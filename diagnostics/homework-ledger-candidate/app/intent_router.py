"""二期·自然语言配置管家 — 意图分流 / 提案 / 确认 / 撤销。

设计要点（严格照 docs/PHASE2_SPEC.md）：
- 首批仅 2 类意图：update_coach_preference（真源 iOS coachPreferenceText）、
  toggle_context（真源 iOS contextInclusionSettings，经 updateContextInclusion 写）。
- 确认前零业务库写入：提案存进程内 TTL 缓存（不建表）。
- account-scoped：每个端点复用 main.principal_from_request（其内部已 set_current_account），
  并把 account_id 编入 proposal_id + 校验时比对，严防串号。
- 不信任 LLM：toggle_context 的 key 必须命中白名单；confidence<0.6 或槽缺 → needs_clarification。
- 不在 import 期引用 main（main 会 include 本 router，循环），LLM/依赖一律函数内惰性导入。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(prefix="/api/intent", tags=["intent"])

# ---------------------------------------------------------------------------
# 进程内提案缓存（不落业务库）
# ---------------------------------------------------------------------------
# key = proposal_id；account 隔离靠 proposal_id 里编入 account 前缀 + 校验时比对。
_PROPOSALS: dict[str, dict] = {}
_PROPOSAL_TTL = 600  # pending 10 分钟
_UNDO_TTL = 24 * 3600  # confirmed 后 24h 可撤窗口

# toggle_context 白名单（对应 iOS ContextInclusionSettings 字段）。
_CONTEXT_KEYS = {
    "visual": "画面",
    "observation": "观察",
    "history": "历史",
    "mistakes": "错题",
    "knowledge": "知识点",
    "memory": "长期记忆",
    "strategy": "策略",
    "debug": "调试",
}

_INTENT_TYPES = {"update_coach_preference", "toggle_context", "qa", "grade_homework", "photo_answer"}
_CONFIDENCE_FLOOR = 0.6


def _now() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _prune() -> None:
    """惰性清理过期提案。pending 用 _PROPOSAL_TTL，confirmed 用 _UNDO_TTL，undone 直接清。"""
    now = _now()
    dead: list[str] = []
    for pid, prop in _PROPOSALS.items():
        status = prop.get("status")
        if status == "pending" and now > prop.get("expires_at_ts", 0):
            dead.append(pid)
        elif status == "confirmed" and now > prop.get("confirmed_at_ts", 0) + _UNDO_TTL:
            dead.append(pid)
        elif status == "undone" and now > prop.get("undone_at_ts", 0) + 60:
            dead.append(pid)
    for pid in dead:
        _PROPOSALS.pop(pid, None)


def _make_proposal_id(account_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", account_id or "local")
    return f"ip_{safe}_{uuid.uuid4().hex}"


def _proposal_account(proposal_id: str) -> str:
    # ip_<account>_<uuid32hex>
    if not proposal_id.startswith("ip_"):
        return ""
    body = proposal_id[3:]
    parts = body.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else ""


# ---------------------------------------------------------------------------
# 账户依赖（严防裸端点 / 串号）
# ---------------------------------------------------------------------------
def _principal(request: Request) -> dict:
    """复用 main.principal_from_request：其内部已 set_current_account(user[account_id])
    并 ensure_account_db，无需在此重复。required=True 确保未登录直接 401。"""
    from . import main  # 惰性导入，规避 main<->intent_router 循环

    return main.principal_from_request(request, required=True)


# ---------------------------------------------------------------------------
# LLM 单次抽槽分类（复用 main 既有 LLM 调用链：run_with_llm_gate + llm.analyze_text）
# ---------------------------------------------------------------------------
_CLASSIFY_PROMPT = """你是「知进拍学」App 的指令分类器。用户在提问输入框里打了一句话，你要判断这是
(A) 普通学习提问（qa），还是 (B) 一条配置指令。配置指令首批只支持两类：

1) update_coach_preference —— 修改“辅导偏好/讲解风格/老师口气”，例如“以后讲题啰嗦一点”“先给提示再讲解”“别老是夸我”。
   槽位 slots: {{"mode": "replace" 或 "append", "text": "用一句话概括的新偏好正文"}}。
   text 必须非空；用户若是“追加/再加上/还要”，mode=append，否则 replace。

2) toggle_context —— 打开/关闭某一类“上下文开关”。可选 key 只能是下面之一：
   visual(画面) observation(观察) history(历史) mistakes(错题) knowledge(知识点) memory(长期记忆) strategy(策略) debug(调试)。
   槽位 slots: {{"key": <上面英文之一>, "value": true 表示打开 / false 表示关闭}}。
   例如“把长期记忆关掉”→ {{"key":"memory","value":false}}；“打开错题上下文”→ {{"key":"mistakes","value":true}}。
   若用户说的开关不在上表（比如“关掉声音”“关闭通知”），不要硬套，请判为 qa 或低置信度。

另外两类是「进入功能模块」的意图（不是配置、也不是直接问答，而是要打开拍照流程）：

3) grade_homework —— 用户想【批改作业/检查对错】，例如“帮我批改这页作业”“检查一下我做得对不对”“我改完了看看错没错”“判一下这张卷子”。slots 可为空。

4) photo_answer —— 用户想【拍照把整页/好几道题分出来解答】，例如“帮我把这几道题分一下”“这一页题拍照解答”“这些题怎么做（一次好几道）”。slots 可为空。
   注意：如果只是问【单独一道题】“这题怎么做/讲讲这道题”，那是普通 qa（现有问答已会自动看画面），不要判为 photo_answer。

如果这句话其实是在【提问】（哪怕含“删/关/记住”等字，但语义是问知识），intent_type 必须是 "qa"。
不确定时给低 confidence。绝不要编造 key，绝不要把不在白名单里的开关塞进 key。

当前用户的配置现状（供你判断 before/after，仅参考，不要照抄）：
{state_summary}

只输出一个 JSON 对象，不要任何多余文字、不要 markdown 代码块：
{{"intent_type": "update_coach_preference"|"toggle_context"|"grade_homework"|"photo_answer"|"qa",
  "confidence": 0.0~1.0 的小数,
  "slots": {{}},
  "summary": "一句中文，概括你打算做的事（qa 时可为空串）",
  "clarification": null 或 "需要向用户澄清的一句话"}}

用户这句话：
{user_text}
"""


def _state_summary(app_state: dict) -> str:
    pref = (app_state.get("coach_preference_text") or "").strip()
    ci = app_state.get("context_inclusion") or {}
    lines = [f"- 辅导偏好正文：{pref or '（空，使用默认策略）'}"]
    on = [_CONTEXT_KEYS[k] for k in _CONTEXT_KEYS if bool(ci.get(k))]
    off = [_CONTEXT_KEYS[k] for k in _CONTEXT_KEYS if k in ci and not bool(ci.get(k))]
    if on:
        lines.append("- 已开启的上下文：" + "、".join(on))
    if off:
        lines.append("- 已关闭的上下文：" + "、".join(off))
    return "\n".join(lines)


def _extract_json(raw: str) -> dict:
    if not raw:
        return {}
    text = raw.strip()
    # 去掉可能的 ```json ... ``` 包裹
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    # 退路：抓第一个 {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


async def _classify(text: str, app_state: dict, account_id: str) -> dict:
    """单次 LLM 调用做抽槽分类。失败/异常一律降级为 qa（让 iOS 照常走 QA）。"""
    from . import llm, main  # 惰性导入规避循环

    settings = main.effective_llm_settings(account_id)
    prompt = _CLASSIFY_PROMPT.format(
        state_summary=_state_summary(app_state),
        user_text=text[:1200],
    )
    try:
        raw = await main.run_with_llm_gate(
            "intent_classify",
            None,
            lambda: llm.analyze_text(settings, prompt, max_tokens=400),
            priority=main.LLM_PRIORITY_REALTIME,
            account_id=account_id,
        )
    except Exception:
        return {"intent_type": "qa", "confidence": 0.0, "slots": {}, "summary": "", "clarification": None}
    parsed = _extract_json(raw)
    if not parsed:
        return {"intent_type": "qa", "confidence": 0.0, "slots": {}, "summary": "", "clarification": None}
    return parsed


# ---------------------------------------------------------------------------
# diff / proposal 构建（用 app_state 快照算 before）
# ---------------------------------------------------------------------------
def _build_preference_proposal(slots: dict, summary: str, app_state: dict) -> dict | None:
    text = str(slots.get("text") or "").strip()
    if not text:
        return None
    mode = slots.get("mode") if slots.get("mode") in ("replace", "append") else "replace"
    before_text = (app_state.get("coach_preference_text") or "").strip()
    if mode == "append" and before_text:
        after_text = (before_text + "\n" + text).strip()
    else:
        after_text = text
    return {
        "category": "preference",
        "title": "调整辅导偏好",
        "summary": summary or f"把辅导偏好设为：{text}",
        "diff": [{
            "label": "辅导偏好",
            "before": before_text or "（默认）",
            "after": after_text,
        }],
        "confirm_label": "确认修改",
        "reversible": True,
        # echo_state 落地与撤销所需的真源目标值与旧值
        "_echo": {
            "category": "preference",
            "coach_preference_text": after_text,
            "before": {"coach_preference_text": before_text},
        },
    }


def _build_context_proposal(slots: dict, summary: str, app_state: dict) -> dict | None:
    key = str(slots.get("key") or "").strip()
    if key not in _CONTEXT_KEYS:  # 白名单硬校验（不信任 LLM）
        return None
    if not isinstance(slots.get("value"), bool):
        return None
    value = bool(slots.get("value"))
    ci = app_state.get("context_inclusion") or {}
    before_val = bool(ci.get(key)) if key in ci else None
    label = _CONTEXT_KEYS[key]
    action = "打开" if value else "关闭"
    return {
        "category": "context_toggle",
        "title": f"调整上下文：{action}{label}",
        "summary": summary or f"本轮起{action}{label}上下文",
        "diff": [{
            "label": label,
            "before": ("开" if before_val else "关") if before_val is not None else "未知",
            "after": "开" if value else "关",
        }],
        "confirm_label": "确认修改",
        "reversible": before_val is not None,
        "_echo": {
            "category": "context_toggle",
            "context_inclusion": {key: value},
            "before": {"context_inclusion": ({key: before_val} if before_val is not None else {})},
        },
    }


def _public_proposal(prop: dict) -> dict:
    """对外暴露的 proposal（剥掉以 _ 开头的内部字段，如 _echo）。"""
    return {k: v for k, v in prop.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------
@router.post("/route")
async def intent_route(request: Request) -> dict:
    """Propose：分类 + 抽槽 + 生成预览，写内存 pending，不执行、不写业务库。"""
    user = _principal(request)
    account_id = user["account_id"]
    _prune()

    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(422, "invalid JSON body")
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(422, "text is required")
    app_state = body.get("app_state") if isinstance(body.get("app_state"), dict) else {}

    result = await _classify(text, app_state, account_id)
    intent_type = result.get("intent_type")
    if intent_type not in _INTENT_TYPES:
        intent_type = "qa"

    if intent_type == "qa":
        return {"intent_kind": "qa"}

    confidence = 0.0
    try:
        confidence = float(result.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    slots = result.get("slots") if isinstance(result.get("slots"), dict) else {}
    summary = str(result.get("summary") or "").strip()
    clarification = result.get("clarification")

    # P4：进入功能模块（拍照答题/作业批改）。不确定就当普通提问，绝不误打断问答。
    if intent_type in ("grade_homework", "photo_answer"):
        if confidence < _CONFIDENCE_FLOOR:
            return {"intent_kind": "qa"}
        return {
            "intent_kind": "module",
            "module": "grade" if intent_type == "grade_homework" else "answer",
            "summary": summary,
        }

    # 低置信度 → 不出卡片，请前端澄清（或当 QA）。
    if confidence < _CONFIDENCE_FLOOR:
        return {
            "intent_kind": "config",
            "proposal": None,
            "needs_clarification": True,
            "clarification": clarification or "我不太确定你是想改设置还是在提问，能再说清楚一点吗？",
        }

    if intent_type == "update_coach_preference":
        built = _build_preference_proposal(slots, summary, app_state)
    else:  # toggle_context
        built = _build_context_proposal(slots, summary, app_state)

    if not built:  # 槽缺 / 白名单未命中 → 不出卡片
        return {
            "intent_kind": "config",
            "proposal": None,
            "needs_clarification": True,
            "clarification": clarification or "这条配置指令我没看明白，能换个说法吗？",
        }

    pid = _make_proposal_id(account_id)
    now = _now()
    expires_ts = now + _PROPOSAL_TTL
    built["id"] = pid
    built["expires_at"] = _iso(expires_ts)
    _PROPOSALS[pid] = {
        "account_id": account_id,
        "status": "pending",
        "created_at_ts": now,
        "expires_at_ts": expires_ts,
        "proposal": built,
    }
    return {"intent_kind": "config", "proposal": _public_proposal(built)}


@router.post("/confirm")
async def intent_confirm(request: Request) -> dict:
    """执行：标记 confirmed（首批不写任何业务库），回吐 echo_state 供 iOS 落本地真源。"""
    user = _principal(request)
    account_id = user["account_id"]
    _prune()

    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(422, "invalid JSON body")
    proposal_id = str(body.get("proposal_id") or "").strip()
    if not proposal_id:
        raise HTTPException(422, "proposal_id is required")

    entry = _PROPOSALS.get(proposal_id)
    if not entry:
        raise HTTPException(404, "proposal not found or expired")
    if entry["account_id"] != account_id or _proposal_account(proposal_id) != re.sub(r"[^a-zA-Z0-9_-]", "_", account_id):
        raise HTTPException(403, "proposal does not belong to this account")
    if entry["status"] != "pending":
        raise HTTPException(409, f"proposal not pending (status={entry['status']})")
    if _now() > entry.get("expires_at_ts", 0):
        _PROPOSALS.pop(proposal_id, None)
        raise HTTPException(410, "proposal expired")

    now = _now()
    entry["status"] = "confirmed"
    entry["confirmed_at_ts"] = now

    prop = entry["proposal"]
    echo = prop.get("_echo") or {}
    return {
        "ok": True,
        "proposal_id": proposal_id,
        "applied": {"category": prop.get("category"), "summary": prop.get("summary")},
        "undoable": bool(prop.get("reversible")) and bool(echo.get("before")),
        "echo_state": echo,
    }


@router.post("/undo")
async def intent_undo(request: Request) -> dict:
    """撤销：回放 before 旧值。携带 current_app_state 做 TOCTOU 二次校验，漂移则 409 拒绝。"""
    user = _principal(request)
    account_id = user["account_id"]
    _prune()

    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(422, "invalid JSON body")
    proposal_id = str(body.get("proposal_id") or "").strip()
    if not proposal_id:
        raise HTTPException(422, "proposal_id is required")
    current = body.get("current_app_state") if isinstance(body.get("current_app_state"), dict) else {}

    entry = _PROPOSALS.get(proposal_id)
    if not entry:
        raise HTTPException(404, "proposal not found or expired")
    if entry["account_id"] != account_id:
        raise HTTPException(403, "proposal does not belong to this account")
    if entry["status"] != "confirmed":
        raise HTTPException(409, f"proposal not undoable (status={entry['status']})")
    if _now() > entry.get("confirmed_at_ts", 0) + _UNDO_TTL:
        raise HTTPException(410, "undo window expired")

    prop = entry["proposal"]
    echo = prop.get("_echo") or {}
    before = echo.get("before") or {}
    category = prop.get("category")

    # current_app_state 缺失/空 → 无法做 TOCTOU 校验，拒绝盲目回写（不静默放行）。
    if not current:
        raise HTTPException(409, "state_unverifiable")

    # TOCTOU：当前值必须仍等于我当初改成的值，否则拒绝（不盲目回写）。
    if category == "preference":
        if "coach_preference_text" not in current:
            raise HTTPException(409, "state_unverifiable")
        applied_val = (echo.get("coach_preference_text") or "")
        cur_val = (current.get("coach_preference_text") or "")
        if cur_val != applied_val:
            raise HTTPException(409, "state_drifted")
        undo_echo = {
            "category": "preference",
            "coach_preference_text": before.get("coach_preference_text") or "",
        }
    elif category == "context_toggle":
        applied_ci = echo.get("context_inclusion") or {}
        cur_ci = current.get("context_inclusion") if isinstance(current.get("context_inclusion"), dict) else None
        if cur_ci is None:
            raise HTTPException(409, "state_unverifiable")
        for k, v in applied_ci.items():
            if k not in cur_ci:
                raise HTTPException(409, "state_unverifiable")
            if bool(cur_ci.get(k)) != bool(v):
                raise HTTPException(409, "state_drifted")
        undo_echo = {
            "category": "context_toggle",
            "context_inclusion": before.get("context_inclusion") or {},
        }
    else:
        raise HTTPException(409, "unsupported category")

    entry["status"] = "undone"
    entry["undone_at_ts"] = _now()
    return {"ok": True, "proposal_id": proposal_id, "status": "undone", "echo_state": undo_echo}
