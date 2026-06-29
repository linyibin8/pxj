import asyncio
import base64
import hmac
import hashlib
import html
import json
import re
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import jwt
try:
    from cryptography.fernet import Fernet
except Exception:  # pragma: no cover - deployment dependency guard
    Fernet = None
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.security.utils import get_authorization_scheme_param
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

from . import embeddings, llm, memory_store, prompts
from .config import get_settings
from .db import account_db_path, connect, connect_control, ensure_account_db, init_db, list_account_ids, set_current_account, utc_now

app = FastAPI(title="知进拍学")
AUTH_SCHEME = "Bearer"
AUTH_PASSWORD_MIN_LENGTH = 8
AUTH_DEFAULT_TOKEN_BYTES = 32
AUTH_HASH_ITERATIONS = 210_000
AUTH_TOKEN_ALGORITHM = "HS256"
AUTH_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PROFILE_TYPES = {"student", "parent", "teacher"}
MODEL_PROVIDERS = {
    "evowit": {"label": "EvoWit 默认模型", "supports_custom_base_url": True, "openai_compatible": True,
               "default_base_url": "http://100.64.0.5:39000/v1", "default_model": "evowit-agent27b",
               "key_hint": "默认使用我们提供的模型，无需填写 Key"},
    "openai": {"label": "OpenAI", "supports_custom_base_url": True, "openai_compatible": True,
               "default_base_url": "https://api.openai.com/v1", "default_model": "gpt-4o-mini",
               "key_hint": "填写你的 OpenAI API Key（sk-...）"},
    "anthropic": {"label": "Anthropic (Claude)", "supports_custom_base_url": True, "openai_compatible": False,
                  "default_base_url": "https://api.anthropic.com", "default_model": "claude-sonnet-4-5",
                  "via_gateway": True,
                  "key_hint": "Anthropic 原生格式，请经网关接入（见下方网关）"},
    "gemini": {"label": "Google Gemini", "supports_custom_base_url": True, "openai_compatible": True,
               "default_base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "default_model": "gemini-2.0-flash",
               "key_hint": "填写你的 Google AI Studio API Key"},
    "zhipu": {"label": "智谱 AI (GLM)", "supports_custom_base_url": True, "openai_compatible": True,
              "default_base_url": "https://open.bigmodel.cn/api/paas/v4", "default_model": "glm-4-flash",
              "key_hint": "填写你的智谱 API Key"},
    "openai-compatible": {"label": "OpenAI-compatible", "supports_custom_base_url": True, "openai_compatible": True,
                          "default_base_url": "", "default_model": "",
                          "key_hint": "任意兼容 OpenAI /chat/completions 的服务"},
}
DEFAULT_ACCOUNT_ID = "local"
llm_gate_lock = asyncio.Lock()
llm_gate_inflight = 0
llm_gate_waiting = 0
llm_gate_last_started_at = 0.0
llm_gate_wait_seq = 0
llm_gate_waiters: dict[str, dict] = {}
llm_gate_inflight_realtime = 0
llm_gate_inflight_background = 0
llm_gate_last_realtime_at = 0.0
# 每个经过 llm_gate 的任务的注册表（按 task_id），用于「按账号查看在跑的后台任务并取消」。
# 受 llm_gate_lock 保护。记录：account_id/user_id/label/lane/session_id/state/created/cancel_requested/future。
llm_tasks: dict[str, dict] = {}
task_dispatcher_task: asyncio.Task | None = None


class LLMTaskCancelled(Exception):
    """用户在任务等待或运行阶段主动取消（账号内取消自己的后台任务）。"""


def task_display_title(label: str) -> str:
    text = (label or "").lower()
    table = (
        ("teaching_visualization", "生成可视化讲解"),
        ("final_report", "生成学习报告"),
        ("question_extraction_session", "提取观察题目"),
        ("qa_session_summary", "生成问答小结"),
        ("memory_consolidation", "整理长期记忆"),
        ("memory_extract", "整理长期记忆"),
        ("distill", "提炼学习要点"),
        ("vision", "分析题目画面"),
        ("profile", "更新学习画像"),
        ("observation", "智能观察分析"),
    )
    for key, name in table:
        if key in text:
            return name
    return "后台生成任务"
THUMBNAIL_MAX_SIDE = 640
THUMBNAIL_QUALITY = 82
FINAL_REPORT_WAIT_SECONDS = 90
DEFAULT_SESSION_ANALYSIS_LIMIT = 20
MAX_SESSION_ANALYSIS_LIMIT = 100
SESSION_OVERVIEW_IMAGE_FALLBACK_LIMIT = 80
DEVICE_CONTROL_POLL_INTERVAL_SECONDS = 1.0
DEVICE_CONTROL_ONLINE_SECONDS = 8
CONTROL_COMMAND_TTL_SECONDS = 120
CONTROL_COMMAND_TYPES = {
    "single_capture",
    "start_burst",
    "stop_burst",
    "voice_question",
    "ok_followup",
    "end_qa",
    "set_goal",
}
CONTROL_COMMAND_ACK_STATUSES = {"applied", "failed", "ignored"}
FINAL_REPORT_PROMPT_CHAR_LIMIT = 28000
FINAL_REPORT_TIMELINE_CHAR_LIMIT = 8000
FINAL_REPORT_ANALYSES_CHAR_LIMIT = 17000
FINAL_REPORT_ANALYSIS_MAX_CHARS = 1400
FINAL_REPORT_ANALYSIS_MIN_CHARS = 180
FINAL_REPORT_CAPTURE_META_CHAR_LIMIT = 180
FINAL_REPORT_DISTILL_TRIGGER_CHARS = 22000
FINAL_REPORT_DISTILL_TRIGGER_ANALYSES = 24
FINAL_REPORT_DISTILL_SOURCE_CHAR_LIMIT = 64000
FINAL_REPORT_DISTILL_CHUNK_CHAR_LIMIT = 6000
FINAL_REPORT_DISTILL_ANALYSIS_MAX_CHARS = 1100
FINAL_REPORT_DISTILL_ANALYSIS_MIN_CHARS = 260
FINAL_REPORT_DISTILLED_NOTES_CHAR_LIMIT = 14000
FINAL_REPORT_DISTILL_MAX_TOKENS = 1600
BATCH_PREVIOUS_CONTEXT_LIMIT = 5000
BATCH_PREVIOUS_ANALYSIS_LIMIT = 6
OBSERVATION_LOOKBACK_LIMIT = 80
VISUAL_DUPLICATE_DISTANCE = 3.2
VISUAL_TEXT_DUPLICATE_DISTANCE = 5.2
TEXT_DUPLICATE_DISTANCE = 0.22
# Keyframe override for 智能观察 dedup: even when the frame is visually a near
# duplicate (camera barely moved), keep it as new content if the OCR token set
# changed meaningfully (the student wrote a new step). Guards against dropping the
# one frame that actually carries new information.
KEYFRAME_TEXT_CHANGE_DISTANCE = 0.45  # Jaccard distance of text tokens vs the matched previous frame
KEYFRAME_MIN_NEW_TOKENS = 3  # require this many genuinely-new tokens (filters single-token OCR noise)
# 语音抓拍快筛: an eligible follow-up frame this close (visually) to the previous QA
# image, with this much text overlap, is the same page -> answer on carried context
# (text-only fast path) instead of a redundant ~6s vision call.
QA_DUPLICATE_FOLLOWUP_TOKEN_OVERLAP = 0.75
LEARNING_ITEM_CONTENT_LIMIT = 900
LEARNING_ITEM_TITLE_LIMIT = 120
LEARNING_ITEM_SUMMARY_LIMIT = 6000
SESSION_GOAL_CHAR_LIMIT = 1200
ASSISTANT_FOCUS_CHAR_LIMIT = 1800
REPORT_PROCESS_CHAR_LIMIT = 6000
QA_PROMPT_CHAR_LIMIT = 16000
QA_RECENT_ANALYSIS_LIMIT = 5
QA_RECENT_EVENT_LIMIT = 8
QA_CONTEXT_ITEM_LIMIT = 12
QA_QUESTION_CHAR_LIMIT = 1200
QA_ANSWER_CHAR_LIMIT = 12000
QA_HTML_CHAR_LIMIT = 20000
TEACHING_VISUALIZATION_PROMPT_CHAR_LIMIT = 22000
TEACHING_VISUALIZATION_SOURCE_CHAR_LIMIT = 9000
TEACHING_VISUALIZATION_HTML_CHAR_LIMIT = 260000
TEACHING_VISUALIZATION_MAX_TOKENS = 7000
TEACHING_VISUALIZATION_SOURCE_TYPES = {"qa_event", "analysis", "custom"}
TEACHING_VISUALIZATION_CSP = (
    "default-src 'self' data: blob: https://cdn.jsdelivr.net https://unpkg.com; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://unpkg.com; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://unpkg.com; "
    "img-src 'self' data: blob:; "
    "font-src 'self' data: https://cdn.jsdelivr.net https://unpkg.com; "
    "connect-src 'none'; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'self'"
)
QA_VISUAL_REVIEW_INTENTS = {"answer_check", "correction_check", "visual_check"}
QA_MIN_TEXT_FOR_CONTEXT = 4
QA_MIN_TEXT_WITH_RECTANGLE_FOR_CONTEXT = 2
QA_MIN_RECTANGLES_WITHOUT_TEXT_FOR_CONTEXT = 2
QA_IMAGE_LIGHT_COVERAGE_MIN = 0.06
QA_IMAGE_EDGE_DENSITY_MIN = 0.018
QA_IMAGE_CONTRAST_MIN = 5.5
QA_IMAGE_MATERIAL_CONFIDENCE_MIN = 0.34
IMAGE_VALIDITY_STRONG_MATERIAL_CONFIDENCE = 0.45
IMAGE_VALIDITY_SOFT_MATERIAL_CONFIDENCE = 0.34
IMAGE_VALIDITY_MIN_TEXT_TOKENS = 2
IMAGE_VALIDITY_MIN_RECTANGLES = 2
IMAGE_VALIDITY_MIN_LIGHT_COVERAGE = 0.06
IMAGE_VALIDITY_MIN_EDGE_DENSITY = 0.018
IMAGE_VALIDITY_MIN_CONTRAST = 5.5
IMAGE_VALIDITY_BLUR_MIN_WITHOUT_TEXT = 0.24
MISTAKE_REASON_LIMIT = 700
ASSET_FIELD_LIMIT = 180
ASSET_SOURCE_SUMMARY_LIMIT = 600
ASSET_DOCUMENT_BODY_LIMIT = 5000
ASSET_PAGE_SIZE_DEFAULT = 25
ASSET_PAGE_SIZE_MAX = 100
TASK_RECOVERY_DELAY_SECONDS = 15
TASK_RETRY_DELAY_SECONDS = 45
TASK_STALE_RUNNING_SECONDS = 300
LLM_PRIORITY_REALTIME = 0
LLM_PRIORITY_BACKGROUND = 100
TASK_PRIORITY_BACKGROUND = 100
TASK_PRIORITY_FINAL_REPORT = 120
TASK_PRIORITY_MEMORY = 150
MEMORY_CONSOLIDATION_INTERVAL_SECONDS = 3600
MEMORY_EVENT_TEXT_LIMIT = 1200
MEMORY_PROFILE_CHAR_LIMIT = 4000
MEMORY_PROFILE_RECENT_EVENT_LIMIT = 80
# "candidate" = 智能观察异步提取出的"可疑错题候选"，尚未进正式错题本/复习队列；学生确认(import)
# 后转 confirmed 才入本，忽略(dismiss)转 ignored。它被刻意排除在 ACTIVE/REVIEW_QUEUE 之外。
MISTAKE_STATUS_VALUES = {"candidate", "suspected", "incomplete", "confirmed", "ignored", "corrected", "mastered"}
MISTAKE_STATUS_ALIASES = {"resolved": "mastered"}
MISTAKE_REVIEW_STATE_VALUES = {"new", "queued", "scheduled", "reviewing", "done", "mastered", "ignored"}
MISTAKE_REVIEW_STATE_ALIASES = {"archived": "ignored", "due": "scheduled", "later": "scheduled"}
ACTIVE_MISTAKE_STATUSES = {"suspected", "incomplete", "confirmed", "corrected"}
REVIEW_QUEUE_STATUSES = {"suspected", "incomplete", "confirmed", "corrected"}
REVIEW_SCHEDULE_DAYS = {
    "new": 0,
    "queued": 1,
    "scheduled": 1,
    "reviewing": 1,
    "done": 3,
    "corrected": 2,
}
REVIEW_EVENT_RESULTS = {"correct", "incorrect", "postpone", "mastered"}
REVIEW_EVENT_ALIASES = {
    "right": "correct",
    "ok": "correct",
    "wrong": "incorrect",
    "again": "incorrect",
    "delay": "postpone",
    "later": "postpone",
    "schedule": "postpone",
    "done": "correct",
}
UNCLEAR_ANALYSIS_SIGNALS = (
    "未识别",
    "看不清",
    "看不清楚",
    "不清楚",
    "模糊",
    "太小",
    "被遮挡",
    "遮挡",
    "反光",
    "无法确认",
    "无法辨认",
    "无法识别",
    "未能识别",
    "识别不清",
    "不清晰",
    "读不清",
    "无法读取",
    "难以辨认",
    "无法判断",
    "未完整入镜",
    "没有完整放进",
    "没有完整进入",
    "未完整进入",
    "画面外",
    "视野外",
    "超出画面",
    "超出视野",
    "边缘被裁切",
    "被裁切",
    "截断",
    "不完整",
    "拍摄不完整",
    "只拍到一部分",
    "只拍到局部",
    "部分可见",
    "只有一半",
    "主体偏出画面",
    "内容缺失",
    "未拍全",
    "没拍全",
    "没拍完整",
    "调整相机",
    "完整放进画面",
    "完整放进拍摄区域",
)


@app.on_event("startup")
def startup() -> None:
    global task_dispatcher_task
    init_db()
    settings = get_settings()
    app.mount("/images", StaticFiles(directory=settings.data_dir / "images"), name="images")
    recover_interrupted_tasks()
    schedule_memory_consolidation_if_due()
    try:
        loop = asyncio.get_running_loop()
        if task_dispatcher_task is None or task_dispatcher_task.done():
            task_dispatcher_task = loop.create_task(task_dispatcher_loop())
    except RuntimeError:
        pass


def emit_log(message: str, *, session_id: str | None = None, device_id: str | None = None, level: str = "info", source: str = "backend") -> None:
    account_id = get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    if session_id:
        try:
            with connect() as conn:
                row = conn.execute("SELECT account_id FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row and row["account_id"]:
                account_id = row["account_id"]
        except Exception:
            account_id = get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    with connect() as conn:
        conn.execute(
            "INSERT INTO logs(account_id, session_id, device_id, level, source, message, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (account_id, session_id, device_id, level, source, message, utc_now()),
        )


def normalize_llm_max_concurrency(value: int) -> int:
    return max(1, min(4, int(value or 1)))


def normalize_llm_min_interval(value: float) -> float:
    return max(0.0, min(120.0, float(value or 0.0)))


def session_llm_identity(session_id: str | None = None, *, account_id: str = "", user_id: str = "") -> tuple[str, str]:
    settings = get_settings()
    resolved_account_id = account_id or settings.default_account_id or DEFAULT_ACCOUNT_ID
    resolved_user_id = user_id or ""
    if session_id:
        try:
            with connect() as conn:
                row = conn.execute(
                    "SELECT account_id, created_by_user_id FROM sessions WHERE id=?",
                    (session_id,),
                ).fetchone()
            if row:
                resolved_account_id = row["account_id"] or resolved_account_id
                resolved_user_id = row["created_by_user_id"] or resolved_user_id
        except Exception:
            pass
    return resolved_account_id, resolved_user_id


def emit_llm_gate_log(message: str, *, session_id: str | None = None, level: str = "info") -> None:
    try:
        emit_log(message, session_id=session_id, level=level)
    except Exception:
        pass


def record_task_run(
    task_kind: str,
    *,
    account_id: str | None = None,
    session_id: str | None = None,
    analysis_id: str | None = None,
    payload: dict | None = None,
    delay_seconds: int = TASK_RECOVERY_DELAY_SECONDS,
    priority: int = TASK_PRIORITY_BACKGROUND,
) -> str:
    task_id = uuid.uuid4().hex
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    available_at = (now_dt + timedelta(seconds=max(0, delay_seconds))).isoformat()
    account_id = account_id or get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    if session_id:
        try:
            with connect() as conn:
                row = conn.execute("SELECT account_id FROM sessions WHERE id=?", (session_id,)).fetchone()
            if row and row["account_id"]:
                account_id = row["account_id"]
        except Exception:
            account_id = get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    try:
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO task_runs(
                    id, account_id, task_kind, status, session_id, analysis_id, payload,
                    priority, available_at, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    account_id,
                    task_kind,
                    "queued",
                    session_id,
                    analysis_id,
                    json_dumps(payload or {}),
                    int(priority),
                    available_at,
                    now,
                    now,
                ),
            )
    except Exception:
        return task_id
    return task_id


def mark_task_run(task_id: str | None, status: str, *, error: str = "") -> None:
    if not task_id:
        return
    now = utc_now()
    try:
        with connect() as conn:
            if status == "running":
                conn.execute(
                    """
                    UPDATE task_runs
                    SET status=?, attempts=attempts + 1, started_at=?, last_error='', updated_at=?
                    WHERE id=?
                    """,
                    (status, now, now, task_id),
                )
            elif status in {"done", "failed"}:
                conn.execute(
                    """
                    UPDATE task_runs
                    SET status=?, last_error=?, finished_at=?, updated_at=?
                    WHERE id=?
                    """,
                    (status, truncate_text(error, 1000), now, now, task_id),
                )
            else:
                conn.execute("UPDATE task_runs SET status=?, updated_at=? WHERE id=?", (status, now, task_id))
    except Exception:
        pass


def recover_interrupted_tasks() -> None:
    now = datetime.now(timezone.utc)
    stale_before = (now - timedelta(seconds=TASK_STALE_RUNNING_SECONDS)).isoformat()
    now_text = now.isoformat()
    try:
        with connect() as conn:
            conn.execute(
                """
                UPDATE task_runs
                SET status='queued', available_at=?, updated_at=?, last_error='recovered after process restart'
                WHERE status='running' AND (started_at='' OR started_at < ?)
                """,
                (now_text, now_text, stale_before),
            )
    except Exception:
        pass


def auth_secret_key() -> str:
    settings = get_settings()
    configured = settings.auth_secret_key.strip()
    if configured:
        return configured
    secret_path = settings.data_dir / "auth_secret.key"
    if secret_path.exists():
        value = secret_path.read_text(encoding="utf-8").strip()
        if value:
            return value
    token = base64.urlsafe_b64encode(uuid.uuid4().bytes + uuid.uuid4().bytes).decode("ascii").rstrip("=")
    secret_path.write_text(token, encoding="utf-8")
    return token


def normalize_email(value: object) -> str:
    return str(value or "").strip().lower()[:254]


def hash_password(password: str) -> str:
    salt = uuid.uuid4().bytes
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, AUTH_HASH_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        AUTH_HASH_ITERATIONS,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        scheme, iterations_raw, salt_raw, digest_raw = str(password_hash or "").split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
        salt = base64.b64decode(salt_raw.encode("ascii"))
        expected = base64.b64decode(digest_raw.encode("ascii"))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def auth_public_config() -> dict:
    settings = get_settings()
    return {
        "auth_required": bool(settings.auth_required),
        "registration_enabled": bool(settings.registration_enabled),
        "token_ttl_minutes": int(settings.auth_token_ttl_minutes or 0),
    }


def make_access_token(user: dict) -> str:
    now = datetime.now(timezone.utc)
    ttl_minutes = max(5, int(get_settings().auth_token_ttl_minutes or 0))
    payload = {
        "sub": user["id"],
        "account_id": user["account_id"],
        "email": user["email"],
        "role": user.get("role") or "member",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ttl_minutes)).timestamp()),
    }
    return jwt.encode(payload, auth_secret_key(), algorithm=AUTH_TOKEN_ALGORITHM)


def public_user(user: dict) -> dict:
    return {
        "id": user.get("id", ""),
        "account_id": user.get("account_id", ""),
        "email": user.get("email", ""),
        "display_name": user.get("display_name", ""),
        "role": user.get("role", ""),
        "status": user.get("status", ""),
        "created_at": user.get("created_at", ""),
        "updated_at": user.get("updated_at", ""),
        "last_login_at": user.get("last_login_at", ""),
    }


def default_principal() -> dict:
    account_id = get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    set_current_account(account_id)
    return {
        "authenticated": False,
        "account_id": account_id,
        "user_id": "",
        "email": "",
        "role": "legacy",
    }


def bind_account_context_from_token(request: Request | None) -> None:
    """Lightweight: set the per-account DB context from the JWT account_id without a DB round-trip.

    Used by high-frequency, best-effort endpoints (e.g. log ingest) that should land in the
    caller's account DB but don't need full user verification.
    """
    default_id = get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    authorization = request.headers.get("Authorization", "") if request else ""
    scheme, token = get_authorization_scheme_param(authorization)
    if token and scheme.lower() == AUTH_SCHEME.lower():
        try:
            payload = jwt.decode(token, auth_secret_key(), algorithms=[AUTH_TOKEN_ALGORITHM])
            account_id = str(payload.get("account_id") or "")
            if account_id:
                set_current_account(account_id)
                return
        except Exception:
            pass
    set_current_account(default_id)


def principal_from_request(request: Request | None, *, required: bool | None = None) -> dict:
    settings = get_settings()
    must_auth = settings.auth_required if required is None else required
    authorization = request.headers.get("Authorization", "") if request else ""
    scheme, token = get_authorization_scheme_param(authorization)
    if not token:
        if must_auth:
            raise HTTPException(401, "login required")
        return default_principal()
    if scheme.lower() != AUTH_SCHEME.lower():
        raise HTTPException(401, "invalid authorization scheme")
    try:
        payload = jwt.decode(token, auth_secret_key(), algorithms=[AUTH_TOKEN_ALGORITHM])
    except Exception:
        raise HTTPException(401, "invalid or expired token")
    user_id = str(payload.get("sub") or "")
    account_id = str(payload.get("account_id") or "")
    if not user_id or not account_id:
        raise HTTPException(401, "invalid token")
    with connect_control() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE id=? AND account_id=? AND status='active'",
            (user_id, account_id),
        ).fetchone()
    if not row:
        raise HTTPException(401, "user not found or disabled")
    user = dict(row)
    set_current_account(user["account_id"])
    ensure_account_db(user["account_id"])
    return {
        "authenticated": True,
        "account_id": user["account_id"],
        "user_id": user["id"],
        "email": user["email"],
        "role": user["role"],
        "user": public_user(user),
    }


def account_filter(principal: dict, alias: str = "sessions") -> tuple[str, list[object]]:
    if principal.get("authenticated") or get_settings().auth_required:
        return f"{alias}.account_id=?", [principal.get("account_id") or DEFAULT_ACCOUNT_ID]
    return "1=1", []


def require_account_session(conn, session_id: str, principal: dict) -> dict:
    if principal.get("authenticated") or get_settings().auth_required:
        row = conn.execute("SELECT * FROM sessions WHERE id=? AND account_id=?", (session_id, principal["account_id"])).fetchone()
    else:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not row:
        raise HTTPException(404, "session not found")
    return dict(row)


def clean_auth_text(value: object, max_chars: int = 160) -> str:
    return str(value or "").strip()[:max_chars]


def account_profiles(account_id: str) -> list[dict]:
    with connect_control() as conn:
        return [
            dict(row)
            for row in conn.execute(
                """
                SELECT *
                FROM identity_profiles
                WHERE account_id=? AND status='active'
                ORDER BY
                    CASE profile_type WHEN 'student' THEN 0 WHEN 'parent' THEN 1 WHEN 'teacher' THEN 2 ELSE 3 END,
                    created_at ASC
                """,
                (account_id,),
            )
        ]


def create_identity_profile(account_id: str, body: dict, *, user_id: str = "") -> dict:
    profile_type = clean_auth_text(body.get("profile_type") or body.get("profileType") or "student", 40)
    if profile_type not in PROFILE_TYPES:
        raise HTTPException(422, "profile_type must be student, parent, or teacher")
    display_name = clean_auth_text(body.get("display_name") or body.get("displayName") or body.get("name"), 120)
    if not display_name:
        display_name = {"student": "默认学生", "parent": "家长", "teacher": "老师"}[profile_type]
    student_id = clean_auth_text(body.get("student_id") or body.get("studentId"), 80)
    relation = clean_auth_text(body.get("relation"), 80)
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    now = utc_now()
    profile_id = uuid.uuid4().hex
    if not user_id:
        with connect_control() as conn:
            owner = conn.execute(
                "SELECT id FROM users WHERE account_id=? AND status='active' ORDER BY created_at ASC LIMIT 1",
                (account_id,),
            ).fetchone()
        user_id = owner["id"] if owner else ""
    with connect_control() as conn:
        conn.execute(
            """
            INSERT INTO identity_profiles(
                id, account_id, user_id, profile_type, display_name, student_id,
                relation, metadata, status, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
            """,
            (
                profile_id,
                account_id,
                user_id,
                profile_type,
                display_name,
                student_id,
                relation,
                json_dumps(metadata),
                now,
                now,
            ),
        )
        row = conn.execute("SELECT * FROM identity_profiles WHERE id=?", (profile_id,)).fetchone()
    set_current_account(account_id)
    ensure_account_db(account_id)
    return dict(row)


def account_default_student_id(account_id: str) -> str:
    with connect_control() as conn:
        row = conn.execute(
            """
            SELECT id
            FROM identity_profiles
            WHERE account_id=? AND profile_type='student' AND status='active'
            ORDER BY created_at ASC
            LIMIT 1
            """,
            (account_id,),
        ).fetchone()
    return row["id"] if row else ""


def resolve_student_profile(account_id: str, requested_id: str = "") -> str:
    requested_id = clean_auth_text(requested_id, 80)
    if requested_id:
        with connect_control() as conn:
            row = conn.execute(
                "SELECT id FROM identity_profiles WHERE id=? AND account_id=? AND profile_type='student' AND status='active'",
                (requested_id, account_id),
            ).fetchone()
        if not row:
            raise HTTPException(422, "student profile not found")
        return requested_id
    return account_default_student_id(account_id)


def active_model_config(account_id: str = "", user_id: str = "") -> dict:
    settings = get_settings()
    if account_id:
        try:
            with connect_control() as conn:
                row = conn.execute(
                    """
                    SELECT *
                    FROM model_configs
                    WHERE account_id=? AND enabled=1
                    ORDER BY is_default DESC, updated_at DESC
                    LIMIT 1
                    """,
                    (account_id,),
                ).fetchone()
        except Exception:
            row = None
        if row:
            data = dict(row)
            return {
                "id": data["id"],
                "account_id": data["account_id"],
                "owner_user_id": data.get("owner_user_id", ""),
                "provider": data["provider"],
                "name": data["name"],
                "base_url": data["base_url"],
                "model": data["model"],
                "enabled": bool(data["enabled"]),
                "is_default": bool(data["is_default"]),
                "max_concurrency": int(data.get("max_concurrency") or settings.llm_max_concurrency),
                "min_interval_seconds": float(data.get("min_interval_seconds") or 0),
                "metadata": parse_json_object(data.get("metadata")),
                "api_key_configured": bool(data.get("api_key_encrypted")),
            }
    return {
        "id": "system-default",
        "account_id": account_id or "",
        "owner_user_id": user_id or "",
        "provider": settings.llm_provider,
        "name": "EvoWit 默认模型",
        "base_url": settings.llm_base_url,
        "model": settings.llm_model,
        "enabled": True,
        "is_default": True,
        "max_concurrency": normalize_llm_max_concurrency(settings.llm_max_concurrency),
        "min_interval_seconds": normalize_llm_min_interval(settings.llm_min_interval_seconds),
        "metadata": {"managed_by": "system"},
                "api_key_configured": bool(settings.llm_api_key),
    }


def effective_llm_settings(account_id: str = "", user_id: str = ""):
    settings = get_settings()
    config = active_model_config(account_id, user_id)
    api_key = settings.llm_api_key
    config_id = config.get("id") or ""
    if config_id and config_id != "system-default" and config.get("api_key_configured"):
        try:
            with connect_control() as conn:
                row = conn.execute(
                    """
                    SELECT api_key_encrypted
                    FROM model_configs
                    WHERE id=? AND account_id=? AND enabled=1
                    """,
                    (config_id, config.get("account_id") or account_id),
                ).fetchone()
            if row and row["api_key_encrypted"]:
                api_key = decrypt_model_secret(row["api_key_encrypted"])
        except Exception:
            api_key = settings.llm_api_key
    updates = {
        "llm_provider": config.get("provider") or settings.llm_provider,
        "llm_base_url": config.get("base_url") or settings.llm_base_url,
        "llm_api_key": api_key,
        "llm_model": config.get("model") or settings.llm_model,
        "llm_max_concurrency": normalize_llm_max_concurrency(config.get("max_concurrency") or settings.llm_max_concurrency),
        "llm_min_interval_seconds": normalize_llm_min_interval(
            config.get("min_interval_seconds")
            if config.get("min_interval_seconds") is not None
            else settings.llm_min_interval_seconds
        ),
    }
    if hasattr(settings, "model_copy"):
        return settings.model_copy(update=updates)
    return settings.copy(update=updates)


def effective_llm_settings_for_session(session_id: str | None = None, *, account_id: str = "", user_id: str = ""):
    resolved_account_id, resolved_user_id = session_llm_identity(session_id, account_id=account_id, user_id=user_id)
    return effective_llm_settings(resolved_account_id, resolved_user_id)


def model_config_public(row: dict) -> dict:
    item = dict(row)
    item["enabled"] = bool(item.get("enabled"))
    item["is_default"] = bool(item.get("is_default"))
    item["api_key_configured"] = bool(item.get("api_key_encrypted"))
    item.pop("api_key_encrypted", None)
    item["metadata"] = parse_json_object(item.get("metadata"))
    return item


def model_secret_fernet():
    if Fernet is None:
        raise HTTPException(500, "cryptography is required for model key encryption")
    digest = hashlib.sha256(auth_secret_key().encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_model_secret(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return model_secret_fernet().encrypt(text.encode("utf-8")).decode("ascii")


def decrypt_model_secret(value: str) -> str:
    if not value:
        return ""
    return model_secret_fernet().decrypt(value.encode("ascii")).decode("utf-8")


def record_llm_usage_start(label: str, session_id: str | None, lane: str, account_id: str, user_id: str = "") -> str:
    settings = get_settings()
    config = active_model_config(account_id, user_id)
    event_id = uuid.uuid4().hex
    now = utc_now()
    try:
        with connect() as conn:
            conn.execute(
                """
                INSERT INTO llm_usage_events(
                    id, account_id, user_id, session_id, request_label, lane, provider,
                    model, base_url, status, started_at, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?)
                """,
                (
                    event_id,
                    account_id or settings.default_account_id or DEFAULT_ACCOUNT_ID,
                    user_id or "",
                    session_id or "",
                    truncate_text(label, 180),
                    lane,
                    config.get("provider") or settings.llm_provider,
                    config.get("model") or settings.llm_model,
                    config.get("base_url") or settings.llm_base_url,
                    now,
                    now,
                    now,
                ),
            )
    except Exception:
        return ""
    return event_id


def record_llm_usage_finish(event_id: str, status: str, *, error: str = "", started_monotonic: float | None = None) -> None:
    if not event_id:
        return
    duration_ms = 0
    if started_monotonic is not None:
        duration_ms = max(0, int((asyncio.get_running_loop().time() - started_monotonic) * 1000))
    now = utc_now()
    try:
        with connect() as conn:
            conn.execute(
                """
                UPDATE llm_usage_events
                SET status=?, duration_ms=?, error=?, finished_at=?, updated_at=?
                WHERE id=?
                """,
                (status, duration_ms, truncate_text(error, 800), now, now, event_id),
            )
    except Exception:
        return


def llm_usage_snapshot(account_id: str = "") -> dict:
    account_id = account_id or get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    with connect() as conn:
        total_row = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) AS done_count,
                SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed_count,
                SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) AS running_count,
                AVG(CASE WHEN duration_ms > 0 THEN duration_ms ELSE NULL END) AS avg_duration_ms
            FROM llm_usage_events
            WHERE account_id=? AND created_at >= ?
            """,
            (account_id, cutoff),
        ).fetchone()
        by_model = [
            dict(row)
            for row in conn.execute(
                """
                SELECT provider, model, status, COUNT(*) AS count
                FROM llm_usage_events
                WHERE account_id=? AND created_at >= ?
                GROUP BY provider, model, status
                ORDER BY count DESC
                LIMIT 20
                """,
                (account_id, cutoff),
            )
        ]
        recent = [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, session_id, request_label, lane, provider, model, status,
                       duration_ms, error, started_at, finished_at, created_at
                FROM llm_usage_events
                WHERE account_id=?
                ORDER BY created_at DESC
                LIMIT 20
                """,
                (account_id,),
            )
        ]
    total = int(total_row["total"] or 0) if total_row else 0
    done = int(total_row["done_count"] or 0) if total_row else 0
    failed = int(total_row["failed_count"] or 0) if total_row else 0
    running = int(total_row["running_count"] or 0) if total_row else 0
    return {
        "account_id": account_id,
        "window_hours": 24,
        "total": total,
        "done": done,
        "failed": failed,
        "running": running,
        "success_rate": round(done / total, 3) if total else 0,
        "avg_duration_ms": round(float(total_row["avg_duration_ms"]), 1) if total_row and total_row["avg_duration_ms"] is not None else 0,
        "by_model": by_model,
        "recent": recent,
    }


def schedule_memory_consolidation_if_due(*, force: bool = False, account_id: str = "") -> str | None:
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    account_id = account_id or get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    try:
        with connect() as conn:
            event_count = conn.execute("SELECT COUNT(*) AS count FROM memory_events WHERE account_id=?", (account_id,)).fetchone()["count"]
            if not event_count:
                return None
            pending = conn.execute(
                """
                SELECT id
                FROM task_runs
                WHERE task_kind='memory_consolidation' AND account_id=? AND status IN ('queued', 'running')
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (account_id,),
            ).fetchone()
            if pending and not force:
                return pending["id"]
            profile = conn.execute("SELECT updated_at FROM memory_profiles WHERE account_id=? AND scope='global'", (account_id,)).fetchone()
            if profile and not force:
                updated_at = parse_datetime(profile["updated_at"])
                if updated_at and (now_dt - updated_at).total_seconds() < MEMORY_CONSOLIDATION_INTERVAL_SECONDS:
                    return None
            last_done = conn.execute(
                """
                SELECT finished_at, updated_at
                FROM task_runs
                WHERE task_kind='memory_consolidation' AND account_id=? AND status='done'
                ORDER BY finished_at DESC, updated_at DESC
                LIMIT 1
                """,
                (account_id,),
            ).fetchone()
            if last_done and not force:
                last_at = parse_datetime(last_done["finished_at"] or last_done["updated_at"])
                if last_at and (now_dt - last_at).total_seconds() < MEMORY_CONSOLIDATION_INTERVAL_SECONDS:
                    return None
        return record_task_run(
            "memory_consolidation",
            account_id=account_id,
            payload={"account_id": account_id, "scope": "global", "scheduled_at": now, "force": force},
            delay_seconds=0,
            priority=TASK_PRIORITY_MEMORY,
        )
    except Exception:
        return None


def claim_task_run(*, task_id: str | None = None, respect_available_at: bool = True) -> dict | None:
    now = utc_now()
    where = ["status='queued'", "attempts < max_attempts"]
    params: list[object] = []
    if task_id:
        where.append("id=?")
        params.append(task_id)
    if respect_available_at:
        where.append("available_at <= ?")
        params.append(now)
    where_sql = " AND ".join(where)
    with connect() as conn:
        row = conn.execute(
            f"""
            SELECT id
            FROM task_runs
            WHERE {where_sql}
            ORDER BY priority ASC, available_at ASC, created_at ASC
            LIMIT 1
            """,
            params,
        ).fetchone()
        if not row:
            return None
        cursor = conn.execute(
            """
            UPDATE task_runs
            SET status='running', attempts=attempts + 1, started_at=?, updated_at=?, last_error=''
            WHERE id=? AND status='queued' AND attempts < max_attempts
            """,
            (now, now, row["id"]),
        )
        if cursor.rowcount != 1:
            return None
        claimed = conn.execute("SELECT * FROM task_runs WHERE id=?", (row["id"],)).fetchone()
    return dict(claimed) if claimed else None


def claim_next_task_run() -> dict | None:
    return claim_task_run(respect_available_at=True)


async def execute_task_run_by_id(task_id: str) -> None:
    task = claim_task_run(task_id=task_id, respect_available_at=False)
    if task:
        await execute_task_run(task)


async def execute_next_task_run_now() -> None:
    task = claim_task_run(respect_available_at=False)
    if task:
        await execute_task_run(task)


def reschedule_task_run(task_id: str, error: str) -> None:
    now_dt = datetime.now(timezone.utc)
    next_at = (now_dt + timedelta(seconds=TASK_RETRY_DELAY_SECONDS)).isoformat()
    now = now_dt.isoformat()
    with connect() as conn:
        row = conn.execute("SELECT attempts, max_attempts FROM task_runs WHERE id=?", (task_id,)).fetchone()
        if not row:
            return
        if int(row["attempts"] or 0) >= int(row["max_attempts"] or 3):
            conn.execute(
                """
                UPDATE task_runs
                SET status='failed', last_error=?, finished_at=?, updated_at=?
                WHERE id=?
                """,
                (truncate_text(error, 1000), now, now, task_id),
            )
        else:
            conn.execute(
                """
                UPDATE task_runs
                SET status='queued', last_error=?, available_at=?, updated_at=?
                WHERE id=?
                """,
                (truncate_text(error, 1000), next_at, now, task_id),
            )


async def execute_task_run(task: dict) -> None:
    task_id = task["id"]
    if task.get("account_id"):
        set_current_account(task["account_id"])
        ensure_account_db(task["account_id"])
    payload = {}
    try:
        loaded = json.loads(task.get("payload") or "{}")
        payload = loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        payload = {}
    try:
        if task["task_kind"] == "vision_analysis":
            analysis_id = task.get("analysis_id") or payload.get("analysis_id")
            session_id = task.get("session_id") or payload.get("session_id")
            if not analysis_id or not session_id:
                raise RuntimeError("vision task missing analysis_id/session_id")
            with connect() as conn:
                row = conn.execute("SELECT batch_id, prompt, scope, status FROM analyses WHERE id=?", (analysis_id,)).fetchone()
            if not row:
                raise RuntimeError(f"analysis not found: {analysis_id}")
            if row["status"] == "done":
                mark_task_run(task_id, "done")
                return
            filenames = [str(name) for name in payload.get("filenames") or []]
            summarize = bool(payload.get("scope") == "batch" or row["scope"] == "batch")
            status, content = await run_analysis(analysis_id, session_id, row["batch_id"], row["prompt"], filenames, summarize, task_id, already_claimed=True)
            if status == "done":
                mark_task_run(task_id, "done")
            else:
                reschedule_task_run(task_id, content)
            return
        if task["task_kind"] == "final_report":
            analysis_id = task.get("analysis_id") or payload.get("analysis_id")
            session_id = task.get("session_id") or payload.get("session_id")
            if not analysis_id or not session_id:
                raise RuntimeError("final report task missing analysis_id/session_id")
            with connect() as conn:
                row = conn.execute("SELECT status FROM analyses WHERE id=?", (analysis_id,)).fetchone()
            if row and row["status"] == "done":
                mark_task_run(task_id, "done")
                return
            status, content = await run_final_report(analysis_id, session_id, task_id, already_claimed=True)
            if status == "done":
                mark_task_run(task_id, "done")
            else:
                reschedule_task_run(task_id, content)
            return
        if task["task_kind"] == "memory_consolidation":
            await run_memory_consolidation(task_id=task_id, account_id=payload.get("account_id") or task.get("account_id") or "")
            mark_task_run(task_id, "done")
            return
        if task["task_kind"] == "qa_session_summary":
            analysis_id = task.get("analysis_id") or payload.get("analysis_id")
            session_id = task.get("session_id") or payload.get("session_id")
            if not analysis_id or not session_id:
                raise RuntimeError("QA summary task missing analysis_id/session_id")
            with connect() as conn:
                row = conn.execute("SELECT status FROM analyses WHERE id=?", (analysis_id,)).fetchone()
            if row and row["status"] == "done":
                mark_task_run(task_id, "done")
                return
            status, content = await run_qa_session_summary(analysis_id, session_id, task_id, already_claimed=True)
            if status == "done":
                mark_task_run(task_id, "done")
            else:
                reschedule_task_run(task_id, content)
            return
        if task["task_kind"] == "question_extraction_session":
            session_id = task.get("session_id") or payload.get("session_id")
            if not session_id:
                raise RuntimeError("question extraction task missing session_id")
            filenames = [str(name) for name in payload.get("filenames") or [] if str(name or "").strip()]
            await run_session_question_extraction(session_id, filenames, task_id=task_id)
            mark_task_run(task_id, "done")
            return
        raise RuntimeError(f"unknown task kind: {task['task_kind']}")
    except LLMTaskCancelled as exc:
        mark_task_run(task_id, "failed", error=str(exc) or "cancelled")
    except Exception as exc:
        reschedule_task_run(task_id, str(exc))


async def task_dispatcher_loop() -> None:
    while True:
        dispatched = False
        try:
            for account_id in list_account_ids():
                set_current_account(account_id)
                recover_interrupted_tasks()
                task = claim_next_task_run()
                if task:
                    # execute_task_run re-binds the account context itself.
                    asyncio.create_task(execute_task_run(task))
                    dispatched = True
            if dispatched:
                await asyncio.sleep(0.1)
                continue
        except Exception:
            pass
        await asyncio.sleep(2)


def normalize_llm_priority(priority: int | str | None) -> int:
    if isinstance(priority, str):
        normalized = priority.strip().lower()
        if normalized in {"realtime", "interactive", "qa", "voice", "gesture", "high"}:
            return LLM_PRIORITY_REALTIME
        if normalized in {"background", "batch", "vision", "report", "normal", "low"}:
            return LLM_PRIORITY_BACKGROUND
    try:
        return max(0, min(1000, int(priority if priority is not None else LLM_PRIORITY_BACKGROUND)))
    except (TypeError, ValueError):
        return LLM_PRIORITY_BACKGROUND


def free_quota_state(account_id: str) -> dict:
    """Today's realtime usage of the DEFAULT (our) model for an account vs the free cap."""
    settings = get_settings()
    limit = int(settings.free_daily_quota or 0)
    state = {"enabled": limit > 0, "limit": limit, "used": 0, "remaining": -1}
    if limit <= 0:
        return state
    account_id = account_id or settings.default_account_id or DEFAULT_ACCOUNT_ID
    day_start = datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00")
    try:
        with connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS n FROM llm_usage_events
                WHERE account_id=? AND lane='realtime' AND base_url=? AND created_at>=? AND status!='failed'
                """,
                (account_id, settings.llm_base_url, day_start),
            ).fetchone()
        used = int(row["n"] or 0) if row else 0
    except Exception:
        used = 0
    state["used"] = used
    state["remaining"] = max(0, limit - used)
    return state


def llm_gate_waiting_counts() -> dict[str, int]:
    realtime = sum(1 for waiter in llm_gate_waiters.values() if int(waiter.get("priority", LLM_PRIORITY_BACKGROUND)) <= LLM_PRIORITY_REALTIME)
    background = max(0, len(llm_gate_waiters) - realtime)
    return {"realtime": realtime, "background": background, "total": len(llm_gate_waiters)}


def llm_gate_next_waiter_id() -> str | None:
    if not llm_gate_waiters:
        return None
    return min(
        llm_gate_waiters,
        key=lambda key: (
            int(llm_gate_waiters[key].get("priority", LLM_PRIORITY_BACKGROUND)),
            int(llm_gate_waiters[key].get("seq", 0)),
        ),
    )


async def run_with_llm_gate(
    label: str,
    session_id: str | None,
    call,
    *,
    priority: int | str = LLM_PRIORITY_BACKGROUND,
    account_id: str = "",
    user_id: str = "",
):
    global llm_gate_inflight, llm_gate_waiting, llm_gate_last_started_at, llm_gate_wait_seq
    global llm_gate_inflight_realtime, llm_gate_inflight_background, llm_gate_last_realtime_at
    account_id, user_id = session_llm_identity(session_id, account_id=account_id, user_id=user_id)
    settings = effective_llm_settings(account_id, user_id)
    max_concurrency = normalize_llm_max_concurrency(settings.llm_max_concurrency)
    min_interval = normalize_llm_min_interval(settings.llm_min_interval_seconds)
    idle_cooldown = max(0.0, float(getattr(settings, "llm_background_idle_seconds", 12.0) or 0))
    max_defer = max(0.0, float(getattr(settings, "llm_background_max_defer_seconds", 240.0) or 0))
    normalized_priority = normalize_llm_priority(priority)
    lane = "realtime" if normalized_priority <= LLM_PRIORITY_REALTIME else "background"
    base_settings = get_settings()
    if (
        lane == "realtime"
        and int(base_settings.free_daily_quota or 0) > 0
        and settings.llm_base_url == base_settings.llm_base_url
    ):
        quota = free_quota_state(account_id)
        if quota["enabled"] and quota["remaining"] <= 0:
            raise HTTPException(
                429,
                f"今日免费额度已用完（{quota['used']}/{quota['limit']} 次）。可在「账号与模型」里配置你自己的大模型继续使用，或明日自动恢复。",
            )
    warned = False
    idle_notified = False
    waiter_id = uuid.uuid4().hex
    registered_waiter = False
    acquired_slot = False   # 取消语义：拿到放行槽前后清理 llm_tasks 的位置不同
    usage_event_id = ""
    usage_started_at: float | None = None
    started_waiting = asyncio.get_running_loop().time()
    # 注册到任务表（等待阶段就可见、可取消）。
    llm_tasks[waiter_id] = {
        "id": waiter_id,
        "account_id": account_id,
        "user_id": user_id,
        "label": label,
        "lane": lane,
        "session_id": session_id or "",
        "state": "waiting",
        "created": started_waiting,
        "cancel_requested": False,
        "future": None,
    }
    try:
        while True:
            async with llm_gate_lock:
                loop = asyncio.get_running_loop()
                now = loop.time()
                # 等待阶段被取消：直接中止（不占用放行槽）。
                if llm_tasks.get(waiter_id, {}).get("cancel_requested"):
                    raise LLMTaskCancelled()
                if not registered_waiter:
                    llm_gate_wait_seq += 1
                    llm_gate_waiters[waiter_id] = {
                        "priority": normalized_priority,
                        "seq": llm_gate_wait_seq,
                        "label": label,
                        "session_id": session_id or "",
                        "lane": lane,
                    }
                    registered_waiter = True
                next_waiter_id = llm_gate_next_waiter_id()
                wait_for_slot = llm_gate_inflight >= max_concurrency
                wait_for_interval = (
                    lane != "realtime"
                    and llm_gate_last_started_at > 0
                    and now - llm_gate_last_started_at < min_interval
                )
                realtime_waiting = llm_gate_waiting_counts()["realtime"]
                # Background jobs (visualization/report) yield the model to realtime
                # voice/QA: only run once the realtime lane has been quiet for the
                # cooldown, but never defer longer than max_defer (avoid starvation).
                wait_for_idle = (
                    lane != "realtime"
                    and (now - started_waiting) < max_defer
                    and (
                        llm_gate_inflight_realtime > 0
                        or realtime_waiting > 0
                        or (llm_gate_last_realtime_at > 0 and now - llm_gate_last_realtime_at < idle_cooldown)
                    )
                )
                wait_for_priority = next_waiter_id != waiter_id
                if not wait_for_slot and not wait_for_interval and not wait_for_priority and not wait_for_idle:
                    llm_gate_waiters.pop(waiter_id, None)
                    registered_waiter = False
                    llm_gate_waiting = len(llm_gate_waiters)
                    llm_gate_inflight += 1
                    if lane == "realtime":
                        llm_gate_inflight_realtime += 1
                    else:
                        llm_gate_inflight_background += 1
                    llm_gate_last_started_at = now
                    if lane == "realtime":
                        llm_gate_last_realtime_at = now
                    acquired_slot = True
                    rec = llm_tasks.get(waiter_id)
                    if rec is not None:
                        rec["state"] = "running"
                    counts = llm_gate_waiting_counts()
                    emit_llm_gate_log(
                        (
                            f"LLM 请求放行：{label}；优先级={lane}；并发={llm_gate_inflight}/{max_concurrency} "
                            f"(实时={llm_gate_inflight_realtime} 后台={llm_gate_inflight_background})；"
                            f"排队={counts['total']} (实时={counts['realtime']} 后台={counts['background']})；"
                            f"最小间隔={min_interval:.1f}s{'，实时请求已绕过间隔' if lane == 'realtime' else ''}"
                        ),
                        session_id=session_id,
                    )
                    break
                llm_gate_waiting = len(llm_gate_waiters)
                wait_seconds = 1.0
                if wait_for_interval:
                    wait_seconds = max(0.5, min(min_interval - (now - llm_gate_last_started_at), 5.0))
                warn_size = max(1, int(settings.llm_queue_warn_size or 1))
                if not warned and llm_gate_waiting >= warn_size:
                    warned = True
                    counts = llm_gate_waiting_counts()
                    emit_llm_gate_log(
                        (
                            f"LLM 请求排队：{label}；优先级={lane}；当前并发={llm_gate_inflight}，"
                            f"排队={counts['total']} (实时={counts['realtime']} 后台={counts['background']})，"
                            f"限制={max_concurrency}，等待模型空闲"
                        ),
                        session_id=session_id,
                        level="warning",
                    )
                if wait_for_idle and not idle_notified:
                    idle_notified = True
                    emit_llm_gate_log(
                        f"后台任务让行实时语音：{label} 将在模型空闲时生成（语音/问答优先）",
                        session_id=session_id,
                    )
            await asyncio.sleep(wait_seconds)
    except LLMTaskCancelled:
        # 等待阶段被取消：未占放行槽，清理等待者与任务记录后中止。
        async with llm_gate_lock:
            llm_gate_waiters.pop(waiter_id, None)
            llm_gate_waiting = len(llm_gate_waiters)
            llm_tasks.pop(waiter_id, None)
        emit_llm_gate_log(f"后台任务已取消（等待中）：{label}", session_id=session_id, level="warning")
        raise
    finally:
        if registered_waiter:
            async with llm_gate_lock:
                llm_gate_waiters.pop(waiter_id, None)
                llm_gate_waiting = len(llm_gate_waiters)
        if not acquired_slot:
            llm_tasks.pop(waiter_id, None)
    retries = int(getattr(settings, "llm_realtime_retry", 1) or 0) if lane == "realtime" else 0
    try:
        usage_started_at = asyncio.get_running_loop().time()
        usage_event_id = record_llm_usage_start(label, session_id, lane, account_id, user_id)
        attempt = 0
        while True:
            try:
                # 用独立 future 包裹本次调用，便于「运行阶段取消」直接 cancel() 打断。
                inner = asyncio.ensure_future(call())
                rec = llm_tasks.get(waiter_id)
                if rec is not None:
                    rec["future"] = inner
                    if rec.get("cancel_requested"):
                        inner.cancel()   # 取消请求落在「拿到槽位~创建 future」窗口内时补刀
                result = await inner
                break
            except asyncio.CancelledError:
                record_llm_usage_finish(usage_event_id, "cancelled", error="cancelled by user", started_monotonic=usage_started_at)
                emit_llm_gate_log(f"后台任务已取消（运行中）：{label}", session_id=session_id, level="warning")
                raise LLMTaskCancelled()
            except Exception:
                if attempt < retries:
                    attempt += 1
                    emit_llm_gate_log(
                        f"实时请求失败，自动重试 {attempt}/{retries}：{label}",
                        session_id=session_id,
                        level="warning",
                    )
                    await asyncio.sleep(0.8)
                    continue
                raise
        record_llm_usage_finish(usage_event_id, "done", started_monotonic=usage_started_at)
        return result
    except LLMTaskCancelled:
        raise
    except Exception as exc:
        record_llm_usage_finish(usage_event_id, "failed", error=str(exc), started_monotonic=usage_started_at)
        raise
    finally:
        llm_tasks.pop(waiter_id, None)
        async with llm_gate_lock:
            llm_gate_inflight = max(0, llm_gate_inflight - 1)
            if lane == "realtime":
                llm_gate_inflight_realtime = max(0, llm_gate_inflight_realtime - 1)
                llm_gate_last_realtime_at = asyncio.get_running_loop().time()
            else:
                llm_gate_inflight_background = max(0, llm_gate_inflight_background - 1)


def analysis_needs_clarity_warning(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    return any(signal in text for signal in UNCLEAR_ANALYSIS_SIGNALS)


def row_to_dict(row) -> dict:
    return dict(row) if row is not None else {}


def analysis_public_columns() -> str:
    return "id, session_id, batch_id, scope, status, content, created_at, updated_at"


def image_public_columns() -> str:
    return (
        "id, session_id, batch_id, kind, filename, original_name, "
        "page_hint, question_hint, captured_at, sequence_index, created_at"
    )


def image_public_select(alias: str = "images") -> str:
    return (
        f"{alias}.id, {alias}.session_id, {alias}.batch_id, {alias}.kind, {alias}.filename, {alias}.original_name, "
        f"{alias}.page_hint, {alias}.question_hint, {alias}.captured_at, {alias}.sequence_index, {alias}.created_at, "
        "COALESCE(obs.novelty_status, 'unknown') AS novelty_status, "
        "COALESCE(obs.signal_summary, '') AS signal_summary"
    )


def thumbnail_path_for(filename: str) -> Path:
    return get_settings().data_dir / "thumbnails" / f"{Path(filename).stem}.jpg"


def create_thumbnail(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image.convert("RGB"), mask=image.getchannel("A"))
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        image.thumbnail((THUMBNAIL_MAX_SIDE, THUMBNAIL_MAX_SIDE), Image.Resampling.LANCZOS)
        out = BytesIO()
        image.save(out, format="JPEG", quality=THUMBNAIL_QUALITY, optimize=True)
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_bytes(out.getvalue())
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()


def image_path_for_request(filename: str) -> Path:
    if Path(filename).name != filename:
        raise HTTPException(400, "invalid filename")
    settings = get_settings()
    image_dir = (settings.data_dir / "images").resolve()
    path = (image_dir / filename).resolve()
    if not path.is_relative_to(image_dir):
        raise HTTPException(400, "invalid filename")
    if not path.is_file():
        raise HTTPException(404, "image not found")
    return path


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def truncate_text(value: object, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 20:
        return text[:max_chars]
    omitted = len(text) - max_chars
    marker = f"\n...[已截断约 {omitted} 字]...\n"
    remaining = max_chars - len(marker)
    if remaining <= 0:
        return text[:max_chars]
    head_chars = max(1, int(remaining * 0.7))
    tail_chars = max(0, remaining - head_chars)
    tail = text[-tail_chars:].lstrip() if tail_chars else ""
    return f"{text[:head_chars].rstrip()}{marker}{tail}"


def compact_capture_meta(raw: str | None, max_chars: int = FINAL_REPORT_CAPTURE_META_CHAR_LIMIT) -> str:
    text = (raw or "").strip()
    if not text:
        return "{}"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return truncate_text(text, max_chars)
    if not isinstance(data, dict):
        return truncate_text(text, max_chars)
    priority_keys = (
        "signal_summary",
        "presence_summary",
        "presenceSummary",
        "student_presence_status",
        "studentPresenceStatus",
        "student_presence",
        "studentPresence",
        "has_student_presence",
        "hasStudentPresence",
        "hand_count",
        "handCount",
        "face_count",
        "faceCount",
        "body_count",
        "bodyCount",
        "activity_summary",
        "activitySummary",
        "action_summary",
        "actionSummary",
        "page_hint",
        "pageHint",
        "question_hint",
        "questionHint",
        "motion_summary",
        "motionSummary",
        "ocr_summary",
        "ocrSummary",
    )
    compact = {key: data[key] for key in priority_keys if data.get(key) not in (None, "")}
    return truncate_text(json_dumps(compact or data), max_chars)


def capture_meta_dict(raw: str | None) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def parse_capture_meta(raw: str, expected_count: int) -> list[dict]:
    if not raw.strip():
        return [{} for _ in range(expected_count)]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [{} for _ in range(expected_count)]
    if isinstance(data, dict):
        data = data.get("frames") or data.get("images") or data.get("captures") or []
    if not isinstance(data, list):
        return [{} for _ in range(expected_count)]
    parsed: list[dict] = []
    for index in range(expected_count):
        item = data[index] if index < len(data) else {}
        parsed.append(item if isinstance(item, dict) else {})
    return parsed


def meta_string(meta: dict) -> str:
    return json_dumps(meta) if meta else ""


def meta_text(meta: dict, *keys: str) -> str:
    for key in keys:
        value = meta.get(key)
        if value is not None:
            return str(value)
    return ""


def meta_bool(meta: dict, *keys: str) -> bool | None:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y", "present", "visible"}:
                return True
            if normalized in {"false", "0", "no", "n", "absent", "none", "not_detected"}:
                return False
    return None


def meta_int(meta: dict, *keys: str) -> int | None:
    for key in keys:
        value = meta.get(key)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def meta_float(meta: dict, *keys: str) -> float | None:
    for key in keys:
        value = meta.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def meta_list(meta: dict, *keys: str) -> list:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return parsed
            return [part.strip() for part in re.split(r"[,，\s]+", value) if part.strip()]
    return []


def normalize_text_token(value: object) -> str:
    token = str(value or "").strip().replace(" ", "")
    return token[:80]


def normalized_text_tokens(meta: dict) -> list[str]:
    tokens = [normalize_text_token(item) for item in meta_list(meta, "text_tokens", "textTokens", "ocr_tokens", "ocrTokens")]
    seen: set[str] = set()
    result: list[str] = []
    for token in tokens:
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        result.append(token)
        if len(result) >= 80:
            break
    return result


def meta_dict(meta: dict, *keys: str) -> dict:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
    return {}


def image_quality_metrics(path: Path) -> dict:
    try:
        with Image.open(path) as image:
            gray = ImageOps.grayscale(image)
            gray.thumbnail((64, 64), Image.Resampling.BILINEAR)
            pixels = list(gray.getdata())
            width, height = gray.size
    except Exception:
        return {}
    if not pixels or width < 3 or height < 3:
        return {}
    light_pixels = sum(1 for value in pixels if value > 168)
    light_coverage = light_pixels / max(1, len(pixels))
    edge_pixels = 0
    contrast_total = 0.0
    edge_checks = 0
    for y in range(height - 1):
        for x in range(width - 1):
            value = pixels[y * width + x]
            right = pixels[y * width + x + 1]
            below = pixels[(y + 1) * width + x]
            for other in (right, below):
                diff = abs(int(value) - int(other))
                contrast_total += diff
                edge_checks += 1
                if diff > 18:
                    edge_pixels += 1
    edge_density = edge_pixels / max(1, edge_checks)
    contrast = contrast_total / max(1, edge_checks)
    return {
        "light_coverage": light_coverage,
        "edge_density": edge_density,
        "contrast": contrast,
    }


def normalized_quality_reasons(meta: dict) -> set[str]:
    return {str(reason).strip() for reason in meta_list(meta, "reasons", "quality_reasons", "qualityReasons") if str(reason).strip()}


def image_content_verdict(meta: dict, filename: str | None = None) -> dict:
    """Conservative global gate: reject only frames that are clearly unrelated or unusable."""
    source = (
        meta_dict(meta, "image_quality", "imageQuality", "frame_quality", "frameQuality")
        or meta_dict(meta, "qa_frame_quality", "qaFrameQuality")
        or meta
    )
    merged = dict(source)
    text_tokens = normalized_text_tokens(source) or normalized_text_tokens(meta)
    text_count = meta_int(source, "text_count", "textCount", "ocr_count", "ocrCount")
    rectangle_count = meta_int(source, "rectangle_count", "rectangleCount")
    material_confidence = meta_float(source, "material_confidence", "materialConfidence")
    blur_score = meta_float(source, "blur_score", "blurScore")
    light_coverage = meta_float(source, "light_coverage", "lightCoverage")
    edge_density = meta_float(source, "edge_density", "edgeDensity")
    contrast = meta_float(source, "contrast")

    if (
        filename
        and (
            light_coverage is None
            or edge_density is None
            or contrast is None
        )
    ):
        metrics = image_quality_metrics(image_path_for_request(filename))
        merged.update(metrics)
        light_coverage = meta_float(merged, "light_coverage", "lightCoverage")
        edge_density = meta_float(merged, "edge_density", "edgeDensity")
        contrast = meta_float(merged, "contrast")

    if text_count is None:
        text_count = len(text_tokens)
    if blur_score is None and edge_density is not None:
        blur_score = max(0.0, min(1.0, edge_density / 0.06))

    weak_text_count = max(text_count or 0, len(text_tokens))
    weak_rectangle_count = max(rectangle_count or 0, 0)
    has_material = meta_bool(source, "has_study_material", "hasStudyMaterial", "material_visible", "materialVisible")
    has_explicit_evidence = meta_bool(
        source,
        "has_explicit_study_evidence",
        "hasExplicitStudyEvidence",
        "explicit_study_evidence",
        "explicitStudyEvidence",
    )
    should_upload = meta_bool(source, "should_upload", "shouldUpload", "should_use", "shouldUse", "eligible")
    quality_status = meta_text(source, "quality_status", "qualityStatus", "status").strip().lower()
    reasons = normalized_quality_reasons(source)

    metrics_present = light_coverage is not None and edge_density is not None and contrast is not None
    usable_texture = bool(
        metrics_present
        and light_coverage >= IMAGE_VALIDITY_MIN_LIGHT_COVERAGE
        and edge_density >= IMAGE_VALIDITY_MIN_EDGE_DENSITY
        and contrast >= IMAGE_VALIDITY_MIN_CONTRAST
    )
    explicit_study_evidence = (
        has_explicit_evidence is True
        or weak_text_count >= IMAGE_VALIDITY_MIN_TEXT_TOKENS
        or weak_rectangle_count >= IMAGE_VALIDITY_MIN_RECTANGLES
    )
    material_visible = (
        has_material is True
        or explicit_study_evidence
        or (material_confidence is not None and material_confidence >= IMAGE_VALIDITY_STRONG_MATERIAL_CONFIDENCE)
        or (
            material_confidence is not None
            and material_confidence >= IMAGE_VALIDITY_SOFT_MATERIAL_CONFIDENCE
            and usable_texture
        )
    )

    payload = {
        "valid": True,
        "reason": "study_material_visible",
        "detail": "画面包含学习材料线索",
        "text_tokens": text_tokens,
        "text_count": weak_text_count,
        "rectangle_count": weak_rectangle_count,
        "material_confidence": material_confidence,
        "blur_score": blur_score,
        "light_coverage": light_coverage,
        "edge_density": edge_density,
        "contrast": contrast,
    }
    if material_visible and (explicit_study_evidence or usable_texture or has_material is True):
        return payload
    if should_upload is True and material_visible:
        return {**payload, "reason": "client_accepted"}

    explicit_rejection = (
        should_upload is False
        or has_material is False
        or has_explicit_evidence is False
        or quality_status in {"low_quality", "invalid", "rejected", "empty", "unrelated"}
        or bool(reasons & {"no_material", "too_blurry", "heavy_occlusion", "empty_frame", "unrelated"})
    )
    if explicit_rejection and not material_visible:
        return {
            **payload,
            "valid": False,
            "reason": "no_study_material",
            "detail": "画面无明确学习材料，已自动忽略",
        }

    if metrics_present:
        if (
            light_coverage < IMAGE_VALIDITY_MIN_LIGHT_COVERAGE
            or edge_density < IMAGE_VALIDITY_MIN_EDGE_DENSITY
            or contrast < IMAGE_VALIDITY_MIN_CONTRAST
        ) and weak_text_count == 0 and weak_rectangle_count == 0:
            return {
                **payload,
                "valid": False,
                "reason": "image_metrics_low",
                "detail": f"画面缺少可用纸面/屏幕纹理 edge={edge_density:.3f} contrast={contrast:.1f}",
            }
        if blur_score is not None and blur_score < IMAGE_VALIDITY_BLUR_MIN_WITHOUT_TEXT and weak_text_count == 0:
            return {
                **payload,
                "valid": False,
                "reason": "too_blurry",
                "detail": "画面模糊且没有可用文字线索，已自动忽略",
            }
        return {
            **payload,
            "valid": None,
            "reason": "unknown",
            "detail": "画面学习价值不确定，保守保留",
        }

    return {
        **payload,
        "valid": None,
        "reason": "unknown",
        "detail": "缺少足够的画面有效性元数据，保守保留",
    }


def capture_meta_with_image_verdict(meta: dict, verdict: dict) -> dict:
    next_meta = dict(meta or {})
    next_meta["image_content_verdict"] = {
        "valid": verdict.get("valid"),
        "reason": verdict.get("reason") or "",
        "detail": verdict.get("detail") or "",
        "text_count": verdict.get("text_count") or 0,
        "rectangle_count": verdict.get("rectangle_count") or 0,
        "material_confidence": verdict.get("material_confidence"),
        "blur_score": verdict.get("blur_score"),
        "edge_density": verdict.get("edge_density"),
        "contrast": verdict.get("contrast"),
        "light_coverage": verdict.get("light_coverage"),
    }
    next_meta["image_content_valid"] = verdict.get("valid")
    next_meta["image_content_reason"] = verdict.get("reason") or ""
    if verdict.get("valid") is False:
        next_meta["discarded_before_analysis"] = True
        next_meta["discard_reason"] = verdict.get("reason") or "invalid_image"
        next_meta["discard_detail"] = verdict.get("detail") or ""
    return next_meta


def qa_frame_quality_from_meta(meta: dict) -> dict:
    qa_context = meta_dict(meta, "qa_context", "qaContext", "context")
    qa_quality = (
        meta_dict(meta, "qa_frame_quality", "qaFrameQuality", "frame_quality", "frameQuality")
        or meta_dict(qa_context, "qa_frame_quality", "qaFrameQuality", "frame_quality", "frameQuality")
    )
    source = qa_quality if qa_quality else meta
    text_tokens = normalized_text_tokens(source) or normalized_text_tokens(meta) or normalized_text_tokens(qa_context)
    eligible = meta_bool(source, "qa_context_eligible", "qaContextEligible", "eligible", "should_use", "shouldUse")
    client_submitted_for_qa = meta_bool(source, "should_upload_for_qa", "shouldUploadForQA", "submitted_current_frame", "submittedCurrentFrame")
    has_material = meta_bool(source, "has_study_material", "hasStudyMaterial", "material_visible", "materialVisible")
    reliable_qa_context = meta_bool(source, "qa_reliable_context", "qaReliableContext", "reliable_qa_context", "reliableQaContext")
    has_explicit_evidence = meta_bool(source, "has_explicit_study_evidence", "hasExplicitStudyEvidence", "explicit_study_evidence", "explicitStudyEvidence")
    text_count = meta_int(source, "text_count", "textCount", "ocr_count", "ocrCount")
    rectangle_count = meta_int(source, "rectangle_count", "rectangleCount")
    material_confidence = meta_float(source, "material_confidence", "materialConfidence")
    blur_score = meta_float(source, "blur_score", "blurScore")
    light_coverage = meta_float(source, "light_coverage", "lightCoverage")
    edge_density = meta_float(source, "edge_density", "edgeDensity")
    contrast = meta_float(source, "contrast")
    reasons = [str(reason) for reason in meta_list(source, "reasons", "quality_reasons", "qualityReasons") if str(reason).strip()]
    weak_text_count = 0 if text_count is None else text_count
    weak_rectangle_count = 0 if rectangle_count is None else rectangle_count
    student_intent = normalize_qa_student_intent(
        meta_text(qa_context, "student_intent", "studentIntent", "intent_hint", "intentHint", "qa_intent", "qaIntent", "intent")
        or meta_text(meta, "student_intent", "studentIntent", "intent_hint", "intentHint", "qa_intent", "qaIntent", "intent")
    )
    is_visual_review_intent = student_intent in QA_VISUAL_REVIEW_INTENTS
    has_reliable_structure = (
        weak_text_count >= QA_MIN_TEXT_FOR_CONTEXT
        or (weak_text_count >= QA_MIN_TEXT_WITH_RECTANGLE_FOR_CONTEXT and weak_rectangle_count >= 1)
        or weak_rectangle_count >= QA_MIN_RECTANGLES_WITHOUT_TEXT_FOR_CONTEXT
    )
    has_review_material_evidence = (
        is_visual_review_intent
        and (
            has_material is True
            or has_explicit_evidence is True
            or reliable_qa_context is True
            or (material_confidence is not None and material_confidence >= QA_IMAGE_MATERIAL_CONFIDENCE_MIN)
        )
        and (
            weak_text_count >= 1
            or weak_rectangle_count >= 1
            or (
                light_coverage is not None
                and edge_density is not None
                and contrast is not None
                and light_coverage >= QA_IMAGE_LIGHT_COVERAGE_MIN
                and edge_density >= QA_IMAGE_EDGE_DENSITY_MIN
                and contrast >= QA_IMAGE_CONTRAST_MIN
            )
        )
    )
    if eligible is False and has_review_material_evidence:
        return {
            "eligible": True,
            "reason": "visual_review_material",
            "detail": "学生请求核对/订正，当前画面含学习材料证据",
            "text_tokens": text_tokens,
        }
    if eligible is False and client_submitted_for_qa is not True:
        return {
            "eligible": False,
            "reason": "client_rejected",
            "detail": ";".join(reasons) or "客户端判断当前画面不适合做问答上下文",
            "text_tokens": text_tokens,
        }
    if reliable_qa_context is False and not has_review_material_evidence:
        return {
            "eligible": False,
            "reason": "no_explicit_study_evidence",
            "detail": "当前抓拍缺少文字、题目边框或屏幕特征",
            "text_tokens": text_tokens,
        }
    if has_explicit_evidence is False and not has_review_material_evidence:
        return {
            "eligible": False,
            "reason": "no_explicit_study_evidence",
            "detail": "当前抓拍缺少明确学习材料证据",
            "text_tokens": text_tokens,
        }
    if (text_count is not None or rectangle_count is not None) and not has_reliable_structure and not has_review_material_evidence:
        return {
            "eligible": False,
            "reason": "no_explicit_study_evidence",
            "detail": "当前抓拍没有足够题目文字或纸张/屏幕结构，已沿用已有上下文",
            "text_tokens": text_tokens,
        }
    if eligible is True:
        return {
            "eligible": True,
            "reason": "client_accepted",
            "detail": "客户端判断当前画面适合做问答上下文",
            "text_tokens": text_tokens,
        }
    if has_review_material_evidence:
        return {
            "eligible": True,
            "reason": "visual_review_material",
            "detail": "学生请求核对/订正，当前画面含学习材料证据",
            "text_tokens": text_tokens,
        }
    if has_material is False:
        return {
            "eligible": False,
            "reason": "no_study_material",
            "detail": "客户端未检测到课本、试卷或电子屏幕",
            "text_tokens": text_tokens,
        }
    if len(text_tokens) >= QA_MIN_TEXT_FOR_CONTEXT:
        return {"eligible": True, "reason": "ocr_tokens", "detail": "检测到可用文字", "text_tokens": text_tokens}
    if len(text_tokens) >= QA_MIN_TEXT_WITH_RECTANGLE_FOR_CONTEXT and weak_rectangle_count >= 1:
        return {"eligible": True, "reason": "ocr_tokens_with_structure", "detail": "检测到文字和题目结构", "text_tokens": text_tokens}
    if len(text_tokens) > 0:
        return {
            "eligible": False,
            "reason": "weak_ocr_evidence",
            "detail": "当前抓拍文字线索不足，已沿用已有上下文",
            "text_tokens": text_tokens,
        }
    if material_confidence is not None and material_confidence >= QA_IMAGE_MATERIAL_CONFIDENCE_MIN:
        if weak_rectangle_count >= QA_MIN_RECTANGLES_WITHOUT_TEXT_FOR_CONTEXT:
            return {"eligible": True, "reason": "material_confidence", "detail": f"学习材料置信度 {material_confidence:.2f}", "text_tokens": text_tokens}
        return {
            "eligible": False,
            "reason": "weak_visual_evidence",
            "detail": "当前抓拍只有弱视觉特征，已沿用已有上下文",
            "text_tokens": text_tokens,
        }
    if has_material is True and (blur_score is None or blur_score >= 0.25):
        if weak_rectangle_count >= QA_MIN_RECTANGLES_WITHOUT_TEXT_FOR_CONTEXT or len(text_tokens) >= QA_MIN_TEXT_WITH_RECTANGLE_FOR_CONTEXT:
            return {"eligible": True, "reason": "material_visible", "detail": "客户端检测到学习材料可见", "text_tokens": text_tokens}
        return {
            "eligible": False,
            "reason": "weak_visual_evidence",
            "detail": "当前抓拍缺少可用题目信息，已沿用已有上下文",
            "text_tokens": text_tokens,
        }
    if light_coverage is not None and edge_density is not None and contrast is not None:
        if (
            light_coverage >= QA_IMAGE_LIGHT_COVERAGE_MIN
            and edge_density >= QA_IMAGE_EDGE_DENSITY_MIN
            and contrast >= QA_IMAGE_CONTRAST_MIN
            and has_reliable_structure
        ):
            return {
                "eligible": True,
                "reason": "image_metrics",
                "detail": f"图像纹理/亮度可用 edge={edge_density:.3f} contrast={contrast:.1f}",
                "text_tokens": text_tokens,
            }
        return {
            "eligible": False,
            "reason": "image_metrics_low",
            "detail": f"图像缺少清晰纸面/屏幕特征 edge={edge_density:.3f} contrast={contrast:.1f}",
            "text_tokens": text_tokens,
        }
    return {"eligible": None, "reason": "unknown", "detail": "缺少足够的画面质量元数据", "text_tokens": text_tokens}


def qa_uploaded_frame_quality(meta: dict, filename: str | None) -> dict:
    quality = qa_frame_quality_from_meta(meta)
    if quality.get("eligible") is not None:
        return quality
    if filename:
        metrics = image_quality_metrics(image_path_for_request(filename))
        if metrics:
            merged = dict(meta)
            merged.update(metrics)
            return qa_frame_quality_from_meta(merged)
    return {
        "eligible": False,
        "reason": quality.get("reason") or "unknown",
        "detail": quality.get("detail") or "无法确认当前画面是否包含清晰题目",
        "text_tokens": quality.get("text_tokens") or [],
    }


def qa_turn_index(context: dict) -> int:
    try:
        return int(context.get("turn") or context.get("qa_turn") or context.get("qaTurn") or 0)
    except (TypeError, ValueError):
        return 0


def qa_question_has_new_problem_reference(question: str) -> bool:
    text = (question or "").strip().lower()
    if not text:
        return False
    patterns = (
        r"第\s*[\d一二三四五六七八九十]+\s*[题頁页]",
        r"[\d一二三四五六七八九十]+\s*[题頁页]",
        r"problem\s*\d+",
        r"question\s*\d+",
        r"page\s*\d+",
        r"换[一個个]?题",
        r"下一题",
        r"上一题",
    )
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def qa_current_frame_was_submitted(context: dict) -> bool:
    return meta_bool(
        context or {},
        "current_frame_submitted",
        "currentFrameSubmitted",
        "submitted_current_frame",
        "submittedCurrentFrame",
    ) is True


def qa_is_first_or_new_problem_turn(trigger: str, context: dict, question: str) -> bool:
    return qa_turn_index(context) <= 1 or qa_question_has_new_problem_reference(question) or not qa_is_followup_like(trigger, context, question)


def qa_context_string(context: dict, *keys: str) -> str:
    for key in keys:
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_qa_student_intent(value: object) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "review": "answer_check",
        "answer_review": "answer_check",
        "check_answer": "answer_check",
        "answer_check": "answer_check",
        "correction": "correction_check",
        "correction_review": "correction_check",
        "check_correction": "correction_check",
        "correction_check": "correction_check",
        "visual": "visual_check",
        "look": "visual_check",
        "look_check": "visual_check",
        "visual_check": "visual_check",
        "followup": "followup_explain",
        "follow_up": "followup_explain",
        "explain": "followup_explain",
        "followup_explain": "followup_explain",
        "new": "new_question",
        "new_question": "new_question",
        "normal": "new_question",
        "normal_qa": "new_question",
    }
    return aliases.get(raw, "")


def infer_qa_student_intent(trigger: str, context: dict, question: str) -> str:
    explicit = normalize_qa_student_intent(
        qa_context_string(
            context,
            "student_intent",
            "studentIntent",
            "qa_intent",
            "qaIntent",
            "intent_hint",
            "intentHint",
            "intent",
            "intent_type",
            "intentType",
        )
    )
    if explicit:
        return explicit
    text = " ".join(
        part
        for part in (
            question or "",
            qa_context_string(context, "transcript", "recognized_text", "recognizedText"),
            qa_context_string(context, "intent_hint", "intentHint"),
        )
        if part
    ).strip()
    compact = re.sub(r"\s+", "", text.lower())
    if not compact:
        return "followup_explain" if qa_is_followup_like(trigger, context, question) else "new_question"

    correction_terms = (
        "改完",
        "改好了",
        "改好",
        "修改",
        "改了",
        "订正",
        "修正",
        "重写",
        "重新写",
        "写完",
        "做完",
        "算完",
        "填完",
        "改正",
    )
    check_terms = (
        "看看",
        "看下",
        "看一下",
        "再看",
        "检查",
        "核对",
        "批改",
        "对不对",
        "对了吗",
        "对了没",
        "是否正确",
        "是不是对",
        "有没有错",
        "还错",
        "还对",
        "正确吗",
        "可以吗",
        "行不行",
    )
    answer_terms = ("答案", "结果", "步骤", "过程", "这道", "这题", "这里", "这步")
    visual_terms = ("看", "图片", "照片", "拍照", "画面", "镜头", "图上", "这张")
    if any(term in compact for term in correction_terms) and any(term in compact for term in check_terms):
        return "correction_check"
    if any(term in compact for term in check_terms) and any(term in compact for term in answer_terms):
        return "answer_check"
    if any(term in compact for term in ("check", "correct", "right", "wrong")) and any(term in compact for term in ("answer", "work", "solution", "again", "fixed", "changed")):
        return "correction_check" if any(term in compact for term in ("fixed", "changed", "corrected", "revised")) else "answer_check"
    if any(term in compact for term in visual_terms) and any(term in compact for term in check_terms):
        return "visual_check"
    if qa_question_has_new_problem_reference(question):
        return "new_question"
    if qa_is_followup_like(trigger, context, question):
        return "followup_explain"
    return "new_question"


def qa_is_followup_like(trigger: str, context: dict, question: str) -> bool:
    normalized_trigger = (trigger or "").strip().lower()
    if "follow" in normalized_trigger or "ok" in normalized_trigger:
        return True
    if qa_turn_index(context) >= 2:
        return True
    text = (question or "").strip()
    followup_terms = ("继续", "刚才", "这里", "这个", "这步", "为什么", "哪里错", "再讲", "不懂", "追问")
    return any(term in text for term in followup_terms)


def qa_token_overlap_score(current_tokens: list[str], previous_tokens: list[str]) -> float:
    current = {normalize_text_token(token).lower() for token in current_tokens if normalize_text_token(token)}
    previous = {normalize_text_token(token).lower() for token in previous_tokens if normalize_text_token(token)}
    if not current or not previous:
        return 0.0
    exact = len(current & previous)
    partial = 0
    for token in current:
        if token in previous:
            continue
        if any(token in prior or prior in token for prior in previous if len(token) >= 2 and len(prior) >= 2):
            partial += 1
    return (exact + partial * 0.5) / max(1, min(len(current), len(previous)))


def qa_quality_with_relevance(
    quality: dict,
    *,
    question: str,
    trigger: str,
    context: dict,
    student_intent: str = "",
    previous_image_row: dict | None = None,
) -> dict:
    if quality.get("eligible") is not True:
        return quality
    current_tokens = [str(token) for token in quality.get("text_tokens") or [] if str(token).strip()]
    if qa_is_first_or_new_problem_turn(trigger, context, question):
        return {**quality, "relevance": "accepted_first_or_new_problem_current_frame"}
    if not previous_image_row:
        return {**quality, "relevance": "accepted_no_previous_frame"}
    if student_intent in QA_VISUAL_REVIEW_INTENTS:
        return {**quality, "relevance": f"accepted_{student_intent}_current_frame"}
    previous_tokens = normalized_text_tokens(capture_meta_dict(previous_image_row.get("capture_meta")))
    if len(current_tokens) < QA_MIN_TEXT_FOR_CONTEXT:
        return {
            **quality,
            "eligible": False,
            "reason": "weak_followup_frame",
            "detail": "追问抓拍文字线索不足，已沿用上一轮上下文",
            "relevance": "rejected_weak_followup_frame",
            "previous_text_tokens": previous_tokens[:12],
        }
    if len(previous_tokens) < 2:
        return {**quality, "relevance": "accepted_insufficient_previous_ocr_for_overlap"}
    overlap = qa_token_overlap_score(current_tokens, previous_tokens)
    if overlap < 0.12:
        return {
            **quality,
            "eligible": False,
            "reason": "unrelated_followup_frame",
            "detail": f"追问画面与上一张学习画面文字线索不匹配 overlap={overlap:.2f}",
            "relevance": "rejected_followup_token_mismatch",
            "previous_text_tokens": previous_tokens[:12],
        }
    return {**quality, "relevance": "accepted_followup_overlap", "overlap": overlap}


def qa_should_soft_accept_first_frame(quality: dict, *, question: str, trigger: str, context: dict, student_intent: str = "") -> bool:
    if quality.get("eligible") is True:
        return False
    if student_intent in QA_VISUAL_REVIEW_INTENTS:
        return False
    if not qa_current_frame_was_submitted(context):
        return False
    if not qa_is_first_or_new_problem_turn(trigger, context, question):
        return False
    frame_quality = meta_dict(context or {}, "qa_frame_quality", "qaFrameQuality", "frame_quality", "frameQuality")
    if meta_bool(frame_quality, "has_study_material", "hasStudyMaterial", "material_visible", "materialVisible") is False:
        return False
    reasons = {str(reason) for reason in meta_list(frame_quality, "reasons", "quality_reasons", "qualityReasons") if str(reason).strip()}
    if {"no_material", "phone_away"} & reasons:
        return False
    reason = str(quality.get("reason") or "")
    if reason in {"no_study_material", "client_rejected", "image_metrics_low", "unknown"}:
        return False
    tokens = [str(token) for token in quality.get("text_tokens") or [] if str(token).strip()]
    if len(tokens) >= QA_MIN_TEXT_WITH_RECTANGLE_FOR_CONTEXT:
        return True
    if meta_bool(frame_quality, "has_study_material", "hasStudyMaterial", "material_visible", "materialVisible") is not True:
        return False
    detail = str(quality.get("detail") or "")
    return any(marker in detail for marker in ("文字", "题目", "结构", "学习材料", "课本", "试卷", "屏幕"))


def _qa_fingerprint_meta(meta: dict) -> dict:
    """QA frames carry their visual_sample/text tokens under qa_frame_quality; fall
    back to that nested dict when the top level has no fingerprint."""
    if visual_sample_from_meta(meta) or normalized_text_tokens(meta):
        return meta
    nested = meta_dict(meta, "qa_frame_quality", "qaFrameQuality", "frame_quality", "frameQuality")
    return nested or meta


def qa_frame_duplicates_previous(meta: dict, previous_image_row: dict | None) -> bool:
    """True when an eligible follow-up frame is essentially the same page as the
    previous QA image, so re-sending it to the vision model adds nothing and we can
    answer on the carried text context (faster realtime voice path). Conservative:
    reuses the observation dedup distance and additionally requires high text overlap."""
    if not previous_image_row:
        return False
    current = _qa_fingerprint_meta(meta)
    previous = _qa_fingerprint_meta(capture_meta_dict(previous_image_row.get("capture_meta")))
    distance = fingerprint_distance(visual_sample_from_meta(current), visual_sample_from_meta(previous))
    if distance is None:
        current_hash = visual_hash_from_meta(current)
        previous_hash = visual_hash_from_meta(previous)
        return bool(current_hash and current_hash == previous_hash)
    if distance > VISUAL_DUPLICATE_DISTANCE:
        return False
    current_tokens = normalized_text_tokens(current)
    previous_tokens = normalized_text_tokens(previous)
    if not current_tokens or not previous_tokens:
        return True
    return qa_token_overlap_score(current_tokens, previous_tokens) >= QA_DUPLICATE_FOLLOWUP_TOKEN_OVERLAP


def qa_context_rejected_from_meta(meta: dict) -> bool:
    if meta_bool(meta, "qa_context_rejected", "qaContextRejected") is True:
        return True
    quality = qa_frame_quality_from_meta(meta)
    if quality.get("eligible") is False:
        return True
    qa_quality = meta_dict(meta, "qa_frame_quality", "qaFrameQuality", "frame_quality", "frameQuality")
    if meta_bool(qa_quality, "qa_context_rejected", "qaContextRejected") is True:
        return True
    return False


def update_qa_image_context_verdict(image_id: str | None, quality: dict, *, accepted: bool) -> None:
    if not image_id:
        return
    with connect() as conn:
        row = conn.execute("SELECT capture_meta FROM images WHERE id=?", (image_id,)).fetchone()
        if not row:
            return
        meta = capture_meta_dict(row["capture_meta"])
        meta["qa_context_accepted"] = bool(accepted)
        meta["qa_context_rejected"] = not bool(accepted)
        meta["qa_context_verdict"] = "accepted" if accepted else "rejected"
        meta["qa_context_rejected_reason"] = "" if accepted else str(quality.get("reason") or "unknown")
        meta["qa_context_rejected_detail"] = "" if accepted else str(quality.get("detail") or "")
        if quality:
            merged_quality = dict(meta_dict(meta, "qa_frame_quality", "qaFrameQuality"))
            merged_quality.update(quality)
            merged_quality["qa_context_eligible"] = bool(accepted)
            meta["qa_frame_quality"] = merged_quality
        conn.execute("UPDATE images SET capture_meta=? WHERE id=?", (meta_string(meta), image_id))


def qa_prompt_context(context: dict, *, current_image_rejected: bool) -> dict:
    prompt_context = dict(context or {})
    if current_image_rejected:
        for key in ("qa_frame_quality", "qaFrameQuality", "frame_quality", "frameQuality"):
            prompt_context.pop(key, None)
        prompt_context["current_image_rejected"] = True
        prompt_context["current_image_note"] = "The newly captured QA frame was rejected before prompt construction; do not use its OCR or visual metadata."
    return prompt_context


def build_context_trace(
    *,
    prompt_context: dict,
    context_payload: dict,
    student_intent: str,
    turn: int,
    image_filename: str,
    image_id: str,
    image_context_mode: str,
    current_image_rejected: bool,
    retrieved_memories: list,
    memory_gated_off: bool,
) -> dict:
    """Read-only observability side-channel: a per-turn snapshot of *which context
    channels the prompt actually carried* and the durable-memory score breakdown.

    Pure: derives everything from already-computed turn signals, mutates nothing, and
    never affects the answer. `included` = "this channel made it into this turn's prompt"
    (inferred from the assembled `prompt_context`, which only contains a channel's keys
    when the client did not toggle it off). No `requested` dimension, no subject field
    (MUST-2 / MUST-7). New top-level QA key; old clients ignore it."""
    ctx = prompt_context if isinstance(prompt_context, dict) else {}
    raw = context_payload if isinstance(context_payload, dict) else {}

    # visual: a frame was actually selected into the prompt for this turn.
    visual_included = bool(image_filename)
    visual_detail: dict = {}
    if visual_included or current_image_rejected:
        visual_detail = {
            "image_id": image_id or "",
            "filename": image_filename or "",
            "mode": image_context_mode or "",
            "rejected": bool(current_image_rejected),
        }

    # history: carried conversation context present in the prompt.
    history_text = ctx.get("carried_history_context")
    if isinstance(history_text, dict):
        history_chars = len(str(history_text.get("text") or json_dumps(history_text)))
    else:
        history_chars = len(str(history_text or ""))
    history_included = bool(history_text)
    # human-readable preview of the actual history text the prompt carried (parents/students
    # asked to "see the real content", not just a char count). Built from the same dict the
    # model received; truncated to ~200 chars. (Read-only; no new fields sent to the model.)
    history_preview = ""
    if history_included:
        if isinstance(history_text, dict):
            parts = []
            title = str(history_text.get("title") or "").strip()
            summary = str(history_text.get("summary") or "").strip()
            if title:
                parts.append(f"来源：{title}")
            if summary:
                parts.append(f"摘要：{summary}")
            for key in ("recent_questions", "recent_answers", "mistakes", "learning_items"):
                snippets = history_text.get(key)
                if isinstance(snippets, list):
                    parts.extend(str(s).strip() for s in snippets if str(s or "").strip())
            history_preview = truncate_text("\n".join(p for p in parts if p), 200)
        else:
            history_preview = truncate_text(str(history_text or ""), 200)

    # mistakes: review context carried for this turn.
    mistakes_assets = [
        a for a in (ctx.get("structured_context_assets") or [])
        if isinstance(a, dict) and a.get("kind") == "mistake"
    ]
    mistakes_included = bool(ctx.get("review_context")) or bool(mistakes_assets)
    mistakes_count = len(mistakes_assets) + (1 if ctx.get("review_context") else 0)
    # human-readable list of the actual mistakes carried this turn: the active review item
    # (if any) plus the structured mistake assets the client attached. title/detail only.
    mistakes_items: list[dict] = []
    review_ctx = ctx.get("review_context")
    if isinstance(review_ctx, dict):
        review_item = review_ctx.get("item") if isinstance(review_ctx.get("item"), dict) else {}
        review_title = str(
            review_item.get("title")
            or review_item.get("question_text")
            or review_item.get("displayTitle")
            or "今日复习错题"
        ).strip()
        review_detail = str(
            review_item.get("error_reason")
            or review_item.get("error_type")
            or review_item.get("next_action")
            or review_item.get("location")
            or ""
        ).strip()
        if review_title:
            mistakes_items.append({
                "title": truncate_text(review_title, 60),
                "detail": truncate_text(review_detail, 120),
                "active": True,
            })
    for a in mistakes_assets[:6]:
        title = str(a.get("title") or "").strip()
        detail = str(a.get("detail") or "").strip()
        if title or detail:
            mistakes_items.append({
                "title": truncate_text(title or "相关错题", 60),
                "detail": truncate_text(detail, 120),
                "active": False,
            })

    # knowledge: semantic knowledge hits attached server-side this turn.
    semantic_knowledge = ctx.get("semantic_knowledge") if isinstance(ctx.get("semantic_knowledge"), list) else []
    knowledge_assets = [
        a for a in (ctx.get("structured_context_assets") or [])
        if isinstance(a, dict) and a.get("kind") == "knowledge"
    ]
    knowledge_included = bool(semantic_knowledge) or bool(knowledge_assets)
    knowledge_hits = [
        {
            "kind": str(h.get("kind") or ""),
            "score": h.get("score"),
            "preview": truncate_text(str(h.get("text") or ""), 80),
        }
        for h in semantic_knowledge[:5]
        if isinstance(h, dict)
    ]

    # memory: durable agent memories retrieved for this turn (with breakdown).
    memory_list = retrieved_memories if isinstance(retrieved_memories, list) else []
    memory_items = [
        {
            "id": str(m.get("id") or ""),
            "kind": str(m.get("kind") or ""),
            "text": truncate_text(str(m.get("text") or ""), 120),
            "score": m.get("score"),
            "breakdown": m.get("breakdown") if isinstance(m.get("breakdown"), dict) else {},
        }
        for m in memory_list
        if isinstance(m, dict) and m.get("text")
    ]
    memory_included = (not memory_gated_off) and bool(memory_items)

    # observation: background observation context carried this turn.
    observation = ctx.get("observation_context") if isinstance(ctx.get("observation_context"), dict) else {}
    observation_included = bool(observation)

    # strategy: dynamic strategy / coach preference carried this turn.
    # 只用受开关约束的 dynamic_strategy 判定 included；strategy_context 恒在顶层不能作依据(否则关策略仍误报)
    strategy_included = bool(ctx.get("dynamic_strategy"))

    return {
        "version": 1,
        "turn": turn,
        "student_intent": student_intent,
        "channels": [
            {"key": "visual", "included": visual_included, "detail": visual_detail},
            {"key": "history", "included": history_included, "detail": {"chars": history_chars, "preview": history_preview} if history_included else {}},
            {"key": "mistakes", "included": mistakes_included, "detail": {"count": mistakes_count, "items": mistakes_items} if mistakes_included else {}},
            {"key": "knowledge", "included": knowledge_included, "detail": {"semantic_hits": knowledge_hits} if knowledge_included else {}},
            {
                "key": "memory",
                "included": memory_included,
                "detail": {
                    "gated_off": bool(memory_gated_off),
                    "weights": {
                        "semantic": memory_store.W_SEMANTIC,
                        "recency": memory_store.W_RECENCY,
                        "importance": memory_store.W_IMPORTANCE,
                        "usage": memory_store.W_USAGE,
                    },
                    "memories": [] if memory_gated_off else memory_items,
                },
            },
            {"key": "observation", "included": observation_included, "detail": {} if not observation_included else {"frames": observation.get("buffered_frame_count") or 0}},
            {
                "key": "strategy",
                "included": strategy_included,
                "detail": {
                    "learning_mode": raw.get("learning_mode_title") or raw.get("learning_mode") or None,
                    "coach_depth": raw.get("coach_depth_title") or raw.get("coach_depth") or None,
                } if strategy_included else {},
            },
        ],
    }


def dynamic_strategy_context(
    session: dict,
    context: dict,
    *,
    question: str,
    trigger_type: str,
    image_row: dict | None,
    image_context_mode: str,
) -> str:
    client_dynamic = context.get("dynamic_strategy") if isinstance(context.get("dynamic_strategy"), dict) else {}
    strategy_context = context.get("strategy_context") if isinstance(context.get("strategy_context"), dict) else {}
    observation = context.get("observation_context") if isinstance(context.get("observation_context"), dict) else {}
    frame_quality = context.get("qa_frame_quality") if isinstance(context.get("qa_frame_quality"), dict) else {}
    agent_memories = context.get("agent_memories") if isinstance(context.get("agent_memories"), list) else []
    memory_candidates = context.get("memory_digest_candidates") or context.get("long_term_memory_candidates") or []
    if not isinstance(memory_candidates, list):
        memory_candidates = []
    formed_memories = important_memory_events(5)
    lines = [
        f"本轮触发：{trigger_type}；turn={context.get('turn') or client_dynamic.get('turn') or 'unknown'}；intent={context.get('student_intent') or 'unknown'}",
        f"用户当前问题：{truncate_text(question, 180)}",
        f"偏好：场景={context.get('learning_mode_title') or strategy_context.get('learning_mode_title') or '未指定'}；回答方式={context.get('coach_depth_title') or strategy_context.get('coach_depth_title') or '未指定'}",
        f"当前画面：mode={image_context_mode}；selected_image_id={(image_row or {}).get('id') or context.get('selected_image_id') or 'none'}；submitted={context.get('current_frame_submitted')}",
        f"画面质量：eligible={frame_quality.get('qa_context_eligible') or frame_quality.get('eligible')}；study_material={frame_quality.get('has_study_material')}；summary={truncate_text(frame_quality.get('signal_summary') or frame_quality.get('message') or '', 180)}",
        f"观察状态：active={observation.get('is_active') or context.get('is_observing')}；frames={observation.get('buffered_frame_count') or 0}；state={observation.get('upload_state') or ''}",
    ]
    if context.get("student_goal") or session.get("student_goal"):
        lines.append(f"本轮目标：{truncate_text(context.get('student_goal') or session.get('student_goal') or '', 240)}")
    if context.get("carried_history_context"):
        lines.append(f"历史上下文：{truncate_text(context.get('carried_history_context'), 360)}")
    if context.get("review_context"):
        lines.append(f"复习上下文：{truncate_text(context.get('review_context'), 360)}")
    if formed_memories:
        lines.append(
            "重点形成记忆："
            + "；".join(truncate_text(event.get("text"), 150) for event in formed_memories)
        )
    if agent_memories:
        lines.append(
            "相关记忆（按语义+新近度+重要性检索）："
            + "；".join(
                truncate_text(m.get("text"), 120)
                for m in agent_memories[:5]
                if isinstance(m, dict) and m.get("text")
            )
        )
    elif memory_candidates:
        lines.append("记忆整理候选：" + "；".join(truncate_text(item, 120) for item in memory_candidates[:4]))
    if client_dynamic:
        lines.append(f"客户端动态策略：{truncate_text(json_dumps(client_dynamic), 700)}")
    lines.append("执行：综合本轮偏好、当前画面、当前回合、观察状态、历史上下文和记忆整理；优先回答用户当前问题，不让旧上下文覆盖新问题。")
    return truncate_text("\n".join(lines), 1800)


def qa_rejected_client_frame_quality(context: dict) -> dict | None:
    quality = qa_frame_quality_from_meta(context or {})
    if quality.get("eligible") is False:
        return quality
    return None


def short_hash(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def text_tokens_hash(tokens: list[str]) -> str:
    return short_hash("\n".join(sorted(tokens)))


def visual_sample_from_meta(meta: dict) -> list[int]:
    values = meta_list(meta, "visual_sample", "visualSample", "fingerprint_values", "fingerprintValues")
    sample: list[int] = []
    for value in values[:1024]:
        try:
            sample.append(max(0, min(255, int(value))))
        except (TypeError, ValueError):
            continue
    return sample


def parse_visual_sample(raw: str | None) -> list[int]:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    sample: list[int] = []
    for value in data[:1024]:
        try:
            sample.append(max(0, min(255, int(value))))
        except (TypeError, ValueError):
            continue
    return sample


def visual_hash_from_sample(sample: list[int]) -> str:
    if not sample:
        return ""
    mean = sum(sample) / len(sample)
    bits = "".join("1" if value >= mean else "0" for value in sample)
    if not bits:
        return ""
    return f"{int(bits, 2):0{(len(bits) + 3) // 4}x}"


def visual_hash_from_meta(meta: dict) -> str:
    explicit = meta_text(meta, "visual_hash", "visualHash", "fingerprint_hash", "fingerprintHash")
    if explicit:
        return explicit[:256]
    return visual_hash_from_sample(visual_sample_from_meta(meta))


def fingerprint_distance(values: list[int], other: list[int]) -> float | None:
    if not values or not other or len(values) != len(other):
        return None
    mean = sum(values) / len(values)
    other_mean = sum(other) / len(other)
    total = sum(abs((lhs - mean) - (rhs - other_mean)) for lhs, rhs in zip(values, other))
    return total / len(values)


def text_token_distance(tokens: list[str], other: list[str]) -> float | None:
    if not tokens or not other:
        return None
    lhs = set(tokens)
    rhs = set(other)
    union = lhs | rhs
    if not union:
        return None
    return 1.0 - (len(lhs & rhs) / len(union))


def visual_hash_distance(lhs: str, rhs: str) -> int | None:
    if not lhs or not rhs or len(lhs) != len(rhs):
        return None
    try:
        return (int(lhs, 16) ^ int(rhs, 16)).bit_count()
    except ValueError:
        return None


def best_previous_observation(session_id: str, visual_hash: str, visual_sample: list[int], text_tokens: list[str]) -> dict | None:
    if not (visual_hash or visual_sample or text_tokens):
        return None
    with connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT image_id, visual_hash, visual_sample, text_tokens, sequence_index, captured_at, novelty_status
                FROM session_observations
                WHERE session_id=? AND novelty_status IN ('novel', 'duplicate', 'unknown')
                ORDER BY sequence_index DESC, created_at DESC
                LIMIT ?
                """,
                (session_id, OBSERVATION_LOOKBACK_LIMIT),
            )
        ]
    best: dict | None = None
    for row in rows:
        previous_sample = parse_visual_sample(row.get("visual_sample"))
        previous_tokens = []
        try:
            parsed_tokens = json.loads(row.get("text_tokens") or "[]")
            if isinstance(parsed_tokens, list):
                previous_tokens = [normalize_text_token(token) for token in parsed_tokens if normalize_text_token(token)]
        except json.JSONDecodeError:
            previous_tokens = []
        sample_distance = fingerprint_distance(visual_sample, previous_sample)
        token_distance = text_token_distance(text_tokens, previous_tokens)
        hash_distance = visual_hash_distance(visual_hash, row.get("visual_hash") or "")
        exact_hash = bool(visual_hash and visual_hash == row.get("visual_hash"))
        # Keyframe override: when the camera barely moved but the student wrote
        # meaningfully new text, this frame carries new content and must not be
        # collapsed into the previous (visually similar) one.
        new_token_count = len(set(text_tokens) - set(previous_tokens)) if text_tokens else 0
        significant_new_text = (
            token_distance is not None
            and token_distance >= KEYFRAME_TEXT_CHANGE_DISTANCE
            and new_token_count >= KEYFRAME_MIN_NEW_TOKENS
        )
        duplicate = exact_hash
        if not significant_new_text:
            if sample_distance is not None:
                duplicate = duplicate or sample_distance <= VISUAL_DUPLICATE_DISTANCE
                duplicate = duplicate or (
                    sample_distance <= VISUAL_TEXT_DUPLICATE_DISTANCE
                    and (token_distance is None or token_distance <= TEXT_DUPLICATE_DISTANCE)
                )
            elif hash_distance is not None:
                duplicate = duplicate or hash_distance <= 6
        if token_distance is not None and token_distance <= 0.08 and exact_hash:
            duplicate = True
        if not duplicate:
            continue
        score = min(
            sample_distance if sample_distance is not None else 999,
            float(hash_distance) if hash_distance is not None else 999,
            0 if exact_hash else 999,
        )
        candidate = {
            "image_id": row.get("image_id") or "",
            "visual_distance": sample_distance,
            "text_distance": token_distance,
            "score": score,
        }
        if best is None or candidate["score"] < best["score"]:
            best = candidate
    return best


def observation_from_meta(session_id: str, batch_id: str | None, image_id: str, captured_at: str, sequence_index: int, meta: dict) -> dict:
    text_tokens = normalized_text_tokens(meta)
    visual_sample = visual_sample_from_meta(meta)
    visual_hash = visual_hash_from_meta(meta)
    text_hash = meta_text(meta, "text_hash", "textHash") or text_tokens_hash(text_tokens)
    duplicate_of = meta_text(meta, "duplicate_of_image_id", "duplicateOfImageId", "duplicate_of", "duplicateOf")
    client_novelty = meta_text(meta, "novelty_status", "noveltyStatus")
    visual_distance = meta_float(meta, "visual_distance", "visualDistance")
    text_distance = meta_float(meta, "text_distance", "textDistance")
    image_verdict = meta_dict(meta, "image_content_verdict", "imageContentVerdict")
    content_valid = meta_bool(meta, "image_content_valid", "imageContentValid")
    if content_valid is None and image_verdict:
        content_valid = meta_bool(image_verdict, "valid")
    if content_valid is False:
        novelty_status = "invalid"
        duplicate_of = ""
    elif duplicate_of:
        novelty_status = "duplicate"
    else:
        previous = best_previous_observation(session_id, visual_hash, visual_sample, text_tokens)
        if previous:
            duplicate_of = previous.get("image_id") or ""
            visual_distance = previous.get("visual_distance")
            text_distance = previous.get("text_distance")
            novelty_status = "duplicate"
        elif client_novelty in {"novel", "duplicate", "unknown"}:
            novelty_status = client_novelty
        else:
            novelty_status = "novel" if (visual_hash or text_tokens or visual_sample) else "unknown"
    return {
        "id": uuid.uuid4().hex,
        "session_id": session_id,
        "batch_id": batch_id,
        "image_id": image_id,
        "captured_at": captured_at,
        "sequence_index": sequence_index,
        "visual_hash": visual_hash,
        "visual_sample": json_dumps(visual_sample) if visual_sample else "",
        "text_hash": text_hash,
        "text_tokens": json_dumps(text_tokens),
        "signal_summary": meta_text(meta, "signal_summary", "signalSummary"),
        "novelty_status": novelty_status,
        "duplicate_of_image_id": duplicate_of,
        "image_content_valid": content_valid,
        "discard_reason": meta_text(meta, "discard_reason", "discardReason")
        or meta_text(image_verdict, "reason"),
        "discard_detail": meta_text(meta, "discard_detail", "discardDetail")
        or meta_text(image_verdict, "detail"),
        "visual_distance": visual_distance,
        "text_distance": text_distance,
        "created_at": utc_now(),
    }


def insert_observation(observation: dict) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO session_observations(
                id, session_id, batch_id, image_id, captured_at, sequence_index,
                visual_hash, visual_sample, text_hash, text_tokens, signal_summary,
                novelty_status, duplicate_of_image_id, visual_distance, text_distance, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                observation["id"],
                observation["session_id"],
                observation["batch_id"],
                observation["image_id"],
                observation["captured_at"],
                observation["sequence_index"],
                observation["visual_hash"],
                observation["visual_sample"],
                observation["text_hash"],
                observation["text_tokens"],
                observation["signal_summary"],
                observation["novelty_status"],
                observation["duplicate_of_image_id"],
                observation["visual_distance"],
                observation["text_distance"],
                observation["created_at"],
            ),
        )


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "未知"
    seconds = max(0, seconds)
    if seconds < 60:
        return f"{seconds:.0f} 秒"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} 分钟"
    hours = minutes / 60
    return f"{hours:.1f} 小时"


def parse_date_or_datetime(value: str | None) -> datetime | None:
    return parse_datetime(value)


def normalize_mistake_status(value: object, *, default: str = "suspected") -> str:
    status = str(value or default).strip()
    status = MISTAKE_STATUS_ALIASES.get(status, status)
    if status not in MISTAKE_STATUS_VALUES:
        raise HTTPException(422, f"invalid mistake status: {status}")
    return status


def normalize_review_state(value: object, *, default: str = "new") -> str:
    state = str(value or default).strip()
    state = MISTAKE_REVIEW_STATE_ALIASES.get(state, state)
    if state not in MISTAKE_REVIEW_STATE_VALUES:
        raise HTTPException(422, f"invalid review state: {state}")
    return state


def review_due_at_for(status: str, review_state: str, base_time: datetime | None = None) -> str:
    status = MISTAKE_STATUS_ALIASES.get(status, status)
    review_state = MISTAKE_REVIEW_STATE_ALIASES.get(review_state, review_state)
    if status in {"ignored", "mastered"} or review_state in {"ignored", "mastered"}:
        return ""
    base = base_time or datetime.now(timezone.utc)
    key = status if status == "corrected" else review_state
    days = REVIEW_SCHEDULE_DAYS.get(key, 1)
    return (base + timedelta(days=days)).isoformat()


def review_due_sort_value(value: str | None) -> str:
    return value or "9999-12-31T23:59:59+00:00"


def normalize_review_event_result(value: object) -> str:
    result = str(value or "").strip().lower()
    result = REVIEW_EVENT_ALIASES.get(result, result)
    if result not in REVIEW_EVENT_RESULTS:
        raise HTTPException(422, f"invalid review result: {result}")
    return result


def optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        raise HTTPException(422, "invalid integer value")


def optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise HTTPException(422, "invalid numeric value")


_MISTAKE_METHOD_LABELS = {
    "manual": "你手动加入",
    "photo_grading": "拍题批改时自动识别",
    "observation": "智能观察学习时识别",
    "qa": "问答中发现",
}
_MISTAKE_STATUS_LABELS = {
    "candidate": "候选 · 待导入",
    "suspected": "疑似 · 待确认",
    "incomplete": "疑似未完成",
    "confirmed": "已确认",
    "corrected": "已订正",
    "mastered": "已掌握",
    "ignored": "已忽略",
}


def build_mistake_provenance(item: dict) -> dict:
    """给每一道错题生成"它是怎么来的"说明：何时 / 怎么采集 / 为什么判为错 / 判定依据 / 置信度。
    旧数据没有 detection_method，则从 source_summary 推断，保证每道题都说得清。"""
    method = (item.get("detection_method") or "").strip()
    summary = item.get("source_summary") or ""
    if not method:
        if "手动" in summary:
            method = "manual"
        elif "观察" in summary:
            method = "observation"
        elif "问答" in summary:
            method = "qa"
        elif summary:
            method = "photo_grading"
    method_label = _MISTAKE_METHOD_LABELS.get(method, "自动整理")
    status = item.get("status") or "suspected"
    student = (item.get("student_answer") or "").strip()
    expected = (item.get("expected_answer") or "").strip()
    reason = (item.get("error_reason") or "").strip()
    evidence = (item.get("evidence") or "").strip()
    # 判定依据：优先"作答↔参考答案对比"，其次错因/证据。
    if status == "incomplete":
        basis = "这道题留空了、没有作答"
    elif student and expected and student != expected:
        basis = f"你的作答与参考答案不一致（作答：{truncate_text(student, 40)}；参考：{truncate_text(expected, 40)}）"
    elif student:
        basis = f"基于你的作答判断：{truncate_text(student, 60)}"
    elif reason and not is_placeholder_mistake_text(reason):
        basis = truncate_text(reason, 80)
    elif evidence:
        basis = truncate_text(evidence, 80)
    else:
        basis = "来自批改/学习记录的判断"
    why = reason if (reason and not is_placeholder_mistake_text(reason)) else basis
    return {
        "method": method,
        "method_label": method_label,
        "when": item.get("created_at") or item.get("first_seen_at") or "",
        "why": truncate_text(why, 120),
        "basis": basis,
        "confidence_label": _MISTAKE_STATUS_LABELS.get(status, "疑似 · 待确认"),
        "source_summary": truncate_text(summary, 80),
        "review_count": int(item.get("review_count") or 0),
    }


def mistake_row_to_dict(row) -> dict:
    item = dict(row)
    item["knowledge_points"] = json_list(item.get("knowledge_points"))
    item["source_image_ids"] = json_list(item.get("source_image_ids"))
    item["source_image_details"] = json_list_of_dicts(item.get("source_image_details"))
    item["provenance"] = build_mistake_provenance(item)
    return item


def student_presence_status(meta: dict, fallback_text: str = "") -> str:
    if meta_bool(meta, "has_device_interaction_presence", "hasDeviceInteractionPresence", "user_operation_presence", "userOperationPresence") is True:
        return "present"
    device_presence = meta.get("device_interaction_presence") or meta.get("deviceInteractionPresence")
    if isinstance(device_presence, dict) and meta_bool(device_presence, "present", "is_present", "isPresent") is True:
        return "present"

    explicit = meta_text(
        meta,
        "student_presence_status",
        "studentPresenceStatus",
        "student_presence",
        "studentPresence",
        "presence_status",
        "presenceStatus",
    ).strip().lower()
    explicit_present = {
        "present",
        "visible",
        "detected",
        "student_present",
        "person_present",
        "human_present",
        "hand_visible",
        "active",
    }
    explicit_absent = {
        "absent",
        "not_present",
        "not_detected",
        "no_student",
        "no_person",
        "no_human",
        "empty",
        "none",
    }
    if explicit in explicit_present:
        return "present"
    if explicit in explicit_absent:
        return "absent"

    has_presence = meta_bool(meta, "has_student_presence", "hasStudentPresence", "student_present", "studentPresent")
    if has_presence is True:
        return "present"

    counts = [
        meta_int(meta, "hand_count", "handCount", "hands"),
        meta_int(meta, "face_count", "faceCount", "faces"),
        meta_int(meta, "body_count", "bodyCount", "bodies", "person_count", "personCount"),
    ]
    if any((count or 0) > 0 for count in counts):
        return "present"

    text = " ".join(
        part
        for part in (
            meta_text(meta, "presence_summary", "presenceSummary"),
            meta_text(meta, "activity_summary", "activitySummary"),
            meta_text(meta, "action_summary", "actionSummary"),
            meta_text(meta, "signal_summary", "signalSummary"),
            fallback_text,
        )
        if part
    ).lower()
    strong_present_terms = (
        "左手",
        "右手",
        "手指",
        "手掌",
        "握笔",
        "持笔",
        "笔尖",
        "书写",
        "正在写",
        "人体",
        "人脸",
        "身体",
        "头部",
        "hand",
        "face",
        "body",
        "person",
        "human",
        "writing",
    )
    if any(term in text for term in strong_present_terms):
        return "present"
    absent_terms = (
        "未检测到学生",
        "未检测到人",
        "未见学生",
        "未见人体",
        "未见手",
        "无人",
        "没人",
        "学生不在",
        "不在场",
        "no student",
        "no person",
        "not detected",
        "not present",
        "absent",
    )
    if any(term in text for term in absent_terms):
        return "absent"
    return "unknown"


def presence_label(status: str) -> str:
    if status == "present":
        return "有学生/手/笔证据"
    if status == "absent":
        return "未检测到学生/疑似不在场"
    return "在场未识别"


def meta_activity_summary(meta: dict, fallback_text: str = "") -> str:
    device_presence = meta.get("device_interaction_presence") or meta.get("deviceInteractionPresence")
    if isinstance(device_presence, dict) and meta_bool(device_presence, "present", "is_present", "isPresent") is True:
        summary = meta_text(device_presence, "summary", "detail", "reason")
        operation = meta_text(device_presence, "operation", "kind")
        if summary:
            return truncate_text(summary, 120)
        if operation:
            return truncate_text(f"用户操作手机：{operation}，证明用户在场", 120)
        return "用户刚操作过手机，证明用户在场"
    for key in (
        "activity_summary",
        "activitySummary",
        "action_summary",
        "actionSummary",
        "presence_summary",
        "presenceSummary",
        "signal_summary",
        "signalSummary",
    ):
        value = meta.get(key)
        if value not in (None, ""):
            return truncate_text(value, 120)
    return truncate_text(fallback_text, 120) if fallback_text else "无"


def build_time_weight_summary(session: dict, images: list[dict]) -> str:
    ordered = sorted(images, key=image_sort_key)
    if not ordered:
        return "时间权重摘要：无抓拍图片，无法判断学生是否在场或活动持续时长。"

    finish = parse_datetime(session.get("finished_at"))
    totals = {"present": 0.0, "absent": 0.0, "unknown": 0.0}
    intervals: list[dict] = []
    for index, row in enumerate(ordered):
        current = parse_datetime(row.get("captured_at")) or parse_datetime(row.get("created_at"))
        next_time = None
        next_label = "下一张"
        if index + 1 < len(ordered):
            next_row = ordered[index + 1]
            next_time = parse_datetime(next_row.get("captured_at")) or parse_datetime(next_row.get("created_at"))
        elif finish:
            next_time = finish
            next_label = "结束"
        seconds = None
        if current and next_time and next_time >= current:
            seconds = (next_time - current).total_seconds()
        meta = capture_meta_dict(row.get("capture_meta"))
        fallback_signal = row.get("signal_summary") or ""
        status = student_presence_status(meta, fallback_signal)
        is_tail = index + 1 == len(ordered) and bool(finish)
        tail_has_weak_followup = is_tail and status == "present" and seconds is not None and seconds > 15
        if tail_has_weak_followup:
            status = "unknown"
        if seconds is not None:
            totals[status] = totals.get(status, 0.0) + seconds
        hints = []
        if row.get("page_hint"):
            hints.append(f"page={row.get('page_hint')}")
        if row.get("question_hint"):
            hints.append(f"question={row.get('question_hint')}")
        intervals.append(
            {
                "sequence_index": row.get("sequence_index") or 0,
                "seconds": seconds,
                "status": status,
                "next_label": next_label,
                "activity": (
                    meta_activity_summary(meta, fallback_signal) + "；尾段无后续画面，不能确认持续在场"
                    if tail_has_weak_followup
                    else meta_activity_summary(meta, fallback_signal)
                ),
                "hints": "，".join(hints) if hints else "页/题未识别",
                "is_tail": is_tail,
            }
        )

    top_intervals = sorted(
        [item for item in intervals if item["seconds"] is not None],
        key=lambda item: item["seconds"] or 0,
        reverse=True,
    )[:6]
    longest_lines = []
    for item in top_intervals:
        tail_note = "；最后一张到结束，无后续画面证据" if item["is_tail"] else ""
        longest_lines.append(
            (
                f"sequence={item['sequence_index']} 到{item['next_label']}：{format_duration(item['seconds'])}，"
                f"{presence_label(item['status'])}，{item['hints']}，活动线索={item['activity']}{tail_note}"
            )
        )
    if not longest_lines:
        longest_lines.append("无可计算间隔。")
    tail_gap = "未知"
    if intervals and intervals[-1]["is_tail"]:
        tail_gap = format_duration(intervals[-1]["seconds"])
    total_observed = sum(totals.values())
    uncertain_seconds = totals.get("absent", 0.0) + totals.get("unknown", 0.0)
    uncertain_ratio = uncertain_seconds / total_observed if total_observed > 0 else 0
    quality_note = "在场证据充足，可按主要动作段估算。"
    if uncertain_ratio >= 0.5:
        quality_note = "疑似不在场/在场未识别时段占比较高，题目耗时只能低置信度估计。"
    elif uncertain_ratio >= 0.25:
        quality_note = "存在较多在场未识别时段，题目耗时需标注疑似。"
    return (
        "时间权重摘要：报告主次必须按相邻抓拍间隔和画面动作累计，不要把拍到的题目平均分配到总时长；"
        "用户操作手机本身是强在场证据；有学生/手/笔/书写证据的时段优先；视觉未检测到学生只能说明画面未确认，不能单独判定学生离开；无后续画面证据的时段不能自动算作题目耗时。\n"
        f"- 有学生/手/笔证据时长：{format_duration(totals.get('present'))}；"
        f"未检测到学生/疑似不在场时长：{format_duration(totals.get('absent'))}；"
        f"在场未识别时长：{format_duration(totals.get('unknown'))}；"
        f"最后一张到结束无新增画面：{tail_gap}；数据质量提示：{quality_note}\n"
        "- 最长观察间隔（按持续时间降序）：\n"
        + "\n".join(f"  {index + 1}. {line}" for index, line in enumerate(longest_lines))
    )


async def save_upload(
    upload: UploadFile,
    session_id: str,
    kind: str,
    batch_id: str | None = None,
    *,
    page_hint: str = "",
    question_hint: str = "",
    captured_at: str | None = None,
    sequence_index: int = 0,
    capture_meta: dict | None = None,
) -> tuple[str, str, dict]:
    settings = get_settings()
    ext = Path(upload.filename or "capture.jpg").suffix.lower() or ".jpg"
    image_id = uuid.uuid4().hex
    filename = f"{session_id}_{batch_id or 'single'}_{image_id}{ext}"
    target = settings.data_dir / "images" / filename
    captured_at = captured_at or utc_now()
    with target.open("wb") as out:
        shutil.copyfileobj(upload.file, out)
    capture_meta = capture_meta_with_image_verdict(capture_meta or {}, image_content_verdict(capture_meta or {}, filename))
    try:
        create_thumbnail(target, thumbnail_path_for(filename))
    except Exception as exc:
        emit_log(f"生成缩略图失败：{exc}", session_id=session_id, level="warning")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO images(
                id, session_id, batch_id, kind, filename, original_name,
                page_hint, question_hint, captured_at, sequence_index, capture_meta, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                session_id,
                batch_id,
                kind,
                filename,
                upload.filename or "",
                page_hint,
                question_hint,
                captured_at,
                sequence_index,
                meta_string(capture_meta or {}),
                utc_now(),
            ),
        )
    observation = observation_from_meta(session_id, batch_id, image_id, captured_at, sequence_index, capture_meta or {})
    insert_observation(observation)
    observation["capture_meta"] = meta_string(capture_meta or {})
    return image_id, filename, observation


def build_batch_prompt(environment: str, image_rows: list[dict], previous_context: str = "", strategy_context: str = "") -> str:
    capture_lines = "\n".join(
        (
            f"{index + 1}. sequence_index={row['sequence_index']}，captured_at={row['captured_at'] or '未知'}，"
            f"filename={row['filename']}，client_meta={row['capture_meta'] or '{}'}"
        )
        for index, row in enumerate(image_rows)
    )
    return prompts.render_prompt(
        "batch_analysis",
        image_count=len(image_rows),
        environment=(environment or "未提供") + ("\n" + strategy_context if strategy_context else ""),
        capture_lines=capture_lines,
        previous_context=previous_context or "无此前批次记录。",
    )


def duplicate_batch_content(image_rows: list[dict]) -> str:
    lines = [
        (
            f"{index + 1}. sequence_index={row.get('sequence_index')}，captured_at={row.get('captured_at') or '未知'}，"
            f"与已保存关键画面重复，duplicate_of={row.get('duplicate_of_image_id') or '未知'}，"
            f"signal={row.get('signal_summary') or '无'}"
        )
        for index, row in enumerate(image_rows)
    ]
    return (
        "本批次没有新增关键画面，未调用视觉大模型。\n"
        "后端已把这些帧作为重复观察写入时间线，可用于估算停留时长，但不会重复记录题目/知识点。\n"
        + "\n".join(lines)
    )


def skipped_batch_content(image_rows: list[dict]) -> str:
    invalid_rows = [row for row in image_rows if row.get("novelty_status") == "invalid"]
    duplicate_rows = [row for row in image_rows if row.get("novelty_status") == "duplicate"]
    lines: list[str] = []
    for row in invalid_rows:
        lines.append(
            f"- sequence_index={row.get('sequence_index')}，captured_at={row.get('captured_at') or '未知'}，"
            f"无相关学习画面，reason={row.get('discard_reason') or 'invalid_image'}，"
            f"detail={row.get('discard_detail') or '画面无相关'}"
        )
    for row in duplicate_rows:
        lines.append(
            f"- sequence_index={row.get('sequence_index')}，captured_at={row.get('captured_at') or '未知'}，"
            f"重复观察，duplicate_of={row.get('duplicate_of_image_id') or '未知'}，"
            f"signal={row.get('signal_summary') or '无'}"
        )
    if invalid_rows and not duplicate_rows:
        head = "本批次画面无相关学习内容，已自动忽略，未调用视觉大模型。"
    elif invalid_rows:
        head = "本批次没有新增可解析学习画面：无相关画面已忽略，重复画面只写入时间线，未调用视觉大模型。"
    else:
        return duplicate_batch_content(image_rows)
    return head + ("\n" + "\n".join(lines) if lines else "")


def image_row_has_valid_content(row: dict) -> bool:
    if row.get("novelty_status") == "invalid":
        return False
    meta = capture_meta_dict(row.get("capture_meta"))
    verdict = meta_dict(meta, "image_content_verdict", "imageContentVerdict")
    valid = meta_bool(meta, "image_content_valid", "imageContentValid")
    if valid is None and verdict:
        valid = meta_bool(verdict, "valid")
    return valid is not False


def image_sort_key(row: dict) -> tuple[int, datetime, str]:
    parsed = parse_datetime(row.get("captured_at")) or parse_datetime(row.get("created_at")) or datetime.min.replace(tzinfo=timezone.utc)
    return (int(row.get("sequence_index") or 0), parsed, row.get("id") or "")


def report_time_bounds(session: dict, images: list[dict]) -> tuple[datetime | None, datetime | None, float | None]:
    times = [parsed for row in images if (parsed := (parse_datetime(row.get("captured_at")) or parse_datetime(row.get("created_at"))))]
    if not times:
        return None, parse_datetime(session.get("finished_at")), None
    start = min(times)
    fallback_end = max(times)
    finish = parse_datetime(session.get("finished_at"))
    end = finish if finish and finish >= fallback_end else fallback_end
    return start, end, (end - start).total_seconds()


def build_timeline_lines(session: dict, images: list[dict]) -> list[str]:
    ordered = sorted(images, key=image_sort_key)
    finish = parse_datetime(session.get("finished_at"))
    lines: list[str] = []
    for index, row in enumerate(ordered):
        current = parse_datetime(row.get("captured_at")) or parse_datetime(row.get("created_at"))
        next_time = None
        if index + 1 < len(ordered):
            next_row = ordered[index + 1]
            next_time = parse_datetime(next_row.get("captured_at")) or parse_datetime(next_row.get("created_at"))
        elif finish:
            next_time = finish
        seconds_to_next = None
        if current and next_time and next_time >= current:
            seconds_to_next = (next_time - current).total_seconds()
        hints = []
        if row.get("page_hint"):
            hints.append(f"page_hint={row.get('page_hint')}")
        if row.get("question_hint"):
            hints.append(f"question_hint={row.get('question_hint')}")
        lines.append(
            (
                f"{index + 1}. sequence_index={row.get('sequence_index') or 0}，"
                f"captured_at={row.get('captured_at') or row.get('created_at') or '未知'}，"
                f"到下一张/结束间隔={format_duration(seconds_to_next)}，"
                f"batch={row.get('batch_id') or 'single'}，"
                f"{'，'.join(hints) + '，' if hints else ''}"
                f"capture_meta={compact_capture_meta(row.get('capture_meta'))}"
            )
        )
    return lines


def representative_indices(count: int) -> list[int]:
    if count <= 0:
        return []
    edge_count = min(10, count)
    indexes = [*range(edge_count), *range(max(edge_count, count - edge_count), count)]
    if count > edge_count * 2:
        sample_count = min(12, count - edge_count * 2)
        indexes.extend(round(step * (count - 1) / (sample_count + 1)) for step in range(1, sample_count + 1))
    seen = set()
    ordered: list[int] = []
    for index in indexes:
        if 0 <= index < count and index not in seen:
            ordered.append(index)
            seen.add(index)
    return ordered


def render_limited_lines(lines: list[str], max_chars: int, omitted_unit: str) -> tuple[str, bool]:
    full = "\n".join(lines)
    if len(full) <= max_chars:
        return full, False
    selected = sorted(representative_indices(len(lines)))
    while selected:
        rendered = render_selected_lines(lines, selected, omitted_unit)
        if len(rendered) <= max_chars:
            return rendered, True
        middle = len(selected) // 2
        selected.pop(middle)
    return truncate_text(full, max_chars), True


def render_selected_lines(lines: list[str], selected: list[int], omitted_unit: str) -> str:
    parts: list[str] = []
    previous = -1
    for index in selected:
        omitted = index - previous - 1
        if omitted > 0:
            parts.append(f"... 已省略 {omitted} {omitted_unit}，保留代表性开头/中段/结尾 ...")
        parts.append(lines[index])
        previous = index
    trailing = len(lines) - previous - 1
    if trailing > 0:
        parts.append(f"... 已省略 {trailing} {omitted_unit}，保留代表性开头/中段/结尾 ...")
    return "\n".join(parts)


def build_limited_batch_notes(analyses: list[dict], max_chars: int) -> tuple[str, bool]:
    rows = [row for row in analyses if row["scope"] != "final"]
    if not rows:
        return "暂无批次视觉分析内容。", False
    per_note_limit = max(
        FINAL_REPORT_ANALYSIS_MIN_CHARS,
        min(FINAL_REPORT_ANALYSIS_MAX_CHARS, max_chars // max(len(rows), 1) - 80),
    )
    note_truncated = any(len((row["content"] or "无内容").strip()) > per_note_limit for row in rows)
    notes = [
        (
            f"【{index + 1}. {row['scope']} / {row['status']} / batch={row['batch_id'] or '无'} / created_at={row['created_at']}】\n"
            f"{truncate_text(row['content'] or '无内容', per_note_limit)}"
        )
        for index, row in enumerate(rows)
    ]
    rendered, notes_compressed = render_limited_lines(notes, max_chars, "条批次分析")
    return rendered, note_truncated or notes_compressed


def normalize_for_dedupe(text: str) -> str:
    return " ".join(text.split()).lower()


def dedupe_analyses(analyses: list[dict]) -> tuple[list[dict], int]:
    rows = [row for row in analyses if row["scope"] != "final" and row.get("status") == "done" and (row.get("content") or "").strip()]
    seen: set[str] = set()
    unique: list[dict] = []
    duplicate_count = 0
    for row in rows:
        normalized = normalize_for_dedupe(row.get("content") or "")
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        if digest in seen:
            duplicate_count += 1
            continue
        seen.add(digest)
        unique.append(row)
    return unique, duplicate_count


def important_analysis_lines(text: str, max_chars: int) -> str:
    source = (text or "").strip()
    if not source:
        return "无内容"
    keywords = (
        "页",
        "题",
        "题目",
        "题干",
        "答案",
        "手写",
        "算式",
        "草稿",
        "订正",
        "作答",
        "学习科目",
        "学习方式",
        "人体",
        "手",
        "头",
        "身体",
        "笔",
        "书写",
        "停留",
        "耗时",
        "时间",
        "变化",
        "翻页",
        "未识别",
        "疑似",
    )
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    selected: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = normalize_for_dedupe(line)
        if normalized in seen:
            continue
        if any(keyword in line for keyword in keywords):
            selected.append(line)
            seen.add(normalized)
    if not selected:
        selected = lines[:8]
    extracted = "\n".join(selected)
    if len(extracted) < min(max_chars, len(source)) // 3:
        fallback = render_selected_lines(lines, representative_indices(len(lines)), "行分析内容")
        extracted = f"{extracted}\n{fallback}" if extracted else fallback
    return truncate_text(extracted, max_chars)


def final_report_evidence_stats(images: list[dict], analyses: list[dict]) -> dict:
    non_final = [row for row in analyses if row["scope"] != "final"]
    done_non_final = [row for row in non_final if row.get("status") == "done"]
    raw_analysis_chars = sum(len(row.get("content") or "") for row in non_final)
    raw_prompt_chars = sum(len(row.get("prompt") or "") for row in non_final)
    raw_capture_meta_chars = sum(len(row.get("capture_meta") or "") for row in images)
    unique_done, duplicate_count = dedupe_analyses(analyses)
    return {
        "image_count": len(images),
        "analysis_count": len(non_final),
        "done_analysis_count": len(done_non_final),
        "unique_done_analysis_count": len(unique_done),
        "duplicate_done_analysis_count": duplicate_count,
        "raw_analysis_chars": raw_analysis_chars,
        "raw_prompt_chars": raw_prompt_chars,
        "raw_capture_meta_chars": raw_capture_meta_chars,
        "raw_evidence_chars": raw_analysis_chars + raw_capture_meta_chars,
    }


def should_distill_final_evidence(images: list[dict], analyses: list[dict]) -> bool:
    stats = final_report_evidence_stats(images, analyses)
    return (
        stats["raw_evidence_chars"] > FINAL_REPORT_DISTILL_TRIGGER_CHARS
        or stats["done_analysis_count"] > FINAL_REPORT_DISTILL_TRIGGER_ANALYSES
        or stats["image_count"] > 120
    )


def build_evidence_note(row: dict, index: int, per_note_limit: int) -> str:
    content = important_analysis_lines(row.get("content") or "无内容", per_note_limit)
    return (
        f"【{index}. {row['scope']} / {row['status']} / batch={row['batch_id'] or '无'} / created_at={row['created_at']}】\n"
        f"{content}"
    )


def build_previous_batch_context(session_id: str, exclude_batch_id: str | None = None) -> str:
    with connect() as conn:
        params: list = [session_id]
        batch_filter = ""
        if exclude_batch_id:
            batch_filter = "AND COALESCE(batch_id, '') != ?"
            params.append(exclude_batch_id)
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT scope, status, batch_id, content, created_at
                FROM analyses
                WHERE session_id=? AND scope='batch' AND status='done' AND content != ''
                {batch_filter}
                ORDER BY created_at DESC
                LIMIT {BATCH_PREVIOUS_ANALYSIS_LIMIT}
                """,
                params,
            )
        ]
    if not rows:
        return "无此前批次记录。"
    notes = [build_evidence_note(row, index + 1, 900) for index, row in enumerate(reversed(rows))]
    rendered = "\n\n".join(notes)
    return truncate_text(rendered, BATCH_PREVIOUS_CONTEXT_LIMIT)


def clean_learning_line(line: str) -> str:
    cleaned = re.sub(r"^\s*[-*•\d.、\)\]]+\s*", "", line.strip())
    cleaned = cleaned.strip("# ：:;；\t ")
    return truncate_text(cleaned, LEARNING_ITEM_CONTENT_LIMIT)


def content_after_label(line: str) -> str:
    parts = re.split(r"[：:]", line, maxsplit=1)
    if len(parts) == 2 and 2 <= len(parts[1].strip()):
        return parts[1].strip()
    return line


def learning_item_type_for_line(line: str) -> str | None:
    text = line.strip()
    if not text:
        return None
    if any(keyword in text for keyword in ("知识点", "考点", "易错点", "概念", "公式", "方法")):
        return "knowledge"
    if any(keyword in text for keyword in ("板块", "章节", "单元", "页面", "页码", "材料", "练习册", "试卷")):
        return "section"
    if any(keyword in text for keyword in ("题目", "题号", "题干", "问题", "第")) and any(keyword in text for keyword in ("题", "页", "问")):
        return "question"
    if any(keyword in text for keyword in ("答案", "作答", "手写", "算式", "草稿", "订正", "空白", "未作答")):
        return "answer"
    return None


def title_for_learning_item(content: str) -> str:
    one_line = " ".join(content.split())
    one_line = re.sub(r"^[一二三四五六七八九十0-9]+[.、]\s*", "", one_line)
    return truncate_text(one_line, LEARNING_ITEM_TITLE_LIMIT)


def normalize_learning_content(content: str) -> str:
    normalized = normalize_for_dedupe(content)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def extract_learning_items(content: str) -> list[dict]:
    items: list[dict] = []
    seen: set[tuple[str, str]] = set()
    current_type: str | None = None
    context_subject = extract_subject_from_text(content)
    for raw_line in (content or "").splitlines():
        line = clean_learning_line(raw_line)
        if not line:
            continue
        inferred_type = learning_item_type_for_line(line)
        if inferred_type:
            current_type = inferred_type
        item_type = inferred_type or current_type
        if item_type not in {"question", "section", "knowledge", "answer"}:
            continue
        candidate = clean_learning_line(content_after_label(line))
        if not candidate or len(candidate) < 4:
            continue
        if candidate in {"未识别", "未知", "无", "无内容", "无可分析内容"}:
            continue
        if "未识别" in candidate and len(candidate) < 12:
            continue
        normalized = normalize_learning_content(candidate)
        if len(normalized) < 4:
            continue
        key = (item_type, normalized)
        if key in seen:
            continue
        seen.add(key)
        items.append(
            enrich_asset_fields_from_text(
                {
                    "item_type": item_type,
                    "title": title_for_learning_item(candidate),
                    "content": candidate,
                    "content_hash": short_hash(normalized),
                },
                f"{line}\n{content}",
                context_subject,
            )
        )
        if len(items) >= 80:
            break
    return items


def mistake_status_for_text(text: str) -> str | None:
    source = text or ""
    if any(keyword in source for keyword in ("错题", "错误", "错因", "做错", "算错", "答案错误", "不正确")):
        return "suspected"
    if any(keyword in source for keyword in ("空白", "未作答", "不会", "未完成")):
        return "incomplete"
    return None


def mistake_is_empty_sentinel(text: str) -> bool:
    """The "no mistakes" placeholder lines (e.g. 暂无明确错题候选) contain the word
    错题 and would otherwise be parsed into a bogus mistake item."""
    s = text or ""
    return ("暂无" in s or "无明确" in s) and ("错题" in s or "候选" in s)


def report_has_student_answer(content: str) -> bool:
    """True when the report shows at least one real student answer somewhere.
    Used to tell an unstarted/blank new paper (no answers anywhere) apart from a
    worked paper with a few questions left blank: in the former, blank questions are
    NOT mistakes; in the latter, the blanks ARE kept as 未完成 (疑似不会做)."""
    placeholders = {"未识别", "未知", "无", "无内容", "无可分析内容", "空白", "未作答", "暂无"}
    for raw_line in (content or "").splitlines():
        line = clean_learning_line(raw_line)
        if not line or learning_item_type_for_line(line) != "answer":
            continue
        # A wrong answer still means the student wrote something, so it counts as an
        # attempt; only skip lines that report the answer as blank/missing.
        if any(marker in line for marker in ("空白", "未作答", "未完成")):
            continue
        body = clean_learning_line(content_after_label(line))
        if not body or len(body) < 2:
            continue
        if body in placeholders or any(marker in body for marker in ("未识别", "空白", "未作答")):
            continue
        return True
    return False


# 智能观察报告里的描述性词汇——这些行描述"屏幕上在显示什么/停留多久"，不是批改结果，
# 绝不能进错题本（之前错题本里满屏"题目耗时线索/差异题目/无新增纸质题目/画面静止"就是它们漏进来的）。
_MISTAKE_OBSERVATION_NOISE = (
    "画面", "耗时", "线索", "差异题目", "无新增", "界面", "屏幕", "截图", "静止",
    "动作证据", "阅读动作", "工作台", "持续显示", "停留", "翻页", "在场", "未检测到",
    "估计", "本批次", "连续", "秒拍摄", "疑似正在",
)

_MISTAKE_PLACEHOLDER_VALUES = {"", "无", "未识别", "未知", "暂无", "空白", "未作答", "-", "—", "无内容", "无可分析内容"}


def looks_like_observation_noise(text: str) -> bool:
    return any(marker in (text or "") for marker in _MISTAKE_OBSERVATION_NOISE)


def is_placeholder_mistake_text(text: str) -> bool:
    body = (text or "").strip()
    return len(body) < 2 or body in _MISTAKE_PLACEHOLDER_VALUES


def is_blank_mistake_value(text: str) -> bool:
    """按取值（而非长度）判空，用于结构化错题字段：单字符答案如“5”“6”“x”是合法作答，
    不能像 is_placeholder_mistake_text 那样按 len<2 误杀。"""
    body = (text or "").strip()
    return not body or body in _MISTAKE_PLACEHOLDER_VALUES


def extract_mistake_items(content: str, learning_items: list[dict] | None = None) -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()
    knowledge_lines = [item["content"] for item in (learning_items or []) if item.get("item_type") == "knowledge"]
    context_subject = extract_subject_from_text(content)
    # A wholly-blank paper (no student answers anywhere) is an unstarted/new paper,
    # not a set of mistakes. Only treat blank questions as 未完成错题 once at least
    # one question on the material has actually been answered.
    has_student_answer = report_has_student_answer(content)
    recent_question = ""
    recent_answer = ""
    recent_expected = ""
    for raw_line in (content or "").splitlines():
        line = clean_learning_line(raw_line)
        if not line:
            continue
        item_type = learning_item_type_for_line(line)
        body = clean_learning_line(content_after_label(line))
        status = mistake_status_for_text(line)
        if item_type == "question" and body and not status:
            recent_question = body
        if item_type == "answer" and body and not status:
            recent_answer = body
        expected = extract_named_detail(line, ("参考答案", "正确答案", "标准答案", "应为"), ASSET_FIELD_LIMIT)
        if expected:
            recent_expected = expected
        if not status:
            continue
        if mistake_is_empty_sentinel(line):
            continue
        if status == "incomplete" and not has_student_answer:
            continue
        title_source = recent_question or body or line
        # 质量门槛（错题本去噪）：一道真正的错题必须有"具体证据"——学生写了作答、
        # 或明确写了错因、或给了参考答案。只是某行里出现了"错"字（常见于观察报告/泛述）
        # 不足以判为错题。智能观察的描述性行（画面/耗时/界面…）一律排除。
        explicit_reason = extract_named_detail(line, ("错因", "错误原因", "疑似错因"), MISTAKE_REASON_LIMIT)
        has_real_answer = not is_placeholder_mistake_text(recent_answer)
        has_expected = not is_placeholder_mistake_text(recent_expected)
        if looks_like_observation_noise(title_source) or looks_like_observation_noise(line):
            if not (explicit_reason and has_real_answer):
                continue
        if status == "suspected" and not (has_real_answer or explicit_reason or has_expected):
            continue
        digest = short_hash(normalize_learning_content(f"{title_source}\n{line}"))
        if digest in seen:
            continue
        seen.add(digest)
        error_reason = truncate_text(
            explicit_reason or body or line,
            MISTAKE_REASON_LIMIT,
        )
        correction = extract_named_detail(line, ("订正", "改正", "订正痕迹", "修正"), ASSET_SOURCE_SUMMARY_LIMIT)
        next_action = extract_named_detail(line, ("下一步", "建议", "帮助建议", "下一步帮助建议"), ASSET_SOURCE_SUMMARY_LIMIT)
        mistake = enrich_asset_fields_from_text(
            {
                "title": title_for_learning_item(title_source),
                "question_text": recent_question,
                "student_answer": recent_answer if item_type != "question" else "",
                "expected_answer": recent_expected,
                "error_reason": error_reason,
                "knowledge_points": knowledge_lines[:8],
                "status": status,
                "evidence": line,
                "error_type": extract_error_type(line),
                "correction": correction,
                "next_action": next_action,
                "content_hash": digest,
            },
            f"{line}\n{recent_question}\n{recent_answer}\n{content}",
            context_subject,
        )
        items.append(mistake)
        if len(items) >= 40:
            break
    return items


# ── 错题本结构化入库（精准识别） ─────────────────────────────────────────────
# 旧路径用正则反解观察报告散文，导致“差异题目/题目耗时线索/画面静止…”这类小节标题和
# 画面描述被误判成错题。新路径让“看得见图片”的模型在主动拍题时直接吐机读 JSON
# （以“错题JSON：”开头），后端只解析它，并要求“学生确有手写作答 + 可见判错证据”才入库。
_MISTAKE_JSON_MARKER = re.compile(r"错题\s*JSON", re.IGNORECASE)
# 观察异步候选用独立标记，避免和主动拍题的“错题JSON”混淆。
_MISTAKE_CANDIDATE_JSON_MARKER = re.compile(r"错题\s*候选\s*JSON", re.IGNORECASE)

_MISTAKE_JSON_FIELD_ALIASES = {
    "question": ("题目", "题干", "题", "question", "question_text"),
    "student": ("学生作答", "学生答案", "我的作答", "作答", "student_answer"),
    "expected": ("正确答案", "参考答案", "标准答案", "应为", "expected_answer"),
    "reason": ("错因", "错误原因", "疑似错因", "reason", "error_reason"),
    "error_type": ("错误类型", "错因类型", "类型", "error_type"),
    "evidence": ("证据", "判错证据", "evidence"),
    "correction": ("订正", "改正", "修正", "correction"),
    "knowledge": ("知识点", "考点", "knowledge", "knowledge_points"),
    "region": ("区域", "region", "bbox"),
    "verdict": ("判定", "判定结果", "error_verdict", "verdict"),
    "question_index": ("题序", "题号序", "question_index"),
}

# 逐题批改（全题对错权威）用独立标记，和“错题JSON/错题候选JSON”区分开。
_BATCH_GRADING_JSON_MARKER = re.compile(r"批改\s*JSON", re.IGNORECASE)

# 展示用：分析正文末尾会追加机读结构化块（错题候选JSON/错题JSON/批改JSON：[...]）。这些块只供后端解析，
# 不该回放给用户看（历史观察回放会直接显示 analyses[].content）。模板约定结构化块在人读正文之后追加，
# 故从最早出现的标记处截断即可。仅用于 API 返回时清洗，DB 原文保留供最终报告蒸馏。
_STRUCTURED_JSON_DISPLAY_MARKER = re.compile(r"(错题候选\s*JSON|错题\s*JSON|批改\s*JSON)", re.IGNORECASE)


def strip_structured_json_for_display(content: object) -> str:
    text = content if isinstance(content, str) else ("" if content is None else str(content))
    if not text:
        return text
    match = _STRUCTURED_JSON_DISPLAY_MARKER.search(text)
    if not match:
        return text
    return text[: match.start()].rstrip()


def sanitize_analyses_for_display(analyses: list[dict]) -> list[dict]:
    """把分析列表里每条 content 的结构化机读块去掉（只影响返回客户端的副本，不改 DB）。"""
    cleaned: list[dict] = []
    for row in analyses:
        item = dict(row)
        if "content" in item:
            item["content"] = strip_structured_json_for_display(item.get("content"))
        cleaned.append(item)
    return cleaned


def _region_bbox(raw: object) -> dict:
    """把归一化区域框 {x,y,w,h}（取值[0,1]，原点左上、x 右 y 下、相对已校正竖直图）规整成
    干净的 dict；无法识别（缺字段/非数字/非 dict）时返回 {}，调用方据此维持现状。"""
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, dict):
        return {}
    box: dict = {}
    for key in ("x", "y", "w", "h"):
        try:
            box[key] = float(raw.get(key))
        except (TypeError, ValueError):
            return {}
    return box


def _iter_json_arrays(text: str):
    """Yield top-level `[...]` substrings via bracket matching (string-aware), so we
    can recover the JSON array even when the model wraps it in prose or code fences."""
    depth = 0
    start = -1
    in_str = False
    esc = False
    for index, ch in enumerate(text or ""):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            if depth == 0:
                start = index
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start : index + 1]
                start = -1


def has_mistake_json_block(content: str, marker: re.Pattern = _MISTAKE_JSON_MARKER) -> bool:
    return bool(marker.search(content or ""))


def extract_mistake_json(content: str, marker: re.Pattern = _MISTAKE_JSON_MARKER) -> list[dict]:
    text = content or ""
    found = marker.search(text)
    regions = [text[found.end() :]] if found else []
    regions.append(text)
    for region in regions:
        for candidate in _iter_json_arrays(region):
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, list) and any(isinstance(entry, dict) for entry in data):
                return [entry for entry in data if isinstance(entry, dict)]
    return []


def _mistake_json_field(entry: dict, key: str) -> object:
    for alias in _MISTAKE_JSON_FIELD_ALIASES[key]:
        if alias in entry and entry[alias] not in (None, ""):
            return entry[alias]
    return ""


def _mistake_json_knowledge(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [clean_asset_field(item) for item in raw if clean_asset_field(item)]
    value = clean_asset_field(raw)
    return [value] if value else []


def parse_structured_mistakes(
    content: str,
    learning_items: list[dict] | None = None,
    *,
    marker: re.Pattern = _MISTAKE_JSON_MARKER,
    force_status: str | None = None,
) -> list[dict]:
    """解析模型直出的机读错题 JSON。force_status='candidate' 时走观察候选模式：门槛更宽
    （因为后面有学生人工确认），但仍要求有题面+真实作答/订正证据，并剔除观察画面噪声。"""
    entries = extract_mistake_json(content, marker)
    if not entries:
        return []
    context_subject = extract_subject_from_text(content)
    fallback_knowledge = [
        item["content"] for item in (learning_items or []) if item.get("item_type") == "knowledge"
    ]
    items: list[dict] = []
    seen: set[str] = set()
    for entry in entries:
        question = clean_asset_field(_mistake_json_field(entry, "question"))
        student = clean_asset_field(_mistake_json_field(entry, "student"))
        expected = clean_asset_field(_mistake_json_field(entry, "expected"))
        reason = clean_asset_field(_mistake_json_field(entry, "reason"), MISTAKE_REASON_LIMIT)
        error_type = clean_asset_field(_mistake_json_field(entry, "error_type"))
        evidence = clean_asset_field(_mistake_json_field(entry, "evidence"), ASSET_SOURCE_SUMMARY_LIMIT)
        correction = clean_asset_field(_mistake_json_field(entry, "correction"), ASSET_SOURCE_SUMMARY_LIMIT)
        entry_knowledge = _mistake_json_knowledge(_mistake_json_field(entry, "knowledge"))
        region_box = _region_bbox(_mistake_json_field(entry, "region"))
        region_ref = json_dumps(region_box) if region_box else ""
        question_index = clean_asset_field(_mistake_json_field(entry, "question_index"))
        combined = " ".join((error_type, reason, evidence))
        is_incomplete = any(token in combined for token in ("未完成", "空白", "未作答", "不会"))
        has_student = not is_blank_mistake_value(student)
        has_correction_evidence = bool(correction) or any(
            token in evidence for token in ("划掉", "订正", "改", "红叉", "错")
        )
        if force_status == "candidate":
            # 候选模式（较宽松）：有题面，且学生确有作答 / 有订正划掉证据 / 明确留空未完成。
            if not question:
                continue
            if not (has_student or has_correction_evidence or is_incomplete):
                continue
        else:
            # 精准门槛：要么学生确有手写作答（疑似做错），要么明确标注留空未完成且有题面。
            if not has_student and not (is_incomplete and question):
                continue
        # 画面/界面/耗时等观察噪声一律不进错题本/候选，哪怕模型误塞进了 JSON。
        if any(looks_like_observation_noise(value) for value in (question, student, reason, evidence)):
            continue
        title_source = question or student or reason
        if is_blank_mistake_value(title_source):
            continue
        status = force_status or ("suspected" if has_student else "incomplete")
        digest = short_hash(normalize_learning_content(f"{title_source}\n{student}\n{reason}"))
        if digest in seen:
            continue
        seen.add(digest)
        mistake = enrich_asset_fields_from_text(
            {
                "title": title_for_learning_item(title_source),
                "question_text": question,
                "student_answer": student,
                "expected_answer": expected,
                "error_reason": truncate_text(reason or evidence or title_source, MISTAKE_REASON_LIMIT),
                "knowledge_points": (entry_knowledge or fallback_knowledge)[:8],
                # 区域 bbox -> location_ref(JSON 字符串)、题序 -> question_ref；缺失时留空，
                # 交给 enrich_asset_fields_from_text 回退到既有页码/题号启发式。
                "location_ref": region_ref,
                "question_ref": question_index,
                "status": status,
                "evidence": truncate_text(evidence or student or question, ASSET_SOURCE_SUMMARY_LIMIT),
                "error_type": error_type or extract_error_type(combined),
                "correction": correction,
                "next_action": "",
                "content_hash": digest,
            },
            f"{question}\n{student}\n{reason}\n{evidence}\n{content}",
            context_subject,
        )
        items.append(mistake)
        if len(items) >= 40:
            break
    return items


def parse_batch_grading(content: str) -> list[dict]:
    """解析模型输出的“批改JSON：[...]”逐题判定数组（全题对错权威，覆盖检测到的所有题，
    含做对/空白/未识别）。坐标系：归一化[0,1]、原点左上、x 右 y 下、相对已校正竖直图。
    解析失败或没有该标记一律返回 []，绝不影响错题抽取链路。"""
    if not _BATCH_GRADING_JSON_MARKER.search(content or ""):
        return []
    try:
        entries = extract_mistake_json(content, _BATCH_GRADING_JSON_MARKER)
    except Exception:
        return []
    results: list[dict] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        verdict = str(_mistake_json_field(entry, "verdict") or "").strip()
        question = clean_asset_field(_mistake_json_field(entry, "question"))
        student = clean_asset_field(_mistake_json_field(entry, "student"))
        question_index = clean_asset_field(_mistake_json_field(entry, "question_index"))
        if not (question or student or question_index or verdict):
            continue
        results.append(
            {
                "question_index": question_index,
                "region": _region_bbox(_mistake_json_field(entry, "region")),
                "question": question,
                "student_answer": student,
                "verdict": verdict,
                "expected_answer": clean_asset_field(_mistake_json_field(entry, "expected")),
                "correction": clean_asset_field(_mistake_json_field(entry, "correction"), ASSET_SOURCE_SUMMARY_LIMIT),
                "error_reason": clean_asset_field(_mistake_json_field(entry, "reason"), MISTAKE_REASON_LIMIT),
                "knowledge_points": _mistake_json_knowledge(_mistake_json_field(entry, "knowledge")),
            }
        )
        if len(results) >= 60:
            break
    return results


def json_list(raw: str | None) -> list[str]:
    try:
        data = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if str(item)]


def json_list_of_dicts(raw: str | None) -> list[dict]:
    try:
        data = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    result: list[dict] = []
    for item in data:
        if isinstance(item, dict):
            result.append(dict(item))
    return result


def clean_asset_field(value: object, max_chars: int = ASSET_FIELD_LIMIT) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    if text in {"未知", "未识别", "无", "暂无", "不确定"}:
        return ""
    return truncate_text(text, max_chars)


def split_inline_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    parts = [part.strip() for part in re.split(r"[；;，,]\s*", text or "") if part.strip()]
    for part in parts:
        nested = re.match(r"[^=：:]{1,24}[：:]\s*(.+[=：:].*)", part)
        if nested:
            part = nested.group(1).strip()
        match = re.match(r"([^=：:]{1,20})[=：:]\s*(.+)", part)
        if not match:
            continue
        key = match.group(1).strip()
        value = clean_asset_field(match.group(2).strip())
        if value:
            fields[key] = value
    return fields


def extract_subject_from_text(text: str) -> str:
    source = text or ""
    for key, value in split_inline_fields(source).items():
        if key in {"科目", "学科", "学习科目"}:
            return value
    match = re.search(r"(?:科目|学科|学习科目)\s*[=：:]\s*([^，,；;\n]+)", source)
    if match:
        return clean_asset_field(match.group(1))
    subjects = ("数学", "语文", "英语", "物理", "化学", "生物", "科学", "历史", "地理", "政治", "道法", "信息技术")
    for subject in subjects:
        if subject in source:
            return subject
    if re.search(r"\d+\s*[+\-×xX*/÷=]\s*\d+", source) or any(
        keyword in source for keyword in ("加法", "减法", "乘法", "除法", "进位", "退位", "方程", "几何", "计算")
    ):
        return "数学"
    if re.search(r"[A-Za-z]{2,}", source) and any(keyword in source for keyword in ("单词", "句子", "语法", "翻译", "阅读理解")):
        return "英语"
    if any(keyword in source for keyword in ("拼音", "作文", "阅读", "课文", "生字", "修辞", "古诗")):
        return "语文"
    return ""


def extract_page_ref(text: str) -> str:
    source = text or ""
    for key, value in split_inline_fields(source).items():
        if key in {"页码", "页面", "页", "page", "page_hint"}:
            return value
    patterns = [
        r"第\s*([一二三四五六七八九十百千万\d]+)\s*页",
        r"页码\s*[=：:]\s*([^，,；;\n]+)",
        r"页面\s*[=：:]\s*([^，,；;\n]+)",
        r"page[_\s-]*hint\s*[=：:]\s*([^，,；;\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if pattern.startswith("第"):
                value = f"第 {value} 页"
            return clean_asset_field(value)
    return ""


def extract_question_ref(text: str) -> str:
    source = text or ""
    for key, value in split_inline_fields(source).items():
        if key in {"题号", "题", "问题", "question", "question_hint"}:
            return value
    patterns = [
        r"第\s*([一二三四五六七八九十百千万\d]+)\s*(?:题|问|小题)",
        r"题号\s*[=：:]\s*([^，,；;\n]+)",
        r"question[_\s-]*hint\s*[=：:]\s*([^，,；;\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if pattern.startswith("第"):
                value = f"第 {value} 题"
            return clean_asset_field(value)
    return ""


def compose_location_ref(page_ref: str, question_ref: str, fallback: str = "") -> str:
    parts = [part for part in (clean_asset_field(page_ref), clean_asset_field(question_ref)) if part]
    if parts:
        return truncate_text(" ".join(parts), ASSET_FIELD_LIMIT)
    source = fallback or ""
    page = extract_page_ref(source)
    question = extract_question_ref(source)
    parts = [part for part in (page, question) if part]
    if parts:
        return truncate_text(" ".join(parts), ASSET_FIELD_LIMIT)
    return ""


def extract_error_type(text: str) -> str:
    source = text or ""
    for key, value in split_inline_fields(source).items():
        if key in {"错误类型", "错因类型", "类型"}:
            return value
    checks = [
        ("计算错误", ("算错", "计算错误", "计算", "进位", "退位")),
        ("审题错误", ("审题", "看错", "题意", "条件漏看")),
        ("概念不清", ("概念", "公式", "定义", "知识点不熟")),
        ("步骤遗漏", ("步骤", "漏步", "过程", "推导")),
        ("书写/抄写错误", ("抄错", "写错", "书写", "符号")),
        ("未完成", ("空白", "未作答", "未完成", "不会")),
        ("订正痕迹", ("订正", "改正", "划掉", "擦掉")),
    ]
    for label, keywords in checks:
        if any(keyword in source for keyword in keywords):
            return label
    return ""


def extract_named_detail(text: str, labels: tuple[str, ...], max_chars: int = ASSET_SOURCE_SUMMARY_LIMIT) -> str:
    source = text or ""
    for key, value in split_inline_fields(source).items():
        if key in labels:
            return truncate_text(value, max_chars)
    label_pattern = "|".join(re.escape(label) for label in labels)
    match = re.search(rf"(?:{label_pattern})\s*[=：:]\s*([^。\n；;]+)", source)
    if match:
        return truncate_text(match.group(1).strip(), max_chars)
    return ""


def merge_text_value(existing: str, incoming: str, max_chars: int = ASSET_FIELD_LIMIT) -> str:
    old = str(existing or "").strip()
    new = str(incoming or "").strip()
    if not new:
        return truncate_text(old, max_chars)
    if not old:
        return truncate_text(new, max_chars)
    if new in old:
        return truncate_text(old, max_chars)
    if old in new:
        return truncate_text(new, max_chars)
    if len(new) > len(old) and old in {"未知", "未识别"}:
        return truncate_text(new, max_chars)
    return truncate_text(old, max_chars)


def merge_json_lists(existing_raw: str | None, incoming: list[str]) -> list[str]:
    return list(dict.fromkeys([*json_list(existing_raw), *(incoming or [])]))


def merge_source_details(existing_raw: str | None, incoming: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for detail in [*json_list_of_dicts(existing_raw), *(incoming or [])]:
        image_id = str(detail.get("image_id") or detail.get("id") or "")
        key = image_id or json_dumps(detail)
        if not key:
            continue
        previous = merged.get(key, {})
        next_detail = {**previous, **{k: v for k, v in detail.items() if v not in (None, "", [])}}
        merged[key] = next_detail
    return list(merged.values())


def merge_mistake_status(existing: str, incoming: str) -> str:
    priority = {
        "candidate": 0,
        "suspected": 1,
        "incomplete": 2,
        "confirmed": 3,
        "corrected": 4,
        "mastered": 5,
        "ignored": 5,
    }
    old = existing or "suspected"
    new = incoming or old
    return new if priority.get(new, 0) > priority.get(old, 0) else old


def source_image_details_from_rows(images: list[dict]) -> list[dict]:
    details: list[dict] = []
    for row in images:
        filename = row.get("filename") or ""
        image_id = row.get("id") or row.get("image_id") or ""
        details.append(
            {
                "image_id": image_id,
                "filename": filename,
                "thumbnail_url": f"/api/images/{filename}/thumbnail" if filename else "",
                "image_url": f"/images/{filename}" if filename else "",
                "captured_at": row.get("captured_at") or "",
                "sequence_index": int(row.get("sequence_index") or 0),
                "page_hint": row.get("page_hint") or "",
                "question_hint": row.get("question_hint") or "",
            }
        )
    return details


def source_summary_from_images(images: list[dict]) -> str:
    if not images:
        return ""
    first = images[0]
    last = images[-1]
    first_seq = int(first.get("sequence_index") or 0)
    last_seq = int(last.get("sequence_index") or first_seq)
    pages = list(dict.fromkeys(clean_asset_field(row.get("page_hint")) for row in images if clean_asset_field(row.get("page_hint"))))
    questions = list(dict.fromkeys(clean_asset_field(row.get("question_hint")) for row in images if clean_asset_field(row.get("question_hint"))))
    parts = [
        f"{len(images)} 张图",
        f"seq {first_seq}-{last_seq}",
        f"{first.get('captured_at') or '未知'} → {last.get('captured_at') or first.get('captured_at') or '未知'}",
    ]
    if pages:
        parts.append("页码提示=" + "、".join(pages[:4]))
    if questions:
        parts.append("题号提示=" + "、".join(questions[:4]))
    return truncate_text("；".join(parts), ASSET_SOURCE_SUMMARY_LIMIT)


def source_refs_from_images(images: list[dict]) -> dict:
    pages = list(dict.fromkeys(clean_asset_field(row.get("page_hint")) for row in images if clean_asset_field(row.get("page_hint"))))
    questions = list(dict.fromkeys(clean_asset_field(row.get("question_hint")) for row in images if clean_asset_field(row.get("question_hint"))))
    page_ref = "、".join(pages[:4])
    question_ref = "、".join(questions[:4])
    return {
        "page_ref": truncate_text(page_ref, ASSET_FIELD_LIMIT),
        "question_ref": truncate_text(question_ref, ASSET_FIELD_LIMIT),
        "location_ref": compose_location_ref(page_ref, question_ref),
    }


def enrich_asset_fields_from_text(item: dict, text: str, fallback_subject: str = "") -> dict:
    combined = "\n".join(str(part or "") for part in (text, item.get("content"), item.get("title"), item.get("question_text"), item.get("error_reason")))
    subject = item.get("subject") or extract_subject_from_text(combined) or fallback_subject
    page_ref = item.get("page_ref") or extract_page_ref(combined)
    question_ref = item.get("question_ref") or extract_question_ref(combined)
    location_ref = item.get("location_ref") or compose_location_ref(page_ref, question_ref, combined)
    next_item = dict(item)
    next_item["subject"] = clean_asset_field(subject)
    next_item["page_ref"] = clean_asset_field(page_ref)
    next_item["question_ref"] = clean_asset_field(question_ref)
    next_item["location_ref"] = clean_asset_field(location_ref)
    return next_item


def clean_user_text(value: object, max_chars: int = SESSION_GOAL_CHAR_LIMIT) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return truncate_text(text, max_chars)


def clean_string_list(value: object, *, limit: int = 8, max_chars: int = ASSET_FIELD_LIMIT) -> list[str]:
    if isinstance(value, str):
        raw_items = [part.strip() for part in re.split(r"[；;、,\n]", value)]
    elif isinstance(value, list):
        raw_items = [str(part or "").strip() for part in value]
    else:
        raw_items = []
    items: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        clean = clean_user_text(raw, max_chars)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        items.append(clean)
        if len(items) >= limit:
            break
    return items


def infer_need_tags_from_text(text: str) -> list[str]:
    source = (text or "").lower()
    checks = [
        ("mistake_book", ("错题", "错因", "错误", "订正", "不会", "做错", "错在哪里")),
        ("knowledge_points", ("知识点", "考点", "概念", "公式", "方法", "板块")),
        ("step_check", ("步骤", "过程", "算式", "推导", "解法", "检查")),
        ("answer_check", ("答案", "对不对", "批改", "核对", "正确", "错误")),
        ("explain", ("讲解", "解释", "教我", "为什么", "思路")),
        ("report", ("报告", "总结", "复盘", "学习记录", "家长")),
        ("coach_hint", ("先提示", "提示优先", "先给提示", "不要直接给答案", "别直接给答案", "启发", "引导", "先想", "自己试")),
        ("coach_step", ("一步步", "分步", "逐步", "慢慢讲", "带着做", "拆步骤")),
        ("coach_check_only", ("只检查", "只核对", "不代做", "只告诉对错", "批改模式", "检查答案")),
        ("coach_full", ("完整讲解", "完整解析", "讲透", "详细讲解")),
        ("review_plan", ("复习", "错题复习", "巩固", "再练", "回顾", "掌握")),
    ]
    tags: list[str] = []
    for tag, keywords in checks:
        if any(keyword in source for keyword in keywords):
            tags.append(tag)
    return tags


def merge_tags(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for tag in group:
            if tag and tag not in seen:
                seen.add(tag)
                merged.append(tag)
    return merged


def infer_needs_from_analysis(content: str, existing_tags: list[str] | None = None) -> list[str]:
    tags = infer_need_tags_from_text(content)
    text = content or ""
    if any(keyword in text for keyword in ("未识别", "看不清", "模糊", "反光")):
        tags.append("capture_quality")
    if any(keyword in text for keyword in ("草稿", "订正", "改成", "擦除", "划掉")):
        tags.append("revision_trace")
    return merge_tags(existing_tags or [], tags)


def tag_label(tag: str) -> str:
    labels = {
        "mistake_book": "整理错题本",
        "knowledge_points": "提炼知识点",
        "step_check": "检查步骤/过程",
        "answer_check": "核对答案",
        "explain": "生成讲解",
        "report": "生成学习报告",
        "coach_hint": "学习教练：先提示",
        "coach_step": "学习教练：分步讲",
        "coach_check_only": "学习教练：只检查不代做",
        "coach_full": "学习教练：完整讲解",
        "review_plan": "安排复习巩固",
        "capture_quality": "提醒补拍不清晰内容",
        "revision_trace": "关注订正/草稿变化",
    }
    return labels.get(tag, tag)


def session_strategy(session: dict | None) -> dict:
    if not session:
        return {"student_goal": "", "assistant_focus": "", "inferred_needs": [], "report_style": ""}
    return {
        "student_goal": session.get("student_goal") or "",
        "assistant_focus": session.get("assistant_focus") or "",
        "inferred_needs": json_list(session.get("inferred_needs") or "[]"),
        "report_style": session.get("report_style") or "",
    }


COACH_SCENE_CANDIDATES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("answer_check", ("answer_check", "学习场景=检查", "检查答案", "只检查", "批改", "核对"), "学习场景=检查答案：先判断对/错/不清楚，再给一条可执行订正步骤。"),
    ("review", ("review", "学习场景=复习", "复习", "错题复习", "巩固", "再练"), "学习场景=复习错题：优先回顾错因、知识点和相似题迁移，不只重复原答案。"),
    ("homework", ("homework", "学习场景=写作业", "写作业", "学习过程", "连拍", "练习册"), "学习场景=写作业记录：持续观察读题、书写、停顿和订正，少打断，多记录关键变化。"),
    ("concept", ("concept", "学习场景=讲知识点", "讲知识点", "知识点", "概念", "公式"), "学习场景=讲知识点：先用学生画面里的题做例子，再提炼概念和易错点。"),
    ("single_problem", ("single_problem", "学习场景=拍题", "拍题", "拍一道题", "单题", "题目解析"), "学习场景=拍一道题：先识别题目、学生答案和卡点，再回答当前问题。"),
)
COACH_STYLE_CANDIDATES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("check_only", ("check_only", "回答方式=只检查", "只检查", "只核对", "不代做", "只告诉对错", "coach_check_only"), "回答方式=只检查：只判断当前答案/过程是否正确，必要时给一个订正方向，不展开代做。"),
    ("hint_first", ("hint_first", "回答方式=先提示", "先提示", "提示优先", "不要直接给答案", "别直接给答案", "先想", "自己试", "coach_hint"), "回答方式=先提示：默认不直接代做，先给 1-2 个关键提示和学生可以先试的一小步。"),
    ("step_by_step", ("step_by_step", "回答方式=分步讲", "一步步", "分步", "逐步", "带着做", "coach_step"), "回答方式=分步讲：把推理拆成短步骤，每步说明为什么这样做。"),
    ("full_explain", ("full_explain", "回答方式=完整讲", "完整讲解", "完整解析", "讲透", "coach_full"), "回答方式=完整讲解：学生明确需要时给完整解法、结论和一个巩固动作。"),
)


def strategy_priority_sources(strategy: dict) -> list[tuple[str, str]]:
    return [
        ("assistant_focus", str(strategy.get("assistant_focus") or "")),
        ("report_style", str(strategy.get("report_style") or "")),
        ("student_goal", str(strategy.get("student_goal") or "")),
        ("inferred_needs", " ".join(str(tag or "") for tag in strategy.get("inferred_needs") or [])),
    ]


def pick_coach_preference(strategy: dict, candidates: tuple[tuple[str, tuple[str, ...], str], ...]) -> tuple[str, str, str]:
    for source_name, source_text in strategy_priority_sources(strategy):
        source = source_text.lower()
        if not source:
            continue
        for key, tokens, line in candidates:
            if any(token in source for token in tokens):
                return key, line, source_name
    return "", "", ""


def coach_preference_lines(strategy: dict) -> list[str]:
    lines: list[str] = []
    scene_key, scene_line, scene_source = pick_coach_preference(strategy, COACH_SCENE_CANDIDATES)
    style_key, style_line, style_source = pick_coach_preference(strategy, COACH_STYLE_CANDIDATES)
    if scene_line:
        lines.append(scene_line)
    if style_line:
        lines.append(style_line)
    if scene_key or style_key:
        lines.append(
            "偏好优先级=assistant_focus > report_style > student_goal > inferred_needs；"
            f"当前生效场景={scene_key or '未指定'}({scene_source or 'none'})；"
            f"当前生效回答方式={style_key or '未指定'}({style_source or 'none'})。"
        )
    return lines


def build_strategy_context(strategy: dict) -> str:
    lines: list[str] = []
    if strategy.get("student_goal"):
        lines.append(f"学生/用户本回合要求：{strategy['student_goal']}")
    if strategy.get("assistant_focus"):
        lines.append(f"当前智能体关注策略：{strategy['assistant_focus']}")
    needs = strategy.get("inferred_needs") or []
    if needs:
        lines.append("系统已推断的帮助需求：" + "、".join(tag_label(tag) for tag in needs))
    if strategy.get("report_style"):
        lines.append(f"报告风格要求：{strategy['report_style']}")
    coach_lines = coach_preference_lines(strategy)
    if coach_lines:
        lines.append("学习教练偏好：\n" + "\n".join(f"- {line}" for line in coach_lines))
    if not lines:
        lines.append("学生暂未输入明确要求；请根据画面动态判断是否需要错题本、知识点、步骤检查、答案核对或报告总结。")
    lines.append(
        "动态执行规则：本批识别要优先服务上述要求；若画面显示学生在订正、反复停留、答案疑似错误或题干清晰，"
        "请主动补充错题本候选、知识点、错误原因和下一步帮助建议。写错题本候选时尽量使用短字段："
        "科目=...；页码=...；题号=...；学生答案=...；参考答案=...；错因=...；错误类型=...；订正=...；下一步=...。"
    )
    lines.append(
        "学习教练执行规则：优先促进学生自己完成；除非学生明确要求完整答案或场景需要核对结论，"
        "默认先给提示、指出卡点、给一个可立即执行的小任务。检查类请求先说对/错/不清楚，再给下一步订正。"
        "面向家长的总结请用三句话说清：学了什么、卡在哪里、下一次怎么复习。"
    )
    return truncate_text("\n".join(lines), ASSISTANT_FOCUS_CHAR_LIMIT)


def update_session_needs(session_id: str, new_tags: list[str], focus_note: str = "") -> list[str]:
    with connect() as conn:
        row = conn.execute("SELECT inferred_needs, assistant_focus FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            return []
        merged = merge_tags(json_list(row["inferred_needs"]), new_tags)
        current_focus = row["assistant_focus"] or ""
        focus = current_focus
        if focus_note and focus_note not in current_focus:
            focus = truncate_text((current_focus + "\n" + focus_note).strip(), ASSISTANT_FOCUS_CHAR_LIMIT)
        conn.execute(
            "UPDATE sessions SET inferred_needs=?, assistant_focus=?, updated_at=? WHERE id=?",
            (json_dumps(merged), focus, utc_now(), session_id),
        )
    return merged


def record_report_event(session_id: str, event_type: str, title: str, content: str, analysis_id: str | None = None) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO report_events(session_id, analysis_id, event_type, title, content, created_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (session_id, analysis_id, event_type, title, truncate_text(content, REPORT_PROCESS_CHAR_LIMIT), utc_now()),
        )


# 正在生成中的可视化 (account_id, source_type, source_id)。用于对前端轮询去重，
# 避免一次点击在后台堆叠多份「生成可视化讲解」。
_VIZ_INFLIGHT: set[tuple[str, str, str]] = set()
# 落库为 running 的可视化超过该秒数仍未完成则视为「已失效」，允许重新触发（防卡死）。
VIZ_INFLIGHT_STALE_SECONDS = 240


def _iso_within_seconds(value: object, seconds: float) -> bool:
    """判断 ISO 时间字符串是否在最近 `seconds` 秒内（用于判定任务是否仍新鲜在跑）。"""
    if not value:
        return False
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - parsed).total_seconds() <= seconds


def visualization_dir() -> Path:
    path = get_settings().data_dir / "visualizations"
    path.mkdir(parents=True, exist_ok=True)
    return path


def visualization_file_path(filename: str) -> Path:
    if Path(filename).name != filename or not filename.endswith(".html"):
        raise HTTPException(400, "invalid visualization filename")
    base = visualization_dir().resolve()
    path = (base / filename).resolve()
    if not path.is_relative_to(base):
        raise HTTPException(400, "invalid visualization filename")
    if not path.is_file():
        raise HTTPException(404, "visualization not found")
    return path


def teaching_visualization_keywords() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        (
            "solid_geometry",
            (
                "立体几何",
                "空间几何",
                "空间想象",
                "空间思维",
                "三视图",
                "截面",
                "正方体",
                "长方体",
                "棱锥",
                "棱柱",
                "圆柱",
                "圆锥",
                "球",
                "线面角",
                "二面角",
                "异面直线",
                "点到平面",
                "面面垂直",
                "线面垂直",
                "体积",
                "表面积",
                "solid geometry",
                "dihedral",
                "line-plane",
            ),
        ),
        (
            "analytic_geometry",
            (
                "解析几何",
                "圆锥曲线",
                "椭圆",
                "双曲线",
                "抛物线",
                "焦点",
                "准线",
                "弦长",
                "轨迹",
                "斜率",
                "离心率",
                "数量积",
                "定点",
                "定值",
                "切线",
                "analytic geometry",
                "conic",
                "ellipse",
                "hyperbola",
                "parabola",
                "locus",
            ),
        ),
        (
            "geometry_or_spatial",
            (
                "几何",
                "图形",
                "图像",
                "坐标系",
                "向量",
                "平面直角坐标系",
                "函数图像",
                "旋转",
                "平移",
                "相似",
                "全等",
                "角度",
                "面积",
                "可视化",
                "滑块",
                "动态演示",
                "交互",
                "geometry",
                "visualize",
                "interactive",
            ),
        ),
    )


def teaching_visualization_candidate(text: str) -> dict:
    normalized = (text or "").lower()
    hits: list[str] = []
    topic_type = ""
    for candidate_type, keywords in teaching_visualization_keywords():
        matched = [keyword for keyword in keywords if keyword.lower() in normalized]
        if matched:
            hits.extend(matched[:5])
            if not topic_type or candidate_type in {"solid_geometry", "analytic_geometry"}:
                topic_type = candidate_type
    if not hits:
        return {"eligible": False, "topic_type": "", "reason": ""}
    topic_label = {
        "solid_geometry": "立体几何/空间思维",
        "analytic_geometry": "解析几何/圆锥曲线",
        "geometry_or_spatial": "几何/图形可视化",
    }.get(topic_type or "geometry_or_spatial", "几何/图形可视化")
    return {
        "eligible": True,
        "topic_type": topic_type or "geometry_or_spatial",
        "reason": f"{topic_label}：{', '.join(dict.fromkeys(hits[:6]))}",
    }


def strip_code_fence(text: str) -> str:
    stripped = (text or "").strip()
    fence = re.match(r"^```(?:html)?\s*(.*?)\s*```$", stripped, flags=re.IGNORECASE | re.DOTALL)
    return fence.group(1).strip() if fence else stripped


def extract_html_document(text: str) -> str:
    stripped = strip_code_fence(text)
    match = re.search(r"<!doctype html\b.*?</html>", stripped, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(0).strip()
    match = re.search(r"<html\b.*?</html>", stripped, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return "<!doctype html>\n" + match.group(0).strip()
    if "<body" in stripped.lower() or "<main" in stripped.lower():
        return "<!doctype html>\n<html lang=\"zh-CN\">\n<head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>可视化讲解</title></head>\n<body>\n" + stripped + "\n</body>\n</html>"
    raise ValueError("model did not return an HTML document")


def sanitize_teaching_html(html_text: str) -> str:
    document = extract_html_document(html_text)
    document = re.sub(r"<script\b[^>]*\bsrc\s*=\s*(['\"])(?!https://cdn\.jsdelivr\.net/|https://unpkg\.com/).*?</script>", "", document, flags=re.IGNORECASE | re.DOTALL)
    document = re.sub(r"\s+on[a-z]+\s*=\s*(['\"]).*?\1", "", document, flags=re.IGNORECASE | re.DOTALL)
    document = re.sub(r"\s+href\s*=\s*(['\"])\s*javascript:.*?\1", " href=\"#\"", document, flags=re.IGNORECASE | re.DOTALL)
    document = re.sub(r"\s+src\s*=\s*(['\"])\s*javascript:.*?\1", "", document, flags=re.IGNORECASE | re.DOTALL)
    document = re.sub(r"<meta[^>]+http-equiv\s*=\s*(['\"]?)refresh\1[^>]*>", "", document, flags=re.IGNORECASE)
    if "<!doctype html" not in document[:80].lower():
        document = "<!doctype html>\n" + document
    if "<meta charset" not in document.lower():
        document = document.replace("<head>", '<head>\n<meta charset="utf-8">', 1)
    if "viewport" not in document.lower():
        document = document.replace("<head>", '<head>\n<meta name="viewport" content="width=device-width, initial-scale=1">', 1)
    if len(document) > TEACHING_VISUALIZATION_HTML_CHAR_LIMIT:
        raise ValueError("generated HTML is too large")
    return document


def visualization_row_to_dict(row: dict) -> dict:
    item = dict(row)
    filename = item.get("html_filename") or ""
    item["url"] = f"/visualizations/{filename}" if filename else ""
    item["can_open"] = bool(filename and (visualization_dir() / filename).is_file())
    return item


def visualizations_for_sources(source_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    pairs = [(source_type, source_id) for source_type, source_id in source_pairs if source_type and source_id]
    if not pairs:
        return {}
    clauses = " OR ".join("(source_type=? AND source_id=?)" for _ in pairs)
    params: list[object] = []
    for source_type, source_id in pairs:
        params.extend([source_type, source_id])
    with connect() as conn:
        rows = [dict(row) for row in conn.execute(f"SELECT * FROM teaching_visualizations WHERE {clauses}", params)]
    return {(row["source_type"], row["source_id"]): visualization_row_to_dict(row) for row in rows}


def latest_visualization_for_source(source_type: str, source_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM teaching_visualizations
            WHERE source_type=? AND source_id=?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (source_type, source_id),
        ).fetchone()
    return visualization_row_to_dict(dict(row)) if row else None


def attach_visualization_metadata(items: list[dict], source_type: str, *, text_keys: tuple[str, ...]) -> list[dict]:
    existing = visualizations_for_sources([(source_type, str(item.get("id") or "")) for item in items])
    for item in items:
        source_id = str(item.get("id") or "")
        source_text = "\n".join(str(item.get(key) or "") for key in text_keys)
        candidate = teaching_visualization_candidate(source_text)
        item["visualization_candidate"] = bool(candidate["eligible"])
        item["visualization_topic_type"] = candidate["topic_type"]
        item["visualization_reason"] = candidate["reason"]
        item["visualization"] = existing.get((source_type, source_id))
    return items


def visualization_title_from_source(source_type: str, source: dict, topic_type: str) -> str:
    if source_type == "qa_event":
        title = source.get("question") or source.get("answer") or "互动教学可视化"
    elif source_type == "analysis":
        title = source.get("content") or "解析可视化"
    else:
        title = source.get("title") or source.get("text") or "互动教学可视化"
    title = re.sub(r"\s+", " ", str(title or "")).strip()
    title = re.sub(r"[#*_`<>]", "", title)
    if len(title) > 36:
        title = title[:34].rstrip() + "..."
    prefix = {
        "solid_geometry": "立体几何",
        "analytic_geometry": "解析几何",
        "geometry_or_spatial": "可视化讲解",
    }.get(topic_type, "可视化讲解")
    return f"{prefix}：{title}" if title else prefix


def build_teaching_visualization_prompt(
    *,
    source_type: str,
    source: dict,
    source_text: str,
    topic_type: str,
    title: str,
    extra_instruction: str = "",
) -> str:
    edulab_mode = {
        "solid_geometry": (
            "Use the edulab edu-solid-geometry spirit: a self-contained lesson page with formulas/steps on the left "
            "and an interactive Three.js 3D model on the right. Include rotation, zoom, step highlights, camera/reset controls, "
            "and sliders when parameters can vary."
        ),
        "analytic_geometry": (
            "Use the edulab edu-analytic-geometry spirit: a self-contained lesson page with KaTeX formulas, a 2D Canvas board, "
            "sliders for parameters, real-time readouts, trace/point/line overlays, and range or invariant indicators."
        ),
        "geometry_or_spatial": (
            "Use the edulab teaching-page spirit: make the abstract geometry or spatial reasoning visible with Canvas/SVG/Three.js, "
            "formula cards, step controls, and sliders or rotation where useful."
        ),
    }.get(topic_type, "Create an interactive teaching visualization page.")
    return truncate_text(
        f"""
You are generating an interactive HTML teaching artifact for 知进拍学.
The user wants edulab-style output when geometry, spatial reasoning, or visual math appears.
Use the configured OpenAI-compatible model as the generator; return only a complete HTML document.

Source type: {source_type}
Topic type: {topic_type}
Page title: {title}

edulab direction:
{edulab_mode}

Content to visualize:
{truncate_text(source_text, TEACHING_VISUALIZATION_SOURCE_CHAR_LIMIT)}

Extra instruction from user or UI:
{extra_instruction or "无"}

Hard requirements:
- Output exactly one complete self-contained HTML document, starting with <!doctype html>. No Markdown fences, no explanation outside HTML.
- Chinese UI copy by default.
- Make it useful as a teaching page: problem statement, known conditions, formula area, step-by-step explanation, answer/check area, and an interactive visualization.
- Include controls a student expects: step buttons, reset button, at least one slider when a parameter or viewpoint can vary, and 3D drag/rotation for solid geometry.
- For solid geometry or spatial-thinking content, use Three.js from https://cdn.jsdelivr.net when needed. For 2D geometry, Canvas/SVG is fine. For formulas, use KaTeX or MathJax CDN if helpful.
- Do not rely on network except CDN libraries from jsdelivr or unpkg. Do not call APIs. Do not use forms or external links.
- Keep the page robust if the exact problem is partially unclear: state visible assumptions and let the visualization teach the method rather than inventing unsupported facts.
- Keep CSS restrained and app-like: compact panels, clear contrast, no marketing hero, no decorative gradient orbs.
- All JavaScript must be inline and safe to run inside an iframe.
- Do not use inline event-handler attributes such as onclick/oninput/onchange; attach listeners from the inline script instead.
- Avoid auto-playing audio/video, alerts, prompts, or popups.
""".strip(),
        TEACHING_VISUALIZATION_PROMPT_CHAR_LIMIT,
    )


def teaching_visualization_source(source_type: str, source_id: str, *, session_id: str = "", text: str = "", title: str = "") -> tuple[dict, str]:
    if source_type == "qa_event":
        with connect() as conn:
            row = conn.execute("SELECT * FROM qa_events WHERE id=?", (source_id,)).fetchone()
        if not row:
            raise HTTPException(404, "qa event not found")
        source = qa_event_row_to_dict(dict(row))
        if session_id and source.get("session_id") != session_id:
            raise HTTPException(404, "qa event not found in session")
        source_text = "\n".join(
            [
                f"学生问题：{source.get('question') or ''}",
                f"AI 回答：{source.get('answer') or ''}",
                f"触发方式：{source.get('trigger_type') or ''}",
                f"图像上下文：{source.get('image_context_mode') or ''}",
            ]
        )
        return source, source_text
    if source_type == "analysis":
        with connect() as conn:
            row = conn.execute("SELECT * FROM analyses WHERE id=?", (source_id,)).fetchone()
        if not row:
            raise HTTPException(404, "analysis not found")
        source = dict(row)
        if session_id and source.get("session_id") != session_id:
            raise HTTPException(404, "analysis not found in session")
        source_text = "\n".join(
            [
                f"解析范围：{source.get('scope') or ''}",
                f"解析内容：{source.get('content') or ''}",
            ]
        )
        return source, source_text
    if source_type == "custom":
        source = {"id": source_id, "session_id": session_id, "title": title, "text": text}
        return source, text
    raise HTTPException(422, "unsupported visualization source_type")


def ensure_visualization_session(session_id: str, title: str = "") -> str:
    session_id = clean_user_text(session_id, 120) or uuid.uuid4().hex
    now = utc_now()
    with connect() as conn:
        row = conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not row:
            conn.execute(
                """
                INSERT INTO sessions(
                    id, device_id, mode, title, status, created_at, updated_at,
                    student_goal, assistant_focus, inferred_needs, report_style
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, '', '', '[]', '')
                """,
                (
                    session_id,
                    "visualization",
                    "custom_visualization",
                    title or "教学可视化",
                    "active",
                    now,
                    now,
                ),
            )
    return session_id


def write_teaching_visualization_html(visualization_id: str, html_text: str) -> str:
    filename = f"{visualization_id}.html"
    target = visualization_dir() / filename
    temp = target.with_name(f".{filename}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(html_text, encoding="utf-8")
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()
    return filename


async def generate_teaching_visualization(
    *,
    source_type: str,
    source_id: str,
    session_id: str = "",
    source_text: str = "",
    title: str = "",
    force: bool = False,
    extra_instruction: str = "",
) -> dict:
    if source_type not in TEACHING_VISUALIZATION_SOURCE_TYPES:
        raise HTTPException(422, "unsupported visualization source_type")
    existing = latest_visualization_for_source(source_type, source_id)
    if existing and existing.get("status") == "ready" and existing.get("can_open") and not force:
        return existing

    source, resolved_text = teaching_visualization_source(source_type, source_id, session_id=session_id, text=source_text, title=title)
    resolved_session_id = str(source.get("session_id") or session_id or "")
    content = source_text or resolved_text
    candidate = teaching_visualization_candidate(content + "\n" + extra_instruction)
    if not candidate["eligible"] and not force:
        raise HTTPException(422, "source is not a geometry or spatial visualization candidate")
    topic_type = candidate["topic_type"] or "geometry_or_spatial"
    resolved_title = title or visualization_title_from_source(source_type, source, topic_type)
    prompt = build_teaching_visualization_prompt(
        source_type=source_type,
        source=source,
        source_text=content,
        topic_type=topic_type,
        title=resolved_title,
        extra_instruction=extra_instruction,
    )
    visualization_id = existing.get("id") if existing else uuid.uuid4().hex
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO teaching_visualizations(
                id, session_id, source_type, source_id, status, title, topic_type,
                trigger_reason, prompt, html_filename, error, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, 'running', ?, ?, ?, ?, '', '', ?, ?)
            ON CONFLICT(source_type, source_id) DO UPDATE SET
                status='running',
                session_id=excluded.session_id,
                title=excluded.title,
                topic_type=excluded.topic_type,
                trigger_reason=excluded.trigger_reason,
                prompt=excluded.prompt,
                error='',
                updated_at=excluded.updated_at
            """,
            (
                visualization_id,
                resolved_session_id,
                source_type,
                source_id,
                resolved_title,
                topic_type,
                candidate["reason"],
                prompt,
                now,
                now,
            ),
        )
        row = conn.execute("SELECT id FROM teaching_visualizations WHERE source_type=? AND source_id=?", (source_type, source_id)).fetchone()
        if row:
            visualization_id = row["id"]
    try:
        settings = effective_llm_settings_for_session(resolved_session_id or None)
        raw_html = await run_with_llm_gate(
            f"teaching_visualization:{source_type}:{source_id[:8]}",
            resolved_session_id or None,
            lambda: llm.analyze_text(settings, prompt, max_tokens=TEACHING_VISUALIZATION_MAX_TOKENS),
            priority=LLM_PRIORITY_BACKGROUND,
        )
        clean_html = sanitize_teaching_html(raw_html)
        html_filename = write_teaching_visualization_html(visualization_id, clean_html)
        status = "ready"
        error = ""
    except Exception as exc:
        html_filename = ""
        status = "failed"
        error = llm.format_llm_error(exc) if not isinstance(exc, ValueError) else str(exc)
    with connect() as conn:
        conn.execute(
            """
            UPDATE teaching_visualizations
            SET status=?, html_filename=?, error=?, updated_at=?
            WHERE id=?
            """,
            (status, html_filename, truncate_text(error, 1200), utc_now(), visualization_id),
        )
        row = conn.execute("SELECT * FROM teaching_visualizations WHERE id=?", (visualization_id,)).fetchone()
    result = visualization_row_to_dict(dict(row)) if row else {}
    if status == "failed":
        emit_log(
            f"教学可视化生成失败：source={source_type}/{source_id} error={truncate_text(error, 180)}",
            session_id=resolved_session_id or None,
            source="visualization",
            level="error",
        )
    else:
        emit_log(
            f"教学可视化生成完成：source={source_type}/{source_id} url={result.get('url')}",
            session_id=resolved_session_id or None,
            source="visualization",
        )
    return result


def asset_document_payload(asset_kind: str, row: dict) -> dict:
    if asset_kind == "learning":
        lines = [
            f"类型：{row.get('item_type') or 'learning'}",
            f"标题：{row.get('title') or ''}",
            f"内容：{row.get('content') or ''}",
            f"科目：{row.get('subject') or ''}",
            f"位置：{row.get('location_ref') or ''}",
            f"页码：{row.get('page_ref') or ''}",
            f"题号：{row.get('question_ref') or ''}",
            f"来源：{row.get('source_summary') or ''}",
        ]
    else:
        knowledge_points = json_list(row.get("knowledge_points"))
        lines = [
            f"类型：错题本候选",
            f"标题：{row.get('title') or ''}",
            f"状态：{row.get('status') or ''}",
            f"复习状态：{row.get('review_state') or ''}",
            f"下次复习：{row.get('next_review_at') or ''}",
            f"科目：{row.get('subject') or ''}",
            f"位置：{row.get('location_ref') or ''}",
            f"页码：{row.get('page_ref') or ''}",
            f"题号：{row.get('question_ref') or ''}",
            f"题目：{row.get('question_text') or ''}",
            f"学生答案：{row.get('student_answer') or ''}",
            f"参考答案：{row.get('expected_answer') or ''}",
            f"错在哪里：{row.get('error_reason') or row.get('evidence') or ''}",
            f"错误类型：{row.get('error_type') or ''}",
            f"订正：{row.get('correction') or ''}",
            f"下一步：{row.get('next_action') or ''}",
            f"知识点：{'、'.join(knowledge_points)}",
            f"证据：{row.get('evidence') or ''}",
            f"来源：{row.get('source_summary') or ''}",
        ]
    body = truncate_text("\n".join(line for line in lines if not line.endswith("：")), ASSET_DOCUMENT_BODY_LIMIT)
    search_text = normalize_learning_content(body)
    return {
        "subject": row.get("subject") or "",
        "page_ref": row.get("page_ref") or "",
        "question_ref": row.get("question_ref") or "",
        "location_ref": row.get("location_ref") or "",
        "title": row.get("title") or "",
        "body": body,
        "search_text": search_text,
        "source_image_ids": row.get("source_image_ids") or "[]",
        "source_image_details": row.get("source_image_details") or "[]",
        "first_seen_at": row.get("first_seen_at") or "",
        "last_seen_at": row.get("last_seen_at") or "",
    }


def sync_asset_document(conn, asset_kind: str, asset_id: str) -> None:
    if asset_kind == "learning":
        row = conn.execute("SELECT * FROM learning_items WHERE id=?", (asset_id,)).fetchone()
    elif asset_kind == "mistake":
        row = conn.execute("SELECT * FROM mistake_items WHERE id=?", (asset_id,)).fetchone()
    else:
        return
    if not row:
        return
    data = dict(row)
    payload = asset_document_payload(asset_kind, data)
    now = utc_now()
    document_id = f"{asset_kind}:{asset_id}"
    conn.execute(
        """
        INSERT INTO asset_documents(
            id, session_id, asset_kind, asset_id, subject, page_ref, question_ref,
            location_ref, title, body, search_text, source_image_ids, source_image_details,
            first_seen_at, last_seen_at, created_at, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(asset_kind, asset_id) DO UPDATE SET
            subject=excluded.subject,
            page_ref=excluded.page_ref,
            question_ref=excluded.question_ref,
            location_ref=excluded.location_ref,
            title=excluded.title,
            body=excluded.body,
            search_text=excluded.search_text,
            source_image_ids=excluded.source_image_ids,
            source_image_details=excluded.source_image_details,
            first_seen_at=excluded.first_seen_at,
            last_seen_at=excluded.last_seen_at,
            updated_at=excluded.updated_at
        """,
        (
            document_id,
            data["session_id"],
            asset_kind,
            asset_id,
            payload["subject"],
            payload["page_ref"],
            payload["question_ref"],
            payload["location_ref"],
            payload["title"],
            payload["body"],
            payload["search_text"],
            payload["source_image_ids"],
            payload["source_image_details"],
            payload["first_seen_at"],
            payload["last_seen_at"],
            data.get("created_at") or now,
            now,
        ),
    )


def backfill_asset_documents(limit: int = 1200) -> dict:
    with connect() as conn:
        backfill_asset_source_details(conn, "learning_items", "learning")
        backfill_asset_source_details(conn, "mistake_items", "mistake")
        learning_ids = [
            row["id"]
            for row in conn.execute(
                """
                SELECT li.id
                FROM learning_items li
                LEFT JOIN asset_documents ad ON ad.asset_kind='learning' AND ad.asset_id=li.id
                WHERE ad.id IS NULL
                ORDER BY li.created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
        ]
        remaining = max(0, limit - len(learning_ids))
        mistake_ids = [
            row["id"]
            for row in conn.execute(
                """
                SELECT mi.id
                FROM mistake_items mi
                LEFT JOIN asset_documents ad ON ad.asset_kind='mistake' AND ad.asset_id=mi.id
                WHERE ad.id IS NULL
                ORDER BY mi.created_at DESC
                LIMIT ?
                """,
                (remaining,),
            )
        ]
        for asset_id in learning_ids:
            sync_asset_document(conn, "learning", asset_id)
        for asset_id in mistake_ids:
            sync_asset_document(conn, "mistake", asset_id)
    return {"learning": len(learning_ids), "mistake": len(mistake_ids)}


def backfill_asset_source_details(conn, table: str, asset_kind: str, limit: int = 400) -> int:
    rows = [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT id, source_image_ids, source_image_details, source_summary, page_ref, question_ref, location_ref
            FROM {table}
            WHERE source_image_ids != '[]'
              AND (source_image_details = '[]' OR source_image_details = '' OR source_summary = '')
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
    ]
    updated = 0
    for row in rows:
        image_ids = json_list(row.get("source_image_ids"))
        if not image_ids:
            continue
        placeholders = ", ".join("?" for _ in image_ids)
        images = [
            dict(image)
            for image in conn.execute(
                f"""
                SELECT id, filename, page_hint, question_hint, captured_at, sequence_index
                FROM images
                WHERE id IN ({placeholders})
                ORDER BY sequence_index, captured_at, created_at
                """,
                image_ids,
            )
        ]
        if not images:
            continue
        details = source_image_details_from_rows(images)
        summary = row.get("source_summary") or source_summary_from_images(images)
        refs = source_refs_from_images(images)
        page_ref = row.get("page_ref") or refs["page_ref"]
        question_ref = row.get("question_ref") or refs["question_ref"]
        location_ref = row.get("location_ref") or refs["location_ref"]
        conn.execute(
            f"""
            UPDATE {table}
            SET source_image_details=?, source_summary=?, page_ref=?, question_ref=?, location_ref=?, updated_at=?
            WHERE id=?
            """,
            (json_dumps(details), summary, page_ref, question_ref, location_ref, utc_now(), row["id"]),
        )
        sync_asset_document(conn, asset_kind, row["id"])
        updated += 1
    return updated


def upsert_learning_item(conn, item: dict, source: dict) -> str:
    now = utc_now()
    existing = conn.execute(
        """
        SELECT *
        FROM learning_items
        WHERE session_id=? AND item_type=? AND content_hash=?
        """,
        (source["session_id"], item["item_type"], item["content_hash"]),
    ).fetchone()
    source_image_ids = list(dict.fromkeys(source.get("source_image_ids") or []))
    source_image_details = source.get("source_image_details") or []
    source_summary = source.get("source_summary") or ""
    if existing:
        existing_ids = json_list(existing["source_image_ids"])
        merged_ids = list(dict.fromkeys([*existing_ids, *source_image_ids]))
        merged_details = merge_source_details(existing["source_image_details"], source_image_details)
        content = existing["content"]
        title = existing["title"]
        if len(item["content"]) > len(content):
            content = item["content"]
            title = item["title"]
        subject = merge_text_value(existing["subject"], item.get("subject", ""))
        page_ref = merge_text_value(existing["page_ref"], item.get("page_ref", ""))
        question_ref = merge_text_value(existing["question_ref"], item.get("question_ref", ""))
        location_ref = merge_text_value(existing["location_ref"], item.get("location_ref", "") or compose_location_ref(page_ref, question_ref, item.get("content", "")))
        summary = merge_text_value(existing["source_summary"], source_summary, ASSET_SOURCE_SUMMARY_LIMIT)
        conn.execute(
            """
            UPDATE learning_items
            SET batch_id=?, analysis_id=?, title=?, content=?, subject=?, page_ref=?,
                question_ref=?, location_ref=?, source_summary=?, source_image_details=?, last_seen_at=?,
                last_sequence_index=?, source_image_ids=?, evidence_count=evidence_count + 1,
                confidence=?, updated_at=?
            WHERE id=?
            """,
            (
                source.get("batch_id"),
                source.get("analysis_id"),
                title,
                content,
                subject,
                page_ref,
                question_ref,
                location_ref,
                summary,
                json_dumps(merged_details),
                source.get("last_seen_at") or existing["last_seen_at"],
                source.get("last_sequence_index") or existing["last_sequence_index"],
                json_dumps(merged_ids),
                source.get("confidence", "llm"),
                now,
                existing["id"],
            ),
        )
        sync_asset_document(conn, "learning", existing["id"])
        return existing["id"]
    item_id = uuid.uuid4().hex
    conn.execute(
        """
        INSERT INTO learning_items(
            id, session_id, batch_id, analysis_id, item_type, title, content,
            subject, page_ref, question_ref, location_ref, source_summary, source_image_details, content_hash,
            first_seen_at, last_seen_at, first_sequence_index, last_sequence_index,
            source_image_ids, evidence_count, confidence, created_at, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item_id,
            source["session_id"],
            source.get("batch_id"),
            source.get("analysis_id"),
            item["item_type"],
            item["title"],
            item["content"],
            item.get("subject", ""),
            item.get("page_ref", ""),
            item.get("question_ref", ""),
            item.get("location_ref", ""),
            source_summary,
            json_dumps(source_image_details),
            item["content_hash"],
            source.get("first_seen_at") or now,
            source.get("last_seen_at") or source.get("first_seen_at") or now,
            source.get("first_sequence_index") or 0,
            source.get("last_sequence_index") or source.get("first_sequence_index") or 0,
            json_dumps(source_image_ids),
            1,
            source.get("confidence", "llm"),
            now,
            now,
        ),
    )
    sync_asset_document(conn, "learning", item_id)
    return item_id


def upsert_mistake_item(conn, item: dict, source: dict, learning_item_id: str | None = None) -> str:
    now = utc_now()
    existing = conn.execute(
        "SELECT * FROM mistake_items WHERE session_id=? AND content_hash=?",
        (source["session_id"], item["content_hash"]),
    ).fetchone()
    source_image_ids = list(dict.fromkeys(source.get("source_image_ids") or []))
    source_image_details = source.get("source_image_details") or []
    source_summary = source.get("source_summary") or ""
    if existing:
        merged_ids = list(dict.fromkeys([*json_list(existing["source_image_ids"]), *source_image_ids]))
        merged_details = merge_source_details(existing["source_image_details"], source_image_details)
        question_text = merge_text_value(existing["question_text"], item.get("question_text", ""), LEARNING_ITEM_CONTENT_LIMIT)
        student_answer = merge_text_value(existing["student_answer"], item.get("student_answer", ""), LEARNING_ITEM_CONTENT_LIMIT)
        expected_answer = merge_text_value(existing["expected_answer"], item.get("expected_answer", ""), LEARNING_ITEM_CONTENT_LIMIT)
        error_reason = merge_text_value(existing["error_reason"], item.get("error_reason", ""), MISTAKE_REASON_LIMIT)
        evidence = merge_text_value(existing["evidence"], item.get("evidence", ""), MISTAKE_REASON_LIMIT)
        conn.execute(
            """
            UPDATE mistake_items
            SET learning_item_id=COALESCE(?, learning_item_id), batch_id=?, analysis_id=?,
                title=?, question_text=?, student_answer=?, expected_answer=?,
                error_reason=?, knowledge_points=?, subject=?, page_ref=?, question_ref=?,
                location_ref=?, error_type=?, correction=?, next_action=?, source_summary=?,
                source_image_details=?, status=?,
                review_state=CASE
                    WHEN review_state IN ('mastered', 'ignored') THEN review_state
                    WHEN status IN ('ignored', 'mastered') THEN review_state
                    WHEN review_state='' THEN 'queued'
                    ELSE review_state
                END,
                next_review_at=CASE
                    WHEN status IN ('ignored', 'mastered') OR review_state IN ('mastered', 'ignored') THEN next_review_at
                    WHEN next_review_at='' THEN ?
                    ELSE next_review_at
                END,
                evidence=?, source_image_ids=?, last_seen_at=?, updated_at=?
            WHERE id=?
            """,
            (
                learning_item_id,
                source.get("batch_id"),
                source.get("analysis_id"),
                merge_text_value(existing["title"], item.get("title", "")),
                question_text,
                student_answer,
                expected_answer,
                error_reason,
                json_dumps(merge_json_lists(existing["knowledge_points"], item.get("knowledge_points") or [])),
                merge_text_value(existing["subject"], item.get("subject", "")),
                merge_text_value(existing["page_ref"], item.get("page_ref", "")),
                merge_text_value(existing["question_ref"], item.get("question_ref", "")),
                merge_text_value(existing["location_ref"], item.get("location_ref", "")),
                merge_text_value(existing["error_type"], item.get("error_type", "")),
                merge_text_value(existing["correction"], item.get("correction", ""), ASSET_SOURCE_SUMMARY_LIMIT),
                merge_text_value(existing["next_action"], item.get("next_action", ""), ASSET_SOURCE_SUMMARY_LIMIT),
                merge_text_value(existing["source_summary"], source_summary, ASSET_SOURCE_SUMMARY_LIMIT),
                json_dumps(merged_details),
                merge_mistake_status(existing["status"], item.get("status", "suspected")),
                review_due_at_for(merge_mistake_status(existing["status"], item.get("status", "suspected")), existing["review_state"] or "queued"),
                evidence,
                json_dumps(merged_ids),
                source.get("last_seen_at") or existing["last_seen_at"],
                now,
                existing["id"],
            ),
        )
        sync_asset_document(conn, "mistake", existing["id"])
        return existing["id"]
    mistake_id = uuid.uuid4().hex
    insert_status = item.get("status", "suspected")
    # 候选不排进复习队列：review_state=new、不设到期时间；导入(confirmed)后才由 update_mistake_item 排程。
    insert_review_state = "new" if insert_status == "candidate" else "queued"
    insert_next_review_at = "" if insert_status == "candidate" else review_due_at_for(insert_status, "queued")
    conn.execute(
        """
        INSERT INTO mistake_items(
            id, session_id, learning_item_id, batch_id, analysis_id, title,
            question_text, student_answer, expected_answer, error_reason,
            knowledge_points, subject, page_ref, question_ref, location_ref,
            error_type, correction, next_action, source_summary, source_image_details,
            status, review_state, next_review_at, evidence, source_image_ids,
            first_seen_at, last_seen_at, content_hash, detection_method, created_at, updated_at
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            mistake_id,
            source["session_id"],
            learning_item_id,
            source.get("batch_id"),
            source.get("analysis_id"),
            item["title"],
            item.get("question_text", ""),
            item.get("student_answer", ""),
            item.get("expected_answer", ""),
            item.get("error_reason", ""),
            json_dumps(item.get("knowledge_points") or []),
            item.get("subject", ""),
            item.get("page_ref", ""),
            item.get("question_ref", ""),
            item.get("location_ref", ""),
            item.get("error_type", ""),
            item.get("correction", ""),
            item.get("next_action", ""),
            source_summary,
            json_dumps(source_image_details),
            insert_status,
            insert_review_state,
            insert_next_review_at,
            item.get("evidence", ""),
            json_dumps(source_image_ids),
            source.get("first_seen_at") or now,
            source.get("last_seen_at") or source.get("first_seen_at") or now,
            item["content_hash"],
            source.get("detection_method") or "",
            now,
            now,
        ),
    )
    sync_asset_document(conn, "mistake", mistake_id)
    return mistake_id


def build_manual_mistake_item(body: dict) -> dict:
    question = clean_user_text(
        body.get("question_text") or body.get("question") or body.get("user_question"),
        LEARNING_ITEM_CONTENT_LIMIT,
    )
    answer = clean_user_text(body.get("answer") or body.get("assistant_answer"), LEARNING_ITEM_CONTENT_LIMIT)
    student_answer = clean_user_text(body.get("student_answer"), LEARNING_ITEM_CONTENT_LIMIT)
    expected_answer = clean_user_text(body.get("expected_answer"), LEARNING_ITEM_CONTENT_LIMIT)
    error_reason = clean_user_text(
        body.get("error_reason") or body.get("reason") or "用户从本轮 AI 回答手动加入错题本",
        MISTAKE_REASON_LIMIT,
    )
    correction = clean_user_text(body.get("correction") or answer, ASSET_SOURCE_SUMMARY_LIMIT)
    next_action = clean_user_text(
        body.get("next_action") or "复习时先复述错因，再做一道相似题。",
        ASSET_SOURCE_SUMMARY_LIMIT,
    )
    title = clean_user_text(body.get("title") or question or answer or "手动加入错题", LEARNING_ITEM_TITLE_LIMIT)
    if not question and not answer:
        raise HTTPException(422, "question or answer is required")
    knowledge_points = clean_string_list(body.get("knowledge_points"))
    content_seed = "\n".join(
        part
        for part in (
            title,
            question,
            student_answer,
            expected_answer,
            error_reason,
            correction,
        )
        if part
    )
    item = enrich_asset_fields_from_text(
        {
            "title": title_for_learning_item(title),
            "question_text": question,
            "student_answer": student_answer,
            "expected_answer": expected_answer,
            "error_reason": error_reason,
            "knowledge_points": knowledge_points,
            "status": normalize_mistake_status(body.get("status"), default="confirmed"),
            "evidence": clean_user_text(body.get("evidence") or answer or question, MISTAKE_REASON_LIMIT),
            "error_type": clean_user_text(body.get("error_type"), ASSET_FIELD_LIMIT),
            "correction": correction,
            "next_action": next_action,
            "content_hash": short_hash(normalize_learning_content(content_seed)),
        },
        content_seed,
        clean_user_text(body.get("subject"), ASSET_FIELD_LIMIT),
    )
    return item


def create_manual_mistake_item(session_id: str, body: dict) -> dict:
    init_db()
    with connect() as conn:
        session = conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not session:
            raise HTTPException(404, "session not found")
        source = {
            "session_id": session_id,
            "source_image_ids": clean_string_list(body.get("source_image_ids"), limit=20, max_chars=80),
            "source_image_details": [],
            "source_summary": clean_user_text(body.get("source_summary") or "来自问答手动加入错题本", ASSET_SOURCE_SUMMARY_LIMIT),
            "confidence": "manual",
            "detection_method": clean_user_text(body.get("detection_method") or "manual", 40),
            "first_seen_at": utc_now(),
            "last_seen_at": utc_now(),
        }
        mistake_id = upsert_mistake_item(conn, build_manual_mistake_item(body), source)
        row = conn.execute("SELECT * FROM mistake_items WHERE id=?", (mistake_id,)).fetchone()
    if not row:
        raise HTTPException(500, "mistake not stored")
    return mistake_row_to_dict(row)


def source_images_for_analysis(session_id: str, batch_id: str | None, filenames: list[str]) -> list[dict]:
    with connect() as conn:
        if filenames:
            placeholders = ", ".join("?" for _ in filenames)
            rows = [
                dict(row)
                for row in conn.execute(
                    f"""
                    SELECT id, filename, page_hint, question_hint, captured_at, sequence_index
                    FROM images
                    WHERE session_id=? AND filename IN ({placeholders})
                    ORDER BY sequence_index, captured_at, created_at
                    """,
                    [session_id, *filenames],
                )
            ]
        elif batch_id:
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT id, filename, page_hint, question_hint, captured_at, sequence_index
                    FROM images
                    WHERE session_id=? AND batch_id=?
                    ORDER BY sequence_index, captured_at, created_at
                    """,
                    (session_id, batch_id),
                )
            ]
        else:
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT id, filename, page_hint, question_hint, captured_at, sequence_index
                    FROM images
                    WHERE session_id=? AND batch_id IS NULL
                    ORDER BY sequence_index, captured_at, created_at
                    """,
                    (session_id,),
                )
            ]
    return rows


def store_learning_items_from_analysis(
    session_id: str,
    batch_id: str | None,
    analysis_id: str,
    content: str,
    filenames: list[str],
    detection_method: str = "photo_grading",
) -> dict:
    items = extract_learning_items(content)
    # 正式错题本只由主动拍题/批改喂养（精准 JSON）。被动智能观察(observation)不直接写错题本，
    # 而是异步提取“可疑错题候选”(status=candidate)，由学生人工确认后才导入正式错题本，
    # 避免“题目耗时线索/差异题目/画面静止”这类观察噪声灌入。
    if detection_method == "observation":
        mistakes = parse_structured_mistakes(
            content, items, marker=_MISTAKE_CANDIDATE_JSON_MARKER, force_status="candidate"
        )
    else:
        # 优先解析模型直出的机读 JSON（精准）；模型没吐 JSON 时退回旧启发式（已带去噪门槛），
        # 避免主动拍题的错题零召回。
        mistakes = parse_structured_mistakes(content, items)
        if not mistakes and not has_mistake_json_block(content):
            mistakes = extract_mistake_items(content, items)
    if not items and not mistakes:
        return {"learning_item_count": 0, "mistake_item_count": 0}
    images = source_images_for_analysis(session_id, batch_id, filenames)
    source_image_ids = [row["id"] for row in images]
    source_image_details = source_image_details_from_rows(images)
    source_summary = source_summary_from_images(images)
    first = images[0] if images else {}
    last = images[-1] if images else first
    source = {
        "session_id": session_id,
        "batch_id": batch_id,
        "analysis_id": analysis_id,
        "source_image_ids": source_image_ids,
        "source_image_details": source_image_details,
        "source_summary": source_summary,
        "first_seen_at": first.get("captured_at") or utc_now(),
        "last_seen_at": last.get("captured_at") or first.get("captured_at") or utc_now(),
        "first_sequence_index": int(first.get("sequence_index") or 0),
        "last_sequence_index": int(last.get("sequence_index") or first.get("sequence_index") or 0),
        "confidence": "llm",
        "detection_method": detection_method,
    }
    learning_ids: list[str] = []
    with connect() as conn:
        for item in items:
            learning_ids.append(upsert_learning_item(conn, item, source))
        related_learning_id = learning_ids[0] if learning_ids else None
        for mistake in mistakes:
            upsert_mistake_item(conn, mistake, source, related_learning_id)
    return {"learning_item_count": len(items), "mistake_item_count": len(mistakes)}


def build_learning_items_context(session_id: str, max_chars: int = 2200) -> str:
    with connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT item_type, title, first_seen_at, last_seen_at, evidence_count
                FROM learning_items
                WHERE session_id=?
                ORDER BY first_sequence_index, first_seen_at, created_at
                LIMIT 80
                """,
                (session_id,),
            )
        ]
    if not rows:
        return ""
    labels = {"question": "题目", "section": "板块", "knowledge": "知识点", "answer": "作答"}
    lines = [
        (
            f"{index + 1}. {labels.get(row['item_type'], row['item_type'])}：{row['title']} "
            f"（first={row['first_seen_at'] or '未知'}，last={row['last_seen_at'] or '未知'}，证据={row['evidence_count']}）"
        )
        for index, row in enumerate(rows)
    ]
    return truncate_text("\n".join(lines), max_chars)


def learning_items_for_session(session_id: str, limit: int = 120) -> list[dict]:
    with connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, session_id, batch_id, analysis_id, item_type, title, content,
                       subject, page_ref, question_ref, location_ref, source_summary,
                       source_image_details,
                       first_seen_at, last_seen_at, first_sequence_index, last_sequence_index,
                       source_image_ids, evidence_count, confidence, created_at, updated_at
                FROM learning_items
                WHERE session_id=?
                ORDER BY first_sequence_index, first_seen_at, created_at
                LIMIT ?
                """,
                (session_id, limit),
            )
        ]
    for row in rows:
        row["source_image_ids"] = json_list(row.get("source_image_ids"))
        row["source_image_details"] = json_list_of_dicts(row.get("source_image_details"))
    return rows


def mistake_items_for_session(session_id: str, limit: int = 120) -> list[dict]:
    with connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, session_id, learning_item_id, batch_id, analysis_id, title,
                       question_text, student_answer, expected_answer, error_reason,
                       knowledge_points, subject, page_ref, question_ref, location_ref,
                       error_type, correction, next_action, source_summary, source_image_details,
                       status, review_state, next_review_at, last_reviewed_at, review_count,
                       review_note, confirmed_at, ignored_at, corrected_at, mastered_at,
                       evidence, source_image_ids,
                       first_seen_at, last_seen_at, created_at, updated_at
                FROM mistake_items
                WHERE session_id=?
                ORDER BY first_seen_at, created_at
                LIMIT ?
                """,
                (session_id, limit),
            )
        ]
    return [mistake_row_to_dict(row) for row in rows]


def report_events_for_session(session_id: str, limit: int = 120) -> list[dict]:
    with connect() as conn:
        return [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, session_id, analysis_id, event_type, title, content, created_at
                FROM report_events
                WHERE session_id=?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            )
        ]


QA_SECTION_ALIASES = {
    "题号": "题目",
    "题目": "题目",
    "问题": "题目",
    "已知": "关键条件",
    "条件": "关键条件",
    "关键条件": "关键条件",
    "学生答案": "学生答案",
    "作答": "学生答案",
    "检查结果": "检查结果",
    "结果": "检查结果",
    "正确计算": "解题步骤",
    "计算": "解题步骤",
    "解法": "解题步骤",
    "思路": "解题思路",
    "解题思路": "解题思路",
    "步骤": "解题步骤",
    "过程": "解题步骤",
    "错因": "错因提醒",
    "错误原因": "错因提醒",
    "订正": "订正建议",
    "改正": "订正建议",
    "结论": "结论",
    "答案": "结论",
    "知识点": "知识点",
    "知识点候选": "知识点",
    "知识板块": "知识点",
    "考点": "知识点",
    "先想一想": "先想一想",
    "先试一下": "先想一想",
    "提示": "先想一想",
    "下一步小任务": "下一步小任务",
    "小任务": "下一步小任务",
    "追问建议": "追问建议",
    "可以追问": "追问建议",
    "下一步": "追问建议",
}
QA_SECTION_CLASS = {
    "题目": "topic",
    "关键条件": "facts",
    "学生答案": "student",
    "检查结果": "check",
    "解题思路": "idea",
    "解题步骤": "steps",
    "错因提醒": "warning",
    "订正建议": "fix",
    "结论": "result",
    "知识点": "knowledge",
    "先想一想": "idea",
    "下一步小任务": "follow",
    "追问建议": "follow",
}
QA_SECTION_ORDER = [
    "题目",
    "关键条件",
    "先想一想",
    "学生答案",
    "检查结果",
    "解题思路",
    "解题步骤",
    "错因提醒",
    "订正建议",
    "结论",
    "知识点",
    "下一步小任务",
    "追问建议",
]
# 给每个栏目配一个学生看得懂的图标，便于扫读。
QA_SECTION_ICON = {
    "题目": "📖",
    "关键条件": "🔑",
    "先想一想": "💭",
    "学生答案": "✏️",
    "检查结果": "✅",
    "解题思路": "💡",
    "解题步骤": "🪜",
    "错因提醒": "⚠️",
    "订正建议": "🛠️",
    "结论": "🎯",
    "知识点": "📚",
    "下一步小任务": "🚀",
    "追问建议": "💬",
}
QA_SECTION_PATTERN = re.compile(r"^\s*(?:#{1,4}\s*)?(?:[-*]\s*)?([A-Za-z\u4e00-\u9fff]{2,12})[：:]\s*(.*)$")
QA_NUMBERED_PATTERN = re.compile(r"^\s*(?:\d+[\.、)]|[（(]\d+[）)]|[-*])\s*(.+)$")


def normalize_qa_section_title(raw: str) -> str:
    title = re.sub(r"\s+", "", raw or "").strip("#:-： ")
    return QA_SECTION_ALIASES.get(title, title if title in QA_SECTION_CLASS else "")


def linkify_escaped_text(escaped_text: str) -> str:
    pattern = re.compile(r"(https?://[^\s<]+)")
    return pattern.sub(lambda match: f'<a href="{match.group(1)}" target="_blank" rel="noopener">{match.group(1)}</a>', escaped_text)


def inline_qa_markup(text: object) -> str:
    escaped = html.escape(str(text or "").strip())
    escaped = re.sub(r"(\d+(?:\.\d+)?)\s*([+\-×x*/÷=])\s*(\d+(?:\.\d+)?)", r'<span class="qa-math">\1 \2 \3</span>', escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    return linkify_escaped_text(escaped)


def qa_answer_html(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""
    sections: dict[str, list[str]] = {}
    loose: list[str] = []
    current = ""
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        match = QA_SECTION_PATTERN.match(line)
        if match:
            title = normalize_qa_section_title(match.group(1))
            if title:
                current = title
                remainder = match.group(2).strip()
                if remainder:
                    sections.setdefault(current, []).append(remainder)
                continue
        numbered = QA_NUMBERED_PATTERN.match(line)
        if current and numbered:
            sections.setdefault(current, []).append(numbered.group(1).strip())
            continue
        if current:
            sections.setdefault(current, []).append(line)
        else:
            loose.append(line)

    # 没有任何标签时，说明这是一段简短的自然回复（核对/闲聊/概念一句话），
    # 直接当作普通段落渲染，不要硬塞进“解题步骤”卡片，保持简约。
    if loose and not sections:
        plain = "".join(
            f"<p>{inline_qa_markup(line)}</p>"
            for line in loose
            if str(line).strip()
        )
        return truncate_text(f'<div class="qa-rich-answer qa-plain-reply">{plain}</div>', QA_HTML_CHAR_LIMIT)
    if loose:
        sections.setdefault("解题思路", []).extend(loose)

    parts = ['<div class="qa-rich-answer">']
    for title in QA_SECTION_ORDER:
        lines = [line for line in sections.get(title, []) if str(line).strip()]
        if not lines:
            continue
        class_name = QA_SECTION_CLASS.get(title, "plain")
        icon = QA_SECTION_ICON.get(title, "")
        icon_html = f'<span class="qa-ico" aria-hidden="true">{icon}</span>' if icon else ""
        parts.append(f'<section class="qa-card-section qa-{class_name}">')
        parts.append(f"<h4>{icon_html}{html.escape(title)}</h4>")
        if len(lines) == 1:
            parts.append(f"<p>{inline_qa_markup(lines[0])}</p>")
        else:
            parts.append("<ol>")
            for line in lines:
                parts.append(f"<li>{inline_qa_markup(line)}</li>")
            parts.append("</ol>")
        parts.append("</section>")

    extra_titles = [title for title in sections if title not in QA_SECTION_ORDER]
    for title in extra_titles:
        lines = [line for line in sections.get(title, []) if str(line).strip()]
        if not lines:
            continue
        parts.append('<section class="qa-card-section qa-plain">')
        parts.append(f"<h4>{html.escape(title)}</h4>")
        parts.append("<ol>" if len(lines) > 1 else "")
        for line in lines:
            tag = "li" if len(lines) > 1 else "p"
            parts.append(f"<{tag}>{inline_qa_markup(line)}</{tag}>")
        parts.append("</ol>" if len(lines) > 1 else "")
        parts.append("</section>")

    parts.append("</div>")
    return truncate_text("".join(parts), QA_HTML_CHAR_LIMIT)


# 大模型常把所有“题目/关键条件/学生答案/检查结果/...”标签和分题挤在一行返回，
# 前端按行解析时就变成一整段没有换行的文字。这里在已知标签和分题号前补回换行，
# 让网页和 iOS 都能按段落/卡片渲染。已经换行的内容不会被重复拆分。
QA_REFLOW_LABELS = sorted(set(QA_SECTION_ALIASES), key=len, reverse=True)
QA_REFLOW_LABEL_PATTERN = re.compile(
    r"(?<!\n)(?<![一-鿿])[^\S\n]*"
    r"((?:" + "|".join(re.escape(label) for label in QA_REFLOW_LABELS) + r")[：:])"
)
QA_REFLOW_PROBLEM_PATTERN = re.compile(r"(?<!\n)[^\S\n]*(题\s*\d+\s*[：:])")
QA_REFLOW_OPTION_PATTERN = re.compile(
    r"(?<!\n)[^\S\n]*(\d+\s*[\.、)]\s*(?:举一反三|总结知识点|按用户偏好))"
)


def reflow_qa_answer(answer: str) -> str:
    text = (answer or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return text
    text = QA_REFLOW_PROBLEM_PATTERN.sub(r"\n\1", text)
    text = QA_REFLOW_OPTION_PATTERN.sub(r"\n\1", text)
    text = QA_REFLOW_LABEL_PATTERN.sub(r"\n\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def qa_event_row_to_dict(row: dict) -> dict:
    item = dict(row)
    for key in ("focus", "context", "gesture"):
        raw = item.get(key)
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = {"raw": raw} if raw else {}
            item[key] = parsed
    context = item.get("context") if isinstance(item.get("context"), dict) else {}
    item["used_image_context"] = bool(context.get("used_image_context")) if context else bool(item.get("image_filename"))
    item["image_context_mode"] = str(context.get("image_context_mode") or "")
    item["selected_image_id"] = str(context.get("selected_image_id") or item.get("image_id") or "")
    item["selected_image_filename"] = str(context.get("selected_image_filename") or item.get("image_filename") or "")
    item["uploaded_image_id"] = str(context.get("uploaded_image_id") or "")
    item["uploaded_image_filename"] = str(context.get("uploaded_image_filename") or "")
    item["current_image_rejected"] = bool(context.get("current_image_rejected")) if context else False
    item["rejected_image_id"] = str(context.get("rejected_image_id") or "")
    item["rejected_image_filename"] = str(context.get("rejected_image_filename") or "")
    item["answer"] = reflow_qa_answer(item.get("answer") or "")
    item["answer_html"] = qa_answer_html(item.get("answer") or "")
    item["actionable"] = qa_turn_actionable(item.get("question") or "", item.get("answer") or "")
    return item


# 纯闲聊/打招呼/致谢等问题。这类回合不该弹「举一反三/加入错题本/生成可视化」等学习动作。
_QA_CHITCHAT_PHRASES = {
    "hi", "hello", "hey", "yo", "ok", "okay", "thanks", "thx", "bye",
    "你好", "您好", "哈喽", "嗨", "在吗", "在不在", "在么", "你在吗",
    "早", "早安", "早上好", "中午好", "下午好", "晚上好", "晚安",
    "谢谢", "谢谢你", "谢啦", "多谢", "辛苦了", "好的", "好滴", "嗯", "哦", "哦哦",
    "拜拜", "再见", "晚点聊", "哈哈", "哈哈哈", "测试", "test", "你是谁", "你叫什么",
}


def qa_turn_actionable(question: str, answer: str) -> bool:
    """判断这轮问答是否值得展示学习动作按钮（举一反三/错题本/可视化等）。
    目的：用户只发「hi/你好/谢谢」这类闲聊时，不要一本正经地给学习按钮。
    策略保守：默认 True，仅当问题明显是闲聊/过短且无学习信号时才 False，避免误伤真实问题。"""
    q = (question or "").strip()
    if not q:
        # 无问题文本（多为拍题/语音带图）——按真实学习处理。
        return True
    core = re.sub(r"[\s，。！？、；：,.!?;:~～…·\-—()（）\"'“”‘’*@#]+", "", q.lower())
    if not core:
        return True
    if core in _QA_CHITCHAT_PHRASES:
        return False
    # 学习信号：数字、字母变量、常见学科/求解类词。命中则一定算实质问题。
    if any(ch.isdigit() for ch in core):
        return True
    study_markers = (
        "为什么", "怎么", "如何", "求", "解", "算", "证明", "推导", "题", "答案", "公式",
        "讲", "解释", "什么是", "区别", "举例", "例子", "翻译", "单词", "语法", "默写",
        "+", "-", "×", "÷", "=", "x", "y", "∫", "√",
    )
    if any(marker in q.lower() for marker in study_markers):
        return True
    # 既无学习信号、问题又很短（≤6 个有效字符），多半是闲聊。
    if len(core) <= 6:
        return False
    return True


def qa_answer_is_unhelpful_image_failure(answer: str) -> bool:
    text = re.sub(r"\s+", "", answer or "")
    if not text:
        return False
    image_failure_terms = (
        "未识别",
        "没识别",
        "无法识别",
        "没有识别",
        "看不清",
        "没看清",
        "无法看清",
        "无法确认图片",
        "图片不清楚",
        "照片不清楚",
        "拍摄清晰",
        "清晰题目",
        "清晰的题目",
        "有效题目",
        "无有效题目",
        "有效输入",
        "无有效输入",
        "输入具体问题",
        "上传题目",
        "上传图片",
        "上传清晰",
        "等待有效输入",
        "重新拍",
        "重拍",
        "移到镜头",
        "对准",
    )
    return any(term in text for term in image_failure_terms)


def qa_safe_followup_fallback_answer(question: str) -> str:
    return (
        "解题思路：我们接着刚才那题讲。\n"
        "步骤：先把上一轮已经确定的条件放在一起，再解释你追问的那一步为什么成立，最后写成一句结论。\n"
        "结论：你可以继续问具体哪一步。"
    )


def qa_events_for_session(session_id: str, limit: int = 60) -> list[dict]:
    with connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT qa_events.*, images.filename AS image_filename
                FROM qa_events
                LEFT JOIN images ON images.id = qa_events.image_id
                WHERE qa_events.session_id=?
                ORDER BY qa_events.created_at DESC, qa_events.id DESC
                LIMIT ?
                """,
                (session_id, limit),
            )
        ]
    items = [qa_event_row_to_dict(row) for row in rows]
    return attach_visualization_metadata(items, "qa_event", text_keys=("question", "answer"))


def memory_event_row_to_dict(row: dict) -> dict:
    item = dict(row)
    item["payload"] = parse_json_object(item.get("payload"))
    return item


def memory_events(limit: int = 80, account_id: str = "") -> list[dict]:
    account_id = account_id or get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    with connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT *
                FROM memory_events
                WHERE account_id=?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (account_id, limit),
            )
        ]
    return [memory_event_row_to_dict(row) for row in rows]


def important_memory_events(limit: int = 8, account_id: str = "") -> list[dict]:
    account_id = account_id or get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    with connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT *
                FROM memory_events
                WHERE account_id=? AND message_type IN ('formed_memory', 'mistake_memory', 'explicit_memory')
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (account_id, limit),
            )
        ]
    return [memory_event_row_to_dict(row) for row in rows]


def memory_profile(scope: str = "global", account_id: str = "") -> dict:
    account_id = account_id or get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    with connect() as conn:
        row = conn.execute("SELECT * FROM memory_profiles WHERE account_id=? AND scope=?", (account_id, scope)).fetchone()
    if not row:
        return {"account_id": account_id, "scope": scope, "profile": "", "source_count": 0, "latest_event_at": "", "updated_at": ""}
    return dict(row)


def fallback_memory_profile(existing_profile: str, events: list[dict]) -> str:
    recent_lines = []
    for event in events[-12:]:
        text = truncate_text(event.get("text"), 80)
        if text:
            recent_lines.append(f"- {text}")
    parts = []
    if existing_profile:
        parts.append("已有画像：\n" + truncate_text(existing_profile, 1400))
    if recent_lines:
        parts.append("最近输入摘要：\n" + "\n".join(recent_lines))
    parts.append("整理建议：继续根据用户的提问方式、学科卡点、偏好反馈和常见错误更新画像。")
    return truncate_text("\n\n".join(parts), MEMORY_PROFILE_CHAR_LIMIT)


def build_memory_consolidation_prompt(existing_profile: str, events: list[dict]) -> str:
    event_lines = []
    for event in events:
        event_lines.append(
            (
                f"- {event.get('created_at')} "
                f"[{event.get('message_type') or event.get('source') or 'user'}] "
                f"{truncate_text(event.get('text'), 180)}"
            )
        )
    return f"""
你是学习陪伴 App 的记忆整理器。请把最近用户发出的语音转文字和纯文字信息，整理成可持续更新的用户画像。

要求：
- 只记录对后续学习陪伴有帮助的信息：学习目标、常见科目、表达习惯、偏好、易错点、需要避免的回答方式。
- 不要重复逐条抄录原话；要合并归纳。
- 如果证据不足，写“暂未形成稳定判断”。
- 输出中文，结构紧凑，适合下次问答作为上下文候选。

已有画像：
{existing_profile or "暂无"}

最近用户输入：
{chr(10).join(event_lines) if event_lines else "暂无"}
""".strip()


async def run_memory_consolidation(*, task_id: str | None = None, account_id: str = "") -> dict:
    account_id = account_id or get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    events = list(reversed(memory_events(MEMORY_PROFILE_RECENT_EVENT_LIMIT, account_id=account_id)))
    if not events:
        return memory_profile(account_id=account_id)
    existing = memory_profile(account_id=account_id)
    prompt = build_memory_consolidation_prompt(existing.get("profile") or "", events)
    settings = effective_llm_settings(account_id)
    try:
        profile_text = await run_with_llm_gate(
            f"memory_consolidation:{task_id or 'manual'}",
            None,
            lambda: llm.analyze_text(settings, prompt, max_tokens=900),
            priority=LLM_PRIORITY_BACKGROUND,
            account_id=account_id,
        )
        profile_text = truncate_text(profile_text, MEMORY_PROFILE_CHAR_LIMIT)
    except Exception as exc:
        profile_text = fallback_memory_profile(existing.get("profile") or "", events)
        emit_log(f"记忆整理 LLM 失败，使用本地摘要：{truncate_text(str(exc), 180)}", level="warning", source="memory")
    latest_event_at = max((str(event.get("created_at") or "") for event in events), default="")
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO memory_profiles(account_id, scope, profile, source_count, latest_event_at, updated_at)
            VALUES(?, 'global', ?, ?, ?, ?)
            ON CONFLICT(account_id, scope) DO UPDATE SET
                profile=excluded.profile,
                source_count=excluded.source_count,
                latest_event_at=excluded.latest_event_at,
                updated_at=excluded.updated_at
            """,
            (account_id, profile_text, len(events), latest_event_at, now),
        )
    emit_log(f"记忆整理完成：{len(events)} 条输入", level="info", source="memory")
    return memory_profile(account_id=account_id)


def record_memory_event(
    *,
    session_id: str,
    account_id: str = "",
    qa_event_id: str,
    source: str,
    message_type: str,
    text: str,
    payload: dict | None = None,
) -> dict | None:
    clean_text = clean_user_text(text, MEMORY_EVENT_TEXT_LIMIT)
    if len(clean_text) < 2:
        return None
    event_id = uuid.uuid4().hex
    now = utc_now()
    if not account_id:
        with connect() as conn:
            session = conn.execute("SELECT account_id FROM sessions WHERE id=?", (session_id,)).fetchone()
        account_id = session["account_id"] if session and session["account_id"] else (get_settings().default_account_id or DEFAULT_ACCOUNT_ID)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO memory_events(id, account_id, session_id, qa_event_id, source, message_type, text, payload, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                account_id,
                session_id,
                qa_event_id,
                truncate_text(source, 80),
                truncate_text(message_type, 80),
                clean_text,
                json_dumps(payload or {}),
                now,
            ),
        )
        row = conn.execute("SELECT * FROM memory_events WHERE id=?", (event_id,)).fetchone()
    schedule_memory_consolidation_if_due(account_id=account_id)
    return memory_event_row_to_dict(dict(row)) if row else None


def parse_json_object(raw: str | None) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"raw": truncate_text(text, 1200)}
    return data if isinstance(data, dict) else {"value": data}


def require_control_token(request: Request, body: dict | None = None) -> None:
    expected = get_settings().control_token.strip()
    if not expected:
        return
    provided = (request.headers.get("X-PAI-Control-Token") or "").strip()
    if not provided and body:
        provided = str(body.get("control_token") or body.get("controlToken") or "").strip()
    if provided != expected:
        raise HTTPException(401, "control token required")


def online_cutoff_iso() -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=DEVICE_CONTROL_ONLINE_SECONDS)).isoformat()


def compact_device_state(state: dict) -> dict:
    allowed = {
        "app",
        "platform",
        "session_id",
        "session_title",
        "student_goal",
        "mode_title",
        "upload_state",
        "is_bursting",
        "is_observing",
        "is_listening",
        "is_thinking",
        "is_speaking",
        "qa_state",
        "qa_answer",
        "recognized_text",
        "strategy_sync_state",
        "updated_at",
    }
    compact: dict[str, object] = {}
    for key, value in state.items():
        if key not in allowed:
            continue
        if isinstance(value, str):
            compact[key] = truncate_text(value, 800)
        elif isinstance(value, (bool, int, float)) or value is None:
            compact[key] = value
    return compact


def control_command_row_to_dict(row: dict) -> dict:
    item = dict(row)
    item["payload"] = parse_json_object(item.get("payload"))
    return item


def device_state_row_to_dict(row: dict, now: datetime | None = None) -> dict:
    item = dict(row)
    item["state"] = parse_json_object(item.get("state"))
    last_seen = parse_datetime(item.get("last_seen_at"))
    current = now or datetime.now(timezone.utc)
    item["online"] = bool(last_seen and (current - last_seen).total_seconds() <= DEVICE_CONTROL_ONLINE_SECONDS)
    item["stale_seconds"] = round((current - last_seen).total_seconds(), 1) if last_seen else None
    return item


def expire_old_control_commands(conn, now: str) -> None:
    conn.execute(
        """
        UPDATE control_commands
        SET status='expired', acknowledged_at=CASE WHEN acknowledged_at='' THEN ? ELSE acknowledged_at END,
            error=CASE WHEN error='' THEN 'command expired before delivery' ELSE error END
        WHERE status IN ('pending', 'delivered') AND expires_at < ?
        """,
        (now, now),
    )


def latest_device_state(conn, *, account_id: str = "", device_id: str = "", session_id: str = "", online_only: bool = False) -> dict | None:
    filters: list[str] = []
    params: list[object] = []
    if account_id:
        filters.append("account_id=?")
        params.append(account_id)
    if device_id:
        filters.append("device_id=?")
        params.append(device_id)
    if session_id:
        filters.append("session_id=?")
        params.append(session_id)
    if online_only:
        filters.append("last_seen_at >= ?")
        params.append(online_cutoff_iso())
    where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
    row = conn.execute(
        f"""
        SELECT *
        FROM device_states
        {where_sql}
        ORDER BY last_seen_at DESC
        LIMIT 1
        """,
        params,
    ).fetchone()
    return dict(row) if row else None


def select_commands_for_device(conn, *, device_id: str, session_id: str, limit: int, now: str, account_id: str = "") -> list[dict]:
    filters = ["status='pending'", "expires_at >= ?"]
    params: list[object] = [now]
    if account_id:
        filters.append("account_id=?")
        params.append(account_id)
    route_filters = ["device_id=?"]
    route_params: list[object] = [device_id]
    if session_id:
        route_filters.append("(session_id=? AND device_id='')")
        route_params.append(session_id)
    route_filters.append("(device_id='' AND session_id='')")
    where_sql = " AND ".join(filters) + f" AND ({' OR '.join(route_filters)})"
    rows = [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT *
            FROM control_commands
            WHERE {where_sql}
            ORDER BY created_at ASC
            LIMIT ?
            """,
            [*params, *route_params, limit],
        )
    ]
    if rows:
        ids = [row["id"] for row in rows]
        placeholders = ", ".join("?" for _ in ids)
        conn.execute(
            f"""
            UPDATE control_commands
            SET status='delivered', delivered_at=?
            WHERE id IN ({placeholders}) AND status='pending'
            """,
            [now, *ids],
        )
    return rows


def recent_qa_context(session_id: str, limit: int = QA_RECENT_EVENT_LIMIT) -> str:
    events = [
        event
        for event in reversed(qa_events_for_session(session_id, limit + 3))
        if event.get("status") != "running" or event.get("answer")
    ][-limit:]
    if not events:
        return "No prior QA turns in this learning session."
    lines = []
    for index, event in enumerate(events, start=1):
        context = event.get("context") if isinstance(event.get("context"), dict) else {}
        intent = context.get("student_intent") or context.get("studentIntent") or "unknown"
        lines.append(
            (
                f"{index}. trigger={event.get('trigger_type') or 'unknown'} "
                f"intent={intent} "
                f"status={event.get('status') or 'unknown'} "
                f"image_id={event.get('image_id') or ''} "
                f"question={truncate_text(event.get('question'), 260)} "
                f"answer={truncate_text(event.get('answer'), 420)}"
            )
        )
    return "\n".join(lines)


def recent_analysis_context(session_id: str, limit: int = QA_RECENT_ANALYSIS_LIMIT) -> str:
    with connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT scope, status, content, created_at
                FROM analyses
                WHERE session_id=? AND scope != 'final'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, limit),
            )
        ]
    if not rows:
        return "No previous visual analysis yet."
    lines = []
    for index, row in enumerate(rows, start=1):
        lines.append(
            (
                f"{index}. scope={row.get('scope')} status={row.get('status')} "
                f"created_at={row.get('created_at')}\n"
                f"{truncate_text(row.get('content'), 1400)}"
            )
        )
    return "\n\n".join(lines)


def compact_learning_context(session_id: str) -> str:
    learning = learning_items_for_session(session_id, QA_CONTEXT_ITEM_LIMIT)
    mistakes = mistake_items_for_session(session_id, QA_CONTEXT_ITEM_LIMIT)
    formed_memories = important_memory_events(6)
    sections: list[str] = []
    if learning:
        sections.append(
            "Learning items:\n"
            + "\n".join(
                f"- {item.get('item_type')}: {truncate_text(item.get('title') or item.get('content'), 220)}"
                for item in learning[:QA_CONTEXT_ITEM_LIMIT]
            )
        )
    if mistakes:
        sections.append(
            "Mistake items:\n"
            + "\n".join(
                (
                    f"- {truncate_text(item.get('title') or item.get('question_text'), 180)} "
                    f"status={item.get('status')} reason={truncate_text(item.get('error_reason'), 220)}"
                )
                for item in mistakes[:QA_CONTEXT_ITEM_LIMIT]
            )
            )
    if formed_memories:
        sections.append(
            "Important formed memories:\n"
            + "\n".join(
                (
                    f"- [{event.get('message_type') or 'formed_memory'}] "
                    f"{truncate_text(event.get('text'), 260)}"
                )
                for event in formed_memories
            )
        )
    return "\n\n".join(sections) if sections else "No structured learning or mistake items yet."


def latest_session_image_for_qa(session_id: str, *, exclude_image_id: str | None = None) -> dict | None:
    filters = ["session_id=?", "kind IN ('burst', 'single', 'qa')"]
    params: list[object] = [session_id]
    if exclude_image_id:
        filters.append("id<>?")
        params.append(exclude_image_id)
    where_sql = " AND ".join(filters)
    with connect() as conn:
        cursor = conn.execute(
            f"""
            SELECT *
            FROM images
            WHERE {where_sql}
            ORDER BY sequence_index DESC, captured_at DESC, created_at DESC
            LIMIT 30
            """,
            params,
        )
        rows = [dict(row) for row in cursor]
    for row in rows:
        if row.get("kind") == "qa" and qa_context_rejected_from_meta(capture_meta_dict(row.get("capture_meta"))):
            continue
        return row
    return None


def build_qa_prompt(
    session: dict,
    *,
    question: str,
    trigger_type: str,
    focus: dict,
    context: dict,
    gesture: dict,
    image_row: dict | None,
    image_context_mode: str = "text_only",
) -> str:
    strategy = build_strategy_context(session_strategy(session))
    dynamic_strategy = dynamic_strategy_context(
        session,
        context,
        question=question,
        trigger_type=trigger_type,
        image_row=image_row,
        image_context_mode=image_context_mode,
    )
    image_note = "No current frame was uploaded for this turn."
    if image_row:
        if image_context_mode == "current_frame":
            label = "Current frame"
        elif image_context_mode in {"fallback_after_rejected_current_frame", "ignored_current_frame_fallback"}:
            label = "Fallback recent session frame; the newly captured QA frame was rejected as not relevant/clear enough"
        else:
            label = "Recent session frame"
        image_note = (
            f"{label}: image_id={image_row.get('id')} filename={image_row.get('filename')} "
            f"context_mode={image_context_mode} "
            f"captured_at={image_row.get('captured_at')} sequence_index={image_row.get('sequence_index')} "
            f"meta={compact_capture_meta(image_row.get('capture_meta'), 600)}"
        )
    prompt = f"""
You are the real-time learning assistant in an iOS study camera app.
Answer in Chinese unless the student clearly asks for another language.
Use the current camera frame when provided, the session context, and the student's question.
Be concise, but include enough reasoning steps for a student to continue solving.
If the student asks to check mistakes, compare the visible work, point out likely wrong parts, and give the next correction step.
If the referenced problem is ambiguous, say what you can infer and ask for a short clarification.
Write a concise, targeted answer. Do NOT fill a fixed template — choose the layout that actually fits THIS question, and keep it short.

Layout rules (important):
- Use short labeled lines only when they help. Put each label on its own line with a real newline; never put two labels on one line; start each 题号 (题1、题2) on its own line.
- Include a label ONLY if it carries real, specific content for this question. Omit any label that would be empty, generic, or just repeats the question. A tight answer with 1-3 labels beats one that fills every label.
- Match the layout to the question type, for example:
  · 只是核对对错: 主要给 检查结果，必要时再加 错因/订正；通常不需要 题目/关键条件/解题步骤。
  · 让你讲这道题: 题目（一句）+ 解题思路或步骤 + 结论；只有条件多或存在干扰条件时才用 关键条件。hint_first 偏好下改用 先想一想 代替直接给步骤。
  · 概念/知识点提问: 用 知识点 把概念讲清楚 + 一个小例子，不要硬套 题目/学生答案。
  · 简单追问或闲聊: 直接用一两句自然中文回答，可以完全不用任何标签。
- If you do not use labels, just answer in one or two short plain sentences.

Available labels (pick only the subset you need; keep this relative order when several appear):
题目、关键条件、先想一想、学生答案、检查结果、解题思路、步骤、错因、订正、结论、知识点、下一步小任务、追问建议

Plain-text math formatting rules:
- Write student answers, formulas, and units as ordinary readable text, not Markdown or LaTeX.
- Never wrap math in dollar signs. Do not output $...$, $$...$$, \\(...\\), or \\[...\\].
- Do not use LaTeX commands such as \\times, \\div, \\frac, \\sqrt, or cm^2 in visible answers.
- Prefer symbols and Chinese units directly, for example: 学生答案：10×6÷2=30平方厘米.
- If copying the student's handwritten work, preserve the math meaning but normalize display characters, for example ×, ÷, =, 平方厘米.

Session:
- id={session.get('id')}
- title={session.get('title')}
- status={session.get('status')}
- student_goal={session.get('student_goal') or ''}

Strategy:
{strategy}

Dynamic strategy:
{dynamic_strategy}

Current trigger:
- trigger_type={trigger_type}
- focus={json_dumps(focus)}
- gesture={json_dumps(gesture)}
- client_context={json_dumps(context)}
- inferred_student_intent={context.get('student_intent') or 'unknown'}
{image_note}

Recent QA turns:
{recent_qa_context(session.get('id') or '')}

Recent visual analyses:
{recent_analysis_context(session.get('id') or '')}

Structured learning context:
{compact_learning_context(session.get('id') or '')}

Structured context assets from client:
{json_dumps((context or {}).get('structured_context_assets') or [])}

Semantically related items from this student's own knowledge base (past mistakes/knowledge points retrieved by meaning, ranked by relevance to the current question; use ONLY when genuinely relevant, e.g. for review, similar-problem warnings, or concept reinforcement — never as factual proof):
{json_dumps((context or {}).get('semantic_knowledge') or [])}

Durable memories about THIS student, semantically retrieved for the current question (preferences, recurring mistakes/habits, goals; use for personalization and tone, NEVER as proof of the current answer; ignore any that aren't genuinely relevant):
{json_dumps([{'text': m.get('text'), 'kind': m.get('kind'), 'score': m.get('score')} for m in ((context or {}).get('agent_memories') or [])])}

Context asset use policy:
{json_dumps((context or {}).get('context_use_policy') or {})}

Student question:
{question}

Answer quality rules:
- Treat the current frame as the primary source when context_mode=current_frame.
- Before using old assets, classify this turn as one of: answer_check, new_problem_explain, mistake_review, knowledge_summary, transfer_practice, or followup. Use that intent to choose assets.
- Use context assets by priority: current/anchor frame and current question first; active review mistake next; related mistakes and knowledge points next; formed memories/user profile last for personalization.
- Mistake assets are for review, error diagnosis, and warning about similar traps. Do not force an unrelated mistake into the answer.
- Knowledge assets are for explaining concepts, summarizing key points, and generating similar practice. Do not let them override visible work.
- Memory assets are for preferences, recurring patterns, and recent goals. Never use memory as proof of the current correct answer.
- If you rely on an asset, mention it naturally in Chinese in one short phrase, for example: “结合你之前常错的单位换算...”.
- When Strategy includes 当前生效回答方式 or 当前生效场景, treat that as the only active coach preference even if inferred_needs lists other possible tags.
- Follow the learning coach preference in Strategy and client_context. If the preference is hint_first, lead with 先想一想 and avoid giving the final answer until the student asks or checking requires it.
- If the preference is check_only, do not solve the whole problem. Say whether the visible answer/process is correct, wrong, or unclear, then give one correction direction.
- If the preference is step_by_step or full_explain, keep steps compact; add one actionable 下一步小任务 only when it genuinely helps.
- Add 先想一想 or 下一步小任务 only when it gives a real next action worth doing; for a simple check, a concept reply, or chit-chat, leave it out.
- Do NOT force a 下一步 / 追问建议 block on every answer. Include 追问建议 (one short line) only when an obvious useful follow-up exists; otherwise end naturally without it.
- If inferred_student_intent is correction_check, answer as a same-dialogue correction review: compare the student's latest visible work with the prior QA context, decide whether the revision is now correct, and give one precise next correction if needed.
- If inferred_student_intent is answer_check or visual_check, inspect the provided/current frame first and report what is correct, wrong, or unclear. Keep it connected to the referenced prior problem when the wording says "again", "this", "here", "changed", "改完", "再看看", or similar.
- If context says current_image_rejected=true or context_mode is fallback/text-only, silently ignore the new capture. Do not say the image was not recognized; answer the student's spoken follow-up from recent QA turns, prior valid image context, and structured learning context.
- For math, independently recompute the visible expression before speaking the final answer. Check the final numeric result at least two ways when possible.
- When using decomposition or carrying, do not add the original whole number again after adding its decomposed parts.
- Only ask the student to move or retake the material when there is no prior context to answer from. During follow-up, prefer continuing the explanation from prior context.
- If you mention a student's answer is wrong, state the correct answer and one short correction step.
- 选中题聚焦规则（仅当 focus 含 region/crop/question_text/regions 时生效，老客户端不带这些字段时忽略本段）：
  · 若 focus.crop 为真：当前帧已被裁剪为单独一道题，把整帧当作唯一目标题作答，不要提“画面里还有别的题”或让学生圈选。
  · 若 focus.region 存在且 focus.crop 不为真：画面里可能有多道题，只回答 region 归一化框（x,y,w,h，取值[0,1]，原点左上、x 右 y 下、相对已梯形校正后的竖直上传图）框住的那一道，其它题忽略、不要顺带讲解。
  · 若 focus.question_text 存在：以它作为选中题的题干锚点；当它与画面内容冲突时以画面为准，并简短指出冲突。
  · focus.regions 只用于帮助你理解整页布局，不要据此逐题作答。
- Keep each label compact. Prefer one idea per line. Avoid long Markdown paragraphs and avoid Markdown tables.
- Especially in 学生答案、步骤、结论, use plain text math only; no Markdown math, no LaTeX, no dollar signs.

Return only the answer that should be spoken by TTS. Do not mention internal IDs unless useful for debugging.
""".strip()
    return truncate_text(prompt, QA_PROMPT_CHAR_LIMIT)


def insert_qa_event(
    session_id: str,
    *,
    image_id: str | None,
    source: str,
    trigger_type: str,
    question: str,
    focus: dict,
    context: dict,
    gesture: dict,
) -> dict:
    event_id = uuid.uuid4().hex
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO qa_events(
                id, session_id, image_id, source, trigger_type, question,
                focus, context, gesture, status, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                session_id,
                image_id,
                source,
                trigger_type,
                question,
                json_dumps(focus),
                json_dumps(context),
                json_dumps(gesture),
                "running",
                now,
                now,
            ),
        )
    return qa_events_for_session(session_id, 1)[0]


def update_qa_event(event_id: str, *, status: str, answer: str = "", tts_status: str = "", interrupted_at: str = "") -> dict:
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            UPDATE qa_events
            SET status=?, answer=?, tts_status=?, interrupted_at=?, updated_at=?
            WHERE id=?
            """,
            (status, truncate_text(answer, QA_ANSWER_CHAR_LIMIT), tts_status, interrupted_at, now, event_id),
        )
        row = conn.execute(
            """
            SELECT qa_events.*, images.filename AS image_filename
            FROM qa_events
            LEFT JOIN images ON images.id = qa_events.image_id
            WHERE qa_events.id=?
            """,
            (event_id,),
        ).fetchone()
    if not row:
        return {}
    item = qa_event_row_to_dict(dict(row))
    return attach_visualization_metadata([item], "qa_event", text_keys=("question", "answer"))[0]


def global_learning_column_items(page_size: int, account_id: str = "") -> tuple[list[dict], int, list[dict], int]:
    account_id = account_id or get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    with connect() as conn:
        learning_total = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM learning_items li
            LEFT JOIN sessions ON sessions.id = li.session_id
            WHERE sessions.account_id=?
            """,
            (account_id,),
        ).fetchone()["count"]
        mistake_total = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM mistake_items mi
            LEFT JOIN sessions ON sessions.id = mi.session_id
            WHERE sessions.account_id=?
            """,
            (account_id,),
        ).fetchone()["count"]
        learning_items = [
            dict(row)
            for row in conn.execute(
                """
                SELECT li.id, li.session_id, li.batch_id, li.analysis_id, li.item_type,
                       li.title, li.content, li.subject, li.page_ref, li.question_ref,
                       li.location_ref, li.source_summary, li.source_image_details,
                       li.first_seen_at, li.last_seen_at, li.first_sequence_index,
                       li.last_sequence_index, li.evidence_count, li.confidence,
                       li.created_at, li.updated_at, sessions.title AS session_title,
                       sessions.status AS session_status
                FROM learning_items li
                LEFT JOIN sessions ON sessions.id = li.session_id
                WHERE sessions.account_id=?
                ORDER BY li.last_seen_at DESC, li.updated_at DESC, li.id DESC
                LIMIT ?
                """,
                (account_id, page_size),
            )
        ]
        mistake_items = [
            dict(row)
            for row in conn.execute(
                """
                SELECT mi.id, mi.session_id, mi.learning_item_id, mi.batch_id, mi.analysis_id,
                       mi.title, mi.question_text, mi.student_answer, mi.expected_answer,
                       mi.error_reason, mi.knowledge_points, mi.subject, mi.page_ref,
                       mi.question_ref, mi.location_ref, mi.error_type, mi.correction,
                       mi.next_action, mi.source_summary, mi.source_image_details,
                       mi.status, mi.review_state, mi.next_review_at, mi.last_reviewed_at,
                       mi.review_count, mi.review_note, mi.confirmed_at, mi.ignored_at,
                       mi.corrected_at, mi.mastered_at, mi.evidence, mi.first_seen_at, mi.last_seen_at,
                       mi.created_at, mi.updated_at, sessions.title AS session_title,
                       sessions.status AS session_status
                FROM mistake_items mi
                LEFT JOIN sessions ON sessions.id = mi.session_id
                WHERE sessions.account_id=?
                ORDER BY mi.last_seen_at DESC, mi.updated_at DESC, mi.id DESC
                LIMIT ?
                """,
                (account_id, page_size),
            )
        ]
    for row in learning_items:
        row["source_image_details"] = json_list_of_dicts(row.get("source_image_details"))[:2]
    for row in mistake_items:
        row["knowledge_points"] = json_list(row.get("knowledge_points"))
        row["source_image_details"] = json_list_of_dicts(row.get("source_image_details"))[:2]
    return learning_items, learning_total, mistake_items, mistake_total


def normalize_asset_page(page: int) -> int:
    return max(1, int(page or 1))


def normalize_asset_page_size(page_size: int) -> int:
    return max(1, min(ASSET_PAGE_SIZE_MAX, int(page_size or ASSET_PAGE_SIZE_DEFAULT)))


def asset_like_pattern(text: str) -> str:
    return f"%{text.replace('%', '').replace('_', '').strip()}%"


def browse_learning_assets(
    *,
    account_id: str = "",
    session_id: str = "",
    item_type: str = "",
    subject: str = "",
    location: str = "",
    q: str = "",
    page: int = 1,
    page_size: int = ASSET_PAGE_SIZE_DEFAULT,
) -> dict:
    account_id = account_id or get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    backfill_asset_documents()
    page = normalize_asset_page(page)
    page_size = normalize_asset_page_size(page_size)
    where = []
    params: list[object] = []
    where.append("sessions.account_id = ?")
    params.append(account_id)
    if session_id:
        where.append("li.session_id = ?")
        params.append(session_id)
    if item_type:
        where.append("li.item_type = ?")
        params.append(item_type)
    if subject:
        pattern = asset_like_pattern(subject)
        where.append("(li.subject LIKE ? OR ad.subject LIKE ?)")
        params.extend([pattern, pattern])
    if location:
        pattern = asset_like_pattern(location)
        where.append("(li.page_ref LIKE ? OR li.question_ref LIKE ? OR li.location_ref LIKE ? OR ad.location_ref LIKE ?)")
        params.extend([pattern, pattern, pattern, pattern])
    if q:
        pattern = asset_like_pattern(q)
        where.append(
            """
            (
                li.title LIKE ? OR li.content LIKE ? OR li.subject LIKE ?
                OR li.page_ref LIKE ? OR li.question_ref LIKE ? OR li.location_ref LIKE ?
                OR li.source_summary LIKE ? OR ad.body LIKE ? OR ad.search_text LIKE ?
            )
            """
        )
        params.extend([pattern, pattern, pattern, pattern, pattern, pattern, pattern, pattern, pattern])
    where_sql = "WHERE " + " AND ".join(where) if where else ""
    offset = (page - 1) * page_size
    with connect() as conn:
        total = conn.execute(
            f"""
            SELECT COUNT(*) AS count
            FROM learning_items li
            LEFT JOIN asset_documents ad ON ad.asset_kind='learning' AND ad.asset_id=li.id
            LEFT JOIN sessions ON sessions.id = li.session_id
            {where_sql}
            """,
            params,
        ).fetchone()["count"]
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT li.id, li.session_id, li.batch_id, li.analysis_id, li.item_type,
                       li.title, li.content, li.subject, li.page_ref, li.question_ref,
                       li.location_ref, li.source_summary, li.source_image_details,
                       ad.body AS document_body, li.first_seen_at, li.last_seen_at,
                       li.first_sequence_index, li.last_sequence_index, li.source_image_ids,
                       li.evidence_count, li.confidence, li.created_at, li.updated_at,
                       sessions.title AS session_title, sessions.status AS session_status,
                       sessions.created_at AS session_created_at
                FROM learning_items li
                LEFT JOIN asset_documents ad ON ad.asset_kind='learning' AND ad.asset_id=li.id
                LEFT JOIN sessions ON sessions.id = li.session_id
                {where_sql}
                ORDER BY li.last_seen_at DESC, li.updated_at DESC, li.id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, page_size, offset],
            )
        ]
    for row in rows:
        row["source_image_ids"] = json_list(row.get("source_image_ids"))
        row["source_image_details"] = json_list_of_dicts(row.get("source_image_details"))
    return {
        "kind": "learning",
        "items": rows,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_next": offset + len(rows) < total,
        "has_prev": page > 1,
    }


def browse_mistake_assets(
    *,
    account_id: str = "",
    session_id: str = "",
    status: str = "",
    review_state: str = "",
    error_type: str = "",
    subject: str = "",
    location: str = "",
    q: str = "",
    page: int = 1,
    page_size: int = ASSET_PAGE_SIZE_DEFAULT,
) -> dict:
    account_id = account_id or get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    backfill_asset_documents()
    page = normalize_asset_page(page)
    page_size = normalize_asset_page_size(page_size)
    where = []
    params: list[object] = []
    where.append("sessions.account_id = ?")
    params.append(account_id)
    if session_id:
        where.append("mi.session_id = ?")
        params.append(session_id)
    if status:
        where.append("mi.status = ?")
        params.append(status)
    if review_state:
        where.append("mi.review_state = ?")
        params.append(review_state)
    if error_type:
        pattern = asset_like_pattern(error_type)
        where.append("mi.error_type LIKE ?")
        params.append(pattern)
    if subject:
        pattern = asset_like_pattern(subject)
        where.append("(mi.subject LIKE ? OR ad.subject LIKE ?)")
        params.extend([pattern, pattern])
    if location:
        pattern = asset_like_pattern(location)
        where.append("(mi.page_ref LIKE ? OR mi.question_ref LIKE ? OR mi.location_ref LIKE ? OR ad.location_ref LIKE ?)")
        params.extend([pattern, pattern, pattern, pattern])
    if q:
        pattern = asset_like_pattern(q)
        where.append(
            """
            (
                mi.title LIKE ? OR mi.question_text LIKE ? OR mi.student_answer LIKE ?
                OR mi.expected_answer LIKE ? OR mi.error_reason LIKE ? OR mi.evidence LIKE ?
                OR mi.knowledge_points LIKE ? OR mi.subject LIKE ? OR mi.page_ref LIKE ?
                OR mi.question_ref LIKE ? OR mi.location_ref LIKE ? OR mi.error_type LIKE ?
                OR mi.correction LIKE ? OR mi.next_action LIKE ? OR mi.source_summary LIKE ?
                OR ad.body LIKE ? OR ad.search_text LIKE ?
            )
            """
        )
        params.extend([pattern] * 17)
    where_sql = "WHERE " + " AND ".join(where) if where else ""
    offset = (page - 1) * page_size
    with connect() as conn:
        total = conn.execute(
            f"""
            SELECT COUNT(*) AS count
            FROM mistake_items mi
            LEFT JOIN asset_documents ad ON ad.asset_kind='mistake' AND ad.asset_id=mi.id
            LEFT JOIN sessions ON sessions.id = mi.session_id
            {where_sql}
            """,
            params,
        ).fetchone()["count"]
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT mi.id, mi.session_id, mi.learning_item_id, mi.batch_id, mi.analysis_id,
                       mi.title, mi.question_text, mi.student_answer, mi.expected_answer,
                       mi.error_reason, mi.knowledge_points, mi.subject, mi.page_ref,
                       mi.question_ref, mi.location_ref, mi.error_type, mi.correction,
                       mi.next_action, mi.source_summary, mi.source_image_details,
                       ad.body AS document_body, mi.status, mi.review_state, mi.next_review_at,
                       mi.last_reviewed_at, mi.review_count, mi.review_note,
                       mi.confirmed_at, mi.ignored_at, mi.corrected_at, mi.mastered_at, mi.evidence,
                       mi.source_image_ids, mi.first_seen_at, mi.last_seen_at,
                       mi.created_at, mi.updated_at,
                       sessions.title AS session_title, sessions.status AS session_status,
                       sessions.created_at AS session_created_at
                FROM mistake_items mi
                LEFT JOIN asset_documents ad ON ad.asset_kind='mistake' AND ad.asset_id=mi.id
                LEFT JOIN sessions ON sessions.id = mi.session_id
                {where_sql}
                ORDER BY mi.last_seen_at DESC, mi.updated_at DESC, mi.id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, page_size, offset],
            )
        ]
    for row in rows:
        row["knowledge_points"] = json_list(row.get("knowledge_points"))
        row["source_image_ids"] = json_list(row.get("source_image_ids"))
        row["source_image_details"] = json_list_of_dicts(row.get("source_image_details"))
    return {
        "kind": "mistake",
        "items": rows,
        "total": total,
        "page": page,
        "page_size": page_size,
        "has_next": offset + len(rows) < total,
        "has_prev": page > 1,
    }


def get_mistake_item(mistake_id: str) -> dict:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT mi.*, sessions.title AS session_title, sessions.status AS session_status,
                   sessions.created_at AS session_created_at
            FROM mistake_items mi
            LEFT JOIN sessions ON sessions.id = mi.session_id
            WHERE mi.id=?
            """,
            (mistake_id,),
        ).fetchone()
    if not row:
        raise HTTPException(404, "mistake not found")
    return mistake_row_to_dict(row)


def update_mistake_item(mistake_id: str, updates: dict) -> dict:
    now = utc_now()
    with connect() as conn:
        existing = conn.execute("SELECT * FROM mistake_items WHERE id=?", (mistake_id,)).fetchone()
        if not existing:
            raise HTTPException(404, "mistake not found")
        existing_data = dict(existing)
        status = existing_data["status"]
        review_state = existing_data["review_state"] or "new"
        assignments = []
        params: list[object] = []

        if "status" in updates:
            status = normalize_mistake_status(updates.get("status"), default=status)
            assignments.append("status=?")
            params.append(status)
            timestamp_column = {
                "confirmed": "confirmed_at",
                "ignored": "ignored_at",
                "corrected": "corrected_at",
                "mastered": "mastered_at",
            }.get(status)
            if timestamp_column and not existing_data.get(timestamp_column):
                assignments.append(f"{timestamp_column}=?")
                params.append(now)

        if "review_state" in updates:
            review_state = normalize_review_state(updates.get("review_state"), default=review_state)
            assignments.append("review_state=?")
            params.append(review_state)
            if review_state == "mastered" and not existing_data.get("mastered_at"):
                assignments.append("mastered_at=?")
                params.append(now)
            if review_state == "ignored" and not existing_data.get("ignored_at"):
                assignments.append("ignored_at=?")
                params.append(now)

        if "review_note" in updates:
            assignments.append("review_note=?")
            params.append(clean_user_text(updates.get("review_note"), 1200))

        if "correction" in updates:
            assignments.append("correction=?")
            params.append(clean_user_text(updates.get("correction"), ASSET_SOURCE_SUMMARY_LIMIT))

        if "next_action" in updates:
            assignments.append("next_action=?")
            params.append(clean_user_text(updates.get("next_action"), ASSET_SOURCE_SUMMARY_LIMIT))

        if "error_type" in updates:
            assignments.append("error_type=?")
            params.append(clean_asset_field(updates.get("error_type")))

        if status in {"ignored", "mastered"} or review_state in {"ignored", "mastered"}:
            next_review_at = ""
        elif "next_review_at" in updates:
            raw_next = str(updates.get("next_review_at") or "").strip()
            if raw_next and not parse_date_or_datetime(raw_next):
                raise HTTPException(422, "invalid next_review_at")
            next_review_at = raw_next
        elif status != existing_data["status"] or review_state != existing_data["review_state"]:
            next_review_at = review_due_at_for(status, review_state)
        else:
            next_review_at = existing_data["next_review_at"] or review_due_at_for(status, review_state)
        assignments.append("next_review_at=?")
        params.append(next_review_at)

        if updates.get("mark_reviewed") or review_state in {"done", "mastered"} or status in {"corrected", "mastered"}:
            assignments.append("last_reviewed_at=?")
            params.append(now)
            assignments.append("review_count=review_count + 1")

        if not assignments:
            return mistake_row_to_dict(existing)

        assignments.append("updated_at=?")
        params.append(now)
        params.append(mistake_id)
        conn.execute(f"UPDATE mistake_items SET {', '.join(assignments)} WHERE id=?", params)
        sync_asset_document(conn, "mistake", mistake_id)
        row = conn.execute(
            """
            SELECT mi.*, sessions.title AS session_title, sessions.status AS session_status,
                   sessions.created_at AS session_created_at
            FROM mistake_items mi
            LEFT JOIN sessions ON sessions.id = mi.session_id
            WHERE mi.id=?
            """,
            (mistake_id,),
        ).fetchone()
    return mistake_row_to_dict(row)


def review_event_row_to_dict(row) -> dict:
    item = dict(row)
    try:
        payload = json.loads(item.get("payload") or "{}")
    except json.JSONDecodeError:
        payload = {}
    item["payload"] = payload if isinstance(payload, dict) else {}
    return item


def list_review_events_for_mistake(mistake_id: str, limit: int = 60) -> dict:
    limit = max(1, min(200, int(limit or 60)))
    with connect() as conn:
        mistake = conn.execute("SELECT id FROM mistake_items WHERE id=?", (mistake_id,)).fetchone()
        if not mistake:
            raise HTTPException(404, "mistake not found")
        rows = [
            review_event_row_to_dict(row)
            for row in conn.execute(
                """
                SELECT re.*, mi.title AS mistake_title, mi.subject, mi.page_ref, mi.question_ref, mi.error_type
                FROM review_events re
                LEFT JOIN mistake_items mi ON mi.id = re.mistake_id
                WHERE re.mistake_id=?
                ORDER BY re.reviewed_at DESC, re.created_at DESC
                LIMIT ?
                """,
                (mistake_id, limit),
            )
        ]
    return {"items": rows, "total": len(rows)}


def create_review_event(mistake_id: str, body: dict) -> dict:
    body = body if isinstance(body, dict) else {}
    result = normalize_review_event_result(body.get("result") or body.get("review_result") or body.get("event_result"))
    note = clean_user_text(body.get("review_note") or body.get("note"), 1200)
    source = clean_user_text(body.get("source") or "", 80)
    event_type = clean_user_text(body.get("event_type") or "review", 80) or "review"
    duration_seconds = optional_int(body.get("duration_seconds", body.get("durationSeconds")))
    score = optional_float(body.get("score"))
    reviewed_at_raw = str(body.get("reviewed_at") or body.get("reviewedAt") or "").strip()
    reviewed_dt = parse_date_or_datetime(reviewed_at_raw) if reviewed_at_raw else datetime.now(timezone.utc)
    if reviewed_at_raw and reviewed_dt is None:
        raise HTTPException(422, "invalid reviewed_at")
    reviewed_at = (reviewed_dt or datetime.now(timezone.utc)).isoformat()
    created_at = utc_now()
    event_id = uuid.uuid4().hex

    payload = {
        key: value
        for key, value in body.items()
        if key
        not in {
            "result",
            "review_result",
            "event_result",
            "review_note",
            "note",
            "source",
            "event_type",
            "duration_seconds",
            "durationSeconds",
            "score",
            "reviewed_at",
            "reviewedAt",
        }
    }

    with connect() as conn:
        existing = conn.execute("SELECT * FROM mistake_items WHERE id=?", (mistake_id,)).fetchone()
        if not existing:
            raise HTTPException(404, "mistake not found")
        mistake = dict(existing)
        status = normalize_mistake_status(mistake.get("status"), default="suspected")
        review_state = normalize_review_state(mistake.get("review_state"), default="new")
        next_review_at = ""
        updates = [
            "last_reviewed_at=?",
            "review_count=review_count + 1",
            "review_note=?",
            "updated_at=?",
        ]
        params: list[object] = [reviewed_at, note or mistake.get("review_note") or "", created_at]

        if result == "mastered":
            status = "mastered"
            review_state = "mastered"
            next_review_at = ""
            updates.extend(["status=?", "review_state=?", "next_review_at=?"])
            params.extend([status, review_state, next_review_at])
            if not mistake.get("mastered_at"):
                updates.append("mastered_at=?")
                params.append(reviewed_at)
        elif result == "correct":
            status = "corrected"
            review_state = "done"
            next_review_at = review_due_at_for(status, review_state, reviewed_dt)
            updates.extend(["status=?", "review_state=?", "next_review_at=?"])
            params.extend([status, review_state, next_review_at])
            if not mistake.get("corrected_at"):
                updates.append("corrected_at=?")
                params.append(reviewed_at)
        elif result == "incorrect":
            status = "confirmed"
            review_state = "scheduled"
            next_review_at = review_due_at_for(status, review_state, reviewed_dt)
            updates.extend(["status=?", "review_state=?", "next_review_at=?"])
            params.extend([status, review_state, next_review_at])
            if not mistake.get("confirmed_at"):
                updates.append("confirmed_at=?")
                params.append(reviewed_at)
        elif result == "postpone":
            if status in {"ignored", "mastered"}:
                status = "confirmed"
            review_state = "scheduled"
            requested_next = str(body.get("next_review_at") or body.get("nextReviewAt") or "").strip()
            if requested_next:
                if not parse_date_or_datetime(requested_next):
                    raise HTTPException(422, "invalid next_review_at")
                next_review_at = requested_next
            else:
                next_review_at = review_due_at_for(status, review_state, reviewed_dt)
            updates.extend(["status=?", "review_state=?", "next_review_at=?"])
            params.extend([status, review_state, next_review_at])

        conn.execute(
            """
            INSERT INTO review_events(
                id, mistake_id, session_id, event_type, result, note, source,
                duration_seconds, score, payload, reviewed_at, created_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                mistake_id,
                mistake["session_id"],
                event_type,
                result,
                note,
                source,
                duration_seconds,
                score,
                json_dumps(payload),
                reviewed_at,
                created_at,
            ),
        )
        params.append(mistake_id)
        conn.execute(f"UPDATE mistake_items SET {', '.join(updates)} WHERE id=?", params)
        sync_asset_document(conn, "mistake", mistake_id)
        event_row = conn.execute(
            """
            SELECT re.*, mi.title AS mistake_title, mi.subject, mi.page_ref, mi.question_ref, mi.error_type
            FROM review_events re
            LEFT JOIN mistake_items mi ON mi.id = re.mistake_id
            WHERE re.id=?
            """,
            (event_id,),
        ).fetchone()
        mistake_row = conn.execute(
            """
            SELECT mi.*, sessions.title AS session_title, sessions.status AS session_status,
                   sessions.created_at AS session_created_at
            FROM mistake_items mi
            LEFT JOIN sessions ON sessions.id = mi.session_id
            WHERE mi.id=?
            """,
            (mistake_id,),
        ).fetchone()
    return {"event": review_event_row_to_dict(event_row), "mistake": mistake_row_to_dict(mistake_row)}


def review_queue_items(
    *,
    account_id: str = "",
    status: str = "",
    subject: str = "",
    page_ref: str = "",
    question_ref: str = "",
    item_type: str = "",
    error_type: str = "",
    error_reason: str = "",
    q: str = "",
    due_only: bool = True,
    page_size: int = ASSET_PAGE_SIZE_DEFAULT,
) -> dict:
    account_id = account_id or get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    now = datetime.now(timezone.utc)
    page_size = normalize_asset_page_size(page_size)
    where = [
        "sessions.account_id=?",
        "mi.status IN ('suspected', 'incomplete', 'confirmed', 'corrected')",
        "mi.review_state NOT IN ('mastered', 'ignored')",
    ]
    params: list[object] = [account_id]
    if status:
        where.append("mi.status=?")
        params.append(normalize_mistake_status(status))
    if subject:
        pattern = asset_like_pattern(subject)
        where.append("mi.subject LIKE ?")
        params.append(pattern)
    if page_ref:
        pattern = asset_like_pattern(page_ref)
        where.append("(mi.page_ref LIKE ? OR mi.location_ref LIKE ?)")
        params.extend([pattern, pattern])
    if question_ref:
        pattern = asset_like_pattern(question_ref)
        where.append("(mi.question_ref LIKE ? OR mi.question_text LIKE ? OR mi.location_ref LIKE ?)")
        params.extend([pattern, pattern, pattern])
    if item_type:
        pattern = asset_like_pattern(item_type)
        where.append("(mi.question_ref LIKE ? OR mi.question_text LIKE ? OR mi.title LIKE ?)")
        params.extend([pattern, pattern, pattern])
    if error_type:
        pattern = asset_like_pattern(error_type)
        where.append("mi.error_type LIKE ?")
        params.append(pattern)
    if error_reason:
        pattern = asset_like_pattern(error_reason)
        where.append("(mi.error_reason LIKE ? OR mi.evidence LIKE ?)")
        params.extend([pattern, pattern])
    if q:
        pattern = asset_like_pattern(q)
        where.append(
            """
            (
                mi.title LIKE ? OR mi.question_text LIKE ? OR mi.student_answer LIKE ?
                OR mi.expected_answer LIKE ? OR mi.error_reason LIKE ? OR mi.evidence LIKE ?
                OR mi.knowledge_points LIKE ? OR mi.subject LIKE ? OR mi.page_ref LIKE ?
                OR mi.question_ref LIKE ? OR mi.location_ref LIKE ? OR mi.error_type LIKE ?
                OR mi.correction LIKE ? OR mi.next_action LIKE ?
            )
            """
        )
        params.extend([pattern] * 14)
    if due_only:
        where.append("(mi.next_review_at='' OR mi.next_review_at <= ?)")
        params.append(now.isoformat())
    where_sql = "WHERE " + " AND ".join(where)
    with connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT mi.*, sessions.title AS session_title, sessions.status AS session_status,
                       sessions.created_at AS session_created_at
                FROM mistake_items mi
                LEFT JOIN sessions ON sessions.id = mi.session_id
                {where_sql}
                ORDER BY
                    CASE WHEN mi.next_review_at='' THEN 0 ELSE 1 END,
                    mi.next_review_at ASC,
                    mi.last_seen_at DESC,
                    mi.updated_at DESC
                LIMIT ?
                """,
                [*params, page_size],
            )
        ]
        total = conn.execute(
            f"""
            SELECT COUNT(*) AS count
            FROM mistake_items mi
            LEFT JOIN sessions ON sessions.id = mi.session_id
            {where_sql}
            """,
            params,
        ).fetchone()["count"]
    items = [mistake_row_to_dict(row) for row in rows]
    for item in items:
        due_at = parse_date_or_datetime(item.get("next_review_at"))
        item["is_due"] = not item.get("next_review_at") or (due_at is not None and due_at <= now)
    return {"items": items, "total": total, "due_only": due_only, "page_size": page_size}


def mistake_candidate_items(
    *,
    account_id: str = "",
    subject: str = "",
    q: str = "",
    page_size: int = ASSET_PAGE_SIZE_DEFAULT,
) -> dict:
    """智能观察异步提取出的“可疑错题候选”(status=candidate)，等学生人工确认导入。
    刻意与正式错题本/复习队列分开：review_queue_items 不含 candidate，这里只含 candidate。"""
    account_id = account_id or get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    page_size = normalize_asset_page_size(page_size)
    where = ["sessions.account_id=?", "mi.status='candidate'"]
    params: list[object] = [account_id]
    if subject:
        where.append("mi.subject LIKE ?")
        params.append(asset_like_pattern(subject))
    if q:
        pattern = asset_like_pattern(q)
        where.append(
            "(mi.title LIKE ? OR mi.question_text LIKE ? OR mi.student_answer LIKE ? "
            "OR mi.error_reason LIKE ? OR mi.evidence LIKE ? OR mi.subject LIKE ?)"
        )
        params.extend([pattern] * 6)
    where_sql = "WHERE " + " AND ".join(where)
    with connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT mi.*, sessions.title AS session_title, sessions.status AS session_status,
                       sessions.created_at AS session_created_at
                FROM mistake_items mi
                LEFT JOIN sessions ON sessions.id = mi.session_id
                {where_sql}
                ORDER BY mi.last_seen_at DESC, mi.created_at DESC
                LIMIT ?
                """,
                [*params, page_size],
            )
        ]
        total = conn.execute(
            f"""
            SELECT COUNT(*) AS count
            FROM mistake_items mi
            LEFT JOIN sessions ON sessions.id = mi.session_id
            {where_sql}
            """,
            params,
        ).fetchone()["count"]
    return {"items": [mistake_row_to_dict(row) for row in rows], "total": total, "page_size": page_size}


def owned_mistake_status(mistake_id: str, account_id: str) -> str | None:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT mi.status
            FROM mistake_items mi
            LEFT JOIN sessions ON sessions.id = mi.session_id
            WHERE mi.id=? AND sessions.account_id=?
            """,
            (mistake_id, account_id),
        ).fetchone()
    return row["status"] if row else None


async def _optional_json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        return {}
    return body if isinstance(body, dict) else {}


def add_metric(metrics: dict[str, dict], key: str, *, weight: int = 1, **extra) -> None:
    normalized = str(key or "").strip()
    if not normalized:
        return
    item = metrics.setdefault(normalized, {"label": normalized, "count": 0})
    item["count"] += weight
    for extra_key, extra_value in extra.items():
        if extra_value not in (None, "", []):
            item.setdefault(extra_key, extra_value)


def ranked_metrics(metrics: dict[str, dict], limit: int = 12) -> list[dict]:
    return sorted(metrics.values(), key=lambda item: (-int(item.get("count") or 0), item.get("label") or ""))[:limit]


def student_profile(account_id: str = "") -> dict:
    account_id = account_id or get_settings().default_account_id or DEFAULT_ACCOUNT_ID
    knowledge_metrics: dict[str, dict] = {}
    error_metrics: dict[str, dict] = {}
    subject_metrics: dict[str, dict] = {}
    with connect() as conn:
        mistake_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT mi.*, sessions.title AS session_title, sessions.created_at AS session_created_at
                FROM mistake_items mi
                LEFT JOIN sessions ON sessions.id = mi.session_id
                WHERE sessions.account_id=?
                ORDER BY mi.updated_at DESC, mi.last_seen_at DESC
                LIMIT 800
                """
                ,
                (account_id,),
            )
        ]
        learning_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT li.subject, li.item_type, li.title, li.content, li.evidence_count, li.updated_at
                FROM learning_items li
                LEFT JOIN sessions ON sessions.id = li.session_id
                WHERE sessions.account_id=?
                ORDER BY li.updated_at DESC
                LIMIT 800
                """
                ,
                (account_id,),
            )
        ]
        recent_sessions = [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, title, status, created_at, updated_at, student_goal,
                       (SELECT COUNT(*) FROM learning_items WHERE learning_items.session_id = sessions.id) AS learning_count,
                       (SELECT COUNT(*) FROM mistake_items WHERE mistake_items.session_id = sessions.id) AS mistake_count,
                       (SELECT COUNT(*) FROM mistake_items WHERE mistake_items.session_id = sessions.id AND status IN ('confirmed','corrected','mastered')) AS progressed_mistake_count
                FROM sessions
                WHERE account_id=?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 20
                """
                ,
                (account_id,),
            )
        ]
        recent_review_events = [
            review_event_row_to_dict(row)
            for row in conn.execute(
                """
                SELECT re.*, mi.title AS mistake_title, mi.question_text, mi.subject,
                       mi.page_ref, mi.question_ref, mi.error_type
                FROM review_events re
                LEFT JOIN mistake_items mi ON mi.id = re.mistake_id
                LEFT JOIN sessions ON sessions.id = re.session_id
                WHERE sessions.account_id=?
                ORDER BY re.reviewed_at DESC, re.created_at DESC
                LIMIT 30
                """
                ,
                (account_id,),
            )
        ]
        review_result_rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT result, COUNT(*) AS count
                FROM review_events
                LEFT JOIN sessions ON sessions.id = review_events.session_id
                WHERE sessions.account_id=?
                GROUP BY result
                """
                ,
                (account_id,),
            )
        ]
        review_totals = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(DISTINCT mistake_id) AS reviewed_mistakes,
                AVG(duration_seconds) AS avg_duration_seconds
            FROM review_events
            LEFT JOIN sessions ON sessions.id = review_events.session_id
            WHERE sessions.account_id=?
            """
            ,
            (account_id,),
        ).fetchone()
        queue_counts = conn.execute(
            """
            SELECT
                SUM(CASE WHEN mi.status IN ('suspected','incomplete','confirmed','corrected')
                          AND mi.review_state NOT IN ('mastered','ignored') THEN 1 ELSE 0 END) AS active_count,
                SUM(CASE WHEN mi.status IN ('suspected','incomplete','confirmed','corrected')
                          AND mi.review_state NOT IN ('mastered','ignored')
                          AND (mi.next_review_at='' OR mi.next_review_at <= ?) THEN 1 ELSE 0 END) AS due_count,
                SUM(CASE WHEN mi.status='mastered' OR mi.review_state='mastered' THEN 1 ELSE 0 END) AS mastered_count,
                SUM(CASE WHEN mi.status='candidate' THEN 1 ELSE 0 END) AS candidate_count
            FROM mistake_items mi
            LEFT JOIN sessions ON sessions.id = mi.session_id
            WHERE sessions.account_id=?
            """,
            (utc_now(), account_id),
        ).fetchone()

    for row in mistake_rows:
        # 候选还没被学生确认，不计入知识点/错因/科目画像，避免未确认的猜测污染学情。
        if row.get("status") == "candidate":
            continue
        is_ignored = row.get("status") == "ignored" or row.get("review_state") == "ignored"
        is_mastered = row.get("status") == "mastered" or row.get("review_state") == "mastered"
        if not is_ignored and not is_mastered:
            weight = max(1, int(row.get("review_count") or 0) + 1)
            points = json_list(row.get("knowledge_points"))
            for point in points:
                for part in re.split(r"[、,，;；\n]+", point):
                    add_metric(knowledge_metrics, part, weight=weight, subject=row.get("subject"), last_seen_at=row.get("last_seen_at"))
            if not points:
                add_metric(knowledge_metrics, row.get("title"), weight=1, subject=row.get("subject"))
            add_metric(error_metrics, row.get("error_type") or row.get("error_reason"), weight=weight, subject=row.get("subject"))
        subject = row.get("subject") or "未识别"
        add_metric(subject_metrics, subject, weight=1)
        subject_metrics[subject]["mistake_count"] = subject_metrics[subject].get("mistake_count", 0) + 1
        if row.get("status") in {"corrected", "mastered"}:
            subject_metrics[subject]["progressed_count"] = subject_metrics[subject].get("progressed_count", 0) + 1
        if is_ignored:
            subject_metrics[subject]["ignored_count"] = subject_metrics[subject].get("ignored_count", 0) + 1
        if is_mastered:
            subject_metrics[subject]["mastered_count"] = subject_metrics[subject].get("mastered_count", 0) + 1

    for row in learning_rows:
        subject = row.get("subject") or "未识别"
        add_metric(subject_metrics, subject, weight=0)
        subject_metrics[subject]["learning_count"] = subject_metrics[subject].get("learning_count", 0) + 1

    result_counts = {row["result"] or "unknown": int(row["count"] or 0) for row in review_result_rows}
    total_events = int(review_totals["total"] or 0) if review_totals else 0
    correct_events = result_counts.get("correct", 0) + result_counts.get("mastered", 0)
    incorrect_events = result_counts.get("incorrect", 0)
    review_summary = {
        "total_events": total_events,
        "reviewed_mistakes": int(review_totals["reviewed_mistakes"] or 0) if review_totals else 0,
        "correct_events": correct_events,
        "incorrect_events": incorrect_events,
        "postponed_events": result_counts.get("postpone", 0),
        "mastered_events": result_counts.get("mastered", 0),
        "active_review_count": int(queue_counts["active_count"] or 0) if queue_counts else 0,
        "due_review_count": int(queue_counts["due_count"] or 0) if queue_counts else 0,
        "mastered_mistake_count": int(queue_counts["mastered_count"] or 0) if queue_counts else 0,
        "candidate_count": int(queue_counts["candidate_count"] or 0) if queue_counts else 0,
        "correction_efficiency": round(correct_events / total_events, 3) if total_events else 0,
    }
    if review_totals and review_totals["avg_duration_seconds"] is not None:
        review_summary["avg_duration_seconds"] = round(float(review_totals["avg_duration_seconds"]), 1)

    return {
        "weak_knowledge_points": ranked_metrics(knowledge_metrics),
        "common_error_types": ranked_metrics(error_metrics),
        "subject_breakdown": ranked_metrics(subject_metrics),
        "recent_sessions": recent_sessions,
        "review_summary": review_summary,
        "recent_review_events": recent_review_events,
    }


def folder_size_bytes(path: Path, limit: int = 5000) -> int:
    total = 0
    scanned = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        try:
            total += item.stat().st_size
        except OSError:
            pass
        scanned += 1
        if scanned >= limit:
            break
    return total


def observability_snapshot(account_id: str = "", user_id: str = "") -> dict:
    settings = get_settings()
    account_id = account_id or settings.default_account_id or DEFAULT_ACCOUNT_ID
    data_dir = settings.data_dir
    db_file = account_db_path(account_id)
    with connect() as conn:
        counts = {}
        for table in ("sessions", "logs", "task_runs", "memory_events", "llm_usage_events"):
            try:
                counts[table] = conn.execute(f"SELECT COUNT(*) AS count FROM {table} WHERE account_id=?", (account_id,)).fetchone()["count"]
            except Exception:
                counts[table] = 0
        joined_counts = {
            "images": "images.session_id = sessions.id",
            "analyses": "analyses.session_id = sessions.id",
            "learning_items": "learning_items.session_id = sessions.id",
            "mistake_items": "mistake_items.session_id = sessions.id",
            "review_events": "review_events.session_id = sessions.id",
            "asset_documents": "asset_documents.session_id = sessions.id",
        }
        for table, join_condition in joined_counts.items():
            try:
                counts[table] = conn.execute(
                    f"SELECT COUNT(*) AS count FROM {table} LEFT JOIN sessions ON {join_condition} WHERE sessions.account_id=?",
                    (account_id,),
                ).fetchone()["count"]
            except Exception:
                counts[table] = 0
        recent_failures = [
            dict(row)
            for row in conn.execute(
                """
                SELECT analyses.id, analyses.session_id, analyses.scope, analyses.status,
                       substr(analyses.content, 1, 400) AS content, analyses.updated_at
                FROM analyses
                LEFT JOIN sessions ON sessions.id = analyses.session_id
                WHERE analyses.status='failed' AND sessions.account_id=?
                ORDER BY analyses.updated_at DESC
                LIMIT 20
                """,
                (account_id,),
            )
        ]
        task_summary = [
            dict(row)
            for row in conn.execute(
                "SELECT status, COUNT(*) AS count FROM task_runs WHERE account_id=? GROUP BY status ORDER BY status",
                (account_id,),
            )
        ]
    waiting_counts = llm_gate_waiting_counts()
    active_config = active_model_config(account_id, user_id)
    active_max_concurrency = normalize_llm_max_concurrency(active_config.get("max_concurrency") or settings.llm_max_concurrency)
    active_min_interval = normalize_llm_min_interval(
        active_config.get("min_interval_seconds")
        if active_config.get("min_interval_seconds") is not None
        else settings.llm_min_interval_seconds
    )
    return {
        "account_id": account_id,
        "llm_gate": {
            "inflight": llm_gate_inflight,
            "waiting": llm_gate_waiting,
            "waiting_realtime": waiting_counts["realtime"],
            "waiting_background": waiting_counts["background"],
            "inflight_realtime": llm_gate_inflight_realtime,
            "inflight_background": llm_gate_inflight_background,
            "last_started_at": llm_gate_last_started_at,
            "max_concurrency": active_max_concurrency,
            "min_interval_seconds": active_min_interval,
            "queue_warn_size": max(1, int(settings.llm_queue_warn_size or 1)),
            "active_provider": active_config.get("provider"),
            "active_model": active_config.get("model"),
            "active_base_url": active_config.get("base_url"),
        },
        "llm_usage": llm_usage_snapshot(account_id),
        "quota": free_quota_state(account_id),
        "model_health": {
            "default_base_url": settings.llm_base_url,
            "active_config": active_config,
            "busy": llm_gate_inflight >= active_max_concurrency,
        },
        "cache": {
            "redis_url_configured": bool(settings.redis_url),
            "redis_status": "configured" if settings.redis_url else "not_configured",
        },
        "storage": {
            "data_dir": str(data_dir),
            "db_path": str(db_file),
            "database_path": str(db_file),
            "database_url": settings.database_url,
            "db_bytes": db_file.stat().st_size if db_file.is_file() else 0,
            "images_bytes": folder_size_bytes(data_dir / "images"),
            "thumbnails_bytes": folder_size_bytes(data_dir / "thumbnails"),
        },
        "counts": counts,
        "task_runs": task_summary,
        "recent_failures": recent_failures,
    }


def render_learning_items_for_prompt(items: list[dict], max_chars: int = LEARNING_ITEM_SUMMARY_LIMIT) -> tuple[str, bool]:
    if not items:
        return "暂无结构化学习条目。", False
    labels = {"question": "题目", "section": "板块", "knowledge": "知识点", "answer": "作答"}
    lines = [
        (
            f"{index + 1}. {labels.get(row['item_type'], row['item_type'])}：{row['content']}；"
            f"科目={row.get('subject') or '未识别'}；位置={row.get('location_ref') or '未识别'}；"
            f"first={row['first_seen_at'] or '未知'}；last={row['last_seen_at'] or '未知'}；"
            f"sequence={row['first_sequence_index']}-{row['last_sequence_index']}；证据={row['evidence_count']}"
        )
        for index, row in enumerate(items)
    ]
    return render_limited_lines(lines, max_chars, "条结构化学习记录")


def render_mistake_items_for_prompt(items: list[dict], max_chars: int = 5000) -> tuple[str, bool]:
    if not items:
        return "暂无明确错题本候选。", False
    lines = [
        (
            f"{index + 1}. {row.get('title') or '错题候选'}；状态={row.get('status') or 'suspected'}；"
            f"科目={row.get('subject') or '未识别'}；位置={row.get('location_ref') or '未识别'}；"
            f"题目={row.get('question_text') or '未识别'}；学生答案={row.get('student_answer') or '未识别'}；"
            f"参考答案={row.get('expected_answer') or '未识别'}；错误类型={row.get('error_type') or '未识别'}；"
            f"错因/证据={row.get('error_reason') or row.get('evidence') or '未识别'}；"
            f"订正={row.get('correction') or '未识别'}；下一步={row.get('next_action') or '未识别'}；"
            f"知识点={','.join(row.get('knowledge_points') or []) or '未识别'}"
        )
        for index, row in enumerate(items)
    ]
    return render_limited_lines(lines, max_chars, "条错题本候选")


def build_report_process_note(session: dict, learning_items: list[dict], mistake_items: list[dict], stats: dict) -> str:
    strategy = session_strategy(session)
    strategy_text = build_strategy_context(strategy)
    item_counts: dict[str, int] = {}
    for item in learning_items:
        item_counts[item.get("item_type") or "unknown"] = item_counts.get(item.get("item_type") or "unknown", 0) + 1
    return (
        "报告生成依据摘要：\n"
        f"- 学生/系统目标：{strategy_text}\n"
        f"- 结构化学习条目：题目 {item_counts.get('question', 0)}，板块 {item_counts.get('section', 0)}，"
        f"知识点 {item_counts.get('knowledge', 0)}，作答 {item_counts.get('answer', 0)}\n"
        f"- 错题本候选：{len(mistake_items)} 条\n"
        f"- 视觉证据：抓拍 {stats['image_count']} 张，批次分析 {stats['analysis_count']} 条，"
        f"去重后可用分析 {stats['unique_done_analysis_count']} 条，重复分析 {stats['duplicate_done_analysis_count']} 条\n"
        "- 报告生成方式：先收集关键帧和差异分析，再合并学习条目、错题候选、时间线和用户要求，最后生成报告。"
    )


def build_distill_source_notes(analyses: list[dict], max_chars: int = FINAL_REPORT_DISTILL_SOURCE_CHAR_LIMIT) -> tuple[list[str], dict]:
    unique_rows, duplicate_count = dedupe_analyses(analyses)
    if not unique_rows:
        return ["暂无可提炼的批次视觉分析内容。"], {"duplicate_done_analysis_count": duplicate_count, "source_chars": 0}
    per_note_limit = max(
        FINAL_REPORT_DISTILL_ANALYSIS_MIN_CHARS,
        min(FINAL_REPORT_DISTILL_ANALYSIS_MAX_CHARS, max_chars // max(len(unique_rows), 1) - 120),
    )
    notes = [build_evidence_note(row, index + 1, per_note_limit) for index, row in enumerate(unique_rows)]
    total_chars = sum(len(note) for note in notes)
    if total_chars <= max_chars:
        return notes, {"duplicate_done_analysis_count": duplicate_count, "source_chars": total_chars}
    selected = sorted(representative_indices(len(notes)))
    while selected:
        rendered_notes = [notes[index] for index in selected]
        omitted = len(notes) - len(rendered_notes)
        rendered_chars = sum(len(note) for note in rendered_notes)
        if rendered_chars <= max_chars:
            if omitted:
                rendered_notes.append(f"... 已省略 {omitted} 条重复或代表性较低的批次分析，保留开头/中段/结尾证据 ...")
            return rendered_notes, {"duplicate_done_analysis_count": duplicate_count, "source_chars": rendered_chars, "omitted_note_count": omitted}
        selected.pop(len(selected) // 2)
    fallback = [truncate_text("\n".join(notes), max_chars)]
    return fallback, {"duplicate_done_analysis_count": duplicate_count, "source_chars": len(fallback[0]), "omitted_note_count": len(notes) - 1}


def chunk_text_blocks(blocks: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0
    for block in blocks:
        block_chars = len(block)
        if current and current_chars + block_chars + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_chars = 0
        if block_chars > max_chars:
            chunks.append(truncate_text(block, max_chars))
            continue
        current.append(block)
        current_chars += block_chars + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def build_distill_prompt(session: dict, images: list[dict], notes_chunk: str, chunk_index: int, chunk_count: int, stats: dict) -> str:
    start, end, total_seconds = report_time_bounds(session, images)
    timeline_lines = build_timeline_lines(session, images)
    timeline, timeline_compressed = render_limited_lines(timeline_lines, 5000, "条抓拍时间线") if timeline_lines else ("无抓拍图片。", False)
    time_weight_summary = build_time_weight_summary(session, images)
    timeline = f"{time_weight_summary}\n{timeline}" if timeline else time_weight_summary
    compressed_notice = "时间线已抽样压缩；" if timeline_compressed else ""
    return prompts.render_prompt(
        "distill_final_evidence",
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        compressed_notice=compressed_notice,
        image_count=stats["image_count"],
        analysis_count=stats["analysis_count"],
        unique_done_analysis_count=stats["unique_done_analysis_count"],
        duplicate_done_analysis_count=stats["duplicate_done_analysis_count"],
        start=start.isoformat() if start else "未知",
        end=end.isoformat() if end else session.get("finished_at") or "未知",
        total_duration=format_duration(total_seconds),
        timeline=timeline,
        notes_chunk=notes_chunk,
    )


def build_final_report_prompt(
    session: dict,
    images: list[dict],
    analyses: list[dict],
    distilled_notes: str | None = None,
    learning_items: list[dict] | None = None,
    mistake_items: list[dict] | None = None,
) -> str:
    start, end, total_seconds = report_time_bounds(session, images)
    timeline_lines = build_timeline_lines(session, images)
    if timeline_lines:
        timeline, timeline_compressed = render_limited_lines(timeline_lines, FINAL_REPORT_TIMELINE_CHAR_LIMIT, "条抓拍时间线")
    else:
        timeline, timeline_compressed = "无抓拍图片。", False
    time_weight_summary = build_time_weight_summary(session, images)
    timeline = f"{time_weight_summary}\n{timeline}"
    if distilled_notes is None:
        batch_notes, analyses_compressed = build_limited_batch_notes(analyses, FINAL_REPORT_ANALYSES_CHAR_LIMIT)
        evidence_label = "批次视觉分析"
    else:
        batch_notes = truncate_text(distilled_notes, FINAL_REPORT_DISTILLED_NOTES_CHAR_LIMIT)
        analyses_compressed = len(distilled_notes) > FINAL_REPORT_DISTILLED_NOTES_CHAR_LIMIT
        evidence_label = "分批提炼后的关键证据"
    stats = final_report_evidence_stats(images, analyses)
    item_notes, items_compressed = render_learning_items_for_prompt(learning_items or [])
    mistake_notes, mistakes_compressed = render_mistake_items_for_prompt(mistake_items or [])
    if item_notes:
        process_note = build_report_process_note(session, learning_items or [], mistake_items or [], stats)
        batch_notes = (
            f"【学生目标与动态策略】\n{build_strategy_context(session_strategy(session))}\n\n"
            f"【错题本候选】\n{mistake_notes}\n\n"
            f"【结构化学习条目（已按题目/板块/知识点去重）】\n{item_notes}\n\n"
            f"【报告生成依据】\n{process_note}\n\n"
            f"【{evidence_label}】\n{batch_notes}"
        )
    compressed_notice = ""
    if timeline_compressed or analyses_compressed or items_compressed or mistakes_compressed or distilled_notes is not None:
        compressed_notice = (
            "注意：后端已按模型上下文预算压缩时间线或批次分析，并删除重复描述；"
            "未列出的细节不要编造，只基于可见证据概述。\n\n"
        )
    prompt = prompts.render_prompt(
        "final_report",
        evidence_label=evidence_label,
        compressed_notice=compressed_notice,
        start=start.isoformat() if start else "未知",
        end=end.isoformat() if end else session.get("finished_at") or "未知",
        total_duration=format_duration(total_seconds),
        image_count=len(images),
        analysis_count=stats["analysis_count"],
        raw_analysis_chars=stats["raw_analysis_chars"],
        raw_capture_meta_chars=stats["raw_capture_meta_chars"],
        unique_done_analysis_count=stats["unique_done_analysis_count"],
        duplicate_done_analysis_count=stats["duplicate_done_analysis_count"],
        timeline=timeline,
        batch_notes=batch_notes,
    )
    return truncate_text(prompt, FINAL_REPORT_PROMPT_CHAR_LIMIT)


def build_qa_session_summary_prompt(session: dict, qa_events: list[dict]) -> str:
    qa_lines = []
    for index, event in enumerate(reversed(qa_events), start=1):
        context = event.get("context") if isinstance(event.get("context"), dict) else {}
        qa_lines.append(
            (
                f"{index}. 时间={event.get('created_at')}; "
                f"触发={event.get('trigger_type') or event.get('source')}; "
                f"意图={context.get('student_intent') or 'unknown'}\n"
                f"问：{truncate_text(event.get('question'), 260)}\n"
                f"答：{truncate_text(event.get('answer'), 520)}"
            )
        )
    return truncate_text(
        f"""
请把这个没有观察图片或以实时语音问答为主的学习回合，压缩成一段可查看的回合总结。

要求：
- 用中文，结构清晰，适合 App 历史对话里的“查看报告”。
- 不要编造题目、图片内容或耗时；只能基于问答文本和会话策略。
- 重点保留：用户问了什么、模型怎么答、当前结论/卡点、下一步可以继续问什么。
- 如果信息不足，明确写“证据不足”。

会话：
- id={session.get('id')}
- title={session.get('title')}
- student_goal={session.get('student_goal') or ''}

动态策略：
{build_strategy_context(session_strategy(session))}

问答记录：
{chr(10).join(qa_lines) if qa_lines else "暂无问答记录。"}

请输出这些栏目：
回合摘要：
用户主要问题：
已给出的帮助：
可能的卡点：
下一步建议：
报告生成依据：
""".strip(),
        FINAL_REPORT_PROMPT_CHAR_LIMIT,
    )


async def wait_for_batch_analyses(session_id: str) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + FINAL_REPORT_WAIT_SECONDS
    while True:
        with connect() as conn:
            running = conn.execute(
                "SELECT COUNT(*) AS count FROM analyses WHERE session_id=? AND scope='batch' AND status='running'",
                (session_id,),
            ).fetchone()["count"]
        if running == 0 or loop.time() >= deadline:
            return
        await asyncio.sleep(1)


async def run_analysis(
    analysis_id: str,
    session_id: str,
    batch_id: str | None,
    prompt: str,
    filenames: list[str],
    summarize: bool = False,
    task_id: str | None = None,
    already_claimed: bool = False,
) -> tuple[str, str]:
    if not already_claimed:
        mark_task_run(task_id, "running")
    storage_settings = get_settings()
    settings = effective_llm_settings_for_session(session_id)
    image_paths = [storage_settings.data_dir / "images" / name for name in filenames]
    emit_log(f"开始大模型解析：{len(image_paths)} 张图片", session_id=session_id)
    try:
        content = await run_with_llm_gate(
            f"vision:{analysis_id[:8]} images={len(image_paths)}",
            session_id,
            lambda: llm.analyze_images(settings, prompt, image_paths),
            priority=LLM_PRIORITY_BACKGROUND,
        )
        status = "done"
        if analysis_needs_clarity_warning(content):
            emit_log(
                (
                    "图片拍摄提醒：课本/试卷/屏幕可能未完整进入拍摄区域，"
                    "或题干、数字、页码看不清。"
                    "请调整相机角度/距离，确保材料完整清晰可见后重新拍摄并解析。"
                ),
                session_id=session_id,
                level="warning",
            )
    except Exception as exc:
        content = f"大模型解析失败：{llm.format_llm_error(exc)}"
        status = "failed"
        emit_log(content, session_id=session_id, level="error")
    now = utc_now()
    with connect() as conn:
        conn.execute(
            "UPDATE analyses SET status=?, content=?, updated_at=? WHERE id=?",
            (status, content, now, analysis_id),
        )
        if summarize:
            conn.execute(
                "UPDATE sessions SET status=?, updated_at=? WHERE id=?",
                ("analyzed" if status == "done" else "error", now, session_id),
            )
        else:
            conn.execute("UPDATE sessions SET status=?, updated_at=? WHERE id=?", (status, now, session_id))
    if status == "done":
        counts = store_learning_items_from_analysis(
            session_id, batch_id, analysis_id, content, filenames,
            detection_method="observation" if summarize else "photo_grading",
        )
        # 逐题批改（全题对错权威）：模型已把“批改JSON：[...]”写进 analyses.content（已落库），
        # 端上可直接从分析内容解析读取；这里再解析一次仅用于校验与可观测日志，最小面、不加列、
        # 不触发迁移，且与上面的错题抽取链路互不影响（解析失败返回 []）。
        try:
            grading = parse_batch_grading(content)
        except Exception:
            grading = []
        if grading:
            emit_log(
                f"逐题批改已解析：{len(grading)} 题（全题对错，含做对/空白；已随 analyses.content 落库供端上读取）",
                session_id=session_id,
            )
            record_report_event(
                session_id,
                "batch_grading",
                "逐题批改结果",
                f"batch={batch_id or 'single'}；题数={len(grading)}",
                analysis_id,
            )
        inferred = infer_needs_from_analysis(content)
        if counts["mistake_item_count"]:
            inferred = merge_tags(inferred, ["mistake_book"])
        if counts["learning_item_count"]:
            inferred = merge_tags(inferred, ["knowledge_points"])
        if inferred:
            update_session_needs(session_id, inferred, "根据最新画面分析动态调整关注点：" + "、".join(tag_label(tag) for tag in inferred))
        if counts["learning_item_count"] or counts["mistake_item_count"]:
            emit_log(
                f"已写入学习资产：结构化条目 {counts['learning_item_count']} 条，错题本候选 {counts['mistake_item_count']} 条",
                session_id=session_id,
            )
            record_report_event(
                session_id,
                "batch_assets",
                "批次学习资产入库",
                f"batch={batch_id or 'single'}；结构化条目={counts['learning_item_count']}；错题本候选={counts['mistake_item_count']}；动态需求={','.join(inferred)}",
                analysis_id,
            )
        try:
            indexed_count = await index_knowledge_items(collect_unindexed_knowledge_rows())
            if indexed_count:
                emit_log(f"语义知识库已自动索引 {indexed_count} 条新条目", session_id=session_id)
        except Exception:
            pass
    if not already_claimed:
        mark_task_run(task_id, "done" if status == "done" else "failed", error=content if status != "done" else "")
    emit_log(f"解析完成：{status}", session_id=session_id)
    return status, content


async def distill_final_report_evidence(settings, session: dict, images: list[dict], analyses: list[dict], session_id: str) -> str | None:
    if not should_distill_final_evidence(images, analyses):
        return None
    stats = final_report_evidence_stats(images, analyses)
    notes, source_stats = build_distill_source_notes(analyses)
    chunks = chunk_text_blocks(notes, FINAL_REPORT_DISTILL_CHUNK_CHAR_LIMIT)
    emit_log(
        (
            "最终报告证据过大，先分批提炼："
            f"images={stats['image_count']} analyses={stats['analysis_count']} "
            f"raw_analysis_chars={stats['raw_analysis_chars']} raw_capture_meta_chars={stats['raw_capture_meta_chars']} "
            f"unique_done={stats['unique_done_analysis_count']} duplicates={stats['duplicate_done_analysis_count']} "
            f"source_chars={source_stats.get('source_chars', 0)} chunks={len(chunks)}"
        ),
        session_id=session_id,
    )
    distilled_chunks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        prompt = build_distill_prompt(session, images, chunk, index, len(chunks), stats)
        emit_log(f"开始提炼最终报告证据：第 {index}/{len(chunks)} 批，prompt_chars={len(prompt)}", session_id=session_id)
        try:
            distilled = await run_with_llm_gate(
                f"distill:{session_id[:8]} chunk={index}/{len(chunks)}",
                session_id,
                lambda: llm.analyze_text(settings, prompt, max_tokens=FINAL_REPORT_DISTILL_MAX_TOKENS),
                priority=LLM_PRIORITY_BACKGROUND,
            )
        except Exception as exc:
            emit_log(
                f"最终报告证据提炼失败，回退到直接压缩报告路径：{llm.format_llm_error(exc)}",
                session_id=session_id,
                level="warning",
            )
            return None
        distilled_chunks.append(f"【提炼批次 {index}/{len(chunks)}】\n{distilled.strip()}")
    return "\n\n".join(distilled_chunks)


async def run_qa_session_summary(
    analysis_id: str,
    session_id: str,
    task_id: str | None = None,
    already_claimed: bool = False,
) -> tuple[str, str]:
    if not already_claimed:
        mark_task_run(task_id, "running")
    with connect() as conn:
        session_row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not session_row:
        content = "回合总结失败：session not found"
        if not already_claimed:
            mark_task_run(task_id, "failed", error=content)
        return "failed", content
    session = dict(session_row)
    events = [event for event in qa_events_for_session(session_id, 80) if event.get("question") or event.get("answer")]
    prompt = build_qa_session_summary_prompt(session, events)
    settings = effective_llm_settings_for_session(session_id)
    try:
        content = await run_with_llm_gate(
            f"qa_session_summary:{session_id[:8]}",
            session_id,
            lambda: llm.analyze_text(settings, prompt, max_tokens=1200),
            priority=LLM_PRIORITY_BACKGROUND,
        )
        content = truncate_text(content, QA_ANSWER_CHAR_LIMIT)
        status = "done"
    except Exception as exc:
        status = "failed"
        content = f"回合总结失败：{llm.format_llm_error(exc)}"
    now = utc_now()
    with connect() as conn:
        conn.execute(
            "UPDATE analyses SET status=?, prompt=?, content=?, updated_at=? WHERE id=?",
            (status, prompt, content, now, analysis_id),
        )
        conn.execute(
            "UPDATE sessions SET summary=?, status=?, report_generated_at=?, updated_at=? WHERE id=?",
            (content if status == "done" else "", "completed" if status == "done" else "error", now if status == "done" else "", now, session_id),
        )
    if not already_claimed:
        mark_task_run(task_id, "done" if status == "done" else "failed", error=content if status != "done" else "")
    emit_log(f"QA 回合总结完成：{status}", session_id=session_id, level="info" if status == "done" else "error")
    return status, content


async def run_final_report(analysis_id: str, session_id: str, task_id: str | None = None, already_claimed: bool = False) -> tuple[str, str]:
    if not already_claimed:
        mark_task_run(task_id, "running")
    await wait_for_batch_analyses(session_id)
    settings = effective_llm_settings_for_session(session_id)
    with connect() as conn:
        session_row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not session_row:
            content = f"最终报告生成失败：session not found: {session_id}"
            if not already_claimed:
                mark_task_run(task_id, "failed", error=content)
            return "failed", content
        session = dict(session_row)
        images = [
            dict(row)
            for row in conn.execute(
                """
                SELECT images.*, COALESCE(obs.novelty_status, 'unknown') AS novelty_status
                FROM images
                LEFT JOIN session_observations obs ON obs.image_id = images.id
                WHERE images.session_id=?
                ORDER BY sequence_index, captured_at, created_at
                """,
                (session_id,),
            )
        ]
        images = [row for row in images if image_row_has_valid_content(row)]
        analyses = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM analyses WHERE session_id=? ORDER BY created_at",
                (session_id,),
            )
        ]
        learning_items = [
            dict(row)
            for row in conn.execute(
                """
                SELECT *
                FROM learning_items
                WHERE session_id=?
                ORDER BY first_sequence_index, first_seen_at, created_at
                """,
                (session_id,),
            )
        ]
        mistake_items = mistake_items_for_session(session_id)
    try:
        distilled_notes = await distill_final_report_evidence(settings, session, images, analyses, session_id)
        prompt = build_final_report_prompt(session, images, analyses, distilled_notes, learning_items, mistake_items)
        stats = final_report_evidence_stats(images, analyses)
        process_note = build_report_process_note(session, learning_items, mistake_items, stats)
        record_report_event(session_id, "final_prompt_context", "最终报告生成依据", process_note, analysis_id)
        emit_log(
            (
                "开始生成学习回合最终总结报告："
                f"prompt_chars={len(prompt)} images={stats['image_count']} analyses={stats['analysis_count']} "
                f"raw_evidence_chars={stats['raw_evidence_chars']} distilled={'yes' if distilled_notes else 'no'}"
            ),
            session_id=session_id,
        )
        content = await run_with_llm_gate(
            f"final_report:{session_id[:8]}",
            session_id,
            lambda: llm.analyze_text(settings, prompt),
            priority=LLM_PRIORITY_BACKGROUND,
        )
        status = "done"
    except Exception as exc:
        content = f"最终报告生成失败：{llm.format_llm_error(exc)}"
        status = "failed"
        emit_log(content, session_id=session_id, level="error")
    now = utc_now()
    with connect() as conn:
        conn.execute(
            "UPDATE analyses SET status=?, content=?, updated_at=? WHERE id=?",
            (status, content, now, analysis_id),
        )
        conn.execute(
            "UPDATE sessions SET summary=?, status=?, report_generated_at=?, updated_at=? WHERE id=?",
            (content, "completed" if status == "done" else "error", now if status == "done" else "", now, session_id),
        )
    if not already_claimed:
        mark_task_run(task_id, "done" if status == "done" else "failed", error=content if status != "done" else "")
    emit_log(f"最终总结报告生成完成：{status}", session_id=session_id)
    return status, content


@app.get("/health")
def health() -> dict:
    return {"ok": True, "service": "pxj"}


@app.get("/health/llm")
async def llm_health(request: Request) -> dict:
    principal = principal_from_request(request)
    settings = effective_llm_settings(principal["account_id"], principal.get("user_id", ""))
    try:
        result = await llm.check_health(settings)
        result["usage"] = llm_usage_snapshot(principal["account_id"])
        result["active_config"] = active_model_config(principal["account_id"], principal.get("user_id", ""))
        return result
    except Exception as exc:
        return {
            "ok": False,
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
            "error": llm.format_llm_error(exc),
            "usage": llm_usage_snapshot(principal["account_id"]),
            "active_config": active_model_config(principal["account_id"], principal.get("user_id", "")),
        }


@app.get("/api/auth/config")
def get_auth_config() -> dict:
    return auth_public_config()


@app.post("/api/auth/register")
async def register_user(request: Request) -> dict:
    init_db()
    settings = get_settings()
    if not settings.registration_enabled:
        raise HTTPException(403, "registration disabled")
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(422, "invalid JSON body")
    email = normalize_email(body.get("email"))
    password = str(body.get("password") or "")
    display_name = clean_auth_text(body.get("display_name") or body.get("displayName") or body.get("name"), 120)
    account_name = clean_auth_text(body.get("account_name") or body.get("accountName"), 120) or (display_name or email.split("@")[0])
    student_name = clean_auth_text(body.get("student_name") or body.get("studentName"), 120) or "默认学生"
    if not email or not AUTH_EMAIL_RE.match(email):
        raise HTTPException(422, "valid email is required")
    if len(password) < AUTH_PASSWORD_MIN_LENGTH:
        raise HTTPException(422, f"password must be at least {AUTH_PASSWORD_MIN_LENGTH} characters")
    now = utc_now()
    account_id = uuid.uuid4().hex
    user_id = uuid.uuid4().hex
    member_id = uuid.uuid4().hex
    parent_profile_id = uuid.uuid4().hex
    student_profile_id = uuid.uuid4().hex
    try:
        with connect_control() as conn:
            existing = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
            if existing:
                raise HTTPException(409, "email already registered")
            conn.execute(
                "INSERT INTO accounts(id, name, status, plan, created_at, updated_at) VALUES(?, ?, 'active', 'free', ?, ?)",
                (account_id, account_name, now, now),
            )
            conn.execute(
                """
                INSERT INTO users(
                    id, account_id, email, display_name, password_hash, role, status,
                    created_at, updated_at, last_login_at
                )
                VALUES(?, ?, ?, ?, ?, 'owner', 'active', ?, ?, ?)
                """,
                (user_id, account_id, email, display_name or email, hash_password(password), now, now, now),
            )
            conn.execute(
                """
                INSERT INTO account_members(id, account_id, user_id, role, status, created_at, updated_at)
                VALUES(?, ?, ?, 'owner', 'active', ?, ?)
                """,
                (member_id, account_id, user_id, now, now),
            )
            conn.execute(
                """
                INSERT INTO identity_profiles(
                    id, account_id, user_id, profile_type, display_name, relation, metadata,
                    status, created_at, updated_at
                )
                VALUES(?, ?, ?, 'parent', ?, 'owner', '{}', 'active', ?, ?)
                """,
                (parent_profile_id, account_id, user_id, display_name or "家长", now, now),
            )
            conn.execute(
                """
                INSERT INTO identity_profiles(
                    id, account_id, user_id, profile_type, display_name, student_id, relation, metadata,
                    status, created_at, updated_at
                )
                VALUES(?, ?, ?, 'student', ?, ?, '', '{}', 'active', ?, ?)
                """,
                (student_profile_id, account_id, user_id, student_name, student_profile_id, now, now),
            )
            user = dict(conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone())
    except HTTPException:
        raise
    set_current_account(account_id)
    ensure_account_db(account_id)
    token = make_access_token(user)
    return {
        "access_token": token,
        "token_type": AUTH_SCHEME.lower(),
        "user": public_user(user),
        "account": {"id": account_id, "name": account_name, "plan": "free", "status": "active"},
        "profiles": account_profiles(account_id),
        "auth": auth_public_config(),
    }


@app.post("/api/auth/login")
async def login_user(request: Request) -> dict:
    init_db()
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(422, "invalid JSON body")
    email = normalize_email(body.get("email"))
    password = str(body.get("password") or "")
    with connect_control() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=? AND status='active'", (email,)).fetchone()
        if not row or not verify_password(password, row["password_hash"]):
            raise HTTPException(401, "invalid email or password")
        now = utc_now()
        conn.execute("UPDATE users SET last_login_at=?, updated_at=? WHERE id=?", (now, now, row["id"]))
        user = dict(conn.execute("SELECT * FROM users WHERE id=?", (row["id"],)).fetchone())
    set_current_account(user["account_id"])
    ensure_account_db(user["account_id"])
    return {
        "access_token": make_access_token(user),
        "token_type": AUTH_SCHEME.lower(),
        "user": public_user(user),
        "profiles": account_profiles(user["account_id"]),
        "auth": auth_public_config(),
    }


@app.get("/api/auth/me")
def get_current_user(request: Request) -> dict:
    principal = principal_from_request(request)
    if not principal.get("authenticated"):
        return {"authenticated": False, "auth": auth_public_config()}
    return {
        "authenticated": True,
        "user": principal["user"],
        "profiles": account_profiles(principal["account_id"]),
        "active_model_config": active_model_config(principal["account_id"], principal["user_id"]),
        "auth": auth_public_config(),
    }


@app.get("/api/profiles")
def list_profiles(request: Request) -> dict:
    principal = principal_from_request(request)
    return {"profiles": account_profiles(principal["account_id"])}


@app.post("/api/profiles")
async def post_profile(request: Request) -> dict:
    principal = principal_from_request(request, required=True)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(422, "invalid JSON body")
    profile = create_identity_profile(principal["account_id"], body, user_id=principal["user_id"] if body.get("attach_to_user") else "")
    return {"profile": profile, "profiles": account_profiles(principal["account_id"])}


@app.patch("/api/profiles/{profile_id}")
async def update_profile(profile_id: str, request: Request) -> dict:
    principal = principal_from_request(request, required=True)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(422, "invalid JSON body")
    fields: list[str] = []
    params: list[object] = []
    if any(k in body for k in ("display_name", "displayName", "name")):
        fields.append("display_name=?")
        params.append(clean_auth_text(body.get("display_name") or body.get("displayName") or body.get("name"), 120))
    if "relation" in body:
        fields.append("relation=?")
        params.append(clean_auth_text(body.get("relation"), 80))
    if "student_id" in body or "studentId" in body:
        fields.append("student_id=?")
        params.append(clean_auth_text(body.get("student_id") or body.get("studentId"), 80))
    if isinstance(body.get("metadata"), dict):
        fields.append("metadata=?")
        params.append(json_dumps(body["metadata"]))
    if "status" in body:
        status_value = clean_auth_text(body.get("status"), 20)
        if status_value in ("active", "inactive"):
            fields.append("status=?")
            params.append(status_value)
    if not fields:
        raise HTTPException(422, "no updatable fields provided")
    fields.append("updated_at=?")
    params.append(utc_now())
    params.extend([profile_id, principal["account_id"]])
    with connect_control() as conn:
        cursor = conn.execute(
            f"UPDATE identity_profiles SET {', '.join(fields)} WHERE id=? AND account_id=?",
            params,
        )
        if cursor.rowcount != 1:
            raise HTTPException(404, "profile not found")
        row = conn.execute("SELECT * FROM identity_profiles WHERE id=?", (profile_id,)).fetchone()
    return {"profile": dict(row), "profiles": account_profiles(principal["account_id"])}


@app.delete("/api/profiles/{profile_id}")
def delete_profile(profile_id: str, request: Request) -> dict:
    principal = principal_from_request(request, required=True)
    now = utc_now()
    with connect_control() as conn:
        profile = conn.execute(
            "SELECT * FROM identity_profiles WHERE id=? AND account_id=? AND status='active'",
            (profile_id, principal["account_id"]),
        ).fetchone()
        if not profile:
            raise HTTPException(404, "profile not found")
        if profile["profile_type"] == "student":
            remaining = conn.execute(
                "SELECT COUNT(*) AS c FROM identity_profiles WHERE account_id=? AND profile_type='student' AND status='active' AND id!=?",
                (principal["account_id"], profile_id),
            ).fetchone()["c"]
            if remaining == 0:
                raise HTTPException(422, "cannot remove the last student profile")
        conn.execute(
            "UPDATE identity_profiles SET status='inactive', updated_at=? WHERE id=? AND account_id=?",
            (now, profile_id, principal["account_id"]),
        )
    return {"ok": True, "profiles": account_profiles(principal["account_id"])}


async def index_knowledge_items(items: list[dict]) -> int:
    """Embed + upsert knowledge texts into the active account's knowledge_vectors table."""
    if not embeddings.embed_enabled():
        return 0
    items = [it for it in items if (it.get("text") or "").strip()]
    if not items:
        return 0
    try:
        vectors = await embeddings.embed_texts([it["text"][:1000] for it in items])
    except Exception:
        return 0
    if len(vectors) != len(items):
        return 0
    now = utc_now()
    with connect() as conn:
        for it, vec in zip(items, vectors):
            conn.execute(
                """
                INSERT INTO knowledge_vectors(id, kind, ref_id, student_profile_id, text, embedding, updated_at)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(kind, ref_id) DO UPDATE SET
                    text=excluded.text, embedding=excluded.embedding,
                    student_profile_id=excluded.student_profile_id, updated_at=excluded.updated_at
                """,
                (uuid.uuid4().hex, it["kind"], it["ref_id"], it.get("student_profile_id", ""),
                 it["text"][:2000], json_dumps(vec), now),
            )
    return len(items)


def collect_account_knowledge_rows() -> list[dict]:
    rows: list[dict] = []
    with connect() as conn:
        for r in conn.execute(
            "SELECT id, title, question_text, knowledge_points, subject, error_reason FROM mistake_items WHERE status NOT IN ('deleted')"
        ):
            d = dict(r)
            text = " ".join(str(d.get(k) or "") for k in ("subject", "title", "question_text", "knowledge_points", "error_reason")).strip()
            if text:
                rows.append({"kind": "mistake", "ref_id": d["id"], "student_profile_id": "", "text": text})
        for r in conn.execute("SELECT id, item_type, title, content, subject FROM learning_items"):
            d = dict(r)
            text = " ".join(str(d.get(k) or "") for k in ("subject", "item_type", "title", "content")).strip()
            if text:
                rows.append({"kind": "learning", "ref_id": d["id"], "student_profile_id": "", "text": text})
    return rows


def collect_unindexed_knowledge_rows() -> list[dict]:
    rows = collect_account_knowledge_rows()
    if not rows:
        return []
    with connect() as conn:
        indexed = {(r["kind"], r["ref_id"]) for r in conn.execute("SELECT kind, ref_id FROM knowledge_vectors")}
    return [r for r in rows if (r["kind"], r["ref_id"]) not in indexed]


async def knowledge_semantic_search(query: str, k: int = 5, kinds: list[str] | None = None) -> list[dict]:
    if not embeddings.embed_enabled() or not query.strip():
        return []
    try:
        qvec = await embeddings.embed_text(query[:1000])
    except Exception:
        return []
    if not qvec:
        return []
    results: list[dict] = []
    with connect() as conn:
        sql = "SELECT kind, ref_id, student_profile_id, text, embedding FROM knowledge_vectors"
        params: list = []
        if kinds:
            sql += " WHERE kind IN (%s)" % ",".join("?" * len(kinds))
            params.extend(kinds)
        for r in conn.execute(sql, params):
            try:
                vec = json.loads(r["embedding"])
            except Exception:
                continue
            results.append({
                "kind": r["kind"], "ref_id": r["ref_id"], "student_profile_id": r["student_profile_id"],
                "text": r["text"], "score": round(embeddings.cosine(qvec, vec), 4),
            })
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[: max(1, min(k, 50))]


@app.post("/api/knowledge/reindex")
async def reindex_knowledge(request: Request) -> dict:
    principal_from_request(request, required=True)
    rows = collect_account_knowledge_rows()
    count = await index_knowledge_items(rows)
    return {"ok": True, "indexed": count, "embed_enabled": embeddings.embed_enabled()}


@app.get("/api/memory/agent")
async def list_agent_memories(request: Request, kind: str = "", limit: int = 200) -> dict:
    principal = principal_from_request(request, required=True)
    return {
        "memories": memory_store.list_memories(account_id=principal["account_id"], kind=kind, limit=limit),
        "stats": memory_store.stats(account_id=principal["account_id"]),
    }


@app.get("/api/memory/agent/search")
async def search_agent_memories(request: Request, q: str = "", k: int = 5, min_score: float = 0.0) -> dict:
    """Preview exactly which durable memories a question would pull into context,
    with the full per-memory score breakdown (semantic/recency/importance/usage)."""
    principal = principal_from_request(request, required=True)
    memories = await memory_store.retrieve(
        q,
        account_id=principal["account_id"],
        k=max(1, min(k, 50)),
        min_score=min_score,
        mark_used=False,
    )
    return {"query": q, "memories": memories, "weights": {
        "semantic": memory_store.W_SEMANTIC,
        "recency": memory_store.W_RECENCY,
        "importance": memory_store.W_IMPORTANCE,
        "usage": memory_store.W_USAGE,
    }}


@app.get("/api/memory/deltas")
async def list_memory_deltas(
    request: Request, unseen_only: int = 1, since: str = "", limit: int = 50
) -> dict:
    """Phase 3: per-turn memory deltas for the in-chat "这次更了解你了" chip.
    Passive pull only — no polling, no server-side push."""
    principal = principal_from_request(request, required=True)
    deltas = memory_store.recent_deltas(
        account_id=principal["account_id"],
        unseen_only=bool(unseen_only),
        since=clean_user_text(since, 64),
        limit=max(1, min(int(limit or 50), 200)),
    )
    return {"deltas": deltas}


@app.post("/api/memory/deltas/seen")
async def mark_memory_deltas_seen(request: Request) -> dict:
    """Best-effort debounce (M8): mark deltas as seen so the chip is not re-shown."""
    principal = principal_from_request(request, required=True)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(422, "invalid JSON body")
    ids = body.get("ids")
    if not isinstance(ids, list):
        raise HTTPException(422, "ids must be a list")
    clean_ids = [str(i)[:64] for i in ids if i]
    updated = memory_store.mark_deltas_seen(clean_ids, account_id=principal["account_id"])
    return {"updated": updated}


@app.patch("/api/memory/agent/{memory_id}")
async def patch_agent_memory(memory_id: str, request: Request) -> dict:
    """Phase 3: correct (text) or soft-delete/restore (status) one durable memory.

    - text   -> re-embed in the same transaction (M5); 500 on embed failure (no
                half-written row), so the client can retry.
    - status -> active|superseded. Restoring to active runs restore_guard first and
                returns 409 {error: capacity|duplicate} on conflict (M6). Soft-delete
                only (status flip), never a hard delete.
    Cross-account isolation: the memory_id must belong to the caller's account or 404.
    """
    principal = principal_from_request(request, required=True)
    account_id = principal["account_id"]
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(422, "invalid JSON body")

    has_text = "text" in body and body.get("text") is not None
    has_status = "status" in body and body.get("status") is not None
    if not has_text and not has_status:
        raise HTTPException(422, "text or status is required")

    updated: dict | None = None
    if has_text:
        text = clean_user_text(body.get("text"), memory_store.MEMORY_TEXT_LIMIT)
        if not text:
            raise HTTPException(422, "text is empty")
        try:
            updated = await memory_store.update_text(memory_id, text, account_id=account_id)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(500, f"re-embed failed: {truncate_text(str(exc), 120)}")
        if updated is None:
            raise HTTPException(404, "memory not found")

    if has_status:
        status = clean_user_text(body.get("status"), 32).lower()
        if status not in ("active", "superseded"):
            raise HTTPException(422, "status must be active or superseded")
        if status == "active":
            guard = await memory_store.restore_guard(memory_id, account_id=account_id)
            if not guard.get("ok"):
                reason = guard.get("reason") or "conflict"
                if reason == "missing":
                    raise HTTPException(404, "memory not found")
                raise HTTPException(409, reason)
        result = memory_store.set_status(memory_id, status, account_id=account_id)
        if result is None:
            raise HTTPException(404, "memory not found")
        updated = result

    return {"memory": updated}


@app.post("/api/knowledge/search")
async def search_knowledge(request: Request) -> dict:
    principal_from_request(request, required=True)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(422, "invalid JSON body")
    query = clean_user_text(body.get("query") or body.get("q"), 500)
    if not query:
        raise HTTPException(422, "query is required")
    k = int(body.get("k") or 5)
    kinds = body.get("kinds") if isinstance(body.get("kinds"), list) else None
    results = await knowledge_semantic_search(query, k=k, kinds=kinds)
    return {"query": query, "results": results, "embed_enabled": embeddings.embed_enabled()}


@app.get("/api/model-platforms")
def model_platforms() -> dict:
    return {
        "platforms": [
            {"provider": provider, **details}
            for provider, details in MODEL_PROVIDERS.items()
        ],
        "recommended_gateway": {
            "name": "new-api",
            "purpose": "统一 OpenAI/Anthropic/Gemini/智谱等平台到 OpenAI-compatible API；OpenAI 兼容的平台可直接在上面填 Base URL+Key，Anthropic 等非兼容平台经网关接入。",
            "endpoint": get_settings().llm_gateway_url or "",
            "deploy_target": "ydz@100.64.0.13",
        },
        "system_default": active_model_config(),
    }


@app.get("/api/model-configs")
def list_model_configs(request: Request) -> dict:
    principal = principal_from_request(request)
    with connect_control() as conn:
        rows = [
            model_config_public(row)
            for row in conn.execute(
                "SELECT * FROM model_configs WHERE account_id=? ORDER BY is_default DESC, updated_at DESC",
                (principal["account_id"],),
            )
        ]
    return {"configs": rows, "active": active_model_config(principal["account_id"], principal.get("user_id", ""))}


@app.post("/api/model-configs")
async def upsert_model_config(request: Request) -> dict:
    principal = principal_from_request(request, required=True)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(422, "invalid JSON body")
    provider = clean_auth_text(body.get("provider"), 40)
    if provider not in MODEL_PROVIDERS:
        raise HTTPException(422, "unsupported model provider")
    name = clean_auth_text(body.get("name"), 120) or MODEL_PROVIDERS[provider]["label"]
    base_url = clean_auth_text(body.get("base_url") or body.get("baseUrl"), 500)
    model = clean_auth_text(body.get("model"), 160)
    if not base_url or not model:
        raise HTTPException(422, "base_url and model are required")
    config_id = clean_auth_text(body.get("id"), 80) or uuid.uuid4().hex
    api_key = body.get("api_key", body.get("apiKey"))
    enabled = 0 if body.get("enabled") is False else 1
    is_default = 1 if body.get("is_default") is True or body.get("isDefault") is True else 0
    max_concurrency = max(1, min(64, int(body.get("max_concurrency") or body.get("maxConcurrency") or 1)))
    min_interval_seconds = max(0.0, float(body.get("min_interval_seconds") or body.get("minIntervalSeconds") or 0))
    metadata = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    now = utc_now()
    with connect_control() as conn:
        existing = conn.execute(
            "SELECT * FROM model_configs WHERE id=? AND account_id=?",
            (config_id, principal["account_id"]),
        ).fetchone()
        encrypted_key = encrypt_model_secret(api_key) if api_key not in (None, "") else (existing["api_key_encrypted"] if existing else "")
        if is_default:
            conn.execute("UPDATE model_configs SET is_default=0 WHERE account_id=?", (principal["account_id"],))
        conn.execute(
            """
            INSERT INTO model_configs(
                id, account_id, owner_user_id, provider, name, base_url, api_key_encrypted,
                model, enabled, is_default, max_concurrency, min_interval_seconds,
                metadata, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                provider=excluded.provider,
                name=excluded.name,
                base_url=excluded.base_url,
                api_key_encrypted=excluded.api_key_encrypted,
                model=excluded.model,
                enabled=excluded.enabled,
                is_default=excluded.is_default,
                max_concurrency=excluded.max_concurrency,
                min_interval_seconds=excluded.min_interval_seconds,
                metadata=excluded.metadata,
                updated_at=excluded.updated_at
            """,
            (
                config_id,
                principal["account_id"],
                principal["user_id"],
                provider,
                name,
                base_url,
                encrypted_key,
                model,
                enabled,
                is_default,
                max_concurrency,
                min_interval_seconds,
                json_dumps(metadata),
                now,
                now,
            ),
        )
        if not is_default:
            default_count = conn.execute(
                "SELECT COUNT(*) AS count FROM model_configs WHERE account_id=? AND enabled=1 AND is_default=1",
                (principal["account_id"],),
            ).fetchone()["count"]
            if not default_count:
                conn.execute("UPDATE model_configs SET is_default=1 WHERE id=? AND account_id=?", (config_id, principal["account_id"]))
        row = conn.execute("SELECT * FROM model_configs WHERE id=? AND account_id=?", (config_id, principal["account_id"])).fetchone()
    return {"config": model_config_public(row), "active": active_model_config(principal["account_id"], principal["user_id"])}


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return (Path(__file__).parent / "static" / "dashboard.html").read_text(encoding="utf-8")


@app.get("/prompts", response_class=HTMLResponse)
def prompt_settings_page() -> str:
    return (Path(__file__).parent / "static" / "prompts.html").read_text(encoding="utf-8")


@app.get("/assets", response_class=HTMLResponse)
def asset_browser_page() -> str:
    return (Path(__file__).parent / "static" / "assets.html").read_text(encoding="utf-8")


@app.get("/session-panel", response_class=HTMLResponse)
def session_panel_page() -> str:
    return (Path(__file__).parent / "static" / "session-panel.html").read_text(encoding="utf-8")


@app.get("/api/prompts")
def list_prompts(request: Request) -> dict:
    init_db()
    # Scope to the caller's account (principal_from_request sets the account context);
    # enforces login when auth_required is on. Prompts are per-account isolated.
    principal_from_request(request)
    return {"prompts": prompts.list_prompt_records()}


@app.get("/api/assets")
def list_assets(
    request: Request,
    kind: str = Query("learning", pattern="^(learning|mistake)$"),
    session_id: str = Query(""),
    item_type: str = Query(""),
    status: str = Query(""),
    review_state: str = Query(""),
    error_type: str = Query(""),
    subject: str = Query(""),
    location: str = Query(""),
    q: str = Query(""),
    page: int = Query(1, ge=1),
    page_size: int = Query(ASSET_PAGE_SIZE_DEFAULT, ge=1, le=ASSET_PAGE_SIZE_MAX),
) -> dict:
    init_db()
    principal = principal_from_request(request)
    filters = {
        "kind": kind,
        "session_id": session_id,
        "item_type": item_type,
        "status": status,
        "review_state": review_state,
        "error_type": error_type,
        "subject": subject,
        "location": location,
        "q": q,
    }
    if kind == "mistake":
        result = browse_mistake_assets(
            session_id=session_id,
            status=status,
            review_state=review_state,
            error_type=error_type,
            subject=subject,
            location=location,
            q=q,
            page=page,
            page_size=page_size,
            account_id=principal["account_id"],
        )
    else:
        result = browse_learning_assets(session_id=session_id, item_type=item_type, subject=subject, location=location, q=q, page=page, page_size=page_size, account_id=principal["account_id"])
    result["filters"] = filters
    return result


@app.get("/api/learning-columns")
def learning_columns_overview(request: Request, page_size: int = Query(40, ge=1, le=ASSET_PAGE_SIZE_MAX)) -> dict:
    init_db()
    principal = principal_from_request(request)
    learning_items, learning_total, mistake_items, mistake_total = global_learning_column_items(page_size, account_id=principal["account_id"])
    with connect() as conn:
        sessions = [
            dict(row)
            for row in conn.execute(
                """
                SELECT id, title, status, created_at, updated_at, student_goal,
                       assistant_focus, inferred_needs, report_style,
                       (SELECT COUNT(*) FROM learning_items WHERE learning_items.session_id = sessions.id) AS learning_count,
                       (SELECT COUNT(*) FROM mistake_items WHERE mistake_items.session_id = sessions.id) AS mistake_count
                FROM sessions
                WHERE account_id=? AND (student_goal != '' OR assistant_focus != '' OR inferred_needs != '[]' OR report_style != '')
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 60
                """,
                (principal["account_id"],),
            )
        ]
    return {
        "scope": "global",
        "learning_items": [compact_global_asset_item(item) for item in learning_items],
        "learning_total": learning_total,
        "mistake_items": [compact_global_asset_item(item) for item in mistake_items],
        "mistake_total": mistake_total,
        "strategy_sessions": sessions,
        "strategy_total": len(sessions),
    }


def compact_global_asset_item(item: dict) -> dict:
    compact = dict(item)
    for key in ("document_body", "source_image_ids"):
        compact.pop(key, None)
    if len(str(compact.get("content") or "")) > 260:
        compact["content"] = truncate_text(compact.get("content"), 260)
    if len(str(compact.get("question_text") or "")) > 260:
        compact["question_text"] = truncate_text(compact.get("question_text"), 260)
    if len(str(compact.get("error_reason") or "")) > 220:
        compact["error_reason"] = truncate_text(compact.get("error_reason"), 220)
    if len(str(compact.get("evidence") or "")) > 220:
        compact["evidence"] = truncate_text(compact.get("evidence"), 220)
    details = compact.get("source_image_details")
    if isinstance(details, list):
        compact["source_image_details"] = details[:2]
    return compact


@app.post("/api/sessions/{session_id}/mistakes")
async def post_session_mistake(session_id: str, request: Request) -> dict:
    principal = principal_from_request(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(422, "invalid JSON body")
    with connect() as conn:
        require_account_session(conn, session_id, principal)
    mistake = create_manual_mistake_item(session_id, body)
    emit_log(
        f"加入错题本：{truncate_text(mistake.get('title') or mistake.get('question_text'), 120)}",
        session_id=session_id,
        source="mistake",
    )
    return {"mistake": mistake}


@app.patch("/api/mistakes/{mistake_id}")
async def patch_mistake(mistake_id: str, request: Request) -> dict:
    init_db()
    principal = principal_from_request(request)
    with connect() as conn:
        row = conn.execute(
            """
            SELECT mi.id
            FROM mistake_items mi
            LEFT JOIN sessions ON sessions.id = mi.session_id
            WHERE mi.id=? AND sessions.account_id=?
            """,
            (mistake_id, principal["account_id"]),
        ).fetchone()
    if not row:
        raise HTTPException(404, "mistake not found")
    body = await request.json()
    mistake = update_mistake_item(mistake_id, body if isinstance(body, dict) else {})
    emit_log(
        f"更新错题状态：{mistake_id} status={mistake.get('status')} review_state={mistake.get('review_state')}",
        session_id=mistake.get("session_id"),
        source="dashboard",
    )
    return {"mistake": mistake}


@app.get("/api/mistake-candidates")
def get_mistake_candidates(
    request: Request,
    subject: str = Query(""),
    q: str = Query(""),
    page_size: int = Query(ASSET_PAGE_SIZE_DEFAULT, ge=1, le=ASSET_PAGE_SIZE_MAX),
) -> dict:
    """列出待学生确认的错题候选（观察异步提取）。"""
    init_db()
    principal = principal_from_request(request)
    return mistake_candidate_items(account_id=principal["account_id"], subject=subject, q=q, page_size=page_size)


@app.post("/api/mistakes/{mistake_id}/import")
async def import_mistake_candidate(mistake_id: str, request: Request) -> dict:
    """学生确认候选→导入正式错题本（status=confirmed，进复习队列）。可带 review_note/correction 等内联订正。"""
    init_db()
    principal = principal_from_request(request)
    status = owned_mistake_status(mistake_id, principal["account_id"])
    if status is None:
        raise HTTPException(404, "mistake not found")
    body = await _optional_json_body(request)
    updates: dict = {"status": "confirmed", "review_state": "queued"}
    for key in ("review_note", "correction", "next_action", "error_type"):
        if key in body:
            updates[key] = body[key]
    mistake = update_mistake_item(mistake_id, updates)
    emit_log(
        f"导入错题候选→错题本：{mistake_id}",
        session_id=mistake.get("session_id"),
        source="dashboard",
    )
    return {"mistake": mistake}


@app.post("/api/mistakes/{mistake_id}/dismiss")
async def dismiss_mistake_candidate(mistake_id: str, request: Request) -> dict:
    """学生忽略候选（status=ignored，不进错题本）。"""
    init_db()
    principal = principal_from_request(request)
    status = owned_mistake_status(mistake_id, principal["account_id"])
    if status is None:
        raise HTTPException(404, "mistake not found")
    mistake = update_mistake_item(mistake_id, {"status": "ignored"})
    emit_log(
        f"忽略错题候选：{mistake_id}",
        session_id=mistake.get("session_id"),
        source="dashboard",
    )
    return {"mistake": mistake}


@app.get("/api/mistakes/{mistake_id}/review-events")
def get_mistake_review_events(request: Request, mistake_id: str, limit: int = Query(60, ge=1, le=200)) -> dict:
    init_db()
    principal = principal_from_request(request)
    with connect() as conn:
        row = conn.execute(
            """
            SELECT mi.id
            FROM mistake_items mi
            LEFT JOIN sessions ON sessions.id = mi.session_id
            WHERE mi.id=? AND sessions.account_id=?
            """,
            (mistake_id, principal["account_id"]),
        ).fetchone()
    if not row:
        raise HTTPException(404, "mistake not found")
    return list_review_events_for_mistake(mistake_id, limit)


@app.post("/api/mistakes/{mistake_id}/review-events")
async def post_mistake_review_event(mistake_id: str, request: Request) -> dict:
    init_db()
    principal = principal_from_request(request)
    with connect() as conn:
        row = conn.execute(
            """
            SELECT mi.id
            FROM mistake_items mi
            LEFT JOIN sessions ON sessions.id = mi.session_id
            WHERE mi.id=? AND sessions.account_id=?
            """,
            (mistake_id, principal["account_id"]),
        ).fetchone()
    if not row:
        raise HTTPException(404, "mistake not found")
    body = await request.json()
    result = create_review_event(mistake_id, body if isinstance(body, dict) else {})
    event = result["event"]
    mistake = result["mistake"]
    emit_log(
        f"记录错题复习事件：{mistake_id} result={event.get('result')} review_count={mistake.get('review_count')}",
        session_id=mistake.get("session_id"),
        source=event.get("source") or "dashboard",
    )
    return result


@app.get("/api/review-queue")
def review_queue(
    request: Request,
    status: str = Query(""),
    subject: str = Query(""),
    page_ref: str = Query(""),
    question_ref: str = Query(""),
    item_type: str = Query(""),
    error_type: str = Query(""),
    error_reason: str = Query(""),
    q: str = Query(""),
    due_only: bool = Query(True),
    page_size: int = Query(ASSET_PAGE_SIZE_DEFAULT, ge=1, le=ASSET_PAGE_SIZE_MAX),
) -> dict:
    init_db()
    principal = principal_from_request(request)
    return review_queue_items(
        account_id=principal["account_id"],
        status=status,
        subject=subject,
        page_ref=page_ref,
        question_ref=question_ref,
        item_type=item_type,
        error_type=error_type,
        error_reason=error_reason,
        q=q,
        due_only=due_only,
        page_size=page_size,
    )


@app.get("/api/student-profile")
def get_student_profile(request: Request) -> dict:
    init_db()
    principal = principal_from_request(request)
    return {"profile": student_profile(account_id=principal["account_id"])}


@app.get("/api/observability")
def get_observability(request: Request) -> dict:
    init_db()
    principal = principal_from_request(request)
    return observability_snapshot(account_id=principal["account_id"], user_id=principal.get("user_id", ""))


@app.get("/api/tasks")
async def list_account_tasks(request: Request) -> dict:
    """列出本账号正在跑/排队的后台生成任务（账号隔离）。仅返回 background 通道：
    可视化/报告/记忆整理等耗时任务；实时问答是用户当前请求本身，不在此列。"""
    init_db()
    principal = principal_from_request(request)
    account_id = principal.get("account_id") or DEFAULT_ACCOUNT_ID
    async with llm_gate_lock:
        snapshot = [
            {k: v for k, v in rec.items() if k != "future"}
            for rec in llm_tasks.values()
            if rec.get("account_id") == account_id and rec.get("lane") != "realtime"
        ]
    now = asyncio.get_running_loop().time()
    tasks = []
    for rec in snapshot:
        if rec.get("cancel_requested"):
            continue
        created = float(rec.get("created", now))
        tasks.append({
            "id": rec["id"],
            "label": rec.get("label", ""),
            "title": task_display_title(rec.get("label", "")),
            "lane": rec.get("lane", "background"),
            "state": rec.get("state", "waiting"),
            "session_id": rec.get("session_id", ""),
            "age_seconds": max(0, int(now - created)),
            "cancelable": True,
        })
    tasks.sort(key=lambda t: t["age_seconds"], reverse=True)
    return {"tasks": tasks, "count": len(tasks)}


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_account_task(task_id: str, request: Request) -> dict:
    """取消本账号自己的后台任务（账号隔离：他人任务一律 404，不泄露存在性）。"""
    init_db()
    principal = principal_from_request(request)
    account_id = principal.get("account_id") or DEFAULT_ACCOUNT_ID
    fut = None
    label = ""
    sess = None
    async with llm_gate_lock:
        rec = llm_tasks.get(task_id)
        if rec is None or rec.get("account_id") != account_id:
            raise HTTPException(404, "task not found")
        if rec.get("cancel_requested"):
            return {"ok": True, "already": True}
        rec["cancel_requested"] = True
        rec["state"] = "cancelling"
        fut = rec.get("future")
        label = rec.get("label", "")
        sess = rec.get("session_id") or None
    if fut is not None and not fut.done():
        fut.cancel()
    emit_llm_gate_log(f"用户取消后台任务：{label}（{task_id[:8]}）", session_id=sess, level="warning")
    return {"ok": True}


@app.put("/api/prompts/{prompt_key}")
async def update_prompt(prompt_key: str, request: Request) -> dict:
    init_db()
    principal = principal_from_request(request)
    body = await request.json()
    try:
        record = prompts.set_prompt(prompt_key, str(body.get("content", "")))
    except KeyError:
        raise HTTPException(404, "prompt not found")
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    emit_log(f"更新提示词：{prompt_key}（账号 {principal['account_id']}）", source="dashboard")
    return {"prompt": record}


@app.post("/api/prompts/{prompt_key}/reset")
def reset_prompt(prompt_key: str, request: Request) -> dict:
    init_db()
    principal = principal_from_request(request)
    try:
        record = prompts.reset_prompt(prompt_key)
    except KeyError:
        raise HTTPException(404, "prompt not found")
    emit_log(f"恢复默认提示词：{prompt_key}（账号 {principal['account_id']}）", source="dashboard")
    return {"prompt": record}


@app.post("/api/prompts/reset")
def reset_all_prompts(request: Request) -> dict:
    init_db()
    principal = principal_from_request(request)
    records = prompts.reset_all_prompts()
    emit_log(f"恢复全部默认提示词（账号 {principal['account_id']}）", source="dashboard")
    return {"prompts": records}


@app.get("/api/images/{filename}/thumbnail")
def image_thumbnail(filename: str) -> FileResponse:
    image_path = image_path_for_request(filename)
    thumb_path = thumbnail_path_for(image_path.name)
    if not thumb_path.is_file() or thumb_path.stat().st_mtime < image_path.stat().st_mtime:
        try:
            create_thumbnail(image_path, thumb_path)
        except Exception as exc:
            raise HTTPException(422, f"could not create thumbnail: {exc}") from exc
    return FileResponse(
        thumb_path,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@app.get("/api/sessions")
def list_sessions(request: Request, include_summary: bool = False) -> dict:
    principal = principal_from_request(request)
    summary_expr = "summary" if include_summary else "'' AS summary"
    with connect() as conn:
        sessions = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT
                    id, device_id, mode, title, status, created_at, updated_at,
                    finished_at, report_generated_at, student_goal, assistant_focus, inferred_needs, report_style, {summary_expr},
                    substr(summary, 1, 240) AS summary_preview,
                    (
                        SELECT question FROM qa_events
                        WHERE qa_events.session_id = sessions.id
                          AND TRIM(COALESCE(question, '')) != ''
                        ORDER BY created_at ASC
                        LIMIT 1
                    ) AS first_question,
                    (SELECT COUNT(*) FROM images WHERE images.session_id = sessions.id) AS image_count,
                    (SELECT COUNT(*) FROM analyses WHERE analyses.session_id = sessions.id) AS analysis_count,
                    (SELECT COUNT(*) FROM qa_events WHERE qa_events.session_id = sessions.id) AS qa_count,
                    (SELECT COUNT(*) FROM mistake_items WHERE mistake_items.session_id = sessions.id) AS mistake_count
                FROM sessions
                WHERE account_id=?
                ORDER BY created_at DESC
                LIMIT 100
                """,
                (principal["account_id"],),
            )
        ]
    return {"sessions": sessions}


@app.get("/api/memory")
def get_memory(request: Request, limit: int = Query(80, ge=1, le=200)) -> dict:
    init_db()
    principal = principal_from_request(request)
    task_id = schedule_memory_consolidation_if_due(account_id=principal["account_id"])
    return {
        "profile": memory_profile(account_id=principal["account_id"]),
        "events": memory_events(limit, account_id=principal["account_id"]),
        "consolidation": {
            "interval_seconds": MEMORY_CONSOLIDATION_INTERVAL_SECONDS,
            "scheduled_task_id": task_id or "",
        },
    }


@app.post("/api/memory/consolidate")
async def consolidate_memory(request: Request, background_tasks: BackgroundTasks) -> dict:
    init_db()
    principal = principal_from_request(request)
    task_id = schedule_memory_consolidation_if_due(force=True, account_id=principal["account_id"])
    if task_id:
        background_tasks.add_task(execute_next_task_run_now)
    return {"status": "queued" if task_id else "skipped", "task_id": task_id or ""}


@app.post("/api/sessions/{session_id}/memory")
async def create_session_memory(session_id: str, request: Request, background_tasks: BackgroundTasks) -> dict:
    init_db()
    principal = principal_from_request(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(422, "invalid JSON body")
    with connect() as conn:
        session = require_account_session(conn, session_id, principal)
    text = clean_user_text(body.get("text") or body.get("memory") or body.get("summary"), MEMORY_EVENT_TEXT_LIMIT)
    if len(text) < 2:
        raise HTTPException(422, "text is required")
    event = record_memory_event(
        session_id=session_id,
        account_id=principal["account_id"],
        qa_event_id=clean_user_text(body.get("qa_event_id"), 80),
        source=clean_user_text(body.get("source"), 80) or "ios-manual",
        message_type=clean_user_text(body.get("message_type"), 80) or "formed_memory",
        text=text,
        payload=body.get("payload") if isinstance(body.get("payload"), dict) else {},
    )
    task_id = schedule_memory_consolidation_if_due(force=True, account_id=principal["account_id"])
    if task_id:
        background_tasks.add_task(execute_next_task_run_now)
    emit_log(
        f"形成记忆：{truncate_text(text, 120)}",
        session_id=session_id,
        source="memory",
    )
    return {
        "event": event,
        "profile": memory_profile(account_id=principal["account_id"]),
        "consolidation": {
            "status": "queued" if task_id else "skipped",
            "task_id": task_id or "",
        },
    }


@app.get("/visualizations/{filename}")
def get_teaching_visualization_file(filename: str) -> FileResponse:
    init_db()
    path = visualization_file_path(filename)
    return FileResponse(
        path,
        media_type="text/html; charset=utf-8",
        headers={
            "Content-Security-Policy": TEACHING_VISUALIZATION_CSP,
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "private, max-age=60",
        },
    )


@app.post("/api/visualizations")
async def create_teaching_visualization(request: Request) -> dict:
    init_db()
    principal = principal_from_request(request)  # binds the per-account DB context for this request
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(422, "invalid JSON body")
    source_type = clean_user_text(body.get("source_type") or body.get("sourceType"), 40)
    source_id = clean_user_text(body.get("source_id") or body.get("sourceId"), 120)
    session_id = clean_user_text(body.get("session_id") or body.get("sessionId"), 120)
    text = clean_user_text(body.get("text") or body.get("source_text") or body.get("sourceText"), TEACHING_VISUALIZATION_SOURCE_CHAR_LIMIT)
    title = clean_user_text(body.get("title"), 160)
    extra_instruction = clean_user_text(body.get("extra_instruction") or body.get("instruction") or body.get("extraInstruction"), 1200)
    force_raw = body.get("force", False)
    force = force_raw is True or str(force_raw).strip().lower() in {"1", "true", "yes", "on"}
    if source_type not in TEACHING_VISUALIZATION_SOURCE_TYPES:
        raise HTTPException(422, "source_type must be qa_event, analysis, or custom")
    if source_type == "custom":
        if not text:
            raise HTTPException(422, "text is required for custom visualization")
        session_id = ensure_visualization_session(session_id, title)
        if not source_id:
            source_id = "custom_" + hashlib.sha256((session_id + "\n" + text).encode("utf-8")).hexdigest()[:32]
    if not source_id:
        raise HTTPException(422, "source_id is required")
    account_id = principal["account_id"]
    # If it's already generated and ready, return immediately.
    existing = latest_visualization_for_source(source_type, source_id)
    if existing and existing.get("status") == "ready" and existing.get("can_open") and not force:
        return {"visualization": existing}

    def _pending_payload() -> dict:
        pending = dict(existing) if existing else {
            "id": "",
            "source_type": source_type,
            "source_id": source_id,
            "session_id": session_id,
        }
        pending.update({
            "status": "running",
            "queued": True,
            "message": "已加入空闲生成队列：可视化会在语音/问答空闲时自动生成，完成后可在该回复下点「打开可视化」查看。",
        })
        return pending

    # 去重：同一条回答的可视化只跑一份。客户端会轮询本接口（每隔几秒 POST 一次），
    # 若不去重，每次轮询都会再起一个后台任务，导致「生成可视化讲解」在后台堆叠 5~6 份
    # （还会挤占模型，拖慢实时问答）。这里在本进程内记录在跑的 (账号, 源)；同时把最近落库
    # 为 running 的也视为在跑（覆盖进程重启/落库已开始的情况）。命中则直接返回排队态，不再起任务。
    viz_key = (account_id, source_type, source_id)
    if not force:
        already_running = viz_key in _VIZ_INFLIGHT
        if not already_running and existing and existing.get("status") == "running":
            already_running = _iso_within_seconds(existing.get("updated_at"), VIZ_INFLIGHT_STALE_SECONDS)
        if already_running:
            return {"visualization": _pending_payload()}

    # Otherwise generate asynchronously at BACKGROUND priority so it never competes
    # with realtime voice/QA — it runs when the model is idle. Return a "running"
    # status the client already understands and polls (session/QA overview).
    _VIZ_INFLIGHT.add(viz_key)

    async def _generate_in_background() -> None:
        set_current_account(account_id)
        ensure_account_db(account_id)
        try:
            await generate_teaching_visualization(
                source_type=source_type,
                source_id=source_id,
                session_id=session_id,
                source_text=text,
                title=title,
                force=force,
                extra_instruction=extra_instruction,
            )
        except Exception as exc:
            emit_log(f"可视化生成失败：{exc}", session_id=session_id or None, source="visualization", level="error")
        finally:
            _VIZ_INFLIGHT.discard(viz_key)

    asyncio.create_task(_generate_in_background())
    return {"visualization": _pending_payload()}


@app.get("/api/sessions/{session_id}/overview")
def get_session_overview(
    request: Request,
    session_id: str,
    analysis_limit: int = Query(DEFAULT_SESSION_ANALYSIS_LIMIT, ge=1, le=MAX_SESSION_ANALYSIS_LIMIT),
    analysis_offset: int = Query(0, ge=0),
) -> dict:
    principal = principal_from_request(request)
    with connect() as conn:
        session = require_account_session(conn, session_id, principal)

        image_count = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM images
            LEFT JOIN session_observations obs ON obs.image_id = images.id
            WHERE images.session_id=? AND COALESCE(obs.novelty_status, 'unknown') != 'invalid'
            """,
            (session_id,),
        ).fetchone()["count"]
        analysis_count = conn.execute(
            "SELECT COUNT(*) AS count FROM analyses WHERE session_id=? AND scope != 'final'",
            (session_id,),
        ).fetchone()["count"]
        qa_count = conn.execute(
            "SELECT COUNT(*) AS count FROM qa_events WHERE session_id=?",
            (session_id,),
        ).fetchone()["count"]
        analyses = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT {analysis_public_columns()}
                FROM analyses
                WHERE session_id=? AND scope != 'final'
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (session_id, analysis_limit, analysis_offset),
            )
        ]
        analyses = attach_visualization_metadata(analyses, "analysis", text_keys=("content",))
        analyses = sanitize_analyses_for_display(analyses)
        final_analysis = conn.execute(
            f"""
            SELECT {analysis_public_columns()}
            FROM analyses
            WHERE session_id=? AND scope='final'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()

        batch_ids = [row["batch_id"] for row in analyses if row.get("batch_id")]
        include_single_images = any(row["scope"] == "single" for row in analyses)
        image_params: list = [session_id]
        image_filters: list[str] = []
        if batch_ids:
            placeholders = ", ".join("?" for _ in batch_ids)
            image_filters.append(f"images.batch_id IN ({placeholders})")
            image_params.extend(batch_ids)
        if include_single_images:
            image_filters.append("(images.batch_id IS NULL AND images.kind='single')")

        if image_filters:
            images = [
                dict(row)
                for row in conn.execute(
                    f"""
                    SELECT {image_public_select()}
                    FROM images
                    LEFT JOIN session_observations obs ON obs.image_id = images.id
                    WHERE images.session_id=? AND ({' OR '.join(image_filters)})
                    ORDER BY images.sequence_index, images.captured_at, images.created_at
                    """,
                    image_params,
                )
            ]
        else:
            images = [
                dict(row)
                for row in conn.execute(
                    f"""
                    SELECT {image_public_select()}
                    FROM images
                    LEFT JOIN session_observations obs ON obs.image_id = images.id
                    WHERE images.session_id=?
                    ORDER BY images.sequence_index, images.captured_at, images.created_at
                    LIMIT ?
                    """,
                    (session_id, SESSION_OVERVIEW_IMAGE_FALLBACK_LIMIT),
                )
            ]
        learning_items = learning_items_for_session(session_id)
        mistake_items = mistake_items_for_session(session_id)
        report_events = report_events_for_session(session_id, 40)
        qa_events = qa_events_for_session(session_id, 60)

    return {
        "session": dict(session),
        "images": images,
        "analyses": analyses,
        "final_analysis": dict(final_analysis) if final_analysis else None,
        "learning_items": learning_items,
        "mistake_items": mistake_items,
        "report_events": report_events,
        "qa_events": qa_events,
        "image_count": image_count,
        "analysis_count": analysis_count,
        "qa_count": qa_count,
        "analysis_limit": analysis_limit,
        "analysis_offset": analysis_offset,
        "has_more_analyses": analysis_offset + len(analyses) < analysis_count,
    }


@app.get("/api/sessions/{session_id}")
def get_session(request: Request, session_id: str) -> dict:
    principal = principal_from_request(request)
    return session_payload(session_id, principal)


def session_payload(session_id: str, principal: dict | None = None) -> dict:
    principal = principal or default_principal()
    with connect() as conn:
        session = require_account_session(conn, session_id, principal)
        images = [
            dict(row)
            for row in conn.execute(
                """
                SELECT images.*, COALESCE(obs.novelty_status, 'unknown') AS novelty_status,
                       COALESCE(obs.signal_summary, '') AS signal_summary
                FROM images
                LEFT JOIN session_observations obs ON obs.image_id = images.id
                WHERE images.session_id=?
                ORDER BY images.created_at
                """,
                (session_id,),
            )
        ]
        analyses = [dict(row) for row in conn.execute("SELECT * FROM analyses WHERE session_id=? ORDER BY created_at", (session_id,))]
        analyses = attach_visualization_metadata(analyses, "analysis", text_keys=("content",))
        analyses = sanitize_analyses_for_display(analyses)
    return {
        "session": dict(session),
        "images": images,
        "analyses": analyses,
        "learning_items": learning_items_for_session(session_id),
        "mistake_items": mistake_items_for_session(session_id),
        "report_events": report_events_for_session(session_id, 40),
        "qa_events": qa_events_for_session(session_id, 80),
    }


@app.get("/api/device-control")
def get_device_control(request: Request, session_id: str = "", device_id: str = "", limit: int = Query(8, ge=1, le=40)) -> dict:
    init_db()
    principal = principal_from_request(request)
    require_control_token(request)
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    with connect() as conn:
        expire_old_control_commands(conn, now)
        filters: list[str] = ["account_id=?"]
        params: list[object] = [principal["account_id"]]
        if session_id:
            filters.append("session_id=?")
            params.append(session_id)
        if device_id:
            filters.append("device_id=?")
            params.append(device_id)
        where_sql = f"WHERE {' AND '.join(filters)}" if filters else ""
        devices = [
            device_state_row_to_dict(dict(row), now_dt)
            for row in conn.execute(
                f"""
                SELECT *
                FROM device_states
                {where_sql}
                ORDER BY last_seen_at DESC
                LIMIT ?
                """,
                [*params, limit],
            )
        ]
        if (not devices or not any(device.get("online") for device in devices)) and session_id and not device_id:
            fallback = latest_device_state(conn, account_id=principal["account_id"], online_only=True)
            if fallback:
                fallback_item = device_state_row_to_dict(fallback, now_dt)
                devices = [fallback_item, *[device for device in devices if device.get("device_id") != fallback_item.get("device_id")]]
        command_filters: list[str] = ["account_id=?"]
        command_params: list[object] = [principal["account_id"]]
        if session_id:
            command_filters.append("session_id=?")
            command_params.append(session_id)
        if device_id:
            command_filters.append("device_id=?")
            command_params.append(device_id)
        command_where = f"WHERE {' AND '.join(command_filters)}" if command_filters else ""
        commands = [
            control_command_row_to_dict(dict(row))
            for row in conn.execute(
                f"""
                SELECT *
                FROM control_commands
                {command_where}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                [*command_params, limit],
            )
        ]
    latest = devices[0] if devices else None
    return {
        "devices": devices,
        "latest_device": latest,
        "recent_commands": commands,
        "poll_interval_seconds": DEVICE_CONTROL_POLL_INTERVAL_SECONDS,
        "online_after_seconds": DEVICE_CONTROL_ONLINE_SECONDS,
        "command_ttl_seconds": CONTROL_COMMAND_TTL_SECONDS,
    }


@app.post("/api/device-control/poll")
async def poll_device_control(request: Request) -> dict:
    init_db()
    principal = principal_from_request(request)
    body = await request.json()
    require_control_token(request, body)
    device_id = clean_user_text(body.get("device_id"), 160)
    if not device_id:
        raise HTTPException(422, "device_id is required")
    state = body.get("state")
    state_payload = compact_device_state(state if isinstance(state, dict) else {})
    session_id = clean_user_text(body.get("session_id") or state_payload.get("session_id"), 160)
    source = clean_user_text(body.get("source"), 80) or "ios"
    now = utc_now()
    commands: list[dict] = []
    with connect() as conn:
        expire_old_control_commands(conn, now)
        conn.execute(
            """
            INSERT INTO device_states(device_id, account_id, session_id, source, state, last_seen_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(device_id) DO UPDATE SET
                account_id=excluded.account_id,
                session_id=excluded.session_id,
                source=excluded.source,
                state=excluded.state,
                last_seen_at=excluded.last_seen_at,
                updated_at=excluded.updated_at
            """,
            (device_id, principal["account_id"], session_id, source, json_dumps(state_payload), now, now),
        )
        commands = [
            control_command_row_to_dict(row)
            for row in select_commands_for_device(conn, device_id=device_id, session_id=session_id, limit=6, now=now, account_id=principal["account_id"])
        ]
    return {
        "commands": commands,
        "poll_interval_seconds": DEVICE_CONTROL_POLL_INTERVAL_SECONDS,
        "server_time": now,
    }


@app.post("/api/control-commands")
async def create_control_command(request: Request) -> dict:
    init_db()
    principal = principal_from_request(request)
    body = await request.json()
    require_control_token(request, body)
    command_type = clean_user_text(body.get("command_type") or body.get("type"), 80)
    if command_type not in CONTROL_COMMAND_TYPES:
        raise HTTPException(422, "unsupported command_type")
    payload = body.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    payload = {str(key): (truncate_text(value, 2000) if isinstance(value, str) else value) for key, value in payload.items()}
    session_id = clean_user_text(body.get("session_id"), 160)
    device_id = clean_user_text(body.get("device_id"), 160)
    source = clean_user_text(body.get("source"), 80) or "dashboard"
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    expires_at = (now_dt + timedelta(seconds=CONTROL_COMMAND_TTL_SECONDS)).isoformat()
    with connect() as conn:
        expire_old_control_commands(conn, now)
        if not device_id:
            row = latest_device_state(conn, account_id=principal["account_id"], session_id=session_id, online_only=True) if session_id else None
            if not row:
                row = latest_device_state(conn, account_id=principal["account_id"], online_only=True)
            if not row and session_id:
                row = latest_device_state(conn, account_id=principal["account_id"], session_id=session_id)
            if not row:
                row = latest_device_state(conn, account_id=principal["account_id"])
            if row:
                device_id = str(row.get("device_id") or "")
        if not device_id and not session_id:
            raise HTTPException(409, "no iOS device has checked in yet")
        command_id = uuid.uuid4().hex
        conn.execute(
            """
            INSERT INTO control_commands(
                id, account_id, device_id, session_id, command_type, payload, status, source,
                created_at, expires_at
            )
            VALUES(?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
            """,
            (command_id, principal["account_id"], device_id, session_id, command_type, json_dumps(payload), source, now, expires_at),
        )
        row = conn.execute("SELECT * FROM control_commands WHERE id=?", (command_id,)).fetchone()
    emit_log(
        f"Dashboard command queued: {command_type} target_device={device_id or '*'} target_session={session_id or '*'}",
        session_id=session_id or None,
        device_id=device_id or None,
        source=source,
    )
    return {"command": control_command_row_to_dict(dict(row))}


@app.post("/api/control-commands/{command_id}/ack")
async def acknowledge_control_command(command_id: str, request: Request) -> dict:
    init_db()
    principal_from_request(request)  # binds the per-account DB context for this request
    body = await request.json()
    require_control_token(request, body)
    status = clean_user_text(body.get("status"), 40) or "applied"
    if status not in CONTROL_COMMAND_ACK_STATUSES:
        raise HTTPException(422, "unsupported ack status")
    error = clean_user_text(body.get("error"), 1000)
    device_id = clean_user_text(body.get("device_id"), 160)
    session_id = clean_user_text(body.get("session_id"), 160)
    state = body.get("state")
    state_payload = compact_device_state(state if isinstance(state, dict) else {})
    now = utc_now()
    with connect() as conn:
        if device_id:
            conn.execute(
                """
                INSERT INTO device_states(device_id, session_id, source, state, last_seen_at, updated_at)
                VALUES(?, ?, 'ios', ?, ?, ?)
                ON CONFLICT(device_id) DO UPDATE SET
                    session_id=CASE WHEN excluded.session_id='' THEN device_states.session_id ELSE excluded.session_id END,
                    state=CASE WHEN excluded.state='{}' THEN device_states.state ELSE excluded.state END,
                    last_seen_at=excluded.last_seen_at,
                    updated_at=excluded.updated_at
                """,
                (device_id, session_id, json_dumps(state_payload), now, now),
            )
        cursor = conn.execute(
            """
            UPDATE control_commands
            SET status=?, error=?, acknowledged_at=?
            WHERE id=?
            """,
            (status, error, now, command_id),
        )
        if cursor.rowcount != 1:
            raise HTTPException(404, "command not found")
        row = conn.execute("SELECT * FROM control_commands WHERE id=?", (command_id,)).fetchone()
    return {"command": control_command_row_to_dict(dict(row))}


async def extract_memories_after_qa(
    *,
    session_id: str,
    account_id: str,
    question: str,
    answer: str,
    feedback: str,
    source_event_id: str,
) -> None:
    """Background: distill durable memories from a finished QA turn.

    Runs on the background LLM lane so it never competes with realtime QA on the
    concurrency-1 model. Failures are swallowed (best-effort enrichment).
    """
    async def _call(prompt: str) -> str:
        return await run_with_llm_gate(
            f"memory_extract:{session_id[:8]}",
            session_id,
            lambda: llm.analyze_text(effective_llm_settings_for_session(session_id), prompt, max_tokens=400),
            priority=LLM_PRIORITY_BACKGROUND,
            account_id=account_id,
        )

    try:
        stored = await memory_store.extract_and_store(
            question=question,
            answer=answer,
            feedback=feedback,
            account_id=account_id,
            source_event_id=source_event_id,
            llm_call=_call,
        )
        if stored:
            adds = sum(1 for m in stored if m.get("op") == "add")
            updates = len(stored) - adds
            # Phase 3: record per-turn deltas (op/kind/text are authoritative, from
            # extract_and_store) so iOS can passively pull the "这次更了解你了" chip.
            # Still inside the background task — the QA response path is untouched.
            try:
                memory_store.write_deltas(
                    [
                        {
                            "qa_event_id": source_event_id,
                            "memory_id": m.get("id") or "",
                            "op": m.get("op") or "add",
                            "kind": m.get("kind") or "fact",
                            "text": m.get("text") or "",
                        }
                        for m in stored
                    ],
                    account_id=account_id,
                )
            except Exception:
                pass
            emit_log(
                f"memory extracted: +{adds} new, ~{updates} updated",
                session_id=session_id,
                source="memory",
            )
    except Exception as exc:
        emit_log(
            f"memory extraction failed: {truncate_text(str(exc), 160)}",
            session_id=session_id,
            source="memory",
            level="warning",
        )


@app.post("/api/sessions/{session_id}/qa")
async def ask_session_question(
    session_id: str,
    request: Request,
    question: str = Form(""),
    trigger_type: str = Form("voice"),
    source: str = Form("ios"),
    focus: str = Form("{}"),
    context: str = Form("{}"),
    gesture: str = Form("{}"),
    image: UploadFile | None = File(None),
) -> dict:
    init_db()
    principal = principal_from_request(request)
    focus_payload = parse_json_object(focus)
    context_payload = parse_json_object(context)
    gesture_payload = parse_json_object(gesture)
    transcript_fallback = ""
    for key in ("question", "transcript", "recognized_text", "recognizedText"):
        value = context_payload.get(key)
        if isinstance(value, str) and value.strip():
            transcript_fallback = value
            break
    cleaned_question = clean_user_text(question, QA_QUESTION_CHAR_LIMIT)
    if not cleaned_question:
        cleaned_question = clean_user_text(transcript_fallback, QA_QUESTION_CHAR_LIMIT)
    if not cleaned_question:
        raise HTTPException(422, "question is required")
    trigger = clean_user_text(trigger_type, 80) or "voice"
    source_text = clean_user_text(source, 80) or "ios"
    student_intent = infer_qa_student_intent(trigger, context_payload, cleaned_question)
    context_payload = dict(context_payload)
    context_payload["student_intent"] = student_intent
    context_payload["dialog_state"] = {
        "turn": qa_turn_index(context_payload),
        "is_followup": qa_is_followup_like(trigger, context_payload, cleaned_question),
        "requires_current_visual": student_intent in QA_VISUAL_REVIEW_INTENTS,
        "question_has_new_problem_reference": qa_question_has_new_problem_reference(cleaned_question),
    }
    with connect() as conn:
        session = require_account_session(conn, session_id, principal)
        max_sequence = conn.execute(
            "SELECT COALESCE(MAX(sequence_index), -1) AS max_sequence FROM images WHERE session_id=?",
            (session_id,),
        ).fetchone()["max_sequence"]
    image_id: str | None = None
    image_filename: str | None = None
    image_row: dict | None = None
    uploaded_image_id: str | None = None
    uploaded_image_filename: str | None = None
    uploaded_frame_quality: dict = {}
    rejected_image_id: str | None = None
    rejected_image_filename: str | None = None
    client_rejected_frame_quality = qa_rejected_client_frame_quality(context_payload)
    current_image_rejected = client_rejected_frame_quality is not None
    previous_qa_image_row = latest_session_image_for_qa(session_id)
    image_context_mode = "text_only"
    if image is not None and image.filename:
        capture_meta = {
            "source": source_text,
            "trigger_type": trigger,
            "student_intent": student_intent,
            "qa_focus": focus_payload,
            "qa_context": context_payload,
            "qa_gesture": gesture_payload,
        }
        qa_frame_quality_payload = context_payload.get("qa_frame_quality") or context_payload.get("qaFrameQuality")
        if isinstance(qa_frame_quality_payload, dict):
            capture_meta["qa_frame_quality"] = qa_frame_quality_payload
        image_id, image_filename, _ = await save_upload(
            image,
            session_id,
            "qa",
            batch_id="qa",
            captured_at=utc_now(),
            sequence_index=int(max_sequence or -1) + 1,
            capture_meta=capture_meta,
        )
        uploaded_image_id = image_id
        uploaded_image_filename = image_filename
        uploaded_frame_quality = qa_quality_with_relevance(
            qa_uploaded_frame_quality(capture_meta, image_filename),
            question=cleaned_question,
            trigger=trigger,
            context=context_payload,
            student_intent=student_intent,
            previous_image_row=previous_qa_image_row,
        )
        if qa_should_soft_accept_first_frame(
            uploaded_frame_quality,
            question=cleaned_question,
            trigger=trigger,
            context=context_payload,
            student_intent=student_intent,
        ):
            uploaded_frame_quality = {
                **uploaded_frame_quality,
                "eligible": True,
                "reason": "first_question_current_frame",
                "detail": (
                    uploaded_frame_quality.get("detail")
                    or "首问/新题已上传当前抓拍，后端允许进入视觉模型并由模型继续判断可见内容"
                ),
                "relevance": "accepted_first_or_new_problem_current_frame",
            }
        with connect() as conn:
            row = conn.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
            uploaded_row = dict(row) if row else None
        skip_duplicate_vision = (
            uploaded_frame_quality.get("eligible") is True
            and get_settings().qa_skip_duplicate_frame_vision
            and student_intent not in QA_VISUAL_REVIEW_INTENTS
            and not qa_is_first_or_new_problem_turn(trigger, context_payload, cleaned_question)
            and qa_frame_duplicates_previous(capture_meta, previous_qa_image_row)
        )
        if skip_duplicate_vision:
            current_image_rejected = False
            update_qa_image_context_verdict(uploaded_image_id, uploaded_frame_quality, accepted=False)
            image_id = None
            image_filename = None
            image_row = None
            image_context_mode = "duplicate_current_frame_text_only"
            emit_log(
                "QA 当前抓拍与上一张几乎一致，转纯文本快路径，跳过视觉模型",
                session_id=session_id,
                device_id=source_text,
                source="qa",
                level="info",
            )
        elif uploaded_frame_quality.get("eligible") is True:
            current_image_rejected = False
            update_qa_image_context_verdict(uploaded_image_id, uploaded_frame_quality, accepted=True)
            with connect() as conn:
                row = conn.execute("SELECT * FROM images WHERE id=?", (uploaded_image_id,)).fetchone()
                uploaded_row = dict(row) if row else uploaded_row
            image_row = uploaded_row
            image_context_mode = "current_frame"
        else:
            update_qa_image_context_verdict(uploaded_image_id, uploaded_frame_quality, accepted=False)
            rejected_image_id = uploaded_image_id
            rejected_image_filename = uploaded_image_filename
            current_image_rejected = True
            image_id = None
            image_filename = None
            image_row = latest_session_image_for_qa(session_id, exclude_image_id=rejected_image_id)
            if image_row:
                image_id = image_row.get("id")
                image_filename = image_row.get("filename")
                image_context_mode = "fallback_after_rejected_current_frame"
            else:
                image_context_mode = "rejected_current_frame"
            emit_log(
                (
                    "QA 当前抓拍未进入大模型上下文："
                    f"reason={uploaded_frame_quality.get('reason') or 'unknown'}；"
                    f"detail={uploaded_frame_quality.get('detail') or ''}"
                ),
                session_id=session_id,
                device_id=source_text,
                source="qa",
                level="warning",
            )
    if image_row is None and image_context_mode == "text_only":
        image_row = latest_session_image_for_qa(session_id)
        if image_row:
            image_id = image_row.get("id")
            image_filename = image_row.get("filename")
            image_context_mode = "ignored_current_frame_fallback" if current_image_rejected else "recent_session_frame"
        elif current_image_rejected:
            image_context_mode = "ignored_current_frame_text_only"
            uploaded_frame_quality = client_rejected_frame_quality or uploaded_frame_quality
    context_payload["image_context_mode"] = image_context_mode
    context_payload["used_image_context"] = bool(image_filename)
    context_payload["selected_image_id"] = image_id or ""
    context_payload["selected_image_filename"] = image_filename or ""
    context_payload["uploaded_image_id"] = uploaded_image_id or ""
    context_payload["uploaded_image_filename"] = uploaded_image_filename or ""
    context_payload["current_image_rejected"] = current_image_rejected
    context_payload["rejected_image_id"] = rejected_image_id or ""
    context_payload["rejected_image_filename"] = rejected_image_filename or ""
    event = insert_qa_event(
        session_id,
        image_id=image_id,
        source=source_text,
        trigger_type=trigger,
        question=cleaned_question,
        focus=focus_payload,
        context=context_payload,
        gesture=gesture_payload,
    )
    record_memory_event(
        session_id=session_id,
        account_id=principal["account_id"],
        qa_event_id=event.get("id") or "",
        source=source_text,
        message_type="typed_text" if trigger == "typed_chat" else "voice_to_text",
        text=cleaned_question,
        payload={"trigger_type": trigger, "student_intent": student_intent},
    )
    prompt_context_payload = qa_prompt_context(context_payload, current_image_rejected=current_image_rejected)
    try:
        if embeddings.embed_enabled():
            hits = await knowledge_semantic_search(cleaned_question, k=4)
            relevant = [
                {"kind": h["kind"], "score": h["score"], "text": h["text"][:300]}
                for h in hits if h.get("score", 0) >= 0.35
            ]
            if relevant:
                prompt_context_payload = dict(prompt_context_payload)
                prompt_context_payload["semantic_knowledge"] = relevant
    except Exception:
        pass
    # Memory gate (B-2): the client may turn long-term memory off for this turn
    # (context_inclusion.memory == false), or exclude individual memories by id
    # (memory_excludes). Off -> skip retrieval entirely (so nothing is mark_used);
    # otherwise retrieve but never mark_used the excluded ids. Old clients omit both
    # keys -> full retrieval, unchanged behaviour.
    inclusion = context_payload.get("context_inclusion")
    memory_enabled = True
    if isinstance(inclusion, dict) and inclusion.get("memory") is False:
        memory_enabled = False
    memory_excludes = context_payload.get("memory_excludes")
    exclude_ids = {str(x) for x in memory_excludes if isinstance(x, (str, int))} if isinstance(memory_excludes, list) else set()
    retrieved_memories: list = []
    if memory_enabled:
        try:
            retrieved_memories = await memory_store.retrieve_for_turn(
                cleaned_question,
                account_id=principal["account_id"],
                exclude_ids=exclude_ids,
            )
            if retrieved_memories:
                prompt_context_payload = dict(prompt_context_payload)
                prompt_context_payload["agent_memories"] = retrieved_memories
        except Exception:
            retrieved_memories = []
    # Read-only observability trace of this turn's assembled context (B-1). Computed
    # once from the final prompt context so both the success and failure returns can
    # surface it. Failure here must never break QA, so it is best-effort.
    try:
        context_trace = build_context_trace(
            prompt_context=prompt_context_payload,
            context_payload=context_payload,
            student_intent=student_intent,
            turn=qa_turn_index(context_payload),
            image_filename=image_filename or "",
            image_id=image_id or "",
            image_context_mode=image_context_mode,
            current_image_rejected=current_image_rejected,
            retrieved_memories=retrieved_memories,
            memory_gated_off=not memory_enabled,
        )
    except Exception:
        context_trace = {}
    prompt = build_qa_prompt(
        session,
        question=cleaned_question,
        trigger_type=trigger,
        focus=focus_payload,
        context=prompt_context_payload,
        gesture=gesture_payload,
        image_row=image_row,
        image_context_mode=image_context_mode,
    )
    settings = effective_llm_settings_for_session(session_id)
    try:
        image_paths = [image_path_for_request(image_filename)] if image_filename else []
        if image_paths:
            try:
                answer = await run_with_llm_gate(
                    f"qa_image:{session_id[:8]}",
                    session_id,
                    lambda: llm.analyze_images(settings, prompt, image_paths),
                    priority=LLM_PRIORITY_REALTIME,
                )
            except Exception as image_exc:
                if not current_image_rejected:
                    raise
                emit_log(
                    f"QA fallback image failed, retrying as text follow-up: {truncate_text(str(image_exc), 180)}",
                    session_id=session_id,
                    device_id=source_text,
                    source="qa",
                    level="warning",
                )
                answer = await run_with_llm_gate(
                    f"qa_fallback_text:{session_id[:8]}",
                    session_id,
                    lambda: llm.analyze_text(settings, prompt, max_tokens=1800),
                    priority=LLM_PRIORITY_REALTIME,
                )
        else:
            answer = await run_with_llm_gate(
                f"qa_text:{session_id[:8]}",
                session_id,
                lambda: llm.analyze_text(settings, prompt, max_tokens=1800),
                priority=LLM_PRIORITY_REALTIME,
            )
        if current_image_rejected and qa_answer_is_unhelpful_image_failure(answer):
            retry_context = dict(prompt_context_payload)
            retry_context["current_image_rejected"] = True
            retry_context["current_image_note"] = "The previous draft focused on the rejected image. Rewrite as a direct answer to the student's spoken follow-up using prior context only."
            retry_prompt = build_qa_prompt(
                session,
                question=cleaned_question,
                trigger_type=trigger,
                focus=focus_payload,
                context=retry_context,
                gesture=gesture_payload,
                image_row=image_row,
                image_context_mode=image_context_mode,
            )
            answer = await run_with_llm_gate(
                f"qa_retry_text:{session_id[:8]}",
                session_id,
                lambda: llm.analyze_text(settings, retry_prompt, max_tokens=1800),
                priority=LLM_PRIORITY_REALTIME,
            )
            if qa_answer_is_unhelpful_image_failure(answer):
                answer = qa_safe_followup_fallback_answer(cleaned_question)
        updated = update_qa_event(event["id"], status="done", answer=answer, tts_status="ready")
        update_session_needs(session_id, infer_need_tags_from_text(cleaned_question + " " + answer), focus_note=cleaned_question)
        memory_feedback = "；".join(
            part
            for part in (
                f"场景={context_payload.get('learning_mode_title') or ''}",
                f"回答方式={context_payload.get('coach_depth_title') or ''}",
                str(context_payload.get("coach_preference") or "").strip(),
            )
            if part and not part.endswith("=")
        )
        asyncio.create_task(
            extract_memories_after_qa(
                session_id=session_id,
                account_id=principal["account_id"],
                question=cleaned_question,
                answer=answer,
                feedback=memory_feedback,
                source_event_id=event.get("id") or "",
            )
        )
        emit_log(
            f"QA answered trigger={trigger} image_context={image_context_mode} question={truncate_text(cleaned_question, 120)}",
            session_id=session_id,
            device_id=source_text,
            source="qa",
        )
        return {
            "event": updated,
            "answer": updated.get("answer", ""),
            "image_id": image_id,
            "image_filename": image_filename,
            "used_image_context": bool(image_filename),
            "image_context_mode": image_context_mode,
            "uploaded_image_id": uploaded_image_id,
            "uploaded_image_filename": uploaded_image_filename,
            "current_image_rejected": current_image_rejected,
            "rejected_image_id": rejected_image_id,
            "rejected_image_filename": rejected_image_filename,
            "frame_quality": uploaded_frame_quality or client_rejected_frame_quality or {},
            "student_intent": student_intent,
            "dialog_state": context_payload.get("dialog_state", {}),
            "agent_memories": retrieved_memories,
            "context_trace": context_trace,
        }
    except Exception as exc:
        answer = f"AI 问答失败：{llm.format_llm_error(exc)}"
        updated = update_qa_event(event["id"], status="failed", answer=answer, tts_status="error")
        emit_log(answer, session_id=session_id, device_id=source_text, source="qa", level="error")
        return {
            "event": updated,
            "answer": answer,
            "image_id": image_id,
            "image_filename": image_filename,
            "used_image_context": bool(image_filename),
            "image_context_mode": image_context_mode,
            "uploaded_image_id": uploaded_image_id,
            "uploaded_image_filename": uploaded_image_filename,
            "current_image_rejected": current_image_rejected,
            "rejected_image_id": rejected_image_id,
            "rejected_image_filename": rejected_image_filename,
            "frame_quality": uploaded_frame_quality or client_rejected_frame_quality or {},
            "student_intent": student_intent,
            "dialog_state": context_payload.get("dialog_state", {}),
            "context_trace": context_trace,
        }


def _clamp_unit(value: object) -> float:
    """把任意输入收敛成归一化坐标分量：非数字/NaN → 0.0，越界 → clamp 到 [0,1]。"""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return 0.0
    if num != num:  # NaN
        return 0.0
    if num < 0.0:
        return 0.0
    if num > 1.0:
        return 1.0
    return num


def _segment_question_bbox(raw: object) -> dict:
    """把模型给的题块框 {x,y,w,h} 清洗成归一化[0,1]、原点左上（x 右 y 下、相对整张上传图）的
    干净 bbox；缺字段/非数字/越界一律 clamp，宽高不越过右/下边界。仿 _region_bbox 的清洗口径，
    但对缺失字段更宽容（补 0 而非整框丢弃），保证逐题框始终可用。"""
    box = raw if isinstance(raw, dict) else {}
    x = _clamp_unit(box.get("x"))
    y = _clamp_unit(box.get("y"))
    w = _clamp_unit(box.get("w"))
    h = _clamp_unit(box.get("h"))
    if x + w > 1.0:
        w = max(0.0, 1.0 - x)
    if y + h > 1.0:
        h = max(0.0, 1.0 - y)
    return {"x": x, "y": y, "w": w, "h": h}


def _iter_brace_blocks(text: str):
    """字符串感知地扫描出所有最外层 {...} 子串（花括号匹配，忽略字符串内的括号）。"""
    depth = 0
    start = -1
    in_str = False
    esc = False
    for index, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = index
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start : index + 1]
                start = -1


def _balance_segmentation_json(text: str) -> str:
    """补齐被 max_tokens 截断而未闭合的字符串/中括号/大括号；已闭合则原样返回。"""
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()
    if not in_str and not stack:
        return text
    repaired = text
    if in_str:
        repaired += '"'
    repaired = re.sub(r",\s*$", "", repaired.rstrip())
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def _repair_segmentation_json(text: str) -> str:
    """对视觉模型偶尔吐出的轻微非法 JSON 做容错修复（只在严格解析失败后使用）：
    1) 数值后多一个引号（如 "h": 0.5"）——只在「值位置」（紧跟 : [ , 后的纯数字）去掉这个引号，
       绝不碰 question_text 这种以数字+引号结尾的合法字符串；
    2) 去掉对象/数组里的尾逗号；
    3) 截断未闭合时补齐引号/中括号/大括号。"""
    repaired = re.sub(r'([:\[,]\s*-?\d+(?:\.\d+)?)"', r"\1", text)
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    repaired = _balance_segmentation_json(repaired)
    return repaired


_SEGMENTATION_TARGET_KEYS = ("questions", "is_study_material")


def _load_segmentation_dict(candidate: str, target_keys: tuple[str, ...] = _SEGMENTATION_TARGET_KEYS) -> dict | None:
    """把候选字符串严格解析成含目标键之一的对象；不是就返回 None。target_keys 可换成别的端点的键
    （如重排版 is_printed_question/plain_text），让同一套鲁棒解析复用到非分割响应上。"""
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(data, dict) and any(k in data for k in target_keys):
        return data
    return None


def _regex_segmentation_fallback(text: str) -> dict:
    """所有 JSON 修复都失败时的正则兜底：直接从文本里捞 is_study_material / material_type，
    并逐个 question 对象抽 index / bbox(x,y,w,h) / question_text / has_student_answer。"""
    result: dict = {}
    match = re.search(r'"is_study_material"\s*:\s*(true|false)', text, re.IGNORECASE)
    if match:
        result["is_study_material"] = match.group(1).lower() == "true"
    match = re.search(r'"material_type"\s*:\s*"([^"]*)"', text)
    if match:
        result["material_type"] = match.group(1)
    positions = [m.start() for m in re.finditer(r'"index"\s*:', text)]
    questions: list[dict] = []
    for order, pos in enumerate(positions):
        end = positions[order + 1] if order + 1 < len(positions) else len(text)
        chunk = text[pos:end]
        entry: dict = {}
        index_match = re.search(r'"index"\s*:\s*(\d+)', chunk)
        if index_match:
            entry["index"] = int(index_match.group(1))
        bbox: dict = {}
        for key in ("x", "y", "w", "h"):
            field = re.search(r'"' + key + r'"\s*:\s*(-?\d+(?:\.\d+)?)', chunk)
            if field:
                bbox[key] = float(field.group(1))
        if bbox:
            entry["bbox"] = bbox
        text_match = re.search(r'"question_text"\s*:\s*"((?:[^"\\]|\\.)*)"', chunk)
        if text_match:
            entry["question_text"] = text_match.group(1)
        answer_match = re.search(r'"has_student_answer"\s*:\s*(true|false)', chunk, re.IGNORECASE)
        if answer_match:
            entry["has_student_answer"] = answer_match.group(1).lower() == "true"
        questions.append(entry)
    if questions:
        result["questions"] = questions
    return result if ("questions" in result or "is_study_material" in result) else {}


def _extract_segmentation_object(content: str, target_keys: tuple[str, ...] = _SEGMENTATION_TARGET_KEYS) -> dict:
    """从模型输出里抽出 JSON 对象，容忍代码围栏/前后散文，并对视觉模型偶尔吐出的
    轻微非法 JSON（数值后多引号、尾逗号、截断未闭合）做容错修复后再解析；都失败时退到正则兜底。
    解析顺序：严格整体 → 容错修复整体 → 修复后逐个 {...} 块 → 原文逐个 {...} 块 → 正则兜底。
    target_keys 决定「合法对象」的判据；非分割端点（如重排版）传自己的键，并跳过分割专用的正则兜底。"""
    text = (content or "").strip()
    if not text:
        return {}
    parsed = _load_segmentation_dict(text, target_keys)
    if parsed is not None:
        return parsed
    repaired = _repair_segmentation_json(text)
    parsed = _load_segmentation_dict(repaired, target_keys)
    if parsed is not None:
        return parsed
    for candidate in _iter_brace_blocks(repaired):
        parsed = _load_segmentation_dict(candidate, target_keys)
        if parsed is not None:
            return parsed
        parsed = _load_segmentation_dict(_repair_segmentation_json(candidate), target_keys)
        if parsed is not None:
            return parsed
    for candidate in _iter_brace_blocks(text):
        parsed = _load_segmentation_dict(candidate, target_keys)
        if parsed is not None:
            return parsed
    # 正则兜底仅对分割响应有意义（抽 is_study_material/questions）；其它端点解析失败就返回 {}。
    if target_keys == _SEGMENTATION_TARGET_KEYS:
        return _regex_segmentation_fallback(text)
    return {}


def _build_segmentation_response(raw: str) -> dict:
    """把视觉模型的原始文本解析成稳定的题目分割契约；任何缺失/越界都被清洗成默认值，
    解析失败时退回 {is_study_material: False, material_type: "none", questions: []}。
    新增 low_quality（像题目但太糊/读不清，建议整图问答或重拍）与 bbox_approx（bbox 为近似定位，
    非精确，仅供排序/粗定位）两个布尔字段；两者均向后兼容。"""
    data = _extract_segmentation_object(raw)
    questions: list[dict] = []
    questions_raw = data.get("questions")
    dropped_empty = 0
    if isinstance(questions_raw, list):
        for position, entry in enumerate(questions_raw, start=1):
            if not isinstance(entry, dict):
                continue
            try:
                index = int(entry.get("index"))
            except (TypeError, ValueError):
                index = position
            question_text = truncate_text(entry.get("question_text") or "", 600)
            # 模糊兜底吐空框：question_text 去空白后为空的题没有任何辨题价值，过滤掉。
            if not question_text.strip():
                dropped_empty += 1
                continue
            questions.append(
                {
                    "index": index,
                    "bbox": _segment_question_bbox(entry.get("bbox")),
                    "question_text": question_text,
                    "has_student_answer": bool(entry.get("has_student_answer")),
                }
            )
    is_material = bool(data.get("is_study_material"))
    if questions:
        is_material = True
    material_type = str(data.get("material_type") or "").strip().lower()
    if material_type not in {"book", "worksheet", "screen", "other", "none"}:
        material_type = "other" if is_material else "none"
    if not is_material:
        questions = []
        material_type = "none"
    # low_quality：像题目但读不清/退化，建议整图问答或重拍。两种触发：
    #   1) 模型判为题目材料、但过滤后一道可读题都没有（全是空框 / 太糊）；
    #   2) 切出了题，但平均文本极短（斜拍/远拍 OCR 退化的简单兜底）。
    low_quality = False
    if is_material and not questions:
        low_quality = True
    elif questions:
        avg_text_len = sum(len(item["question_text"].strip()) for item in questions) / len(questions)
        if avg_text_len < 6 or dropped_empty >= len(questions):
            low_quality = True
    return {
        "is_study_material": is_material,
        "material_type": material_type,
        "questions": questions,
        "low_quality": low_quality,
        "bbox_approx": True,
    }


@app.post("/api/sessions/{session_id}/segment-questions")
async def segment_session_questions(
    session_id: str,
    request: Request,
    image: UploadFile = File(...),
    source: str = Form("ios"),
) -> dict:
    """判断画面里有没有题目并逐题给归一化 bbox（供 iOS 端调用）。复用现有 27B 视觉大模型，
    不引入新模型、不改 QA/批改主流程；只输出分割结果，不解题。"""
    init_db()
    principal = principal_from_request(request)
    if not image.filename:
        raise HTTPException(422, "image is required")
    with connect() as conn:
        require_account_session(conn, session_id, principal)
    source_text = clean_user_text(source, 80) or "ios"
    _, filename, _ = await save_upload(
        image,
        session_id,
        "segment",
        batch_id="segment",
        captured_at=utc_now(),
        capture_meta={"source": source_text, "purpose": "question_segmentation"},
    )
    settings = effective_llm_settings_for_session(session_id)
    prompt = prompts.render_prompt("question_segmentation")
    image_paths = [image_path_for_request(filename)]
    try:
        raw, _used_frontier = await _quality_vision_analyze(
            session_id, f"segment:{session_id[:8]}", settings,
            _SEGMENT_INSTRUCTIONS, prompt, image_paths,
            get_settings().grading_llm_seg_effort, LLM_PRIORITY_REALTIME,
        )
    except HTTPException:
        raise
    except Exception as exc:
        emit_log(
            f"题目分割调用视觉模型失败：{truncate_text(str(exc), 180)}",
            session_id=session_id,
            device_id=source_text,
            source="segment",
            level="warning",
        )
        raise HTTPException(502, "question segmentation failed")
    result = _build_segmentation_response(raw)
    emit_log(
        f"题目分割完成：is_study_material={result['is_study_material']}，"
        f"material_type={result['material_type']}，题数={len(result['questions'])}",
        session_id=session_id,
        device_id=source_text,
        source="segment",
    )
    result["raw"] = truncate_text(raw, 4000)
    return result


_GRADING_VERDICTS = {"对", "错", "部分对", "不确定", "未作答", "未识别"}
# 这三档是“声称判定了对错”的结论，必须有正确答案做依据，否则降级为“不确定”。
_GRADING_DECISIVE_VERDICTS = {"对", "错", "部分对"}

# 需要读图/几何推理的线索：VLM 对这类题过度自信、会编造正确答案硬判对错（实测同一道几何题
# 三张照片给三个不同标准答案），是模型能力硬伤，服务端必须强制兜底为“不确定”。
# 命中任一“读图线索”即视为不可靠；几何图形词需叠加“面积/周长/体积”才算几何推理题。
_GRADING_FIGURE_HINTS = (
    "如图", "下图", "上图", "右图", "左图", "图中", "图所示", "见图", "阴影",
)
_GRADING_SHAPE_HINTS = (
    "三角形", "梯形", "平行四边形", "正方形", "长方形", "圆", "扇形", "多边形", "直角三角形",
)
_GRADING_MEASURE_HINTS = ("面积", "周长", "体积")


def _grading_needs_figure(question_text: str) -> bool:
    """判断这道题是否“需要读图/几何推理”，从而 VLM 判分不可靠、需强制降级为“不确定”。
    判据（任一命中即真）：题干含读图线索（如图/下图/…/阴影 之一）；或含几何图形词
    （三角形/梯形/…/直角三角形 之一）且含度量词（面积/周长/体积 之一）。
    纯算术/口算/纯数字应用题不含上述线索 → 返回 False，模型的对错判定照常保留。"""
    text = question_text or ""
    if any(hint in text for hint in _GRADING_FIGURE_HINTS):
        return True
    if any(shape in text for shape in _GRADING_SHAPE_HINTS) and any(
        measure in text for measure in _GRADING_MEASURE_HINTS
    ):
        return True
    return False


def _build_grading_response(raw: str, enforce_figure_gate: bool = True) -> dict:
    """把整页批改视觉模型的原始文本解析成稳定的逐题批改契约。复用题目分割的鲁棒 JSON 抽取
    （_extract_segmentation_object）与 bbox 清洗（_segment_question_bbox）。每题取
    index/bbox/question_text/student_answer/verdict(白名单校验，非法→"不确定")/correct_answer/
    correction/error_reason/knowledge；question_text 与 student_answer 都为空的题过滤掉。
    一致性兜底：verdict∈{对,错,部分对} 但 correct_answer 为空（缺判分依据）→降级为"不确定"。
    bbox 仅供粗排序（bbox_approx=True），前端用端上 Vision 的精确 bbox 定位。
    整体解析失败时退回 {is_study_material: False, questions: [], low_quality: True}。"""
    data = _extract_segmentation_object(raw)
    if not data:
        return {
            "is_study_material": False,
            "questions": [],
            "low_quality": True,
            "bbox_approx": True,
        }
    questions: list[dict] = []
    questions_raw = data.get("questions")
    if isinstance(questions_raw, list):
        for position, entry in enumerate(questions_raw, start=1):
            if not isinstance(entry, dict):
                continue
            try:
                index = int(entry.get("index"))
            except (TypeError, ValueError):
                index = position
            question_text = truncate_text(entry.get("question_text") or "", 600)
            student_answer = truncate_text(entry.get("student_answer") or "", 600)
            # 题干与学生作答都为空的题没有任何批改价值，过滤掉。
            if not question_text.strip() and not student_answer.strip():
                continue
            verdict = str(entry.get("verdict") or "").strip()
            # 非法判定不要默认成判分档“未识别”，统一退到“不确定”，避免无依据地给出对错。
            if verdict not in _GRADING_VERDICTS:
                verdict = "不确定"
            correct_answer = truncate_text(entry.get("correct_answer") or "", 600)
            # 一致性兜底：声称判了对/错/部分对，却没给出正确答案，说明判定缺乏依据，降级为“不确定”。
            if verdict in _GRADING_DECISIVE_VERDICTS and not correct_answer.strip():
                verdict = "不确定"
            # 可靠性闸门：读图/几何推理题 VLM 判分不可靠，强制降级为“不确定”并清空正确答案，
            # 仅保留 student_answer/question_text/correction(只当思路)/error_reason/knowledge。
            # enforce_figure_gate=False 给前沿大模型(GPT-5.5)路径：它自己会算几何并诚实判不确定，不强制。
            needs_figure = enforce_figure_gate and _grading_needs_figure(question_text)
            if needs_figure and verdict in _GRADING_DECISIVE_VERDICTS:
                verdict = "不确定"
                correct_answer = ""
            questions.append(
                {
                    "index": index,
                    "bbox": _segment_question_bbox(entry.get("bbox")),
                    "question_text": question_text,
                    "student_answer": student_answer,
                    "verdict": verdict,
                    "correct_answer": correct_answer,
                    "correction": truncate_text(entry.get("correction") or "", 600),
                    "error_reason": truncate_text(entry.get("error_reason") or "", 600),
                    "knowledge": truncate_text(entry.get("knowledge") or "", 300),
                    # gradable=False 表示这是读图/几何题、判分不可靠，前端据此不显示 ✓/✗。
                    "gradable": not needs_figure,
                }
            )
    is_material = bool(data.get("is_study_material"))
    if questions:
        is_material = True
    if not is_material:
        questions = []
    # like 题目材料 but no gradable question survived → 建议整图问答或重拍。
    low_quality = bool(is_material and not questions)
    return {
        "is_study_material": is_material,
        "questions": questions,
        "low_quality": low_quality,
        "bbox_approx": True,
        # 服务端只对“可靠子集”（纯算术/口算）给出对错；读图/几何题强制“不确定”。
        "reliable_subset_only": True,
    }


_grading_llm_sem: asyncio.Semaphore | None = None


def _grading_llm_enabled() -> bool:
    s = get_settings()
    return bool(s.grading_llm_url and s.grading_llm_key)


def _grading_llm_semaphore() -> asyncio.Semaphore:
    global _grading_llm_sem
    if _grading_llm_sem is None:
        _grading_llm_sem = asyncio.Semaphore(max(1, get_settings().grading_llm_max_concurrency))
    return _grading_llm_sem


async def _quality_vision_analyze(
    session_id: str,
    label: str,
    settings,
    instructions: str,
    prompt: str,
    image_paths: list,
    effort: str,
    fallback_priority,
) -> tuple[str, bool]:
    """精批/精准分题：有 GPT-5.5 网关就走它（外部，不占本地 27B GPU gate，自带并发上限）；
    否则回退本地 27B（走 llm_gate）。返回 (raw_text, used_frontier)。"""
    if _grading_llm_enabled():
        s = get_settings()
        async with _grading_llm_semaphore():
            raw = await llm.analyze_images_responses(s, instructions, prompt, image_paths, effort)
        return raw, True
    raw = await run_with_llm_gate(
        label, session_id,
        lambda: llm.analyze_images(settings, prompt, image_paths),
        priority=fallback_priority,
    )
    return raw, False


_GRADE_INSTRUCTIONS = (
    "你是严谨且有能力的中小学作业批改助手，具备读图与几何推理能力。只输出 JSON。"
    "能据图中可见的尺寸/图形算出正确答案时，就判对/错/部分对，并在 correction 写出简要推导、correct_answer 给出结果；"
    "只有当图形被截断、关键尺寸看不清、或你确实无法确信时，才用 verdict=不确定、correct_answer 留空、correction 只给思路。"
    "绝不编造答案，correct_answer 必须与你的推导一致。"
)
_SEGMENT_INSTRUCTIONS = "你是作业题目分割器。只输出 JSON。逐题按题目在图中的真实像素位置给归一化 bbox，不要按题序均匀平铺。"


@app.post("/api/sessions/{session_id}/grade-page")
async def grade_session_questions(
    session_id: str,
    request: Request,
    image: UploadFile = File(...),
    source: str = Form("ios"),
) -> dict:
    """对整页作业逐题批改（供 iOS 端调用）。复用现有 27B 视觉大模型逐题给出对/错+订正；
    鉴权/存图/限流口径与 /segment-questions 完全一致。bbox 仅供粗排序，前端用端上 Vision 精确定位。"""
    init_db()
    principal = principal_from_request(request)
    if not image.filename:
        raise HTTPException(422, "image is required")
    with connect() as conn:
        require_account_session(conn, session_id, principal)
    source_text = clean_user_text(source, 80) or "ios"
    _, filename, _ = await save_upload(
        image,
        session_id,
        "grade",
        batch_id="grade",
        captured_at=utc_now(),
        capture_meta={"source": source_text, "purpose": "page_grading"},
    )
    settings = effective_llm_settings_for_session(session_id)
    prompt = prompts.render_prompt("page_grading")
    image_paths = [image_path_for_request(filename)]
    try:
        raw, used_frontier = await _quality_vision_analyze(
            session_id, f"grade:{session_id[:8]}", settings,
            _GRADE_INSTRUCTIONS, prompt, image_paths,
            get_settings().grading_llm_grade_effort, LLM_PRIORITY_REALTIME,
        )
    except HTTPException:
        raise
    except Exception as exc:
        emit_log(
            f"整页批改调用视觉模型失败：{truncate_text(str(exc), 180)}",
            session_id=session_id,
            device_id=source_text,
            source="grade",
            level="warning",
        )
        raise HTTPException(502, "page grading failed")
    # GPT-5.5 会自己诚实判几何题/自报不确定 → 放开服务端「几何强制不确定」闸门；本地 27B 仍强制。
    result = _build_grading_response(raw, enforce_figure_gate=not used_frontier)
    emit_log(
        f"整页批改完成：is_study_material={result['is_study_material']}，"
        f"题数={len(result['questions'])}，low_quality={result['low_quality']}",
        session_id=session_id,
        device_id=source_text,
        source="grade",
    )
    result["raw"] = truncate_text(raw, 4000)
    return result


def _relayout_str_list(raw: object, item_limit: int = 600, max_items: int = 30) -> list[str]:
    """把模型给的 given/formulas 字段安全清洗成 list[str]：非列表→空列表；逐项转字符串、截断、
    丢掉空白项；最多保留 max_items 条，避免异常输出撑爆响应。"""
    if not isinstance(raw, list):
        return []
    items: list[str] = []
    for entry in raw:
        if isinstance(entry, (dict, list)):
            continue
        text = truncate_text(str(entry) if entry is not None else "", item_limit).strip()
        if text:
            items.append(text)
        if len(items) >= max_items:
            break
    return items


def _build_relayout_response(raw: str) -> dict:
    """把“印刷题干数字化重排版”视觉模型的原始文本解析成稳定契约。复用题目分割的鲁棒 JSON 抽取
    （_extract_segmentation_object）。只取印刷题干字段：is_printed_question(bool)/title/plain_text/
    given(list[str])/ask/formulas(list[str])/figure_note/uncertain，缺字段安全降级为空值。
    顶层强制 must_compare_original=True，前端据此强制显示原图（批改证据/题干须对照原图核验）。
    本函数只搬运模型给的印刷题干文本，不在服务端拼接学生作答、不重建几何图。
    整体解析失败时退回 {is_printed_question: False, plain_text: "", must_compare_original: True}。"""
    data = _extract_segmentation_object(raw, target_keys=("is_printed_question", "plain_text", "title"))
    if not data:
        return {
            "is_printed_question": False,
            "plain_text": "",
            "must_compare_original": True,
        }
    return {
        "is_printed_question": bool(data.get("is_printed_question")),
        "title": truncate_text(data.get("title") or "", 200),
        "plain_text": truncate_text(data.get("plain_text") or "", 4000),
        "given": _relayout_str_list(data.get("given")),
        "ask": truncate_text(data.get("ask") or "", 600),
        "formulas": _relayout_str_list(data.get("formulas")),
        "figure_note": truncate_text(data.get("figure_note") or "", 300),
        "uncertain": bool(data.get("uncertain")),
        # 铁律：题干须对照原图核验；前端强制显示原图，不得仅凭重排文本作答/批改。
        "must_compare_original": True,
    }


@app.post("/api/sessions/{session_id}/relayout-question")
async def relayout_question(
    session_id: str,
    request: Request,
    image: UploadFile = File(...),
    source: str = Form("ios"),
) -> dict:
    """把图中【印刷体题干】整理成干净数字版（B 实验，供 iOS 端调用）。严格只重排印刷题干：
    不转写学生手写作答、不重画几何图（含图只在 figure_note 注明见原图）、看不清写“未识别”。
    复用现有 27B 视觉大模型；鉴权/存图/限流口径与 /segment-questions 完全一致；不解题、不批改。
    响应顶层强制 must_compare_original=true，前端据此强制显示原图。"""
    init_db()
    principal = principal_from_request(request)
    if not image.filename:
        raise HTTPException(422, "image is required")
    with connect() as conn:
        require_account_session(conn, session_id, principal)
    source_text = clean_user_text(source, 80) or "ios"
    _, filename, _ = await save_upload(
        image,
        session_id,
        "relayout",
        batch_id="relayout",
        captured_at=utc_now(),
        capture_meta={"source": source_text, "purpose": "question_relayout"},
    )
    settings = effective_llm_settings_for_session(session_id)
    prompt = prompts.render_prompt("question_relayout")
    image_paths = [image_path_for_request(filename)]
    try:
        raw = await run_with_llm_gate(
            f"relayout:{session_id[:8]}",
            session_id,
            lambda: llm.analyze_images(settings, prompt, image_paths),
            priority=LLM_PRIORITY_REALTIME,
        )
    except HTTPException:
        raise
    except Exception as exc:
        emit_log(
            f"题干重排版调用视觉模型失败：{truncate_text(str(exc), 180)}",
            session_id=session_id,
            device_id=source_text,
            source="relayout",
            level="warning",
        )
        raise HTTPException(502, "question relayout failed")
    result = _build_relayout_response(raw)
    emit_log(
        f"印刷题干重排版完成：is_printed_question={result['is_printed_question']}，"
        f"uncertain={result.get('uncertain')}，文本长度={len(result.get('plain_text') or '')}，"
        f"must_compare_original={result['must_compare_original']}",
        session_id=session_id,
        device_id=source_text,
        source="relayout",
    )
    result["raw"] = truncate_text(raw, 4000)
    return result


# ---------------------------------------------------------------------------
# 观察台 Demo（/observe）：无状态拍照观察式题目提取还原。独立入口、不污染真实数据。
# 红线：不调用 save_upload、不写任何用户表（images/errors/knowledge/sessions），
# 原图只在前端持有 blob，后端只用临时文件且 finally unlink。每图只触发【一次】VLM 调用。
# ---------------------------------------------------------------------------
_OBSERVE_QTYPES = {
    "选择题", "填空题", "计算题", "口算题", "解答题", "应用题", "判断题", "作文题", "其它",
}
_OBSERVE_GRADE_CLAUSE = (
    "\n额外要求（批改模式）：再给每道题一个 verdict 字段，只能从这六个里选一个："
    "对、错、部分对、不确定、未作答、未识别。判分门槛（对标可靠口算批改，宁可不判也别瞎判）："
    "只有纯口算/印刷算术、你能独立有把握确认正确答案时，才允许判“对/错/部分对”；"
    "凡是需要读图/几何推理、应用题、或你拿不准的，verdict 一律写“不确定”，不要硬判对错；"
    "学生没作答写“未作答”，题干或作答看不清写“未识别”。verdict 放进每道题对象里。"
)


def _observe_options(raw: object) -> list[dict]:
    """把模型给的 options 安全清洗成 [{label,text}] 列表：非列表→[]；逐项取 label/text、截断、
    丢掉 label 与 text 都空的项；最多 12 项，避免异常输出撑爆响应。"""
    if not isinstance(raw, list):
        return []
    items: list[dict] = []
    for entry in raw:
        label = ""
        text = ""
        if isinstance(entry, dict):
            label = truncate_text(entry.get("label") or "", 16).strip()
            text = truncate_text(entry.get("text") or "", 300).strip()
        elif entry is not None and not isinstance(entry, list):
            text = truncate_text(str(entry), 300).strip()
        if not label and not text:
            continue
        items.append({"label": label, "text": text})
        if len(items) >= 12:
            break
    return items


_OBSERVE_PUNCT_RE = re.compile(r"[\s　.,;:!?。，、；：！？_\-—–·…\"'“”‘’()（）\[\]【】]+")


def _observe_normalize_stem(stem: str) -> str:
    """题干归一化：去空白/标点、统一小写，用于跨图去重指纹。stem 已不含学生手写作答。"""
    return _OBSERVE_PUNCT_RE.sub("", (stem or "").lower())


def _observe_simhash(stem: str) -> str:
    """对归一化题干的 3-gram 计算 64 位 SimHash，返回 16 位十六进制。前端用 Hamming 距离判近似同题。"""
    text = _observe_normalize_stem(stem)
    if not text:
        return "0" * 16
    grams = [text[i : i + 3] for i in range(max(1, len(text) - 2))]
    v = [0] * 64
    for gram in grams:
        h = int(hashlib.sha1(gram.encode("utf-8")).hexdigest()[:16], 16)
        for bit in range(64):
            v[bit] += 1 if (h >> bit) & 1 else -1
    out = 0
    for bit in range(64):
        if v[bit] > 0:
            out |= 1 << bit
    return f"{out:016x}"


def _build_observe_response(raw: str, do_grade: bool) -> dict:
    """把观察台单图提取的视觉模型原始文本解析成稳定契约。复用题目分割的鲁棒 JSON 抽取
    （_extract_segmentation_object）。每题清洗 index/number/subject/qtype/stem/options/blanks/
    figure_note/has_student_answer/student_answer，并由服务端追加去重用的 fingerprint(40hex)+simhash(16hex)。
    do_grade=True 时校验 verdict 白名单（非法→“不确定”），并对命中 _grading_needs_figure（读图/几何）的题
    强制降级为“不确定”，避免模型编造标准答案（沿用 grade-page 的可靠性闸门）。
    stem 与 student_answer 都为空的题过滤掉；整体解析失败退回 {is_study_material: False, questions: [], low_quality: True}。"""
    data = _extract_segmentation_object(raw, target_keys=("is_study_material", "questions"))
    if not data:
        return {"is_study_material": False, "questions": [], "low_quality": True}
    questions: list[dict] = []
    questions_raw = data.get("questions")
    if isinstance(questions_raw, list):
        for position, entry in enumerate(questions_raw, start=1):
            if not isinstance(entry, dict):
                continue
            try:
                index = int(entry.get("index"))
            except (TypeError, ValueError):
                index = position
            stem = truncate_text(entry.get("stem") or "", 800)
            student_answer = truncate_text(entry.get("student_answer") or "", 600)
            # 题干与学生作答都为空的题没有提取价值，过滤掉。
            if not stem.strip() and not student_answer.strip():
                continue
            qtype = str(entry.get("qtype") or "").strip()
            if qtype not in _OBSERVE_QTYPES:
                qtype = "其它"
            try:
                blanks = max(0, int(entry.get("blanks")))
            except (TypeError, ValueError):
                blanks = 0
            stem_norm = _observe_normalize_stem(stem)
            question = {
                "index": index,
                "number": truncate_text(entry.get("number") or "", 40),
                "subject": truncate_text(entry.get("subject") or "", 40),
                "qtype": qtype,
                "stem": stem,
                "options": _observe_options(entry.get("options")),
                "blanks": blanks,
                "figure_note": truncate_text(entry.get("figure_note") or "", 300),
                "has_student_answer": bool(entry.get("has_student_answer")),
                "student_answer": student_answer,
                # 去重指纹：fingerprint 精确同题，simhash 近似同题（前端按 Hamming<=3 合并）。
                "fingerprint": hashlib.sha1(stem_norm.encode("utf-8")).hexdigest() if stem_norm else "",
                "simhash": _observe_simhash(stem),
            }
            if do_grade:
                verdict = str(entry.get("verdict") or "").strip()
                if verdict not in _GRADING_VERDICTS:
                    verdict = "不确定"
                # 可靠性闸门：读图/几何题 VLM 判分不可靠，强制降级为“不确定”。
                if _grading_needs_figure(stem) and verdict in _GRADING_DECISIVE_VERDICTS:
                    verdict = "不确定"
                question["verdict"] = verdict
                question["gradable"] = not _grading_needs_figure(stem)
            questions.append(question)
    is_material = bool(data.get("is_study_material")) or bool(questions)
    if not is_material:
        questions = []
    return {
        "is_study_material": is_material,
        "questions": questions,
        "low_quality": bool(is_material and not questions),
    }


# 观察台 Demo 的轻量防滥用：同一 batch_token 维度的活跃请求串行（防单入口独占共享 27B），
# 配合前端单批 ≤40 张硬上限 + BACKGROUND 优先级（给课堂 realtime 让路）。
_OBSERVE_BATCH_LOCKS: dict[str, asyncio.Lock] = {}
_OBSERVE_MAX_IMAGE_BYTES = 16 * 1024 * 1024


@app.post("/api/observe-demo/extract")
async def observe_demo_extract(
    request: Request,
    image: UploadFile = File(...),
    do_grade: bool = Form(False),
    batch_token: str = Form(""),
) -> dict:
    """观察台 Demo 单图题目结构化提取（无状态、不污染真实数据）。复用现有 27B 视觉大模型，
    每图只触发一次 analyze_images；BACKGROUND 优先级给课堂实时请求让路。原图写临时文件、用完即删，
    不调用 save_upload、不写任何用户表。默认要求登录；若确需公开演示，可显式设置 PXJ_OBSERVE_DEMO_PUBLIC=true。"""
    init_db()
    settings_for_demo = get_settings()
    principal_from_request(request, required=not settings_for_demo.observe_demo_public)
    if not image.filename:
        raise HTTPException(422, "image is required")
    payload = await image.read()
    if not payload:
        raise HTTPException(422, "image is empty")
    if len(payload) > _OBSERVE_MAX_IMAGE_BYTES:
        raise HTTPException(413, "image too large (Demo 单图上限 16MB)")
    token = clean_user_text(batch_token, 64) or "default"
    lock = _OBSERVE_BATCH_LOCKS.setdefault(token, asyncio.Lock())
    tmp_dir = get_settings().data_dir / "observe_demo"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}.jpg"
    tmp_path.write_bytes(payload)
    settings = effective_llm_settings_for_session(None)
    prompt = prompts.render_prompt("observe_extract")
    if do_grade:
        prompt = prompt + _OBSERVE_GRADE_CLAUSE
    try:
        async with lock:
            raw = await run_with_llm_gate(
                f"observe:{tmp_path.stem[:8]}",
                None,
                lambda: llm.analyze_images(settings, prompt, [tmp_path]),
                priority=LLM_PRIORITY_BACKGROUND,
            )
    except HTTPException:
        raise
    except Exception as exc:
        emit_log(
            f"观察台提取调用视觉模型失败：{truncate_text(str(exc), 180)}",
            source="observe",
            level="warning",
        )
        raise HTTPException(502, "observe extract failed")
    finally:
        tmp_path.unlink(missing_ok=True)
    result = _build_observe_response(raw, do_grade)
    result["filename"] = truncate_text(image.filename, 200)
    result["raw"] = truncate_text(raw, 4000)
    return result


@app.get("/observe", response_class=HTMLResponse)
def observe_demo_page() -> str:
    return (Path(__file__).parent / "static" / "observe.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 题目提取（融入 iPhone/iPad 观察）：有状态会话端点。每张照片提取题目→服务端去重累积成
# 「题目集」存库→会话创建即入历史(status='saved')→还原页/空白卷由服务端渲染 HTML 供 WKWebView 显示与打印。
# 复用 observe_extract 提示词与 _build_observe_response；几何/读图题已被强制“不确定”，本功能不判分。
# ---------------------------------------------------------------------------
def _simhash_hamming(a: str, b: str) -> int:
    if not a or not b:
        return 64
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 64


def _consolidate_question_set(prior: list[dict], new: list[dict]) -> list[dict]:
    """把本张图新提取的题并入已有题目集并去重（服务端是唯一去重者，两端一致）。
    fingerprint 精确命中跳过；否则 simhash 十六进制 Hamming<=3 视为近似同题跳过。保留先到题。"""
    result = list(prior)
    seen_fp = {q.get("fingerprint") for q in result if q.get("fingerprint")}
    for q in new:
        if not (q.get("stem") or "").strip():
            continue
        fp = q.get("fingerprint") or ""
        if fp and fp in seen_fp:
            continue
        sh = q.get("simhash") or ""
        if sh and any(_simhash_hamming(sh, e.get("simhash") or "") <= 3 for e in result):
            continue
        result.append(q)
        if fp:
            seen_fp.add(fp)
    return result


def _latest_question_set(session_id: str) -> list[dict]:
    """读取本会话已存的题目集（report_events 中单条 event_type='question_set'）。"""
    with connect() as conn:
        row = conn.execute(
            "SELECT content FROM report_events WHERE session_id=? AND event_type='question_set' ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    if not row:
        return []
    try:
        data = json.loads(row["content"])
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _save_question_set(session_id: str, qset: list[dict]) -> None:
    """把题目集 upsert 成单条 report_events 行（避免逐张追加撑爆 overview 的 40 条窗口、也避免
    record_report_event 的 6000 字截断毁坏 JSON）。content 为完整 JSON（TEXT 列无长度限制）。"""
    payload = json.dumps(qset, ensure_ascii=False)
    now = utc_now()
    with connect() as conn:
        existing = conn.execute(
            "SELECT id FROM report_events WHERE session_id=? AND event_type='question_set' ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if existing:
            conn.execute("UPDATE report_events SET content=?, created_at=? WHERE id=?", (payload, now, existing["id"]))
        else:
            conn.execute(
                "INSERT INTO report_events(session_id, analysis_id, event_type, title, content, created_at) VALUES(?,?,?,?,?,?)",
                (session_id, None, "question_set", "题目集", payload, now),
            )
        conn.commit()


async def _extract_questions_from_stored_image(
    session_id: str,
    filename: str,
    *,
    source_text: str,
    label_prefix: str = "extract",
) -> dict:
    """对已经保存到 images 目录的一张图提题，并把结果合并进本会话 question_set。

    这个内部函数不改 sessions.status/title；显式拍题提取和智能观察停止后的整轮提题共用它，
    避免普通观察会话被误改成“题目提取”会话。
    """
    settings = effective_llm_settings_for_session(session_id)
    prompt = prompts.render_prompt("observe_extract")
    image_paths = [image_path_for_request(filename)]
    raw = await run_with_llm_gate(
        f"{label_prefix}:{session_id[:8]}",
        session_id,
        lambda: llm.analyze_images(settings, prompt, image_paths),
        priority=LLM_PRIORITY_BACKGROUND,
    )
    per = _build_observe_response(raw, do_grade=False)
    for q in per["questions"]:
        q["src_filename"] = filename
    before = _latest_question_set(session_id)
    consolidated = _consolidate_question_set(before, per["questions"])
    _save_question_set(session_id, consolidated)
    added_count = max(0, len(consolidated) - len(before))
    emit_log(
        (
            f"题目提取完成：图片 {filename} 本图 {len(per['questions'])} 题，"
            f"新增 {added_count} 题，去重后题集 {len(consolidated)} 题，low_quality={per['low_quality']}"
        ),
        session_id=session_id,
        device_id=source_text,
        source="extract",
    )
    return {
        "is_study_material": per["is_study_material"],
        "questions": per["questions"],
        "low_quality": per["low_quality"],
        "question_set": consolidated,
        "question_set_count": len(consolidated),
        "added_count": added_count,
        "filename": filename,
        "raw": truncate_text(raw, 4000),
    }


def _question_extraction_filenames_for_session(session_id: str, limit: int = 80) -> list[str]:
    """选择本轮观察里值得进入整轮提题的图片：跳过服务端已判定 invalid/duplicate 的帧。"""
    limit = max(1, min(int(limit or 80), 160))
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT images.filename
            FROM images
            LEFT JOIN session_observations obs ON obs.image_id = images.id
            WHERE images.session_id=?
              AND images.kind IN ('burst', 'single', 'extract', 'grade')
              AND COALESCE(obs.novelty_status, 'unknown') NOT IN ('invalid', 'duplicate')
            ORDER BY images.sequence_index ASC, images.created_at ASC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    return [row["filename"] for row in rows if row["filename"]]


async def run_session_question_extraction(session_id: str, filenames: list[str], task_id: str | None = None) -> dict:
    """后台整轮提题任务：串行处理会话图片，逐张写入同一个 question_set。"""
    if not filenames:
        filenames = _question_extraction_filenames_for_session(session_id)
    initial_count = len(_latest_question_set(session_id))
    processed = 0
    failed = 0
    low_quality = 0
    emit_log(
        f"本轮题目提取开始：待处理 {len(filenames)} 张关键图，已有题集 {initial_count} 道",
        session_id=session_id,
        source="extract",
    )
    for index, filename in enumerate(filenames, start=1):
        path = image_path_for_request(filename)
        if not path.exists():
            failed += 1
            emit_log(
                f"题目提取跳过缺失图片：{filename}",
                session_id=session_id,
                source="extract",
                level="warning",
            )
            continue
        try:
            result = await _extract_questions_from_stored_image(
                session_id,
                filename,
                source_text="observation",
                label_prefix="extract_all",
            )
            processed += 1
            if result.get("low_quality"):
                low_quality += 1
        except LLMTaskCancelled:
            raise
        except Exception as exc:
            failed += 1
            emit_log(
                f"题目提取失败：{index}/{len(filenames)} {filename}；{truncate_text(str(exc), 180)}",
                session_id=session_id,
                source="extract",
                level="warning",
            )
    final_count = len(_latest_question_set(session_id))
    with connect() as conn:
        conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (utc_now(), session_id))
    emit_log(
        (
            f"本轮题目提取完成：处理 {processed}/{len(filenames)} 张，失败 {failed} 张，"
            f"低质量 {low_quality} 张，新增 {max(0, final_count - initial_count)} 道，题集共 {final_count} 道"
        ),
        session_id=session_id,
        source="extract",
    )
    return {
        "processed": processed,
        "failed": failed,
        "low_quality": low_quality,
        "initial_count": initial_count,
        "question_set_count": final_count,
        "task_id": task_id or "",
    }


@app.post("/api/sessions/{session_id}/extract-questions")
async def extract_session_questions(
    session_id: str,
    request: Request,
    image: UploadFile = File(...),
    source: str = Form("ios"),
) -> dict:
    """题目提取（供 iOS 观察模式调用）：对一张拍清的试卷图提取结构化题目，服务端去重累积成题目集并存库，
    会话置 status='saved' 直接进历史。鉴权/存图/限流口径与 segment-questions 一致；BACKGROUND 优先级给课堂实时让路；
    不判分（do_grade=False，几何题已强制不确定）、不触发任何报告任务（绝不调用 finish）。"""
    init_db()
    principal = principal_from_request(request)
    if not image.filename:
        raise HTTPException(422, "image is required")
    with connect() as conn:
        require_account_session(conn, session_id, principal)
    source_text = clean_user_text(source, 80) or "ios"
    _, filename, _ = await save_upload(
        image,
        session_id,
        "extract",
        batch_id="extract",
        captured_at=utc_now(),
        capture_meta={"source": source_text, "purpose": "question_extraction"},
    )
    try:
        result = await _extract_questions_from_stored_image(
            session_id,
            filename,
            source_text=source_text,
            label_prefix="extract",
        )
    except HTTPException:
        raise
    except Exception as exc:
        emit_log(
            f"题目提取调用视觉模型失败：{truncate_text(str(exc), 180)}",
            session_id=session_id,
            device_id=source_text,
            source="extract",
            level="warning",
        )
        raise HTTPException(502, "question extraction failed")
    subject = next((q.get("subject") for q in result["question_set"] if q.get("subject")), "")
    title = (f"{subject} · 题目提取" if subject else "题目提取")
    with connect() as conn:
        conn.execute(
            "UPDATE sessions SET status='saved', updated_at=?, title=? WHERE id=?",
            (utc_now(), title, session_id),
        )
        conn.commit()
    return {
        "is_study_material": result["is_study_material"],
        "questions": result["questions"],
        "low_quality": result["low_quality"],
        "question_set": result["question_set"],
        "question_set_count": result["question_set_count"],
        "filename": filename,
    }


@app.post("/api/sessions/{session_id}/extract-all-questions")
async def extract_all_session_questions(
    session_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    limit: int = Form(80),
) -> dict:
    """把本轮已保存的关键图片排入后台整轮提题任务，供“停止观察后提取所有题目”按钮调用。"""
    init_db()
    principal = principal_from_request(request)
    with connect() as conn:
        require_account_session(conn, session_id, principal)
        existing = conn.execute(
            """
            SELECT id, status
            FROM task_runs
            WHERE session_id=? AND task_kind='question_extraction_session' AND status IN ('queued','running')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    filenames = _question_extraction_filenames_for_session(session_id, limit=limit)
    if existing:
        return {
            "session_id": session_id,
            "task_id": existing["id"],
            "status": existing["status"],
            "image_count": len(filenames),
            "question_set_count": len(_latest_question_set(session_id)),
        }
    task_id = record_task_run(
        "question_extraction_session",
        session_id=session_id,
        payload={"session_id": session_id, "filenames": filenames},
        priority=TASK_PRIORITY_BACKGROUND,
    )
    emit_log(
        f"已排队提取本轮所有题目：{len(filenames)} 张关键图",
        session_id=session_id,
        source="extract",
    )
    background_tasks.add_task(execute_next_task_run_now)
    return {
        "session_id": session_id,
        "task_id": task_id,
        "status": "queued",
        "image_count": len(filenames),
        "question_set_count": len(_latest_question_set(session_id)),
    }


def _restore_esc(s: object) -> str:
    return html.escape(str(s or ""))


def _restore_stem_html(stem: str) -> str:
    return _restore_esc(stem).replace("未识别", '<span style="color:#c0392b">未识别</span>')


def _restore_options_html(q: dict) -> str:
    opts = q.get("options") or []
    if not isinstance(opts, list) or not opts:
        return ""
    items = "".join(
        f"<li><b>{_restore_esc(o.get('label'))}.</b> {_restore_esc(o.get('text'))}</li>"
        for o in opts if isinstance(o, dict)
    )
    return f'<ul style="margin:6px 0 0;padding-left:18px">{items}</ul>'


def _blank_answer_area(q: dict) -> str:
    qtype = q.get("qtype") or ""
    if qtype == "选择题" and (q.get("options") or []):
        return ""
    try:
        blanks = int(q.get("blanks") or 0)
    except (TypeError, ValueError):
        blanks = 0
    n = blanks if blanks > 0 else (1 if qtype == "口算题" else (12 if qtype == "作文题" else 4))
    line = '<div style="border-bottom:1px solid #999;height:1.7em;margin:7px 0"></div>'
    return f'<div>{line * n}</div>'


def _render_restore_html(qset: list[dict], view: str, base_url: str) -> str:
    """从已存题目集渲染服务端 HTML（还原页 / 空白卷）。含图题只输出『见原图』+原图缩略（走免鉴权
    thumbnail 绝对URL），不重绘几何、不渲染学生手写。view='blank' 带 @media print A4 空白作答区。"""
    is_blank = view == "blank"
    subject = next((q.get("subject") for q in qset if q.get("subject")), "")
    cards = []
    for i, q in enumerate(qset):
        num = _restore_esc(q.get("number")) or (f"{i + 1}." if is_blank else f"第{i + 1}题")
        figure = ""
        fn = q.get("figure_note") or ""
        src = q.get("src_filename") or ""
        if fn or (q.get("qtype") in ("解答题", "应用题") and "图" in (q.get("stem") or "")):
            note = _restore_esc(fn) or "此题含图，见原图"
            img = f'<img src="{base_url}/api/images/{_restore_esc(src)}/thumbnail" style="max-width:60%;border:1px solid #ccc;margin-top:6px;display:block">' if src else ""
            figure = f'<div style="color:#b9770e;font-size:13px;margin-top:6px">🖼 {note}</div>{img}'
        pills = f'<span style="font-size:12px;color:#666">[{_restore_esc(q.get("qtype"))}]</span>'
        if not is_blank and q.get("subject"):
            pills = f'<span style="font-size:12px;color:#666">{_restore_esc(q.get("subject"))} · </span>' + pills
        blanks_note = ""
        if not is_blank and q.get("blanks"):
            blanks_note = f'<div style="font-size:12px;color:#888;margin-top:4px">填空 {int(q["blanks"])} 处</div>'
        body = (
            f'<div style="margin-bottom:6px"><b style="color:#2962ff">{num}</b> {pills}</div>'
            f'<div style="font-size:15px;white-space:pre-wrap">{_restore_stem_html(q.get("stem"))}</div>'
            f'{_restore_options_html(q)}{blanks_note}{figure}'
        )
        if is_blank:
            body += _blank_answer_area(q)
            cards.append(f'<div style="padding:12px 0;border-bottom:1px dashed #ccc;break-inside:avoid;page-break-inside:avoid">{body}</div>')
        else:
            thumb = ""
            if src:
                thumb = f'<div style="flex:0 0 180px"><img src="{base_url}/api/images/{_restore_esc(src)}/thumbnail" style="width:100%;border:1px solid #ddd;border-radius:6px"><div style="font-size:11px;color:#999;text-align:center">原图核对</div></div>'
            cards.append(f'<div style="display:flex;gap:14px;border:1px solid #e0e0e0;border-radius:10px;padding:14px;margin-bottom:12px"><div style="flex:1">{body}</div>{thumb}</div>')
    body_html = "".join(cards) or '<div style="color:#888;text-align:center;padding:40px">还没有提取到题目。把试卷拍清楚、铺满取景框再试。</div>'
    if is_blank:
        head = f'<div style="display:flex;justify-content:space-between;border-bottom:2px solid #111;padding-bottom:8px;margin-bottom:16px;font-size:13px"><span>{_restore_esc(subject)} 空白练习卷</span><span>姓名______ 日期______ 得分____</span></div>'
        return (
            '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<style>@page{size:A4;margin:14mm}body{font-family:-apple-system,"PingFang SC",sans-serif;color:#111;padding:16px;max-width:780px;margin:0 auto}</style>'
            f'</head><body>{head}{body_html}</body></html>'
        )
    return (
        '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{font-family:-apple-system,"PingFang SC",sans-serif;color:#1a1a1a;padding:14px;margin:0}</style>'
        f'</head><body><div style="color:#888;font-size:12px;margin-bottom:10px">题目还原（印刷题干语义重排 + 原图核对）。含图题只标“见原图”，不重绘、不渲染手写作答。</div>{body_html}</body></html>'
    )


@app.get("/api/sessions/{session_id}/restore-page", response_class=HTMLResponse)
def session_restore_page(session_id: str, request: Request, view: str = "restore") -> str:
    """从已存题目集渲染还原页/空白卷 HTML（供 iOS WKWebView 显示与打印）。鉴权口径同其它会话端点。"""
    init_db()
    principal = principal_from_request(request)
    with connect() as conn:
        require_account_session(conn, session_id, principal)
    qset = _latest_question_set(session_id)
    base_url = (get_settings().public_base_url or "").rstrip("/")
    return _render_restore_html(qset, "blank" if view == "blank" else "restore", base_url)


@app.post("/api/solve-single")
async def solve_single(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    device_id: str = Form("iphone"),
    page_hint: str = Form(""),
    question_hint: str = Form(""),
    student_goal: str = Form(""),
    student_profile_id: str = Form(""),
    report_style: str = Form(""),
    assistant_focus: str = Form(""),
) -> dict:
    principal = principal_from_request(request)
    session_id = uuid.uuid4().hex
    now = utc_now()
    resolved_student_profile_id = resolve_student_profile(principal["account_id"], student_profile_id)
    goal = clean_user_text(student_goal)
    style = clean_user_text(report_style, 500)
    requested_focus = clean_user_text(assistant_focus, ASSISTANT_FOCUS_CHAR_LIMIT)
    inferred = infer_need_tags_from_text(goal + " " + style + " " + requested_focus)
    focus = requested_focus or ("根据学生输入初始化关注点：" + "、".join(tag_label(tag) for tag in inferred) if inferred else "")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions(
                id, account_id, student_profile_id, created_by_user_id, device_id, mode, title, status, created_at, updated_at,
                student_goal, assistant_focus, inferred_needs, report_style
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                principal["account_id"],
                resolved_student_profile_id,
                principal.get("user_id", ""),
                device_id,
                "single",
                "单张拍题解析",
                "uploaded",
                now,
                now,
                goal,
                focus,
                json_dumps(inferred),
                style,
            ),
        )
    if goal:
        record_report_event(session_id, "student_goal", "学生本回合要求", goal)
    _, filename, _ = await save_upload(
        image,
        session_id,
        "single",
        page_hint=page_hint,
        question_hint=question_hint,
        sequence_index=0,
    )
    analysis_id = uuid.uuid4().hex
    prompt = prompts.render_prompt("single_analysis", page_hint=page_hint or "未知", question_hint=question_hint or "未知")
    strategy_context = build_strategy_context({"student_goal": goal, "assistant_focus": focus, "inferred_needs": inferred, "report_style": style})
    prompt = f"{prompt}\n\n{strategy_context}"
    with connect() as conn:
        conn.execute(
            "INSERT INTO analyses(id, session_id, scope, status, prompt, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (analysis_id, session_id, "single", "running", prompt, now, now),
        )
    with connect() as conn:
        conn.execute("UPDATE sessions SET status=?, updated_at=? WHERE id=?", ("analyzing", utc_now(), session_id))
    emit_log("收到单张拍题图片，已保存并进入后台解析流程", session_id=session_id, device_id=device_id)
    task_id = record_task_run(
        "vision_analysis",
        session_id=session_id,
        analysis_id=analysis_id,
        payload={"scope": "single", "filenames": [filename]},
        priority=TASK_PRIORITY_BACKGROUND,
    )
    background_tasks.add_task(execute_next_task_run_now)
    return session_payload(session_id, principal)


@app.post("/api/sessions")
def create_session(
    request: Request,
    device_id: str = Form("iphone"),
    mode: str = Form("burst"),
    title: str = Form("智能观察学习回合"),
    student_goal: str = Form(""),
    student_profile_id: str = Form(""),
    report_style: str = Form(""),
    assistant_focus: str = Form(""),
) -> dict:
    principal = principal_from_request(request)
    session_id = uuid.uuid4().hex
    now = utc_now()
    resolved_student_profile_id = resolve_student_profile(principal["account_id"], student_profile_id)
    goal = clean_user_text(student_goal)
    style = clean_user_text(report_style, 500)
    requested_focus = clean_user_text(assistant_focus, ASSISTANT_FOCUS_CHAR_LIMIT)
    inferred = infer_need_tags_from_text(goal + " " + style + " " + requested_focus)
    focus = requested_focus or ("根据学生输入初始化关注点：" + "、".join(tag_label(tag) for tag in inferred) if inferred else "")
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions(
                id, account_id, student_profile_id, created_by_user_id, device_id, mode, title, status, created_at, updated_at,
                student_goal, assistant_focus, inferred_needs, report_style
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                principal["account_id"],
                resolved_student_profile_id,
                principal.get("user_id", ""),
                device_id,
                mode,
                title,
                "created",
                now,
                now,
                goal,
                focus,
                json_dumps(inferred),
                style,
            ),
        )
    emit_log("创建学习回合", session_id=session_id, device_id=device_id)
    if goal:
        record_report_event(session_id, "student_goal", "学生本回合要求", goal)
    return {"session_id": session_id}


@app.patch("/api/sessions/{session_id}/strategy")
async def update_session_strategy(session_id: str, request: Request) -> dict:
    principal = principal_from_request(request)
    body = await request.json()
    has_goal = "student_goal" in body or "goal" in body
    has_style = "report_style" in body
    has_focus = "assistant_focus" in body or "focus" in body
    goal = clean_user_text(body.get("student_goal", body.get("goal", ""))) if has_goal else None
    style = clean_user_text(body.get("report_style", ""), 500) if has_style else None
    focus = clean_user_text(body.get("assistant_focus", body.get("focus", "")), ASSISTANT_FOCUS_CHAR_LIMIT) if has_focus else None
    inferred = infer_need_tags_from_text(" ".join(value for value in (goal, style, focus) if value))
    now = utc_now()
    with connect() as conn:
        session = require_account_session(conn, session_id, principal)
        merged = merge_tags(json_list(session["inferred_needs"]), inferred)
        next_goal = goal if has_goal else session["student_goal"]
        next_style = style if has_style else session["report_style"]
        next_focus = focus if has_focus else session["assistant_focus"]
        conn.execute(
            """
            UPDATE sessions
            SET student_goal=?, report_style=?, assistant_focus=?, inferred_needs=?, updated_at=?
            WHERE id=?
            """,
            (next_goal, next_style, next_focus, json_dumps(merged), now, session_id),
        )
    record_report_event(
        session_id,
        "strategy_update",
        "学习目标/报告策略更新",
        "\n".join(
            [
                f"student_goal={goal if has_goal else '(unchanged)'}",
                f"report_style={style if has_style else '(unchanged)'}",
                f"assistant_focus={focus if has_focus else '(unchanged)'}",
                f"inferred_needs={','.join(inferred)}",
            ]
        ),
    )
    with connect() as conn:
        updated = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    return {"session": dict(updated), "strategy": session_strategy(dict(updated))}


@app.post("/api/sessions/{session_id}/batches")
async def upload_batch(
    session_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    images: list[UploadFile] = File(...),
    device_id: str = Form("iphone"),
    environment: str = Form(""),
    capture_meta: str = Form(""),
) -> dict:
    principal = principal_from_request(request)
    with connect() as conn:
        session = require_account_session(conn, session_id, principal)
    batch_id = uuid.uuid4().hex
    filenames = []
    analysis_filenames = []
    image_rows = []
    analysis_image_rows = []
    capture_items = parse_capture_meta(capture_meta, len(images))
    with connect() as conn:
        max_sequence = conn.execute(
            "SELECT COALESCE(MAX(sequence_index), -1) AS max_sequence FROM images WHERE session_id=?",
            (session_id,),
        ).fetchone()["max_sequence"]
    for index, image in enumerate(images):
        meta = capture_items[index]
        meta_sequence = meta_int(meta, "sequence_index", "sequenceIndex", "index")
        sequence_index = meta_sequence if meta_sequence is not None else (max_sequence + index + 1)
        captured_at = meta_text(meta, "captured_at", "capturedAt", "timestamp", "created_at")
        page_hint = meta_text(meta, "page_hint", "pageHint")
        question_hint = meta_text(meta, "question_hint", "questionHint")
        image_id, filename, observation = await save_upload(
            image,
            session_id,
            "burst",
            batch_id,
            page_hint=page_hint,
            question_hint=question_hint,
            captured_at=captured_at or None,
            sequence_index=sequence_index,
            capture_meta=meta,
        )
        filenames.append(filename)
        row = {
            "image_id": image_id,
            "filename": filename,
            "captured_at": observation.get("captured_at") or captured_at,
            "sequence_index": sequence_index,
            "capture_meta": observation.get("capture_meta") or meta_string(meta),
            "novelty_status": observation.get("novelty_status", "unknown"),
            "duplicate_of_image_id": observation.get("duplicate_of_image_id", ""),
            "signal_summary": observation.get("signal_summary", ""),
            "discard_reason": observation.get("discard_reason", ""),
            "discard_detail": observation.get("discard_detail", ""),
        }
        image_rows.append(row)
        if row["novelty_status"] not in {"duplicate", "invalid"}:
            analysis_filenames.append(filename)
            analysis_image_rows.append(row)
    now = utc_now()
    analysis_id = uuid.uuid4().hex
    previous_context = build_previous_batch_context(session_id, exclude_batch_id=batch_id)
    if analysis_image_rows:
        learning_context = build_learning_items_context(session_id)
        merged_context = previous_context
        if learning_context:
            merged_context = f"{previous_context}\n\n此前已结构化保存的学习条目：\n{learning_context}"
        prompt = build_batch_prompt(environment, analysis_image_rows, merged_context, build_strategy_context(session_strategy(session)))
        with connect() as conn:
            conn.execute(
                "INSERT INTO analyses(id, session_id, batch_id, scope, status, prompt, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (analysis_id, session_id, batch_id, "batch", "running", prompt, now, now),
            )
            conn.execute("UPDATE sessions SET status=?, updated_at=? WHERE id=?", ("analyzing", now, session_id))
        emit_log(
            (
                f"收到智能观察批次：{len(filenames)} 张，其中新增关键画面 {len(analysis_filenames)} 张，"
                f"重复观察 {sum(1 for row in image_rows if row.get('novelty_status') == 'duplicate')} 张，"
                f"无相关画面 {sum(1 for row in image_rows if row.get('novelty_status') == 'invalid')} 张"
            ),
            session_id=session_id,
            device_id=device_id,
        )
        task_id = record_task_run(
            "vision_analysis",
            session_id=session_id,
            analysis_id=analysis_id,
            payload={"scope": "batch", "batch_id": batch_id, "filenames": analysis_filenames},
            priority=TASK_PRIORITY_BACKGROUND,
        )
        background_tasks.add_task(execute_next_task_run_now)
    else:
        invalid_count = sum(1 for row in image_rows if row.get("novelty_status") == "invalid")
        prompt = "无新增可解析关键画面。"
        content = skipped_batch_content(image_rows)
        with connect() as conn:
            conn.execute(
                "INSERT INTO analyses(id, session_id, batch_id, scope, status, prompt, content, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (analysis_id, session_id, batch_id, "batch", "done", prompt, content, now, now),
            )
            conn.execute("UPDATE sessions SET status=?, updated_at=? WHERE id=?", ("analyzed", now, session_id))
        if invalid_count:
            emit_log(
                f"收到智能观察批次：{len(filenames)} 张，其中无相关画面 {invalid_count} 张，已跳过大模型解析",
                session_id=session_id,
                device_id=device_id,
            )
        else:
            emit_log(f"收到重复观察批次：{len(filenames)} 张，无新增关键画面，已跳过大模型解析", session_id=session_id, device_id=device_id)
    return {
        "session_id": session_id,
        "batch_id": batch_id,
        "analysis_id": analysis_id,
        "image_count": len(filenames),
        "analysis_image_count": len(analysis_filenames),
        "duplicate_image_count": sum(1 for row in image_rows if row.get("novelty_status") == "duplicate"),
        "discarded_image_count": sum(1 for row in image_rows if row.get("novelty_status") == "invalid"),
        "skipped_image_count": len(filenames) - len(analysis_filenames),
    }


@app.post("/api/sessions/{session_id}/finish")
async def finish_session(session_id: str, request: Request, background_tasks: BackgroundTasks, device_id: str = Form("iphone")) -> dict:
    principal = principal_from_request(request)
    now = utc_now()
    with connect() as conn:
        session = require_account_session(conn, session_id, principal)
        image_count = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM images
            LEFT JOIN session_observations obs ON obs.image_id = images.id
            WHERE images.session_id=? AND COALESCE(obs.novelty_status, 'unknown') != 'invalid'
            """,
            (session_id,),
        ).fetchone()["count"]
        qa_count = conn.execute(
            "SELECT COUNT(*) AS count FROM qa_events WHERE session_id=?",
            (session_id,),
        ).fetchone()["count"]
        existing = conn.execute(
            "SELECT id, status FROM analyses WHERE session_id=? AND scope='final' ORDER BY created_at DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if existing and existing["status"] == "running":
            analysis_id = existing["id"]
        else:
            analysis_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO analyses(id, session_id, scope, status, prompt, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (analysis_id, session_id, "final", "running", prompts.get_prompt("final_analysis_placeholder"), now, now),
            )
        conn.execute(
            "UPDATE sessions SET status=?, finished_at=?, updated_at=? WHERE id=?",
            ("finalizing", now, now, session_id),
        )
    emit_log("收到结束学习回合请求，开始汇总最终报告", session_id=session_id, device_id=device_id)
    if image_count == 0:
        if qa_count > 0:
            task_id = record_task_run(
                "qa_session_summary",
                session_id=session_id,
                analysis_id=analysis_id,
                payload={"scope": "final", "mode": "qa_session_summary"},
                priority=TASK_PRIORITY_FINAL_REPORT,
            )
            background_tasks.add_task(execute_next_task_run_now)
            return {
                "session_id": session_id,
                "analysis_id": analysis_id,
                "status": "finalizing",
                "image_count": image_count,
                "qa_count": qa_count,
                "summary_mode": "qa_session_summary",
            }
        with connect() as conn:
            empty_report = prompts.get_prompt("empty_report")
            conn.execute(
                "UPDATE analyses SET status=?, content=?, updated_at=? WHERE id=?",
                ("done", empty_report, now, analysis_id),
            )
            conn.execute(
                "UPDATE sessions SET summary=?, status=?, report_generated_at=?, updated_at=? WHERE id=?",
                (empty_report, "completed", now, now, session_id),
            )
        return {"session_id": session_id, "analysis_id": analysis_id, "status": "completed", "image_count": image_count, "qa_count": qa_count}
    task_id = record_task_run(
        "final_report",
        session_id=session_id,
        analysis_id=analysis_id,
        payload={"scope": "final"},
        priority=TASK_PRIORITY_FINAL_REPORT,
    )
    background_tasks.add_task(execute_next_task_run_now)
    return {"session_id": session_id, "analysis_id": analysis_id, "status": "finalizing", "image_count": image_count}


@app.post("/api/logs")
async def ingest_log(request: Request) -> dict:
    bind_account_context_from_token(request)
    body = await request.json()
    emit_log(
        str(body.get("message", "")),
        session_id=body.get("session_id"),
        device_id=body.get("device_id"),
        level=body.get("level", "info"),
        source=body.get("source", "ios"),
    )
    return {"ok": True}


@app.get("/api/logs")
def get_logs(request: Request, session_id: str | None = None, after_id: int = 0) -> dict:
    principal = principal_from_request(request)
    sql = "SELECT * FROM logs WHERE id > ? AND account_id = ?"
    params: list = [after_id, principal["account_id"]]
    if session_id:
        sql += " AND session_id = ?"
        params.append(session_id)
    sql += " ORDER BY id DESC LIMIT 200"
    with connect() as conn:
        logs = [dict(row) for row in conn.execute(sql, params)]
    return {"logs": list(reversed(logs))}


@app.get("/api/logs/stream")
async def stream_logs(request: Request, session_id: str | None = None, after_id: int = 0) -> StreamingResponse:
    principal = principal_from_request(request)

    async def events():
        last_id = after_id
        while True:
            data = get_logs(request, session_id=session_id, after_id=last_id)["logs"]
            for item in data:
                last_id = max(last_id, item["id"])
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(events(), media_type="text/event-stream")


# 二期·自然语言配置管家（intent_router）。独立 router，account-scoped，
# 端点内部复用 principal_from_request / effective_llm_settings / run_with_llm_gate（惰性导入规避循环）。
from . import intent_router as _intent_router  # noqa: E402

app.include_router(_intent_router.router)
