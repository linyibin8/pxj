"""Long-term user-memory layer (write-time extraction + semantic retrieval + scoring).

This is the "memory layer" that decides *which* durable facts about a student get
carried into the next QA turn's context. It is the in-house equivalent of frameworks
like Mem0/Letta, implemented directly on the stack this app already runs:

  * embeddings  -> the local TEI (bge-small-zh) server via `embeddings.py`
  * vector store -> the per-account SQLite `agent_memories` table (brute-force cosine,
                    same approach as `knowledge_vectors`; each account's set is small)
  * extraction  -> the shared LLM, called through the *background* gate so it never
                   competes with realtime QA on the concurrency-1 lane

Retrieval is fully transparent: every returned memory carries the exact score
breakdown (semantic / recency / importance / usage) so the client debug panel and the
web dashboard can show *why* each memory was selected.

The module intentionally has no dependency on `main.py` (to avoid an import cycle):
the caller injects an `llm_call` coroutine for extraction so routing through the LLM
gate stays in `main.py`.
"""
from __future__ import annotations

import asyncio
import json
import math
import uuid
from datetime import datetime, timezone

from . import embeddings
from .config import get_settings
from .db import connect, utc_now

# ---------------------------------------------------------------------------
# Tunables — the retrieval scoring weights are the heart of "which memories get
# carried into context". Keep them here, visible and in one place.
# ---------------------------------------------------------------------------
KIND_VALUES = ("preference", "mistake", "goal", "habit", "fact")

W_SEMANTIC = 0.60       # how close the memory is to the current question
W_RECENCY = 0.15        # newer memories slightly preferred
W_IMPORTANCE = 0.20     # importance assigned at write time
W_USAGE = 0.05          # memories that have proven useful before

RECENCY_HALF_LIFE_DAYS = 21.0   # recency score halves every 3 weeks
USAGE_SATURATION = 5            # use_count beyond this no longer raises the score

DEFAULT_TOP_K = 5
DEFAULT_MIN_SCORE = 0.30        # drop weakly related memories entirely
DEDUPE_COSINE = 0.90           # >= this vs an existing memory => update, not insert
MAX_MEMORIES_PER_ACCOUNT = 400  # soft cap; lowest-value memories pruned past this
MEMORY_TEXT_LIMIT = 400


def _account_id(account_id: str = "") -> str:
    return account_id or get_settings().default_account_id or "local"


# M2: serialize the read-modify-write dedupe section per account so two QA turns
# finishing nearly simultaneously can't both miss an existing memory and each
# insert a near-duplicate. The LLM/embedding calls stay OUTSIDE this lock so the
# extract pipeline's slow stages never block on it.
_account_locks: dict[str, asyncio.Lock] = {}


def _lock_for(account_id: str) -> asyncio.Lock:
    lock = _account_locks.get(account_id)
    if lock is None:
        lock = asyncio.Lock()
        _account_locks[account_id] = lock
    return lock


def _parse_ts(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _recency_score(updated_at: str, *, now: datetime) -> float:
    ts = _parse_ts(updated_at)
    if ts is None:
        return 0.0
    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
    return math.exp(-age_days / RECENCY_HALF_LIFE_DAYS)


def _usage_score(use_count: int) -> float:
    return min(max(use_count, 0), USAGE_SATURATION) / USAGE_SATURATION


def _decode_embedding(raw: str) -> list[float]:
    try:
        vec = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return vec if isinstance(vec, list) else []


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Retrieval  (read path)
# ---------------------------------------------------------------------------
async def retrieve(
    query: str,
    *,
    account_id: str = "",
    k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    mark_used: bool = True,
    exclude_ids: set[str] = frozenset(),
) -> list[dict]:
    """Return the top-k durable memories for `query`, each with a score breakdown.

    This is the single source of truth for "which memories enter this turn's context".
    """
    account_id = _account_id(account_id)
    query = (query or "").strip()
    if not query or not embeddings.embed_enabled():
        return []
    try:
        qvec = await embeddings.embed_text(query[:1000])
    except Exception:
        return []
    if not qvec:
        return []

    now = datetime.now(timezone.utc)
    scored: list[dict] = []
    with connect(account_id) as conn:
        rows = conn.execute(
            "SELECT id, kind, text, embedding, importance, use_count, updated_at "
            "FROM agent_memories WHERE account_id=? AND status='active'",
            (account_id,),
        ).fetchall()

    for row in rows:
        vec = _decode_embedding(row["embedding"])
        if not vec:
            continue
        semantic = _clamp01(embeddings.cosine(qvec, vec))
        recency = _recency_score(row["updated_at"], now=now)
        importance = _clamp01(float(row["importance"] or 0.0))
        usage = _usage_score(int(row["use_count"] or 0))
        final = (
            W_SEMANTIC * semantic
            + W_RECENCY * recency
            + W_IMPORTANCE * importance
            + W_USAGE * usage
        )
        scored.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "text": row["text"],
                "score": round(final, 4),
                "breakdown": {
                    "semantic": round(semantic, 4),
                    "recency": round(recency, 4),
                    "importance": round(importance, 4),
                    "usage": round(usage, 4),
                },
            }
        )

    scored.sort(key=lambda m: m["score"], reverse=True)
    selected = [m for m in scored if m["score"] >= min_score][: max(1, min(k, 50))]
    if exclude_ids:
        # Drop client-excluded memories *before* mark_used so their use_count is not
        # incremented for a turn the user opted out of (MUST-3 side-effect fix).
        selected = [m for m in selected if m["id"] not in exclude_ids]

    if mark_used and selected:
        ids = [m["id"] for m in selected]
        now_iso = utc_now()
        with connect(account_id) as conn:
            conn.executemany(
                "UPDATE agent_memories SET use_count=use_count+1, last_used_at=? WHERE id=?",
                [(now_iso, mid) for mid in ids],
            )
    return selected


# Persona kinds are "who this student is": carried into context regardless of the
# current question (always-on personalization), unlike mistakes/facts which are pulled
# only when semantically relevant. Mirrors the core-vs-recall split in Mem0/Letta.
PERSONA_KINDS = ("preference", "goal", "habit")


def persona_memories(*, account_id: str = "", n: int = 3) -> list[dict]:
    """The always-on personalization slice: top preferences/goals/habits by
    importance then recency, independent of the current question."""
    account_id = _account_id(account_id)
    placeholders = ",".join("?" * len(PERSONA_KINDS))
    with connect(account_id) as conn:
        rows = conn.execute(
            f"SELECT id, kind, text, importance FROM agent_memories "
            f"WHERE account_id=? AND status='active' AND kind IN ({placeholders}) "
            f"ORDER BY importance DESC, updated_at DESC LIMIT ?",
            (account_id, *PERSONA_KINDS, max(1, min(n, 20))),
        ).fetchall()
    return [
        {
            "id": r["id"],
            "kind": r["kind"],
            "text": r["text"],
            "score": None,
            "breakdown": {"reason": "persona_always_on", "importance": round(float(r["importance"] or 0.0), 4)},
        }
        for r in rows
    ]


async def retrieve_for_turn(
    query: str,
    *,
    account_id: str = "",
    recall_k: int = DEFAULT_TOP_K,
    persona_n: int = 2,
    exclude_ids: set[str] = frozenset(),
) -> list[dict]:
    """What actually goes into a QA turn: the always-on persona slice + the
    semantically-recalled memories, persona first, de-duplicated by id.

    `exclude_ids` are memories the client opted out of for this turn: they are
    filtered from both the persona slice and the recalled slice, and (via
    `retrieve`) are never marked used. Defaults to no exclusions (back-compat)."""
    persona = [m for m in persona_memories(account_id=account_id, n=persona_n) if m["id"] not in exclude_ids]
    recalled = await retrieve(query, account_id=account_id, k=recall_k, exclude_ids=exclude_ids)
    seen = {m["id"] for m in persona}
    return persona + [m for m in recalled if m["id"] not in seen]


# ---------------------------------------------------------------------------
# Write path  (extraction + dedupe + store)
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """你是学习陪伴 App 的「记忆抽取器」。从下面这一轮师生互动中，抽取出对**今后**继续陪伴这个学生有用的、稳定的事实。

只抽取持久、可复用的信息，例如：
- preference：学生的学习偏好（喜欢先提示、需要分步、不喜欢直接给答案等）
- mistake：反复出现的卡点/易错点（如「分数约分常忘记先求最大公约数」）
- goal：近期学习目标（如「准备期末数学」「想提高应用题」）
- habit：学习习惯（如「做题爱跳步」「容易粗心看错符号」）
- fact：其它对长期陪伴有用的稳定事实（年级、学科重点等）

不要抽取：这道题的具体答案、一次性的临时内容、与学生无关的客套话。
如果没有任何值得长期记住的信息，返回空数组 []。

严格只输出 JSON 数组，每个元素形如：
{"text": "一句话事实（中文，不超过60字）", "kind": "preference|mistake|goal|habit|fact", "importance": 0.0~1.0}

本轮互动：
学生问题：{question}
助教回答：{answer}
本轮偏好/反馈：{feedback}
"""


def _safe_json_array(raw: str) -> list[dict]:
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except (TypeError, ValueError):
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def _normalize_kind(kind: str) -> str:
    kind = (kind or "").strip().lower()
    return kind if kind in KIND_VALUES else "fact"


async def extract_and_store(
    *,
    question: str,
    answer: str,
    feedback: str = "",
    account_id: str = "",
    source_event_id: str = "",
    llm_call,
) -> list[dict]:
    """Extract durable memories from one QA exchange and persist them.

    `llm_call(prompt: str) -> str` is injected by the caller so the LLM request can be
    routed through the background gate. Returns the list of stored/updated memories.
    Designed to run in a background task — never on the QA response path.
    """
    account_id = _account_id(account_id)
    question = (question or "").strip()
    answer = (answer or "").strip()
    if len(question) < 2 or not embeddings.embed_enabled():
        return []

    prompt = (
        EXTRACTION_PROMPT
        .replace("{question}", question[:800])
        .replace("{answer}", answer[:1200])
        .replace("{feedback}", (feedback or "无")[:400])
    )
    try:
        raw = await llm_call(prompt)
    except Exception:
        return []
    candidates = _safe_json_array(raw)
    if not candidates:
        return []

    texts = []
    cleaned = []
    for item in candidates:
        text = str(item.get("text") or "").strip()[:MEMORY_TEXT_LIMIT]
        if len(text) < 4:
            continue
        try:
            importance = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        cleaned.append(
            {"text": text, "kind": _normalize_kind(item.get("kind")), "importance": _clamp01(importance)}
        )
        texts.append(text)
    if not cleaned:
        return []

    try:
        vectors = await embeddings.embed_texts(texts)
    except Exception:
        vectors = []
    if len(vectors) != len(cleaned):
        return []

    stored: list[dict] = []
    now = utc_now()
    # M2: only the DB read-modify-write is serialized per account; the LLM call and
    # embedding call above already completed outside this lock.
    async with _lock_for(account_id):
        with connect(account_id) as conn:
            existing = conn.execute(
                "SELECT id, embedding, importance, use_count FROM agent_memories "
                "WHERE account_id=? AND status='active'",
                (account_id,),
            ).fetchall()
            existing_vecs = [(r["id"], _decode_embedding(r["embedding"]), r) for r in existing]

            for item, vec in zip(cleaned, vectors):
                # Dedupe by embedding similarity rather than a second LLM call: cheap, and
                # essential given the shared concurrency-1 LLM.
                best_id, best_sim = "", 0.0
                for mid, mvec, _ in existing_vecs:
                    if not mvec:
                        continue
                    sim = embeddings.cosine(vec, mvec)
                    if sim > best_sim:
                        best_id, best_sim = mid, sim
                if best_id and best_sim >= DEDUPE_COSINE:
                    # M1: do NOT overwrite source_event_id on a dedupe-UPDATE — keep the
                    # original provenance so a stale memory isn't re-attributed to this turn.
                    conn.execute(
                        "UPDATE agent_memories SET text=?, kind=?, embedding=?, "
                        "importance=MAX(importance, ?), updated_at=? WHERE id=?",
                        (item["text"], item["kind"], json.dumps(vec), item["importance"], now, best_id),
                    )
                    stored.append({"id": best_id, "op": "update", **item})
                else:
                    mem_id = uuid.uuid4().hex
                    conn.execute(
                        "INSERT INTO agent_memories(id, account_id, kind, text, embedding, importance, "
                        "status, source_event_id, use_count, created_at, updated_at, last_used_at) "
                        "VALUES(?, ?, ?, ?, ?, ?, 'active', ?, 0, ?, ?, '')",
                        (mem_id, account_id, item["kind"], item["text"], json.dumps(vec), item["importance"], source_event_id, now, now),
                    )
                    existing_vecs.append((mem_id, vec, None))
                    stored.append({"id": mem_id, "op": "add", **item})

            _prune(conn, account_id)
    return stored


def _prune(conn, account_id: str) -> None:
    """Keep the active set bounded: past the cap, retire the lowest-value memories
    (lowest importance, then oldest, then least used)."""
    count = conn.execute(
        "SELECT COUNT(*) AS c FROM agent_memories WHERE account_id=? AND status='active'",
        (account_id,),
    ).fetchone()["c"]
    if count <= MAX_MEMORIES_PER_ACCOUNT:
        return
    overflow = count - MAX_MEMORIES_PER_ACCOUNT
    victims = conn.execute(
        "SELECT id FROM agent_memories WHERE account_id=? AND status='active' "
        "ORDER BY importance ASC, use_count ASC, updated_at ASC LIMIT ?",
        (account_id, overflow),
    ).fetchall()
    for v in victims:
        conn.execute("UPDATE agent_memories SET status='superseded' WHERE id=?", (v["id"],))


# ---------------------------------------------------------------------------
# Inspection helpers (for the web dashboard / debug panel / tests)
# ---------------------------------------------------------------------------
def list_memories(*, account_id: str = "", kind: str = "", limit: int = 200) -> list[dict]:
    account_id = _account_id(account_id)
    sql = (
        "SELECT id, kind, text, importance, use_count, created_at, updated_at, last_used_at "
        "FROM agent_memories WHERE account_id=? AND status='active'"
    )
    params: list = [account_id]
    if kind:
        sql += " AND kind=?"
        params.append(kind)
    sql += " ORDER BY updated_at DESC LIMIT ?"
    params.append(max(1, min(limit, 1000)))
    with connect(account_id) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def stats(*, account_id: str = "") -> dict:
    account_id = _account_id(account_id)
    with connect(account_id) as conn:
        total = conn.execute(
            "SELECT COUNT(*) AS c FROM agent_memories WHERE account_id=? AND status='active'",
            (account_id,),
        ).fetchone()["c"]
        by_kind = {
            r["kind"]: r["c"]
            for r in conn.execute(
                "SELECT kind, COUNT(*) AS c FROM agent_memories WHERE account_id=? AND status='active' GROUP BY kind",
                (account_id,),
            ).fetchall()
        }
    return {"total": total, "by_kind": by_kind, "embed_enabled": embeddings.embed_enabled()}


# ---------------------------------------------------------------------------
# Phase 3: rolling-memory deltas + user-facing correction / soft-delete / restore.
# These are the SINGLE write entry points for status/text mutations (the iOS
# "我的学习档案" page and the per-turn delta chip both route through here), so the
# soft-delete invariant (status active/superseded, never hard delete) lives in one
# place. account_id is always part of the WHERE clause for cross-account isolation.
# ---------------------------------------------------------------------------
_MEMORY_PUBLIC_COLUMNS = (
    "id, kind, text, importance, status, source_event_id, use_count, "
    "created_at, updated_at, last_used_at"
)


def _memory_row(conn, memory_id: str, account_id: str) -> dict | None:
    row = conn.execute(
        f"SELECT {_MEMORY_PUBLIC_COLUMNS} FROM agent_memories WHERE id=? AND account_id=?",
        (memory_id, account_id),
    ).fetchone()
    return dict(row) if row else None


def set_status(memory_id: str, status: str, *, account_id: str = "") -> dict | None:
    """Soft-delete / restore: flip agent_memories.status. Never hard-deletes.

    Returns the updated row, or None if the memory does not exist for this account.
    """
    account_id = _account_id(account_id)
    status = (status or "").strip().lower()
    if status not in ("active", "superseded"):
        return None
    now = utc_now()
    with connect(account_id) as conn:
        cur = conn.execute(
            "UPDATE agent_memories SET status=?, updated_at=? WHERE id=? AND account_id=?",
            (status, now, memory_id, account_id),
        )
        if cur.rowcount == 0:
            return None
        return _memory_row(conn, memory_id, account_id)


async def update_text(memory_id: str, text: str, *, account_id: str = "") -> dict | None:
    """Correct a memory's text and re-embed it in the SAME transaction (M5).

    Embedding is computed BEFORE the write; if it fails or is empty we raise and
    never touch the row, so text and embedding can never drift apart. Returns the
    updated row, or None if the memory does not exist for this account.
    """
    account_id = _account_id(account_id)
    text = (text or "").strip()[:MEMORY_TEXT_LIMIT]
    if len(text) < 2:
        raise ValueError("memory text too short")
    if not embeddings.embed_enabled():
        raise RuntimeError("embeddings disabled; cannot re-embed corrected memory")
    vectors = await embeddings.embed_texts([text])
    if not vectors or not vectors[0]:
        raise RuntimeError("re-embed failed for corrected memory")
    vec = vectors[0]
    now = utc_now()
    with connect(account_id) as conn:
        cur = conn.execute(
            "UPDATE agent_memories SET text=?, embedding=?, updated_at=? WHERE id=? AND account_id=?",
            (text, json.dumps(vec), now, memory_id, account_id),
        )
        if cur.rowcount == 0:
            return None
        return _memory_row(conn, memory_id, account_id)


def write_deltas(rows: list[dict], *, account_id: str = "") -> int:
    """Batch-insert per-turn memory deltas (best-effort, fire-and-forget from the
    extract task). `op`/`kind`/`text` come from the authoritative `stored` list."""
    account_id = _account_id(account_id)
    if not rows:
        return 0
    now = utc_now()
    payload = [
        (
            uuid.uuid4().hex,
            account_id,
            str(r.get("qa_event_id") or ""),
            str(r.get("memory_id") or ""),
            (str(r.get("op") or "add") or "add"),
            _normalize_kind(r.get("kind")),
            str(r.get("text") or "")[:MEMORY_TEXT_LIMIT],
            now,
        )
        for r in rows
    ]
    with connect(account_id) as conn:
        conn.executemany(
            "INSERT INTO memory_deltas(id, account_id, qa_event_id, memory_id, op, kind, text, seen, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, 0, ?)",
            payload,
        )
    return len(payload)


def recent_deltas(
    *, account_id: str = "", unseen_only: bool = True, since: str = "", limit: int = 50
) -> list[dict]:
    """Per-turn delta feed for the in-chat "这次更了解你了" chip (passive pull)."""
    account_id = _account_id(account_id)
    sql = (
        "SELECT id, memory_id, op, kind, text, qa_event_id, created_at "
        "FROM memory_deltas WHERE account_id=?"
    )
    params: list = [account_id]
    if unseen_only:
        sql += " AND seen=0"
    if since:
        sql += " AND created_at > ?"
        params.append(since)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(max(1, min(limit, 200)))
    with connect(account_id) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def mark_deltas_seen(ids: list[str], *, account_id: str = "") -> int:
    """Best-effort debounce (M8): mark deltas seen so the chip isn't re-shown.
    Not an idempotency guarantee — the UI tolerates an occasional repeat."""
    account_id = _account_id(account_id)
    ids = [i for i in (ids or []) if i]
    if not ids:
        return 0
    placeholders = ",".join("?" * len(ids))
    with connect(account_id) as conn:
        cur = conn.execute(
            f"UPDATE memory_deltas SET seen=1 WHERE account_id=? AND id IN ({placeholders})",
            (account_id, *ids),
        )
    return cur.rowcount


async def restore_guard(memory_id: str, *, account_id: str = "") -> dict:
    """Validate a soft-deleted memory can be restored to active (M6).

    Returns {"ok": True} if restore is safe, else {"ok": False, "reason": ...}:
      - "missing":   memory does not exist for this account
      - "capacity":  restoring would exceed MAX_MEMORIES_PER_ACCOUNT
      - "duplicate": a near-duplicate active memory already exists (>= DEDUPE_COSINE)
    The actual flip to status=active stays in set_status; this only gates it.
    """
    account_id = _account_id(account_id)
    with connect(account_id) as conn:
        target = conn.execute(
            "SELECT id, text, embedding, status FROM agent_memories WHERE id=? AND account_id=?",
            (memory_id, account_id),
        ).fetchone()
        if target is None:
            return {"ok": False, "reason": "missing"}
        if target["status"] == "active":
            # Already active: restore is a no-op, allow it.
            return {"ok": True}
        active_count = conn.execute(
            "SELECT COUNT(*) AS c FROM agent_memories WHERE account_id=? AND status='active'",
            (account_id,),
        ).fetchone()["c"]
        if active_count + 1 > MAX_MEMORIES_PER_ACCOUNT:
            return {"ok": False, "reason": "capacity"}
        target_vec = _decode_embedding(target["embedding"])
        active = conn.execute(
            "SELECT id, embedding FROM agent_memories WHERE account_id=? AND status='active'",
            (account_id,),
        ).fetchall()
    if target_vec:
        for r in active:
            if r["id"] == memory_id:
                continue
            mvec = _decode_embedding(r["embedding"])
            if not mvec:
                continue
            if embeddings.cosine(target_vec, mvec) >= DEDUPE_COSINE:
                return {"ok": False, "reason": "duplicate"}
    return {"ok": True}
