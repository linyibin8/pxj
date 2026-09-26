import asyncio
import base64
import hmac
import hashlib
import html
import json
import re
import shutil
import sqlite3
import time
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

app = FastAPI(title="鐭ヨ繘鎷嶅")
AUTH_SCHEME = "Bearer"
AUTH_PASSWORD_MIN_LENGTH = 8
AUTH_DEFAULT_TOKEN_BYTES = 32
AUTH_HASH_ITERATIONS = 210_000
AUTH_TOKEN_ALGORITHM = "HS256"
AUTH_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PROFILE_TYPES = {"student", "parent", "teacher"}
MODEL_PROVIDERS = {
    "evowit": {"label": "EvoWit 榛樿妯″瀷", "supports_custom_base_url": True, "openai_compatible": True,
               "default_base_url": "http://100.64.0.5:39000/v1", "default_model": "evowit-agent27b",
               "key_hint": "榛樿浣跨敤鎴戜滑鎻愪緵鐨勬ā鍨嬶紝鏃犻渶濉啓 Key"},
    "openai": {"label": "OpenAI", "supports_custom_base_url": True, "openai_compatible": True,
               "default_base_url": "https://api.openai.com/v1", "default_model": "gpt-4o-mini",
               "key_hint": "濉啓浣犵殑 OpenAI API Key锛坰k-...锛?},
    "anthropic": {"label": "Anthropic (Claude)", "supports_custom_base_url": True, "openai_compatible": False,
                  "default_base_url": "https://api.anthropic.com", "default_model": "claude-sonnet-4-5",
                  "via_gateway": True,
                  "key_hint": "Anthropic 鍘熺敓鏍煎紡锛岃缁忕綉鍏虫帴鍏ワ紙瑙佷笅鏂圭綉鍏筹級"},
    "gemini": {"label": "Google Gemini", "supports_custom_base_url": True, "openai_compatible": True,
               "default_base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "default_model": "gemini-2.0-flash",
               "key_hint": "濉啓浣犵殑 Google AI Studio API Key"},
    "zhipu": {"label": "鏅鸿氨 AI (GLM)", "supports_custom_base_url": True, "openai_compatible": True,
              "default_base_url": "https://open.bigmodel.cn/api/paas/v4", "default_model": "glm-4-flash",
              "key_hint": "濉啓浣犵殑鏅鸿氨 API Key"},
    "openai-compatible": {"label": "OpenAI-compatible", "supports_custom_base_url": True, "openai_compatible": True,
                          "default_base_url": "", "default_model": "",
                          "key_hint": "浠绘剰鍏煎 OpenAI /chat/completions 鐨勬湇鍔?},
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
# 姣忎釜缁忚繃 llm_gate 鐨勪换鍔＄殑娉ㄥ唽琛紙鎸?task_id锛夛紝鐢ㄤ簬銆屾寜璐﹀彿鏌ョ湅鍦ㄨ窇鐨勫悗鍙颁换鍔″苟鍙栨秷銆嶃€?
# 鍙?llm_gate_lock 淇濇姢銆傝褰曪細account_id/user_id/label/lane/session_id/state/created/cancel_requested/future銆?
llm_tasks: dict[str, dict] = {}
task_dispatcher_task: asyncio.Task | None = None


class LLMTaskCancelled(Exception):
    """鐢ㄦ埛鍦ㄤ换鍔＄瓑寰呮垨杩愯闃舵涓诲姩鍙栨秷锛堣处鍙峰唴鍙栨秷鑷繁鐨勫悗鍙颁换鍔★級銆?""


def task_display_title(label: str) -> str:
    text = (label or "").lower()
    table = (
        ("teaching_visualization", "鐢熸垚鍙鍖栬瑙?),
        ("final_report", "鐢熸垚瀛︿範鎶ュ憡"),
        ("question_extraction_session", "鎻愬彇瑙傚療棰樼洰"),
        ("qa_session_summary", "鐢熸垚闂瓟灏忕粨"),
        ("memory_consolidation", "鏁寸悊闀挎湡璁板繂"),
        ("memory_extract", "鏁寸悊闀挎湡璁板繂"),
        ("distill", "鎻愮偧瀛︿範瑕佺偣"),
        ("vision", "鍒嗘瀽棰樼洰鐢婚潰"),
        ("profile", "鏇存柊瀛︿範鐢诲儚"),
        ("observation", "鏅鸿兘瑙傚療鍒嗘瀽"),
    )
    for key, name in table:
        if key in text:
            return name
    return "鍚庡彴鐢熸垚浠诲姟"
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
OBSERVATION_ANALYSIS_MAX_IMAGES = 3
OBSERVATION_LOOKBACK_LIMIT = 80
QUESTION_CROP_BATCH_SIZE = 6
QUESTION_CROP_MIN_WIDTH_PX = 420
QUESTION_CROP_MIN_HEIGHT_PX = 240
QUESTION_CROP_MIN_AREA_PX = 160_000
QUESTION_CROP_THIN_ASPECT_RATIO = 6.0
QUESTION_CROP_EXPAND_MIN_WIDTH_RATIO = 0.72
QUESTION_CROP_EXPAND_MIN_HEIGHT_RATIO = 0.28
QUESTION_CROP_EXPAND_MAX_WIDTH_RATIO = 0.94
QUESTION_CROP_EXPAND_MAX_HEIGHT_RATIO = 0.52
QUESTION_CROP_EXPANDED_QUALITY = 90
VISUAL_DUPLICATE_DISTANCE = 3.2
VISUAL_TEXT_DUPLICATE_DISTANCE = 5.2
TEXT_DUPLICATE_DISTANCE = 0.22
# Keyframe override for 鏅鸿兘瑙傚療 dedup: even when the frame is visually a near
# duplicate (camera barely moved), keep it as new content if the OCR token set
# changed meaningfully (the student wrote a new step). Guards against dropping the
# one frame that actually carries new information.
KEYFRAME_TEXT_CHANGE_DISTANCE = 0.45  # Jaccard distance of text tokens vs the matched previous frame
KEYFRAME_MIN_NEW_TOKENS = 3  # require this many genuinely-new tokens (filters single-token OCR noise)
# 璇煶鎶撴媿蹇瓫: an eligible follow-up frame this close (visually) to the previous QA
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
LLM_PRIORITY_QUESTION_EXTRACTION = 30
LLM_PRIORITY_BACKGROUND = 100
TASK_PRIORITY_QUESTION_EXTRACTION = 30
TASK_PRIORITY_BACKGROUND = 100
TASK_PRIORITY_FINAL_REPORT = 120
TASK_PRIORITY_MEMORY = 150
TASK_STALE_RUNNING_SECONDS_BY_KIND = {
    "question_extraction_session": 1800,
    "final_report": 1200,
}
MEMORY_CONSOLIDATION_INTERVAL_SECONDS = 3600
MEMORY_EVENT_TEXT_LIMIT = 1200
MEMORY_PROFILE_CHAR_LIMIT = 4000
MEMORY_PROFILE_RECENT_EVENT_LIMIT = 80
# "candidate" = 鏅鸿兘瑙傚療寮傛鎻愬彇鍑虹殑"鍙枒閿欓鍊欓€?锛屽皻鏈繘姝ｅ紡閿欓鏈?澶嶄範闃熷垪锛涘鐢熺‘璁?import)
# 鍚庤浆 confirmed 鎵嶅叆鏈紝蹇界暐(dismiss)杞?ignored銆傚畠琚埢鎰忔帓闄ゅ湪 ACTIVE/REVIEW_QUEUE 涔嬪銆?
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
    "鏈瘑鍒?,
    "鐪嬩笉娓?,
    "鐪嬩笉娓呮",
    "涓嶆竻妤?,
    "妯＄硦",
    "澶皬",
    "琚伄鎸?,
    "閬尅",
    "鍙嶅厜",
    "鏃犳硶纭",
    "鏃犳硶杈ㄨ",
    "鏃犳硶璇嗗埆",
    "鏈兘璇嗗埆",
    "璇嗗埆涓嶆竻",
    "涓嶆竻鏅?,
    "璇讳笉娓?,
    "鏃犳硶璇诲彇",
    "闅句互杈ㄨ",
    "鏃犳硶鍒ゆ柇",
    "鏈畬鏁村叆闀?,
    "娌℃湁瀹屾暣鏀捐繘",
    "娌℃湁瀹屾暣杩涘叆",
    "鏈畬鏁磋繘鍏?,
    "鐢婚潰澶?,
    "瑙嗛噹澶?,
    "瓒呭嚭鐢婚潰",
    "瓒呭嚭瑙嗛噹",
    "杈圭紭琚鍒?,
    "琚鍒?,
    "鎴柇",
    "涓嶅畬鏁?,
    "鎷嶆憚涓嶅畬鏁?,
    "鍙媿鍒颁竴閮ㄥ垎",
    "鍙媿鍒板眬閮?,
    "閮ㄥ垎鍙",
    "鍙湁涓€鍗?,
    "涓讳綋鍋忓嚭鐢婚潰",
    "鍐呭缂哄け",
    "鏈媿鍏?,
    "娌℃媿鍏?,
    "娌℃媿瀹屾暣",
    "璋冩暣鐩告満",
    "瀹屾暣鏀捐繘鐢婚潰",
    "瀹屾暣鏀捐繘鎷嶆憚鍖哄煙",
)


@app.on_event("startup")
def startup() -> None:
    global task_dispatcher_task
    init_db()
    settings = get_settings()
    app.mount("/images", StaticFiles(directory=settings.data_dir / "images"), name="images")
    for account_id in list_account_ids():
        set_current_account(account_id)
        recover_interrupted_tasks(force=True)
    set_current_account(settings.default_account_id or DEFAULT_ACCOUNT_ID)
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


def recover_interrupted_tasks(*, force: bool = False) -> None:
    now = datetime.now(timezone.utc)
    now_text = now.isoformat()
    try:
        with connect() as conn:
            conn.execute(
                """
                UPDATE task_runs
                SET priority=?,
                    attempts=CASE WHEN attempts >= max_attempts THEN MAX(0, max_attempts - 1) ELSE attempts END,
                    updated_at=?
                WHERE task_kind='question_extraction_session'
                  AND status IN ('queued', 'running')
                """,
                (TASK_PRIORITY_QUESTION_EXTRACTION, now_text),
            )
            if force:
                conn.execute(
                    """
                    UPDATE task_runs
                    SET status='queued',
                        attempts=CASE WHEN attempts >= max_attempts THEN MAX(0, max_attempts - 1) ELSE attempts END,
                        available_at=?, updated_at=?, last_error='recovered after process restart'
                    WHERE status='running'
                    """,
                    (now_text, now_text),
                )
            else:
                for task_kind, seconds in TASK_STALE_RUNNING_SECONDS_BY_KIND.items():
                    stale_before = (now - timedelta(seconds=seconds)).isoformat()
                    conn.execute(
                        """
                        UPDATE task_runs
                        SET status='queued',
                            attempts=CASE WHEN attempts >= max_attempts THEN MAX(0, max_attempts - 1) ELSE attempts END,
                            available_at=?, updated_at=?, last_error='recovered after long-running task timeout'
                        WHERE status='running' AND task_kind=? AND (started_at='' OR started_at < ?)
                        """,
                        (now_text, now_text, task_kind, stale_before),
                    )
                stale_before = (now - timedelta(seconds=TASK_STALE_RUNNING_SECONDS)).isoformat()
                known_kinds = tuple(TASK_STALE_RUNNING_SECONDS_BY_KIND.keys())
                if known_kinds:
                    placeholders = ",".join("?" for _ in known_kinds)
                    conn.execute(
                        f"""
                        UPDATE task_runs
                        SET status='queued',
                            attempts=CASE WHEN attempts >= max_attempts THEN MAX(0, max_attempts - 1) ELSE attempts END,
                            available_at=?, updated_at=?, last_error='recovered after running task timeout'
                        WHERE status='running'
                          AND task_kind NOT IN ({placeholders})
                          AND (started_at='' OR started_at < ?)
                        """,
                        (now_text, now_text, *known_kinds, stale_before),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE task_runs
                        SET status='queued',
                            attempts=CASE WHEN attempts >= max_attempts THEN MAX(0, max_attempts - 1) ELSE attempts END,
                            available_at=?, updated_at=?, last_error='recovered after running task timeout'
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
        display_name = {"student": "榛樿瀛︾敓", "parent": "瀹堕暱", "teacher": "鑰佸笀"}[profile_type]
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
        "name": "EvoWit 榛樿妯″瀷",
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
            sources = [item for item in payload.get("sources") or [] if isinstance(item, dict)]
            await run_session_question_extraction(session_id, filenames, task_id=task_id, sources=sources)
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
                f"浠婃棩鍏嶈垂棰濆害宸茬敤瀹岋紙{quota['used']}/{quota['limit']} 娆★級銆傚彲鍦ㄣ€岃处鍙蜂笌妯″瀷銆嶉噷閰嶇疆浣犺嚜宸辩殑澶фā鍨嬬户缁娇鐢紝鎴栨槑鏃ヨ嚜鍔ㄦ仮澶嶃€?,
            )
    warned = False
    idle_notified = False
    waiter_id = uuid.uuid4().hex
    registered_waiter = False
    acquired_slot = False   # 鍙栨秷璇箟锛氭嬁鍒版斁琛屾Ы鍓嶅悗娓呯悊 llm_tasks 鐨勪綅缃笉鍚?
    usage_event_id = ""
    usage_started_at: float | None = None
    started_waiting = asyncio.get_running_loop().time()
    # 娉ㄥ唽鍒颁换鍔¤〃锛堢瓑寰呴樁娈靛氨鍙銆佸彲鍙栨秷锛夈€?
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
                # 绛夊緟闃舵琚彇娑堬細鐩存帴涓锛堜笉鍗犵敤鏀捐妲斤級銆?
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
                            f"LLM 璇锋眰鏀捐锛歿label}锛涗紭鍏堢骇={lane}锛涘苟鍙?{llm_gate_inflight}/{max_concurrency} "
                            f"(瀹炴椂={llm_gate_inflight_realtime} 鍚庡彴={llm_gate_inflight_background})锛?
                            f"鎺掗槦={counts['total']} (瀹炴椂={counts['realtime']} 鍚庡彴={counts['background']})锛?
                            f"鏈€灏忛棿闅?{min_interval:.1f}s{'锛屽疄鏃惰姹傚凡缁曡繃闂撮殧' if lane == 'realtime' else ''}"
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
                            f"LLM 璇锋眰鎺掗槦锛歿label}锛涗紭鍏堢骇={lane}锛涘綋鍓嶅苟鍙?{llm_gate_inflight}锛?
                            f"鎺掗槦={counts['total']} (瀹炴椂={counts['realtime']} 鍚庡彴={counts['background']})锛?
                            f"闄愬埗={max_concurrency}锛岀瓑寰呮ā鍨嬬┖闂?
                        ),
                        session_id=session_id,
                        level="warning",
                    )
                if wait_for_idle and not idle_notified:
                    idle_notified = True
                    emit_llm_gate_log(
                        f"鍚庡彴浠诲姟璁╄瀹炴椂璇煶锛歿label} 灏嗗湪妯″瀷绌洪棽鏃剁敓鎴愶紙璇煶/闂瓟浼樺厛锛?,
                        session_id=session_id,
                    )
            await asyncio.sleep(wait_seconds)
    except LLMTaskCancelled:
        # 绛夊緟闃舵琚彇娑堬細鏈崰鏀捐妲斤紝娓呯悊绛夊緟鑰呬笌浠诲姟璁板綍鍚庝腑姝€?
        async with llm_gate_lock:
            llm_gate_waiters.pop(waiter_id, None)
            llm_gate_waiting = len(llm_gate_waiters)
            llm_tasks.pop(waiter_id, None)
        emit_llm_gate_log(f"鍚庡彴浠诲姟宸插彇娑堬紙绛夊緟涓級锛歿label}", session_id=session_id, level="warning")
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
                # 鐢ㄧ嫭绔?future 鍖呰９鏈璋冪敤锛屼究浜庛€岃繍琛岄樁娈靛彇娑堛€嶇洿鎺?cancel() 鎵撴柇銆?
                inner = asyncio.ensure_future(call())
                rec = llm_tasks.get(waiter_id)
                if rec is not None:
                    rec["future"] = inner
                    if rec.get("cancel_requested"):
                        inner.cancel()   # 鍙栨秷璇锋眰钀藉湪銆屾嬁鍒版Ы浣崀鍒涘缓 future銆嶇獥鍙ｅ唴鏃惰ˉ鍒€
                result = await inner
                break
            except asyncio.CancelledError:
                record_llm_usage_finish(usage_event_id, "cancelled", error="cancelled by user", started_monotonic=usage_started_at)
                emit_llm_gate_log(f"鍚庡彴浠诲姟宸插彇娑堬紙杩愯涓級锛歿label}", session_id=session_id, level="warning")
                raise LLMTaskCancelled()
            except Exception:
                if attempt < retries:
                    attempt += 1
                    emit_llm_gate_log(
                        f"瀹炴椂璇锋眰澶辫触锛岃嚜鍔ㄩ噸璇?{attempt}/{retries}锛歿label}",
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
    marker = f"\n...[宸叉埅鏂害 {omitted} 瀛梋...\n"
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


def json_object_string(value: object, fallback: object | None = None) -> str:
    if isinstance(value, (dict, list)):
        return json_dumps(value)
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(parsed, (dict, list)):
                    return json_dumps(parsed)
    if fallback is not None:
        return json_dumps(fallback)
    return "{}"


def parse_question_crop_manifest_payload(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {"crops": data}
    return {}


def question_crop_manifest_items(data: object) -> list[dict]:
    if isinstance(data, dict):
        data = data.get("crops") or data.get("question_crops") or data.get("items") or []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def parse_question_crop_manifest(raw: str) -> list[dict]:
    return question_crop_manifest_items(parse_question_crop_manifest_payload(raw))


def question_crop_key_is_weak(question_key: str, strength: str = "") -> bool:
    key = str(question_key or "").strip().lower()
    value = str(strength or "").strip().lower()
    return key.startswith("layout:") or value.startswith("weak") or value in {"layout", "layout_weak"}


def question_crop_row_safety(row: dict) -> dict:
    value = row.get("crop_safety")
    if isinstance(value, dict):
        return value
    return question_crop_safety_from_normalized_rect(row.get("normalized_rect"))


def question_crop_row_quality_score(row: dict, image_dir: Path) -> float:
    score = 0.0
    try:
        confidence = float(row.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    score += max(0.0, min(1.0, confidence)) * 2.0

    crop_size = size_tuple_from_payload(row.get("crop_image_size"))
    if crop_size:
        width, height = crop_size
        score += min(width, 1600) / 1600
        score += min(height, 1200) / 1200
        score += min(width * height, 1_600_000) / 1_600_000

    rect = json_object_value(row.get("crop_rect"))
    area = (numeric_value(rect, "width", "w") or 0) * (numeric_value(rect, "height", "h") or 0)
    if 0.08 <= area <= 0.68:
        score += 1.0
    elif 0.035 <= area < 0.08 or 0.68 < area <= 0.78:
        score += 0.25
    elif area > 0:
        score -= 0.75

    source = str(row.get("source") or "").strip()
    if source in {"server_rect_crop", "server_rect_expanded", "server_expanded"}:
        score += 0.35
    if source in {"server_rect_expanded", "server_expanded"}:
        score += 0.15

    safety = question_crop_row_safety(row)
    if safety.get("error"):
        score -= 2.0
    if safety.get("expanded"):
        score += 0.15

    filename = str(row.get("crop_filename") or "").strip()
    if filename:
        path = image_dir / filename
        if path.exists():
            score += 0.5
            try:
                score += min(path.stat().st_size, 600_000) / 1_200_000
            except OSError:
                pass
        else:
            score -= 1.5
    return round(score, 6)


def question_crop_new_row_is_better(new_row: dict, existing_row: dict, image_dir: Path) -> bool:
    new_score = question_crop_row_quality_score(new_row, image_dir)
    existing_score = question_crop_row_quality_score(existing_row, image_dir)
    if new_score > existing_score + 0.08:
        return True
    if existing_score > new_score + 0.08:
        return False
    try:
        new_conf = float(new_row.get("confidence") or 0)
    except (TypeError, ValueError):
        new_conf = 0.0
    try:
        old_conf = float(existing_row.get("confidence") or 0)
    except (TypeError, ValueError):
        old_conf = 0.0
    if new_conf > old_conf + 0.08:
        return True
    new_size = size_tuple_from_payload(new_row.get("crop_image_size")) or (0, 0)
    old_size = size_tuple_from_payload(existing_row.get("crop_image_size")) or (0, 0)
    return new_size[0] * new_size[1] > old_size[0] * old_size[1] * 1.20


def remove_question_crop_file(image_dir: Path, filename: str) -> None:
    clean = str(filename or "").strip()
    if not clean:
        return
    for path in (image_dir / clean, thumbnail_path_for(clean)):
        try:
            if path.exists():
                path.unlink()
        except Exception:
            pass


def replace_question_crop_row(conn, existing_row: dict, new_row: dict, now: str) -> bool:
    params = (
        new_row["batch_id"],
        new_row["image_id"],
        new_row["sequence_index"],
        new_row["manifest_index"],
        new_row["question_index"],
        new_row["question_key"],
        new_row["fingerprint"],
        new_row["crop_hash"],
        new_row["text_hash"],
        new_row["normalized_rect"],
        new_row["crop_rect"],
        new_row["source_image_size"],
        new_row["crop_image_size"],
        new_row["preview_text"],
        new_row["crop_filename"],
        new_row["original_name"],
        new_row["status"],
        new_row["source"],
        new_row["confidence"],
        now,
        existing_row["id"],
        existing_row["session_id"],
    )
    try:
        conn.execute(
            """
            UPDATE session_question_crops
            SET batch_id=?, image_id=?, sequence_index=?, manifest_index=?,
                question_index=?, question_key=?, fingerprint=?, crop_hash=?, text_hash=?,
                normalized_rect=?, crop_rect=?, source_image_size=?, crop_image_size=?,
                preview_text=?, crop_filename=?, original_name=?, status=?, source=?,
                confidence=?, updated_at=?
            WHERE id=? AND session_id=?
            """,
            params,
        )
        return True
    except sqlite3.IntegrityError:
        conn.execute(
            """
            UPDATE session_question_crops
            SET question_index=?, question_key=?, fingerprint=?, crop_hash=?, text_hash=?,
                normalized_rect=?, crop_rect=?, source_image_size=?, crop_image_size=?,
                preview_text=?, crop_filename=?, original_name=?, status=?, source=?,
                confidence=?, updated_at=?
            WHERE id=? AND session_id=?
            """,
            (
                new_row["question_index"],
                new_row["question_key"],
                new_row["fingerprint"],
                new_row["crop_hash"],
                new_row["text_hash"],
                new_row["normalized_rect"],
                new_row["crop_rect"],
                new_row["source_image_size"],
                new_row["crop_image_size"],
                new_row["preview_text"],
                new_row["crop_filename"],
                new_row["original_name"],
                new_row["status"],
                new_row["source"],
                new_row["confidence"],
                now,
                existing_row["id"],
                existing_row["session_id"],
            ),
        )
        return True


def crop_manifest_int(item: dict, *keys: str) -> int | None:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def crop_manifest_float(item: dict, *keys: str) -> float | None:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        return number if number == number else None
    return None


def crop_manifest_str_list(item: dict, *keys: str) -> list[str]:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            return [truncate_text(str(part), 80) for part in value if str(part).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [truncate_text(str(part), 80) for part in parsed if str(part).strip()]
            return [truncate_text(part.strip(), 80) for part in text.split(",") if part.strip()]
    return []


def json_object_value(value: object) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return dict(parsed)
    return {}


def numeric_value(item: dict, *keys: str) -> float | None:
    for key in keys:
        value = item.get(key)
        if value is None or value == "":
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number:
            return number
    return None


def int_value(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def image_row_allows_question_crop(row: dict) -> bool:
    return bool(row.get("filename")) and row.get("novelty_status") not in {"duplicate", "invalid"}


def size_payload(width: float | int, height: float | int) -> dict:
    return {"width": int(round(width)), "height": int(round(height))}


def size_tuple_from_payload(value: object) -> tuple[int, int] | None:
    payload = json_object_value(value)
    width = numeric_value(payload, "width", "w")
    height = numeric_value(payload, "height", "h")
    if width is None or height is None or width <= 0 or height <= 0:
        return None
    return int(round(width)), int(round(height))


def image_size_for_path(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            return image.size
    except Exception:
        return None


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def question_crop_rect_to_pixels(
    crop_rect: object,
    source_image_size: object,
    actual_image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    rect = json_object_value(crop_rect)
    image_width, image_height = actual_image_size
    x = numeric_value(rect, "x", "left", "minX", "min_x")
    y = numeric_value(rect, "y", "top", "minY", "min_y")
    width = numeric_value(rect, "width", "w")
    height = numeric_value(rect, "height", "h")
    if x is None or y is None or width is None or height is None or width <= 0 or height <= 0:
        return None

    source_size = size_tuple_from_payload(source_image_size) or actual_image_size
    source_width, source_height = source_size
    normalized = max(abs(x), abs(y), abs(width), abs(height), abs(x + width), abs(y + height)) <= 1.5
    if normalized:
        left = x * image_width
        top = y * image_height
        rect_width = width * image_width
        rect_height = height * image_height
    else:
        scale_x = image_width / max(1, source_width)
        scale_y = image_height / max(1, source_height)
        left = x * scale_x
        top = y * scale_y
        rect_width = width * scale_x
        rect_height = height * scale_y

    left = max(0.0, min(float(image_width), left))
    top = max(0.0, min(float(image_height), top))
    right = max(left + 1.0, min(float(image_width), left + rect_width))
    bottom = max(top + 1.0, min(float(image_height), top + rect_height))
    if right <= left or bottom <= top:
        return None
    return (
        max(0, int(left)),
        max(0, int(top)),
        min(image_width, int(right + 0.999)),
        min(image_height, int(bottom + 0.999)),
    )


def question_crop_rect_payload_from_box(box: tuple[int, int, int, int], image_size: tuple[int, int]) -> dict:
    image_width, image_height = image_size
    left, top, right, bottom = box
    return {
        "x": round(left / max(1, image_width), 6),
        "y": round(top / max(1, image_height), 6),
        "width": round(max(0, right - left) / max(1, image_width), 6),
        "height": round(max(0, bottom - top) / max(1, image_height), 6),
        "coordinate_space": "source_image_normalized",
    }


def question_crop_expansion_reasons(
    crop_size: tuple[int, int] | None,
    source_size: tuple[int, int],
    crop_box: tuple[int, int, int, int],
) -> list[str]:
    source_width, source_height = source_size
    box_width = max(1, crop_box[2] - crop_box[0])
    box_height = max(1, crop_box[3] - crop_box[1])
    crop_width, crop_height = crop_size or (box_width, box_height)
    min_width = min(
        max(QUESTION_CROP_MIN_WIDTH_PX, int(source_width * 0.32)),
        max(1, int(source_width * 0.55)),
    )
    min_height = min(QUESTION_CROP_MIN_HEIGHT_PX, max(1, int(source_height * 0.22)))
    min_area = min(QUESTION_CROP_MIN_AREA_PX, max(1, int(source_width * source_height * 0.10)))
    reasons: list[str] = []
    if crop_size is None:
        reasons.append("unreadable_client_crop")
    if crop_width < min_width:
        reasons.append("too_narrow")
    if crop_height < min_height:
        reasons.append("too_short")
    if crop_width * crop_height < min_area:
        reasons.append("area_too_small")
    aspect_ratio = max(crop_width / max(1, crop_height), crop_height / max(1, crop_width))
    if aspect_ratio >= QUESTION_CROP_THIN_ASPECT_RATIO:
        reasons.append("thin_strip")
    return reasons


def fit_interval(start: float, length: float, limit: int) -> tuple[int, int]:
    length = max(1.0, min(float(limit), length))
    start = max(0.0, min(float(limit) - length, start))
    end = min(float(limit), start + length)
    return int(start), int(end + 0.999)


def expanded_question_crop_box(
    crop_box: tuple[int, int, int, int],
    source_size: tuple[int, int],
    reasons: list[str],
) -> tuple[int, int, int, int]:
    source_width, source_height = source_size
    left, top, right, bottom = crop_box
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    target_width = float(crop_width)
    target_height = float(crop_height)
    reason_set = set(reasons)
    if reason_set & {"too_narrow", "area_too_small", "thin_strip", "unreadable_client_crop"}:
        target_width = max(target_width, crop_width * 1.8)
    if "too_narrow" in reason_set:
        target_width = max(target_width, source_width * QUESTION_CROP_EXPAND_MIN_WIDTH_RATIO)
    if "thin_strip" in reason_set and crop_width < source_width * 0.55:
        target_width = max(target_width, source_width * QUESTION_CROP_EXPAND_MIN_WIDTH_RATIO)
    if reason_set & {"too_short", "area_too_small", "thin_strip", "unreadable_client_crop"}:
        target_height = max(target_height, crop_height * 3.2, source_height * QUESTION_CROP_EXPAND_MIN_HEIGHT_RATIO)
    if "too_short" in reason_set:
        target_height = max(target_height, source_height * QUESTION_CROP_EXPAND_MIN_HEIGHT_RATIO)

    target_width = min(target_width, source_width * QUESTION_CROP_EXPAND_MAX_WIDTH_RATIO)
    target_height = min(target_height, source_height * QUESTION_CROP_EXPAND_MAX_HEIGHT_RATIO)
    extra_width = max(0.0, target_width - crop_width)
    extra_height = max(0.0, target_height - crop_height)
    fitted_left, fitted_right = fit_interval(left - extra_width * 0.5, target_width, source_width)
    # Bias extra vertical context downward: answers and figures usually follow the stem line.
    fitted_top, fitted_bottom = fit_interval(top - extra_height * 0.35, target_height, source_height)
    return fitted_left, fitted_top, fitted_right, fitted_bottom


def save_expanded_question_crop(source_path: Path, target_path: Path, box: tuple[int, int, int, int]) -> tuple[int, int]:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image)
        cropped = image.crop(box)
        if cropped.mode in ("RGBA", "LA"):
            background = Image.new("RGB", cropped.size, (255, 255, 255))
            background.paste(cropped.convert("RGB"), mask=cropped.getchannel("A"))
            cropped = background
        elif cropped.mode != "RGB":
            cropped = cropped.convert("RGB")
        out = BytesIO()
        cropped.save(out, format="JPEG", quality=QUESTION_CROP_EXPANDED_QUALITY, optimize=True)
        crop_size = cropped.size
    temp = target_path.with_name(f".{target_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_bytes(out.getvalue())
        temp.replace(target_path)
    finally:
        if temp.exists():
            temp.unlink()
    return crop_size


def create_thumbnail_safe(path: Path, filename: str, session_id: str) -> None:
    try:
        create_thumbnail(path, thumbnail_path_for(filename))
    except Exception as exc:
        emit_log(
            f"question crop thumbnail failed: {truncate_text(str(exc), 160)}",
            session_id=session_id,
            level="warning",
        )


def question_crop_trace_payload(
    normalized_rect: object,
    *,
    client_crop_rect: object,
    crop_safety: dict,
) -> dict:
    payload = json_object_value(normalized_rect)
    if json_object_value(client_crop_rect):
        payload.setdefault("client_crop_rect", json_object_value(client_crop_rect))
    payload["crop_safety"] = crop_safety
    return payload


def maybe_expand_question_crop_file(
    *,
    session_id: str,
    batch_id: str,
    crop_id: str,
    image_dir: Path,
    client_crop_filename: str,
    client_crop_path: Path,
    source_image_filename: str,
    crop_rect: object,
    source_image_size: object,
    crop_image_size: object,
    normalized_rect: object,
    client_source: str,
    crop_hash: str,
) -> dict:
    started = time.perf_counter()
    client_crop_size = image_size_for_path(client_crop_path) or size_tuple_from_payload(crop_image_size)
    source_image_size_payload = size_tuple_from_payload(source_image_size)
    result = {
        "filename": client_crop_filename,
        "path": client_crop_path,
        "source": "client_crop",
        "crop_rect": json_object_value(crop_rect),
        "source_image_size": size_payload(*(source_image_size_payload or (0, 0))) if source_image_size_payload else json_object_value(source_image_size),
        "crop_image_size": size_payload(*(client_crop_size or (0, 0))) if client_crop_size else json_object_value(crop_image_size),
        "crop_hash": crop_hash or "",
        "expanded": False,
        "normalized_rect": json_object_value(normalized_rect),
        "crop_safety": {},
    }
    source_image_path = image_dir / Path(source_image_filename or "").name
    actual_source_size = image_size_for_path(source_image_path) if source_image_filename else None
    crop_box = question_crop_rect_to_pixels(crop_rect, source_image_size, actual_source_size) if actual_source_size else None
    reason_source_size = actual_source_size or source_image_size_payload
    reason_box = crop_box
    if reason_box is None and client_crop_size:
        reason_box = (0, 0, max(1, client_crop_size[0]), max(1, client_crop_size[1]))
    reasons = question_crop_expansion_reasons(client_crop_size, reason_source_size, reason_box) if reason_source_size and reason_box else []
    crop_safety = {
        "source": result["source"],
        "client_source": truncate_text(client_source or "", 80),
        "expanded": False,
        "reason": reasons,
        "client_crop_filename": client_crop_filename,
        "client_crop_image_size": result["crop_image_size"],
        "client_crop_rect": json_object_value(crop_rect),
        "source_image_size": size_payload(*(actual_source_size or source_image_size_payload or (0, 0))),
        "duration_ms": 0,
    }
    if reasons and actual_source_size and crop_box:
        expanded_box = expanded_question_crop_box(crop_box, actual_source_size, reasons)
        original_area = max(1, (crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1]))
        expanded_area = max(1, (expanded_box[2] - expanded_box[0]) * (expanded_box[3] - expanded_box[1]))
        if expanded_area > original_area:
            expanded_filename = f"{session_id}_{batch_id}_qcrop_{crop_id}_expanded.jpg"
            expanded_path = image_dir / expanded_filename
            try:
                expanded_size = save_expanded_question_crop(source_image_path, expanded_path, expanded_box)
                expanded_rect = question_crop_rect_payload_from_box(expanded_box, actual_source_size)
                crop_safety.update(
                    {
                        "source": "server_expanded",
                        "expanded": True,
                        "expanded_crop_filename": expanded_filename,
                        "expanded_crop_rect": expanded_rect,
                        "expanded_crop_image_size": size_payload(*expanded_size),
                        "original_crop_box_px": {
                            "left": crop_box[0],
                            "top": crop_box[1],
                            "right": crop_box[2],
                            "bottom": crop_box[3],
                        },
                        "expanded_crop_box_px": {
                            "left": expanded_box[0],
                            "top": expanded_box[1],
                            "right": expanded_box[2],
                            "bottom": expanded_box[3],
                        },
                    }
                )
                result.update(
                    {
                        "filename": expanded_filename,
                        "path": expanded_path,
                        "source": "server_expanded",
                        "crop_rect": expanded_rect,
                        "source_image_size": size_payload(*actual_source_size),
                        "crop_image_size": size_payload(*expanded_size),
                        "crop_hash": file_sha1(expanded_path),
                        "expanded": True,
                    }
                )
            except Exception as exc:
                crop_safety["error"] = truncate_text(str(exc), 180)
    elif reasons and (not actual_source_size or not crop_box):
        crop_safety["error"] = "source_image_or_crop_rect_unavailable"

    if not result["crop_hash"] and result["path"].exists():
        try:
            result["crop_hash"] = file_sha1(result["path"])
        except Exception:
            result["crop_hash"] = ""
    crop_safety["source"] = result["source"]
    crop_safety["duration_ms"] = max(0, int((time.perf_counter() - started) * 1000))
    result["crop_safety"] = crop_safety
    result["normalized_rect"] = question_crop_trace_payload(
        normalized_rect,
        client_crop_rect=crop_rect,
        crop_safety=crop_safety,
    )
    return result


def save_question_crop_from_source_rect(
    *,
    session_id: str,
    batch_id: str,
    crop_id: str,
    image_dir: Path,
    source_image_filename: str,
    crop_rect: object,
    source_image_size: object,
    normalized_rect: object,
    client_source: str,
) -> dict:
    started = time.perf_counter()
    source_image_size_payload = size_tuple_from_payload(source_image_size)
    filename = f"{session_id}_{batch_id}_qcrop_{crop_id}_rect.jpg"
    target = image_dir / filename
    source_image_path = image_dir / Path(source_image_filename or "").name
    actual_source_size = image_size_for_path(source_image_path) if source_image_filename else None
    result = {
        "filename": filename,
        "path": target,
        "source": "server_rect_crop",
        "crop_rect": json_object_value(crop_rect),
        "source_image_size": size_payload(*(actual_source_size or source_image_size_payload or (0, 0))),
        "crop_image_size": {},
        "crop_hash": "",
        "expanded": False,
        "normalized_rect": json_object_value(normalized_rect),
        "crop_safety": {},
    }
    crop_safety = {
        "source": result["source"],
        "client_source": truncate_text(client_source or "", 80),
        "transfer_mode": "rect_only",
        "expanded": False,
        "reason": [],
        "client_crop_filename": "",
        "client_crop_image_size": {},
        "client_crop_rect": json_object_value(crop_rect),
        "source_image_size": result["source_image_size"],
        "duration_ms": 0,
    }

    crop_box = question_crop_rect_to_pixels(crop_rect, source_image_size, actual_source_size) if actual_source_size else None
    if not actual_source_size or not source_image_path.is_file():
        crop_safety["error"] = "source_image_unavailable"
    elif not crop_box:
        crop_safety["error"] = "crop_rect_unavailable"
    else:
        box_width = max(1, crop_box[2] - crop_box[0])
        box_height = max(1, crop_box[3] - crop_box[1])
        reasons = question_crop_expansion_reasons((box_width, box_height), actual_source_size, crop_box)
        crop_safety["reason"] = reasons
        crop_safety["original_crop_box_px"] = {
            "left": crop_box[0],
            "top": crop_box[1],
            "right": crop_box[2],
            "bottom": crop_box[3],
        }
        final_box = crop_box
        if reasons:
            expanded_box = expanded_question_crop_box(crop_box, actual_source_size, reasons)
            original_area = max(1, box_width * box_height)
            expanded_area = max(1, (expanded_box[2] - expanded_box[0]) * (expanded_box[3] - expanded_box[1]))
            if expanded_area > original_area:
                final_box = expanded_box
                crop_safety.update(
                    {
                        "source": "server_rect_expanded",
                        "expanded": True,
                        "expanded_crop_box_px": {
                            "left": expanded_box[0],
                            "top": expanded_box[1],
                            "right": expanded_box[2],
                            "bottom": expanded_box[3],
                        },
                    }
                )
        try:
            crop_size = save_expanded_question_crop(source_image_path, target, final_box)
            final_rect = question_crop_rect_payload_from_box(final_box, actual_source_size)
            crop_safety.update(
                {
                    "expanded_crop_rect": final_rect,
                    "expanded_crop_image_size": size_payload(*crop_size),
                }
            )
            result.update(
                {
                    "source": crop_safety["source"],
                    "crop_rect": final_rect,
                    "source_image_size": size_payload(*actual_source_size),
                    "crop_image_size": size_payload(*crop_size),
                    "crop_hash": file_sha1(target),
                    "expanded": bool(crop_safety.get("expanded")),
                }
            )
        except Exception as exc:
            crop_safety["error"] = truncate_text(str(exc), 180)

    crop_safety["source"] = result["source"]
    crop_safety["duration_ms"] = max(0, int((time.perf_counter() - started) * 1000))
    result["crop_safety"] = crop_safety
    result["normalized_rect"] = question_crop_trace_payload(
        normalized_rect,
        client_crop_rect=crop_rect,
        crop_safety=crop_safety,
    )
    return result


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
            return [part.strip() for part in re.split(r"[,锛孿s]+", value) if part.strip()]
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
        "detail": "鐢婚潰鍖呭惈瀛︿範鏉愭枡绾跨储",
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
            "detail": "鐢婚潰鏃犳槑纭涔犳潗鏂欙紝宸茶嚜鍔ㄥ拷鐣?,
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
                "detail": f"鐢婚潰缂哄皯鍙敤绾搁潰/灞忓箷绾圭悊 edge={edge_density:.3f} contrast={contrast:.1f}",
            }
        if blur_score is not None and blur_score < IMAGE_VALIDITY_BLUR_MIN_WITHOUT_TEXT and weak_text_count == 0:
            return {
                **payload,
                "valid": False,
                "reason": "too_blurry",
                "detail": "鐢婚潰妯＄硦涓旀病鏈夊彲鐢ㄦ枃瀛楃嚎绱紝宸茶嚜鍔ㄥ拷鐣?,
            }
        return {
            **payload,
            "valid": None,
            "reason": "unknown",
            "detail": "鐢婚潰瀛︿範浠峰€间笉纭畾锛屼繚瀹堜繚鐣?,
        }

    return {
        **payload,
        "valid": None,
        "reason": "unknown",
        "detail": "缂哄皯瓒冲鐨勭敾闈㈡湁鏁堟€у厓鏁版嵁锛屼繚瀹堜繚鐣?,
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
            "detail": "瀛︾敓璇锋眰鏍稿/璁㈡锛屽綋鍓嶇敾闈㈠惈瀛︿範鏉愭枡璇佹嵁",
            "text_tokens": text_tokens,
        }
    if eligible is False and client_submitted_for_qa is not True:
        return {
            "eligible": False,
            "reason": "client_rejected",
            "detail": ";".join(reasons) or "瀹㈡埛绔垽鏂綋鍓嶇敾闈笉閫傚悎鍋氶棶绛斾笂涓嬫枃",
            "text_tokens": text_tokens,
        }
    if reliable_qa_context is False and not has_review_material_evidence:
        return {
            "eligible": False,
            "reason": "no_explicit_study_evidence",
            "detail": "褰撳墠鎶撴媿缂哄皯鏂囧瓧銆侀鐩竟妗嗘垨灞忓箷鐗瑰緛",
            "text_tokens": text_tokens,
        }
    if has_explicit_evidence is False and not has_review_material_evidence:
        return {
            "eligible": False,
            "reason": "no_explicit_study_evidence",
            "detail": "褰撳墠鎶撴媿缂哄皯鏄庣‘瀛︿範鏉愭枡璇佹嵁",
            "text_tokens": text_tokens,
        }
    if (text_count is not None or rectangle_count is not None) and not has_reliable_structure and not has_review_material_evidence:
        return {
            "eligible": False,
            "reason": "no_explicit_study_evidence",
            "detail": "褰撳墠鎶撴媿娌℃湁瓒冲棰樼洰鏂囧瓧鎴栫焊寮?灞忓箷缁撴瀯锛屽凡娌跨敤宸叉湁涓婁笅鏂?,
            "text_tokens": text_tokens,
        }
    if eligible is True:
        return {
            "eligible": True,
            "reason": "client_accepted",
            "detail": "瀹㈡埛绔垽鏂綋鍓嶇敾闈㈤€傚悎鍋氶棶绛斾笂涓嬫枃",
            "text_tokens": text_tokens,
        }
    if has_review_material_evidence:
        return {
            "eligible": True,
            "reason": "visual_review_material",
            "detail": "瀛︾敓璇锋眰鏍稿/璁㈡锛屽綋鍓嶇敾闈㈠惈瀛︿範鏉愭枡璇佹嵁",
            "text_tokens": text_tokens,
        }
    if has_material is False:
        return {
            "eligible": False,
            "reason": "no_study_material",
            "detail": "瀹㈡埛绔湭妫€娴嬪埌璇炬湰銆佽瘯鍗锋垨鐢靛瓙灞忓箷",
            "text_tokens": text_tokens,
        }
    if len(text_tokens) >= QA_MIN_TEXT_FOR_CONTEXT:
        return {"eligible": True, "reason": "ocr_tokens", "detail": "妫€娴嬪埌鍙敤鏂囧瓧", "text_tokens": text_tokens}
    if len(text_tokens) >= QA_MIN_TEXT_WITH_RECTANGLE_FOR_CONTEXT and weak_rectangle_count >= 1:
        return {"eligible": True, "reason": "ocr_tokens_with_structure", "detail": "妫€娴嬪埌鏂囧瓧鍜岄鐩粨鏋?, "text_tokens": text_tokens}
    if len(text_tokens) > 0:
        return {
            "eligible": False,
            "reason": "weak_ocr_evidence",
            "detail": "褰撳墠鎶撴媿鏂囧瓧绾跨储涓嶈冻锛屽凡娌跨敤宸叉湁涓婁笅鏂?,
            "text_tokens": text_tokens,
        }
    if material_confidence is not None and material_confidence >= QA_IMAGE_MATERIAL_CONFIDENCE_MIN:
        if weak_rectangle_count >= QA_MIN_RECTANGLES_WITHOUT_TEXT_FOR_CONTEXT:
            return {"eligible": True, "reason": "material_confidence", "detail": f"瀛︿範鏉愭枡缃俊搴?{material_confidence:.2f}", "text_tokens": text_tokens}
        return {
            "eligible": False,
            "reason": "weak_visual_evidence",
            "detail": "褰撳墠鎶撴媿鍙湁寮辫瑙夌壒寰侊紝宸叉部鐢ㄥ凡鏈変笂涓嬫枃",
            "text_tokens": text_tokens,
        }
    if has_material is True and (blur_score is None or blur_score >= 0.25):
        if weak_rectangle_count >= QA_MIN_RECTANGLES_WITHOUT_TEXT_FOR_CONTEXT or len(text_tokens) >= QA_MIN_TEXT_WITH_RECTANGLE_FOR_CONTEXT:
            return {"eligible": True, "reason": "material_visible", "detail": "瀹㈡埛绔娴嬪埌瀛︿範鏉愭枡鍙", "text_tokens": text_tokens}
        return {
            "eligible": False,
            "reason": "weak_visual_evidence",
            "detail": "褰撳墠鎶撴媿缂哄皯鍙敤棰樼洰淇℃伅锛屽凡娌跨敤宸叉湁涓婁笅鏂?,
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
                "detail": f"鍥惧儚绾圭悊/浜害鍙敤 edge={edge_density:.3f} contrast={contrast:.1f}",
                "text_tokens": text_tokens,
            }
        return {
            "eligible": False,
            "reason": "image_metrics_low",
            "detail": f"鍥惧儚缂哄皯娓呮櫚绾搁潰/灞忓箷鐗瑰緛 edge={edge_density:.3f} contrast={contrast:.1f}",
            "text_tokens": text_tokens,
        }
    return {"eligible": None, "reason": "unknown", "detail": "缂哄皯瓒冲鐨勭敾闈㈣川閲忓厓鏁版嵁", "text_tokens": text_tokens}


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
        "detail": quality.get("detail") or "鏃犳硶纭褰撳墠鐢婚潰鏄惁鍖呭惈娓呮櫚棰樼洰",
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
        r"绗琝s*[\d涓€浜屼笁鍥涗簲鍏竷鍏節鍗乚+\s*[棰橀爜椤礭",
        r"[\d涓€浜屼笁鍥涗簲鍏竷鍏節鍗乚+\s*[棰橀爜椤礭",
        r"problem\s*\d+",
        r"question\s*\d+",
        r"page\s*\d+",
        r"鎹涓€鍊嬩釜]?棰?,
        r"涓嬩竴棰?,
        r"涓婁竴棰?,
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
        "鏀瑰畬",
        "鏀瑰ソ浜?,
        "鏀瑰ソ",
        "淇敼",
        "鏀逛簡",
        "璁㈡",
        "淇",
        "閲嶅啓",
        "閲嶆柊鍐?,
        "鍐欏畬",
        "鍋氬畬",
        "绠楀畬",
        "濉畬",
        "鏀规",
    )
    check_terms = (
        "鐪嬬湅",
        "鐪嬩笅",
        "鐪嬩竴涓?,
        "鍐嶇湅",
        "妫€鏌?,
        "鏍稿",
        "鎵规敼",
        "瀵逛笉瀵?,
        "瀵逛簡鍚?,
        "瀵逛簡娌?,
        "鏄惁姝ｇ‘",
        "鏄笉鏄",
        "鏈夋病鏈夐敊",
        "杩橀敊",
        "杩樺",
        "姝ｇ‘鍚?,
        "鍙互鍚?,
        "琛屼笉琛?,
    )
    answer_terms = ("绛旀", "缁撴灉", "姝ラ", "杩囩▼", "杩欓亾", "杩欓", "杩欓噷", "杩欐")
    visual_terms = ("鐪?, "鍥剧墖", "鐓х墖", "鎷嶇収", "鐢婚潰", "闀滃ご", "鍥句笂", "杩欏紶")
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
    followup_terms = ("缁х画", "鍒氭墠", "杩欓噷", "杩欎釜", "杩欐", "涓轰粈涔?, "鍝噷閿?, "鍐嶈", "涓嶆噦", "杩介棶")
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
            "detail": "杩介棶鎶撴媿鏂囧瓧绾跨储涓嶈冻锛屽凡娌跨敤涓婁竴杞笂涓嬫枃",
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
            "detail": f"杩介棶鐢婚潰涓庝笂涓€寮犲涔犵敾闈㈡枃瀛楃嚎绱笉鍖归厤 overlap={overlap:.2f}",
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
    return any(marker in detail for marker in ("鏂囧瓧", "棰樼洰", "缁撴瀯", "瀛︿範鏉愭枡", "璇炬湰", "璇曞嵎", "灞忓箷"))


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
                parts.append(f"鏉ユ簮锛歿title}")
            if summary:
                parts.append(f"鎽樿锛歿summary}")
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
            or "浠婃棩澶嶄範閿欓"
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
                "title": truncate_text(title or "鐩稿叧閿欓", 60),
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
    # 鍙敤鍙楀紑鍏崇害鏉熺殑 dynamic_strategy 鍒ゅ畾 included锛泂trategy_context 鎭掑湪椤跺眰涓嶈兘浣滀緷鎹?鍚﹀垯鍏崇瓥鐣ヤ粛璇姤)
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
        f"鏈疆瑙﹀彂锛歿trigger_type}锛泃urn={context.get('turn') or client_dynamic.get('turn') or 'unknown'}锛沬ntent={context.get('student_intent') or 'unknown'}",
        f"鐢ㄦ埛褰撳墠闂锛歿truncate_text(question, 180)}",
        f"鍋忓ソ锛氬満鏅?{context.get('learning_mode_title') or strategy_context.get('learning_mode_title') or '鏈寚瀹?}锛涘洖绛旀柟寮?{context.get('coach_depth_title') or strategy_context.get('coach_depth_title') or '鏈寚瀹?}",
        f"褰撳墠鐢婚潰锛歮ode={image_context_mode}锛泂elected_image_id={(image_row or {}).get('id') or context.get('selected_image_id') or 'none'}锛泂ubmitted={context.get('current_frame_submitted')}",
        f"鐢婚潰璐ㄩ噺锛歟ligible={frame_quality.get('qa_context_eligible') or frame_quality.get('eligible')}锛泂tudy_material={frame_quality.get('has_study_material')}锛泂ummary={truncate_text(frame_quality.get('signal_summary') or frame_quality.get('message') or '', 180)}",
        f"瑙傚療鐘舵€侊細active={observation.get('is_active') or context.get('is_observing')}锛沠rames={observation.get('buffered_frame_count') or 0}锛泂tate={observation.get('upload_state') or ''}",
    ]
    if context.get("student_goal") or session.get("student_goal"):
        lines.append(f"鏈疆鐩爣锛歿truncate_text(context.get('student_goal') or session.get('student_goal') or '', 240)}")
    if context.get("carried_history_context"):
        lines.append(f"鍘嗗彶涓婁笅鏂囷細{truncate_text(context.get('carried_history_context'), 360)}")
    if context.get("review_context"):
        lines.append(f"澶嶄範涓婁笅鏂囷細{truncate_text(context.get('review_context'), 360)}")
    if formed_memories:
        lines.append(
            "閲嶇偣褰㈡垚璁板繂锛?
            + "锛?.join(truncate_text(event.get("text"), 150) for event in formed_memories)
        )
    if agent_memories:
        lines.append(
            "鐩稿叧璁板繂锛堟寜璇箟+鏂拌繎搴?閲嶈鎬ф绱級锛?
            + "锛?.join(
                truncate_text(m.get("text"), 120)
                for m in agent_memories[:5]
                if isinstance(m, dict) and m.get("text")
            )
        )
    elif memory_candidates:
        lines.append("璁板繂鏁寸悊鍊欓€夛細" + "锛?.join(truncate_text(item, 120) for item in memory_candidates[:4]))
    if client_dynamic:
        lines.append(f"瀹㈡埛绔姩鎬佺瓥鐣ワ細{truncate_text(json_dumps(client_dynamic), 700)}")
    lines.append("鎵ц锛氱患鍚堟湰杞亸濂姐€佸綋鍓嶇敾闈€佸綋鍓嶅洖鍚堛€佽瀵熺姸鎬併€佸巻鍙蹭笂涓嬫枃鍜岃蹇嗘暣鐞嗭紱浼樺厛鍥炵瓟鐢ㄦ埛褰撳墠闂锛屼笉璁╂棫涓婁笅鏂囪鐩栨柊闂銆?)
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
        return "鏈煡"
    seconds = max(0, seconds)
    if seconds < 60:
        return f"{seconds:.0f} 绉?
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f} 鍒嗛挓"
    hours = minutes / 60
    return f"{hours:.1f} 灏忔椂"


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
    "manual": "浣犳墜鍔ㄥ姞鍏?,
    "photo_grading": "鎷嶉鎵规敼鏃惰嚜鍔ㄨ瘑鍒?,
    "observation": "鏅鸿兘瑙傚療瀛︿範鏃惰瘑鍒?,
    "qa": "闂瓟涓彂鐜?,
}
_MISTAKE_STATUS_LABELS = {
    "candidate": "鍊欓€?路 寰呭鍏?,
    "suspected": "鐤戜技 路 寰呯‘璁?,
    "incomplete": "鐤戜技鏈畬鎴?,
    "confirmed": "宸茬‘璁?,
    "corrected": "宸茶姝?,
    "mastered": "宸叉帉鎻?,
    "ignored": "宸插拷鐣?,
}


def build_mistake_provenance(item: dict) -> dict:
    """缁欐瘡涓€閬撻敊棰樼敓鎴?瀹冩槸鎬庝箞鏉ョ殑"璇存槑锛氫綍鏃?/ 鎬庝箞閲囬泦 / 涓轰粈涔堝垽涓洪敊 / 鍒ゅ畾渚濇嵁 / 缃俊搴︺€?
    鏃ф暟鎹病鏈?detection_method锛屽垯浠?source_summary 鎺ㄦ柇锛屼繚璇佹瘡閬撻閮借寰楁竻銆?""
    method = (item.get("detection_method") or "").strip()
    summary = item.get("source_summary") or ""
    if not method:
        if "鎵嬪姩" in summary:
            method = "manual"
        elif "瑙傚療" in summary:
            method = "observation"
        elif "闂瓟" in summary:
            method = "qa"
        elif summary:
            method = "photo_grading"
    method_label = _MISTAKE_METHOD_LABELS.get(method, "鑷姩鏁寸悊")
    status = item.get("status") or "suspected"
    student = (item.get("student_answer") or "").strip()
    expected = (item.get("expected_answer") or "").strip()
    reason = (item.get("error_reason") or "").strip()
    evidence = (item.get("evidence") or "").strip()
    # 鍒ゅ畾渚濇嵁锛氫紭鍏?浣滅瓟鈫斿弬鑰冪瓟妗堝姣?锛屽叾娆￠敊鍥?璇佹嵁銆?
    if status == "incomplete":
        basis = "杩欓亾棰樼暀绌轰簡銆佹病鏈変綔绛?
    elif student and expected and student != expected:
        basis = f"浣犵殑浣滅瓟涓庡弬鑰冪瓟妗堜笉涓€鑷达紙浣滅瓟锛歿truncate_text(student, 40)}锛涘弬鑰冿細{truncate_text(expected, 40)}锛?
    elif student:
        basis = f"鍩轰簬浣犵殑浣滅瓟鍒ゆ柇锛歿truncate_text(student, 60)}"
    elif reason and not is_placeholder_mistake_text(reason):
        basis = truncate_text(reason, 80)
    elif evidence:
        basis = truncate_text(evidence, 80)
    else:
        basis = "鏉ヨ嚜鎵规敼/瀛︿範璁板綍鐨勫垽鏂?
    why = reason if (reason and not is_placeholder_mistake_text(reason)) else basis
    return {
        "method": method,
        "method_label": method_label,
        "when": item.get("created_at") or item.get("first_seen_at") or "",
        "why": truncate_text(why, 120),
        "basis": basis,
        "confidence_label": _MISTAKE_STATUS_LABELS.get(status, "鐤戜技 路 寰呯‘璁?),
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
        "宸︽墜",
        "鍙虫墜",
        "鎵嬫寚",
        "鎵嬫帉",
        "鎻＄瑪",
        "鎸佺瑪",
        "绗斿皷",
        "涔﹀啓",
        "姝ｅ湪鍐?,
        "浜轰綋",
        "浜鸿劯",
        "韬綋",
        "澶撮儴",
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
        "鏈娴嬪埌瀛︾敓",
        "鏈娴嬪埌浜?,
        "鏈瀛︾敓",
        "鏈浜轰綋",
        "鏈鎵?,
        "鏃犱汉",
        "娌′汉",
        "瀛︾敓涓嶅湪",
        "涓嶅湪鍦?,
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
        return "鏈夊鐢?鎵?绗旇瘉鎹?
    if status == "absent":
        return "鏈娴嬪埌瀛︾敓/鐤戜技涓嶅湪鍦?
    return "鍦ㄥ満鏈瘑鍒?


def meta_activity_summary(meta: dict, fallback_text: str = "") -> str:
    device_presence = meta.get("device_interaction_presence") or meta.get("deviceInteractionPresence")
    if isinstance(device_presence, dict) and meta_bool(device_presence, "present", "is_present", "isPresent") is True:
        summary = meta_text(device_presence, "summary", "detail", "reason")
        operation = meta_text(device_presence, "operation", "kind")
        if summary:
            return truncate_text(summary, 120)
        if operation:
            return truncate_text(f"鐢ㄦ埛鎿嶄綔鎵嬫満锛歿operation}锛岃瘉鏄庣敤鎴峰湪鍦?, 120)
        return "鐢ㄦ埛鍒氭搷浣滆繃鎵嬫満锛岃瘉鏄庣敤鎴峰湪鍦?
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
    return truncate_text(fallback_text, 120) if fallback_text else "鏃?


def build_time_weight_summary(session: dict, images: list[dict]) -> str:
    ordered = sorted(images, key=image_sort_key)
    if not ordered:
        return "鏃堕棿鏉冮噸鎽樿锛氭棤鎶撴媿鍥剧墖锛屾棤娉曞垽鏂鐢熸槸鍚﹀湪鍦烘垨娲诲姩鎸佺画鏃堕暱銆?

    finish = parse_datetime(session.get("finished_at"))
    totals = {"present": 0.0, "absent": 0.0, "unknown": 0.0}
    intervals: list[dict] = []
    for index, row in enumerate(ordered):
        current = parse_datetime(row.get("captured_at")) or parse_datetime(row.get("created_at"))
        next_time = None
        next_label = "涓嬩竴寮?
        if index + 1 < len(ordered):
            next_row = ordered[index + 1]
            next_time = parse_datetime(next_row.get("captured_at")) or parse_datetime(next_row.get("created_at"))
        elif finish:
            next_time = finish
            next_label = "缁撴潫"
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
                    meta_activity_summary(meta, fallback_signal) + "锛涘熬娈垫棤鍚庣画鐢婚潰锛屼笉鑳界‘璁ゆ寔缁湪鍦?
                    if tail_has_weak_followup
                    else meta_activity_summary(meta, fallback_signal)
                ),
                "hints": "锛?.join(hints) if hints else "椤?棰樻湭璇嗗埆",
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
        tail_note = "锛涙渶鍚庝竴寮犲埌缁撴潫锛屾棤鍚庣画鐢婚潰璇佹嵁" if item["is_tail"] else ""
        longest_lines.append(
            (
                f"sequence={item['sequence_index']} 鍒皗item['next_label']}锛歿format_duration(item['seconds'])}锛?
                f"{presence_label(item['status'])}锛寋item['hints']}锛屾椿鍔ㄧ嚎绱?{item['activity']}{tail_note}"
            )
        )
    if not longest_lines:
        longest_lines.append("鏃犲彲璁＄畻闂撮殧銆?)
    tail_gap = "鏈煡"
    if intervals and intervals[-1]["is_tail"]:
        tail_gap = format_duration(intervals[-1]["seconds"])
    total_observed = sum(totals.values())
    uncertain_seconds = totals.get("absent", 0.0) + totals.get("unknown", 0.0)
    uncertain_ratio = uncertain_seconds / total_observed if total_observed > 0 else 0
    quality_note = "鍦ㄥ満璇佹嵁鍏呰冻锛屽彲鎸変富瑕佸姩浣滄浼扮畻銆?
    if uncertain_ratio >= 0.5:
        quality_note = "鐤戜技涓嶅湪鍦?鍦ㄥ満鏈瘑鍒椂娈靛崰姣旇緝楂橈紝棰樼洰鑰楁椂鍙兘浣庣疆淇″害浼拌銆?
    elif uncertain_ratio >= 0.25:
        quality_note = "瀛樺湪杈冨鍦ㄥ満鏈瘑鍒椂娈碉紝棰樼洰鑰楁椂闇€鏍囨敞鐤戜技銆?
    return (
        "鏃堕棿鏉冮噸鎽樿锛氭姤鍛婁富娆″繀椤绘寜鐩搁偦鎶撴媿闂撮殧鍜岀敾闈㈠姩浣滅疮璁★紝涓嶈鎶婃媿鍒扮殑棰樼洰骞冲潎鍒嗛厤鍒版€绘椂闀匡紱"
        "鐢ㄦ埛鎿嶄綔鎵嬫満鏈韩鏄己鍦ㄥ満璇佹嵁锛涙湁瀛︾敓/鎵?绗?涔﹀啓璇佹嵁鐨勬椂娈典紭鍏堬紱瑙嗚鏈娴嬪埌瀛︾敓鍙兘璇存槑鐢婚潰鏈‘璁わ紝涓嶈兘鍗曠嫭鍒ゅ畾瀛︾敓绂诲紑锛涙棤鍚庣画鐢婚潰璇佹嵁鐨勬椂娈典笉鑳借嚜鍔ㄧ畻浣滈鐩€楁椂銆俓n"
        f"- 鏈夊鐢?鎵?绗旇瘉鎹椂闀匡細{format_duration(totals.get('present'))}锛?
        f"鏈娴嬪埌瀛︾敓/鐤戜技涓嶅湪鍦烘椂闀匡細{format_duration(totals.get('absent'))}锛?
        f"鍦ㄥ満鏈瘑鍒椂闀匡細{format_duration(totals.get('unknown'))}锛?
        f"鏈€鍚庝竴寮犲埌缁撴潫鏃犳柊澧炵敾闈細{tail_gap}锛涙暟鎹川閲忔彁绀猴細{quality_note}\n"
        "- 鏈€闀胯瀵熼棿闅旓紙鎸夋寔缁椂闂撮檷搴忥級锛歕n"
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
        emit_log(f"鐢熸垚缂╃暐鍥惧け璐ワ細{exc}", session_id=session_id, level="warning")
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


async def save_question_crop_uploads(
    uploads: list[UploadFile],
    manifest_raw: str,
    *,
    session_id: str,
    batch_id: str,
    image_rows: list[dict],
) -> dict:
    manifest_payload = parse_question_crop_manifest_payload(manifest_raw)
    manifest_items = question_crop_manifest_items(manifest_payload)
    client_metrics = json_object_value(manifest_payload.get("metrics")) if manifest_payload else {}
    if not manifest_items:
        return {
            "saved": [],
            "saved_count": 0,
            "duplicate_count": 0,
            "skipped_count": len(uploads or []),
            "expanded_count": 0,
            "client_crop_count": 0,
            "rect_only_count": 0,
            "replaced_count": 0,
            "expansion_duration_ms": 0,
            "client_metrics": client_metrics,
        }

    sequence_to_image: dict[int, dict] = {}
    index_to_image: dict[int, dict] = {}
    id_to_image: dict[str, dict] = {}
    for index, row in enumerate(image_rows):
        if not image_row_allows_question_crop(row):
            continue
        sequence_index = int_value(row.get("sequence_index"))
        if sequence_index is not None:
            sequence_to_image.setdefault(sequence_index, row)
        index_to_image[index] = row
        image_id = str(row.get("image_id") or "").strip()
        if image_id:
            id_to_image.setdefault(image_id, row)
    saved: list[dict] = []
    duplicate_count = 0
    skipped_count = 0
    expanded_count = 0
    client_crop_count = 0
    rect_only_count = 0
    replaced_count = 0
    expansion_duration_ms = 0
    now = utc_now()
    settings = get_settings()
    image_dir = settings.data_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    with connect() as conn:
        for manifest_index, item in enumerate(manifest_items):
            file_index = crop_manifest_int(item, "file_index", "fileIndex")
            upload_index = file_index if file_index is not None else manifest_index
            upload = uploads[upload_index] if 0 <= upload_index < len(uploads or []) else None
            transfer_mode = meta_text(item, "transfer_mode", "transferMode", "upload_mode", "uploadMode").strip().lower()
            crop_prepared_value = item.get("crop_prepared", item.get("cropPrepared"))
            crop_prepared = str(crop_prepared_value).strip().lower() not in {"0", "false", "no"} if crop_prepared_value is not None else upload is not None
            use_client_upload = upload is not None and crop_prepared and transfer_mode not in {"rect_only", "source_rect", "server_rect"}
            source_sequence = crop_manifest_int(
                item,
                "source_sequence_index",
                "sourceSequenceIndex",
                "sequence_index",
                "sequenceIndex",
            )
            source_image_index = crop_manifest_int(item, "source_image_index", "sourceImageIndex", "frame_index", "frameIndex")
            source_image_id = truncate_text(meta_text(item, "source_image_id", "sourceImageId"), 80)
            row_candidates = [
                row
                for row in (
                    sequence_to_image.get(source_sequence) if source_sequence is not None else None,
                    index_to_image.get(source_image_index) if source_image_index is not None else None,
                    id_to_image.get(source_image_id) if source_image_id else None,
                )
                if row is not None
            ]
            row_ids = {str(row.get("image_id") or "") for row in row_candidates if row.get("image_id")}
            if len(row_ids) > 1:
                skipped_count += 1
                continue
            image_row = row_candidates[0] if row_candidates else None
            if image_row is None:
                skipped_count += 1
                continue

            question_key = truncate_text(meta_text(item, "question_key", "questionKey", "key"), 160)
            fingerprint = truncate_text(meta_text(item, "fingerprint"), 120)
            crop_hash = truncate_text(meta_text(item, "crop_hash", "cropHash"), 120)
            preview_text = truncate_text(meta_text(item, "preview_text", "previewText", "ocr_text", "ocrText", "preview"), 600)
            question_key_strength = truncate_text(meta_text(item, "question_key_strength", "questionKeyStrength", "key_strength", "keyStrength"), 40)
            weak_question_key = question_crop_key_is_weak(question_key, question_key_strength)
            dedupe_question_key = "" if weak_question_key else question_key
            dedupe_fingerprint = "" if weak_question_key else fingerprint
            text_hash_source = "" if weak_question_key else (question_key or preview_text or fingerprint)
            text_hash = hashlib.sha1(text_hash_source.encode("utf-8")).hexdigest() if text_hash_source else ""
            normalized_rect = item.get("normalized_rect") or item.get("normalizedRect") or {}
            crop_rect = item.get("crop_rect") or item.get("cropRect") or {}
            source_image_size = item.get("source_image_size") or item.get("sourceImageSize") or {}
            crop_image_size = item.get("crop_image_size") or item.get("cropImageSize") or {}
            question_index = crop_manifest_int(item, "question_index", "questionIndex", "index") or 0
            confidence = crop_manifest_float(item, "confidence")
            client_source = truncate_text(meta_text(item, "source") or ("ios-observation-crop" if use_client_upload else "ios-observation-rect"), 80)
            client_crop_quality = truncate_text(meta_text(item, "crop_quality", "cropQuality"), 80)
            client_crop_reasons = crop_manifest_str_list(item, "crop_risk_reasons", "cropRiskReasons", "crop_reasons", "cropReasons")
            client_crop_area = crop_manifest_float(item, "crop_area", "cropArea")
            frame_candidate_count = crop_manifest_int(item, "frame_candidate_count", "frameCandidateCount")
            frame_candidate_limit = crop_manifest_int(item, "frame_candidate_limit", "frameCandidateLimit")
            frame_limited_candidate_count = crop_manifest_int(item, "frame_limited_candidate_count", "frameLimitedCandidateCount")
            existing_identity_row: dict | None = None
            if dedupe_question_key or dedupe_fingerprint or text_hash or crop_hash:
                existing = conn.execute(
                    """
                    SELECT *
                    FROM session_question_crops
                    WHERE session_id=? AND status='ready'
                      AND (
                        (? != '' AND question_key=?)
                        OR (? != '' AND fingerprint=?)
                        OR (? != '' AND text_hash=?)
                        OR (? != '' AND crop_hash=?)
                      )
                    LIMIT 1
                    """,
                    (
                        session_id,
                        dedupe_question_key, dedupe_question_key,
                        dedupe_fingerprint, dedupe_fingerprint,
                        text_hash, text_hash,
                        crop_hash, crop_hash,
                    ),
                ).fetchone()
                if existing:
                    existing_identity_row = row_to_dict(existing)

            crop_id = uuid.uuid4().hex
            original_name = ""
            if use_client_upload and upload is not None:
                ext = Path(upload.filename or "question-crop.jpg").suffix.lower() or ".jpg"
                if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                    ext = ".jpg"
                filename = f"{session_id}_{batch_id}_qcrop_{crop_id}{ext}"
                target = image_dir / filename
                with target.open("wb") as out:
                    shutil.copyfileobj(upload.file, out)
                original_name = truncate_text(upload.filename or "", 240)
                crop_result = maybe_expand_question_crop_file(
                    session_id=session_id,
                    batch_id=batch_id,
                    crop_id=crop_id,
                    image_dir=image_dir,
                    client_crop_filename=filename,
                    client_crop_path=target,
                    source_image_filename=image_row.get("filename") or "",
                    crop_rect=crop_rect,
                    source_image_size=source_image_size,
                    crop_image_size=crop_image_size,
                    normalized_rect=normalized_rect,
                    client_source=client_source,
                    crop_hash=crop_hash,
                )
            else:
                crop_result = save_question_crop_from_source_rect(
                    session_id=session_id,
                    batch_id=batch_id,
                    crop_id=crop_id,
                    image_dir=image_dir,
                    source_image_filename=image_row.get("filename") or "",
                    crop_rect=crop_rect,
                    source_image_size=source_image_size,
                    normalized_rect=normalized_rect,
                    client_source=client_source,
                )
            if (
                client_crop_quality
                or client_crop_reasons
                or question_key_strength
                or client_crop_area is not None
                or frame_candidate_count is not None
                or frame_candidate_limit is not None
                or frame_limited_candidate_count is not None
            ):
                crop_safety = crop_result.get("crop_safety") or {}
                if client_crop_quality:
                    crop_safety["client_crop_quality"] = client_crop_quality
                if client_crop_reasons:
                    crop_safety["client_crop_reasons"] = client_crop_reasons
                client_telemetry = crop_safety.get("client_telemetry") if isinstance(crop_safety.get("client_telemetry"), dict) else {}
                if question_key_strength:
                    client_telemetry["question_key_strength"] = question_key_strength
                if client_crop_area is not None:
                    client_telemetry["crop_area"] = round(float(client_crop_area), 6)
                if confidence is not None:
                    client_telemetry["confidence"] = round(float(confidence), 6)
                if source_sequence is not None:
                    client_telemetry["source_sequence_index"] = source_sequence
                if source_image_index is not None:
                    client_telemetry["source_image_index"] = source_image_index
                if frame_candidate_count is not None:
                    client_telemetry["frame_candidate_count"] = frame_candidate_count
                if frame_candidate_limit is not None:
                    client_telemetry["frame_candidate_limit"] = frame_candidate_limit
                if frame_limited_candidate_count is not None:
                    client_telemetry["frame_limited_candidate_count"] = frame_limited_candidate_count
                if client_telemetry:
                    crop_safety["client_telemetry"] = client_telemetry
                crop_result["crop_safety"] = crop_safety
                crop_result["normalized_rect"] = question_crop_trace_payload(
                    crop_result.get("normalized_rect"),
                    client_crop_rect=crop_rect,
                    crop_safety=crop_safety,
                )
            filename = crop_result["filename"]
            target = crop_result["path"]
            if not target.exists():
                skipped_count += 1
                continue
            crop_hash = truncate_text(crop_result.get("crop_hash") or crop_hash, 120)
            existing_hash_row: dict | None = None
            if crop_hash:
                existing_by_hash = conn.execute(
                    """
                    SELECT *
                    FROM session_question_crops
                    WHERE session_id=? AND status='ready' AND crop_hash=?
                    LIMIT 1
                    """,
                    (session_id, crop_hash),
                ).fetchone()
                if existing_by_hash:
                    existing_hash_row = row_to_dict(existing_by_hash)

            row = {
                "id": crop_id,
                "session_id": session_id,
                "batch_id": batch_id,
                "image_id": image_row["image_id"],
                "sequence_index": int(image_row.get("sequence_index") or 0),
                "manifest_index": manifest_index,
                "question_index": question_index,
                "question_key": question_key,
                "fingerprint": fingerprint,
                "crop_hash": crop_hash,
                "text_hash": text_hash,
                "normalized_rect": json_object_string(crop_result.get("normalized_rect"), {}),
                "crop_rect": json_object_string(crop_result.get("crop_rect"), {}),
                "source_image_size": json_object_string(crop_result.get("source_image_size"), {}),
                "crop_image_size": json_object_string(crop_result.get("crop_image_size"), {}),
                "preview_text": preview_text,
                "crop_filename": filename,
                "original_name": original_name,
                "status": "ready",
                "source": crop_result.get("source") or "client_crop",
                "confidence": confidence,
                "src_filename": image_row.get("filename") or "",
                "crop_safety": crop_result.get("crop_safety") or {},
            }
            existing_match = existing_identity_row
            if existing_hash_row and (not existing_match or existing_hash_row.get("id") != existing_match.get("id")):
                existing_match = existing_hash_row
            if existing_match:
                if question_crop_new_row_is_better(row, existing_match, image_dir):
                    create_thumbnail_safe(target, filename, session_id)
                    old_filename = str(existing_match.get("crop_filename") or "").strip()
                    updated_row = {**row, "id": existing_match.get("id") or row["id"]}
                    replace_question_crop_row(conn, existing_match, updated_row, now)
                    if old_filename and old_filename != filename:
                        remove_question_crop_file(image_dir, old_filename)
                    replaced_count += 1
                    if crop_result.get("expanded"):
                        expanded_count += 1
                    if use_client_upload:
                        if not crop_result.get("expanded"):
                            client_crop_count += 1
                    else:
                        rect_only_count += 1
                    expansion_duration_ms += int((crop_result.get("crop_safety") or {}).get("duration_ms") or 0)
                    continue
                duplicate_count += 1
                remove_question_crop_file(image_dir, filename)
                continue

            if crop_result.get("expanded"):
                expanded_count += 1
            if use_client_upload:
                if not crop_result.get("expanded"):
                    client_crop_count += 1
            else:
                rect_only_count += 1
            expansion_duration_ms += int((crop_result.get("crop_safety") or {}).get("duration_ms") or 0)
            create_thumbnail_safe(target, filename, session_id)
            conn.execute(
                """
                INSERT INTO session_question_crops(
                    id, session_id, batch_id, image_id, sequence_index, manifest_index,
                    question_index, question_key, fingerprint, crop_hash, text_hash,
                    normalized_rect, crop_rect, source_image_size, crop_image_size,
                    preview_text, crop_filename, original_name, status, source, confidence,
                    created_at, updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row["id"],
                    row["session_id"],
                    row["batch_id"],
                    row["image_id"],
                    row["sequence_index"],
                    row["manifest_index"],
                    row["question_index"],
                    row["question_key"],
                    row["fingerprint"],
                    row["crop_hash"],
                    row["text_hash"],
                    row["normalized_rect"],
                    row["crop_rect"],
                    row["source_image_size"],
                    row["crop_image_size"],
                    row["preview_text"],
                    row["crop_filename"],
                    row["original_name"],
                    row["status"],
                    row["source"],
                    row["confidence"],
                    now,
                    now,
                ),
            )
            saved.append(row)
        conn.commit()

    if saved or skipped_count or duplicate_count or replaced_count:
        client_duration_ms = int_value(client_metrics.get("analysis_duration_ms")) or 0
        client_weak_layout = int_value(client_metrics.get("weak_layout_candidate_count")) or 0
        client_strong_ocr = int_value(client_metrics.get("strong_ocr_candidate_count")) or 0
        client_limited = int_value(client_metrics.get("limited_candidate_count")) or 0
        client_low_confidence = int_value(client_metrics.get("low_confidence_candidate_count")) or 0
        emit_log(
            (
                f"question crop safety summary: saved={len(saved)}, expanded={expanded_count}, "
                f"client_crop={client_crop_count}, rect_only={rect_only_count}, duplicate={duplicate_count}, "
                f"replaced={replaced_count}, skipped={skipped_count}, duration_ms={expansion_duration_ms}, "
                f"client_ms={client_duration_ms}, client_weak_layout={client_weak_layout}, "
                f"client_strong_ocr={client_strong_ocr}, client_limited={client_limited}, "
                f"client_low_confidence={client_low_confidence}"
            ),
            session_id=session_id,
            source="question_crop",
        )
    return {
        "saved": saved,
        "saved_count": len(saved),
        "duplicate_count": duplicate_count,
        "skipped_count": skipped_count,
        "expanded_count": expanded_count,
        "client_crop_count": client_crop_count,
        "rect_only_count": rect_only_count,
        "replaced_count": replaced_count,
        "expansion_duration_ms": expansion_duration_ms,
        "client_metrics": client_metrics,
    }


def build_batch_prompt(environment: str, image_rows: list[dict], previous_context: str = "", strategy_context: str = "") -> str:
    capture_lines = "\n".join(
        (
            f"{index + 1}. sequence_index={row['sequence_index']}锛宑aptured_at={row['captured_at'] or '鏈煡'}锛?
            f"filename={row['filename']}锛宑lient_meta={row['capture_meta'] or '{}'}"
        )
        for index, row in enumerate(image_rows)
    )
    return prompts.render_prompt(
        "batch_analysis",
        image_count=len(image_rows),
        environment=(environment or "鏈彁渚?) + ("\n" + strategy_context if strategy_context else ""),
        capture_lines=capture_lines,
        previous_context=previous_context or "鏃犳鍓嶆壒娆¤褰曘€?,
    )


def duplicate_batch_content(image_rows: list[dict]) -> str:
    lines = [
        (
            f"{index + 1}. sequence_index={row.get('sequence_index')}锛宑aptured_at={row.get('captured_at') or '鏈煡'}锛?
            f"涓庡凡淇濆瓨鍏抽敭鐢婚潰閲嶅锛宒uplicate_of={row.get('duplicate_of_image_id') or '鏈煡'}锛?
            f"signal={row.get('signal_summary') or '鏃?}"
        )
        for index, row in enumerate(image_rows)
    ]
    return (
        "鏈壒娆℃病鏈夋柊澧炲叧閿敾闈紝鏈皟鐢ㄨ瑙夊ぇ妯″瀷銆俓n"
        "鍚庣宸叉妸杩欎簺甯т綔涓洪噸澶嶈瀵熷啓鍏ユ椂闂寸嚎锛屽彲鐢ㄤ簬浼扮畻鍋滅暀鏃堕暱锛屼絾涓嶄細閲嶅璁板綍棰樼洰/鐭ヨ瘑鐐广€俓n"
        + "\n".join(lines)
    )


def skipped_batch_content(image_rows: list[dict]) -> str:
    invalid_rows = [row for row in image_rows if row.get("novelty_status") == "invalid"]
    duplicate_rows = [row for row in image_rows if row.get("novelty_status") == "duplicate"]
    lines: list[str] = []
    for row in invalid_rows:
        lines.append(
            f"- sequence_index={row.get('sequence_index')}锛宑aptured_at={row.get('captured_at') or '鏈煡'}锛?
            f"鏃犵浉鍏冲涔犵敾闈紝reason={row.get('discard_reason') or 'invalid_image'}锛?
            f"detail={row.get('discard_detail') or '鐢婚潰鏃犵浉鍏?}"
        )
    for row in duplicate_rows:
        lines.append(
            f"- sequence_index={row.get('sequence_index')}锛宑aptured_at={row.get('captured_at') or '鏈煡'}锛?
            f"閲嶅瑙傚療锛宒uplicate_of={row.get('duplicate_of_image_id') or '鏈煡'}锛?
            f"signal={row.get('signal_summary') or '鏃?}"
        )
    if invalid_rows and not duplicate_rows:
        head = "鏈壒娆＄敾闈㈡棤鐩稿叧瀛︿範鍐呭锛屽凡鑷姩蹇界暐锛屾湭璋冪敤瑙嗚澶фā鍨嬨€?
    elif invalid_rows:
        head = "鏈壒娆℃病鏈夋柊澧炲彲瑙ｆ瀽瀛︿範鐢婚潰锛氭棤鐩稿叧鐢婚潰宸插拷鐣ワ紝閲嶅鐢婚潰鍙啓鍏ユ椂闂寸嚎锛屾湭璋冪敤瑙嗚澶фā鍨嬨€?
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
                f"{index + 1}. sequence_index={row.get('sequence_index') or 0}锛?
                f"captured_at={row.get('captured_at') or row.get('created_at') or '鏈煡'}锛?
                f"鍒颁笅涓€寮?缁撴潫闂撮殧={format_duration(seconds_to_next)}锛?
                f"batch={row.get('batch_id') or 'single'}锛?
                f"{'锛?.join(hints) + '锛? if hints else ''}"
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
            parts.append(f"... 宸茬渷鐣?{omitted} {omitted_unit}锛屼繚鐣欎唬琛ㄦ€у紑澶?涓/缁撳熬 ...")
        parts.append(lines[index])
        previous = index
    trailing = len(lines) - previous - 1
    if trailing > 0:
        parts.append(f"... 宸茬渷鐣?{trailing} {omitted_unit}锛屼繚鐣欎唬琛ㄦ€у紑澶?涓/缁撳熬 ...")
    return "\n".join(parts)


def build_limited_batch_notes(analyses: list[dict], max_chars: int) -> tuple[str, bool]:
    rows = [row for row in analyses if row["scope"] != "final"]
    if not rows:
        return "鏆傛棤鎵规瑙嗚鍒嗘瀽鍐呭銆?, False
    per_note_limit = max(
        FINAL_REPORT_ANALYSIS_MIN_CHARS,
        min(FINAL_REPORT_ANALYSIS_MAX_CHARS, max_chars // max(len(rows), 1) - 80),
    )
    note_truncated = any(len((row["content"] or "鏃犲唴瀹?).strip()) > per_note_limit for row in rows)
    notes = [
        (
            f"銆恵index + 1}. {row['scope']} / {row['status']} / batch={row['batch_id'] or '鏃?} / created_at={row['created_at']}銆慭n"
            f"{truncate_text(row['content'] or '鏃犲唴瀹?, per_note_limit)}"
        )
        for index, row in enumerate(rows)
    ]
    rendered, notes_compressed = render_limited_lines(notes, max_chars, "鏉℃壒娆″垎鏋?)
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
        return "鏃犲唴瀹?
    keywords = (
        "椤?,
        "棰?,
        "棰樼洰",
        "棰樺共",
        "绛旀",
        "鎵嬪啓",
        "绠楀紡",
        "鑽夌",
        "璁㈡",
        "浣滅瓟",
        "瀛︿範绉戠洰",
        "瀛︿範鏂瑰紡",
        "浜轰綋",
        "鎵?,
        "澶?,
        "韬綋",
        "绗?,
        "涔﹀啓",
        "鍋滅暀",
        "鑰楁椂",
        "鏃堕棿",
        "鍙樺寲",
        "缈婚〉",
        "鏈瘑鍒?,
        "鐤戜技",
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
        fallback = render_selected_lines(lines, representative_indices(len(lines)), "琛屽垎鏋愬唴瀹?)
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
    content = important_analysis_lines(row.get("content") or "鏃犲唴瀹?, per_note_limit)
    return (
        f"銆恵index}. {row['scope']} / {row['status']} / batch={row['batch_id'] or '鏃?} / created_at={row['created_at']}銆慭n"
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
        return "鏃犳鍓嶆壒娆¤褰曘€?
    notes = [build_evidence_note(row, index + 1, 900) for index, row in enumerate(reversed(rows))]
    rendered = "\n\n".join(notes)
    return truncate_text(rendered, BATCH_PREVIOUS_CONTEXT_LIMIT)


def clean_learning_line(line: str) -> str:
    cleaned = re.sub(r"^\s*[-*鈥d.銆乗)\]]+\s*", "", line.strip())
    cleaned = cleaned.strip("# 锛?;锛沑t ")
    return truncate_text(cleaned, LEARNING_ITEM_CONTENT_LIMIT)


def content_after_label(line: str) -> str:
    parts = re.split(r"[锛?]", line, maxsplit=1)
    if len(parts) == 2 and 2 <= len(parts[1].strip()):
        return parts[1].strip()
    return line


def learning_item_type_for_line(line: str) -> str | None:
    text = line.strip()
    if not text:
        return None
    if any(keyword in text for keyword in ("鐭ヨ瘑鐐?, "鑰冪偣", "鏄撻敊鐐?, "姒傚康", "鍏紡", "鏂规硶")):
        return "knowledge"
    if any(keyword in text for keyword in ("鏉垮潡", "绔犺妭", "鍗曞厓", "椤甸潰", "椤电爜", "鏉愭枡", "缁冧範鍐?, "璇曞嵎")):
        return "section"
    if any(keyword in text for keyword in ("棰樼洰", "棰樺彿", "棰樺共", "闂", "绗?)) and any(keyword in text for keyword in ("棰?, "椤?, "闂?)):
        return "question"
    if any(keyword in text for keyword in ("绛旀", "浣滅瓟", "鎵嬪啓", "绠楀紡", "鑽夌", "璁㈡", "绌虹櫧", "鏈綔绛?)):
        return "answer"
    return None


def title_for_learning_item(content: str) -> str:
    one_line = " ".join(content.split())
    one_line = re.sub(r"^[涓€浜屼笁鍥涗簲鍏竷鍏節鍗?-9]+[.銆乚\s*", "", one_line)
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
        if candidate in {"鏈瘑鍒?, "鏈煡", "鏃?, "鏃犲唴瀹?, "鏃犲彲鍒嗘瀽鍐呭"}:
            continue
        if "鏈瘑鍒? in candidate and len(candidate) < 12:
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
    if any(keyword in source for keyword in ("閿欓", "閿欒", "閿欏洜", "鍋氶敊", "绠楅敊", "绛旀閿欒", "涓嶆纭?)):
        return "suspected"
    if any(keyword in source for keyword in ("绌虹櫧", "鏈綔绛?, "涓嶄細", "鏈畬鎴?)):
        return "incomplete"
    return None


def mistake_is_empty_sentinel(text: str) -> bool:
    """The "no mistakes" placeholder lines (e.g. 鏆傛棤鏄庣‘閿欓鍊欓€? contain the word
    閿欓 and would otherwise be parsed into a bogus mistake item."""
    s = text or ""
    return ("鏆傛棤" in s or "鏃犳槑纭? in s) and ("閿欓" in s or "鍊欓€? in s)


def report_has_student_answer(content: str) -> bool:
    """True when the report shows at least one real student answer somewhere.
    Used to tell an unstarted/blank new paper (no answers anywhere) apart from a
    worked paper with a few questions left blank: in the former, blank questions are
    NOT mistakes; in the latter, the blanks ARE kept as 鏈畬鎴?(鐤戜技涓嶄細鍋?."""
    placeholders = {"鏈瘑鍒?, "鏈煡", "鏃?, "鏃犲唴瀹?, "鏃犲彲鍒嗘瀽鍐呭", "绌虹櫧", "鏈綔绛?, "鏆傛棤"}
    for raw_line in (content or "").splitlines():
        line = clean_learning_line(raw_line)
        if not line or learning_item_type_for_line(line) != "answer":
            continue
        # A wrong answer still means the student wrote something, so it counts as an
        # attempt; only skip lines that report the answer as blank/missing.
        if any(marker in line for marker in ("绌虹櫧", "鏈綔绛?, "鏈畬鎴?)):
            continue
        body = clean_learning_line(content_after_label(line))
        if not body or len(body) < 2:
            continue
        if body in placeholders or any(marker in body for marker in ("鏈瘑鍒?, "绌虹櫧", "鏈綔绛?)):
            continue
        return True
    return False


# 鏅鸿兘瑙傚療鎶ュ憡閲岀殑鎻忚堪鎬ц瘝姹団€斺€旇繖浜涜鎻忚堪"灞忓箷涓婂湪鏄剧ず浠€涔?鍋滅暀澶氫箙"锛屼笉鏄壒鏀圭粨鏋滐紝
# 缁濅笉鑳借繘閿欓鏈紙涔嬪墠閿欓鏈噷婊″睆"棰樼洰鑰楁椂绾跨储/宸紓棰樼洰/鏃犳柊澧炵焊璐ㄩ鐩?鐢婚潰闈欐"灏辨槸瀹冧滑婕忚繘鏉ョ殑锛夈€?
_MISTAKE_OBSERVATION_NOISE = (
    "鐢婚潰", "鑰楁椂", "绾跨储", "宸紓棰樼洰", "鏃犳柊澧?, "鐣岄潰", "灞忓箷", "鎴浘", "闈欐",
    "鍔ㄤ綔璇佹嵁", "闃呰鍔ㄤ綔", "宸ヤ綔鍙?, "鎸佺画鏄剧ず", "鍋滅暀", "缈婚〉", "鍦ㄥ満", "鏈娴嬪埌",
    "浼拌", "鏈壒娆?, "杩炵画", "绉掓媿鎽?, "鐤戜技姝ｅ湪",
)

_MISTAKE_PLACEHOLDER_VALUES = {"", "鏃?, "鏈瘑鍒?, "鏈煡", "鏆傛棤", "绌虹櫧", "鏈綔绛?, "-", "鈥?, "鏃犲唴瀹?, "鏃犲彲鍒嗘瀽鍐呭"}


def looks_like_observation_noise(text: str) -> bool:
    return any(marker in (text or "") for marker in _MISTAKE_OBSERVATION_NOISE)


def is_placeholder_mistake_text(text: str) -> bool:
    body = (text or "").strip()
    return len(body) < 2 or body in _MISTAKE_PLACEHOLDER_VALUES


def is_blank_mistake_value(text: str) -> bool:
    """鎸夊彇鍊硷紙鑰岄潪闀垮害锛夊垽绌猴紝鐢ㄤ簬缁撴瀯鍖栭敊棰樺瓧娈碉細鍗曞瓧绗︾瓟妗堝鈥?鈥濃€?鈥濃€渪鈥濇槸鍚堟硶浣滅瓟锛?
    涓嶈兘鍍?is_placeholder_mistake_text 閭ｆ牱鎸?len<2 璇潃銆?""
    body = (text or "").strip()
    return not body or body in _MISTAKE_PLACEHOLDER_VALUES


def extract_mistake_items(content: str, learning_items: list[dict] | None = None) -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()
    knowledge_lines = [item["content"] for item in (learning_items or []) if item.get("item_type") == "knowledge"]
    context_subject = extract_subject_from_text(content)
    # A wholly-blank paper (no student answers anywhere) is an unstarted/new paper,
    # not a set of mistakes. Only treat blank questions as 鏈畬鎴愰敊棰?once at least
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
        expected = extract_named_detail(line, ("鍙傝€冪瓟妗?, "姝ｇ‘绛旀", "鏍囧噯绛旀", "搴斾负"), ASSET_FIELD_LIMIT)
        if expected:
            recent_expected = expected
        if not status:
            continue
        if mistake_is_empty_sentinel(line):
            continue
        if status == "incomplete" and not has_student_answer:
            continue
        title_source = recent_question or body or line
        # 璐ㄩ噺闂ㄦ锛堥敊棰樻湰鍘诲櫔锛夛細涓€閬撶湡姝ｇ殑閿欓蹇呴』鏈?鍏蜂綋璇佹嵁"鈥斺€斿鐢熷啓浜嗕綔绛斻€?
        # 鎴栨槑纭啓浜嗛敊鍥犮€佹垨缁欎簡鍙傝€冪瓟妗堛€傚彧鏄煇琛岄噷鍑虹幇浜?閿?瀛楋紙甯歌浜庤瀵熸姤鍛?娉涜堪锛?
        # 涓嶈冻浠ュ垽涓洪敊棰樸€傛櫤鑳借瀵熺殑鎻忚堪鎬ц锛堢敾闈?鑰楁椂/鐣岄潰鈥︼級涓€寰嬫帓闄ゃ€?
        explicit_reason = extract_named_detail(line, ("閿欏洜", "閿欒鍘熷洜", "鐤戜技閿欏洜"), MISTAKE_REASON_LIMIT)
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
        correction = extract_named_detail(line, ("璁㈡", "鏀规", "璁㈡鐥曡抗", "淇"), ASSET_SOURCE_SUMMARY_LIMIT)
        next_action = extract_named_detail(line, ("涓嬩竴姝?, "寤鸿", "甯姪寤鸿", "涓嬩竴姝ュ府鍔╁缓璁?), ASSET_SOURCE_SUMMARY_LIMIT)
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


# 鈹€鈹€ 閿欓鏈粨鏋勫寲鍏ュ簱锛堢簿鍑嗚瘑鍒級 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 鏃ц矾寰勭敤姝ｅ垯鍙嶈В瑙傚療鎶ュ憡鏁ｆ枃锛屽鑷粹€滃樊寮傞鐩?棰樼洰鑰楁椂绾跨储/鐢婚潰闈欐鈥︹€濊繖绫诲皬鑺傛爣棰樺拰
# 鐢婚潰鎻忚堪琚鍒ゆ垚閿欓銆傛柊璺緞璁┾€滅湅寰楄鍥剧墖鈥濈殑妯″瀷鍦ㄤ富鍔ㄦ媿棰樻椂鐩存帴鍚愭満璇?JSON
# 锛堜互鈥滈敊棰楯SON锛氣€濆紑澶达級锛屽悗绔彧瑙ｆ瀽瀹冿紝骞惰姹傗€滃鐢熺‘鏈夋墜鍐欎綔绛?+ 鍙鍒ら敊璇佹嵁鈥濇墠鍏ュ簱銆?
_MISTAKE_JSON_MARKER = re.compile(r"閿欓\s*JSON", re.IGNORECASE)
# 瑙傚療寮傛鍊欓€夌敤鐙珛鏍囪锛岄伩鍏嶅拰涓诲姩鎷嶉鐨勨€滈敊棰楯SON鈥濇贩娣嗐€?
_MISTAKE_CANDIDATE_JSON_MARKER = re.compile(r"閿欓\s*鍊欓€塡s*JSON", re.IGNORECASE)

_MISTAKE_JSON_FIELD_ALIASES = {
    "question": ("棰樼洰", "棰樺共", "棰?, "question", "question_text"),
    "student": ("瀛︾敓浣滅瓟", "瀛︾敓绛旀", "鎴戠殑浣滅瓟", "浣滅瓟", "student_answer"),
    "expected": ("姝ｇ‘绛旀", "鍙傝€冪瓟妗?, "鏍囧噯绛旀", "搴斾负", "expected_answer"),
    "reason": ("閿欏洜", "閿欒鍘熷洜", "鐤戜技閿欏洜", "reason", "error_reason"),
    "error_type": ("閿欒绫诲瀷", "閿欏洜绫诲瀷", "绫诲瀷", "error_type"),
    "evidence": ("璇佹嵁", "鍒ら敊璇佹嵁", "evidence"),
    "correction": ("璁㈡", "鏀规", "淇", "correction"),
    "knowledge": ("鐭ヨ瘑鐐?, "鑰冪偣", "knowledge", "knowledge_points"),
    "region": ("鍖哄煙", "region", "bbox"),
    "verdict": ("鍒ゅ畾", "鍒ゅ畾缁撴灉", "error_verdict", "verdict"),
    "question_index": ("棰樺簭", "棰樺彿搴?, "question_index"),
}

# 閫愰鎵规敼锛堝叏棰樺閿欐潈濞侊級鐢ㄧ嫭绔嬫爣璁帮紝鍜屸€滈敊棰楯SON/閿欓鍊欓€塉SON鈥濆尯鍒嗗紑銆?
_BATCH_GRADING_JSON_MARKER = re.compile(r"鎵规敼\s*JSON", re.IGNORECASE)

# 灞曠ず鐢細鍒嗘瀽姝ｆ枃鏈熬浼氳拷鍔犳満璇荤粨鏋勫寲鍧楋紙閿欓鍊欓€塉SON/閿欓JSON/鎵规敼JSON锛歔...]锛夈€傝繖浜涘潡鍙緵鍚庣瑙ｆ瀽锛?
# 涓嶈鍥炴斁缁欑敤鎴风湅锛堝巻鍙茶瀵熷洖鏀句細鐩存帴鏄剧ず analyses[].content锛夈€傛ā鏉跨害瀹氱粨鏋勫寲鍧楀湪浜鸿姝ｆ枃涔嬪悗杩藉姞锛?
# 鏁呬粠鏈€鏃╁嚭鐜扮殑鏍囪澶勬埅鏂嵆鍙€備粎鐢ㄤ簬 API 杩斿洖鏃舵竻娲楋紝DB 鍘熸枃淇濈暀渚涙渶缁堟姤鍛婅捀棣忋€?
_STRUCTURED_JSON_DISPLAY_MARKER = re.compile(r"(閿欓鍊欓€塡s*JSON|閿欓\s*JSON|鎵规敼\s*JSON)", re.IGNORECASE)


def strip_structured_json_for_display(content: object) -> str:
    text = content if isinstance(content, str) else ("" if content is None else str(content))
    if not text:
        return text
    match = _STRUCTURED_JSON_DISPLAY_MARKER.search(text)
    if not match:
        return text
    return text[: match.start()].rstrip()


def sanitize_analyses_for_display(analyses: list[dict]) -> list[dict]:
    """鎶婂垎鏋愬垪琛ㄩ噷姣忔潯 content 鐨勭粨鏋勫寲鏈鸿鍧楀幓鎺夛紙鍙奖鍝嶈繑鍥炲鎴风鐨勫壇鏈紝涓嶆敼 DB锛夈€?""
    cleaned: list[dict] = []
    for row in analyses:
        item = dict(row)
        if "content" in item:
            item["content"] = strip_structured_json_for_display(item.get("content"))
        cleaned.append(item)
    return cleaned


def _region_bbox(raw: object) -> dict:
    """鎶婂綊涓€鍖栧尯鍩熸 {x,y,w,h}锛堝彇鍊糩0,1]锛屽師鐐瑰乏涓娿€亁 鍙?y 涓嬨€佺浉瀵瑰凡鏍℃绔栫洿鍥撅級瑙勬暣鎴?
    骞插噣鐨?dict锛涙棤娉曡瘑鍒紙缂哄瓧娈?闈炴暟瀛?闈?dict锛夋椂杩斿洖 {}锛岃皟鐢ㄦ柟鎹缁存寔鐜扮姸銆?""
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
    """瑙ｆ瀽妯″瀷鐩村嚭鐨勬満璇婚敊棰?JSON銆俧orce_status='candidate' 鏃惰蛋瑙傚療鍊欓€夋ā寮忥細闂ㄦ鏇村
    锛堝洜涓哄悗闈㈡湁瀛︾敓浜哄伐纭锛夛紝浣嗕粛瑕佹眰鏈夐闈?鐪熷疄浣滅瓟/璁㈡璇佹嵁锛屽苟鍓旈櫎瑙傚療鐢婚潰鍣０銆?""
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
        is_incomplete = any(token in combined for token in ("鏈畬鎴?, "绌虹櫧", "鏈綔绛?, "涓嶄細"))
        has_student = not is_blank_mistake_value(student)
        has_correction_evidence = bool(correction) or any(
            token in evidence for token in ("鍒掓帀", "璁㈡", "鏀?, "绾㈠弶", "閿?)
        )
        if force_status == "candidate":
            # 鍊欓€夋ā寮忥紙杈冨鏉撅級锛氭湁棰橀潰锛屼笖瀛︾敓纭湁浣滅瓟 / 鏈夎姝ｅ垝鎺夎瘉鎹?/ 鏄庣‘鐣欑┖鏈畬鎴愩€?
            if not question:
                continue
            if not (has_student or has_correction_evidence or is_incomplete):
                continue
        else:
            # 绮惧噯闂ㄦ锛氳涔堝鐢熺‘鏈夋墜鍐欎綔绛旓紙鐤戜技鍋氶敊锛夛紝瑕佷箞鏄庣‘鏍囨敞鐣欑┖鏈畬鎴愪笖鏈夐闈€?
            if not has_student and not (is_incomplete and question):
                continue
        # 鐢婚潰/鐣岄潰/鑰楁椂绛夎瀵熷櫔澹颁竴寰嬩笉杩涢敊棰樻湰/鍊欓€夛紝鍝€曟ā鍨嬭濉炶繘浜?JSON銆?
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
                # 鍖哄煙 bbox -> location_ref(JSON 瀛楃涓?銆侀搴?-> question_ref锛涚己澶辨椂鐣欑┖锛?
                # 浜ょ粰 enrich_asset_fields_from_text 鍥為€€鍒版棦鏈夐〉鐮?棰樺彿鍚彂寮忋€?
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
    """瑙ｆ瀽妯″瀷杈撳嚭鐨勨€滄壒鏀笿SON锛歔...]鈥濋€愰鍒ゅ畾鏁扮粍锛堝叏棰樺閿欐潈濞侊紝瑕嗙洊妫€娴嬪埌鐨勬墍鏈夐锛?
    鍚仛瀵?绌虹櫧/鏈瘑鍒級銆傚潗鏍囩郴锛氬綊涓€鍖朳0,1]銆佸師鐐瑰乏涓娿€亁 鍙?y 涓嬨€佺浉瀵瑰凡鏍℃绔栫洿鍥俱€?
    瑙ｆ瀽澶辫触鎴栨病鏈夎鏍囪涓€寰嬭繑鍥?[]锛岀粷涓嶅奖鍝嶉敊棰樻娊鍙栭摼璺€?""
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
    if text in {"鏈煡", "鏈瘑鍒?, "鏃?, "鏆傛棤", "涓嶇‘瀹?}:
        return ""
    return truncate_text(text, max_chars)


def split_inline_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    parts = [part.strip() for part in re.split(r"[锛?锛?]\s*", text or "") if part.strip()]
    for part in parts:
        nested = re.match(r"[^=锛?]{1,24}[锛?]\s*(.+[=锛?].*)", part)
        if nested:
            part = nested.group(1).strip()
        match = re.match(r"([^=锛?]{1,20})[=锛?]\s*(.+)", part)
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
        if key in {"绉戠洰", "瀛︾", "瀛︿範绉戠洰"}:
            return value
    match = re.search(r"(?:绉戠洰|瀛︾|瀛︿範绉戠洰)\s*[=锛?]\s*([^锛?锛?\n]+)", source)
    if match:
        return clean_asset_field(match.group(1))
    subjects = ("鏁板", "璇枃", "鑻辫", "鐗╃悊", "鍖栧", "鐢熺墿", "绉戝", "鍘嗗彶", "鍦扮悊", "鏀挎不", "閬撴硶", "淇℃伅鎶€鏈?)
    for subject in subjects:
        if subject in source:
            return subject
    if re.search(r"\d+\s*[+\-脳xX*/梅=]\s*\d+", source) or any(
        keyword in source for keyword in ("鍔犳硶", "鍑忔硶", "涔樻硶", "闄ゆ硶", "杩涗綅", "閫€浣?, "鏂圭▼", "鍑犱綍", "璁＄畻")
    ):
        return "鏁板"
    if re.search(r"[A-Za-z]{2,}", source) and any(keyword in source for keyword in ("鍗曡瘝", "鍙ュ瓙", "璇硶", "缈昏瘧", "闃呰鐞嗚В")):
        return "鑻辫"
    if any(keyword in source for keyword in ("鎷奸煶", "浣滄枃", "闃呰", "璇炬枃", "鐢熷瓧", "淇緸", "鍙よ瘲")):
        return "璇枃"
    return ""


def extract_page_ref(text: str) -> str:
    source = text or ""
    for key, value in split_inline_fields(source).items():
        if key in {"椤电爜", "椤甸潰", "椤?, "page", "page_hint"}:
            return value
    patterns = [
        r"绗琝s*([涓€浜屼笁鍥涗簲鍏竷鍏節鍗佺櫨鍗冧竾\d]+)\s*椤?,
        r"椤电爜\s*[=锛?]\s*([^锛?锛?\n]+)",
        r"椤甸潰\s*[=锛?]\s*([^锛?锛?\n]+)",
        r"page[_\s-]*hint\s*[=锛?]\s*([^锛?锛?\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if pattern.startswith("绗?):
                value = f"绗?{value} 椤?
            return clean_asset_field(value)
    return ""


def extract_question_ref(text: str) -> str:
    source = text or ""
    for key, value in split_inline_fields(source).items():
        if key in {"棰樺彿", "棰?, "闂", "question", "question_hint"}:
            return value
    patterns = [
        r"绗琝s*([涓€浜屼笁鍥涗簲鍏竷鍏節鍗佺櫨鍗冧竾\d]+)\s*(?:棰榺闂畖灏忛)",
        r"棰樺彿\s*[=锛?]\s*([^锛?锛?\n]+)",
        r"question[_\s-]*hint\s*[=锛?]\s*([^锛?锛?\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if pattern.startswith("绗?):
                value = f"绗?{value} 棰?
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
        if key in {"閿欒绫诲瀷", "閿欏洜绫诲瀷", "绫诲瀷"}:
            return value
    checks = [
        ("璁＄畻閿欒", ("绠楅敊", "璁＄畻閿欒", "璁＄畻", "杩涗綅", "閫€浣?)),
        ("瀹￠閿欒", ("瀹￠", "鐪嬮敊", "棰樻剰", "鏉′欢婕忕湅")),
        ("姒傚康涓嶆竻", ("姒傚康", "鍏紡", "瀹氫箟", "鐭ヨ瘑鐐逛笉鐔?)),
        ("姝ラ閬楁紡", ("姝ラ", "婕忔", "杩囩▼", "鎺ㄥ")),
        ("涔﹀啓/鎶勫啓閿欒", ("鎶勯敊", "鍐欓敊", "涔﹀啓", "绗﹀彿")),
        ("鏈畬鎴?, ("绌虹櫧", "鏈綔绛?, "鏈畬鎴?, "涓嶄細")),
        ("璁㈡鐥曡抗", ("璁㈡", "鏀规", "鍒掓帀", "鎿︽帀")),
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
    match = re.search(rf"(?:{label_pattern})\s*[=锛?]\s*([^銆俓n锛?]+)", source)
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
    if len(new) > len(old) and old in {"鏈煡", "鏈瘑鍒?}:
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
        f"{len(images)} 寮犲浘",
        f"seq {first_seq}-{last_seq}",
        f"{first.get('captured_at') or '鏈煡'} 鈫?{last.get('captured_at') or first.get('captured_at') or '鏈煡'}",
    ]
    if pages:
        parts.append("椤电爜鎻愮ず=" + "銆?.join(pages[:4]))
    if questions:
        parts.append("棰樺彿鎻愮ず=" + "銆?.join(questions[:4]))
    return truncate_text("锛?.join(parts), ASSET_SOURCE_SUMMARY_LIMIT)


def source_refs_from_images(images: list[dict]) -> dict:
    pages = list(dict.fromkeys(clean_asset_field(row.get("page_hint")) for row in images if clean_asset_field(row.get("page_hint"))))
    questions = list(dict.fromkeys(clean_asset_field(row.get("question_hint")) for row in images if clean_asset_field(row.get("question_hint"))))
    page_ref = "銆?.join(pages[:4])
    question_ref = "銆?.join(questions[:4])
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
        raw_items = [part.strip() for part in re.split(r"[锛?銆?\n]", value)]
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
        ("mistake_book", ("閿欓", "閿欏洜", "閿欒", "璁㈡", "涓嶄細", "鍋氶敊", "閿欏湪鍝噷")),
        ("knowledge_points", ("鐭ヨ瘑鐐?, "鑰冪偣", "姒傚康", "鍏紡", "鏂规硶", "鏉垮潡")),
        ("step_check", ("姝ラ", "杩囩▼", "绠楀紡", "鎺ㄥ", "瑙ｆ硶", "妫€鏌?)),
        ("answer_check", ("绛旀", "瀵逛笉瀵?, "鎵规敼", "鏍稿", "姝ｇ‘", "閿欒")),
        ("explain", ("璁茶В", "瑙ｉ噴", "鏁欐垜", "涓轰粈涔?, "鎬濊矾")),
        ("report", ("鎶ュ憡", "鎬荤粨", "澶嶇洏", "瀛︿範璁板綍", "瀹堕暱")),
        ("coach_hint", ("鍏堟彁绀?, "鎻愮ず浼樺厛", "鍏堢粰鎻愮ず", "涓嶈鐩存帴缁欑瓟妗?, "鍒洿鎺ョ粰绛旀", "鍚彂", "寮曞", "鍏堟兂", "鑷繁璇?)),
        ("coach_step", ("涓€姝ユ", "鍒嗘", "閫愭", "鎱㈡參璁?, "甯︾潃鍋?, "鎷嗘楠?)),
        ("coach_check_only", ("鍙鏌?, "鍙牳瀵?, "涓嶄唬鍋?, "鍙憡璇夊閿?, "鎵规敼妯″紡", "妫€鏌ョ瓟妗?)),
        ("coach_full", ("瀹屾暣璁茶В", "瀹屾暣瑙ｆ瀽", "璁查€?, "璇︾粏璁茶В")),
        ("review_plan", ("澶嶄範", "閿欓澶嶄範", "宸╁浐", "鍐嶇粌", "鍥為【", "鎺屾彙")),
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
    if any(keyword in text for keyword in ("鏈瘑鍒?, "鐪嬩笉娓?, "妯＄硦", "鍙嶅厜")):
        tags.append("capture_quality")
    if any(keyword in text for keyword in ("鑽夌", "璁㈡", "鏀规垚", "鎿﹂櫎", "鍒掓帀")):
        tags.append("revision_trace")
    return merge_tags(existing_tags or [], tags)


def tag_label(tag: str) -> str:
    labels = {
        "mistake_book": "鏁寸悊閿欓鏈?,
        "knowledge_points": "鎻愮偧鐭ヨ瘑鐐?,
        "step_check": "妫€鏌ユ楠?杩囩▼",
        "answer_check": "鏍稿绛旀",
        "explain": "鐢熸垚璁茶В",
        "report": "鐢熸垚瀛︿範鎶ュ憡",
        "coach_hint": "瀛︿範鏁欑粌锛氬厛鎻愮ず",
        "coach_step": "瀛︿範鏁欑粌锛氬垎姝ヨ",
        "coach_check_only": "瀛︿範鏁欑粌锛氬彧妫€鏌ヤ笉浠ｅ仛",
        "coach_full": "瀛︿範鏁欑粌锛氬畬鏁磋瑙?,
        "review_plan": "瀹夋帓澶嶄範宸╁浐",
        "capture_quality": "鎻愰啋琛ユ媿涓嶆竻鏅板唴瀹?,
        "revision_trace": "鍏虫敞璁㈡/鑽夌鍙樺寲",
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
    ("answer_check", ("answer_check", "瀛︿範鍦烘櫙=妫€鏌?, "妫€鏌ョ瓟妗?, "鍙鏌?, "鎵规敼", "鏍稿"), "瀛︿範鍦烘櫙=妫€鏌ョ瓟妗堬細鍏堝垽鏂/閿?涓嶆竻妤氾紝鍐嶇粰涓€鏉″彲鎵ц璁㈡姝ラ銆?),
    ("review", ("review", "瀛︿範鍦烘櫙=澶嶄範", "澶嶄範", "閿欓澶嶄範", "宸╁浐", "鍐嶇粌"), "瀛︿範鍦烘櫙=澶嶄範閿欓锛氫紭鍏堝洖椤鹃敊鍥犮€佺煡璇嗙偣鍜岀浉浼奸杩佺Щ锛屼笉鍙噸澶嶅師绛旀銆?),
    ("homework", ("homework", "瀛︿範鍦烘櫙=鍐欎綔涓?, "鍐欎綔涓?, "瀛︿範杩囩▼", "杩炴媿", "缁冧範鍐?), "瀛︿範鍦烘櫙=鍐欎綔涓氳褰曪細鎸佺画瑙傚療璇婚銆佷功鍐欍€佸仠椤垮拰璁㈡锛屽皯鎵撴柇锛屽璁板綍鍏抽敭鍙樺寲銆?),
    ("concept", ("concept", "瀛︿範鍦烘櫙=璁茬煡璇嗙偣", "璁茬煡璇嗙偣", "鐭ヨ瘑鐐?, "姒傚康", "鍏紡"), "瀛︿範鍦烘櫙=璁茬煡璇嗙偣锛氬厛鐢ㄥ鐢熺敾闈㈤噷鐨勯鍋氫緥瀛愶紝鍐嶆彁鐐兼蹇靛拰鏄撻敊鐐广€?),
    ("single_problem", ("single_problem", "瀛︿範鍦烘櫙=鎷嶉", "鎷嶉", "鎷嶄竴閬撻", "鍗曢", "棰樼洰瑙ｆ瀽"), "瀛︿範鍦烘櫙=鎷嶄竴閬撻锛氬厛璇嗗埆棰樼洰銆佸鐢熺瓟妗堝拰鍗＄偣锛屽啀鍥炵瓟褰撳墠闂銆?),
)
COACH_STYLE_CANDIDATES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("check_only", ("check_only", "鍥炵瓟鏂瑰紡=鍙鏌?, "鍙鏌?, "鍙牳瀵?, "涓嶄唬鍋?, "鍙憡璇夊閿?, "coach_check_only"), "鍥炵瓟鏂瑰紡=鍙鏌ワ細鍙垽鏂綋鍓嶇瓟妗?杩囩▼鏄惁姝ｇ‘锛屽繀瑕佹椂缁欎竴涓姝ｆ柟鍚戯紝涓嶅睍寮€浠ｅ仛銆?),
    ("hint_first", ("hint_first", "鍥炵瓟鏂瑰紡=鍏堟彁绀?, "鍏堟彁绀?, "鎻愮ず浼樺厛", "涓嶈鐩存帴缁欑瓟妗?, "鍒洿鎺ョ粰绛旀", "鍏堟兂", "鑷繁璇?, "coach_hint"), "鍥炵瓟鏂瑰紡=鍏堟彁绀猴細榛樿涓嶇洿鎺ヤ唬鍋氾紝鍏堢粰 1-2 涓叧閿彁绀哄拰瀛︾敓鍙互鍏堣瘯鐨勪竴灏忔銆?),
    ("step_by_step", ("step_by_step", "鍥炵瓟鏂瑰紡=鍒嗘璁?, "涓€姝ユ", "鍒嗘", "閫愭", "甯︾潃鍋?, "coach_step"), "鍥炵瓟鏂瑰紡=鍒嗘璁诧細鎶婃帹鐞嗘媶鎴愮煭姝ラ锛屾瘡姝ヨ鏄庝负浠€涔堣繖鏍峰仛銆?),
    ("full_explain", ("full_explain", "鍥炵瓟鏂瑰紡=瀹屾暣璁?, "瀹屾暣璁茶В", "瀹屾暣瑙ｆ瀽", "璁查€?, "coach_full"), "鍥炵瓟鏂瑰紡=瀹屾暣璁茶В锛氬鐢熸槑纭渶瑕佹椂缁欏畬鏁磋В娉曘€佺粨璁哄拰涓€涓珐鍥哄姩浣溿€?),
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
            "鍋忓ソ浼樺厛绾?assistant_focus > report_style > student_goal > inferred_needs锛?
            f"褰撳墠鐢熸晥鍦烘櫙={scene_key or '鏈寚瀹?}({scene_source or 'none'})锛?
            f"褰撳墠鐢熸晥鍥炵瓟鏂瑰紡={style_key or '鏈寚瀹?}({style_source or 'none'})銆?
        )
    return lines


def build_strategy_context(strategy: dict) -> str:
    lines: list[str] = []
    if strategy.get("student_goal"):
        lines.append(f"瀛︾敓/鐢ㄦ埛鏈洖鍚堣姹傦細{strategy['student_goal']}")
    if strategy.get("assistant_focus"):
        lines.append(f"褰撳墠鏅鸿兘浣撳叧娉ㄧ瓥鐣ワ細{strategy['assistant_focus']}")
    needs = strategy.get("inferred_needs") or []
    if needs:
        lines.append("绯荤粺宸叉帹鏂殑甯姪闇€姹傦細" + "銆?.join(tag_label(tag) for tag in needs))
    if strategy.get("report_style"):
        lines.append(f"鎶ュ憡椋庢牸瑕佹眰锛歿strategy['report_style']}")
    coach_lines = coach_preference_lines(strategy)
    if coach_lines:
        lines.append("瀛︿範鏁欑粌鍋忓ソ锛歕n" + "\n".join(f"- {line}" for line in coach_lines))
    if not lines:
        lines.append("瀛︾敓鏆傛湭杈撳叆鏄庣‘瑕佹眰锛涜鏍规嵁鐢婚潰鍔ㄦ€佸垽鏂槸鍚﹂渶瑕侀敊棰樻湰銆佺煡璇嗙偣銆佹楠ゆ鏌ャ€佺瓟妗堟牳瀵规垨鎶ュ憡鎬荤粨銆?)
    lines.append(
        "鍔ㄦ€佹墽琛岃鍒欙細鏈壒璇嗗埆瑕佷紭鍏堟湇鍔′笂杩拌姹傦紱鑻ョ敾闈㈡樉绀哄鐢熷湪璁㈡銆佸弽澶嶅仠鐣欍€佺瓟妗堢枒浼奸敊璇垨棰樺共娓呮櫚锛?
        "璇蜂富鍔ㄨˉ鍏呴敊棰樻湰鍊欓€夈€佺煡璇嗙偣銆侀敊璇師鍥犲拰涓嬩竴姝ュ府鍔╁缓璁€傚啓閿欓鏈€欓€夋椂灏介噺浣跨敤鐭瓧娈碉細"
        "绉戠洰=...锛涢〉鐮?...锛涢鍙?...锛涘鐢熺瓟妗?...锛涘弬鑰冪瓟妗?...锛涢敊鍥?...锛涢敊璇被鍨?...锛涜姝?...锛涗笅涓€姝?...銆?
    )
    lines.append(
        "瀛︿範鏁欑粌鎵ц瑙勫垯锛氫紭鍏堜績杩涘鐢熻嚜宸卞畬鎴愶紱闄ら潪瀛︾敓鏄庣‘瑕佹眰瀹屾暣绛旀鎴栧満鏅渶瑕佹牳瀵圭粨璁猴紝"
        "榛樿鍏堢粰鎻愮ず銆佹寚鍑哄崱鐐广€佺粰涓€涓彲绔嬪嵆鎵ц鐨勫皬浠诲姟銆傛鏌ョ被璇锋眰鍏堣瀵?閿?涓嶆竻妤氾紝鍐嶇粰涓嬩竴姝ヨ姝ｃ€?
        "闈㈠悜瀹堕暱鐨勬€荤粨璇风敤涓夊彞璇濊娓咃細瀛︿簡浠€涔堛€佸崱鍦ㄥ摢閲屻€佷笅涓€娆℃€庝箞澶嶄範銆?
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


# 姝ｅ湪鐢熸垚涓殑鍙鍖?(account_id, source_type, source_id)銆傜敤浜庡鍓嶇杞鍘婚噸锛?
# 閬垮厤涓€娆＄偣鍑诲湪鍚庡彴鍫嗗彔澶氫唤銆岀敓鎴愬彲瑙嗗寲璁茶В銆嶃€?
_VIZ_INFLIGHT: set[tuple[str, str, str]] = set()
# 钀藉簱涓?running 鐨勫彲瑙嗗寲瓒呰繃璇ョ鏁颁粛鏈畬鎴愬垯瑙嗕负銆屽凡澶辨晥銆嶏紝鍏佽閲嶆柊瑙﹀彂锛堥槻鍗℃锛夈€?
VIZ_INFLIGHT_STALE_SECONDS = 240


def _iso_within_seconds(value: object, seconds: float) -> bool:
    """鍒ゆ柇 ISO 鏃堕棿瀛楃涓叉槸鍚﹀湪鏈€杩?`seconds` 绉掑唴锛堢敤浜庡垽瀹氫换鍔℃槸鍚︿粛鏂伴矞鍦ㄨ窇锛夈€?""
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
                "绔嬩綋鍑犱綍",
                "绌洪棿鍑犱綍",
                "绌洪棿鎯宠薄",
                "绌洪棿鎬濈淮",
                "涓夎鍥?,
                "鎴潰",
                "姝ｆ柟浣?,
                "闀挎柟浣?,
                "妫遍敟",
                "妫辨煴",
                "鍦嗘煴",
                "鍦嗛敟",
                "鐞?,
                "绾块潰瑙?,
                "浜岄潰瑙?,
                "寮傞潰鐩寸嚎",
                "鐐瑰埌骞抽潰",
                "闈㈤潰鍨傜洿",
                "绾块潰鍨傜洿",
                "浣撶Н",
                "琛ㄩ潰绉?,
                "solid geometry",
                "dihedral",
                "line-plane",
            ),
        ),
        (
            "analytic_geometry",
            (
                "瑙ｆ瀽鍑犱綍",
                "鍦嗛敟鏇茬嚎",
                "妞渾",
                "鍙屾洸绾?,
                "鎶涚墿绾?,
                "鐒︾偣",
                "鍑嗙嚎",
                "寮﹂暱",
                "杞ㄨ抗",
                "鏂滅巼",
                "绂诲績鐜?,
                "鏁伴噺绉?,
                "瀹氱偣",
                "瀹氬€?,
                "鍒囩嚎",
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
                "鍑犱綍",
                "鍥惧舰",
                "鍥惧儚",
                "鍧愭爣绯?,
                "鍚戦噺",
                "骞抽潰鐩磋鍧愭爣绯?,
                "鍑芥暟鍥惧儚",
                "鏃嬭浆",
                "骞崇Щ",
                "鐩镐技",
                "鍏ㄧ瓑",
                "瑙掑害",
                "闈㈢Н",
                "鍙鍖?,
                "婊戝潡",
                "鍔ㄦ€佹紨绀?,
                "浜や簰",
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
        "solid_geometry": "绔嬩綋鍑犱綍/绌洪棿鎬濈淮",
        "analytic_geometry": "瑙ｆ瀽鍑犱綍/鍦嗛敟鏇茬嚎",
        "geometry_or_spatial": "鍑犱綍/鍥惧舰鍙鍖?,
    }.get(topic_type or "geometry_or_spatial", "鍑犱綍/鍥惧舰鍙鍖?)
    return {
        "eligible": True,
        "topic_type": topic_type or "geometry_or_spatial",
        "reason": f"{topic_label}锛歿', '.join(dict.fromkeys(hits[:6]))}",
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
        return "<!doctype html>\n<html lang=\"zh-CN\">\n<head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>鍙鍖栬瑙?/title></head>\n<body>\n" + stripped + "\n</body>\n</html>"
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
        title = source.get("question") or source.get("answer") or "浜掑姩鏁欏鍙鍖?
    elif source_type == "analysis":
        title = source.get("content") or "瑙ｆ瀽鍙鍖?
    else:
        title = source.get("title") or source.get("text") or "浜掑姩鏁欏鍙鍖?
    title = re.sub(r"\s+", " ", str(title or "")).strip()
    title = re.sub(r"[#*_`<>]", "", title)
    if len(title) > 36:
        title = title[:34].rstrip() + "..."
    prefix = {
        "solid_geometry": "绔嬩綋鍑犱綍",
        "analytic_geometry": "瑙ｆ瀽鍑犱綍",
        "geometry_or_spatial": "鍙鍖栬瑙?,
    }.get(topic_type, "鍙鍖栬瑙?)
    return f"{prefix}锛歿title}" if title else prefix


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
You are generating an interactive HTML teaching artifact for 鐭ヨ繘鎷嶅.
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
{extra_instruction or "鏃?}

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
                f"瀛︾敓闂锛歿source.get('question') or ''}",
                f"AI 鍥炵瓟锛歿source.get('answer') or ''}",
                f"瑙﹀彂鏂瑰紡锛歿source.get('trigger_type') or ''}",
                f"鍥惧儚涓婁笅鏂囷細{source.get('image_context_mode') or ''}",
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
                f"瑙ｆ瀽鑼冨洿锛歿source.get('scope') or ''}",
                f"瑙ｆ瀽鍐呭锛歿source.get('content') or ''}",
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
                    title or "鏁欏鍙鍖?,
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
            f"鏁欏鍙鍖栫敓鎴愬け璐ワ細source={source_type}/{source_id} error={truncate_text(error, 180)}",
            session_id=resolved_session_id or None,
            source="visualization",
            level="error",
        )
    else:
        emit_log(
            f"鏁欏鍙鍖栫敓鎴愬畬鎴愶細source={source_type}/{source_id} url={result.get('url')}",
            session_id=resolved_session_id or None,
            source="visualization",
        )
    return result


def asset_document_payload(asset_kind: str, row: dict) -> dict:
    if asset_kind == "learning":
        lines = [
            f"绫诲瀷锛歿row.get('item_type') or 'learning'}",
            f"鏍囬锛歿row.get('title') or ''}",
            f"鍐呭锛歿row.get('content') or ''}",
            f"绉戠洰锛歿row.get('subject') or ''}",
            f"浣嶇疆锛歿row.get('location_ref') or ''}",
            f"椤电爜锛歿row.get('page_ref') or ''}",
            f"棰樺彿锛歿row.get('question_ref') or ''}",
            f"鏉ユ簮锛歿row.get('source_summary') or ''}",
        ]
    else:
        knowledge_points = json_list(row.get("knowledge_points"))
        lines = [
            f"绫诲瀷锛氶敊棰樻湰鍊欓€?,
            f"鏍囬锛歿row.get('title') or ''}",
            f"鐘舵€侊細{row.get('status') or ''}",
            f"澶嶄範鐘舵€侊細{row.get('review_state') or ''}",
            f"涓嬫澶嶄範锛歿row.get('next_review_at') or ''}",
            f"绉戠洰锛歿row.get('subject') or ''}",
            f"浣嶇疆锛歿row.get('location_ref') or ''}",
            f"椤电爜锛歿row.get('page_ref') or ''}",
            f"棰樺彿锛歿row.get('question_ref') or ''}",
            f"棰樼洰锛歿row.get('question_text') or ''}",
            f"瀛︾敓绛旀锛歿row.get('student_answer') or ''}",
            f"鍙傝€冪瓟妗堬細{row.get('expected_answer') or ''}",
            f"閿欏湪鍝噷锛歿row.get('error_reason') or row.get('evidence') or ''}",
            f"閿欒绫诲瀷锛歿row.get('error_type') or ''}",
            f"璁㈡锛歿row.get('correction') or ''}",
            f"涓嬩竴姝ワ細{row.get('next_action') or ''}",
            f"鐭ヨ瘑鐐癸細{'銆?.join(knowledge_points)}",
            f"璇佹嵁锛歿row.get('evidence') or ''}",
            f"鏉ユ簮锛歿row.get('source_summary') or ''}",
        ]
    body = truncate_text("\n".join(line for line in lines if not line.endswith("锛?)), ASSET_DOCUMENT_BODY_LIMIT)
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
    # 鍊欓€変笉鎺掕繘澶嶄範闃熷垪锛歳eview_state=new銆佷笉璁惧埌鏈熸椂闂达紱瀵煎叆(confirmed)鍚庢墠鐢?update_mistake_item 鎺掔▼銆?
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
        body.get("error_reason") or body.get("reason") or "鐢ㄦ埛浠庢湰杞?AI 鍥炵瓟鎵嬪姩鍔犲叆閿欓鏈?,
        MISTAKE_REASON_LIMIT,
    )
    correction = clean_user_text(body.get("correction") or answer, ASSET_SOURCE_SUMMARY_LIMIT)
    next_action = clean_user_text(
        body.get("next_action") or "澶嶄範鏃跺厛澶嶈堪閿欏洜锛屽啀鍋氫竴閬撶浉浼奸銆?,
        ASSET_SOURCE_SUMMARY_LIMIT,
    )
    title = clean_user_text(body.get("title") or question or answer or "鎵嬪姩鍔犲叆閿欓", LEARNING_ITEM_TITLE_LIMIT)
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
            "source_summary": clean_user_text(body.get("source_summary") or "鏉ヨ嚜闂瓟鎵嬪姩鍔犲叆閿欓鏈?, ASSET_SOURCE_SUMMARY_LIMIT),
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
    # 姝ｅ紡閿欓鏈彧鐢变富鍔ㄦ媿棰?鎵规敼鍠傚吇锛堢簿鍑?JSON锛夈€傝鍔ㄦ櫤鑳借瀵?observation)涓嶇洿鎺ュ啓閿欓鏈紝
    # 鑰屾槸寮傛鎻愬彇鈥滃彲鐤戦敊棰樺€欓€夆€?status=candidate)锛岀敱瀛︾敓浜哄伐纭鍚庢墠瀵煎叆姝ｅ紡閿欓鏈紝
    # 閬垮厤鈥滈鐩€楁椂绾跨储/宸紓棰樼洰/鐢婚潰闈欐鈥濊繖绫昏瀵熷櫔澹扮亴鍏ャ€?
    if detection_method == "observation":
        mistakes = parse_structured_mistakes(
            content, items, marker=_MISTAKE_CANDIDATE_JSON_MARKER, force_status="candidate"
        )
    else:
        # 浼樺厛瑙ｆ瀽妯″瀷鐩村嚭鐨勬満璇?JSON锛堢簿鍑嗭級锛涙ā鍨嬫病鍚?JSON 鏃堕€€鍥炴棫鍚彂寮忥紙宸插甫鍘诲櫔闂ㄦ锛夛紝
        # 閬垮厤涓诲姩鎷嶉鐨勯敊棰橀浂鍙洖銆?
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
    labels = {"question": "棰樼洰", "section": "鏉垮潡", "knowledge": "鐭ヨ瘑鐐?, "answer": "浣滅瓟"}
    lines = [
        (
            f"{index + 1}. {labels.get(row['item_type'], row['item_type'])}锛歿row['title']} "
            f"锛坒irst={row['first_seen_at'] or '鏈煡'}锛宭ast={row['last_seen_at'] or '鏈煡'}锛岃瘉鎹?{row['evidence_count']}锛?
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
    "棰樺彿": "棰樼洰",
    "棰樼洰": "棰樼洰",
    "闂": "棰樼洰",
    "宸茬煡": "鍏抽敭鏉′欢",
    "鏉′欢": "鍏抽敭鏉′欢",
    "鍏抽敭鏉′欢": "鍏抽敭鏉′欢",
    "瀛︾敓绛旀": "瀛︾敓绛旀",
    "浣滅瓟": "瀛︾敓绛旀",
    "妫€鏌ョ粨鏋?: "妫€鏌ョ粨鏋?,
    "缁撴灉": "妫€鏌ョ粨鏋?,
    "姝ｇ‘璁＄畻": "瑙ｉ姝ラ",
    "璁＄畻": "瑙ｉ姝ラ",
    "瑙ｆ硶": "瑙ｉ姝ラ",
    "鎬濊矾": "瑙ｉ鎬濊矾",
    "瑙ｉ鎬濊矾": "瑙ｉ鎬濊矾",
    "姝ラ": "瑙ｉ姝ラ",
    "杩囩▼": "瑙ｉ姝ラ",
    "閿欏洜": "閿欏洜鎻愰啋",
    "閿欒鍘熷洜": "閿欏洜鎻愰啋",
    "璁㈡": "璁㈡寤鸿",
    "鏀规": "璁㈡寤鸿",
    "缁撹": "缁撹",
    "绛旀": "缁撹",
    "鐭ヨ瘑鐐?: "鐭ヨ瘑鐐?,
    "鐭ヨ瘑鐐瑰€欓€?: "鐭ヨ瘑鐐?,
    "鐭ヨ瘑鏉垮潡": "鐭ヨ瘑鐐?,
    "鑰冪偣": "鐭ヨ瘑鐐?,
    "鍏堟兂涓€鎯?: "鍏堟兂涓€鎯?,
    "鍏堣瘯涓€涓?: "鍏堟兂涓€鎯?,
    "鎻愮ず": "鍏堟兂涓€鎯?,
    "涓嬩竴姝ュ皬浠诲姟": "涓嬩竴姝ュ皬浠诲姟",
    "灏忎换鍔?: "涓嬩竴姝ュ皬浠诲姟",
    "杩介棶寤鸿": "杩介棶寤鸿",
    "鍙互杩介棶": "杩介棶寤鸿",
    "涓嬩竴姝?: "杩介棶寤鸿",
}
QA_SECTION_CLASS = {
    "棰樼洰": "topic",
    "鍏抽敭鏉′欢": "facts",
    "瀛︾敓绛旀": "student",
    "妫€鏌ョ粨鏋?: "check",
    "瑙ｉ鎬濊矾": "idea",
    "瑙ｉ姝ラ": "steps",
    "閿欏洜鎻愰啋": "warning",
    "璁㈡寤鸿": "fix",
    "缁撹": "result",
    "鐭ヨ瘑鐐?: "knowledge",
    "鍏堟兂涓€鎯?: "idea",
    "涓嬩竴姝ュ皬浠诲姟": "follow",
    "杩介棶寤鸿": "follow",
}
QA_SECTION_ORDER = [
    "棰樼洰",
    "鍏抽敭鏉′欢",
    "鍏堟兂涓€鎯?,
    "瀛︾敓绛旀",
    "妫€鏌ョ粨鏋?,
    "瑙ｉ鎬濊矾",
    "瑙ｉ姝ラ",
    "閿欏洜鎻愰啋",
    "璁㈡寤鸿",
    "缁撹",
    "鐭ヨ瘑鐐?,
    "涓嬩竴姝ュ皬浠诲姟",
    "杩介棶寤鸿",
]
# 缁欐瘡涓爮鐩厤涓€涓鐢熺湅寰楁噦鐨勫浘鏍囷紝渚夸簬鎵銆?
QA_SECTION_ICON = {
    "棰樼洰": "馃摉",
    "鍏抽敭鏉′欢": "馃攽",
    "鍏堟兂涓€鎯?: "馃挱",
    "瀛︾敓绛旀": "鉁忥笍",
    "妫€鏌ョ粨鏋?: "鉁?,
    "瑙ｉ鎬濊矾": "馃挕",
    "瑙ｉ姝ラ": "馃獪",
    "閿欏洜鎻愰啋": "鈿狅笍",
    "璁㈡寤鸿": "馃洜锔?,
    "缁撹": "馃幆",
    "鐭ヨ瘑鐐?: "馃摎",
    "涓嬩竴姝ュ皬浠诲姟": "馃殌",
    "杩介棶寤鸿": "馃挰",
}
QA_SECTION_PATTERN = re.compile(r"^\s*(?:#{1,4}\s*)?(?:[-*]\s*)?([A-Za-z\u4e00-\u9fff]{2,12})[锛?]\s*(.*)$")
QA_NUMBERED_PATTERN = re.compile(r"^\s*(?:\d+[\.銆?]|[锛?]\d+[锛?]|[-*])\s*(.+)$")


def normalize_qa_section_title(raw: str) -> str:
    title = re.sub(r"\s+", "", raw or "").strip("#:-锛?")
    return QA_SECTION_ALIASES.get(title, title if title in QA_SECTION_CLASS else "")


def linkify_escaped_text(escaped_text: str) -> str:
    pattern = re.compile(r"(https?://[^\s<]+)")
    return pattern.sub(lambda match: f'<a href="{match.group(1)}" target="_blank" rel="noopener">{match.group(1)}</a>', escaped_text)


def inline_qa_markup(text: object) -> str:
    escaped = html.escape(str(text or "").strip())
    escaped = re.sub(r"(\d+(?:\.\d+)?)\s*([+\-脳x*/梅=])\s*(\d+(?:\.\d+)?)", r'<span class="qa-math">\1 \2 \3</span>', escaped)
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

    # 娌℃湁浠讳綍鏍囩鏃讹紝璇存槑杩欐槸涓€娈电畝鐭殑鑷劧鍥炲锛堟牳瀵?闂茶亰/姒傚康涓€鍙ヨ瘽锛夛紝
    # 鐩存帴褰撲綔鏅€氭钀芥覆鏌擄紝涓嶈纭杩涒€滆В棰樻楠も€濆崱鐗囷紝淇濇寔绠€绾︺€?
    if loose and not sections:
        plain = "".join(
            f"<p>{inline_qa_markup(line)}</p>"
            for line in loose
            if str(line).strip()
        )
        return truncate_text(f'<div class="qa-rich-answer qa-plain-reply">{plain}</div>', QA_HTML_CHAR_LIMIT)
    if loose:
        sections.setdefault("瑙ｉ鎬濊矾", []).extend(loose)

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


# 澶фā鍨嬪父鎶婃墍鏈夆€滈鐩?鍏抽敭鏉′欢/瀛︾敓绛旀/妫€鏌ョ粨鏋?...鈥濇爣绛惧拰鍒嗛鎸ゅ湪涓€琛岃繑鍥烇紝
# 鍓嶇鎸夎瑙ｆ瀽鏃跺氨鍙樻垚涓€鏁存娌℃湁鎹㈣鐨勬枃瀛椼€傝繖閲屽湪宸茬煡鏍囩鍜屽垎棰樺彿鍓嶈ˉ鍥炴崲琛岋紝
# 璁╃綉椤靛拰 iOS 閮借兘鎸夋钀?鍗＄墖娓叉煋銆傚凡缁忔崲琛岀殑鍐呭涓嶄細琚噸澶嶆媶鍒嗐€?
QA_REFLOW_LABELS = sorted(set(QA_SECTION_ALIASES), key=len, reverse=True)
QA_REFLOW_LABEL_PATTERN = re.compile(
    r"(?<!\n)(?<![涓€-榭縘)[^\S\n]*"
    r"((?:" + "|".join(re.escape(label) for label in QA_REFLOW_LABELS) + r")[锛?])"
)
QA_REFLOW_PROBLEM_PATTERN = re.compile(r"(?<!\n)[^\S\n]*(棰榎s*\d+\s*[锛?])")
QA_REFLOW_OPTION_PATTERN = re.compile(
    r"(?<!\n)[^\S\n]*(\d+\s*[\.銆?]\s*(?:涓句竴鍙嶄笁|鎬荤粨鐭ヨ瘑鐐箌鎸夌敤鎴峰亸濂?)"
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


# 绾棽鑱?鎵撴嫑鍛?鑷磋阿绛夐棶棰樸€傝繖绫诲洖鍚堜笉璇ュ脊銆屼妇涓€鍙嶄笁/鍔犲叆閿欓鏈?鐢熸垚鍙鍖栥€嶇瓑瀛︿範鍔ㄤ綔銆?
_QA_CHITCHAT_PHRASES = {
    "hi", "hello", "hey", "yo", "ok", "okay", "thanks", "thx", "bye",
    "浣犲ソ", "鎮ㄥソ", "鍝堝柦", "鍡?, "鍦ㄥ悧", "鍦ㄤ笉鍦?, "鍦ㄤ箞", "浣犲湪鍚?,
    "鏃?, "鏃╁畨", "鏃╀笂濂?, "涓崍濂?, "涓嬪崍濂?, "鏅氫笂濂?, "鏅氬畨",
    "璋㈣阿", "璋㈣阿浣?, "璋㈠暒", "澶氳阿", "杈涜嫤浜?, "濂界殑", "濂芥淮", "鍡?, "鍝?, "鍝﹀摝",
    "鎷滄嫓", "鍐嶈", "鏅氱偣鑱?, "鍝堝搱", "鍝堝搱鍝?, "娴嬭瘯", "test", "浣犳槸璋?, "浣犲彨浠€涔?,
}


def qa_turn_actionable(question: str, answer: str) -> bool:
    """鍒ゆ柇杩欒疆闂瓟鏄惁鍊煎緱灞曠ず瀛︿範鍔ㄤ綔鎸夐挳锛堜妇涓€鍙嶄笁/閿欓鏈?鍙鍖栫瓑锛夈€?
    鐩殑锛氱敤鎴峰彧鍙戙€宧i/浣犲ソ/璋㈣阿銆嶈繖绫婚棽鑱婃椂锛屼笉瑕佷竴鏈缁忓湴缁欏涔犳寜閽€?
    绛栫暐淇濆畧锛氶粯璁?True锛屼粎褰撻棶棰樻槑鏄炬槸闂茶亰/杩囩煭涓旀棤瀛︿範淇″彿鏃舵墠 False锛岄伩鍏嶈浼ょ湡瀹為棶棰樸€?""
    q = (question or "").strip()
    if not q:
        # 鏃犻棶棰樻枃鏈紙澶氫负鎷嶉/璇煶甯﹀浘锛夆€斺€旀寜鐪熷疄瀛︿範澶勭悊銆?
        return True
    core = re.sub(r"[\s锛屻€傦紒锛熴€侊紱锛?.!?;:~锝炩€β穃-鈥?)锛堬級\"'鈥溾€濃€樷€?@#]+", "", q.lower())
    if not core:
        return True
    if core in _QA_CHITCHAT_PHRASES:
        return False
    # 瀛︿範淇″彿锛氭暟瀛椼€佸瓧姣嶅彉閲忋€佸父瑙佸绉?姹傝В绫昏瘝銆傚懡涓垯涓€瀹氱畻瀹炶川闂銆?
    if any(ch.isdigit() for ch in core):
        return True
    study_markers = (
        "涓轰粈涔?, "鎬庝箞", "濡備綍", "姹?, "瑙?, "绠?, "璇佹槑", "鎺ㄥ", "棰?, "绛旀", "鍏紡",
        "璁?, "瑙ｉ噴", "浠€涔堟槸", "鍖哄埆", "涓句緥", "渚嬪瓙", "缈昏瘧", "鍗曡瘝", "璇硶", "榛樺啓",
        "+", "-", "脳", "梅", "=", "x", "y", "鈭?, "鈭?,
    )
    if any(marker in q.lower() for marker in study_markers):
        return True
    # 鏃㈡棤瀛︿範淇″彿銆侀棶棰樺張寰堢煭锛堚墹6 涓湁鏁堝瓧绗︼級锛屽鍗婃槸闂茶亰銆?
    if len(core) <= 6:
        return False
    return True


def qa_answer_is_unhelpful_image_failure(answer: str) -> bool:
    text = re.sub(r"\s+", "", answer or "")
    if not text:
        return False
    image_failure_terms = (
        "鏈瘑鍒?,
        "娌¤瘑鍒?,
        "鏃犳硶璇嗗埆",
        "娌℃湁璇嗗埆",
        "鐪嬩笉娓?,
        "娌＄湅娓?,
        "鏃犳硶鐪嬫竻",
        "鏃犳硶纭鍥剧墖",
        "鍥剧墖涓嶆竻妤?,
        "鐓х墖涓嶆竻妤?,
        "鎷嶆憚娓呮櫚",
        "娓呮櫚棰樼洰",
        "娓呮櫚鐨勯鐩?,
        "鏈夋晥棰樼洰",
        "鏃犳湁鏁堥鐩?,
        "鏈夋晥杈撳叆",
        "鏃犳湁鏁堣緭鍏?,
        "杈撳叆鍏蜂綋闂",
        "涓婁紶棰樼洰",
        "涓婁紶鍥剧墖",
        "涓婁紶娓呮櫚",
        "绛夊緟鏈夋晥杈撳叆",
        "閲嶆柊鎷?,
        "閲嶆媿",
        "绉诲埌闀滃ご",
        "瀵瑰噯",
    )
    return any(term in text for term in image_failure_terms)


def qa_safe_followup_fallback_answer(question: str) -> str:
    return (
        "瑙ｉ鎬濊矾锛氭垜浠帴鐫€鍒氭墠閭ｉ璁层€俓n"
        "姝ラ锛氬厛鎶婁笂涓€杞凡缁忕‘瀹氱殑鏉′欢鏀惧湪涓€璧凤紝鍐嶈В閲婁綘杩介棶鐨勯偅涓€姝ヤ负浠€涔堟垚绔嬶紝鏈€鍚庡啓鎴愪竴鍙ョ粨璁恒€俓n"
        "缁撹锛氫綘鍙互缁х画闂叿浣撳摢涓€姝ャ€?
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
        parts.append("宸叉湁鐢诲儚锛歕n" + truncate_text(existing_profile, 1400))
    if recent_lines:
        parts.append("鏈€杩戣緭鍏ユ憳瑕侊細\n" + "\n".join(recent_lines))
    parts.append("鏁寸悊寤鸿锛氱户缁牴鎹敤鎴风殑鎻愰棶鏂瑰紡銆佸绉戝崱鐐广€佸亸濂藉弽棣堝拰甯歌閿欒鏇存柊鐢诲儚銆?)
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
浣犳槸瀛︿範闄即 App 鐨勮蹇嗘暣鐞嗗櫒銆傝鎶婃渶杩戠敤鎴峰彂鍑虹殑璇煶杞枃瀛楀拰绾枃瀛椾俊鎭紝鏁寸悊鎴愬彲鎸佺画鏇存柊鐨勭敤鎴风敾鍍忋€?

瑕佹眰锛?
- 鍙褰曞鍚庣画瀛︿範闄即鏈夊府鍔╃殑淇℃伅锛氬涔犵洰鏍囥€佸父瑙佺鐩€佽〃杈句範鎯€佸亸濂姐€佹槗閿欑偣銆侀渶瑕侀伩鍏嶇殑鍥炵瓟鏂瑰紡銆?
- 涓嶈閲嶅閫愭潯鎶勫綍鍘熻瘽锛涜鍚堝苟褰掔撼銆?
- 濡傛灉璇佹嵁涓嶈冻锛屽啓鈥滄殏鏈舰鎴愮ǔ瀹氬垽鏂€濄€?
- 杈撳嚭涓枃锛岀粨鏋勭揣鍑戯紝閫傚悎涓嬫闂瓟浣滀负涓婁笅鏂囧€欓€夈€?

宸叉湁鐢诲儚锛?
{existing_profile or "鏆傛棤"}

鏈€杩戠敤鎴疯緭鍏ワ細
{chr(10).join(event_lines) if event_lines else "鏆傛棤"}
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
        emit_log(f"璁板繂鏁寸悊 LLM 澶辫触锛屼娇鐢ㄦ湰鍦版憳瑕侊細{truncate_text(str(exc), 180)}", level="warning", source="memory")
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
    emit_log(f"璁板繂鏁寸悊瀹屾垚锛歿len(events)} 鏉¤緭鍏?, level="info", source="memory")
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
Write a concise, targeted answer. Do NOT fill a fixed template 鈥?choose the layout that actually fits THIS question, and keep it short.

Layout rules (important):
- Use short labeled lines only when they help. Put each label on its own line with a real newline; never put two labels on one line; start each 棰樺彿 (棰?銆侀2) on its own line.
- Include a label ONLY if it carries real, specific content for this question. Omit any label that would be empty, generic, or just repeats the question. A tight answer with 1-3 labels beats one that fills every label.
- Match the layout to the question type, for example:
  路 鍙槸鏍稿瀵归敊: 涓昏缁?妫€鏌ョ粨鏋滐紝蹇呰鏃跺啀鍔?閿欏洜/璁㈡锛涢€氬父涓嶉渶瑕?棰樼洰/鍏抽敭鏉′欢/瑙ｉ姝ラ銆?
  路 璁╀綘璁茶繖閬撻: 棰樼洰锛堜竴鍙ワ級+ 瑙ｉ鎬濊矾鎴栨楠?+ 缁撹锛涘彧鏈夋潯浠跺鎴栧瓨鍦ㄥ共鎵版潯浠舵椂鎵嶇敤 鍏抽敭鏉′欢銆俬int_first 鍋忓ソ涓嬫敼鐢?鍏堟兂涓€鎯?浠ｆ浛鐩存帴缁欐楠ゃ€?
  路 姒傚康/鐭ヨ瘑鐐规彁闂? 鐢?鐭ヨ瘑鐐?鎶婃蹇佃娓呮 + 涓€涓皬渚嬪瓙锛屼笉瑕佺‖濂?棰樼洰/瀛︾敓绛旀銆?
  路 绠€鍗曡拷闂垨闂茶亰: 鐩存帴鐢ㄤ竴涓ゅ彞鑷劧涓枃鍥炵瓟锛屽彲浠ュ畬鍏ㄤ笉鐢ㄤ换浣曟爣绛俱€?
- If you do not use labels, just answer in one or two short plain sentences.

Available labels (pick only the subset you need; keep this relative order when several appear):
棰樼洰銆佸叧閿潯浠躲€佸厛鎯充竴鎯炽€佸鐢熺瓟妗堛€佹鏌ョ粨鏋溿€佽В棰樻€濊矾銆佹楠ゃ€侀敊鍥犮€佽姝ｃ€佺粨璁恒€佺煡璇嗙偣銆佷笅涓€姝ュ皬浠诲姟銆佽拷闂缓璁?

Plain-text math formatting rules:
- Write student answers, formulas, and units as ordinary readable text, not Markdown or LaTeX.
- Never wrap math in dollar signs. Do not output $...$, $$...$$, \\(...\\), or \\[...\\].
- Do not use LaTeX commands such as \\times, \\div, \\frac, \\sqrt, or cm^2 in visible answers.
- Prefer symbols and Chinese units directly, for example: 瀛︾敓绛旀锛?0脳6梅2=30骞虫柟鍘樼背.
- If copying the student's handwritten work, preserve the math meaning but normalize display characters, for example 脳, 梅, =, 骞虫柟鍘樼背.

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

Semantically related items from this student's own knowledge base (past mistakes/knowledge points retrieved by meaning, ranked by relevance to the current question; use ONLY when genuinely relevant, e.g. for review, similar-problem warnings, or concept reinforcement 鈥?never as factual proof):
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
- If you rely on an asset, mention it naturally in Chinese in one short phrase, for example: 鈥滅粨鍚堜綘涔嬪墠甯搁敊鐨勫崟浣嶆崲绠?..鈥?
- When Strategy includes 褰撳墠鐢熸晥鍥炵瓟鏂瑰紡 or 褰撳墠鐢熸晥鍦烘櫙, treat that as the only active coach preference even if inferred_needs lists other possible tags.
- Follow the learning coach preference in Strategy and client_context. If the preference is hint_first, lead with 鍏堟兂涓€鎯?and avoid giving the final answer until the student asks or checking requires it.
- If the preference is check_only, do not solve the whole problem. Say whether the visible answer/process is correct, wrong, or unclear, then give one correction direction.
- If the preference is step_by_step or full_explain, keep steps compact; add one actionable 涓嬩竴姝ュ皬浠诲姟 only when it genuinely helps.
- Add 鍏堟兂涓€鎯?or 涓嬩竴姝ュ皬浠诲姟 only when it gives a real next action worth doing; for a simple check, a concept reply, or chit-chat, leave it out.
- Do NOT force a 涓嬩竴姝?/ 杩介棶寤鸿 block on every answer. Include 杩介棶寤鸿 (one short line) only when an obvious useful follow-up exists; otherwise end naturally without it.
- If inferred_student_intent is correction_check, answer as a same-dialogue correction review: compare the student's latest visible work with the prior QA context, decide whether the revision is now correct, and give one precise next correction if needed.
- If inferred_student_intent is answer_check or visual_check, inspect the provided/current frame first and report what is correct, wrong, or unclear. Keep it connected to the referenced prior problem when the wording says "again", "this", "here", "changed", "鏀瑰畬", "鍐嶇湅鐪?, or similar.
- If context says current_image_rejected=true or context_mode is fallback/text-only, silently ignore the new capture. Do not say the image was not recognized; answer the student's spoken follow-up from recent QA turns, prior valid image context, and structured learning context.
- For math, independently recompute the visible expression before speaking the final answer. Check the final numeric result at least two ways when possible.
- When using decomposition or carrying, do not add the original whole number again after adding its decomposed parts.
- Only ask the student to move or retake the material when there is no prior context to answer from. During follow-up, prefer continuing the explanation from prior context.
- If you mention a student's answer is wrong, state the correct answer and one short correction step.
- 閫変腑棰樿仛鐒﹁鍒欙紙浠呭綋 focus 鍚?region/crop/question_text/regions 鏃剁敓鏁堬紝鑰佸鎴风涓嶅甫杩欎簺瀛楁鏃跺拷鐣ユ湰娈碉級锛?
  路 鑻?focus.crop 涓虹湡锛氬綋鍓嶅抚宸茶瑁佸壀涓哄崟鐙竴閬撻锛屾妸鏁村抚褰撲綔鍞竴鐩爣棰樹綔绛旓紝涓嶈鎻愨€滅敾闈㈤噷杩樻湁鍒殑棰樷€濇垨璁╁鐢熷湀閫夈€?
  路 鑻?focus.region 瀛樺湪涓?focus.crop 涓嶄负鐪燂細鐢婚潰閲屽彲鑳芥湁澶氶亾棰橈紝鍙洖绛?region 褰掍竴鍖栨锛坸,y,w,h锛屽彇鍊糩0,1]锛屽師鐐瑰乏涓娿€亁 鍙?y 涓嬨€佺浉瀵瑰凡姊舰鏍℃鍚庣殑绔栫洿涓婁紶鍥撅級妗嗕綇鐨勯偅涓€閬擄紝鍏跺畠棰樺拷鐣ャ€佷笉瑕侀『甯﹁瑙ｃ€?
  路 鑻?focus.question_text 瀛樺湪锛氫互瀹冧綔涓洪€変腑棰樼殑棰樺共閿氱偣锛涘綋瀹冧笌鐢婚潰鍐呭鍐茬獊鏃朵互鐢婚潰涓哄噯锛屽苟绠€鐭寚鍑哄啿绐併€?
  路 focus.regions 鍙敤浜庡府鍔╀綘鐞嗚В鏁撮〉甯冨眬锛屼笉瑕佹嵁姝ら€愰浣滅瓟銆?
- Keep each label compact. Prefer one idea per line. Avoid long Markdown paragraphs and avoid Markdown tables.
- Especially in 瀛︾敓绛旀銆佹楠ゃ€佺粨璁? use plain text math only; no Markdown math, no LaTeX, no dollar signs.

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
    """鏅鸿兘瑙傚療寮傛鎻愬彇鍑虹殑鈥滃彲鐤戦敊棰樺€欓€夆€?status=candidate)锛岀瓑瀛︾敓浜哄伐纭瀵煎叆銆?
    鍒绘剰涓庢寮忛敊棰樻湰/澶嶄範闃熷垪鍒嗗紑锛歳eview_queue_items 涓嶅惈 candidate锛岃繖閲屽彧鍚?candidate銆?""
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
        # 鍊欓€夎繕娌¤瀛︾敓纭锛屼笉璁″叆鐭ヨ瘑鐐?閿欏洜/绉戠洰鐢诲儚锛岄伩鍏嶆湭纭鐨勭寽娴嬫薄鏌撳鎯呫€?
        if row.get("status") == "candidate":
            continue
        is_ignored = row.get("status") == "ignored" or row.get("review_state") == "ignored"
        is_mastered = row.get("status") == "mastered" or row.get("review_state") == "mastered"
        if not is_ignored and not is_mastered:
            weight = max(1, int(row.get("review_count") or 0) + 1)
            points = json_list(row.get("knowledge_points"))
            for point in points:
                for part in re.split(r"[銆?锛?锛沑n]+", point):
                    add_metric(knowledge_metrics, part, weight=weight, subject=row.get("subject"), last_seen_at=row.get("last_seen_at"))
            if not points:
                add_metric(knowledge_metrics, row.get("title"), weight=1, subject=row.get("subject"))
            add_metric(error_metrics, row.get("error_type") or row.get("error_reason"), weight=weight, subject=row.get("subject"))
        subject = row.get("subject") or "鏈瘑鍒?
        add_metric(subject_metrics, subject, weight=1)
        subject_metrics[subject]["mistake_count"] = subject_metrics[subject].get("mistake_count", 0) + 1
        if row.get("status") in {"corrected", "mastered"}:
            subject_metrics[subject]["progressed_count"] = subject_metrics[subject].get("progressed_count", 0) + 1
        if is_ignored:
            subject_metrics[subject]["ignored_count"] = subject_metrics[subject].get("ignored_count", 0) + 1
        if is_mastered:
            subject_metrics[subject]["mastered_count"] = subject_metrics[subject].get("mastered_count", 0) + 1

    for row in learning_rows:
        subject = row.get("subject") or "鏈瘑鍒?
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
        return "鏆傛棤缁撴瀯鍖栧涔犳潯鐩€?, False
    labels = {"question": "棰樼洰", "section": "鏉垮潡", "knowledge": "鐭ヨ瘑鐐?, "answer": "浣滅瓟"}
    lines = [
        (
            f"{index + 1}. {labels.get(row['item_type'], row['item_type'])}锛歿row['content']}锛?
            f"绉戠洰={row.get('subject') or '鏈瘑鍒?}锛涗綅缃?{row.get('location_ref') or '鏈瘑鍒?}锛?
            f"first={row['first_seen_at'] or '鏈煡'}锛沴ast={row['last_seen_at'] or '鏈煡'}锛?
            f"sequence={row['first_sequence_index']}-{row['last_sequence_index']}锛涜瘉鎹?{row['evidence_count']}"
        )
        for index, row in enumerate(items)
    ]
    return render_limited_lines(lines, max_chars, "鏉＄粨鏋勫寲瀛︿範璁板綍")


def render_mistake_items_for_prompt(items: list[dict], max_chars: int = 5000) -> tuple[str, bool]:
    if not items:
        return "鏆傛棤鏄庣‘閿欓鏈€欓€夈€?, False
    lines = [
        (
            f"{index + 1}. {row.get('title') or '閿欓鍊欓€?}锛涚姸鎬?{row.get('status') or 'suspected'}锛?
            f"绉戠洰={row.get('subject') or '鏈瘑鍒?}锛涗綅缃?{row.get('location_ref') or '鏈瘑鍒?}锛?
            f"棰樼洰={row.get('question_text') or '鏈瘑鍒?}锛涘鐢熺瓟妗?{row.get('student_answer') or '鏈瘑鍒?}锛?
            f"鍙傝€冪瓟妗?{row.get('expected_answer') or '鏈瘑鍒?}锛涢敊璇被鍨?{row.get('error_type') or '鏈瘑鍒?}锛?
            f"閿欏洜/璇佹嵁={row.get('error_reason') or row.get('evidence') or '鏈瘑鍒?}锛?
            f"璁㈡={row.get('correction') or '鏈瘑鍒?}锛涗笅涓€姝?{row.get('next_action') or '鏈瘑鍒?}锛?
            f"鐭ヨ瘑鐐?{','.join(row.get('knowledge_points') or []) or '鏈瘑鍒?}"
        )
        for index, row in enumerate(items)
    ]
    return render_limited_lines(lines, max_chars, "鏉￠敊棰樻湰鍊欓€?)


def build_report_process_note(session: dict, learning_items: list[dict], mistake_items: list[dict], stats: dict) -> str:
    strategy = session_strategy(session)
    strategy_text = build_strategy_context(strategy)
    item_counts: dict[str, int] = {}
    for item in learning_items:
        item_counts[item.get("item_type") or "unknown"] = item_counts.get(item.get("item_type") or "unknown", 0) + 1
    return (
        "鎶ュ憡鐢熸垚渚濇嵁鎽樿锛歕n"
        f"- 瀛︾敓/绯荤粺鐩爣锛歿strategy_text}\n"
        f"- 缁撴瀯鍖栧涔犳潯鐩細棰樼洰 {item_counts.get('question', 0)}锛屾澘鍧?{item_counts.get('section', 0)}锛?
        f"鐭ヨ瘑鐐?{item_counts.get('knowledge', 0)}锛屼綔绛?{item_counts.get('answer', 0)}\n"
        f"- 閿欓鏈€欓€夛細{len(mistake_items)} 鏉n"
        f"- 瑙嗚璇佹嵁锛氭姄鎷?{stats['image_count']} 寮狅紝鎵规鍒嗘瀽 {stats['analysis_count']} 鏉★紝"
        f"鍘婚噸鍚庡彲鐢ㄥ垎鏋?{stats['unique_done_analysis_count']} 鏉★紝閲嶅鍒嗘瀽 {stats['duplicate_done_analysis_count']} 鏉n"
        "- 鎶ュ憡鐢熸垚鏂瑰紡锛氬厛鏀堕泦鍏抽敭甯у拰宸紓鍒嗘瀽锛屽啀鍚堝苟瀛︿範鏉＄洰銆侀敊棰樺€欓€夈€佹椂闂寸嚎鍜岀敤鎴疯姹傦紝鏈€鍚庣敓鎴愭姤鍛娿€?
    )


def build_distill_source_notes(analyses: list[dict], max_chars: int = FINAL_REPORT_DISTILL_SOURCE_CHAR_LIMIT) -> tuple[list[str], dict]:
    unique_rows, duplicate_count = dedupe_analyses(analyses)
    if not unique_rows:
        return ["鏆傛棤鍙彁鐐肩殑鎵规瑙嗚鍒嗘瀽鍐呭銆?], {"duplicate_done_analysis_count": duplicate_count, "source_chars": 0}
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
                rendered_notes.append(f"... 宸茬渷鐣?{omitted} 鏉￠噸澶嶆垨浠ｈ〃鎬ц緝浣庣殑鎵规鍒嗘瀽锛屼繚鐣欏紑澶?涓/缁撳熬璇佹嵁 ...")
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


def chunk_sequence(items: list, size: int) -> list[list]:
    size = max(1, int(size or 1))
    return [items[index:index + size] for index in range(0, len(items), size)]


def build_distill_prompt(session: dict, images: list[dict], notes_chunk: str, chunk_index: int, chunk_count: int, stats: dict) -> str:
    start, end, total_seconds = report_time_bounds(session, images)
    timeline_lines = build_timeline_lines(session, images)
    timeline, timeline_compressed = render_limited_lines(timeline_lines, 5000, "鏉℃姄鎷嶆椂闂寸嚎") if timeline_lines else ("鏃犳姄鎷嶅浘鐗囥€?, False)
    time_weight_summary = build_time_weight_summary(session, images)
    timeline = f"{time_weight_summary}\n{timeline}" if timeline else time_weight_summary
    compressed_notice = "鏃堕棿绾垮凡鎶芥牱鍘嬬缉锛? if timeline_compressed else ""
    return prompts.render_prompt(
        "distill_final_evidence",
        chunk_index=chunk_index,
        chunk_count=chunk_count,
        compressed_notice=compressed_notice,
        image_count=stats["image_count"],
        analysis_count=stats["analysis_count"],
        unique_done_analysis_count=stats["unique_done_analysis_count"],
        duplicate_done_analysis_count=stats["duplicate_done_analysis_count"],
        start=start.isoformat() if start else "鏈煡",
        end=end.isoformat() if end else session.get("finished_at") or "鏈煡",
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
        timeline, timeline_compressed = render_limited_lines(timeline_lines, FINAL_REPORT_TIMELINE_CHAR_LIMIT, "鏉℃姄鎷嶆椂闂寸嚎")
    else:
        timeline, timeline_compressed = "鏃犳姄鎷嶅浘鐗囥€?, False
    time_weight_summary = build_time_weight_summary(session, images)
    timeline = f"{time_weight_summary}\n{timeline}"
    if distilled_notes is None:
        batch_notes, analyses_compressed = build_limited_batch_notes(analyses, FINAL_REPORT_ANALYSES_CHAR_LIMIT)
        evidence_label = "鎵规瑙嗚鍒嗘瀽"
    else:
        batch_notes = truncate_text(distilled_notes, FINAL_REPORT_DISTILLED_NOTES_CHAR_LIMIT)
        analyses_compressed = len(distilled_notes) > FINAL_REPORT_DISTILLED_NOTES_CHAR_LIMIT
        evidence_label = "鍒嗘壒鎻愮偧鍚庣殑鍏抽敭璇佹嵁"
    stats = final_report_evidence_stats(images, analyses)
    item_notes, items_compressed = render_learning_items_for_prompt(learning_items or [])
    mistake_notes, mistakes_compressed = render_mistake_items_for_prompt(mistake_items or [])
    if item_notes:
        process_note = build_report_process_note(session, learning_items or [], mistake_items or [], stats)
        batch_notes = (
            f"銆愬鐢熺洰鏍囦笌鍔ㄦ€佺瓥鐣ャ€慭n{build_strategy_context(session_strategy(session))}\n\n"
            f"銆愰敊棰樻湰鍊欓€夈€慭n{mistake_notes}\n\n"
            f"銆愮粨鏋勫寲瀛︿範鏉＄洰锛堝凡鎸夐鐩?鏉垮潡/鐭ヨ瘑鐐瑰幓閲嶏級銆慭n{item_notes}\n\n"
            f"銆愭姤鍛婄敓鎴愪緷鎹€慭n{process_note}\n\n"
            f"銆恵evidence_label}銆慭n{batch_notes}"
        )
    compressed_notice = ""
    if timeline_compressed or analyses_compressed or items_compressed or mistakes_compressed or distilled_notes is not None:
        compressed_notice = (
            "娉ㄦ剰锛氬悗绔凡鎸夋ā鍨嬩笂涓嬫枃棰勭畻鍘嬬缉鏃堕棿绾挎垨鎵规鍒嗘瀽锛屽苟鍒犻櫎閲嶅鎻忚堪锛?
            "鏈垪鍑虹殑缁嗚妭涓嶈缂栭€狅紝鍙熀浜庡彲瑙佽瘉鎹杩般€俓n\n"
        )
    prompt = prompts.render_prompt(
        "final_report",
        evidence_label=evidence_label,
        compressed_notice=compressed_notice,
        start=start.isoformat() if start else "鏈煡",
        end=end.isoformat() if end else session.get("finished_at") or "鏈煡",
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
                f"{index}. 鏃堕棿={event.get('created_at')}; "
                f"瑙﹀彂={event.get('trigger_type') or event.get('source')}; "
                f"鎰忓浘={context.get('student_intent') or 'unknown'}\n"
                f"闂細{truncate_text(event.get('question'), 260)}\n"
                f"绛旓細{truncate_text(event.get('answer'), 520)}"
            )
        )
    return truncate_text(
        f"""
璇锋妸杩欎釜娌℃湁瑙傚療鍥剧墖鎴栦互瀹炴椂璇煶闂瓟涓轰富鐨勫涔犲洖鍚堬紝鍘嬬缉鎴愪竴娈靛彲鏌ョ湅鐨勫洖鍚堟€荤粨銆?

瑕佹眰锛?
- 鐢ㄤ腑鏂囷紝缁撴瀯娓呮櫚锛岄€傚悎 App 鍘嗗彶瀵硅瘽閲岀殑鈥滄煡鐪嬫姤鍛娾€濄€?
- 涓嶈缂栭€犻鐩€佸浘鐗囧唴瀹规垨鑰楁椂锛涘彧鑳藉熀浜庨棶绛旀枃鏈拰浼氳瘽绛栫暐銆?
- 閲嶇偣淇濈暀锛氱敤鎴烽棶浜嗕粈涔堛€佹ā鍨嬫€庝箞绛斻€佸綋鍓嶇粨璁?鍗＄偣銆佷笅涓€姝ュ彲浠ョ户缁棶浠€涔堛€?
- 濡傛灉淇℃伅涓嶈冻锛屾槑纭啓鈥滆瘉鎹笉瓒斥€濄€?

浼氳瘽锛?
- id={session.get('id')}
- title={session.get('title')}
- student_goal={session.get('student_goal') or ''}

鍔ㄦ€佺瓥鐣ワ細
{build_strategy_context(session_strategy(session))}

闂瓟璁板綍锛?
{chr(10).join(qa_lines) if qa_lines else "鏆傛棤闂瓟璁板綍銆?}

璇疯緭鍑鸿繖浜涙爮鐩細
鍥炲悎鎽樿锛?
鐢ㄦ埛涓昏闂锛?
宸茬粰鍑虹殑甯姪锛?
鍙兘鐨勫崱鐐癸細
涓嬩竴姝ュ缓璁細
鎶ュ憡鐢熸垚渚濇嵁锛?
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
    emit_log(f"寮€濮嬪ぇ妯″瀷瑙ｆ瀽锛歿len(image_paths)} 寮犲浘鐗?, session_id=session_id)
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
                    "鍥剧墖鎷嶆憚鎻愰啋锛氳鏈?璇曞嵎/灞忓箷鍙兘鏈畬鏁磋繘鍏ユ媿鎽勫尯鍩燂紝"
                    "鎴栭骞层€佹暟瀛椼€侀〉鐮佺湅涓嶆竻銆?
                    "璇疯皟鏁寸浉鏈鸿搴?璺濈锛岀‘淇濇潗鏂欏畬鏁存竻鏅板彲瑙佸悗閲嶆柊鎷嶆憚骞惰В鏋愩€?
                ),
                session_id=session_id,
                level="warning",
            )
    except Exception as exc:
        content = f"澶фā鍨嬭В鏋愬け璐ワ細{llm.format_llm_error(exc)}"
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
        # 閫愰鎵规敼锛堝叏棰樺閿欐潈濞侊級锛氭ā鍨嬪凡鎶娾€滄壒鏀笿SON锛歔...]鈥濆啓杩?analyses.content锛堝凡钀藉簱锛夛紝
        # 绔笂鍙洿鎺ヤ粠鍒嗘瀽鍐呭瑙ｆ瀽璇诲彇锛涜繖閲屽啀瑙ｆ瀽涓€娆′粎鐢ㄤ簬鏍￠獙涓庡彲瑙傛祴鏃ュ織锛屾渶灏忛潰銆佷笉鍔犲垪銆?
        # 涓嶈Е鍙戣縼绉伙紝涓斾笌涓婇潰鐨勯敊棰樻娊鍙栭摼璺簰涓嶅奖鍝嶏紙瑙ｆ瀽澶辫触杩斿洖 []锛夈€?
        try:
            grading = parse_batch_grading(content)
        except Exception:
            grading = []
        if grading:
            emit_log(
                f"閫愰鎵规敼宸茶В鏋愶細{len(grading)} 棰橈紙鍏ㄩ瀵归敊锛屽惈鍋氬/绌虹櫧锛涘凡闅?analyses.content 钀藉簱渚涚涓婅鍙栵級",
                session_id=session_id,
            )
            record_report_event(
                session_id,
                "batch_grading",
                "閫愰鎵规敼缁撴灉",
                f"batch={batch_id or 'single'}锛涢鏁?{len(grading)}",
                analysis_id,
            )
        inferred = infer_needs_from_analysis(content)
        if counts["mistake_item_count"]:
            inferred = merge_tags(inferred, ["mistake_book"])
        if counts["learning_item_count"]:
            inferred = merge_tags(inferred, ["knowledge_points"])
        if inferred:
            update_session_needs(session_id, inferred, "鏍规嵁鏈€鏂扮敾闈㈠垎鏋愬姩鎬佽皟鏁村叧娉ㄧ偣锛? + "銆?.join(tag_label(tag) for tag in inferred))
        if counts["learning_item_count"] or counts["mistake_item_count"]:
            emit_log(
                f"宸插啓鍏ュ涔犺祫浜э細缁撴瀯鍖栨潯鐩?{counts['learning_item_count']} 鏉★紝閿欓鏈€欓€?{counts['mistake_item_count']} 鏉?,
                session_id=session_id,
            )
            record_report_event(
                session_id,
                "batch_assets",
                "鎵规瀛︿範璧勪骇鍏ュ簱",
                f"batch={batch_id or 'single'}锛涚粨鏋勫寲鏉＄洰={counts['learning_item_count']}锛涢敊棰樻湰鍊欓€?{counts['mistake_item_count']}锛涘姩鎬侀渶姹?{','.join(inferred)}",
                analysis_id,
            )
        try:
            indexed_count = await index_knowledge_items(collect_unindexed_knowledge_rows())
            if indexed_count:
                emit_log(f"璇箟鐭ヨ瘑搴撳凡鑷姩绱㈠紩 {indexed_count} 鏉℃柊鏉＄洰", session_id=session_id)
        except Exception:
            pass
    if not already_claimed:
        mark_task_run(task_id, "done" if status == "done" else "failed", error=content if status != "done" else "")
    emit_log(f"瑙ｆ瀽瀹屾垚锛歿status}", session_id=session_id)
    return status, content


async def distill_final_report_evidence(settings, session: dict, images: list[dict], analyses: list[dict], session_id: str) -> str | None:
    if not should_distill_final_evidence(images, analyses):
        return None
    stats = final_report_evidence_stats(images, analyses)
    notes, source_stats = build_distill_source_notes(analyses)
    chunks = chunk_text_blocks(notes, FINAL_REPORT_DISTILL_CHUNK_CHAR_LIMIT)
    emit_log(
        (
            "鏈€缁堟姤鍛婅瘉鎹繃澶э紝鍏堝垎鎵规彁鐐硷細"
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
        emit_log(f"寮€濮嬫彁鐐兼渶缁堟姤鍛婅瘉鎹細绗?{index}/{len(chunks)} 鎵癸紝prompt_chars={len(prompt)}", session_id=session_id)
        try:
            distilled = await run_with_llm_gate(
                f"distill:{session_id[:8]} chunk={index}/{len(chunks)}",
                session_id,
                lambda: llm.analyze_text(settings, prompt, max_tokens=FINAL_REPORT_DISTILL_MAX_TOKENS),
                priority=LLM_PRIORITY_BACKGROUND,
            )
        except Exception as exc:
            emit_log(
                f"鏈€缁堟姤鍛婅瘉鎹彁鐐煎け璐ワ紝鍥為€€鍒扮洿鎺ュ帇缂╂姤鍛婅矾寰勶細{llm.format_llm_error(exc)}",
                session_id=session_id,
                level="warning",
            )
            return None
        distilled_chunks.append(f"銆愭彁鐐兼壒娆?{index}/{len(chunks)}銆慭n{distilled.strip()}")
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
        content = "鍥炲悎鎬荤粨澶辫触锛歴ession not found"
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
        content = f"鍥炲悎鎬荤粨澶辫触锛歿llm.format_llm_error(exc)}"
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
    emit_log(f"QA 鍥炲悎鎬荤粨瀹屾垚锛歿status}", session_id=session_id, level="info" if status == "done" else "error")
    return status, content


async def run_final_report(analysis_id: str, session_id: str, task_id: str | None = None, already_claimed: bool = False) -> tuple[str, str]:
    if not already_claimed:
        mark_task_run(task_id, "running")
    await wait_for_batch_analyses(session_id)
    settings = effective_llm_settings_for_session(session_id)
    with connect() as conn:
        session_row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        if not session_row:
            content = f"鏈€缁堟姤鍛婄敓鎴愬け璐ワ細session not found: {session_id}"
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
        record_report_event(session_id, "final_prompt_context", "鏈€缁堟姤鍛婄敓鎴愪緷鎹?, process_note, analysis_id)
        emit_log(
            (
                "寮€濮嬬敓鎴愬涔犲洖鍚堟渶缁堟€荤粨鎶ュ憡锛?
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
        content = f"鏈€缁堟姤鍛婄敓鎴愬け璐ワ細{llm.format_llm_error(exc)}"
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
    emit_log(f"鏈€缁堟€荤粨鎶ュ憡鐢熸垚瀹屾垚锛歿status}", session_id=session_id)
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
    student_name = clean_auth_text(body.get("student_name") or body.get("studentName"), 120) or "榛樿瀛︾敓"
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
                (parent_profile_id, account_id, user_id, display_name or "瀹堕暱", now, now),
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
    """Phase 3: per-turn memory deltas for the in-chat "杩欐鏇翠簡瑙ｄ綘浜? chip.
    Passive pull only 鈥?no polling, no server-side push."""
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
            "purpose": "缁熶竴 OpenAI/Anthropic/Gemini/鏅鸿氨绛夊钩鍙板埌 OpenAI-compatible API锛汷penAI 鍏煎鐨勫钩鍙板彲鐩存帴鍦ㄤ笂闈㈠～ Base URL+Key锛孉nthropic 绛夐潪鍏煎骞冲彴缁忕綉鍏虫帴鍏ャ€?,
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
        f"鍔犲叆閿欓鏈細{truncate_text(mistake.get('title') or mistake.get('question_text'), 120)}",
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
        f"鏇存柊閿欓鐘舵€侊細{mistake_id} status={mistake.get('status')} review_state={mistake.get('review_state')}",
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
    """鍒楀嚭寰呭鐢熺‘璁ょ殑閿欓鍊欓€夛紙瑙傚療寮傛鎻愬彇锛夈€?""
    init_db()
    principal = principal_from_request(request)
    return mistake_candidate_items(account_id=principal["account_id"], subject=subject, q=q, page_size=page_size)


@app.post("/api/mistakes/{mistake_id}/import")
async def import_mistake_candidate(mistake_id: str, request: Request) -> dict:
    """瀛︾敓纭鍊欓€夆啋瀵煎叆姝ｅ紡閿欓鏈紙status=confirmed锛岃繘澶嶄範闃熷垪锛夈€傚彲甯?review_note/correction 绛夊唴鑱旇姝ｃ€?""
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
        f"瀵煎叆閿欓鍊欓€夆啋閿欓鏈細{mistake_id}",
        session_id=mistake.get("session_id"),
        source="dashboard",
    )
    return {"mistake": mistake}


@app.post("/api/mistakes/{mistake_id}/dismiss")
async def dismiss_mistake_candidate(mistake_id: str, request: Request) -> dict:
    """瀛︾敓蹇界暐鍊欓€夛紙status=ignored锛屼笉杩涢敊棰樻湰锛夈€?""
    init_db()
    principal = principal_from_request(request)
    status = owned_mistake_status(mistake_id, principal["account_id"])
    if status is None:
        raise HTTPException(404, "mistake not found")
    mistake = update_mistake_item(mistake_id, {"status": "ignored"})
    emit_log(
        f"蹇界暐閿欓鍊欓€夛細{mistake_id}",
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
        f"璁板綍閿欓澶嶄範浜嬩欢锛歿mistake_id} result={event.get('result')} review_count={mistake.get('review_count')}",
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
    """鍒楀嚭鏈处鍙锋鍦ㄨ窇/鎺掗槦鐨勫悗鍙扮敓鎴愪换鍔★紙璐﹀彿闅旂锛夈€備粎杩斿洖 background 閫氶亾锛?
    鍙鍖?鎶ュ憡/璁板繂鏁寸悊绛夎€楁椂浠诲姟锛涘疄鏃堕棶绛旀槸鐢ㄦ埛褰撳墠璇锋眰鏈韩锛屼笉鍦ㄦ鍒椼€?""
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
    """鍙栨秷鏈处鍙疯嚜宸辩殑鍚庡彴浠诲姟锛堣处鍙烽殧绂伙細浠栦汉浠诲姟涓€寰?404锛屼笉娉勯湶瀛樺湪鎬э級銆?""
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
    emit_llm_gate_log(f"鐢ㄦ埛鍙栨秷鍚庡彴浠诲姟锛歿label}锛坽task_id[:8]}锛?, session_id=sess, level="warning")
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
    emit_log(f"鏇存柊鎻愮ず璇嶏細{prompt_key}锛堣处鍙?{principal['account_id']}锛?, source="dashboard")
    return {"prompt": record}


@app.post("/api/prompts/{prompt_key}/reset")
def reset_prompt(prompt_key: str, request: Request) -> dict:
    init_db()
    principal = principal_from_request(request)
    try:
        record = prompts.reset_prompt(prompt_key)
    except KeyError:
        raise HTTPException(404, "prompt not found")
    emit_log(f"鎭㈠榛樿鎻愮ず璇嶏細{prompt_key}锛堣处鍙?{principal['account_id']}锛?, source="dashboard")
    return {"prompt": record}


@app.post("/api/prompts/reset")
def reset_all_prompts(request: Request) -> dict:
    init_db()
    principal = principal_from_request(request)
    records = prompts.reset_all_prompts()
    emit_log(f"鎭㈠鍏ㄩ儴榛樿鎻愮ず璇嶏紙璐﹀彿 {principal['account_id']}锛?, source="dashboard")
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
                    (SELECT COUNT(*) FROM mistake_items WHERE mistake_items.session_id = sessions.id) AS mistake_count,
                    (SELECT COUNT(*) FROM extracted_questions WHERE extracted_questions.session_id = sessions.id) AS stored_question_count,
                    (
                        SELECT content FROM report_events
                        WHERE report_events.session_id = sessions.id
                          AND event_type='question_set'
                        ORDER BY id DESC
                        LIMIT 1
                    ) AS question_set_payload
                FROM sessions
                WHERE account_id=?
                ORDER BY created_at DESC
                LIMIT 100
                """,
                (principal["account_id"],),
            )
        ]
    for session in sessions:
        stored_count = int(session.pop("stored_question_count", 0) or 0)
        payload = session.pop("question_set_payload", "") or ""
        question_count = stored_count
        if question_count == 0 and payload:
            try:
                parsed = json.loads(payload)
                question_count = len(parsed) if isinstance(parsed, list) else 0
            except (json.JSONDecodeError, TypeError):
                question_count = 0
        session["question_count"] = question_count
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
        f"褰㈡垚璁板繂锛歿truncate_text(text, 120)}",
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
            "message": "宸插姞鍏ョ┖闂茬敓鎴愰槦鍒楋細鍙鍖栦細鍦ㄨ闊?闂瓟绌洪棽鏃惰嚜鍔ㄧ敓鎴愶紝瀹屾垚鍚庡彲鍦ㄨ鍥炲涓嬬偣銆屾墦寮€鍙鍖栥€嶆煡鐪嬨€?,
        })
        return pending

    # 鍘婚噸锛氬悓涓€鏉″洖绛旂殑鍙鍖栧彧璺戜竴浠姐€傚鎴风浼氳疆璇㈡湰鎺ュ彛锛堟瘡闅斿嚑绉?POST 涓€娆★級锛?
    # 鑻ヤ笉鍘婚噸锛屾瘡娆¤疆璇㈤兘浼氬啀璧蜂竴涓悗鍙颁换鍔★紝瀵艰嚧銆岀敓鎴愬彲瑙嗗寲璁茶В銆嶅湪鍚庡彴鍫嗗彔 5~6 浠?
    # 锛堣繕浼氭尋鍗犳ā鍨嬶紝鎷栨參瀹炴椂闂瓟锛夈€傝繖閲屽湪鏈繘绋嬪唴璁板綍鍦ㄨ窇鐨?(璐﹀彿, 婧?锛涘悓鏃舵妸鏈€杩戣惤搴?
    # 涓?running 鐨勪篃瑙嗕负鍦ㄨ窇锛堣鐩栬繘绋嬮噸鍚?钀藉簱宸插紑濮嬬殑鎯呭喌锛夈€傚懡涓垯鐩存帴杩斿洖鎺掗槦鎬侊紝涓嶅啀璧蜂换鍔°€?
    viz_key = (account_id, source_type, source_id)
    if not force:
        already_running = viz_key in _VIZ_INFLIGHT
        if not already_running and existing and existing.get("status") == "running":
            already_running = _iso_within_seconds(existing.get("updated_at"), VIZ_INFLIGHT_STALE_SECONDS)
        if already_running:
            return {"visualization": _pending_payload()}

    # Otherwise generate asynchronously at BACKGROUND priority so it never competes
    # with realtime voice/QA 鈥?it runs when the model is idle. Return a "running"
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
            emit_log(f"鍙鍖栫敓鎴愬け璐ワ細{exc}", session_id=session_id or None, source="visualization", level="error")
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
            # extract_and_store) so iOS can passively pull the "杩欐鏇翠簡瑙ｄ綘浜? chip.
            # Still inside the background task 鈥?the QA response path is untouched.
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
            sequence_index=int(max_sequence if max_sequence is not None else -1) + 1,
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
                    or "棣栭棶/鏂伴宸蹭笂浼犲綋鍓嶆姄鎷嶏紝鍚庣鍏佽杩涘叆瑙嗚妯″瀷骞剁敱妯″瀷缁х画鍒ゆ柇鍙鍐呭"
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
                "QA 褰撳墠鎶撴媿涓庝笂涓€寮犲嚑涔庝竴鑷达紝杞函鏂囨湰蹇矾寰勶紝璺宠繃瑙嗚妯″瀷",
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
                    "QA 褰撳墠鎶撴媿鏈繘鍏ュぇ妯″瀷涓婁笅鏂囷細"
                    f"reason={uploaded_frame_quality.get('reason') or 'unknown'}锛?
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
        memory_feedback = "锛?.join(
            part
            for part in (
                f"鍦烘櫙={context_payload.get('learning_mode_title') or ''}",
                f"鍥炵瓟鏂瑰紡={context_payload.get('coach_depth_title') or ''}",
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
        answer = f"AI 闂瓟澶辫触锛歿llm.format_llm_error(exc)}"
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
    """鎶婁换鎰忚緭鍏ユ敹鏁涙垚褰掍竴鍖栧潗鏍囧垎閲忥細闈炴暟瀛?NaN 鈫?0.0锛岃秺鐣?鈫?clamp 鍒?[0,1]銆?""
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
    """鎶婃ā鍨嬬粰鐨勯鍧楁 {x,y,w,h} 娓呮礂鎴愬綊涓€鍖朳0,1]銆佸師鐐瑰乏涓婏紙x 鍙?y 涓嬨€佺浉瀵规暣寮犱笂浼犲浘锛夌殑
    骞插噣 bbox锛涚己瀛楁/闈炴暟瀛?瓒婄晫涓€寰?clamp锛屽楂樹笉瓒婅繃鍙?涓嬭竟鐣屻€備豢 _region_bbox 鐨勬竻娲楀彛寰勶紝
    浣嗗缂哄け瀛楁鏇村瀹癸紙琛?0 鑰岄潪鏁存涓㈠純锛夛紝淇濊瘉閫愰妗嗗缁堝彲鐢ㄣ€?""
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
    """瀛楃涓叉劅鐭ュ湴鎵弿鍑烘墍鏈夋渶澶栧眰 {...} 瀛愪覆锛堣姳鎷彿鍖归厤锛屽拷鐣ュ瓧绗︿覆鍐呯殑鎷彿锛夈€?""
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
    """琛ラ綈琚?max_tokens 鎴柇鑰屾湭闂悎鐨勫瓧绗︿覆/涓嫭鍙?澶ф嫭鍙凤紱宸查棴鍚堝垯鍘熸牱杩斿洖銆?""
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
    """瀵硅瑙夋ā鍨嬪伓灏斿悙鍑虹殑杞诲井闈炴硶 JSON 鍋氬閿欎慨澶嶏紙鍙湪涓ユ牸瑙ｆ瀽澶辫触鍚庝娇鐢級锛?
    1) 鏁板€煎悗澶氫竴涓紩鍙凤紙濡?"h": 0.5"锛夆€斺€斿彧鍦ㄣ€屽€间綅缃€嶏紙绱ц窡 : [ , 鍚庣殑绾暟瀛楋級鍘绘帀杩欎釜寮曞彿锛?
       缁濅笉纰?question_text 杩欑浠ユ暟瀛?寮曞彿缁撳熬鐨勫悎娉曞瓧绗︿覆锛?
    2) 鍘绘帀瀵硅薄/鏁扮粍閲岀殑灏鹃€楀彿锛?
    3) 鎴柇鏈棴鍚堟椂琛ラ綈寮曞彿/涓嫭鍙?澶ф嫭鍙枫€?""
    repaired = re.sub(r'([:\[,]\s*-?\d+(?:\.\d+)?)"', r"\1", text)
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    repaired = _balance_segmentation_json(repaired)
    return repaired


_SEGMENTATION_TARGET_KEYS = ("questions", "is_study_material")


def _load_segmentation_dict(candidate: str, target_keys: tuple[str, ...] = _SEGMENTATION_TARGET_KEYS) -> dict | None:
    """鎶婂€欓€夊瓧绗︿覆涓ユ牸瑙ｆ瀽鎴愬惈鐩爣閿箣涓€鐨勫璞★紱涓嶆槸灏辫繑鍥?None銆倀arget_keys 鍙崲鎴愬埆鐨勭鐐圭殑閿?
    锛堝閲嶆帓鐗?is_printed_question/plain_text锛夛紝璁╁悓涓€濂楅瞾妫掕В鏋愬鐢ㄥ埌闈炲垎鍓插搷搴斾笂銆?""
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(data, dict) and any(k in data for k in target_keys):
        return data
    return None


def _regex_segmentation_fallback(text: str) -> dict:
    """鎵€鏈?JSON 淇閮藉け璐ユ椂鐨勬鍒欏厹搴曪細鐩存帴浠庢枃鏈噷鎹?is_study_material / material_type锛?
    骞堕€愪釜 question 瀵硅薄鎶?index / bbox(x,y,w,h) / question_text / has_student_answer銆?""
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
    """浠庢ā鍨嬭緭鍑洪噷鎶藉嚭 JSON 瀵硅薄锛屽蹇嶄唬鐮佸洿鏍?鍓嶅悗鏁ｆ枃锛屽苟瀵硅瑙夋ā鍨嬪伓灏斿悙鍑虹殑
    杞诲井闈炴硶 JSON锛堟暟鍊煎悗澶氬紩鍙枫€佸熬閫楀彿銆佹埅鏂湭闂悎锛夊仛瀹归敊淇鍚庡啀瑙ｆ瀽锛涢兘澶辫触鏃堕€€鍒版鍒欏厹搴曘€?
    瑙ｆ瀽椤哄簭锛氫弗鏍兼暣浣?鈫?瀹归敊淇鏁翠綋 鈫?淇鍚庨€愪釜 {...} 鍧?鈫?鍘熸枃閫愪釜 {...} 鍧?鈫?姝ｅ垯鍏滃簳銆?
    target_keys 鍐冲畾銆屽悎娉曞璞°€嶇殑鍒ゆ嵁锛涢潪鍒嗗壊绔偣锛堝閲嶆帓鐗堬級浼犺嚜宸辩殑閿紝骞惰烦杩囧垎鍓蹭笓鐢ㄧ殑姝ｅ垯鍏滃簳銆?""
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
    # 姝ｅ垯鍏滃簳浠呭鍒嗗壊鍝嶅簲鏈夋剰涔夛紙鎶?is_study_material/questions锛夛紱鍏跺畠绔偣瑙ｆ瀽澶辫触灏辫繑鍥?{}銆?
    if target_keys == _SEGMENTATION_TARGET_KEYS:
        return _regex_segmentation_fallback(text)
    return {}


def _build_segmentation_response(raw: str) -> dict:
    """鎶婅瑙夋ā鍨嬬殑鍘熷鏂囨湰瑙ｆ瀽鎴愮ǔ瀹氱殑棰樼洰鍒嗗壊濂戠害锛涗换浣曠己澶?瓒婄晫閮借娓呮礂鎴愰粯璁ゅ€硷紝
    瑙ｆ瀽澶辫触鏃堕€€鍥?{is_study_material: False, material_type: "none", questions: []}銆?
    鏂板 low_quality锛堝儚棰樼洰浣嗗お绯?璇讳笉娓咃紝寤鸿鏁村浘闂瓟鎴栭噸鎷嶏級涓?bbox_approx锛坆box 涓鸿繎浼煎畾浣嶏紝
    闈炵簿纭紝浠呬緵鎺掑簭/绮楀畾浣嶏級涓や釜甯冨皵瀛楁锛涗袱鑰呭潎鍚戝悗鍏煎銆?""
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
            # 妯＄硦鍏滃簳鍚愮┖妗嗭細question_text 鍘荤┖鐧藉悗涓虹┖鐨勯娌℃湁浠讳綍杈ㄩ浠峰€硷紝杩囨护鎺夈€?
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
    # low_quality锛氬儚棰樼洰浣嗚涓嶆竻/閫€鍖栵紝寤鸿鏁村浘闂瓟鎴栭噸鎷嶃€備袱绉嶈Е鍙戯細
    #   1) 妯″瀷鍒や负棰樼洰鏉愭枡銆佷絾杩囨护鍚庝竴閬撳彲璇婚閮芥病鏈夛紙鍏ㄦ槸绌烘 / 澶硦锛夛紱
    #   2) 鍒囧嚭浜嗛锛屼絾骞冲潎鏂囨湰鏋佺煭锛堟枩鎷?杩滄媿 OCR 閫€鍖栫殑绠€鍗曞厹搴曪級銆?
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
    """鍒ゆ柇鐢婚潰閲屾湁娌℃湁棰樼洰骞堕€愰缁欏綊涓€鍖?bbox锛堜緵 iOS 绔皟鐢級銆傚鐢ㄧ幇鏈?27B 瑙嗚澶фā鍨嬶紝
    涓嶅紩鍏ユ柊妯″瀷銆佷笉鏀?QA/鎵规敼涓绘祦绋嬶紱鍙緭鍑哄垎鍓茬粨鏋滐紝涓嶈В棰樸€?""
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
            f"棰樼洰鍒嗗壊璋冪敤瑙嗚妯″瀷澶辫触锛歿truncate_text(str(exc), 180)}",
            session_id=session_id,
            device_id=source_text,
            source="segment",
            level="warning",
        )
        raise HTTPException(502, "question segmentation failed")
    result = _build_segmentation_response(raw)
    emit_log(
        f"棰樼洰鍒嗗壊瀹屾垚锛歩s_study_material={result['is_study_material']}锛?
        f"material_type={result['material_type']}锛岄鏁?{len(result['questions'])}",
        session_id=session_id,
        device_id=source_text,
        source="segment",
    )
    result["raw"] = truncate_text(raw, 4000)
    return result


_GRADING_VERDICTS = {"瀵?, "閿?, "閮ㄥ垎瀵?, "涓嶇‘瀹?, "鏈綔绛?, "鏈瘑鍒?}
# 杩欎笁妗ｆ槸鈥滃０绉板垽瀹氫簡瀵归敊鈥濈殑缁撹锛屽繀椤绘湁姝ｇ‘绛旀鍋氫緷鎹紝鍚﹀垯闄嶇骇涓衡€滀笉纭畾鈥濄€?
_GRADING_DECISIVE_VERDICTS = {"瀵?, "閿?, "閮ㄥ垎瀵?}

# 闇€瑕佽鍥?鍑犱綍鎺ㄧ悊鐨勭嚎绱細VLM 瀵硅繖绫婚杩囧害鑷俊銆佷細缂栭€犳纭瓟妗堢‖鍒ゅ閿欙紙瀹炴祴鍚屼竴閬撳嚑浣曢
# 涓夊紶鐓х墖缁欎笁涓笉鍚屾爣鍑嗙瓟妗堬級锛屾槸妯″瀷鑳藉姏纭激锛屾湇鍔＄蹇呴』寮哄埗鍏滃簳涓衡€滀笉纭畾鈥濄€?
# 鍛戒腑浠讳竴鈥滆鍥剧嚎绱⑩€濆嵆瑙嗕负涓嶅彲闈狅紱鍑犱綍鍥惧舰璇嶉渶鍙犲姞鈥滈潰绉?鍛ㄩ暱/浣撶Н鈥濇墠绠楀嚑浣曟帹鐞嗛銆?
_GRADING_FIGURE_HINTS = (
    "濡傚浘", "涓嬪浘", "涓婂浘", "鍙冲浘", "宸﹀浘", "鍥句腑", "鍥炬墍绀?, "瑙佸浘", "闃村奖",
)
_GRADING_SHAPE_HINTS = (
    "涓夎褰?, "姊舰", "骞宠鍥涜竟褰?, "姝ｆ柟褰?, "闀挎柟褰?, "鍦?, "鎵囧舰", "澶氳竟褰?, "鐩磋涓夎褰?,
)
_GRADING_MEASURE_HINTS = ("闈㈢Н", "鍛ㄩ暱", "浣撶Н")


def _grading_needs_figure(question_text: str) -> bool:
    """鍒ゆ柇杩欓亾棰樻槸鍚︹€滈渶瑕佽鍥?鍑犱綍鎺ㄧ悊鈥濓紝浠庤€?VLM 鍒ゅ垎涓嶅彲闈犮€侀渶寮哄埗闄嶇骇涓衡€滀笉纭畾鈥濄€?
    鍒ゆ嵁锛堜换涓€鍛戒腑鍗崇湡锛夛細棰樺共鍚鍥剧嚎绱紙濡傚浘/涓嬪浘/鈥?闃村奖 涔嬩竴锛夛紱鎴栧惈鍑犱綍鍥惧舰璇?
    锛堜笁瑙掑舰/姊舰/鈥?鐩磋涓夎褰?涔嬩竴锛変笖鍚害閲忚瘝锛堥潰绉?鍛ㄩ暱/浣撶Н 涔嬩竴锛夈€?
    绾畻鏈?鍙ｇ畻/绾暟瀛楀簲鐢ㄩ涓嶅惈涓婅堪绾跨储 鈫?杩斿洖 False锛屾ā鍨嬬殑瀵归敊鍒ゅ畾鐓у父淇濈暀銆?""
    text = question_text or ""
    if any(hint in text for hint in _GRADING_FIGURE_HINTS):
        return True
    if any(shape in text for shape in _GRADING_SHAPE_HINTS) and any(
        measure in text for measure in _GRADING_MEASURE_HINTS
    ):
        return True
    return False


def _build_grading_response(raw: str, enforce_figure_gate: bool = True) -> dict:
    """鎶婃暣椤垫壒鏀硅瑙夋ā鍨嬬殑鍘熷鏂囨湰瑙ｆ瀽鎴愮ǔ瀹氱殑閫愰鎵规敼濂戠害銆傚鐢ㄩ鐩垎鍓茬殑椴佹 JSON 鎶藉彇
    锛坃extract_segmentation_object锛変笌 bbox 娓呮礂锛坃segment_question_bbox锛夈€傛瘡棰樺彇
    index/bbox/question_text/student_answer/verdict(鐧藉悕鍗曟牎楠岋紝闈炴硶鈫?涓嶇‘瀹?)/correct_answer/
    correction/error_reason/knowledge锛泀uestion_text 涓?student_answer 閮戒负绌虹殑棰樿繃婊ゆ帀銆?
    涓€鑷存€у厹搴曪細verdict鈭坽瀵?閿?閮ㄥ垎瀵箎 浣?correct_answer 涓虹┖锛堢己鍒ゅ垎渚濇嵁锛夆啋闄嶇骇涓?涓嶇‘瀹?銆?
    bbox 浠呬緵绮楁帓搴忥紙bbox_approx=True锛夛紝鍓嶇鐢ㄧ涓?Vision 鐨勭簿纭?bbox 瀹氫綅銆?
    鏁翠綋瑙ｆ瀽澶辫触鏃堕€€鍥?{is_study_material: False, questions: [], low_quality: True}銆?""
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
            # 棰樺共涓庡鐢熶綔绛旈兘涓虹┖鐨勯娌℃湁浠讳綍鎵规敼浠峰€硷紝杩囨护鎺夈€?
            if not question_text.strip() and not student_answer.strip():
                continue
            verdict = str(entry.get("verdict") or "").strip()
            # 闈炴硶鍒ゅ畾涓嶈榛樿鎴愬垽鍒嗘。鈥滄湭璇嗗埆鈥濓紝缁熶竴閫€鍒扳€滀笉纭畾鈥濓紝閬垮厤鏃犱緷鎹湴缁欏嚭瀵归敊銆?
            if verdict not in _GRADING_VERDICTS:
                verdict = "涓嶇‘瀹?
            correct_answer = truncate_text(entry.get("correct_answer") or "", 600)
            # 涓€鑷存€у厹搴曪細澹扮О鍒や簡瀵?閿?閮ㄥ垎瀵癸紝鍗存病缁欏嚭姝ｇ‘绛旀锛岃鏄庡垽瀹氱己涔忎緷鎹紝闄嶇骇涓衡€滀笉纭畾鈥濄€?
            if verdict in _GRADING_DECISIVE_VERDICTS and not correct_answer.strip():
                verdict = "涓嶇‘瀹?
            # 鍙潬鎬ч椄闂細璇诲浘/鍑犱綍鎺ㄧ悊棰?VLM 鍒ゅ垎涓嶅彲闈狅紝寮哄埗闄嶇骇涓衡€滀笉纭畾鈥濆苟娓呯┖姝ｇ‘绛旀锛?
            # 浠呬繚鐣?student_answer/question_text/correction(鍙綋鎬濊矾)/error_reason/knowledge銆?
            # enforce_figure_gate=False 缁欏墠娌垮ぇ妯″瀷(GPT-5.5)璺緞锛氬畠鑷繁浼氱畻鍑犱綍骞惰瘹瀹炲垽涓嶇‘瀹氾紝涓嶅己鍒躲€?
            needs_figure = enforce_figure_gate and _grading_needs_figure(question_text)
            if needs_figure and verdict in _GRADING_DECISIVE_VERDICTS:
                verdict = "涓嶇‘瀹?
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
                    # gradable=False 琛ㄧず杩欐槸璇诲浘/鍑犱綍棰樸€佸垽鍒嗕笉鍙潬锛屽墠绔嵁姝や笉鏄剧ず 鉁?鉁椼€?
                    "gradable": not needs_figure,
                }
            )
    is_material = bool(data.get("is_study_material"))
    if questions:
        is_material = True
    if not is_material:
        questions = []
    # like 棰樼洰鏉愭枡 but no gradable question survived 鈫?寤鸿鏁村浘闂瓟鎴栭噸鎷嶃€?
    low_quality = bool(is_material and not questions)
    return {
        "is_study_material": is_material,
        "questions": questions,
        "low_quality": low_quality,
        "bbox_approx": True,
        # 鏈嶅姟绔彧瀵光€滃彲闈犲瓙闆嗏€濓紙绾畻鏈?鍙ｇ畻锛夌粰鍑哄閿欙紱璇诲浘/鍑犱綍棰樺己鍒垛€滀笉纭畾鈥濄€?
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
    """绮炬壒/绮惧噯鍒嗛锛氭湁 GPT-5.5 缃戝叧灏辫蛋瀹冿紙澶栭儴锛屼笉鍗犳湰鍦?27B GPU gate锛岃嚜甯﹀苟鍙戜笂闄愶級锛?
    鍚﹀垯鍥為€€鏈湴 27B锛堣蛋 llm_gate锛夈€傝繑鍥?(raw_text, used_frontier)銆?""
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
    "浣犳槸涓ヨ皑涓旀湁鑳藉姏鐨勪腑灏忓浣滀笟鎵规敼鍔╂墜锛屽叿澶囪鍥句笌鍑犱綍鎺ㄧ悊鑳藉姏銆傚彧杈撳嚭 JSON銆?
    "鑳芥嵁鍥句腑鍙鐨勫昂瀵?鍥惧舰绠楀嚭姝ｇ‘绛旀鏃讹紝灏卞垽瀵?閿?閮ㄥ垎瀵癸紝骞跺湪 correction 鍐欏嚭绠€瑕佹帹瀵笺€乧orrect_answer 缁欏嚭缁撴灉锛?
    "鍙湁褰撳浘褰㈣鎴柇銆佸叧閿昂瀵哥湅涓嶆竻銆佹垨浣犵‘瀹炴棤娉曠‘淇℃椂锛屾墠鐢?verdict=涓嶇‘瀹氥€乧orrect_answer 鐣欑┖銆乧orrection 鍙粰鎬濊矾銆?
    "缁濅笉缂栭€犵瓟妗堬紝correct_answer 蹇呴』涓庝綘鐨勬帹瀵间竴鑷淬€?
)
_SEGMENT_INSTRUCTIONS = "浣犳槸浣滀笟棰樼洰鍒嗗壊鍣ㄣ€傚彧杈撳嚭 JSON銆傞€愰鎸夐鐩湪鍥句腑鐨勭湡瀹炲儚绱犱綅缃粰褰掍竴鍖?bbox锛屼笉瑕佹寜棰樺簭鍧囧寑骞抽摵銆?


@app.post("/api/sessions/{session_id}/grade-page")
async def grade_session_questions(
    session_id: str,
    request: Request,
    image: UploadFile = File(...),
    source: str = Form("ios"),
) -> dict:
    """瀵规暣椤典綔涓氶€愰鎵规敼锛堜緵 iOS 绔皟鐢級銆傚鐢ㄧ幇鏈?27B 瑙嗚澶фā鍨嬮€愰缁欏嚭瀵?閿?璁㈡锛?
    閴存潈/瀛樺浘/闄愭祦鍙ｅ緞涓?/segment-questions 瀹屽叏涓€鑷淬€俠box 浠呬緵绮楁帓搴忥紝鍓嶇鐢ㄧ涓?Vision 绮剧‘瀹氫綅銆?""
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
            f"鏁撮〉鎵规敼璋冪敤瑙嗚妯″瀷澶辫触锛歿truncate_text(str(exc), 180)}",
            session_id=session_id,
            device_id=source_text,
            source="grade",
            level="warning",
        )
        raise HTTPException(502, "page grading failed")
    # GPT-5.5 浼氳嚜宸辫瘹瀹炲垽鍑犱綍棰?鑷姤涓嶇‘瀹?鈫?鏀惧紑鏈嶅姟绔€屽嚑浣曞己鍒朵笉纭畾銆嶉椄闂紱鏈湴 27B 浠嶅己鍒躲€?
    result = _build_grading_response(raw, enforce_figure_gate=not used_frontier)
    emit_log(
        f"鏁撮〉鎵规敼瀹屾垚锛歩s_study_material={result['is_study_material']}锛?
        f"棰樻暟={len(result['questions'])}锛宭ow_quality={result['low_quality']}",
        session_id=session_id,
        device_id=source_text,
        source="grade",
    )
    result["raw"] = truncate_text(raw, 4000)
    return result


def _relayout_str_list(raw: object, item_limit: int = 600, max_items: int = 30) -> list[str]:
    """鎶婃ā鍨嬬粰鐨?given/formulas 瀛楁瀹夊叏娓呮礂鎴?list[str]锛氶潪鍒楄〃鈫掔┖鍒楄〃锛涢€愰」杞瓧绗︿覆銆佹埅鏂€?
    涓㈡帀绌虹櫧椤癸紱鏈€澶氫繚鐣?max_items 鏉★紝閬垮厤寮傚父杈撳嚭鎾戠垎鍝嶅簲銆?""
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
    """鎶娾€滃嵃鍒烽骞叉暟瀛楀寲閲嶆帓鐗堚€濊瑙夋ā鍨嬬殑鍘熷鏂囨湰瑙ｆ瀽鎴愮ǔ瀹氬绾︺€傚鐢ㄩ鐩垎鍓茬殑椴佹 JSON 鎶藉彇
    锛坃extract_segmentation_object锛夈€傚彧鍙栧嵃鍒烽骞插瓧娈碉細is_printed_question(bool)/title/plain_text/
    given(list[str])/ask/formulas(list[str])/figure_note/uncertain锛岀己瀛楁瀹夊叏闄嶇骇涓虹┖鍊笺€?
    椤跺眰寮哄埗 must_compare_original=True锛屽墠绔嵁姝ゅ己鍒舵樉绀哄師鍥撅紙鎵规敼璇佹嵁/棰樺共椤诲鐓у師鍥炬牳楠岋級銆?
    鏈嚱鏁板彧鎼繍妯″瀷缁欑殑鍗板埛棰樺共鏂囨湰锛屼笉鍦ㄦ湇鍔＄鎷兼帴瀛︾敓浣滅瓟銆佷笉閲嶅缓鍑犱綍鍥俱€?
    鏁翠綋瑙ｆ瀽澶辫触鏃堕€€鍥?{is_printed_question: False, plain_text: "", must_compare_original: True}銆?""
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
        # 閾佸緥锛氶骞查』瀵圭収鍘熷浘鏍搁獙锛涘墠绔己鍒舵樉绀哄師鍥撅紝涓嶅緱浠呭嚟閲嶆帓鏂囨湰浣滅瓟/鎵规敼銆?
        "must_compare_original": True,
    }


@app.post("/api/sessions/{session_id}/relayout-question")
async def relayout_question(
    session_id: str,
    request: Request,
    image: UploadFile = File(...),
    source: str = Form("ios"),
) -> dict:
    """鎶婂浘涓€愬嵃鍒蜂綋棰樺共銆戞暣鐞嗘垚骞插噣鏁板瓧鐗堬紙B 瀹為獙锛屼緵 iOS 绔皟鐢級銆備弗鏍煎彧閲嶆帓鍗板埛棰樺共锛?
    涓嶈浆鍐欏鐢熸墜鍐欎綔绛斻€佷笉閲嶇敾鍑犱綍鍥撅紙鍚浘鍙湪 figure_note 娉ㄦ槑瑙佸師鍥撅級銆佺湅涓嶆竻鍐欌€滄湭璇嗗埆鈥濄€?
    澶嶇敤鐜版湁 27B 瑙嗚澶фā鍨嬶紱閴存潈/瀛樺浘/闄愭祦鍙ｅ緞涓?/segment-questions 瀹屽叏涓€鑷达紱涓嶈В棰樸€佷笉鎵规敼銆?
    鍝嶅簲椤跺眰寮哄埗 must_compare_original=true锛屽墠绔嵁姝ゅ己鍒舵樉绀哄師鍥俱€?""
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
            f"棰樺共閲嶆帓鐗堣皟鐢ㄨ瑙夋ā鍨嬪け璐ワ細{truncate_text(str(exc), 180)}",
            session_id=session_id,
            device_id=source_text,
            source="relayout",
            level="warning",
        )
        raise HTTPException(502, "question relayout failed")
    result = _build_relayout_response(raw)
    emit_log(
        f"鍗板埛棰樺共閲嶆帓鐗堝畬鎴愶細is_printed_question={result['is_printed_question']}锛?
        f"uncertain={result.get('uncertain')}锛屾枃鏈暱搴?{len(result.get('plain_text') or '')}锛?
        f"must_compare_original={result['must_compare_original']}",
        session_id=session_id,
        device_id=source_text,
        source="relayout",
    )
    result["raw"] = truncate_text(raw, 4000)
    return result


# ---------------------------------------------------------------------------
# 瑙傚療鍙?Demo锛?observe锛夛細鏃犵姸鎬佹媿鐓ц瀵熷紡棰樼洰鎻愬彇杩樺師銆傜嫭绔嬪叆鍙ｃ€佷笉姹℃煋鐪熷疄鏁版嵁銆?
# 绾㈢嚎锛氫笉璋冪敤 save_upload銆佷笉鍐欎换浣曠敤鎴疯〃锛坕mages/errors/knowledge/sessions锛夛紝
# 鍘熷浘鍙湪鍓嶇鎸佹湁 blob锛屽悗绔彧鐢ㄤ复鏃舵枃浠朵笖 finally unlink銆傛瘡鍥惧彧瑙﹀彂銆愪竴娆°€慥LM 璋冪敤銆?
# ---------------------------------------------------------------------------
_OBSERVE_QTYPES = {
    "閫夋嫨棰?, "濉┖棰?, "璁＄畻棰?, "鍙ｇ畻棰?, "瑙ｇ瓟棰?, "搴旂敤棰?, "鍒ゆ柇棰?, "浣滄枃棰?, "鍏跺畠",
}
_OBSERVE_GRADE_CLAUSE = (
    "\n棰濆瑕佹眰锛堟壒鏀规ā寮忥級锛氬啀缁欐瘡閬撻涓€涓?verdict 瀛楁锛屽彧鑳戒粠杩欏叚涓噷閫変竴涓細"
    "瀵广€侀敊銆侀儴鍒嗗銆佷笉纭畾銆佹湭浣滅瓟銆佹湭璇嗗埆銆傚垽鍒嗛棬妲涳紙瀵规爣鍙潬鍙ｇ畻鎵规敼锛屽畞鍙笉鍒や篃鍒瀻鍒わ級锛?
    "鍙湁绾彛绠?鍗板埛绠楁湳銆佷綘鑳界嫭绔嬫湁鎶婃彙纭姝ｇ‘绛旀鏃讹紝鎵嶅厑璁稿垽鈥滃/閿?閮ㄥ垎瀵光€濓紱"
    "鍑℃槸闇€瑕佽鍥?鍑犱綍鎺ㄧ悊銆佸簲鐢ㄩ銆佹垨浣犳嬁涓嶅噯鐨勶紝verdict 涓€寰嬪啓鈥滀笉纭畾鈥濓紝涓嶈纭垽瀵归敊锛?
    "瀛︾敓娌′綔绛斿啓鈥滄湭浣滅瓟鈥濓紝棰樺共鎴栦綔绛旂湅涓嶆竻鍐欌€滄湭璇嗗埆鈥濄€倂erdict 鏀捐繘姣忛亾棰樺璞￠噷銆?
)


_OBSERVE_EXTRACT_STABILITY_CLAUSE = (
    "\n\nAdditional extraction stability rules:\n"
    "1. Keep one top-level printed question as one JSON question. Do not split a single printed exercise such as "
    "\"绠椾竴绠梊", \"缁冧竴缁僜", \"鍙ｇ畻\", or a grid/list of formulas into separate questions for every formula.\n"
    "2. If an image crop only shows an isolated short formula like \"5x99\" or \"6x999\" and the top-level question number "
    "or instruction is not visible, treat it as insufficient context and do not output it as a standalone question.\n"
    "3. When several visible formulas clearly belong to one printed exercise, merge them into the same question stem, "
    "preserving the printed instruction and formula list once.\n"
)


def _observe_options(raw: object) -> list[dict]:
    """鎶婃ā鍨嬬粰鐨?options 瀹夊叏娓呮礂鎴?[{label,text}] 鍒楄〃锛氶潪鍒楄〃鈫抂]锛涢€愰」鍙?label/text銆佹埅鏂€?
    涓㈡帀 label 涓?text 閮界┖鐨勯」锛涙渶澶?12 椤癸紝閬垮厤寮傚父杈撳嚭鎾戠垎鍝嶅簲銆?""
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


_OBSERVE_PUNCT_RE = re.compile(r"[\s銆€.,;:!?銆傦紝銆侊紱锛氾紒锛焈\-鈥斺€撀封€"'鈥溾€濃€樷€?)锛堬級\[\]銆愩€慮+")


def _observe_normalize_stem(stem: str) -> str:
    """棰樺共褰掍竴鍖栵細鍘荤┖鐧?鏍囩偣銆佺粺涓€灏忓啓锛岀敤浜庤法鍥惧幓閲嶆寚绾广€俿tem 宸蹭笉鍚鐢熸墜鍐欎綔绛斻€?""
    return _OBSERVE_PUNCT_RE.sub("", (stem or "").lower())


def _observe_simhash(stem: str) -> str:
    """瀵瑰綊涓€鍖栭骞茬殑 3-gram 璁＄畻 64 浣?SimHash锛岃繑鍥?16 浣嶅崄鍏繘鍒躲€傚墠绔敤 Hamming 璺濈鍒よ繎浼煎悓棰樸€?""
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


_OBSERVE_PLACEHOLDER_STEM_SIGNALS = (
    "棰樼洰鏂囧瓧鏈瘑鍒?,
    "棰樺共鏂囧瓧鏈瘑鍒?,
    "棰樼洰鏂囧瓧琚伄鎸?,
    "棰樺共鏂囧瓧琚伄鎸?,
    "鏂囧瓧鏈瘑鍒?,
    "浠呭彲瑙佸浘绀?,
    "浠呰鍥剧ず",
    "浠呰鎻掑浘",
)

_OBSERVE_PLACEHOLDER_CONTEXT_SIGNALS = (
    "缁撳悎鍥剧ず",
    "鍥剧ず涓?,
    "鍚埧灞?,
    "鍚彃鍥?,
    "鍚浘绀?,
)

_OBSERVE_GENERIC_STEM_NORMS = {
    "鍋氫竴鍋氬～涓€濉?,
}


def _observe_question_placeholder(stem: str) -> bool:
    compact = re.sub(r"[\s()锛堬級銆愩€慭[\]{}銆屻€嶃€庛€忋€傦紝銆侊紱;锛?,.!?锛侊紵]+", "", stem or "")
    if not compact:
        return True
    if any(signal in compact for signal in _OBSERVE_PLACEHOLDER_STEM_SIGNALS):
        return True
    if ("鏈瘑鍒? in compact or "琚伄鎸? in compact) and any(
        signal in compact for signal in _OBSERVE_PLACEHOLDER_CONTEXT_SIGNALS
    ):
        return True
    return False


def _observe_question_too_generic(question: dict) -> bool:
    stem_norm = _observe_normalize_stem(question.get("stem") or "")
    return not (question.get("number") or "").strip() and stem_norm in _OBSERVE_GENERIC_STEM_NORMS


def _observe_short_formula_fragment(question: dict) -> bool:
    if str(question.get("number") or "").strip():
        return False
    stem = str(question.get("stem") or "").strip().lower()
    if not stem:
        return False
    compact = re.sub(r"\s+", "", stem)
    compact = compact.replace("\u00d7", "x").replace("\uff0a", "*").replace("\u00f7", "/").replace("\uff1d", "=")
    compact = compact.replace("\u2014", "-").replace("\uff0d", "-")
    if len(compact) > 96:
        return False
    if re.fullmatch(r"\d+(?:[x*/+\-=]\d+)+=?", compact) is None:
        return False
    operator_count = len(re.findall(r"[x*/+\-=]", compact))
    return len(compact) <= 12 or operator_count >= 2


def _observe_question_should_keep(question: dict) -> bool:
    stem = (question.get("stem") or "").strip()
    if not stem or stem in {"鏈瘑鍒?, "鏃犳硶璇嗗埆", "鐪嬩笉娓?}:
        return False
    if _observe_question_placeholder(stem):
        return False
    if _observe_question_too_generic(question):
        return False
    if _observe_short_formula_fragment(question):
        return False
    return True


def _build_observe_response(raw: str, do_grade: bool) -> dict:
    """鎶婅瀵熷彴鍗曞浘鎻愬彇鐨勮瑙夋ā鍨嬪師濮嬫枃鏈В鏋愭垚绋冲畾濂戠害銆傚鐢ㄩ鐩垎鍓茬殑椴佹 JSON 鎶藉彇
    锛坃extract_segmentation_object锛夈€傛瘡棰樻竻娲?index/number/subject/qtype/stem/options/blanks/
    figure_note/has_student_answer/student_answer锛屽苟鐢辨湇鍔＄杩藉姞鍘婚噸鐢ㄧ殑 fingerprint(40hex)+simhash(16hex)銆?
    do_grade=True 鏃舵牎楠?verdict 鐧藉悕鍗曪紙闈炴硶鈫掆€滀笉纭畾鈥濓級锛屽苟瀵瑰懡涓?_grading_needs_figure锛堣鍥?鍑犱綍锛夌殑棰?
    寮哄埗闄嶇骇涓衡€滀笉纭畾鈥濓紝閬垮厤妯″瀷缂栭€犳爣鍑嗙瓟妗堬紙娌跨敤 grade-page 鐨勫彲闈犳€ч椄闂級銆?
    stem 涓?student_answer 閮戒负绌虹殑棰樿繃婊ゆ帀锛涙暣浣撹В鏋愬け璐ラ€€鍥?{is_study_material: False, questions: [], low_quality: True}銆?""
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
            input_index = None
            for input_key in ("input_index", "inputIndex", "source_index", "sourceIndex"):
                try:
                    candidate_input_index = int(entry.get(input_key))
                except (TypeError, ValueError):
                    continue
                if candidate_input_index > 0:
                    input_index = candidate_input_index
                    break
            stem = truncate_text(entry.get("stem") or "", 800)
            student_answer = truncate_text(entry.get("student_answer") or "", 600)
            # 棰樺共涓庡鐢熶綔绛旈兘涓虹┖鐨勯娌℃湁鎻愬彇浠峰€硷紝杩囨护鎺夈€?
            if not stem.strip() and not student_answer.strip():
                continue
            qtype = str(entry.get("qtype") or "").strip()
            if qtype not in _OBSERVE_QTYPES:
                qtype = "鍏跺畠"
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
                # 鍘婚噸鎸囩汗锛歠ingerprint 绮剧‘鍚岄锛宻imhash 杩戜技鍚岄锛堝墠绔寜 Hamming<=3 鍚堝苟锛夈€?
                "fingerprint": hashlib.sha1(stem_norm.encode("utf-8")).hexdigest() if stem_norm else "",
                "simhash": _observe_simhash(stem),
            }
            if input_index is not None:
                question["input_index"] = input_index
            if not _observe_question_should_keep(question):
                continue
            if do_grade:
                verdict = str(entry.get("verdict") or "").strip()
                if verdict not in _GRADING_VERDICTS:
                    verdict = "涓嶇‘瀹?
                # 鍙潬鎬ч椄闂細璇诲浘/鍑犱綍棰?VLM 鍒ゅ垎涓嶅彲闈狅紝寮哄埗闄嶇骇涓衡€滀笉纭畾鈥濄€?
                if _grading_needs_figure(stem) and verdict in _GRADING_DECISIVE_VERDICTS:
                    verdict = "涓嶇‘瀹?
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


# 瑙傚療鍙?Demo 鐨勮交閲忛槻婊ョ敤锛氬悓涓€ batch_token 缁村害鐨勬椿璺冭姹備覆琛岋紙闃插崟鍏ュ彛鐙崰鍏变韩 27B锛夛紝
# 閰嶅悎鍓嶇鍗曟壒 鈮?0 寮犵‖涓婇檺 + BACKGROUND 浼樺厛绾э紙缁欒鍫?realtime 璁╄矾锛夈€?
_OBSERVE_BATCH_LOCKS: dict[str, asyncio.Lock] = {}
_OBSERVE_MAX_IMAGE_BYTES = 16 * 1024 * 1024


@app.post("/api/observe-demo/extract")
async def observe_demo_extract(
    request: Request,
    image: UploadFile = File(...),
    do_grade: bool = Form(False),
    batch_token: str = Form(""),
) -> dict:
    """瑙傚療鍙?Demo 鍗曞浘棰樼洰缁撴瀯鍖栨彁鍙栵紙鏃犵姸鎬併€佷笉姹℃煋鐪熷疄鏁版嵁锛夈€傚鐢ㄧ幇鏈?27B 瑙嗚澶фā鍨嬶紝
    姣忓浘鍙Е鍙戜竴娆?analyze_images锛汢ACKGROUND 浼樺厛绾х粰璇惧爞瀹炴椂璇锋眰璁╄矾銆傚師鍥惧啓涓存椂鏂囦欢銆佺敤瀹屽嵆鍒狅紝
    涓嶈皟鐢?save_upload銆佷笉鍐欎换浣曠敤鎴疯〃銆傞粯璁よ姹傜櫥褰曪紱鑻ョ‘闇€鍏紑婕旂ず锛屽彲鏄惧紡璁剧疆 PXJ_OBSERVE_DEMO_PUBLIC=true銆?""
    init_db()
    settings_for_demo = get_settings()
    principal_from_request(request, required=not settings_for_demo.observe_demo_public)
    if not image.filename:
        raise HTTPException(422, "image is required")
    payload = await image.read()
    if not payload:
        raise HTTPException(422, "image is empty")
    if len(payload) > _OBSERVE_MAX_IMAGE_BYTES:
        raise HTTPException(413, "image too large (Demo 鍗曞浘涓婇檺 16MB)")
    token = clean_user_text(batch_token, 64) or "default"
    lock = _OBSERVE_BATCH_LOCKS.setdefault(token, asyncio.Lock())
    tmp_dir = get_settings().data_dir / "observe_demo"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}.jpg"
    tmp_path.write_bytes(payload)
    settings = effective_llm_settings_for_session(None)
    prompt = prompts.render_prompt("observe_extract") + _OBSERVE_EXTRACT_STABILITY_CLAUSE
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
            f"瑙傚療鍙版彁鍙栬皟鐢ㄨ瑙夋ā鍨嬪け璐ワ細{truncate_text(str(exc), 180)}",
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
# 棰樼洰鎻愬彇锛堣瀺鍏?iPhone/iPad 瑙傚療锛夛細鏈夌姸鎬佷細璇濈鐐广€傛瘡寮犵収鐗囨彁鍙栭鐩啋鏈嶅姟绔幓閲嶇疮绉垚
# 銆岄鐩泦銆嶅瓨搴撯啋浼氳瘽鍒涘缓鍗冲叆鍘嗗彶(status='saved')鈫掕繕鍘熼〉/绌虹櫧鍗风敱鏈嶅姟绔覆鏌?HTML 渚?WKWebView 鏄剧ず涓庢墦鍗般€?
# 澶嶇敤 observe_extract 鎻愮ず璇嶄笌 _build_observe_response锛涘嚑浣?璇诲浘棰樺凡琚己鍒垛€滀笉纭畾鈥濓紝鏈姛鑳戒笉鍒ゅ垎銆?
# ---------------------------------------------------------------------------
def _simhash_hamming(a: str, b: str) -> int:
    if not a or not b:
        return 64
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return 64


def _question_norm_text(question: dict) -> str:
    text = " ".join(
        str(question.get(key) or "")
        for key in ("stem", "number", "qtype", "subject")
    )
    options = question.get("options")
    if isinstance(options, list):
        text += " " + " ".join(str(item.get("text") or "") for item in options if isinstance(item, dict))
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _question_stem_norm_text(question: dict) -> str:
    stem = str(question.get("stem") or "")
    return "".join(ch.lower() for ch in stem if ch.isalnum())


def _text_gram_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 16 and shorter in longer:
        return 1.0
    if len(shorter) < 12:
        return 0.0
    width = 2 if len(shorter) < 40 else 3
    a_grams = {a[index:index + width] for index in range(0, max(1, len(a) - width + 1))}
    b_grams = {b[index:index + width] for index in range(0, max(1, len(b) - width + 1))}
    if not a_grams or not b_grams:
        return 0.0
    return len(a_grams & b_grams) / max(1, min(len(a_grams), len(b_grams)))


def _common_prefix_len(a: str, b: str) -> int:
    count = 0
    for left, right in zip(a, b):
        if left != right:
            break
        count += 1
    return count


def _questions_look_same(a: dict, b: dict) -> bool:
    fp_a = a.get("fingerprint") or ""
    fp_b = b.get("fingerprint") or ""
    if fp_a and fp_a == fp_b:
        return True
    sh_a = a.get("simhash") or ""
    sh_b = b.get("simhash") or ""
    if sh_a and sh_b and _simhash_hamming(sh_a, sh_b) <= 3:
        return True
    norm_a = _question_norm_text(a)
    norm_b = _question_norm_text(b)
    similarity = _text_gram_similarity(norm_a, norm_b)
    stem_a = _question_stem_norm_text(a)
    stem_b = _question_stem_norm_text(b)
    stem_similarity = _text_gram_similarity(stem_a, stem_b)
    shorter_stem_len = min(len(stem_a), len(stem_b))
    if shorter_stem_len >= 8 and (stem_a in stem_b or stem_b in stem_a):
        return True
    same_number = bool(a.get("number")) and str(a.get("number")) == str(b.get("number"))
    same_type = bool(a.get("qtype")) and str(a.get("qtype")) == str(b.get("qtype"))
    one_unnumbered = bool(a.get("number")) != bool(b.get("number"))
    common_prefix = _common_prefix_len(stem_a, stem_b)
    if same_number and shorter_stem_len >= 4 and (stem_a in stem_b or stem_b in stem_a):
        return True
    if shorter_stem_len >= 10 and stem_similarity >= 0.98:
        return True
    if same_number and shorter_stem_len >= 12 and common_prefix >= 8 and stem_similarity >= 0.32:
        return True
    if same_number and shorter_stem_len >= 10 and common_prefix >= min(14, shorter_stem_len):
        return True
    if same_number and shorter_stem_len >= 6 and stem_similarity >= 0.42:
        return True
    if one_unnumbered and shorter_stem_len >= 14 and stem_similarity >= 0.56 and (same_type or similarity >= 0.54):
        return True
    if shorter_stem_len >= 18 and stem_similarity >= 0.94:
        return True
    if same_number and shorter_stem_len >= 18 and stem_similarity >= 0.62:
        return True
    if same_type and shorter_stem_len >= 18 and stem_similarity >= 0.82:
        return True
    if similarity >= 0.86 and (same_number or same_type):
        return True
    if sh_a and sh_b and _simhash_hamming(sh_a, sh_b) <= 8 and similarity >= 0.72:
        return True
    return False


def _question_quality_score(question: dict) -> int:
    stem = question.get("stem") or ""
    stem_norm = _question_stem_norm_text(question)
    score = min(len(stem_norm), 220)
    if (question.get("number") or "").strip():
        score += 28
    if (question.get("qtype") or "").strip() and question.get("qtype") != "鍏跺畠":
        score += 4
    if question.get("figure_note"):
        score += 3
    if isinstance(question.get("options"), list):
        score += min(12, len(question["options"]) * 3)
    if "鏈瘑鍒? in stem or "琚伄鎸? in stem:
        score -= 35
    if _observe_question_too_generic(question):
        score -= 120
    if _observe_question_placeholder(stem):
        score -= 1000
    return score


def _question_number_value(question: dict) -> int | None:
    match = re.search(r"\d+", str(question.get("number") or ""))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _order_question_set_if_safe(questions: list[dict]) -> list[dict]:
    numbered = [_question_number_value(question) for question in questions]
    present_numbers = [number for number in numbered if number is not None]
    if len(present_numbers) < 2:
        return questions
    # 澶氶〉缁冧範缁忓父浼氫粠 1 閲嶆柊缂栧彿锛涜繖绉嶆儏鍐典繚鐣欐媿鎽勯『搴忥紝閬垮厤璺ㄩ〉涔卞簭銆?    if len(set(present_numbers)) != len(present_numbers):
        return questions
    if len(present_numbers) < max(2, len(questions) - 1):
        return questions
    return [
        item
        for _, item in sorted(
            enumerate(questions),
            key=lambda pair: (
                _question_number_value(pair[1]) if _question_number_value(pair[1]) is not None else 10**6 + pair[0],
                pair[0],
            ),
        )
    ]


def _dedupe_question_set(questions: list[dict]) -> list[dict]:
    result: list[dict] = []
    for question in questions:
        if not _observe_question_should_keep(question):
            continue
        replaced = False
        for index, existing in enumerate(result):
            if _questions_look_same(question, existing):
                if _question_quality_score(question) > _question_quality_score(existing) + 6:
                    result[index] = question
                replaced = True
                break
        if replaced:
            continue
        result.append(question)
    return _order_question_set_if_safe(result)


def _consolidate_question_set(prior: list[dict], new: list[dict]) -> list[dict]:
    """鎶婃湰寮犲浘鏂版彁鍙栫殑棰樺苟鍏ュ凡鏈夐鐩泦骞跺幓閲嶏紙鏈嶅姟绔槸鍞竴鍘婚噸鑰咃紝涓ょ涓€鑷达級銆?    fingerprint 绮剧‘鍛戒腑璺宠繃锛涘惁鍒?simhash/棰樺共鐩镐技搴﹀懡涓涓鸿繎浼煎悓棰樸€?    鍚岄鍚屾椂鍑虹幇鐭増/鍗犱綅鐗?瀹屾暣鐗堟椂锛屼繚鐣欐洿瀹屾暣鐨勭増鏈€?""
    return _dedupe_question_set([*prior, *new])


def _latest_question_set(session_id: str) -> list[dict]:
    """璇诲彇鏈細璇濆凡瀛樼殑棰樼洰闆嗭紙report_events 涓崟鏉?event_type='question_set'锛夈€?""
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


def _question_storage_key(question: dict, index: int) -> str:
    base = str(question.get("fingerprint") or "").strip()
    if not base:
        base = str(question.get("simhash") or "").strip()
    if not base:
        base = _question_norm_text(question)
    if not base:
        base = f"question-{index}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _question_options_json(question: dict) -> str:
    options = question.get("options")
    return json_dumps(options if isinstance(options, list) else [])


def _question_blanks_value(question: dict) -> int:
    try:
        return max(0, int(question.get("blanks") or 0))
    except (TypeError, ValueError):
        return 0


def _replace_session_extracted_questions(conn, session_id: str, qset: list[dict], now: str) -> None:
    conn.execute("DELETE FROM extracted_questions WHERE session_id=?", (session_id,))
    for index, question in enumerate(qset, start=1):
        question_key = _question_storage_key(question, index)
        conn.execute(
            """
            INSERT INTO extracted_questions(
                id, session_id, question_index, question_key, number, subject, qtype, stem,
                options, blanks, figure_note, has_student_answer, student_answer,
                fingerprint, simhash, src_filename, source_image_id, source_crop_id,
                crop_filename, crop_rect, crop_hash, payload, first_seen_at, last_seen_at,
                created_at, updated_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                uuid.uuid4().hex,
                session_id,
                index,
                question_key,
                truncate_text(question.get("number") or "", 80),
                truncate_text(question.get("subject") or "", 80),
                truncate_text(question.get("qtype") or "", 80),
                question.get("stem") or "",
                _question_options_json(question),
                _question_blanks_value(question),
                question.get("figure_note") or "",
                1 if question.get("has_student_answer") else 0,
                question.get("student_answer") or "",
                truncate_text(question.get("fingerprint") or "", 80),
                truncate_text(question.get("simhash") or "", 80),
                truncate_text(question.get("src_filename") or "", 240),
                truncate_text(question.get("source_image_id") or "", 80),
                truncate_text(question.get("source_crop_id") or "", 80),
                truncate_text(question.get("crop_filename") or "", 240),
                json_object_string(question.get("crop_rect"), {}),
                truncate_text(question.get("crop_hash") or "", 120),
                json_dumps(question),
                now,
                now,
                now,
                now,
            ),
        )


def _stored_question_rows(session_id: str) -> list[dict]:
    with connect() as conn:
        rows = [
            row_to_dict(row)
            for row in conn.execute(
                """
                SELECT *
                FROM extracted_questions
                WHERE session_id=?
                ORDER BY question_index ASC, created_at ASC
                """,
                (session_id,),
            ).fetchall()
        ]
    return rows


def _stored_questions_for_response(session_id: str) -> list[dict]:
    rows = _stored_question_rows(session_id)
    if rows:
        crop_lookup: dict[str, str] | None = None

        def lookup_crop_filename(row: dict) -> str:
            nonlocal crop_lookup
            existing = str(row.get("crop_filename") or "").strip()
            if existing:
                return existing
            crop_id = str(row.get("source_crop_id") or "").strip()
            crop_hash = str(row.get("crop_hash") or "").strip()
            if not crop_id and not crop_hash:
                return ""
            if crop_lookup is None:
                with connect() as conn:
                    crop_rows = conn.execute(
                        """
                        SELECT id, crop_hash, crop_filename
                        FROM session_question_crops
                        WHERE session_id=? AND crop_filename!=''
                        """,
                        (session_id,),
                    ).fetchall()
                crop_lookup = {}
                for crop_row in crop_rows:
                    filename = str(crop_row["crop_filename"] or "").strip()
                    if not filename:
                        continue
                    if crop_row["id"]:
                        crop_lookup[f"id:{crop_row['id']}"] = filename
                    if crop_row["crop_hash"]:
                        crop_lookup[f"hash:{crop_row['crop_hash']}"] = filename
            return crop_lookup.get(f"id:{crop_id}") or crop_lookup.get(f"hash:{crop_hash}") or ""

        result: list[dict] = []
        for row in rows:
            try:
                payload = json.loads(row.get("payload") or "{}")
            except (json.JSONDecodeError, TypeError):
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            payload.setdefault("number", row.get("number") or "")
            payload.setdefault("subject", row.get("subject") or "")
            payload.setdefault("qtype", row.get("qtype") or "")
            payload.setdefault("stem", row.get("stem") or "")
            payload.setdefault("figure_note", row.get("figure_note") or "")
            payload.setdefault("fingerprint", row.get("fingerprint") or "")
            payload.setdefault("simhash", row.get("simhash") or "")
            payload.setdefault("src_filename", row.get("src_filename") or "")
            payload.setdefault("source_image_id", row.get("source_image_id") or "")
            payload.setdefault("source_crop_id", row.get("source_crop_id") or "")
            payload["crop_filename"] = str(payload.get("crop_filename") or "").strip() or lookup_crop_filename(row)
            payload.setdefault("crop_rect", row.get("crop_rect") or "{}")
            payload.setdefault("crop_hash", row.get("crop_hash") or "")
            result.append(payload)
        return result
    qset = _latest_question_set(session_id)
    if qset:
        now = utc_now()
        with connect() as conn:
            _replace_session_extracted_questions(conn, session_id, qset, now)
        return qset
    return []


def _save_question_set(session_id: str, qset: list[dict]) -> None:
    """鎶婇鐩泦 upsert 鎴愬崟鏉?report_events 琛岋紙閬垮厤閫愬紶杩藉姞鎾戠垎 overview 鐨?40 鏉＄獥鍙ｃ€佷篃閬垮厤
    record_report_event 鐨?6000 瀛楁埅鏂瘉鍧?JSON锛夈€俢ontent 涓哄畬鏁?JSON锛圱EXT 鍒楁棤闀垮害闄愬埗锛夈€?""
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
                (session_id, None, "question_set", "棰樼洰闆?, payload, now),
            )
        _replace_session_extracted_questions(conn, session_id, qset, now)
        conn.commit()


def _apply_question_source_meta(question: dict, source_meta: dict, filename: str) -> None:
    source_type = str(source_meta.get("type") or "").lower()
    question["src_filename"] = source_meta.get("src_filename") or filename
    if source_meta.get("input_index"):
        question["source_input_index"] = int(source_meta.get("input_index") or 0)
    if source_meta.get("source_image_id"):
        question["source_image_id"] = source_meta.get("source_image_id")
    if source_meta.get("source_crop_id"):
        question["source_crop_id"] = source_meta.get("source_crop_id")
        question["source_type"] = "question_crop"
    crop_filename = str(source_meta.get("crop_filename") or "").strip()
    if source_type == "crop" and not crop_filename:
        crop_filename = str(source_meta.get("filename") or filename).strip()
    if crop_filename:
        question["crop_filename"] = crop_filename
    if source_meta.get("crop_rect"):
        question["crop_rect"] = source_meta.get("crop_rect")
    if source_meta.get("crop_hash"):
        question["crop_hash"] = source_meta.get("crop_hash")
    if source_meta.get("crop_source"):
        question["crop_source"] = source_meta.get("crop_source")
    if source_meta.get("source_image_size"):
        question["source_image_size"] = source_meta.get("source_image_size")
    if source_meta.get("crop_image_size"):
        question["crop_image_size"] = source_meta.get("crop_image_size")
    if source_meta.get("crop_safety"):
        question["crop_safety"] = source_meta.get("crop_safety")
    if source_meta.get("client_question_key"):
        question["client_question_key"] = source_meta.get("client_question_key")
    if source_meta.get("client_ocr_text"):
        question["client_ocr_text"] = source_meta.get("client_ocr_text")
    if source_meta.get("fallback_reason"):
        question["source_fallback_reason"] = source_meta.get("fallback_reason")


def _question_source_match_text(source: dict) -> str:
    text = " ".join(
        str(source.get(key) or "")
        for key in ("client_ocr_text", "client_question_key", "question_index")
    )
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _best_source_for_extracted_question(question: dict, sources: list[dict], fallback_index: int) -> dict:
    if not sources:
        return {}
    if len(sources) == 1:
        return sources[0]
    try:
        input_index = int(question.get("input_index") or 0)
    except (TypeError, ValueError):
        input_index = 0
    if 1 <= input_index <= len(sources):
        return sources[input_index - 1]
    qnorm = _question_norm_text(question)
    qnum = str(question.get("number") or "").strip()
    best_source = sources[min(fallback_index, len(sources) - 1)]
    best_score = -1.0
    for index, source in enumerate(sources):
        source_norm = _question_source_match_text(source)
        score = _text_gram_similarity(qnorm, source_norm)
        source_question_index = str(source.get("question_index") or "").strip()
        if qnum and source_question_index and qnum == source_question_index:
            score += 0.25
        if qnum and source_norm.startswith(qnum):
            score += 0.12
        score -= abs(index - fallback_index) * 0.01
        if score > best_score:
            best_score = score
            best_source = source
    return best_source


async def _extract_questions_from_stored_source_group(
    session_id: str,
    sources: list[dict],
    *,
    label_prefix: str = "extract_crop_batch",
) -> dict:
    sources = [{**source, "input_index": index + 1} for index, source in enumerate(sources)]
    settings = effective_llm_settings_for_session(session_id)
    prompt = prompts.render_prompt("observe_extract") + _OBSERVE_EXTRACT_STABILITY_CLAUSE
    if len(sources) > 1:
        prompt += (
            "\n\n琛ュ厖锛氭湰娆¤緭鍏ユ槸鎸夐瑁佸壀鍥炬壒閲忚瘑鍒€?
            "姣忓紶鍥剧墖閫氬父瀵瑰簲涓€閬撻鎴栦竴閬撻鐨勫眬閮紝璇锋寜鍥剧墖椤哄簭鎻愬彇鍗板埛棰樼洰锛?
            "鍚屼竴閬撻閲嶅鍑虹幇鏃跺彧淇濈暀鏇村畬鏁寸殑涓€鏉°€?
        )
    if len(sources) > 1:
        prompt += (
            "\n"
            "Keep input_index attached to the question even when deduping repeated inputs."
        )
    image_paths = [image_path_for_request(str(source.get("filename") or "")) for source in sources]
    raw = await run_with_llm_gate(
        f"{label_prefix}:{session_id[:8]} images={len(image_paths)}",
        session_id,
        lambda: llm.analyze_images(settings, prompt, image_paths),
        priority=LLM_PRIORITY_QUESTION_EXTRACTION,
    )
    per = _build_observe_response(raw, do_grade=False)
    for index, q in enumerate(per["questions"]):
        matched_source = _best_source_for_extracted_question(q, sources, index)
        _apply_question_source_meta(q, matched_source, matched_source.get("filename") or sources[0].get("filename") or "")
    before = _latest_question_set(session_id)
    consolidated = _consolidate_question_set(before, per["questions"])
    _save_question_set(session_id, consolidated)
    added_count = max(0, len(consolidated) - len(before))
    crop_count = sum(1 for source in sources if source.get("type") == "crop")
    emit_log(
        (
            f"棰樼洰鎵归噺鎻愬彇瀹屾垚锛氭潵婧?{len(sources)} 涓紙瑁佸壀棰樺浘 {crop_count}锛夛紝"
            f"鏈壒 {len(per['questions'])} 棰橈紝鏂板 {added_count} 棰橈紝"
            f"鍘婚噸鍚庨闆?{len(consolidated)} 棰橈紝low_quality={per['low_quality']}"
        ),
        session_id=session_id,
        source="extract",
    )
    return {
        **per,
        "question_set": consolidated,
        "question_set_count": len(consolidated),
    }


async def _extract_questions_from_stored_image(
    session_id: str,
    filename: str,
    *,
    source_text: str,
    label_prefix: str = "extract",
    source_meta: dict | None = None,
) -> dict:
    """瀵瑰凡缁忎繚瀛樺埌 images 鐩綍鐨勪竴寮犲浘鎻愰锛屽苟鎶婄粨鏋滃悎骞惰繘鏈細璇?question_set銆?
    杩欎釜鍐呴儴鍑芥暟涓嶆敼 sessions.status/title锛涙樉寮忔媿棰樻彁鍙栧拰鏅鸿兘瑙傚療鍋滄鍚庣殑鏁磋疆鎻愰鍏辩敤瀹冿紝
    閬垮厤鏅€氳瀵熶細璇濊璇敼鎴愨€滈鐩彁鍙栤€濅細璇濄€?    """
    settings = effective_llm_settings_for_session(session_id)
    prompt = prompts.render_prompt("observe_extract") + _OBSERVE_EXTRACT_STABILITY_CLAUSE
    image_paths = [image_path_for_request(filename)]
    raw = await run_with_llm_gate(
        f"{label_prefix}:{session_id[:8]}",
        session_id,
        lambda: llm.analyze_images(settings, prompt, image_paths),
        priority=LLM_PRIORITY_QUESTION_EXTRACTION,
    )
    per = _build_observe_response(raw, do_grade=False)
    source_meta = source_meta or {}
    for q in per["questions"]:
        _apply_question_source_meta(q, source_meta, filename)
    before = _latest_question_set(session_id)
    consolidated = _consolidate_question_set(before, per["questions"])
    _save_question_set(session_id, consolidated)
    added_count = max(0, len(consolidated) - len(before))
    emit_log(
        (
            f"棰樼洰鎻愬彇瀹屾垚锛氬浘鐗?{filename} 鏈浘 {len(per['questions'])} 棰橈紝"
            f"鏂板 {added_count} 棰橈紝鍘婚噸鍚庨闆?{len(consolidated)} 棰橈紝low_quality={per['low_quality']}"
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


def _question_extraction_signal_count(summary: str, label: str) -> int:
    match = re.search(rf"{re.escape(label)}\s*(\d+)", summary or "")
    if not match:
        return 0
    try:
        return max(0, int(match.group(1)))
    except ValueError:
        return 0


def _question_extraction_frame_score(row: dict) -> float:
    summary = row.get("signal_summary") or ""
    text_count = _question_extraction_signal_count(summary, "鏂囧瓧")
    rect_count = _question_extraction_signal_count(summary, "鐭╁舰")
    score = text_count * 10 + rect_count * 3
    if text_count == 0:
        score -= 80
    if "瀛︾敓鍦ㄥ満" in summary or "鎵? in summary:
        score -= 8
    visual_distance = row.get("visual_distance")
    try:
        if visual_distance is not None and float(visual_distance) < 1.0:
            score -= 4
    except (TypeError, ValueError):
        pass
    return score


def _question_extraction_filenames_for_session(session_id: str, limit: int = 80) -> list[str]:
    """閫夋嫨鏈疆瑙傚療閲屽€煎緱杩涘叆鏁磋疆鎻愰鐨勫浘鐗囷細鍙烦杩?invalid/duplicate锛屼笉鎸変笂浼犳壒娆′涪椤点€?""
    limit = max(1, min(int(limit or 80), 160))
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT images.filename, images.batch_id, images.sequence_index, images.created_at,
                   obs.visual_distance, COALESCE(obs.signal_summary, '') AS signal_summary
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
    selected: list[str] = []
    seen: set[str] = set()
    for raw_row in rows:
        row = row_to_dict(raw_row)
        filename = row.get("filename")
        if filename and filename not in seen:
            selected.append(filename)
            seen.add(filename)
    return selected


def _question_extraction_image_sources_for_session(session_id: str, limit: int = 80) -> list[dict]:
    limit = max(1, min(int(limit or 80), 160))
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT images.id AS image_id, images.filename, images.batch_id, images.sequence_index, images.created_at,
                   obs.visual_distance, COALESCE(obs.signal_summary, '') AS signal_summary
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
    sources: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        item = row_to_dict(row)
        filename = item.get("filename") or ""
        if not filename or filename in seen:
            continue
        seen.add(filename)
        sources.append(
            {
                "type": "image",
                "filename": filename,
                "src_filename": filename,
                "source_image_id": item.get("image_id") or "",
                "sequence_index": item.get("sequence_index") or 0,
                "signal_summary": item.get("signal_summary") or "",
                "visual_distance": item.get("visual_distance"),
            }
        )
    return sources


def question_crop_safety_from_normalized_rect(value: object) -> dict:
    payload = json_object_value(value)
    safety = payload.get("crop_safety")
    return safety if isinstance(safety, dict) else {}


def ensure_question_crop_row_safety(session_id: str, item: dict) -> dict:
    source = str(item.get("source") or "").strip()
    existing_safety = question_crop_safety_from_normalized_rect(item.get("normalized_rect"))
    if source in {"server_expanded", "server_rect_crop", "server_rect_expanded"} or (source == "client_crop" and existing_safety):
        item["crop_safety"] = existing_safety
        return item
    filename = str(item.get("crop_filename") or "").strip()
    if not filename:
        return item
    try:
        client_crop_path = image_path_for_request(filename)
    except HTTPException:
        return item
    image_dir = get_settings().data_dir / "images"
    crop_result = maybe_expand_question_crop_file(
        session_id=session_id,
        batch_id=str(item.get("batch_id") or "existing"),
        crop_id=str(item.get("id") or uuid.uuid4().hex),
        image_dir=image_dir,
        client_crop_filename=filename,
        client_crop_path=client_crop_path,
        source_image_filename=str(item.get("src_filename") or ""),
        crop_rect=item.get("crop_rect") or {},
        source_image_size=item.get("source_image_size") or {},
        crop_image_size=item.get("crop_image_size") or {},
        normalized_rect=item.get("normalized_rect") or {},
        client_source=source or "client_crop",
        crop_hash=str(item.get("crop_hash") or ""),
    )
    if crop_result.get("expanded"):
        create_thumbnail_safe(crop_result["path"], crop_result["filename"], session_id)
    should_update = (
        bool(crop_result.get("expanded"))
        or source != crop_result.get("source")
        or not existing_safety
        or (not item.get("crop_hash") and crop_result.get("crop_hash"))
    )
    if should_update:
        with connect() as conn:
            conn.execute(
                """
                UPDATE session_question_crops
                SET crop_filename=?, crop_rect=?, crop_hash=?, source_image_size=?,
                    crop_image_size=?, normalized_rect=?, source=?, updated_at=?
                WHERE id=? AND session_id=?
                """,
                (
                    crop_result.get("filename") or filename,
                    json_object_string(crop_result.get("crop_rect"), {}),
                    truncate_text(crop_result.get("crop_hash") or item.get("crop_hash") or "", 120),
                    json_object_string(crop_result.get("source_image_size"), {}),
                    json_object_string(crop_result.get("crop_image_size"), {}),
                    json_object_string(crop_result.get("normalized_rect"), {}),
                    crop_result.get("source") or "client_crop",
                    utc_now(),
                    item.get("id") or "",
                    session_id,
                ),
            )
            conn.commit()
    if crop_result.get("expanded"):
        safety = crop_result.get("crop_safety") or {}
        emit_log(
            (
                f"question crop expanded on use: crop_id={item.get('id') or ''}, "
                f"reasons={','.join(safety.get('reason') or [])}, "
                f"client_size={safety.get('client_crop_image_size') or {}}, "
                f"expanded_size={safety.get('expanded_crop_image_size') or {}}, "
                f"duration_ms={safety.get('duration_ms') or 0}"
            ),
            session_id=session_id,
            source="question_crop",
        )
    item.update(
        {
            "crop_filename": crop_result.get("filename") or filename,
            "crop_rect": json_object_string(crop_result.get("crop_rect"), {}),
            "crop_hash": truncate_text(crop_result.get("crop_hash") or item.get("crop_hash") or "", 120),
            "source_image_size": json_object_string(crop_result.get("source_image_size"), {}),
            "crop_image_size": json_object_string(crop_result.get("crop_image_size"), {}),
            "normalized_rect": json_object_string(crop_result.get("normalized_rect"), {}),
            "source": crop_result.get("source") or "client_crop",
            "crop_safety": crop_result.get("crop_safety") or {},
        }
    )
    return item


def _question_crop_sources_for_session(session_id: str, limit: int = 160) -> list[dict]:
    limit = max(1, min(int(limit or 160), 240))
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT c.*, images.filename AS src_filename
            FROM session_question_crops c
            LEFT JOIN images ON images.id = c.image_id
            LEFT JOIN session_observations obs ON obs.image_id = c.image_id
            WHERE c.session_id=? AND c.status='ready' AND c.crop_filename!=''
              AND COALESCE(obs.novelty_status, 'unknown') NOT IN ('invalid', 'duplicate')
            ORDER BY c.sequence_index ASC, c.manifest_index ASC, c.created_at ASC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    sources: list[dict] = []
    seen_keys: set[str] = set()
    for row in rows:
        item = ensure_question_crop_row_safety(session_id, row_to_dict(row))
        dedupe_key = item.get("question_key") or item.get("fingerprint") or item.get("crop_hash") or item.get("crop_filename")
        if dedupe_key and dedupe_key in seen_keys:
            continue
        if dedupe_key:
            seen_keys.add(dedupe_key)
        crop_safety = item.get("crop_safety") if isinstance(item.get("crop_safety"), dict) else question_crop_safety_from_normalized_rect(item.get("normalized_rect"))
        sources.append(
            {
                "type": "crop",
                "filename": item.get("crop_filename") or "",
                "crop_filename": item.get("crop_filename") or "",
                "src_filename": item.get("src_filename") or "",
                "source_image_id": item.get("image_id") or "",
                "source_crop_id": item.get("id") or "",
                "sequence_index": item.get("sequence_index") or 0,
                "question_index": item.get("question_index") or 0,
                "crop_rect": item.get("crop_rect") or "{}",
                "crop_hash": item.get("crop_hash") or "",
                "crop_source": item.get("source") or "client_crop",
                "source_image_size": item.get("source_image_size") or "{}",
                "crop_image_size": item.get("crop_image_size") or "{}",
                "crop_safety": crop_safety,
                "client_question_key": item.get("question_key") or "",
                "client_ocr_text": item.get("preview_text") or "",
                "confidence": item.get("confidence"),
            }
        )
    return sources


def _question_full_frame_fallback_reason(stats: dict[str, float]) -> str:
    if not stats:
        return "no_crops"
    crop_count = int(stats.get("count") or 0)
    total_area = float(stats.get("area") or 0)
    max_area = float(stats.get("max_area") or 0)
    weak_count = int(stats.get("weak_count") or 0)
    low_conf_count = int(stats.get("low_conf_count") or 0)
    error_count = int(stats.get("error_count") or 0)
    limited_count = int(stats.get("limited_count") or 0)
    if crop_count <= 0:
        return "no_crops"
    if error_count >= crop_count:
        return "all_crop_generation_errors"
    if limited_count > 0:
        return "limited_candidate_frame"
    if crop_count <= 1 and total_area < 0.42:
        return "single_low_coverage_crop"
    if crop_count <= 2 and total_area < 0.34:
        return "sparse_low_coverage_crops"
    if total_area < 0.18:
        return "tiny_crop_coverage"
    if max_area < 0.16 and total_area < 0.50:
        return "no_large_question_crop"
    if weak_count >= crop_count and crop_count <= 3:
        return "all_weak_crop_keys"
    if low_conf_count >= crop_count and crop_count <= 3:
        return "low_confidence_crops"
    return ""


def _question_extraction_sources_for_session(session_id: str, limit: int = 80) -> list[dict]:
    """Prefer client-side question crops; fall back to full frames that produced no crops."""
    crop_sources = _question_crop_sources_for_session(session_id, limit=max(limit * 4, 40))
    image_sources = _question_extraction_image_sources_for_session(session_id, limit=limit)
    cropped_image_ids = {source.get("source_image_id") for source in crop_sources if source.get("source_image_id")}
    crop_stats_by_image: dict[str, dict[str, float]] = {}
    for source in crop_sources:
        image_id = str(source.get("source_image_id") or "")
        if not image_id:
            continue
        rect = json_object_value(source.get("crop_rect") or {})
        area = max(0.0, float(rect.get("width") or 0) * float(rect.get("height") or 0))
        stats = crop_stats_by_image.setdefault(
            image_id,
            {
                "count": 0,
                "area": 0.0,
                "max_area": 0.0,
                "weak_count": 0,
                "low_conf_count": 0,
                "error_count": 0,
                "limited_count": 0,
                "max_frame_candidate_count": 0,
                "frame_candidate_limit": 0,
            },
        )
        stats["count"] += 1
        stats["area"] += area
        stats["max_area"] = max(float(stats.get("max_area") or 0), area)
        if question_crop_key_is_weak(str(source.get("client_question_key") or "")):
            stats["weak_count"] += 1
        try:
            confidence = float(source.get("confidence") or 0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence and confidence < 0.42:
            stats["low_conf_count"] += 1
        safety = source.get("crop_safety") if isinstance(source.get("crop_safety"), dict) else {}
        if safety.get("error"):
            stats["error_count"] += 1
        client_telemetry = safety.get("client_telemetry") if isinstance(safety.get("client_telemetry"), dict) else {}
        try:
            limited_count = int(client_telemetry.get("frame_limited_candidate_count") or 0)
        except (TypeError, ValueError):
            limited_count = 0
        try:
            frame_candidate_count = int(client_telemetry.get("frame_candidate_count") or 0)
        except (TypeError, ValueError):
            frame_candidate_count = 0
        try:
            frame_candidate_limit = int(client_telemetry.get("frame_candidate_limit") or 0)
        except (TypeError, ValueError):
            frame_candidate_limit = 0
        stats["limited_count"] = max(float(stats.get("limited_count") or 0), limited_count)
        stats["max_frame_candidate_count"] = max(float(stats.get("max_frame_candidate_count") or 0), frame_candidate_count)
        stats["frame_candidate_limit"] = max(float(stats.get("frame_candidate_limit") or 0), frame_candidate_limit)

    fallback_images: list[dict] = []
    for source in image_sources:
        image_id = str(source.get("source_image_id") or "")
        if not image_id:
            reason = "missing_source_image_id"
        elif image_id not in cropped_image_ids:
            reason = "no_crops"
        else:
            reason = _question_full_frame_fallback_reason(crop_stats_by_image.get(image_id, {}))
        if not reason:
            continue
        fallback = {**source, "fallback_reason": reason}
        if image_id in crop_stats_by_image:
            fallback["crop_fallback_stats"] = dict(crop_stats_by_image[image_id])
        fallback_images.append(fallback)
    if fallback_images:
        reason_counts: dict[str, int] = defaultdict(int)
        for source in fallback_images:
            reason_counts[str(source.get("fallback_reason") or "unknown")] += 1
        emit_log(
            f"question extraction full-frame fallback: {dict(sorted(reason_counts.items()))}",
            session_id=session_id,
            source="extract",
        )
    return [*crop_sources, *fallback_images]


def _normalize_question_source(source: dict) -> dict | None:
    if not isinstance(source, dict):
        return None
    filename = str(source.get("filename") or "").strip()
    if not filename:
        return None
    source_type = str(source.get("type") or "image").strip().lower()
    if source_type not in {"crop", "image"}:
        source_type = "image"
    crop_filename = str(source.get("crop_filename") or "").strip()
    if source_type == "crop" and not crop_filename:
        crop_filename = filename
    return {
        "type": source_type,
        "filename": filename,
        "crop_filename": crop_filename,
        "src_filename": str(source.get("src_filename") or filename).strip(),
        "source_image_id": str(source.get("source_image_id") or "").strip(),
        "source_crop_id": str(source.get("source_crop_id") or "").strip(),
        "sequence_index": source.get("sequence_index") or 0,
        "question_index": source.get("question_index") or 0,
        "crop_rect": source.get("crop_rect") or "{}",
        "crop_hash": str(source.get("crop_hash") or "").strip(),
        "crop_source": str(source.get("crop_source") or source.get("source") or "").strip(),
        "source_image_size": source.get("source_image_size") or "{}",
        "crop_image_size": source.get("crop_image_size") or "{}",
        "crop_safety": source.get("crop_safety") if isinstance(source.get("crop_safety"), dict) else {},
        "client_question_key": str(source.get("client_question_key") or "").strip(),
        "client_ocr_text": str(source.get("client_ocr_text") or "").strip(),
        "fallback_reason": str(source.get("fallback_reason") or "").strip(),
        "crop_fallback_stats": source.get("crop_fallback_stats") if isinstance(source.get("crop_fallback_stats"), dict) else {},
    }


def _merge_question_extraction_sources(current_sources: list[dict], payload_sources: list[dict], legacy_filenames: list[str]) -> list[dict]:
    result: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for raw in [*current_sources, *payload_sources]:
        source = _normalize_question_source(raw)
        if not source:
            continue
        key = (source["type"], source.get("source_crop_id") or source["filename"])
        if key in seen:
            continue
        seen.add(key)
        result.append(source)
    if result:
        return result
    for filename in legacy_filenames:
        source = _normalize_question_source({"type": "image", "filename": filename, "src_filename": filename})
        if not source:
            continue
        key = (source["type"], source["filename"])
        if key in seen:
            continue
        seen.add(key)
        result.append(source)
    return result


async def _run_grouped_session_question_extraction(
    *,
    session_id: str,
    task_id: str | None,
    extraction_sources: list[dict],
    initial_count: int,
    crop_count: int,
    image_count: int,
) -> dict:
    processed = 0
    failed = 0
    low_quality = 0
    groups: list[list[dict]] = []
    source_index = 0
    while source_index < len(extraction_sources):
        source = extraction_sources[source_index]
        if source.get("type") == "crop":
            group: list[dict] = []
            while (
                source_index < len(extraction_sources)
                and extraction_sources[source_index].get("type") == "crop"
                and len(group) < QUESTION_CROP_BATCH_SIZE
            ):
                group.append(extraction_sources[source_index])
                source_index += 1
            groups.append(group)
            continue
        groups.append([source])
        source_index += 1

    emit_log(
        f"棰樼洰鎻愬彇鎵ц璁″垝锛歿len(groups)} 涓ā鍨嬫壒娆★紝鏉ユ簮 {len(extraction_sources)} 涓紙瑁佸壀棰樺浘 {crop_count}銆佹暣鍥惧厹搴?{image_count}锛?,
        session_id=session_id,
        source="extract",
    )

    for group_index, raw_group in enumerate(groups, start=1):
        valid_group: list[dict] = []
        for source in raw_group:
            filename = source.get("filename") or ""
            try:
                path = image_path_for_request(filename)
            except HTTPException:
                failed += 1
                emit_log(
                    f"棰樼洰鎻愬彇璺宠繃缂哄け鍥剧墖锛歿filename}",
                    session_id=session_id,
                    source="extract",
                    level="warning",
                )
                continue
            if not path.exists():
                failed += 1
                emit_log(
                    f"棰樼洰鎻愬彇璺宠繃缂哄け鍥剧墖锛歿filename}",
                    session_id=session_id,
                    source="extract",
                    level="warning",
                )
                continue
            valid_group.append(source)
        if not valid_group:
            continue
        try:
            if len(valid_group) > 1 and all(source.get("type") == "crop" for source in valid_group):
                result = await _extract_questions_from_stored_source_group(
                    session_id,
                    valid_group,
                    label_prefix="extract_crop_batch",
                )
            else:
                source = valid_group[0]
                filename = source.get("filename") or ""
                result = await _extract_questions_from_stored_image(
                    session_id,
                    filename,
                    source_text="observation_crop" if source.get("type") == "crop" else "observation",
                    label_prefix="extract_crop" if source.get("type") == "crop" else "extract_all",
                    source_meta=source,
                )
            processed += len(valid_group)
            if result.get("low_quality"):
                low_quality += len(valid_group)
        except LLMTaskCancelled:
            raise
        except Exception as exc:
            failed += len(valid_group)
            filenames_text = ", ".join(str(source.get("filename") or "") for source in valid_group[:3])
            if len(valid_group) > 3:
                filenames_text += " ..."
            emit_log(
                f"棰樼洰鎻愬彇澶辫触锛氭壒娆?{group_index}/{len(groups)}锛屾潵婧?{len(valid_group)} 涓紝{filenames_text}锛歿truncate_text(str(exc), 180)}",
                session_id=session_id,
                source="extract",
                level="warning",
            )

    final_count = len(_latest_question_set(session_id))
    with connect() as conn:
        conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (utc_now(), session_id))
    emit_log(
        (
            f"鏈疆棰樼洰鎻愬彇瀹屾垚锛氬鐞?{processed}/{len(extraction_sources)} 涓潵婧愶紝澶辫触 {failed} 涓紝"
            f"浣庤川閲?{low_quality} 寮狅紝鏂板 {max(0, final_count - initial_count)} 閬擄紝棰橀泦鍏?{final_count} 閬?
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
        "crop_source_count": crop_count,
        "image_source_count": image_count,
        "task_id": task_id or "",
    }


async def run_session_question_extraction(
    session_id: str,
    filenames: list[str],
    task_id: str | None = None,
    sources: list[dict] | None = None,
) -> dict:
    """鍚庡彴鏁磋疆鎻愰浠诲姟锛氫覆琛屽鐞嗕細璇濆浘鐗囷紝閫愬紶鍐欏叆鍚屼竴涓?question_set銆?""
    current_sources = _question_extraction_sources_for_session(session_id)
    extraction_sources = _merge_question_extraction_sources(current_sources, sources or [], filenames)
    initial_count = len(_latest_question_set(session_id))
    processed = 0
    failed = 0
    low_quality = 0
    crop_count = sum(1 for source in extraction_sources if source.get("type") == "crop")
    image_count = len(extraction_sources) - crop_count
    emit_log(
        f"鏈疆棰樼洰鎻愬彇寮€濮嬶細寰呭鐞?{len(extraction_sources)} 涓潵婧愶紙瑁佸壀棰樺浘 {crop_count}銆佹暣鍥惧厹搴?{image_count}锛夛紝宸叉湁棰橀泦 {initial_count} 閬?,
        session_id=session_id,
        source="extract",
    )
    return await _run_grouped_session_question_extraction(
        session_id=session_id,
        task_id=task_id,
        extraction_sources=extraction_sources,
        initial_count=initial_count,
        crop_count=crop_count,
        image_count=image_count,
    )
    for index, source in enumerate(extraction_sources, start=1):
        filename = source.get("filename") or ""
        try:
            path = image_path_for_request(filename)
        except HTTPException:
            failed += 1
            emit_log(
                f"棰樼洰鎻愬彇璺宠繃缂哄け鍥剧墖锛歿filename}",
                session_id=session_id,
                source="extract",
                level="warning",
            )
            continue
        if not path.exists():
            failed += 1
            emit_log(
                f"棰樼洰鎻愬彇璺宠繃缂哄け鍥剧墖锛歿filename}",
                session_id=session_id,
                source="extract",
                level="warning",
            )
            continue
        try:
            result = await _extract_questions_from_stored_image(
                session_id,
                filename,
                source_text="observation_crop" if source.get("type") == "crop" else "observation",
                label_prefix="extract_crop" if source.get("type") == "crop" else "extract_all",
                source_meta=source,
            )
            processed += 1
            if result.get("low_quality"):
                low_quality += 1
        except LLMTaskCancelled:
            raise
        except Exception as exc:
            failed += 1
            emit_log(
                f"棰樼洰鎻愬彇澶辫触锛歿index}/{len(filenames)} {filename}锛泏truncate_text(str(exc), 180)}",
                session_id=session_id,
                source="extract",
                level="warning",
            )
    final_count = len(_latest_question_set(session_id))
    with connect() as conn:
        conn.execute("UPDATE sessions SET updated_at=? WHERE id=?", (utc_now(), session_id))
    emit_log(
        (
            f"鏈疆棰樼洰鎻愬彇瀹屾垚锛氬鐞?{processed}/{len(extraction_sources)} 涓潵婧愶紝澶辫触 {failed} 涓紝"
            f"浣庤川閲?{low_quality} 寮狅紝鏂板 {max(0, final_count - initial_count)} 閬擄紝棰橀泦鍏?{final_count} 閬?
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
        "crop_source_count": crop_count,
        "image_source_count": image_count,
        "task_id": task_id or "",
    }


@app.post("/api/sessions/{session_id}/extract-questions")
async def extract_session_questions(
    session_id: str,
    request: Request,
    image: UploadFile = File(...),
    source: str = Form("ios"),
) -> dict:
    """棰樼洰鎻愬彇锛堜緵 iOS 瑙傚療妯″紡璋冪敤锛夛細瀵逛竴寮犳媿娓呯殑璇曞嵎鍥炬彁鍙栫粨鏋勫寲棰樼洰锛屾湇鍔＄鍘婚噸绱Н鎴愰鐩泦骞跺瓨搴擄紝
    浼氳瘽缃?status='saved' 鐩存帴杩涘巻鍙层€傞壌鏉?瀛樺浘/闄愭祦鍙ｅ緞涓?segment-questions 涓€鑷达紱BACKGROUND 浼樺厛绾х粰璇惧爞瀹炴椂璁╄矾锛?
    涓嶅垽鍒嗭紙do_grade=False锛屽嚑浣曢宸插己鍒朵笉纭畾锛夈€佷笉瑙﹀彂浠讳綍鎶ュ憡浠诲姟锛堢粷涓嶈皟鐢?finish锛夈€?""
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
            f"棰樼洰鎻愬彇璋冪敤瑙嗚妯″瀷澶辫触锛歿truncate_text(str(exc), 180)}",
            session_id=session_id,
            device_id=source_text,
            source="extract",
            level="warning",
        )
        raise HTTPException(502, "question extraction failed")
    subject = next((q.get("subject") for q in result["question_set"] if q.get("subject")), "")
    title = (f"{subject} 路 棰樼洰鎻愬彇" if subject else "棰樼洰鎻愬彇")
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
    """鎶婃湰杞凡淇濆瓨鐨勫叧閿浘鐗囨帓鍏ュ悗鍙版暣杞彁棰樹换鍔★紝渚涒€滃仠姝㈣瀵熷悗鎻愬彇鎵€鏈夐鐩€濇寜閽皟鐢ㄣ€?""
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
    sources = _question_extraction_sources_for_session(session_id, limit=limit)
    crop_source_count = sum(1 for source in sources if source.get("type") == "crop")
    image_source_count = len(sources) - crop_source_count
    if existing:
        with connect() as conn:
            conn.execute(
                """
                UPDATE task_runs
                SET priority=CASE WHEN priority > ? THEN ? ELSE priority END, updated_at=?
                WHERE id=?
                """,
                (TASK_PRIORITY_QUESTION_EXTRACTION, TASK_PRIORITY_QUESTION_EXTRACTION, utc_now(), existing["id"]),
            )
        background_tasks.add_task(execute_next_task_run_now)
        return {
            "session_id": session_id,
            "task_id": existing["id"],
            "status": existing["status"],
            "image_count": len(filenames),
            "source_count": len(sources),
            "raw_image_count": len(filenames),
            "image_source_count": image_source_count,
            "crop_source_count": crop_source_count,
            "question_set_count": len(_latest_question_set(session_id)),
        }
    task_id = record_task_run(
        "question_extraction_session",
        session_id=session_id,
        payload={"session_id": session_id, "filenames": filenames, "sources": sources, "sources_version": 2},
        priority=TASK_PRIORITY_QUESTION_EXTRACTION,
    )
    emit_log(
        f"宸叉帓闃熸彁鍙栨湰杞墍鏈夐鐩細{len(sources)} 涓潵婧愶紙瑁佸壀棰樺浘 {crop_source_count}銆佹暣鍥惧厹搴?{image_source_count}锛涘師濮嬪叧閿浘 {len(filenames)}锛?,
        session_id=session_id,
        source="extract",
    )
    emit_log(
        f"棰樼洰鎻愬彇鏉ユ簮鏄庣粏锛氬疄闄呮潵婧?{len(sources)} 涓紝瑁佸壀棰樺浘 {crop_source_count} 涓紝鏁村浘鍏滃簳 {image_source_count} 涓紝鍘熷鍏抽敭鍥?{len(filenames)} 寮?,
        session_id=session_id,
        source="extract",
    )
    background_tasks.add_task(execute_next_task_run_now)
    return {
        "session_id": session_id,
        "task_id": task_id,
        "status": "queued",
        "image_count": len(filenames),
        "source_count": len(sources),
        "raw_image_count": len(filenames),
        "image_source_count": image_source_count,
        "crop_source_count": crop_source_count,
        "question_set_count": len(_latest_question_set(session_id)),
    }


def _restore_esc(s: object) -> str:
    return html.escape(str(s or ""))


def _restore_stem_html(stem: str) -> str:
    return _restore_esc(stem).replace("鏈瘑鍒?, '<span style="color:#c0392b">鏈瘑鍒?/span>')


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
    if qtype == "閫夋嫨棰? and (q.get("options") or []):
        return ""
    try:
        blanks = int(q.get("blanks") or 0)
    except (TypeError, ValueError):
        blanks = 0
    n = blanks if blanks > 0 else (1 if qtype == "鍙ｇ畻棰? else (12 if qtype == "浣滄枃棰? else 4))
    line = '<div style="border-bottom:1px solid #999;height:1.7em;margin:7px 0"></div>'
    return f'<div>{line * n}</div>'


def _render_restore_html(qset: list[dict], view: str, base_url: str) -> str:
    """浠庡凡瀛橀鐩泦娓叉煋鏈嶅姟绔?HTML锛堣繕鍘熼〉 / 绌虹櫧鍗凤級銆傚惈鍥鹃鍙緭鍑恒€庤鍘熷浘銆?鍘熷浘缂╃暐锛堣蛋鍏嶉壌鏉?
    thumbnail 缁濆URL锛夛紝涓嶉噸缁樺嚑浣曘€佷笉娓叉煋瀛︾敓鎵嬪啓銆倂iew='blank' 甯?@media print A4 绌虹櫧浣滅瓟鍖恒€?""
    is_blank = view == "blank"
    subject = next((q.get("subject") for q in qset if q.get("subject")), "")
    cards = []
    for i, q in enumerate(qset):
        num = _restore_esc(q.get("number")) or (f"{i + 1}." if is_blank else f"绗瑊i + 1}棰?)
        figure = ""
        fn = q.get("figure_note") or ""
        src = q.get("src_filename") or ""
        crop_src = q.get("crop_filename") or ""
        if fn or (q.get("qtype") in ("瑙ｇ瓟棰?, "搴旂敤棰?) and "鍥? in (q.get("stem") or "")):
            note = _restore_esc(fn) or "姝ら鍚浘锛岃鍘熷浘"
            img = f'<img src="{base_url}/api/images/{_restore_esc(src)}/thumbnail" style="max-width:60%;border:1px solid #ccc;margin-top:6px;display:block">' if src else ""
            figure = f'<div style="color:#b9770e;font-size:13px;margin-top:6px">馃柤 {note}</div>{img}'
        pills = f'<span style="font-size:12px;color:#666">[{_restore_esc(q.get("qtype"))}]</span>'
        if not is_blank and q.get("subject"):
            pills = f'<span style="font-size:12px;color:#666">{_restore_esc(q.get("subject"))} 路 </span>' + pills
        blanks_note = ""
        if not is_blank and q.get("blanks"):
            blanks_note = f'<div style="font-size:12px;color:#888;margin-top:4px">濉┖ {int(q["blanks"])} 澶?/div>'
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
            thumb_src = crop_src or src
            if thumb_src:
                caption = "棰樼洰瑁佸壀" if crop_src else "鍘熷浘鏍稿"
                thumb = f'<div style="flex:0 0 180px"><img src="{base_url}/api/images/{_restore_esc(thumb_src)}/thumbnail" style="width:100%;border:1px solid #ddd;border-radius:6px"><div style="font-size:11px;color:#999;text-align:center">{caption}</div></div>'
            cards.append(f'<div style="display:flex;gap:14px;border:1px solid #e0e0e0;border-radius:10px;padding:14px;margin-bottom:12px"><div style="flex:1">{body}</div>{thumb}</div>')
    body_html = "".join(cards) or '<div style="color:#888;text-align:center;padding:40px">杩樻病鏈夋彁鍙栧埌棰樼洰銆傛妸璇曞嵎鎷嶆竻妤氥€侀摵婊″彇鏅鍐嶈瘯銆?/div>'
    if is_blank:
        head = f'<div style="display:flex;justify-content:space-between;border-bottom:2px solid #111;padding-bottom:8px;margin-bottom:16px;font-size:13px"><span>{_restore_esc(subject)} 绌虹櫧缁冧範鍗?/span><span>濮撳悕______ 鏃ユ湡______ 寰楀垎____</span></div>'
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
        f'</head><body><div style="color:#888;font-size:12px;margin-bottom:10px">棰樼洰杩樺師锛堝嵃鍒烽骞茶涔夐噸鎺?+ 鍘熷浘鏍稿锛夈€傚惈鍥鹃鍙爣鈥滆鍘熷浘鈥濓紝涓嶉噸缁樸€佷笉娓叉煋鎵嬪啓浣滅瓟銆?/div>{body_html}</body></html>'
    )


@app.get("/api/sessions/{session_id}/restore-page", response_class=HTMLResponse)
def session_restore_page(session_id: str, request: Request, view: str = "restore") -> str:
    """浠庡凡瀛橀鐩泦娓叉煋杩樺師椤?绌虹櫧鍗?HTML锛堜緵 iOS WKWebView 鏄剧ず涓庢墦鍗帮級銆傞壌鏉冨彛寰勫悓鍏跺畠浼氳瘽绔偣銆?""
    init_db()
    principal = principal_from_request(request)
    with connect() as conn:
        require_account_session(conn, session_id, principal)
    qset = _stored_questions_for_response(session_id)
    base_url = (get_settings().public_base_url or "").rstrip("/")
    return _render_restore_html(qset, "blank" if view == "blank" else "restore", base_url)


@app.get("/api/sessions/{session_id}/question-set")
def session_question_set_status(session_id: str, request: Request) -> dict:
    """杞婚噺杩斿洖鏁磋疆棰樼洰鎻愬彇鐘舵€侊紝渚涘鎴风鍦ㄥ悗鍙颁换鍔″畬鎴愬悗鍒锋柊杩樺師椤点€?""
    init_db()
    principal = principal_from_request(request)
    with connect() as conn:
        require_account_session(conn, session_id, principal)
        task = conn.execute(
            """
            SELECT id, status, last_error, updated_at, finished_at
            FROM task_runs
            WHERE session_id=? AND task_kind='question_extraction_session'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    qset = _stored_questions_for_response(session_id)
    return {
        "session_id": session_id,
        "question_set_count": len(qset),
        "latest_task": dict(task) if task else None,
    }


@app.get("/api/sessions/{session_id}/questions")
def session_extracted_questions(session_id: str, request: Request) -> dict:
    """杩斿洖鏈洖鍚堝凡缁忓叆搴撶殑棰樼洰鍒楄〃锛屼緵鍘嗗彶鍏ュ彛/棰樺簱椤佃鍙栥€?""
    init_db()
    principal = principal_from_request(request)
    with connect() as conn:
        require_account_session(conn, session_id, principal)
    questions = _stored_questions_for_response(session_id)
    return {
        "session_id": session_id,
        "question_count": len(questions),
        "questions": questions,
    }


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
    focus = requested_focus or ("鏍规嵁瀛︾敓杈撳叆鍒濆鍖栧叧娉ㄧ偣锛? + "銆?.join(tag_label(tag) for tag in inferred) if inferred else "")
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
                "鍗曞紶鎷嶉瑙ｆ瀽",
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
        record_report_event(session_id, "student_goal", "瀛︾敓鏈洖鍚堣姹?, goal)
    _, filename, _ = await save_upload(
        image,
        session_id,
        "single",
        page_hint=page_hint,
        question_hint=question_hint,
        sequence_index=0,
    )
    analysis_id = uuid.uuid4().hex
    prompt = prompts.render_prompt("single_analysis", page_hint=page_hint or "鏈煡", question_hint=question_hint or "鏈煡")
    strategy_context = build_strategy_context({"student_goal": goal, "assistant_focus": focus, "inferred_needs": inferred, "report_style": style})
    prompt = f"{prompt}\n\n{strategy_context}"
    with connect() as conn:
        conn.execute(
            "INSERT INTO analyses(id, session_id, scope, status, prompt, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (analysis_id, session_id, "single", "running", prompt, now, now),
        )
    with connect() as conn:
        conn.execute("UPDATE sessions SET status=?, updated_at=? WHERE id=?", ("analyzing", utc_now(), session_id))
    emit_log("鏀跺埌鍗曞紶鎷嶉鍥剧墖锛屽凡淇濆瓨骞惰繘鍏ュ悗鍙拌В鏋愭祦绋?, session_id=session_id, device_id=device_id)
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
    title: str = Form("鏅鸿兘瑙傚療瀛︿範鍥炲悎"),
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
    focus = requested_focus or ("鏍规嵁瀛︾敓杈撳叆鍒濆鍖栧叧娉ㄧ偣锛? + "銆?.join(tag_label(tag) for tag in inferred) if inferred else "")
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
    emit_log("鍒涘缓瀛︿範鍥炲悎", session_id=session_id, device_id=device_id)
    if goal:
        record_report_event(session_id, "student_goal", "瀛︾敓鏈洖鍚堣姹?, goal)
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
        "瀛︿範鐩爣/鎶ュ憡绛栫暐鏇存柊",
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
    question_crops: list[UploadFile] | None = File(None),
    device_id: str = Form("iphone"),
    environment: str = Form(""),
    capture_meta: str = Form(""),
    question_crop_manifest: str = Form(""),
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
    crop_upload_result = await save_question_crop_uploads(
        question_crops or [],
        question_crop_manifest,
        session_id=session_id,
        batch_id=batch_id,
        image_rows=image_rows,
    )
    crop_saved_count = int(crop_upload_result.get("saved_count") or 0)
    crop_duplicate_count = int(crop_upload_result.get("duplicate_count") or 0)
    crop_skipped_count = int(crop_upload_result.get("skipped_count") or 0)
    crop_expanded_count = int(crop_upload_result.get("expanded_count") or 0)
    crop_client_count = int(crop_upload_result.get("client_crop_count") or 0)
    crop_rect_only_count = int(crop_upload_result.get("rect_only_count") or 0)
    crop_replaced_count = int(crop_upload_result.get("replaced_count") or 0)
    crop_expansion_duration_ms = int(crop_upload_result.get("expansion_duration_ms") or 0)
    crop_client_metrics = crop_upload_result.get("client_metrics")
    if not isinstance(crop_client_metrics, dict):
        crop_client_metrics = {}
    now = utc_now()
    analysis_id = uuid.uuid4().hex
    previous_context = build_previous_batch_context(session_id, exclude_batch_id=batch_id)
    if analysis_image_rows:
        learning_context = build_learning_items_context(session_id)
        merged_context = previous_context
        if learning_context:
            merged_context = f"{previous_context}\n\n姝ゅ墠宸茬粨鏋勫寲淇濆瓨鐨勫涔犳潯鐩細\n{learning_context}"
        analysis_chunks = chunk_sequence(analysis_image_rows, OBSERVATION_ANALYSIS_MAX_IMAGES)
        analysis_records: list[tuple[str, list[dict], list[str]]] = []
        for chunk_rows in analysis_chunks:
            chunk_analysis_id = uuid.uuid4().hex
            chunk_filenames = [row["filename"] for row in chunk_rows]
            analysis_records.append((chunk_analysis_id, chunk_rows, chunk_filenames))
        with connect() as conn:
            for chunk_analysis_id, chunk_rows, _ in analysis_records:
                prompt = build_batch_prompt(environment, chunk_rows, merged_context, build_strategy_context(session_strategy(session)))
                conn.execute(
                    "INSERT INTO analyses(id, session_id, batch_id, scope, status, prompt, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                    (chunk_analysis_id, session_id, batch_id, "batch", "running", prompt, now, now),
                )
            conn.execute("UPDATE sessions SET status=?, updated_at=? WHERE id=?", ("analyzing", now, session_id))
        analysis_id = analysis_records[0][0]
        emit_log(
            (
                f"鏀跺埌鏅鸿兘瑙傚療鎵规锛歿len(filenames)} 寮狅紝鍏朵腑鏂板鍏抽敭鐢婚潰 {len(analysis_filenames)} 寮狅紝"
                f"閲嶅瑙傚療 {sum(1 for row in image_rows if row.get('novelty_status') == 'duplicate')} 寮狅紝"
                f"鏃犵浉鍏崇敾闈?{sum(1 for row in image_rows if row.get('novelty_status') == 'invalid')} 寮狅紱"
                f"棰樼洰瑁佸壀 {crop_saved_count} 寮狅紙閲嶅 {crop_duplicate_count}銆佽烦杩?{crop_skipped_count}锛夛紱"
                f"鎷嗗垎涓?{len(analysis_records)} 涓皬鍒嗘瀽浠诲姟"
            ),
            session_id=session_id,
            device_id=device_id,
        )
        for chunk_index, (chunk_analysis_id, _, chunk_filenames) in enumerate(analysis_records, start=1):
            record_task_run(
                "vision_analysis",
                session_id=session_id,
                analysis_id=chunk_analysis_id,
                payload={
                    "scope": "batch",
                    "batch_id": batch_id,
                    "filenames": chunk_filenames,
                    "chunk_index": chunk_index,
                    "chunk_count": len(analysis_records),
                },
                priority=TASK_PRIORITY_BACKGROUND,
            )
        background_tasks.add_task(execute_next_task_run_now)
    else:
        invalid_count = sum(1 for row in image_rows if row.get("novelty_status") == "invalid")
        prompt = "鏃犳柊澧炲彲瑙ｆ瀽鍏抽敭鐢婚潰銆?
        content = skipped_batch_content(image_rows)
        with connect() as conn:
            conn.execute(
                "INSERT INTO analyses(id, session_id, batch_id, scope, status, prompt, content, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (analysis_id, session_id, batch_id, "batch", "done", prompt, content, now, now),
            )
            conn.execute("UPDATE sessions SET status=?, updated_at=? WHERE id=?", ("analyzed", now, session_id))
        if invalid_count:
            emit_log(
                f"鏀跺埌鏅鸿兘瑙傚療鎵规锛歿len(filenames)} 寮狅紝鍏朵腑鏃犵浉鍏崇敾闈?{invalid_count} 寮狅紝棰樼洰瑁佸壀 {crop_saved_count} 寮狅紝宸茶烦杩囧ぇ妯″瀷瑙ｆ瀽",
                session_id=session_id,
                device_id=device_id,
            )
        else:
            emit_log(f"鏀跺埌閲嶅瑙傚療鎵规锛歿len(filenames)} 寮狅紝棰樼洰瑁佸壀 {crop_saved_count} 寮狅紝鏃犳柊澧炲叧閿敾闈紝宸茶烦杩囧ぇ妯″瀷瑙ｆ瀽", session_id=session_id, device_id=device_id)
    return {
        "session_id": session_id,
        "batch_id": batch_id,
        "analysis_id": analysis_id,
        "image_count": len(filenames),
        "analysis_image_count": len(analysis_filenames),
        "question_crop_count": crop_saved_count,
        "question_crop_expanded_count": crop_expanded_count,
        "question_crop_client_count": crop_client_count,
        "question_crop_rect_only_count": crop_rect_only_count,
        "question_crop_replaced_count": crop_replaced_count,
        "question_crop_server_duration_ms": crop_expansion_duration_ms,
        "question_crop_client_metrics": crop_client_metrics,
        "question_crop_duplicate_count": crop_duplicate_count,
        "question_crop_skipped_count": crop_skipped_count,
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
    emit_log("鏀跺埌缁撴潫瀛︿範鍥炲悎璇锋眰锛屽紑濮嬫眹鎬绘渶缁堟姤鍛?, session_id=session_id, device_id=device_id)
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


# 浜屾湡路鑷劧璇█閰嶇疆绠″锛坕ntent_router锛夈€傜嫭绔?router锛宎ccount-scoped锛?
# 绔偣鍐呴儴澶嶇敤 principal_from_request / effective_llm_settings / run_with_llm_gate锛堟儼鎬у鍏ヨ閬垮惊鐜級銆?
from . import intent_router as _intent_router  # noqa: E402

app.include_router(_intent_router.router)

