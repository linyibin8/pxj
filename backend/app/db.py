import contextvars
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import get_settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Per-account database routing
#
# Architecture: one shared control DB (auth/identity/model-config/job-queue)
# plus one SQLite file per account holding that account's learning data. The
# active account for the current request/task lives in a ContextVar so the
# ~580 existing connect() call sites stay unchanged; only control-plane code
# uses connect_control(). The full schema is created in every DB file, so the
# unused tables in each file are simply empty.
# ---------------------------------------------------------------------------

CONTROL_DB_NAME = "control.sqlite3"
_current_account: contextvars.ContextVar[str] = contextvars.ContextVar("pxj_current_account", default="")
_ready_accounts: set[str] = set()
_ready_lock = threading.Lock()
_control_ready = False


def set_current_account(account_id: str | None):
    return _current_account.set(account_id or "")


def reset_current_account(token) -> None:
    try:
        _current_account.reset(token)
    except Exception:
        pass


def _safe_account(account_id: str | None) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", account_id or "")[:120]
    return cleaned or (get_settings().default_account_id or "local")


def current_account_id() -> str:
    return _safe_account(_current_account.get() or "")


def control_db_path() -> Path:
    return get_settings().data_dir / CONTROL_DB_NAME


def account_db_path(account_id: str | None = None) -> Path:
    aid = _safe_account(account_id or _current_account.get() or "")
    directory = get_settings().data_dir / "accounts"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{aid}.sqlite3"


def db_path() -> Path:
    """Back-compat: path of the current account's database file."""
    return account_db_path()


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connect_control() -> Iterator[sqlite3.Connection]:
    """Connection to the shared control DB (accounts/users/identity/model_configs/task_runs)."""
    global _control_ready
    if not _control_ready:
        init_control_db()
    conn = _open(control_db_path())
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


@contextmanager
def connect(account_id: str | None = None) -> Iterator[sqlite3.Connection]:
    """Connection to a single account's data DB (defaults to the active account)."""
    aid = _safe_account(account_id or _current_account.get() or "")
    if aid not in _ready_accounts:
        ensure_account_db(aid)
    conn = _open(account_db_path(aid))
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_control_db() -> None:
    global _control_ready
    with _ready_lock:
        conn = _open(control_db_path())
        try:
            _init_schema(conn)
            conn.commit()
        finally:
            conn.close()
        _control_ready = True


def ensure_account_db(account_id: str) -> None:
    aid = _safe_account(account_id)
    if aid in _ready_accounts:
        return
    with _ready_lock:
        if aid in _ready_accounts:
            return
        conn = _open(account_db_path(aid))
        try:
            _init_schema(conn)
            conn.commit()
        finally:
            conn.close()
        _ready_accounts.add(aid)


def list_account_ids() -> list[str]:
    """All account ids that have a data DB file (for cross-account background work)."""
    directory = get_settings().data_dir / "accounts"
    ids: list[str] = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.sqlite3")):
            ids.append(path.stem)
    default_id = _safe_account(get_settings().default_account_id or "local")
    if default_id not in ids:
        ids.append(default_id)
    return ids


def init_db() -> None:
    init_control_db()
    ensure_account_db(get_settings().default_account_id or "local")


def _init_schema(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS accounts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                plan TEXT NOT NULL DEFAULT 'free',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL DEFAULT '',
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'owner',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login_at TEXT NOT NULL DEFAULT '',
                FOREIGN KEY(account_id) REFERENCES accounts(id)
            );

            CREATE TABLE IF NOT EXISTS account_members (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'owner',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(account_id) REFERENCES accounts(id),
                FOREIGN KEY(user_id) REFERENCES users(id),
                UNIQUE(account_id, user_id)
            );

            CREATE TABLE IF NOT EXISTS identity_profiles (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                user_id TEXT NOT NULL DEFAULT '',
                profile_type TEXT NOT NULL,
                display_name TEXT NOT NULL,
                student_id TEXT NOT NULL DEFAULT '',
                relation TEXT NOT NULL DEFAULT '',
                metadata TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(account_id) REFERENCES accounts(id),
                FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS model_configs (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                owner_user_id TEXT NOT NULL DEFAULT '',
                provider TEXT NOT NULL,
                name TEXT NOT NULL,
                base_url TEXT NOT NULL,
                api_key_encrypted TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                is_default INTEGER NOT NULL DEFAULT 0,
                max_concurrency INTEGER NOT NULL DEFAULT 1,
                min_interval_seconds REAL NOT NULL DEFAULT 0,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(account_id) REFERENCES accounts(id),
                FOREIGN KEY(owner_user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS llm_usage_events (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL DEFAULT '',
                user_id TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL DEFAULT '',
                request_label TEXT NOT NULL DEFAULT '',
                lane TEXT NOT NULL DEFAULT '',
                provider TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                base_url TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'running',
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                duration_ms INTEGER NOT NULL DEFAULT 0,
                error TEXT NOT NULL DEFAULT '',
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL DEFAULT 'local',
                student_profile_id TEXT NOT NULL DEFAULT '',
                created_by_user_id TEXT NOT NULL DEFAULT '',
                device_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                batch_id TEXT,
                kind TEXT NOT NULL,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                page_hint TEXT NOT NULL DEFAULT '',
                question_hint TEXT NOT NULL DEFAULT '',
                captured_at TEXT NOT NULL DEFAULT '',
                sequence_index INTEGER NOT NULL DEFAULT 0,
                capture_meta TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                batch_id TEXT,
                scope TEXT NOT NULL,
                status TEXT NOT NULL,
                prompt TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                device_id TEXT,
                level TEXT NOT NULL,
                source TEXT NOT NULL,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS prompts (
                prompt_key TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_observations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                batch_id TEXT,
                image_id TEXT,
                captured_at TEXT NOT NULL DEFAULT '',
                sequence_index INTEGER NOT NULL DEFAULT 0,
                visual_hash TEXT NOT NULL DEFAULT '',
                visual_sample TEXT NOT NULL DEFAULT '',
                text_hash TEXT NOT NULL DEFAULT '',
                text_tokens TEXT NOT NULL DEFAULT '[]',
                signal_summary TEXT NOT NULL DEFAULT '',
                novelty_status TEXT NOT NULL DEFAULT 'unknown',
                duplicate_of_image_id TEXT NOT NULL DEFAULT '',
                visual_distance REAL,
                text_distance REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                FOREIGN KEY(image_id) REFERENCES images(id)
            );

            CREATE TABLE IF NOT EXISTS learning_items (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                batch_id TEXT,
                analysis_id TEXT,
                item_type TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                subject TEXT NOT NULL DEFAULT '',
                page_ref TEXT NOT NULL DEFAULT '',
                question_ref TEXT NOT NULL DEFAULT '',
                location_ref TEXT NOT NULL DEFAULT '',
                source_summary TEXT NOT NULL DEFAULT '',
                source_image_details TEXT NOT NULL DEFAULT '[]',
                content_hash TEXT NOT NULL,
                first_seen_at TEXT NOT NULL DEFAULT '',
                last_seen_at TEXT NOT NULL DEFAULT '',
                first_sequence_index INTEGER NOT NULL DEFAULT 0,
                last_sequence_index INTEGER NOT NULL DEFAULT 0,
                source_image_ids TEXT NOT NULL DEFAULT '[]',
                evidence_count INTEGER NOT NULL DEFAULT 1,
                confidence TEXT NOT NULL DEFAULT 'observed',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                FOREIGN KEY(analysis_id) REFERENCES analyses(id),
                UNIQUE(session_id, item_type, content_hash)
            );

            CREATE TABLE IF NOT EXISTS mistake_items (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                learning_item_id TEXT,
                batch_id TEXT,
                analysis_id TEXT,
                title TEXT NOT NULL,
                question_text TEXT NOT NULL DEFAULT '',
                student_answer TEXT NOT NULL DEFAULT '',
                expected_answer TEXT NOT NULL DEFAULT '',
                error_reason TEXT NOT NULL DEFAULT '',
                knowledge_points TEXT NOT NULL DEFAULT '[]',
                subject TEXT NOT NULL DEFAULT '',
                page_ref TEXT NOT NULL DEFAULT '',
                question_ref TEXT NOT NULL DEFAULT '',
                location_ref TEXT NOT NULL DEFAULT '',
                error_type TEXT NOT NULL DEFAULT '',
                correction TEXT NOT NULL DEFAULT '',
                next_action TEXT NOT NULL DEFAULT '',
                source_summary TEXT NOT NULL DEFAULT '',
                source_image_details TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'suspected',
                review_state TEXT NOT NULL DEFAULT 'new',
                next_review_at TEXT NOT NULL DEFAULT '',
                last_reviewed_at TEXT NOT NULL DEFAULT '',
                review_count INTEGER NOT NULL DEFAULT 0,
                review_note TEXT NOT NULL DEFAULT '',
                confirmed_at TEXT NOT NULL DEFAULT '',
                ignored_at TEXT NOT NULL DEFAULT '',
                corrected_at TEXT NOT NULL DEFAULT '',
                mastered_at TEXT NOT NULL DEFAULT '',
                evidence TEXT NOT NULL DEFAULT '',
                source_image_ids TEXT NOT NULL DEFAULT '[]',
                first_seen_at TEXT NOT NULL DEFAULT '',
                last_seen_at TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                FOREIGN KEY(learning_item_id) REFERENCES learning_items(id),
                FOREIGN KEY(analysis_id) REFERENCES analyses(id),
                UNIQUE(session_id, content_hash)
            );

            CREATE TABLE IF NOT EXISTS report_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                analysis_id TEXT,
                event_type TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                FOREIGN KEY(analysis_id) REFERENCES analyses(id)
            );

            CREATE TABLE IF NOT EXISTS extracted_questions (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                question_index INTEGER NOT NULL DEFAULT 0,
                question_key TEXT NOT NULL,
                number TEXT NOT NULL DEFAULT '',
                subject TEXT NOT NULL DEFAULT '',
                qtype TEXT NOT NULL DEFAULT '',
                stem TEXT NOT NULL DEFAULT '',
                options TEXT NOT NULL DEFAULT '[]',
                blanks INTEGER NOT NULL DEFAULT 0,
                figure_note TEXT NOT NULL DEFAULT '',
                has_student_answer INTEGER NOT NULL DEFAULT 0,
                student_answer TEXT NOT NULL DEFAULT '',
                fingerprint TEXT NOT NULL DEFAULT '',
                simhash TEXT NOT NULL DEFAULT '',
                src_filename TEXT NOT NULL DEFAULT '',
                source_image_id TEXT NOT NULL DEFAULT '',
                source_crop_id TEXT NOT NULL DEFAULT '',
                crop_filename TEXT NOT NULL DEFAULT '',
                crop_rect TEXT NOT NULL DEFAULT '{}',
                crop_hash TEXT NOT NULL DEFAULT '',
                payload TEXT NOT NULL DEFAULT '{}',
                first_seen_at TEXT NOT NULL DEFAULT '',
                last_seen_at TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                UNIQUE(session_id, question_key)
            );

            CREATE TABLE IF NOT EXISTS session_question_crops (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                batch_id TEXT NOT NULL DEFAULT '',
                image_id TEXT NOT NULL,
                sequence_index INTEGER NOT NULL DEFAULT 0,
                manifest_index INTEGER NOT NULL DEFAULT 0,
                question_index INTEGER NOT NULL DEFAULT 0,
                question_key TEXT NOT NULL DEFAULT '',
                fingerprint TEXT NOT NULL DEFAULT '',
                crop_hash TEXT NOT NULL DEFAULT '',
                text_hash TEXT NOT NULL DEFAULT '',
                normalized_rect TEXT NOT NULL DEFAULT '{}',
                crop_rect TEXT NOT NULL DEFAULT '{}',
                source_image_size TEXT NOT NULL DEFAULT '{}',
                crop_image_size TEXT NOT NULL DEFAULT '{}',
                preview_text TEXT NOT NULL DEFAULT '',
                crop_filename TEXT NOT NULL DEFAULT '',
                original_name TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'ready',
                source TEXT NOT NULL DEFAULT 'client_manifest',
                confidence REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                FOREIGN KEY(image_id) REFERENCES images(id),
                UNIQUE(session_id, image_id, manifest_index)
            );

            CREATE TABLE IF NOT EXISTS qa_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                image_id TEXT,
                source TEXT NOT NULL DEFAULT '',
                trigger_type TEXT NOT NULL DEFAULT '',
                question TEXT NOT NULL,
                answer TEXT NOT NULL DEFAULT '',
                focus TEXT NOT NULL DEFAULT '{}',
                context TEXT NOT NULL DEFAULT '{}',
                gesture TEXT NOT NULL DEFAULT '{}',
                tts_status TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'running',
                interrupted_at TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                FOREIGN KEY(image_id) REFERENCES images(id)
            );

            CREATE TABLE IF NOT EXISTS teaching_visualizations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'ready',
                title TEXT NOT NULL DEFAULT '',
                topic_type TEXT NOT NULL DEFAULT '',
                trigger_reason TEXT NOT NULL DEFAULT '',
                prompt TEXT NOT NULL DEFAULT '',
                html_filename TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                UNIQUE(source_type, source_id)
            );

            CREATE TABLE IF NOT EXISTS asset_documents (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                asset_kind TEXT NOT NULL,
                asset_id TEXT NOT NULL,
                subject TEXT NOT NULL DEFAULT '',
                page_ref TEXT NOT NULL DEFAULT '',
                question_ref TEXT NOT NULL DEFAULT '',
                location_ref TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                body TEXT NOT NULL DEFAULT '',
                search_text TEXT NOT NULL DEFAULT '',
                source_image_ids TEXT NOT NULL DEFAULT '[]',
                source_image_details TEXT NOT NULL DEFAULT '[]',
                first_seen_at TEXT NOT NULL DEFAULT '',
                last_seen_at TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id),
                UNIQUE(asset_kind, asset_id)
            );

            CREATE TABLE IF NOT EXISTS task_runs (
                id TEXT PRIMARY KEY,
                task_kind TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                session_id TEXT,
                analysis_id TEXT,
                payload TEXT NOT NULL DEFAULT '{}',
                priority INTEGER NOT NULL DEFAULT 100,
                attempts INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL DEFAULT 3,
                last_error TEXT NOT NULL DEFAULT '',
                available_at TEXT NOT NULL DEFAULT '',
                started_at TEXT NOT NULL DEFAULT '',
                finished_at TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL DEFAULT '',
                qa_event_id TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                message_type TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL,
                payload TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_profiles (
                account_id TEXT NOT NULL DEFAULT 'local',
                scope TEXT NOT NULL,
                profile TEXT NOT NULL DEFAULT '',
                source_count INTEGER NOT NULL DEFAULT 0,
                latest_event_at TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                PRIMARY KEY(account_id, scope)
            );

            CREATE TABLE IF NOT EXISTS review_events (
                id TEXT PRIMARY KEY,
                mistake_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                event_type TEXT NOT NULL DEFAULT 'review',
                result TEXT NOT NULL,
                note TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                duration_seconds INTEGER,
                score REAL,
                payload TEXT NOT NULL DEFAULT '{}',
                reviewed_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(mistake_id) REFERENCES mistake_items(id),
                FOREIGN KEY(session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS device_states (
                device_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                state TEXT NOT NULL DEFAULT '{}',
                last_seen_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS control_commands (
                id TEXT PRIMARY KEY,
                device_id TEXT NOT NULL DEFAULT '',
                session_id TEXT NOT NULL DEFAULT '',
                command_type TEXT NOT NULL,
                payload TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'pending',
                source TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                delivered_at TEXT NOT NULL DEFAULT '',
                acknowledged_at TEXT NOT NULL DEFAULT '',
                expires_at TEXT NOT NULL
            );
            """
        )
        migrate_memory_profiles_schema(conn)
        ensure_column(conn, "sessions", "finished_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "sessions", "account_id", "TEXT NOT NULL DEFAULT 'local'")
        ensure_column(conn, "sessions", "student_profile_id", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "sessions", "created_by_user_id", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "sessions", "report_generated_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "sessions", "student_goal", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "sessions", "assistant_focus", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "sessions", "inferred_needs", "TEXT NOT NULL DEFAULT '[]'")
        ensure_column(conn, "sessions", "report_style", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "logs", "account_id", "TEXT NOT NULL DEFAULT 'local'")
        ensure_column(conn, "task_runs", "account_id", "TEXT NOT NULL DEFAULT 'local'")
        ensure_column(conn, "memory_events", "account_id", "TEXT NOT NULL DEFAULT 'local'")
        ensure_column(conn, "memory_profiles", "account_id", "TEXT NOT NULL DEFAULT 'local'")
        ensure_column(conn, "device_states", "account_id", "TEXT NOT NULL DEFAULT 'local'")
        ensure_column(conn, "control_commands", "account_id", "TEXT NOT NULL DEFAULT 'local'")
        ensure_column(conn, "llm_usage_events", "account_id", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "llm_usage_events", "user_id", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "llm_usage_events", "session_id", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "llm_usage_events", "request_label", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "llm_usage_events", "lane", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "llm_usage_events", "provider", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "llm_usage_events", "model", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "llm_usage_events", "base_url", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "llm_usage_events", "status", "TEXT NOT NULL DEFAULT 'running'")
        ensure_column(conn, "llm_usage_events", "prompt_tokens", "INTEGER")
        ensure_column(conn, "llm_usage_events", "completion_tokens", "INTEGER")
        ensure_column(conn, "llm_usage_events", "total_tokens", "INTEGER")
        ensure_column(conn, "llm_usage_events", "duration_ms", "INTEGER NOT NULL DEFAULT 0")
        ensure_column(conn, "llm_usage_events", "error", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "llm_usage_events", "started_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "llm_usage_events", "finished_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "llm_usage_events", "created_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "llm_usage_events", "updated_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "images", "page_hint", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "images", "question_hint", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "images", "captured_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "images", "sequence_index", "INTEGER NOT NULL DEFAULT 0")
        ensure_column(conn, "images", "capture_meta", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "session_observations", "visual_sample", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "learning_items", "subject", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "learning_items", "page_ref", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "learning_items", "question_ref", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "learning_items", "location_ref", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "learning_items", "source_summary", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "learning_items", "source_image_details", "TEXT NOT NULL DEFAULT '[]'")
        ensure_column(conn, "mistake_items", "subject", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "mistake_items", "page_ref", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "mistake_items", "question_ref", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "mistake_items", "location_ref", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "mistake_items", "error_type", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "mistake_items", "correction", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "mistake_items", "next_action", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "mistake_items", "source_summary", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "mistake_items", "source_image_details", "TEXT NOT NULL DEFAULT '[]'")
        ensure_column(conn, "mistake_items", "review_state", "TEXT NOT NULL DEFAULT 'new'")
        ensure_column(conn, "mistake_items", "next_review_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "mistake_items", "last_reviewed_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "mistake_items", "review_count", "INTEGER NOT NULL DEFAULT 0")
        ensure_column(conn, "mistake_items", "review_note", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "mistake_items", "confirmed_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "mistake_items", "ignored_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "mistake_items", "corrected_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "mistake_items", "mastered_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "mistake_items", "detection_method", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "question_index", "INTEGER NOT NULL DEFAULT 0")
        ensure_column(conn, "extracted_questions", "question_key", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "number", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "subject", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "qtype", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "stem", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "options", "TEXT NOT NULL DEFAULT '[]'")
        ensure_column(conn, "extracted_questions", "blanks", "INTEGER NOT NULL DEFAULT 0")
        ensure_column(conn, "extracted_questions", "figure_note", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "has_student_answer", "INTEGER NOT NULL DEFAULT 0")
        ensure_column(conn, "extracted_questions", "student_answer", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "fingerprint", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "simhash", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "src_filename", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "source_image_id", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "source_crop_id", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "crop_filename", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "crop_rect", "TEXT NOT NULL DEFAULT '{}'")
        ensure_column(conn, "extracted_questions", "crop_hash", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "payload", "TEXT NOT NULL DEFAULT '{}'")
        ensure_column(conn, "extracted_questions", "first_seen_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "last_seen_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "created_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "extracted_questions", "updated_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "session_question_crops", "batch_id", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "session_question_crops", "image_id", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "session_question_crops", "sequence_index", "INTEGER NOT NULL DEFAULT 0")
        ensure_column(conn, "session_question_crops", "manifest_index", "INTEGER NOT NULL DEFAULT 0")
        ensure_column(conn, "session_question_crops", "question_index", "INTEGER NOT NULL DEFAULT 0")
        ensure_column(conn, "session_question_crops", "question_key", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "session_question_crops", "fingerprint", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "session_question_crops", "crop_hash", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "session_question_crops", "text_hash", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "session_question_crops", "normalized_rect", "TEXT NOT NULL DEFAULT '{}'")
        ensure_column(conn, "session_question_crops", "crop_rect", "TEXT NOT NULL DEFAULT '{}'")
        ensure_column(conn, "session_question_crops", "source_image_size", "TEXT NOT NULL DEFAULT '{}'")
        ensure_column(conn, "session_question_crops", "crop_image_size", "TEXT NOT NULL DEFAULT '{}'")
        ensure_column(conn, "session_question_crops", "preview_text", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "session_question_crops", "crop_filename", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "session_question_crops", "original_name", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "session_question_crops", "status", "TEXT NOT NULL DEFAULT 'ready'")
        ensure_column(conn, "session_question_crops", "source", "TEXT NOT NULL DEFAULT 'client_manifest'")
        ensure_column(conn, "session_question_crops", "confidence", "REAL")
        ensure_column(conn, "session_question_crops", "created_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "session_question_crops", "updated_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "review_events", "event_type", "TEXT NOT NULL DEFAULT 'review'")
        ensure_column(conn, "review_events", "note", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "review_events", "source", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "review_events", "duration_seconds", "INTEGER")
        ensure_column(conn, "review_events", "score", "REAL")
        ensure_column(conn, "review_events", "payload", "TEXT NOT NULL DEFAULT '{}'")
        ensure_column(conn, "review_events", "reviewed_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "device_states", "session_id", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "device_states", "source", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "device_states", "state", "TEXT NOT NULL DEFAULT '{}'")
        ensure_column(conn, "device_states", "last_seen_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "device_states", "updated_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "control_commands", "device_id", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "control_commands", "session_id", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "control_commands", "command_type", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "control_commands", "payload", "TEXT NOT NULL DEFAULT '{}'")
        ensure_column(conn, "control_commands", "status", "TEXT NOT NULL DEFAULT 'pending'")
        ensure_column(conn, "control_commands", "source", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "control_commands", "error", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "control_commands", "created_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "control_commands", "delivered_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "control_commands", "acknowledged_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "control_commands", "expires_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "qa_events", "image_id", "TEXT")
        ensure_column(conn, "qa_events", "source", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "qa_events", "trigger_type", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "qa_events", "focus", "TEXT NOT NULL DEFAULT '{}'")
        ensure_column(conn, "qa_events", "context", "TEXT NOT NULL DEFAULT '{}'")
        ensure_column(conn, "qa_events", "gesture", "TEXT NOT NULL DEFAULT '{}'")
        ensure_column(conn, "qa_events", "tts_status", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "qa_events", "status", "TEXT NOT NULL DEFAULT 'running'")
        ensure_column(conn, "qa_events", "interrupted_at", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "teaching_visualizations", "status", "TEXT NOT NULL DEFAULT 'ready'")
        ensure_column(conn, "teaching_visualizations", "title", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "teaching_visualizations", "topic_type", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "teaching_visualizations", "trigger_reason", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "teaching_visualizations", "prompt", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "teaching_visualizations", "html_filename", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "teaching_visualizations", "error", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "task_runs", "priority", "INTEGER NOT NULL DEFAULT 100")
        ensure_column(conn, "memory_events", "session_id", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "memory_events", "qa_event_id", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "memory_events", "source", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "memory_events", "message_type", "TEXT NOT NULL DEFAULT ''")
        ensure_column(conn, "memory_events", "payload", "TEXT NOT NULL DEFAULT '{}'")
        ensure_column(conn, "memory_profiles", "source_count", "INTEGER NOT NULL DEFAULT 0")
        ensure_column(conn, "memory_profiles", "latest_event_at", "TEXT NOT NULL DEFAULT ''")
        settings = get_settings()
        default_account_id = settings.default_account_id or "local"
        now = utc_now()
        conn.execute(
            """
            INSERT INTO accounts(id, name, status, plan, created_at, updated_at)
            VALUES(?, 'Local Account', 'active', 'legacy', ?, ?)
            ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at
            """,
            (default_account_id, now, now),
        )
        conn.execute("UPDATE sessions SET account_id=? WHERE account_id='' OR account_id IS NULL", (default_account_id,))
        conn.execute("UPDATE logs SET account_id=? WHERE account_id='' OR account_id IS NULL", (default_account_id,))
        conn.execute("UPDATE task_runs SET account_id=? WHERE account_id='' OR account_id IS NULL", (default_account_id,))
        conn.execute("UPDATE memory_events SET account_id=? WHERE account_id='' OR account_id IS NULL", (default_account_id,))
        conn.execute("UPDATE memory_profiles SET account_id=? WHERE account_id='' OR account_id IS NULL", (default_account_id,))
        conn.execute("UPDATE device_states SET account_id=? WHERE account_id='' OR account_id IS NULL", (default_account_id,))
        conn.execute("UPDATE control_commands SET account_id=? WHERE account_id='' OR account_id IS NULL", (default_account_id,))
        conn.execute(
            """
            UPDATE mistake_items
            SET status='mastered',
                review_state=CASE WHEN review_state IN ('', 'new', 'queued', 'scheduled', 'reviewing', 'done') THEN 'mastered' ELSE review_state END,
                next_review_at='',
                mastered_at=CASE WHEN mastered_at='' THEN ? ELSE mastered_at END,
                updated_at=CASE WHEN updated_at='' THEN ? ELSE updated_at END
            WHERE status='resolved'
            """,
            (now, now),
        )
        conn.execute(
            """
            UPDATE mistake_items
            SET next_review_at=''
            WHERE status IN ('ignored', 'mastered') OR review_state IN ('ignored', 'mastered')
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_vectors (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                ref_id TEXT NOT NULL,
                student_profile_id TEXT NOT NULL DEFAULT '',
                text TEXT NOT NULL DEFAULT '',
                embedding TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                UNIQUE(kind, ref_id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_knowledge_vectors_kind ON knowledge_vectors(kind, updated_at DESC)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_memories (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL DEFAULT 'local',
                kind TEXT NOT NULL DEFAULT 'fact',
                text TEXT NOT NULL,
                embedding TEXT NOT NULL DEFAULT '',
                importance REAL NOT NULL DEFAULT 0.5,
                status TEXT NOT NULL DEFAULT 'active',
                source_event_id TEXT NOT NULL DEFAULT '',
                use_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_memories_account ON agent_memories(account_id, status, updated_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_memories_kind ON agent_memories(account_id, kind, status)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_deltas (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL DEFAULT 'local',
                qa_event_id TEXT NOT NULL DEFAULT '',
                memory_id TEXT NOT NULL DEFAULT '',
                op TEXT NOT NULL DEFAULT 'add',
                kind TEXT NOT NULL DEFAULT 'fact',
                text TEXT NOT NULL DEFAULT '',
                seen INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_deltas_account ON memory_deltas(account_id, seen, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_users_account ON users(account_id, status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_account_members_user ON account_members(user_id, status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_identity_profiles_account ON identity_profiles(account_id, profile_type, status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_identity_profiles_student ON identity_profiles(account_id, student_id, profile_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_model_configs_account ON model_configs(account_id, enabled, is_default)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_events_account_created ON llm_usage_events(account_id, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_usage_events_status ON llm_usage_events(status, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_account_created ON sessions(account_id, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_student_created ON sessions(account_id, student_profile_id, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_images_session_sequence ON images(session_id, sequence_index, captured_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_images_session_created ON images(session_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_images_session_batch ON images(session_id, batch_id, sequence_index, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_session_scope ON analyses(session_id, scope, status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_session_created ON analyses(session_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_session_id ON logs(session_id, id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_observations_session_sequence ON session_observations(session_id, sequence_index, captured_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_observations_session_hash ON session_observations(session_id, visual_hash, novelty_status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_learning_items_session_type ON learning_items(session_id, item_type, first_seen_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mistake_items_session_status ON mistake_items(session_id, status, first_seen_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mistake_items_review ON mistake_items(status, review_state, next_review_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_report_events_session ON report_events(session_id, id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_extracted_questions_session_order ON extracted_questions(session_id, question_index)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_extracted_questions_session_updated ON extracted_questions(session_id, updated_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_extracted_questions_key ON extracted_questions(question_key)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_question_crops_session_order ON session_question_crops(session_id, sequence_index, manifest_index)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_question_crops_session_key ON session_question_crops(session_id, question_key)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_question_crops_image ON session_question_crops(image_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_events_session_created ON qa_events(session_id, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_events_image ON qa_events(image_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_visualizations_session ON teaching_visualizations(session_id, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_visualizations_source ON teaching_visualizations(source_type, source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_asset_documents_session_kind ON asset_documents(session_id, asset_kind, last_seen_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_asset_documents_asset ON asset_documents(asset_kind, asset_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_task_runs_status ON task_runs(status, available_at, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_task_runs_status_priority ON task_runs(status, priority, available_at, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_task_runs_session ON task_runs(session_id, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_events_created ON memory_events(created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_events_session ON memory_events(session_id, created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_events_qa ON memory_events(qa_event_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_review_events_mistake ON review_events(mistake_id, reviewed_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_review_events_session ON review_events(session_id, reviewed_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_review_events_result ON review_events(result, reviewed_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_device_states_seen ON device_states(last_seen_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_control_commands_pending_device ON control_commands(status, device_id, expires_at, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_control_commands_pending_session ON control_commands(status, session_id, expires_at, created_at)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_control_commands_created ON control_commands(created_at DESC)")


def ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def migrate_memory_profiles_schema(conn: sqlite3.Connection) -> None:
    table = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memory_profiles'").fetchone()
    if not table:
        return
    columns = [dict(row) for row in conn.execute("PRAGMA table_info(memory_profiles)")]
    pk_columns = [row["name"] for row in sorted((row for row in columns if row["pk"]), key=lambda item: item["pk"])]
    if pk_columns == ["account_id", "scope"]:
        return
    default_account_id = get_settings().default_account_id or "local"
    conn.execute("ALTER TABLE memory_profiles RENAME TO memory_profiles_legacy")
    conn.execute(
        """
        CREATE TABLE memory_profiles (
            account_id TEXT NOT NULL DEFAULT 'local',
            scope TEXT NOT NULL,
            profile TEXT NOT NULL DEFAULT '',
            source_count INTEGER NOT NULL DEFAULT 0,
            latest_event_at TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL,
            PRIMARY KEY(account_id, scope)
        )
        """
    )
    legacy_columns = {row["name"] for row in conn.execute("PRAGMA table_info(memory_profiles_legacy)")}
    account_expr = "account_id" if "account_id" in legacy_columns else f"'{default_account_id}'"
    conn.execute(
        f"""
        INSERT OR REPLACE INTO memory_profiles(account_id, scope, profile, source_count, latest_event_at, updated_at)
        SELECT COALESCE(NULLIF({account_expr}, ''), ?), scope, profile, source_count, latest_event_at, updated_at
        FROM memory_profiles_legacy
        """,
        (default_account_id,),
    )
    conn.execute("DROP TABLE memory_profiles_legacy")
