import argparse
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DATA_TABLES = [
    "control_commands",
    "device_states",
    "llm_usage_events",
    "review_events",
    "memory_profiles",
    "memory_events",
    "asset_documents",
    "teaching_visualizations",
    "qa_events",
    "report_events",
    "mistake_items",
    "learning_items",
    "session_observations",
    "analyses",
    "images",
    "task_runs",
    "logs",
    "sessions",
]

AUTH_TABLES = [
    "model_configs",
    "identity_profiles",
    "account_members",
    "users",
    "accounts",
]

DATA_DIRS = ["images", "thumbnails", "visualizations"]
CONTROL_DB_NAME = "control.sqlite3"
ACCOUNTS_DIR_NAME = "accounts"


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def backup_data_dir(data_dir: Path, backup_dir: Path) -> Path:
    target = backup_dir / f"history-reset-{timestamp()}"
    target.mkdir(parents=True, exist_ok=False)
    control_db = data_dir / CONTROL_DB_NAME
    if control_db.exists():
        shutil.copy2(control_db, target / control_db.name)
    accounts_dir = data_dir / ACCOUNTS_DIR_NAME
    if accounts_dir.exists():
        shutil.copytree(accounts_dir, target / ACCOUNTS_DIR_NAME)
    legacy_db = data_dir / "pxj.sqlite3"
    if legacy_db.exists():
        shutil.copy2(legacy_db, target / legacy_db.name)
    for name in DATA_DIRS:
        source = data_dir / name
        if source.exists():
            shutil.copytree(source, target / name)
    return target


def clear_table(conn: sqlite3.Connection, table: str) -> None:
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if exists:
        conn.execute(f"DELETE FROM {table}")


def clear_files(data_dir: Path) -> None:
    for name in DATA_DIRS:
        path = data_dir / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup and clear pxj operational history data.")
    parser.add_argument("--data-dir", default="backend/data", help="Data directory containing control.sqlite3 and accounts/*.sqlite3")
    parser.add_argument("--backup-dir", default="", help="Backup output directory; defaults to <data-dir>/backups")
    parser.add_argument("--include-users", action="store_true", help="Also remove accounts/users/profiles/model configs")
    parser.add_argument("--yes", action="store_true", help="Actually clear data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    backup_dir = Path(args.backup_dir).resolve() if args.backup_dir else data_dir / "backups"
    account_paths = sorted((data_dir / ACCOUNTS_DIR_NAME).glob("*.sqlite3"))
    legacy_db = data_dir / "pxj.sqlite3"
    if legacy_db.exists():
        account_paths.append(legacy_db)
    control_db = data_dir / CONTROL_DB_NAME
    if not account_paths and not control_db.exists():
        raise SystemExit(f"No database found under: {data_dir}")
    backup_path = backup_data_dir(data_dir, backup_dir)
    if not args.yes:
        print(f"Backup created at {backup_path}")
        print("Dry run only. Re-run with --yes to clear operational data.")
        return 0

    for db_path in account_paths:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("PRAGMA foreign_keys = OFF")
            for table in DATA_TABLES:
                clear_table(conn, table)
            conn.commit()
            conn.execute("VACUUM")
        finally:
            conn.close()
    if args.include_users and control_db.exists():
        conn = sqlite3.connect(control_db)
        try:
            conn.execute("PRAGMA foreign_keys = OFF")
            for table in DATA_TABLES + AUTH_TABLES:
                clear_table(conn, table)
            conn.commit()
            conn.execute("VACUUM")
        finally:
            conn.close()
    clear_files(data_dir)
    print(f"Backup created at {backup_path}")
    print("Operational history cleared.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
