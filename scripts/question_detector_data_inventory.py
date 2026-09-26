"""Inventory PXJ/KPAI data roots for question-detector training readiness."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def is_noise_path(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return "node_modules" in parts or ".next" in parts or ".git" in parts


def count_images(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def inspect_sqlite(db_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(db_path),
        "size_bytes": db_path.stat().st_size,
        "tables": [],
        "counts": {},
        "usable_for_weak_labels": False,
    }
    try:
        with sqlite3.connect(db_path) as conn:
            tables = sorted(row[0] for row in conn.execute("select name from sqlite_master where type='table'"))
            result["tables"] = tables
            for table in ("images", "session_observations", "session_question_crops"):
                if table in tables:
                    result["counts"][table] = conn.execute(f"select count(*) from {table}").fetchone()[0]
            result["usable_for_weak_labels"] = int(result["counts"].get("session_question_crops", 0)) > 0
    except Exception as exc:
        result["error"] = str(exc)
    return result


def inventory_root(root: Path) -> dict[str, Any]:
    dbs = [
        inspect_sqlite(path)
        for path in sorted(root.rglob("*.sqlite3"))
        if path.is_file() and not is_noise_path(path)
    ]
    image_dirs = []
    for candidate in sorted(root.rglob("images")):
        if candidate.is_dir() and not is_noise_path(candidate):
            count = count_images(candidate)
            if count:
                image_dirs.append({"path": str(candidate), "image_count": count})
    crop_ready = [db for db in dbs if db.get("usable_for_weak_labels")]
    return {
        "root": str(root),
        "sqlite_count": len(dbs),
        "sqlite_with_question_crops": len(crop_ready),
        "total_question_crops": sum(int(db.get("counts", {}).get("session_question_crops", 0)) for db in dbs),
        "total_db_images": sum(int(db.get("counts", {}).get("images", 0)) for db in dbs),
        "image_dirs": image_dirs[:50],
        "sqlite": dbs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory available data for question detector training.")
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-data-inventory.json"))
    args = parser.parse_args()
    roots = [root for root in args.roots if root.exists()]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "roots": [inventory_root(root) for root in roots],
    }
    report["summary"] = {
        "root_count": len(report["roots"]),
        "sqlite_with_question_crops": sum(root["sqlite_with_question_crops"] for root in report["roots"]),
        "total_question_crops": sum(root["total_question_crops"] for root in report["roots"]),
        "total_db_images": sum(root["total_db_images"] for root in report["roots"]),
    }
    write_json(args.out, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
