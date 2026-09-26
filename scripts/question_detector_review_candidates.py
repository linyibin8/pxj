"""Build review candidates from historical SQLite data without crop labels.

This does not create detector boxes. It finds image-level positive candidates
from QA events whose answers appear to contain recognized questions, plus safe
textless negative candidates, so they can be reviewed or box-labeled later.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_json_value(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def answer_has_question(answer: str) -> bool:
    text = answer.strip()
    if not text:
        return False
    negative_markers = ("未检测到题目", "当前未检测到题目", "无法识别", "题干文字、题号")
    if any(marker in text for marker in negative_markers):
        return False
    positive_markers = ("题目：", "关键条件", "学生答案", "解题思路", "检查结果", "变式题")
    return any(marker in text for marker in positive_markers)


def safe_textless_negative(row: dict[str, Any]) -> bool:
    kind = str(row.get("kind") or "").strip().lower()
    if kind in {"qa", "extract", "grade"}:
        return False
    tokens = parse_json_value(row.get("text_tokens"))
    if isinstance(tokens, (list, dict)) and tokens:
        return False
    if str(row.get("text_hash") or "").strip():
        return False
    signal = str(row.get("signal_summary") or "").strip().lower()
    return not signal or any(marker in signal for marker in ("文字0", "text0", "text 0", "0 text"))


def copy_candidate_image(data_dir: Path, out_dir: Path, row: dict[str, Any], kind: str) -> str:
    source = data_dir / "images" / str(row.get("filename") or "")
    if not source.is_file():
        return ""
    target = out_dir / kind / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target.relative_to(out_dir).as_posix()


def load_candidates(
    sqlite_path: Path,
    data_dir: Path,
    out_dir: Path,
    positive_limit: int,
    negative_limit: int,
    include_observation_positives: bool,
    observation_positive_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    positive_image_ids: set[str] = set()
    with sqlite3.connect(sqlite_path) as conn:
        conn.row_factory = sqlite3.Row
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if {"qa_events", "images"}.issubset(tables):
            query = """
                SELECT qa.id AS qa_event_id, qa.answer, qa.question,
                       images.id AS image_id, images.filename, images.session_id,
                       images.batch_id, images.kind, images.sequence_index, images.captured_at
                FROM qa_events qa
                JOIN images ON images.id = qa.image_id
                WHERE COALESCE(qa.status, '') = 'done'
                  AND COALESCE(qa.answer, '') != ''
                ORDER BY qa.created_at
            """
            for row in conn.execute(query):
                item = dict(row)
                if not answer_has_question(str(item.get("answer") or "")):
                    continue
                rel = copy_candidate_image(data_dir, out_dir, item, "positive")
                if not rel:
                    continue
                positives.append(
                    {
                        "image": rel,
                        "candidate_kind": "qa_recognized_question",
                        "needs_box_label": True,
                        "sqlite": str(sqlite_path),
                        "image_id": item.get("image_id") or "",
                        "session_id": item.get("session_id") or "",
                        "batch_id": item.get("batch_id") or "",
                        "qa_event_id": item.get("qa_event_id") or "",
                        "question": item.get("question") or "",
                        "answer_snippet": str(item.get("answer") or "")[:500],
                    }
                )
                positive_image_ids.add(str(item.get("image_id") or ""))
                if len(positives) >= positive_limit:
                    break
        if {"session_observations", "images"}.issubset(tables):
            if include_observation_positives:
                query = """
                    SELECT obs.*, images.id AS image_id, images.filename, images.session_id,
                           images.batch_id, images.kind, images.sequence_index, images.captured_at
                    FROM session_observations obs
                    JOIN images ON images.id = obs.image_id
                    WHERE COALESCE(obs.novelty_status, 'unknown') NOT IN ('invalid', 'duplicate')
                    ORDER BY obs.created_at
                """
                observation_added = 0
                for row in conn.execute(query):
                    item = dict(row)
                    image_id = str(item.get("image_id") or "")
                    if image_id in positive_image_ids:
                        continue
                    rel = copy_candidate_image(data_dir, out_dir, item, "positive")
                    if not rel:
                        continue
                    positives.append(
                        {
                            "image": rel,
                            "candidate_kind": "observation_image_unverified",
                            "needs_box_label": True,
                            "review_only": True,
                            "sqlite": str(sqlite_path),
                            "image_id": image_id,
                            "session_id": item.get("session_id") or "",
                            "batch_id": item.get("batch_id") or "",
                            "sequence_index": item.get("sequence_index") or "",
                            "captured_at": item.get("captured_at") or "",
                            "signal_summary": item.get("signal_summary") or "",
                            "text_hash": item.get("text_hash") or "",
                            "visual_hash": item.get("visual_hash") or "",
                            "note": "Unverified observation image candidate; draft boxes must be reviewed before training.",
                        }
                    )
                    positive_image_ids.add(image_id)
                    observation_added += 1
                    if observation_added >= observation_positive_limit:
                        break
            crop_join = ""
            crop_filter = ""
            if "session_question_crops" in tables:
                crop_join = """
                LEFT JOIN session_question_crops c
                  ON c.image_id = obs.image_id
                 AND COALESCE(c.status, 'ready') = 'ready'
                 AND COALESCE(c.crop_filename, '') != ''
                """
                crop_filter = "AND c.id IS NULL"
            query = """
                SELECT obs.*, images.filename, images.kind
                FROM session_observations obs
                JOIN images ON images.id = obs.image_id
                {crop_join}
                LEFT JOIN qa_events qa ON qa.image_id = obs.image_id
                WHERE qa.id IS NULL
                  {crop_filter}
                  AND COALESCE(obs.novelty_status, 'unknown') NOT IN ('invalid', 'duplicate')
                ORDER BY obs.created_at
            """.format(crop_join=crop_join, crop_filter=crop_filter) if "qa_events" in tables else """
                SELECT obs.*, images.filename, images.kind
                FROM session_observations obs
                JOIN images ON images.id = obs.image_id
                {crop_join}
                WHERE COALESCE(obs.novelty_status, 'unknown') NOT IN ('invalid', 'duplicate')
                  {crop_filter}
                ORDER BY obs.created_at
            """.format(crop_join=crop_join, crop_filter=crop_filter)
            for row in conn.execute(query):
                item = dict(row)
                if not safe_textless_negative(item):
                    continue
                rel = copy_candidate_image(data_dir, out_dir, item, "negative")
                if not rel:
                    continue
                negatives.append(
                    {
                        "image": rel,
                        "candidate_kind": "safe_textless_negative",
                        "sqlite": str(sqlite_path),
                        "image_id": item.get("image_id") or "",
                        "session_id": item.get("session_id") or "",
                        "batch_id": item.get("batch_id") or "",
                        "signal_summary": item.get("signal_summary") or "",
                    }
                )
                if len(negatives) >= negative_limit:
                    break
    return positives, negatives


def build_contact_sheet(out_dir: Path, rows: list[dict[str, Any]], name: str, limit: int = 60) -> None:
    tiles: list[Image.Image] = []
    for row in rows[:limit]:
        path = out_dir / str(row.get("image") or "")
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
        except Exception:
            continue
        scale = min(220 / image.width, 170 / image.height)
        thumb = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (240, 230), "white")
        tile.paste(thumb, ((240 - thumb.width) // 2, 8))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 184), str(row.get("candidate_kind") or "")[:34], fill=(0, 0, 0))
        draw.text((8, 204), Path(str(row.get("image") or "")).name[:34], fill=(0, 0, 0))
        tiles.append(tile)
    if not tiles:
        return
    cols = 4
    rows_count = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 240, rows_count * 230), (245, 245, 245))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % cols) * 240, (index // cols) * 230))
    sheet.save(out_dir / f"{name}_contact_sheet.jpg", quality=88)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create review candidate package from historical SQLite data.")
    parser.add_argument("--sqlite", type=Path, action="append", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-review-candidates"))
    parser.add_argument("--positive-limit", type=int, default=200)
    parser.add_argument("--negative-limit", type=int, default=200)
    parser.add_argument("--include-observation-positives", action="store_true", help="Add unverified session_observations as review-only positive candidates.")
    parser.add_argument("--observation-positive-limit", type=int, default=500, help="Maximum unverified observation positives per SQLite when enabled.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for sqlite_path in args.sqlite:
        pos, neg = load_candidates(
            sqlite_path,
            args.data_dir,
            args.out,
            args.positive_limit,
            args.negative_limit,
            args.include_observation_positives,
            args.observation_positive_limit,
        )
        positives.extend(pos)
        negatives.extend(neg)
    write_jsonl(args.out / "positive_candidates.jsonl", positives)
    write_jsonl(args.out / "negative_candidates.jsonl", negatives)
    build_contact_sheet(args.out, positives, "positive")
    build_contact_sheet(args.out, negatives, "negative")
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sqlite": [str(path) for path in args.sqlite],
        "data_dir": str(args.data_dir),
        "positive_candidates": len(positives),
        "negative_candidates": len(negatives),
        "include_observation_positives": bool(args.include_observation_positives),
        "observation_positive_limit": args.observation_positive_limit,
        "outputs": {
            "positive_candidates": "positive_candidates.jsonl",
            "negative_candidates": "negative_candidates.jsonl",
            "positive_contact_sheet": "positive_contact_sheet.jpg",
            "negative_contact_sheet": "negative_contact_sheet.jpg",
        },
    }
    write_json(args.out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
