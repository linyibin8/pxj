"""Turn question segmentation benchmark errors into a review queue.

Input is an output directory from question_segmentation_benchmark_eval.py. The
result follows the same prelabel-style contract as other detector review queues:
images/, annotations/draft_boxes.jsonl, summary.json, and a contact sheet.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line_number, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise SystemExit(f"{path}:{line_number} is not a JSON object")
            rows.append(item)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def clamp_box(raw: Any, width: int, height: int) -> dict[str, int] | None:
    if isinstance(raw, dict):
        values = [raw.get("x"), raw.get("y"), raw.get("width"), raw.get("height")]
    elif isinstance(raw, list) and len(raw) >= 4:
        values = raw[:4]
    else:
        return None
    try:
        x, y, w, h = (float(values[0]), float(values[1]), float(values[2]), float(values[3]))
    except (TypeError, ValueError):
        return None
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    out = {
        "x": int(round(x1)),
        "y": int(round(y1)),
        "width": int(round(x2 - x1)),
        "height": int(round(y2 - y1)),
    }
    if out["width"] <= 2 or out["height"] <= 2:
        return None
    return out


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)[:90] or "image"


def load_benchmark(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        review_id = str(row.get("review_id") or "").strip()
        if review_id:
            rows[review_id] = row
    return rows


def resolve_eval_path(eval_root: Path, raw: str) -> Path:
    path = Path(str(raw or ""))
    if path.is_file():
        return path
    candidate = eval_root / path
    if candidate.is_file():
        return candidate
    candidate = eval_root.parent / path
    if candidate.is_file():
        return candidate
    return path


def copy_image(row: dict[str, Any], out: Path, index: int) -> tuple[str, int, int] | None:
    source = Path(str(row.get("image_path") or ""))
    if not source.is_file():
        return None
    suffix = source.suffix.lower() or ".jpg"
    target_rel = f"images/{index:03d}_{safe_name(Path(str(row.get('image') or source.name)).stem)}{suffix}"
    target = out / target_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    with Image.open(target) as image:
        normalized = ImageOps.exif_transpose(image)
        width, height = normalized.size
    return target_rel, width, height


def error_box(error: dict[str, Any], error_kind: str, width: int, height: int, label: str) -> dict[str, Any] | None:
    raw_bbox = error.get("bbox") or error.get("reference_bbox") or error.get("prediction_bbox")
    bbox = clamp_box(raw_bbox, width, height)
    if bbox is None:
        return None
    if error_kind == "missed":
        flags = ["benchmark_error", "error:missed_reference", "review_gt_or_correct_box"]
        default_status = "pending"
        training_export_default = True
        requires_correction = False
        suggested_action = "approve_or_correct"
        source = "benchmark_missed_reference"
        score = 1.0
    else:
        flags = ["benchmark_error", "error:false_positive_prediction", "default_reject_for_training"]
        default_status = "rejected"
        training_export_default = False
        requires_correction = True
        suggested_action = "reject_or_correct"
        source = "benchmark_false_positive_prediction"
        score = float(error.get("score") or 0.0)
    return {
        "bbox_px": bbox,
        "score": round(score, 4),
        "source": source,
        "quality_flags": flags,
        "annotation_status": "draft_review_required",
        "suggested_review_action": suggested_action,
        "default_review_status": default_status,
        "training_export_default": training_export_default,
        "requires_correction_for_training": requires_correction,
        "error_kind": error_kind,
        "benchmark_label": label,
        "benchmark_iou": error.get("iou"),
        "best_any_iou": error.get("best_any_iou"),
        "reference_index": error.get("reference_index"),
        "prediction_index": error.get("prediction_index"),
    }


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = read_json(args.eval / "summary.json")
    benchmark_path = resolve_eval_path(args.eval, str(summary.get("benchmark_manifest") or ""))
    benchmark = load_benchmark(benchmark_path)
    missed = read_jsonl(args.eval / "missed_reference.jsonl")
    false_positives = read_jsonl(args.eval / "false_positive_predictions.jsonl")
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"missed": [], "false_positive": []})
    for row in missed:
        grouped[str(row.get("review_id") or "")]["missed"].append(row)
    for row in false_positives:
        grouped[str(row.get("review_id") or "")]["false_positive"].append(row)

    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for image_index, (review_id, errors) in enumerate(sorted(grouped.items()), start=1):
        bench = benchmark.get(review_id)
        if not bench:
            stats["missing_benchmark_row"] += 1
            continue
        copied = copy_image(bench, args.out, image_index)
        if copied is None:
            stats["missing_image"] += 1
            continue
        image_rel, width, height = copied
        boxes: list[dict[str, Any]] = []
        for kind, items in errors.items():
            if kind == "missed" and not args.include_missed:
                continue
            if kind == "false_positive" and not args.include_false_positives:
                continue
            for item in items:
                box = error_box(item, "missed" if kind == "missed" else "false_positive", width, height, str(summary.get("label") or "benchmark"))
                if box is None:
                    stats[f"{kind}:bad_box"] += 1
                    continue
                boxes.append(box)
                stats[f"{kind}:boxes"] += 1
        if not boxes:
            stats["images_without_boxes"] += 1
            continue
        rows.append(
            {
                "image": image_rel,
                "source_candidate": bench.get("image") or bench.get("source_image_path") or "",
                "width": width,
                "height": height,
                "boxes": boxes,
                "annotation_status": "draft_review_required",
                "review_id": review_id,
                "metadata": {
                    "benchmark_eval": str(args.eval),
                    "benchmark_label": summary.get("label"),
                    "benchmark_manifest": summary.get("benchmark_manifest"),
                    "cohort": bench.get("cohort"),
                    "source_kind": bench.get("source_kind"),
                    "source_review_id": review_id,
                    "missed_count": len(errors["missed"]),
                    "false_positive_count": len(errors["false_positive"]),
                },
            }
        )
        stats["images"] += 1
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval": str(args.eval),
        "benchmark_manifest": str(benchmark_path),
        "label": summary.get("label"),
        "stats": dict(stats),
        "outputs": {
            "draft_boxes": "annotations/draft_boxes.jsonl",
            "contact_sheet": "benchmark_errors_contact_sheet.jpg",
        },
        "notes": [
            "Missed reference boxes default to pending/approve-or-correct.",
            "False-positive prediction boxes default to rejected and only enter training if corrected or verified.",
        ],
    }
    return rows, report


def build_contact_sheet(out: Path, rows: list[dict[str, Any]], limit: int = 48) -> None:
    if not rows:
        return
    thumb_w, thumb_h, columns = 320, 230, 4
    selected = rows[:limit]
    sheet = Image.new("RGB", (columns * thumb_w, math.ceil(len(selected) / columns) * thumb_h), "white")
    for index, row in enumerate(selected):
        path = out / str(row["image"])
        try:
            with Image.open(path) as image:
                preview = ImageOps.exif_transpose(image).convert("RGB")
        except Exception:
            continue
        original_w, original_h = preview.size
        preview.thumbnail((thumb_w, thumb_h - 26), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
        offset_x = (thumb_w - preview.width) // 2
        offset_y = 24
        canvas.paste(preview, (offset_x, offset_y))
        draw = ImageDraw.Draw(canvas)
        sx = preview.width / max(1, original_w)
        sy = preview.height / max(1, original_h)
        for box in row.get("boxes") or []:
            b = box["bbox_px"]
            color = (30, 140, 70) if box.get("error_kind") == "missed" else (215, 60, 55)
            x = offset_x + b["x"] * sx
            y = offset_y + b["y"] * sy
            draw.rectangle((x, y, x + b["width"] * sx, y + b["height"] * sy), outline=color, width=2)
        meta = row.get("metadata") or {}
        draw.text((6, 5), f"{meta.get('missed_count', 0)} miss / {meta.get('false_positive_count', 0)} fp", fill=(20, 20, 20))
        sheet.paste(canvas, ((index % columns) * thumb_w, (index // columns) * thumb_h))
    sheet.save(out / "benchmark_errors_contact_sheet.jpg", format="JPEG", quality=90, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a review queue from question segmentation benchmark errors.")
    parser.add_argument("--eval", type=Path, required=True, help="Output root from question_segmentation_benchmark_eval.py.")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-segmentation-benchmark-errors"))
    parser.add_argument("--no-missed", dest="include_missed", action="store_false")
    parser.add_argument("--no-false-positives", dest="include_false_positives", action="store_false")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, report = build_rows(args)
    write_jsonl(args.out / "annotations" / "draft_boxes.jsonl", rows)
    write_json(args.out / "summary.json", report)
    build_contact_sheet(args.out, rows)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "images": len(rows),
                "boxes": sum(len(row.get("boxes") or []) for row in rows),
                "stats": report["stats"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
