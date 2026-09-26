"""Convert detector eval misses/false positives into a review queue.

The output intentionally matches the prelabel-review contract:

- images/
- annotations/draft_boxes.jsonl
- hard_examples_contact_sheet.jpg

It can be opened by question_detector_review_workbench.py, turning failed model
behavior into the next active-learning batch.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalize_path(value: str) -> Path:
    return Path(str(value).replace("\\", "/"))


def image_source(dataset_root: Path, file_name: str) -> Path:
    rel = normalize_path(file_name)
    if rel.is_absolute():
        return rel
    return dataset_root / rel


def clamp_box(raw: list[Any], width: int, height: int) -> dict[str, int] | None:
    if len(raw) < 4:
        return None
    try:
        x, y, w, h = (float(raw[0]), float(raw[1]), float(raw[2]), float(raw[3]))
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
    if out["width"] <= 3 or out["height"] <= 3:
        return None
    return out


def box_area_ratio(box: dict[str, int], width: int, height: int) -> float:
    return (box["width"] * box["height"]) / max(1.0, float(width * height))


def copy_image(source: Path, target: Path) -> tuple[int, int] | None:
    try:
        with Image.open(source) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            width, height = normalized.size
            target.parent.mkdir(parents=True, exist_ok=True)
            normalized.save(target, format="JPEG", quality=88, optimize=True)
            return width, height
    except Exception:
        return None


def select_problem_images(summary: dict[str, Any], threshold_key: str | None) -> tuple[str, dict[str, Any]]:
    thresholds = summary.get("thresholds") if isinstance(summary.get("thresholds"), dict) else {}
    if threshold_key is None:
        primary = summary.get("primary_threshold")
        threshold_key = f"{float(primary):.2f}" if primary is not None else next(iter(thresholds), "0.50")
    else:
        try:
            threshold_key = f"{float(threshold_key):.2f}"
        except (TypeError, ValueError):
            threshold_key = str(threshold_key)
    metrics = thresholds.get(threshold_key)
    if not isinstance(metrics, dict):
        raise SystemExit(f"threshold {threshold_key} not found in eval summary")
    problem_images = metrics.get("problem_images")
    if not isinstance(problem_images, dict):
        raise SystemExit("eval summary does not include threshold.problem_images")
    return threshold_key, problem_images


def score_for(kind: str, item: dict[str, Any]) -> float:
    if kind == "false_positive":
        try:
            return float(item.get("score") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 1.0


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = read_json(args.eval_summary)
    threshold_key, problem_images = select_problem_images(summary, args.threshold)
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    sorted_items = sorted(
        problem_images.items(),
        key=lambda item: (len(item[1].get("missed") or []) + len(item[1].get("false_positives") or [])),
        reverse=True,
    )
    for image_id, problem in sorted_items[: args.limit if args.limit else None]:
        file_name = str(problem.get("file_name") or "")
        source = image_source(args.dataset_root, file_name)
        if not source.is_file():
            stats["missing_image"] += 1
            continue
        target_rel = f"images/{Path(file_name).name}"
        copied_size = copy_image(source, args.out / target_rel)
        if copied_size is None:
            stats["bad_image"] += 1
            continue
        width, height = copied_size
        boxes: list[dict[str, Any]] = []
        for kind, items in (("missed", problem.get("missed") or []), ("false_positive", problem.get("false_positives") or [])):
            if kind == "missed" and not args.include_missed:
                continue
            if kind == "false_positive" and not args.include_false_positives:
                continue
            if kind == "false_positive":
                items = sorted(
                    [item for item in items if isinstance(item, dict)],
                    key=lambda item: score_for(kind, item),
                    reverse=True,
                )
                if args.max_false_positives_per_image > 0:
                    skipped = max(0, len(items) - args.max_false_positives_per_image)
                    if skipped:
                        stats["false_positive:per_image_cap_skipped"] += skipped
                    items = items[: args.max_false_positives_per_image]
            for index, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                if args.max_boxes_total > 0 and stats["boxes_total"] >= args.max_boxes_total:
                    stats["box_total_cap_skipped"] += 1
                    continue
                box = clamp_box(item.get("bbox") or [], width, height)
                if box is None:
                    stats[f"{kind}:bad_box"] += 1
                    continue
                area = box_area_ratio(box, width, height)
                if area < args.min_area_ratio or area > args.max_area_ratio:
                    stats[f"{kind}:area_filtered"] += 1
                    continue
                flags = [
                    "model_error",
                    f"error:{kind}",
                    f"eval_iou:{threshold_key}",
                ]
                if kind == "missed":
                    flags.append("review_gt_or_correct_box")
                    suggested_action = "approve_or_correct"
                    default_review_status = "pending"
                    training_export_default = True
                    requires_correction_for_training = False
                else:
                    flags.append("review_false_positive")
                    flags.append("default_reject_for_training")
                    suggested_action = "reject_or_correct"
                    default_review_status = "rejected"
                    training_export_default = False
                    requires_correction_for_training = True
                boxes.append(
                    {
                        "bbox_px": box,
                        "score": round(score_for(kind, item), 4),
                        "source": f"eval_{kind}",
                        "quality_flags": flags,
                        "annotation_status": "draft_review_required",
                        "suggested_review_action": suggested_action,
                        "default_review_status": default_review_status,
                        "training_export_default": training_export_default,
                        "requires_correction_for_training": requires_correction_for_training,
                        "error_kind": kind,
                        "eval_image_id": image_id,
                        "eval_index": index,
                        "best_iou": item.get("best_iou"),
                        "best_any_iou": item.get("best_any_iou"),
                        "question_key": item.get("question_key") or "",
                    }
                )
                stats[f"{kind}:boxes"] += 1
                stats["boxes_total"] += 1
        if not boxes:
            stats["images_without_boxes"] += 1
            continue
        rows.append(
            {
                "image": target_rel,
                "source_candidate": file_name,
                "candidate": {
                    "candidate_kind": "detector_eval_error",
                    "eval_summary": str(args.eval_summary),
                    "eval_image_id": image_id,
                    "threshold": threshold_key,
                    "is_negative": bool(problem.get("is_negative")),
                    "negative_kind": problem.get("negative_kind") or "",
                },
                "width": width,
                "height": height,
                "boxes": boxes,
                "metadata": {
                    "matches": problem.get("matches") or [],
                    "missed_count": len(problem.get("missed") or []),
                    "false_positive_count": len(problem.get("false_positives") or []),
                },
            }
        )
        stats["images"] += 1
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_summary": str(args.eval_summary),
        "dataset_root": str(args.dataset_root),
        "threshold": threshold_key,
        "stats": dict(stats),
        "outputs": {
            "draft_boxes": "annotations/draft_boxes.jsonl",
            "contact_sheet": "hard_examples_contact_sheet.jpg",
        },
    }
    return rows, report


def build_contact_sheet(out: Path, rows: list[dict[str, Any]], limit: int = 48) -> None:
    if not rows:
        return
    thumb_w, thumb_h, columns = 320, 220, 4
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
        preview.thumbnail((thumb_w, thumb_h - 24), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
        offset_x = (thumb_w - preview.width) // 2
        offset_y = 20
        canvas.paste(preview, (offset_x, offset_y))
        draw = ImageDraw.Draw(canvas)
        sx = preview.width / max(1, original_w)
        sy = preview.height / max(1, original_h)
        for box in row.get("boxes") or []:
            b = box["bbox_px"]
            color = (40, 180, 70) if box.get("error_kind") == "missed" else (220, 60, 60)
            x = offset_x + b["x"] * sx
            y = offset_y + b["y"] * sy
            draw.rectangle((x, y, x + b["width"] * sx, y + b["height"] * sy), outline=color, width=2)
        draw.text((6, 4), f"{row['metadata']['missed_count']} miss / {row['metadata']['false_positive_count']} fp", fill=(20, 20, 20))
        sheet.paste(canvas, ((index % columns) * thumb_w, (index // columns) * thumb_h))
    sheet.save(out / "hard_examples_contact_sheet.jpg", format="JPEG", quality=90, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create review queue from detector eval errors.")
    parser.add_argument("--eval-summary", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", default=None, help="Threshold key such as 0.50. Defaults to eval primary threshold.")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--no-missed", dest="include_missed", action="store_false", help="Do not include missed ground-truth boxes.")
    parser.add_argument("--no-false-positives", dest="include_false_positives", action="store_false", help="Do not include false-positive prediction boxes.")
    parser.add_argument("--min-area-ratio", type=float, default=0.0005)
    parser.add_argument("--max-area-ratio", type=float, default=0.98)
    parser.add_argument("--max-false-positives-per-image", type=int, default=8)
    parser.add_argument("--max-boxes-total", type=int, default=240)
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
