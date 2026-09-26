"""Package AI-teacher question boxes for detector training and iOS comparison.

The input JSONL is intentionally simple: one row per source image, with boxes
drawn by a visual teacher rather than by the weak OCR prelabeler. The output is
compatible with question_detector_dataset.py --reviewed-prelabels and with
question_segmentation_benchmark_eval.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_APPROVED_STATUSES = {"approved", "accepted", "corrected", "verified"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stable_id(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def safe_name(path: Path, index: int) -> str:
    stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in path.stem)[:80]
    return f"{index:04d}_{stable_id(str(path.resolve()))}_{stem}.jpg"


def resolve_image(raw: Any, base: Path) -> Path:
    text = str(raw or "").strip()
    if not text:
        raise SystemExit("teacher row is missing image")
    path = Path(text)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([base / path, base.parent / path])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(f"teacher image not found: {text}")


def normalize_box(raw: Any, width: int, height: int) -> dict[str, int] | None:
    if isinstance(raw, dict):
        try:
            x = float(raw.get("x"))
            y = float(raw.get("y"))
            w = float(raw.get("width"))
            h = float(raw.get("height"))
        except (TypeError, ValueError):
            return None
    elif isinstance(raw, list) and len(raw) >= 4:
        try:
            x, y, w, h = [float(value) for value in raw[:4]]
        except (TypeError, ValueError):
            return None
    else:
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
    return (box["width"] * box["height"]) / max(1, width * height)


def validate_box(box: dict[str, int], width: int, height: int) -> list[str]:
    flags: list[str] = []
    area = box_area_ratio(box, width, height)
    if area >= 0.55:
        flags.append("near_full_page")
    if box["width"] / max(1, width) >= 0.96:
        flags.append("too_wide")
    if box["height"] / max(1, height) >= 0.70:
        flags.append("too_tall")
    if area <= 0.003:
        flags.append("too_small")
    return flags


def copy_image(source: Path, target: Path) -> tuple[int, int]:
    with Image.open(source) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        width, height = normalized.size
        target.parent.mkdir(parents=True, exist_ok=True)
        normalized.save(target, format="JPEG", quality=90, optimize=True)
        return width, height


def draw_preview(image_path: Path, row: dict[str, Any], target: Path) -> None:
    with Image.open(image_path) as image:
        canvas = ImageOps.exif_transpose(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", max(22, canvas.width // 110))
    except Exception:
        font = ImageFont.load_default()
    for index, box in enumerate(row.get("boxes") or [], start=1):
        bbox = box.get("bbox_px") if isinstance(box, dict) else None
        if not isinstance(bbox, dict):
            continue
        x = int(bbox["x"])
        y = int(bbox["y"])
        w = int(bbox["width"])
        h = int(bbox["height"])
        status = str(box.get("annotation_status") or row.get("annotation_status") or "")
        color = (0, 180, 80) if status in DEFAULT_APPROVED_STATUSES else (255, 140, 0)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=max(4, canvas.width // 700))
        label = str(box.get("question_label") or box.get("question_index") or index)
        draw.rectangle([x, max(0, y - 34), x + 96, y], fill=color)
        draw.text((x + 8, max(0, y - 30)), label, fill=(255, 255, 255), font=font)
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.thumbnail((1600, 1600))
    canvas.save(target, quality=90)


def contact_sheet(previews: list[Path], target: Path, columns: int = 2, limit: int = 80) -> None:
    if not previews:
        return
    previews = previews[: max(1, limit)]
    thumbs: list[Image.Image] = []
    for path in previews:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((760, 570))
            thumbs.append(thumb.copy())
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 780, rows * 610), "white")
    for index, thumb in enumerate(thumbs):
        x = (index % columns) * 780 + 10
        y = (index // columns) * 610 + 10
        sheet.paste(thumb, (x, y))
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, quality=90)


def package_teacher_rows(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    approved_statuses = {
        item.strip().lower()
        for item in args.approved_statuses.split(",")
        if item.strip()
    } or DEFAULT_APPROVED_STATUSES

    stats: Counter[str] = Counter()
    approved_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    previews: list[Path] = []

    source_rows = read_jsonl(args.teacher_jsonl)
    for index, row in enumerate(source_rows, start=1):
        source = resolve_image(row.get("image") or row.get("image_path"), args.teacher_jsonl.parent)
        if source.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            stats["unsupported_image"] += 1
            continue
        target_name = safe_name(source, index)
        image_rel = f"images/{target_name}"
        target_path = args.out / image_rel
        width, height = copy_image(source, target_path)
        review_id = str(row.get("review_id") or f"teacher:{stable_id(str(source.resolve()))}").strip()

        approved_boxes: list[dict[str, Any]] = []
        all_boxes: list[dict[str, Any]] = []
        for box_index, box_row in enumerate(row.get("boxes") or [], start=1):
            if not isinstance(box_row, dict):
                stats["bad_box_row"] += 1
                continue
            bbox = normalize_box(box_row.get("bbox_px") or box_row.get("bbox"), width, height)
            if bbox is None:
                stats["bad_box"] += 1
                continue
            flags = list(dict.fromkeys([str(flag) for flag in box_row.get("quality_flags") or []]))
            flags.extend(flag for flag in validate_box(bbox, width, height) if flag not in flags)
            status = str(box_row.get("annotation_status") or row.get("annotation_status") or "verified").strip().lower()
            packaged_box = {
                **box_row,
                "bbox_px": bbox,
                "annotation_status": status,
                "score": float(box_row.get("score") or row.get("score") or 1.0),
                "quality_flags": flags,
                "teacher": str(box_row.get("teacher") or row.get("teacher") or args.teacher_name),
                "teacher_note": str(box_row.get("teacher_note") or ""),
                "box_index": box_index,
            }
            all_boxes.append(packaged_box)
            if status in approved_statuses and not any(flag in {"near_full_page", "too_small"} for flag in flags):
                approved_boxes.append(packaged_box)
                stats["approved_boxes"] += 1
            else:
                stats[f"excluded:{status or 'missing_status'}"] += 1

        packaged_row = {
            "review_id": review_id,
            "image": image_rel,
            "source_candidate": str(source),
            "width": width,
            "height": height,
            "annotation_status": "verified" if approved_boxes else "needs_review",
            "teacher": args.teacher_name,
            "boxes": all_boxes,
        }
        all_rows.append(packaged_row)
        if approved_boxes:
            approved_row = {**packaged_row, "boxes": approved_boxes, "annotation_status": "verified"}
            approved_rows.append(approved_row)
            benchmark_rows.append(
                {
                    "review_id": review_id,
                    "image": image_rel,
                    "image_path": str(target_path.resolve()),
                    "source_image": str(source),
                    "width": width,
                    "height": height,
                    "cohort": row.get("cohort") or "ai_teacher_seed",
                    "source_kind": "ai_teacher",
                    "box_count": len(approved_boxes),
                }
            )
            prediction_rows.append(
                {
                    "review_id": review_id,
                    "image": image_rel,
                    "width": width,
                    "height": height,
                    "boxes": [],
                }
            )
        if len(previews) < args.preview_limit:
            preview_path = args.out / "previews" / f"{Path(target_name).stem}_teacher.jpg"
            draw_preview(target_path, {**packaged_row, "boxes": approved_boxes or all_boxes}, preview_path)
            previews.append(preview_path)
        stats["images"] += 1
        if approved_boxes:
            stats["approved_images"] += 1

    write_jsonl(args.out / "annotations" / "approved_boxes.jsonl", approved_rows)
    write_jsonl(args.out / "annotations" / "teacher_all_boxes.jsonl", all_rows)
    write_jsonl(args.out / "teacher_reference_boxes.jsonl", approved_rows)
    write_jsonl(args.out / "benchmark_manifest.jsonl", benchmark_rows)
    write_jsonl(args.out / "ios_predictions_template.jsonl", prediction_rows)
    contact_sheet(previews, args.out / "teacher_preview_contact_sheet.jpg", limit=args.preview_limit)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "teacher_jsonl": str(args.teacher_jsonl),
        "out": str(args.out),
        "teacher": args.teacher_name,
        "approved_statuses": sorted(approved_statuses),
        "images": stats["images"],
        "approved_images": stats["approved_images"],
        "approved_boxes": stats["approved_boxes"],
        "excluded_boxes": sum(value for key, value in stats.items() if key.startswith("excluded:")),
        "stats": dict(sorted(stats.items())),
        "outputs": {
            "approved_boxes": "annotations/approved_boxes.jsonl",
            "teacher_all_boxes": "annotations/teacher_all_boxes.jsonl",
            "teacher_reference_boxes": "teacher_reference_boxes.jsonl",
            "benchmark_manifest": "benchmark_manifest.jsonl",
            "ios_predictions_template": "ios_predictions_template.jsonl",
            "preview": "teacher_preview_contact_sheet.jpg",
        },
        "notes": [
            "Only approved boxes are used for detector training and benchmark reference.",
            "Boxes with partial/uncertain statuses remain in teacher_all_boxes.jsonl for active learning but are not promoted.",
        ],
    }
    write_json(args.out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher-name", default="codex_visual_teacher")
    parser.add_argument("--approved-statuses", default="approved,accepted,corrected,verified")
    parser.add_argument("--preview-limit", type=int, default=80)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    print(json.dumps(package_teacher_rows(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
