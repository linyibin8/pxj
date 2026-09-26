"""Filter remote visual-reference question boxes into reviewable teacher candidates.

The Dell reference directory contains one JSON per image with candidate question
boxes. This script does not trust them as ground truth. It matches references to
local images, rejects obvious bad boxes, and writes a teacher-compatible JSONL
for visual review/packaging.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


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


def image_index(image_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            out[path.name] = path
            out[path.stem] = path
    return out


def normalize_bbox(raw: Any, width: int, height: int, sent_width: int, sent_height: int) -> dict[str, int] | None:
    if not isinstance(raw, list) or len(raw) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in raw[:4]]
    except (TypeError, ValueError):
        return None
    scale_x = width / max(1, sent_width)
    scale_y = height / max(1, sent_height)
    left = max(0.0, min(float(width), x1 * scale_x))
    top = max(0.0, min(float(height), y1 * scale_y))
    right = max(0.0, min(float(width), x2 * scale_x))
    bottom = max(0.0, min(float(height), y2 * scale_y))
    if right <= left or bottom <= top:
        return None
    box = {
        "x": int(round(left)),
        "y": int(round(top)),
        "width": int(round(right - left)),
        "height": int(round(bottom - top)),
    }
    if box["width"] <= 3 or box["height"] <= 3:
        return None
    return box


def box_flags(box: dict[str, int], width: int, height: int, raw_question: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    area = box["width"] * box["height"] / max(1, width * height)
    bw = box["width"] / max(1, width)
    bh = box["height"] / max(1, height)
    text = str(raw_question.get("text") or "")
    if area >= 0.45:
        flags.append("large_area")
    if area >= 0.62:
        flags.append("near_full_page")
    if bw >= 0.92:
        flags.append("very_wide")
    if bh >= 0.62:
        flags.append("very_tall")
    if area <= 0.0025:
        flags.append("too_small")
    if bw <= 0.06:
        flags.append("too_narrow")
    if box["width"] <= 90 and bh >= 0.08:
        flags.append("question_number_strip")
    if bh >= 0.09 and bw <= 0.12:
        flags.append("skinny_vertical_strip")
    if bw >= 0.10 and bh <= 0.018:
        flags.append("thin_horizontal_strip")
    if box["y"] + box["height"] > height * 0.995:
        flags.append("touches_bottom")
    if box["x"] + box["width"] > width * 0.995:
        flags.append("touches_right")
    if text.count("\n") >= 3:
        flags.append("multi_question_text")
    if "【练习" in text and text.count("\n") >= 1:
        flags.append("section_bundle")
    nbox = raw_question.get("nbox")
    if isinstance(nbox, list) and any(isinstance(v, (int, float)) and (v < -0.01 or v > 1.01) for v in nbox):
        flags.append("normalized_out_of_bounds")
    return flags


def should_accept(flags: list[str], args: argparse.Namespace) -> bool:
    blockers = {
        "near_full_page",
        "very_wide",
        "very_tall",
        "too_small",
        "multi_question_text",
        "section_bundle",
        "normalized_out_of_bounds",
        "too_narrow",
        "question_number_strip",
        "skinny_vertical_strip",
        "thin_horizontal_strip",
    }
    if args.allow_edge_touch:
        blockers.discard("touches_bottom")
        blockers.discard("touches_right")
    return not any(flag in blockers for flag in flags)


def draw_preview(image_path: Path, boxes: list[dict[str, Any]], target: Path) -> None:
    with Image.open(image_path) as image:
        canvas = ImageOps.exif_transpose(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", max(18, canvas.width // 120))
    except Exception:
        font = ImageFont.load_default()
    for index, box_row in enumerate(boxes, start=1):
        box = box_row["bbox_px"]
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        status = str(box_row.get("annotation_status") or "")
        color = (0, 180, 80) if status in {"verified", "approved", "accepted", "corrected"} else (255, 130, 0)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=max(3, canvas.width // 900))
        label = str(box_row.get("question_label") or index)
        draw.rectangle([x, max(0, y - 28), x + 86, y], fill=color)
        draw.text((x + 6, max(0, y - 24)), label, fill=(255, 255, 255), font=font)
    canvas.thumbnail((900, 900))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=90)


def make_contact_sheet(previews: list[Path], target: Path, columns: int = 3) -> None:
    if not previews:
        return
    thumbs = []
    for path in previews:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((420, 315))
            thumbs.append((path.name, thumb.copy()))
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 440, rows * 360), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, thumb) in enumerate(thumbs):
        x = (index % columns) * 440 + 10
        y = (index // columns) * 360 + 10
        sheet.paste(thumb, (x, y))
        draw.text((x, y + 320), name[:56], fill=(0, 0, 0))
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, quality=90)


def filter_references(args: argparse.Namespace) -> dict[str, Any]:
    idx = image_index(args.image_dir)
    rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    previews: list[Path] = []
    stats: Counter[str] = Counter()
    for ref_path in sorted(args.reference_dir.glob("*.json")):
        image_name = ref_path.name[:-5]
        image_path = idx.get(image_name) or idx.get(Path(image_name).stem)
        if image_path is None:
            stats["missing_image"] += 1
            continue
        try:
            payload = read_json(ref_path)
            with Image.open(image_path) as image:
                width, height = ImageOps.exif_transpose(image).size
        except Exception:
            stats["bad_reference_or_image"] += 1
            continue
        boxes: list[dict[str, Any]] = []
        rejected_boxes: list[dict[str, Any]] = []
        meta = payload.get("_meta") if isinstance(payload.get("_meta"), dict) else {}
        sent_width = int(meta.get("sent_w") or meta.get("w") or width)
        sent_height = int(meta.get("sent_h") or meta.get("h") or height)
        for q_index, question in enumerate(payload.get("questions") or [], start=1):
            if not isinstance(question, dict):
                continue
            box = normalize_bbox(question.get("bbox"), width, height, sent_width, sent_height)
            if box is None:
                stats["bad_box"] += 1
                continue
            flags = box_flags(box, width, height, question)
            accept = should_accept(flags, args)
            row_box = {
                "question_label": str(question.get("id") or q_index),
                "bbox_px": box,
                "annotation_status": args.accepted_status if accept else "needs_review",
                "score": 0.78 if accept else 0.35,
                "quality_flags": flags + ["dell_reference_candidate"],
                "teacher": "dell_reference_filtered",
                "teacher_note": str(question.get("text") or "")[:240],
            }
            if accept:
                boxes.append(row_box)
                stats["accepted_boxes"] += 1
            else:
                rejected_boxes.append(row_box)
                stats["rejected_boxes"] += 1
                for flag in flags:
                    stats[f"rejected_flag:{flag}"] += 1
        if boxes:
            rows.append(
                {
                    "review_id": f"dellref:{Path(image_name).stem}",
                    "image": str(image_path),
                    "teacher": "dell_reference_filtered",
                    "cohort": "dell_reference_filtered",
                    "boxes": boxes,
                }
            )
            stats["accepted_images"] += 1
            if len(previews) < args.preview_limit:
                preview_path = args.out / "previews" / f"{Path(image_name).stem}.jpg"
                draw_preview(image_path, boxes, preview_path)
                previews.append(preview_path)
        if rejected_boxes:
            rejected_rows.append(
                {
                    "review_id": f"dellref_reject:{Path(image_name).stem}",
                    "image": str(image_path),
                    "teacher": "dell_reference_filtered",
                    "cohort": "dell_reference_rejected",
                    "boxes": rejected_boxes,
                }
            )
        stats["references"] += 1

    write_jsonl(args.out / "teacher_candidate_boxes.jsonl", rows)
    write_jsonl(args.out / "teacher_rejected_boxes.jsonl", rejected_rows)
    make_contact_sheet(previews, args.out / "candidate_preview_contact_sheet.jpg")
    summary = {
        "reference_dir": str(args.reference_dir),
        "image_dir": str(args.image_dir),
        "out": str(args.out),
        "accepted_status": args.accepted_status,
        "stats": dict(sorted(stats.items())),
        "outputs": {
            "teacher_candidate_boxes": "teacher_candidate_boxes.jsonl",
            "teacher_rejected_boxes": "teacher_rejected_boxes.jsonl",
            "candidate_preview": "candidate_preview_contact_sheet.jpg",
        },
    }
    write_json(args.out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview-limit", type=int, default=60)
    parser.add_argument("--allow-edge-touch", action="store_true")
    parser.add_argument(
        "--accepted-status",
        default="candidate",
        help="status written for boxes that pass filters; keep as candidate until visual teacher approval",
    )
    args = parser.parse_args()
    print(json.dumps(filter_references(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
