"""Turn teacher candidate text boxes into reviewable full-question regions.

The Dell visual reference often marks the text/number area of a question, not
the full tappable question block. This script keeps those boxes as anchors,
groups them by page/column, and expands each anchor to the region between it
and the next anchor in reading order. The output is still draft/candidate data;
visual approval is required before it becomes training gold.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


APPROVED_STATUSES = {"approved", "accepted", "corrected", "verified"}
BLOCKED_REJECT_FLAGS = {
    "near_full_page",
    "very_wide",
    "very_tall",
    "too_small",
    "section_bundle",
    "normalized_out_of_bounds",
    "too_narrow",
    "question_number_strip",
    "skinny_vertical_strip",
    "thin_horizontal_strip",
}


@dataclass
class Anchor:
    label: str
    box: dict[str, int]
    source_status: str
    source_flags: list[str]
    source_note: str
    source_teacher: str
    source_score: float

    @property
    def x1(self) -> float:
        return float(self.box["x"])

    @property
    def y1(self) -> float:
        return float(self.box["y"])

    @property
    def x2(self) -> float:
        return float(self.box["x"] + self.box["width"])

    @property
    def y2(self) -> float:
        return float(self.box["y"] + self.box["height"])

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def clean_out(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def numeric_label(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"\d+", text)
    return match.group(0) if match else text


def box_from(raw: Any) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    try:
        x = int(round(float(raw.get("x"))))
        y = int(round(float(raw.get("y"))))
        w = int(round(float(raw.get("width"))))
        h = int(round(float(raw.get("height"))))
    except (TypeError, ValueError):
        return None
    if w <= 3 or h <= 3:
        return None
    return {"x": x, "y": y, "width": w, "height": h}


def clamp_box(box: dict[str, float], width: int, height: int) -> dict[str, int] | None:
    x1 = max(0.0, min(float(width), box["x"]))
    y1 = max(0.0, min(float(height), box["y"]))
    x2 = max(0.0, min(float(width), box["x"] + box["width"]))
    y2 = max(0.0, min(float(height), box["y"] + box["height"]))
    out = {
        "x": int(round(x1)),
        "y": int(round(y1)),
        "width": int(round(x2 - x1)),
        "height": int(round(y2 - y1)),
    }
    if out["width"] <= 3 or out["height"] <= 3:
        return None
    return out


def accept_rejected_anchor(box: dict[str, Any]) -> bool:
    flags = {str(flag) for flag in box.get("quality_flags") or []}
    if flags & BLOCKED_REJECT_FLAGS:
        return False
    return "multi_question_text" in flags or "touches_bottom" in flags or "touches_right" in flags


def anchors_for_row(row: dict[str, Any], rejected_by_image: dict[str, list[dict[str, Any]]]) -> list[Anchor]:
    image = str(row.get("image") or "")
    raw_boxes: list[dict[str, Any]] = []
    raw_boxes.extend(box for box in row.get("boxes") or [] if isinstance(box, dict))
    for box in rejected_by_image.get(image, []):
        if accept_rejected_anchor(box):
            raw_boxes.append(box)

    anchors: list[Anchor] = []
    seen: set[tuple[str, int, int]] = set()
    for raw in raw_boxes:
        box = box_from(raw.get("bbox_px") or raw.get("bbox"))
        if box is None:
            continue
        label = numeric_label(raw.get("question_label") or raw.get("question_index"))
        if not label:
            continue
        key = (label, round(box["x"] / 12), round(box["y"] / 12))
        if key in seen:
            continue
        seen.add(key)
        anchors.append(
            Anchor(
                label=label,
                box=box,
                source_status=str(raw.get("annotation_status") or ""),
                source_flags=[str(flag) for flag in raw.get("quality_flags") or []],
                source_note=str(raw.get("teacher_note") or raw.get("note") or ""),
                source_teacher=str(raw.get("teacher") or row.get("teacher") or ""),
                source_score=float(raw.get("score") or 0.0),
            )
        )
    return anchors


def split_columns(anchors: list[Anchor], width: int) -> list[list[Anchor]]:
    if len(anchors) < 6:
        return [anchors]
    centers = sorted(anchor.cx for anchor in anchors)
    best_gap = 0.0
    best_index: int | None = None
    for idx in range(len(centers) - 1):
        left_count = idx + 1
        right_count = len(centers) - left_count
        if left_count < 2 or right_count < 2:
            continue
        gap = centers[idx + 1] - centers[idx]
        if gap > best_gap:
            best_gap = gap
            best_index = idx
    if best_index is None:
        return [anchors]
    split_x = (centers[best_index] + centers[best_index + 1]) / 2
    left = [anchor for anchor in anchors if anchor.cx < split_x]
    right = [anchor for anchor in anchors if anchor.cx >= split_x]
    if len(left) < 2 or len(right) < 2:
        return [anchors]
    min_gap = max(width * 0.12, 140)
    if best_gap < min_gap:
        return [anchors]
    if split_x < width * 0.22 or split_x > width * 0.78:
        return [anchors]
    return [left, right]


def content_span(anchors: list[Anchor], width: int, margin_ratio: float) -> tuple[float, float]:
    if not anchors:
        return 0.0, float(width)
    min_x = min(anchor.x1 for anchor in anchors)
    max_x = max(anchor.x2 for anchor in anchors)
    span = max_x - min_x
    pad = max(28.0, min(width * 0.08, span * margin_ratio))
    min_x = max(0.0, min_x - pad)
    max_x = min(float(width), max_x + pad)
    min_width = min(float(width), max(width * 0.34, span * 1.75))
    if max_x - min_x < min_width:
        center = (min_x + max_x) / 2
        min_x = center - min_width / 2
        max_x = center + min_width / 2
        if min_x < 0:
            max_x -= min_x
            min_x = 0
        if max_x > width:
            min_x -= max_x - width
            max_x = float(width)
    return max(0.0, min_x), min(float(width), max_x)


def group_column_span(
    group: list[Anchor],
    all_groups: list[list[Anchor]],
    width: int,
    margin_ratio: float,
) -> tuple[float, float]:
    if len(all_groups) == 1:
        return content_span(group, width, max(margin_ratio, 0.90))

    centers = [sum(anchor.cx for anchor in item) / max(1, len(item)) for item in all_groups]
    ordered = sorted(zip(centers, all_groups), key=lambda pair: pair[0])
    group_center = sum(anchor.cx for anchor in group) / max(1, len(group))
    index = next((idx for idx, (_, item) in enumerate(ordered) if item is group), 0)
    if index == 0:
        left = 0.0
        right = (ordered[index][0] + ordered[index + 1][0]) / 2 if len(ordered) > 1 else float(width)
    elif index == len(ordered) - 1:
        left = (ordered[index - 1][0] + ordered[index][0]) / 2
        right = float(width)
    else:
        left = (ordered[index - 1][0] + ordered[index][0]) / 2
        right = (ordered[index][0] + ordered[index + 1][0]) / 2

    anchor_min = min(anchor.x1 for anchor in group)
    anchor_max = max(anchor.x2 for anchor in group)
    pad = max(36.0, min(width * 0.08, (anchor_max - anchor_min) * margin_ratio))
    left = min(left, anchor_min - pad)
    right = max(right, anchor_max + pad)
    min_width = max(width * 0.30, (anchor_max - anchor_min) * 1.55)
    if right - left < min_width:
        left = group_center - min_width / 2
        right = group_center + min_width / 2
    return max(0.0, left), min(float(width), right)


def regionize_group(
    anchors: list[Anchor],
    width: int,
    height: int,
    *,
    x_span: tuple[float, float],
    y_pad_ratio: float,
) -> list[dict[str, Any]]:
    ordered = sorted(anchors, key=lambda item: (item.y1, item.x1))
    if not ordered:
        return []
    median_h = sorted(anchor.box["height"] for anchor in ordered)[len(ordered) // 2]
    x1, x2 = x_span
    rows: list[dict[str, Any]] = []
    for index, anchor in enumerate(ordered):
        next_anchor = ordered[index + 1] if index + 1 < len(ordered) else None
        top_pad = max(10.0, min(44.0, median_h * 0.32))
        y1 = max(0.0, anchor.y1 - top_pad)
        if next_anchor is not None:
            gap = next_anchor.y1 - anchor.y2
            boundary = anchor.y2 + max(0.0, gap * y_pad_ratio)
            y2 = min(float(height), max(anchor.y2 + median_h * 0.35, boundary))
        else:
            terminal_extra = max(median_h * 1.8, (x2 - x1) * 0.20, height * 0.08)
            y2 = min(float(height), anchor.y2 + terminal_extra)
        min_height = max(42.0, median_h * 1.25)
        if y2 - y1 < min_height:
            y2 = min(float(height), y1 + min_height)
        box = clamp_box({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}, width, height)
        if box is None:
            continue
        area = box["width"] * box["height"] / max(1, width * height)
        flags = list(dict.fromkeys(anchor.source_flags + ["regionized_question_candidate"]))
        if area >= 0.42:
            flags.append("large_region_review_required")
        if next_anchor is None:
            flags.append("terminal_region_estimated")
        rows.append(
            {
                "question_label": anchor.label,
                "bbox_px": box,
                "annotation_status": "candidate",
                "default_review_status": "pending",
                "score": round(max(0.35, min(0.86, anchor.source_score or 0.66)), 3),
                "quality_flags": flags,
                "teacher": "codex_regionized_from_dell_reference",
                "teacher_note": (
                    f"Regionized from {anchor.source_teacher or 'candidate'} anchor "
                    f"{anchor.box}; source_status={anchor.source_status}; {anchor.source_note[:180]}"
                ),
            }
        )
    return rows


def regionize_row(row: dict[str, Any], rejected_by_image: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> dict[str, Any] | None:
    image_path = Path(str(row.get("image") or ""))
    if not image_path.is_file():
        return None
    with Image.open(image_path) as image:
        width, height = ImageOps.exif_transpose(image).size
    anchors = anchors_for_row(row, rejected_by_image)
    if not anchors:
        return None
    output_boxes: list[dict[str, Any]] = []
    groups = split_columns(anchors, width)
    for group in groups:
        x_span = group_column_span(group, groups, width, args.x_margin_ratio)
        output_boxes.extend(
            regionize_group(
                group,
                width,
                height,
                x_span=x_span,
                y_pad_ratio=args.y_pad_ratio,
            )
        )
    output_boxes.sort(key=lambda box: (box["bbox_px"]["y"], box["bbox_px"]["x"]))
    if args.max_boxes_per_image > 0 and len(output_boxes) > args.max_boxes_per_image:
        return None
    return {
        "review_id": f"regionized:{Path(str(row.get('image') or '')).stem}",
        "image": str(image_path),
        "source_candidate": str(image_path),
        "width": width,
        "height": height,
        "teacher": "codex_regionized_from_dell_reference",
        "cohort": "regionized_dell_candidate",
        "annotation_status": "draft_review_required",
        "boxes": output_boxes,
        "metadata": {
            "source_review_id": row.get("review_id"),
            "source_teacher": row.get("teacher"),
            "anchor_count": len(anchors),
        },
    }


def draw_preview(row: dict[str, Any], target: Path) -> None:
    with Image.open(row["image"]) as image:
        canvas = ImageOps.exif_transpose(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", max(18, canvas.width // 120))
    except Exception:
        font = ImageFont.load_default()
    for index, box_row in enumerate(row.get("boxes") or [], start=1):
        box = box_row["bbox_px"]
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        status = str(box_row.get("annotation_status") or "")
        color = (0, 180, 80) if status in APPROVED_STATUSES else (255, 140, 0)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=max(3, canvas.width // 800))
        label = str(box_row.get("question_label") or index)
        draw.rectangle([x, max(0, y - 34), x + 100, y], fill=color)
        draw.text((x + 7, max(0, y - 30)), label, fill=(255, 255, 255), font=font)
    canvas.thumbnail((1500, 1500))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=90)


def contact_sheet(previews: list[Path], target: Path, limit: int, columns: int = 2) -> None:
    if not previews:
        return
    thumbs: list[Image.Image] = []
    names: list[str] = []
    for path in previews[:limit]:
        with Image.open(path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((760, 570))
            thumbs.append(thumb.copy())
            names.append(path.name[:76])
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * 780, rows * 610), "white")
    draw = ImageDraw.Draw(sheet)
    for index, thumb in enumerate(thumbs):
        x = (index % columns) * 780 + 10
        y = (index // columns) * 610 + 10
        sheet.paste(thumb, (x, y))
        draw.text((x, y + 575), names[index], fill=(0, 0, 0))
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, quality=90)


def copy_images_and_write_review_format(out: Path, rows: list[dict[str, Any]]) -> None:
    annotations = out / "annotations"
    images = out / "images"
    review_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        source = Path(row["image"])
        target_name = f"{index:04d}_{source.name}"
        target = images / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        review_rows.append({**row, "image": f"images/{target_name}", "source_candidate": str(source)})
    write_jsonl(annotations / "draft_boxes.jsonl", review_rows)
    write_jsonl(out / "candidate_regions.jsonl", rows)


def regionize(args: argparse.Namespace) -> dict[str, Any]:
    clean_out(args.out, args.clean)
    candidate_rows = read_jsonl(args.candidates)
    rejected_rows = read_jsonl(args.rejected)
    rejected_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rejected_rows:
        image = str(row.get("image") or "")
        rejected_by_image[image].extend(box for box in row.get("boxes") or [] if isinstance(box, dict))

    stats: Counter[str] = Counter()
    output_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        if args.skip_legacy and "legacy" in Path(str(row.get("image") or "")).name:
            stats["skipped_legacy"] += 1
            continue
        out_row = regionize_row(row, rejected_by_image, args)
        if out_row is None:
            stats["skipped_unregionizable"] += 1
            continue
        output_rows.append(out_row)
        stats["images"] += 1
        stats["boxes"] += len(out_row.get("boxes") or [])
    if args.limit > 0:
        output_rows = output_rows[: args.limit]

    copy_images_and_write_review_format(args.out, output_rows)

    previews: list[Path] = []
    for row in output_rows[: args.preview_limit]:
        preview_path = args.out / "previews" / f"{Path(row['image']).stem}.jpg"
        draw_preview(row, preview_path)
        previews.append(preview_path)
    contact_sheet(previews, args.out / "regionized_candidate_preview.jpg", args.preview_limit)

    summary = {
        "candidates": str(args.candidates),
        "rejected": str(args.rejected),
        "out": str(args.out),
        "stats": dict(sorted(stats.items())),
        "images": len(output_rows),
        "boxes": sum(len(row.get("boxes") or []) for row in output_rows),
        "outputs": {
            "candidate_regions": "candidate_regions.jsonl",
            "draft_boxes": "annotations/draft_boxes.jsonl",
            "preview": "regionized_candidate_preview.jpg",
        },
        "notes": [
            "Output boxes are candidate full-question regions, not gold labels.",
            "Use the review workbench or visual teacher approval before packaging as training data.",
        ],
    }
    write_json(args.out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview-limit", type=int, default=48)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-boxes-per-image", type=int, default=18)
    parser.add_argument("--y-pad-ratio", type=float, default=0.52)
    parser.add_argument("--x-margin-ratio", type=float, default=0.60)
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    print(json.dumps(regionize(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
