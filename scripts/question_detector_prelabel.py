"""Create draft question-box annotations from image-level review candidates.

This is an active-learning helper, not a ground-truth generator. It uses
conservative document/text-layout heuristics to pre-draw boxes so historical
positive candidates can be reviewed faster in an annotation tool.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


CATEGORY_ID = 1
CATEGORY_NAME = "question_block"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class DraftBox:
    x: int
    y: int
    width: int
    height: int
    score: float
    source: str
    flags: list[str]

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
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


def maybe_clean(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        return np.array(image)
    except Exception:
        return None


def resize_for_processing(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    height, width = rgb.shape[:2]
    longest = max(width, height)
    if longest <= max_side:
        return rgb, 1.0
    scale = max_side / float(longest)
    resized = cv2.resize(rgb, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def clamp_box(x: int, y: int, width: int, height: int, image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x = max(0, min(image_width - 1, int(x)))
    y = max(0, min(image_height - 1, int(y)))
    width = max(1, min(image_width - x, int(width)))
    height = max(1, min(image_height - y, int(height)))
    return x, y, width, height


def detect_page_rect(rgb: np.ndarray) -> tuple[int, int, int, int, str]:
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    bright_cut = max(120, int(np.percentile(value, 63)))
    paper_mask = ((value >= bright_cut) & (saturation <= 95)).astype(np.uint8) * 255
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, width // 180), max(5, height // 180)))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(7, width // 120), max(7, height // 120)))
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    paper_mask = cv2.morphologyEx(paper_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    contours, _ = cv2.findContours(paper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    image_area = width * height
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < image_area * 0.08:
            continue
        if w < width * 0.18 or h < height * 0.18:
            continue
        aspect = w / max(1, h)
        if aspect > 2.15 or aspect < 0.22:
            continue
        if y + h < height * 0.38:
            continue
        fill = cv2.contourArea(contour) / max(1, area)
        center_bias = 1.0 - min(1.0, abs((x + w / 2) - width / 2) / max(1, width / 2))
        if 0.45 <= aspect <= 1.65:
            aspect_score = 1.0
        else:
            aspect_score = 0.45
        vertical_score = min(1.0, max(0.0, (y + h / 2) / max(1, height)))
        score = area / max(1, image_area) + fill * 0.35 + center_bias * 0.12 + aspect_score * 0.22 + vertical_score * 0.12
        candidates.append((score, (x, y, w, h)))
    if not candidates:
        pad_x = int(width * 0.04)
        pad_y = int(height * 0.04)
        return pad_x, pad_y, max(1, width - 2 * pad_x), max(1, height - 2 * pad_y), "full_image_fallback"
    _score, rect = max(candidates, key=lambda item: item[0])
    x, y, w, h = rect
    pad_x = int(w * 0.025)
    pad_y = int(h * 0.025)
    x, y, w, h = clamp_box(x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y, width, height)
    return x, y, w, h, "bright_page"


def dark_layout_mask(page_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    block = max(31, (min(page_rgb.shape[:2]) // 18) | 1)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 13)
    # Remove isolated sensor/noise dots, then reconnect characters into lines.
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    return binary


def text_line_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    height, width = mask.shape[:2]
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, width // 45), max(2, height // 220)))
    line_mask = cv2.dilate(mask, line_kernel, iterations=1)
    contours, _ = cv2.findContours(line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < max(18, width * height * 0.00005):
            continue
        if h < max(3, height * 0.003) or h > height * 0.32:
            continue
        if w < width * 0.025 and area < width * height * 0.0005:
            continue
        boxes.append((x, y, w, h))
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def percentile_value(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(0.0, min(1.0, pct)) * (len(ordered) - 1)
    lower = int(np.floor(rank))
    upper = int(np.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def split_dense_columns(
    line_boxes: list[tuple[int, int, int, int]],
    page_width: int,
    page_height: int,
) -> list[list[tuple[int, int, int, int]]]:
    if len(line_boxes) < 8:
        return [line_boxes]
    candidates = [
        box for box in line_boxes
        if box[2] >= page_width * 0.055 and box[3] <= page_height * 0.12
    ]
    if len(candidates) < 8:
        return [line_boxes]
    left_edge_candidates = [box for box in candidates if box[2] <= page_width * 0.58]
    if len(left_edge_candidates) < 8:
        return [line_boxes]
    lefts = sorted((x, (x, y, w, h)) for x, y, w, h in left_edge_candidates)
    split_gap = page_width * 0.105
    split_indexes = {
        index for index in range(len(lefts) - 1)
        if lefts[index + 1][0] - lefts[index][0] >= split_gap
    }
    if not split_indexes:
        return [line_boxes]
    clusters: list[list[tuple[int, int, int, int]]] = []
    current: list[tuple[int, int, int, int]] = []
    for index, (_left, box) in enumerate(lefts):
        current.append(box)
        if index in split_indexes:
            clusters.append(current)
            current = []
    if current:
        clusters.append(current)

    min_cluster_lines = max(4, int(len(candidates) * 0.16))
    strong_clusters = [cluster for cluster in clusters if len(cluster) >= min_cluster_lines]
    if len(strong_clusters) <= 1:
        return [line_boxes]

    anchors = sorted(float(median([x for x, _y, _w, _h in cluster])) for cluster in strong_clusters)
    if len(anchors) <= 1:
        return [line_boxes]
    boundaries = [0.0]
    for left, right in zip(anchors, anchors[1:]):
        boundaries.append((left + right) / 2)
    boundaries.append(float(page_width))

    bounds: list[tuple[float, float, list[tuple[int, int, int, int]]]] = []
    strong_by_anchor = sorted(zip(anchors, strong_clusters), key=lambda item: item[0])
    for index, (_anchor, cluster) in enumerate(strong_by_anchor):
        min_x = min(x for x, _y, _w, _h in cluster)
        max_x = max(x + w for x, _y, w, _h in cluster)
        pad = max(page_width * 0.025, (max_x - min_x) * 0.10)
        bounds.append((max(boundaries[index], min_x - pad), min(boundaries[index + 1], max_x + pad), []))
    column_count = len(bounds)
    max_line_width_ratio = 0.46 if column_count >= 3 else 0.64
    for box in line_boxes:
        x, _y, w, _h = box
        if w >= page_width * max_line_width_ratio:
            continue
        best_index = min(range(len(bounds)), key=lambda idx: abs(x - anchors[idx]))
        bounds[best_index][2].append(box)
    columns = [sorted(items, key=lambda box: (box[1], box[0])) for _min_x, _max_x, items in bounds if items]
    return columns or [line_boxes]


def merge_dense_line_boxes_into_questions(
    line_boxes: list[tuple[int, int, int, int]],
    page_width: int,
    page_height: int,
) -> list[DraftBox]:
    columns = split_dense_columns(line_boxes, page_width, page_height)
    if not columns:
        return []
    draft: list[DraftBox] = []
    multi_column = len(columns) > 1
    for column_index, column in enumerate(columns):
        if len(column) < 3:
            continue
        heights = [h for _x, _y, _w, h in column]
        typical_h = max(4.0, float(median(heights)))
        if multi_column:
            gap_threshold = max(page_height * 0.004, typical_h * 0.52)
        else:
            gap_threshold = max(page_height * 0.006, typical_h * 0.72)
        min_question_h = max(page_height * 0.026, typical_h * 2.2)
        groups: list[list[tuple[int, int, int, int]]] = []
        current: list[tuple[int, int, int, int]] = []
        current_bottom = 0
        for box in sorted(column, key=lambda item: (item[1], item[0])):
            x, y, w, h = box
            if not current:
                current = [box]
                current_bottom = y + h
                continue
            gap = y - current_bottom
            if gap > gap_threshold:
                groups.append(current)
                current = [box]
            else:
                current.append(box)
            current_bottom = max(current_bottom, y + h)
        if current:
            groups.append(current)

        column_lefts = [x for group in groups for x, _y, _w, _h in group]
        column_rights = [x + w for group in groups for x, _y, w, _h in group]
        if multi_column:
            column_min_x = percentile_value([float(value) for value in column_lefts], 0.22)
            column_max_x = percentile_value([float(value) for value in column_rights], 0.93)
            min_column_width = page_width * (0.27 if len(columns) >= 3 else 0.40)
            if column_max_x - column_min_x < min_column_width:
                column_max_x = min(float(page_width), column_min_x + min_column_width)
        else:
            column_min_x = min(column_lefts)
            column_max_x = max(column_rights)
        column_width = max(1, column_max_x - column_min_x)
        for index, group in enumerate(groups):
            min_x = min(x for x, _y, _w, _h in group)
            min_y = min(y for _x, y, _w, _h in group)
            max_x = max(x + w for x, _y, w, _h in group)
            max_y = max(y + h for _x, y, _w, h in group)
            if index + 1 < len(groups):
                next_top = min(y for _x, y, _w, _h in groups[index + 1])
                max_y = min(next_top - max(2, int(typical_h * 0.28)), max_y + max(int(typical_h * 1.8), 6))
            else:
                max_y = min(page_height, max_y + max(int(typical_h * 2.2), 8))

            if multi_column:
                pad_x = max(page_width * 0.012, typical_h * 1.0)
                target_min_x = max(0, column_min_x - pad_x)
                target_max_x = min(page_width, column_max_x + pad_x)
            else:
                pad_x = max(page_width * 0.04, typical_h * 1.5)
                target_min_x = min_x - pad_x
                target_max_x = max(max_x + pad_x, page_width * 0.90)
            pad_top = max(page_height * 0.004, typical_h * 0.35)
            min_y = max(0, int(min_y - pad_top))
            height = max(min_question_h, max_y - min_y)
            x, y, width, height_px = clamp_box(
                int(target_min_x),
                int(min_y),
                int(target_max_x - target_min_x),
                int(height),
                page_width,
                page_height,
            )
            area_ratio = width * height_px / max(1, page_width * page_height)
            if area_ratio > 0.42:
                continue
            flags: list[str] = []
            if len(group) <= 1:
                flags.append("few_text_lines")
            if multi_column:
                flags.append("dense_column_layout")
            else:
                flags.append("dense_single_column_layout")
            score = 0.42 + min(0.28, len(group) * 0.04) + min(0.16, area_ratio * 1.5)
            draft.append(
                DraftBox(
                    x,
                    y,
                    width,
                    height_px,
                    round(max(0.05, min(0.92, score)), 4),
                    "dense_layout_prelabel",
                    flags,
                )
            )
    return sorted(dedupe_boxes(draft, iou_threshold=0.70), key=lambda item: (item.y, item.x))


def merge_line_boxes_into_questions(
    line_boxes: list[tuple[int, int, int, int]],
    page_width: int,
    page_height: int,
) -> list[DraftBox]:
    if not line_boxes:
        return []
    heights = [h for _x, _y, _w, h in line_boxes]
    typical_h = max(4.0, float(median(heights)))
    gap_threshold = max(page_height * 0.028, typical_h * 2.4)
    min_question_h = max(page_height * 0.055, typical_h * 3.2)
    max_question_h = page_height * 0.58

    groups: list[list[tuple[int, int, int, int]]] = []
    current: list[tuple[int, int, int, int]] = []
    current_bottom = 0
    for box in line_boxes:
        x, y, w, h = box
        if not current:
            current = [box]
            current_bottom = y + h
            continue
        gap = y - current_bottom
        if gap > gap_threshold and len(current) >= 1:
            groups.append(current)
            current = [box]
        else:
            current.append(box)
        current_bottom = max(current_bottom, y + h)
    if current:
        groups.append(current)

    draft: list[DraftBox] = []
    for index, group in enumerate(groups):
        min_x = min(x for x, _y, _w, _h in group)
        min_y = min(y for _x, y, _w, _h in group)
        max_x = max(x + w for x, _y, w, _h in group)
        max_y = max(y + h for _x, y, _w, h in group)
        line_count = len(group)
        # If the block is very narrow, expand horizontally like production crops:
        # the reviewer can trim, but missing figure/answer area is harder to spot.
        pad_x = int(max(page_width * 0.035, typical_h * 1.8))
        pad_y_top = int(max(page_height * 0.008, typical_h * 0.8))
        pad_y_bottom = int(max(page_height * 0.045, typical_h * 3.0))
        if max_x - min_x < page_width * 0.55:
            center = (min_x + max_x) / 2
            span = max(max_x - min_x + 2 * pad_x, page_width * 0.72)
            min_x = int(center - span / 2)
            max_x = int(center + span / 2)
        if index + 1 < len(groups):
            next_top = min(y for _x, y, _w, _h in groups[index + 1])
            if next_top - max_y <= page_height * 0.16:
                max_y = min(max_y + pad_y_bottom, next_top - max(2, int(typical_h * 0.35)))
            else:
                max_y = max_y + pad_y_bottom
        else:
            max_y = max_y + pad_y_bottom
        min_x, min_y, width, height = clamp_box(
            min_x - pad_x,
            min_y - pad_y_top,
            max_x - min_x + 2 * pad_x,
            max_y - min_y + pad_y_top,
            page_width,
            page_height,
        )
        flags: list[str] = []
        if line_count <= 1:
            flags.append("few_text_lines")
        if height < min_question_h:
            flags.append("short_block")
            height = int(min(min_question_h, page_height - min_y))
        if height > max_question_h:
            flags.append("tall_block")
        area_ratio = width * height / max(1, page_width * page_height)
        if area_ratio > 0.75:
            flags.append("near_full_page")
        score = 0.35 + min(0.35, line_count * 0.035) + min(0.2, area_ratio)
        if flags:
            score -= 0.08 * len(flags)
        draft.append(DraftBox(min_x, min_y, width, height, round(max(0.05, min(0.95, score)), 4), "layout_prelabel", flags))
    return draft


def dedupe_boxes(boxes: list[DraftBox], iou_threshold: float = 0.82) -> list[DraftBox]:
    selected: list[DraftBox] = []
    for box in sorted(boxes, key=lambda item: item.score, reverse=True):
        if any(box_iou(box, existing) >= iou_threshold for existing in selected):
            continue
        selected.append(box)
    return sorted(selected, key=lambda item: (item.y, item.x))


def box_iou(a: DraftBox, b: DraftBox) -> float:
    ax2, ay2 = a.x + a.width, a.y + a.height
    bx2, by2 = b.x + b.width, b.y + b.height
    overlap_w = max(0, min(ax2, bx2) - max(a.x, b.x))
    overlap_h = max(0, min(ay2, by2) - max(a.y, b.y))
    overlap = overlap_w * overlap_h
    if overlap <= 0:
        return 0.0
    union = max(1, a.area + b.area - overlap)
    return overlap / union


def quarantine_reasons(
    page_source: str,
    page_rect: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    line_count: int,
    boxes: list[DraftBox],
) -> list[str]:
    x, y, width, height = page_rect
    page_aspect = width / max(1, height)
    page_bottom_ratio = (y + height) / max(1, image_height)
    page_area_ratio = width * height / max(1, image_width * image_height)
    reasons: list[str] = []
    dense_layout_boxes = bool(boxes) and all(box.source == "dense_layout_prelabel" for box in boxes)
    near_full_box_count = sum(1 for box in boxes if "near_full_page" in box.flags or box.area / max(1, width * height) > 0.42)
    if page_source == "full_image_fallback" and not (dense_layout_boxes and len(boxes) >= 3 and near_full_box_count == 0):
        reasons.append("page_rect_fallback")
    if not boxes:
        reasons.append("no_draft_boxes")
    if line_count < 3:
        reasons.append("too_few_text_lines")
    if page_aspect > 1.85 and page_bottom_ratio < 0.68:
        reasons.append("wide_top_band_likely_keyboard_or_table")
    edge_dominant = x <= image_width * 0.015 and y <= image_height * 0.08 and page_area_ratio > 0.62
    if edge_dominant and line_count > 95 and len(boxes) >= 3:
        reasons.append("edge_dominant_dense_layout")
    if len(boxes) >= 6 and not dense_layout_boxes:
        fragmented = sum(1 for box in boxes if "few_text_lines" in box.flags or "short_block" in box.flags)
        if fragmented / max(1, len(boxes)) >= 0.5:
            reasons.append("many_fragmented_short_blocks")
    return reasons


def draft_box_payload(box: DraftBox) -> dict[str, Any]:
    return {
        "bbox_px": {"x": box.x, "y": box.y, "width": box.width, "height": box.height},
        "score": box.score,
        "source": box.source,
        "quality_flags": box.flags,
    }


def draft_boxes_for_image(rgb: np.ndarray, max_side: int) -> tuple[list[DraftBox], dict[str, Any]]:
    small, scale = resize_for_processing(rgb, max_side)
    page_x, page_y, page_w, page_h, page_source = detect_page_rect(small)
    page = small[page_y : page_y + page_h, page_x : page_x + page_w]
    mask = dark_layout_mask(page)
    lines = text_line_boxes(mask)
    boxes = merge_line_boxes_into_questions(lines, page_w, page_h)
    dense_boxes = merge_dense_line_boxes_into_questions(lines, page_w, page_h)
    dense_near_full = sum(1 for box in dense_boxes if box.area / max(1, page_w * page_h) > 0.42)
    dense_layout_allowed = page_source == "full_image_fallback"
    if (
        dense_layout_allowed
        and len(dense_boxes) >= 3
        and dense_near_full == 0
        and len(dense_boxes) >= max(3, len(boxes))
    ):
        boxes = dense_boxes
    if not boxes and lines:
        min_x = min(x for x, _y, _w, _h in lines)
        min_y = min(y for _x, y, _w, _h in lines)
        max_x = max(x + w for x, _y, w, _h in lines)
        max_y = max(y + h for _x, y, _w, h in lines)
        x, y, w, h = clamp_box(min_x - 12, min_y - 12, max_x - min_x + 24, max_y - min_y + 48, page_w, page_h)
        boxes = [DraftBox(x, y, w, h, 0.28, "line_union_fallback", ["single_union_fallback"])]
    processing_boxes = dedupe_boxes(boxes)
    reasons = quarantine_reasons(
        page_source,
        (page_x, page_y, page_w, page_h),
        small.shape[1],
        small.shape[0],
        len(lines),
        processing_boxes,
    )
    mapped: list[DraftBox] = []
    for box in processing_boxes:
        x = int(round((page_x + box.x) / scale))
        y = int(round((page_y + box.y) / scale))
        w = int(round(box.width / scale))
        h = int(round(box.height / scale))
        x, y, w, h = clamp_box(x, y, w, h, rgb.shape[1], rgb.shape[0])
        mapped.append(DraftBox(x, y, w, h, box.score, box.source, [*box.flags, f"page:{page_source}"]))
    metadata = {
        "page_rect_processing": {"x": page_x, "y": page_y, "width": page_w, "height": page_h, "source": page_source},
        "processing_scale": scale,
        "line_box_count": len(lines),
        "dense_draft_box_count": len(dense_boxes),
        "draft_box_count": len(mapped),
        "quarantine_reasons": reasons,
    }
    if reasons:
        metadata["rejected_draft_boxes"] = [draft_box_payload(box) for box in mapped]
        return [], metadata
    return mapped, metadata


def copy_image(source: Path, out_root: Path, rel_name: str) -> Path:
    target = out_root / rel_name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def create_ml_annotation(box: DraftBox) -> dict[str, Any]:
    return {
        "label": CATEGORY_NAME,
        "coordinates": {
            "x": box.x + box.width / 2,
            "y": box.y + box.height / 2,
            "width": box.width,
            "height": box.height,
        },
    }


def label_studio_task(
    image_rel: str,
    boxes: list[DraftBox],
    image_width: int,
    image_height: int,
    model_version: str = "layout_prelabel_v1",
) -> dict[str, Any]:
    results = []
    for index, box in enumerate(boxes, start=1):
        results.append(
            {
                "id": f"draft-{index}",
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": box.x / image_width * 100,
                    "y": box.y / image_height * 100,
                    "width": box.width / image_width * 100,
                    "height": box.height / image_height * 100,
                    "rectanglelabels": [CATEGORY_NAME],
                },
                "score": box.score,
            }
        )
    return {
        "data": {"image": image_rel.replace("\\", "/")},
        "predictions": [{"model_version": model_version, "score": max([box.score for box in boxes], default=0), "result": results}],
    }


def build_preview_sheet(out_dir: Path, rows: list[dict[str, Any]], filename: str, limit: int = 80) -> None:
    tiles: list[Image.Image] = []
    colors = [(40, 180, 70), (220, 120, 40), (60, 120, 220), (180, 70, 180)]
    for row in rows[:limit]:
        path = out_dir / str(row.get("image") or "")
        if not path.is_file():
            continue
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        scale = min(260 / image.width, 190 / image.height)
        thumb = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(thumb)
        for index, ann in enumerate(row.get("boxes", []) or []):
            x = ann["bbox_px"]["x"] * scale
            y = ann["bbox_px"]["y"] * scale
            w = ann["bbox_px"]["width"] * scale
            h = ann["bbox_px"]["height"] * scale
            color = colors[index % len(colors)]
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            draw.text((x + 3, y + 3), str(index + 1), fill=color)
        tile = Image.new("RGB", (280, 235), "white")
        tile.paste(thumb, ((280 - thumb.width) // 2, 8))
        label = f"boxes={len(row.get('boxes', []) or [])} {Path(str(row.get('image') or '')).name[:26]}"
        ImageDraw.Draw(tile).text((8, 205), label, fill=(0, 0, 0))
        tiles.append(tile)
    if not tiles:
        return
    cols = 4
    rows_count = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 280, rows_count * 235), (245, 245, 245))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % cols) * 280, (index // cols) * 235))
    sheet.save(out_dir / filename, quality=88)


def prelabel(args: argparse.Namespace) -> dict[str, Any]:
    maybe_clean(args.out, args.clean)
    candidate_root = args.candidate_root
    rows = read_jsonl(candidate_root / args.positive_manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    draft_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []
    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    createml: list[dict[str, Any]] = []
    label_studio: list[dict[str, Any]] = []
    label_studio_quarantine: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    quarantine_counts: dict[str, int] = {}

    ann_id = 1
    image_id = 0
    for row in rows:
        rel = str(row.get("image") or "")
        source = candidate_root / rel
        if source.suffix.lower() not in IMAGE_SUFFIXES or not source.is_file():
            skipped["missing_image"] = skipped.get("missing_image", 0) + 1
            continue
        rgb = load_rgb(source)
        if rgb is None:
            skipped["unreadable_image"] = skipped.get("unreadable_image", 0) + 1
            continue
        export_rel = (Path("images") / Path(rel).name).as_posix()
        copy_image(source, args.out, export_rel)
        boxes, meta = draft_boxes_for_image(rgb, args.max_side)
        quarantine_reasons_for_image = list(meta.get("quarantine_reasons") or [])
        if quarantine_reasons_for_image:
            for reason in quarantine_reasons_for_image:
                quarantine_counts[reason] = quarantine_counts.get(reason, 0) + 1
            rejected_box_rows = list(meta.get("rejected_draft_boxes") or [])
            quarantine_rows.append(
                {
                    "image": export_rel,
                    "source_candidate": rel,
                    "candidate": row,
                    "width": int(rgb.shape[1]),
                    "height": int(rgb.shape[0]),
                    "boxes": rejected_box_rows,
                    "metadata": meta,
                    "annotation_status": "manual_page_review_required",
                    "quarantine_reasons": quarantine_reasons_for_image,
                }
            )
            label_studio_quarantine.append(
                {
                    **label_studio_task(export_rel, [], int(rgb.shape[1]), int(rgb.shape[0]), "layout_prelabel_quarantine_v1"),
                    "meta": {"quarantine_reasons": quarantine_reasons_for_image, "source_candidate": rel},
                }
            )
            continue
        if not boxes:
            skipped["no_draft_boxes"] = skipped.get("no_draft_boxes", 0) + 1
            continue
        image_id += 1
        image_record = {
            "id": image_id,
            "file_name": export_rel,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
            "source_candidate": rel,
            "candidate_kind": row.get("candidate_kind") or "",
            "source_image_id": row.get("image_id") or "",
            "session_id": row.get("session_id") or "",
            "batch_id": row.get("batch_id") or "",
            "label_count": len(boxes),
            "annotation_status": "draft_review_required",
        }
        coco_images.append(image_record)
        box_rows: list[dict[str, Any]] = []
        for box in boxes:
            box_payload = {**draft_box_payload(box), "annotation_status": "draft_review_required"}
            box_rows.append(box_payload)
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": CATEGORY_ID,
                    "bbox": [box.x, box.y, box.width, box.height],
                    "area": box.area,
                    "iscrowd": 0,
                    "score": box.score,
                    "source": box.source,
                    "quality_flags": box.flags,
                    "annotation_status": "draft_review_required",
                }
            )
            ann_id += 1
        createml.append({"image": export_rel, "annotations": [create_ml_annotation(box) for box in boxes]})
        label_studio.append(label_studio_task(export_rel, boxes, int(rgb.shape[1]), int(rgb.shape[0])))
        draft_rows.append(
            {
                "image": export_rel,
                "source_candidate": rel,
                "candidate": row,
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "boxes": box_rows,
                "metadata": meta,
            }
        )

    write_jsonl(args.out / "annotations" / "draft_boxes.jsonl", draft_rows)
    write_jsonl(args.out / "annotations" / "quarantine.jsonl", quarantine_rows)
    write_json(
        args.out / "annotations" / "coco_draft.json",
        {
            "info": {
                "description": "Draft prelabels for PXJ question detector review; not ground truth.",
                "version": "prelabel_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": str(candidate_root),
            },
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME}],
        },
    )
    write_json(args.out / "annotations" / "createml_draft.json", createml)
    write_json(args.out / "annotations" / "label_studio_tasks.json", label_studio)
    write_json(args.out / "annotations" / "label_studio_quarantine_tasks.json", label_studio_quarantine)
    build_preview_sheet(args.out, draft_rows, "draft_preview_contact_sheet.jpg")
    build_preview_sheet(args.out, quarantine_rows, "quarantine_preview_contact_sheet.jpg", limit=80)

    box_counts = [len(row.get("boxes", []) or []) for row in draft_rows]
    flagged = sum(1 for row in draft_rows for box in row.get("boxes", []) if box.get("quality_flags"))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_root": str(candidate_root),
        "input_candidates": len(rows),
        "images": len(draft_rows),
        "images_with_boxes": sum(1 for count in box_counts if count > 0),
        "quarantined_images": len(quarantine_rows),
        "quarantine_reasons": dict(sorted(quarantine_counts.items())),
        "draft_boxes": len(coco_annotations),
        "flagged_boxes": flagged,
        "mean_boxes_per_image": round(sum(box_counts) / max(1, len(box_counts)), 4),
        "median_boxes_per_image": median(box_counts) if box_counts else 0,
        "skipped": skipped,
        "status": "draft_review_required",
        "outputs": {
            "draft_jsonl": "annotations/draft_boxes.jsonl",
            "coco": "annotations/coco_draft.json",
            "createml": "annotations/createml_draft.json",
            "label_studio": "annotations/label_studio_tasks.json",
            "quarantine_jsonl": "annotations/quarantine.jsonl",
            "label_studio_quarantine": "annotations/label_studio_quarantine_tasks.json",
            "preview": "draft_preview_contact_sheet.jpg",
            "quarantine_preview": "quarantine_preview_contact_sheet.jpg",
        },
        "notes": [
            "Draft boxes are generated by layout heuristics and must be reviewed before detector training.",
            "Do not merge this output into question_detector_dataset.py without human approval of boxes.",
            "Quarantined images need manual page review before any box prelabels are trusted.",
        ],
    }
    write_json(args.out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft-prelabel question boxes from positive review candidates.")
    parser.add_argument("--candidate-root", type=Path, required=True, help="Output root from question_detector_review_candidates.py.")
    parser.add_argument("--positive-manifest", default="positive_candidates.jsonl")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-prelabels"))
    parser.add_argument("--max-side", type=int, default=1400)
    parser.add_argument("--limit", type=int, default=0, help="Optional max candidates for smoke runs; 0 means all.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    summary = prelabel(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
