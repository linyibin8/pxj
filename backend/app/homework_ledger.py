import base64
import html
import json
import re
import shutil
import uuid
from io import BytesIO
from pathlib import Path
from typing import Callable
from urllib.parse import quote, unquote

from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - optional production acceleration
    cv2 = None
    np = None


HOMEWORK_LEDGER_ASSET_ROOT = "homework_ledger"
HOMEWORK_LEDGER_HTML_LIMIT = 240_000


def _json_object(value: object) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return dict(parsed)
    return {}


def _json_array(value: object) -> list:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return []
            if isinstance(parsed, list):
                return list(parsed)
    return []


def _meta_text(item: dict, *keys: str) -> str:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _truncate(value: object, limit: int) -> str:
    text = str(value or "").strip()
    return text[:limit]


def _int_value(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _manifest_list(manifest: dict, *keys: str) -> list[dict]:
    for key in keys:
        value = manifest.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _parse_manifest(raw: str) -> dict:
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "invalid manifest json") from exc
    if not isinstance(data, dict):
        raise HTTPException(400, "manifest must be an object")
    return data


def _file_map(uploads: list[UploadFile] | None) -> dict[str, UploadFile]:
    mapped: dict[str, UploadFile] = {}
    for index, upload in enumerate(uploads or []):
        original = str(upload.filename or "").replace("\\", "/").strip().lstrip("/")
        if original:
            mapped.setdefault(original, upload)
            mapped.setdefault(Path(original).name, upload)
        mapped.setdefault(str(index), upload)
    return mapped


def _upload_ref(item: dict, *keys: str, fallback: str = "") -> str:
    for key in keys:
        value = str(item.get(key) or "").replace("\\", "/").strip().lstrip("/")
        if value:
            return value
    return fallback


def _safe_asset_rel(rel_path: str, default_folder: str, fallback_name: str) -> str:
    clean = str(rel_path or "").replace("\\", "/").strip().lstrip("/")
    parts = [part for part in clean.split("/") if part and part not in {".", ".."}]
    if not parts:
        parts = [fallback_name]
    if parts[0] == "server_refined" and len(parts) >= 3 and parts[1] in {"pages", "crops"}:
        parts = parts[:3]
    elif default_folder in parts:
        start = len(parts) - 1 - list(reversed(parts)).index(default_folder)
        parts = parts[start:]
    elif default_folder == "frames" and "normalized_frames" in parts:
        parts = ["frames", parts[-1]]
    elif len(parts) == 1 or parts[0] not in {"frames", "crops", "server_refined"}:
        parts = [default_folder, parts[-1]]
    return "/".join(parts[:3])


def _frame_id(item: dict, fallback_index: int) -> str:
    return _truncate(_meta_text(item, "id", "frame_id", "frameId") or f"frame_{fallback_index:04d}", 80)


def _evidence_id(item: dict, fallback_index: int) -> str:
    return _truncate(_meta_text(item, "id", "evidence_id", "evidenceId") or f"qev_{fallback_index:04d}", 80)


def _card_id(item: dict, fallback_index: int) -> str:
    return _truncate(_meta_text(item, "id", "card_id", "cardId") or f"q_{fallback_index:04d}", 80)


def _rect_payload(item: dict) -> dict:
    rect = _json_object(item.get("canonical_rect") or item.get("canonicalRect") or item.get("rect") or {})
    if rect and "width" not in rect and "w" in rect:
        rect["width"] = rect.get("w")
    if rect and "height" not in rect and "h" in rect:
        rect["height"] = rect.get("h")
    return rect


def _save_jpeg(image: Image.Image, path: Path, quality: int = 88) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=quality, optimize=True)
    return path.stat().st_size


def _image_metrics(image: Image.Image) -> dict:
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    edges = ImageChops.difference(gray, gray.filter(ImageFilter.SMOOTH_MORE))
    edge_stat = ImageStat.Stat(edges)
    return {
        "brightness": round(float(stat.mean[0]), 3),
        "contrast": round(float(stat.stddev[0]), 3),
        "edge_energy": round(float(edge_stat.mean[0]), 3),
    }


def _paper_score(image: Image.Image) -> float:
    small = image.copy()
    small.thumbnail((280, 280), Image.Resampling.LANCZOS)
    hsv = small.convert("HSV")
    total = max(1, small.width * small.height)
    paper = 0
    table = 0
    for hue, saturation, value in hsv.getdata():
        if value > 150 and saturation < 130:
            paper += 1
        if 10 <= hue <= 32 and saturation > 45 and value < 190:
            table += 1
    return paper / total - table / total * 0.85


def _average_hash_image(image: Image.Image, hash_size: int = 8) -> int:
    gray = ImageOps.grayscale(image).resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    pixels = list(gray.getdata())
    average = sum(pixels) / max(1, len(pixels))
    value = 0
    for index, pixel in enumerate(pixels):
        if pixel >= average:
            value |= 1 << index
    return value


def _average_hash_path(path: Path, hash_size: int = 8) -> int:
    with Image.open(path) as image:
        return _average_hash_image(image, hash_size=hash_size)


def _hash_distance(left: int | None, right: int | None) -> int:
    if left is None or right is None:
        return 64
    return int(left ^ right).bit_count()


def _rect_overlap_smaller(left: dict, right: dict) -> float:
    lx1 = float(left.get("x") or 0)
    ly1 = float(left.get("y") or 0)
    lx2 = lx1 + float(left.get("width") or 0)
    ly2 = ly1 + float(left.get("height") or 0)
    rx1 = float(right.get("x") or 0)
    ry1 = float(right.get("y") or 0)
    rx2 = rx1 + float(right.get("width") or 0)
    ry2 = ry1 + float(right.get("height") or 0)
    inter = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))
    smaller = min(max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1), max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1))
    if smaller <= 0:
        return 0.0
    return inter / smaller


def _rect_center_y(rect: dict) -> float:
    return float(rect.get("y") or 0) + float(rect.get("height") or 0) / 2


def _server_page_side(page_episode_id: object) -> str:
    text = str(page_episode_id or "")
    for suffix in ("_left", "_right", "_page"):
        if text.endswith(suffix):
            return suffix.lstrip("_")
    return "page"


def _quality_float(quality: dict, key: str) -> float:
    try:
        return float(quality.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def _is_low_information_crop(quality: dict) -> bool:
    paper = _quality_float(quality, "paper_score")
    edge = _quality_float(quality, "edge_energy")
    contrast = _quality_float(quality, "contrast")
    area = _quality_float(quality, "area")
    return paper > 0.88 and edge < 0.95 and contrast < 18 and area < 0.32


def _merge_source_frames(target: dict, source: dict) -> None:
    frames = []
    for value in _json_array(target.get("source_frames")) + _json_array(source.get("source_frames")):
        text = str(value or "")
        if text and text not in frames:
            frames.append(text)
    target["source_frames"] = frames
    reasons = []
    for value in _json_array(target.get("merge_reasons")) + _json_array(source.get("merge_reasons")) + ["server_duplicate_merged"]:
        text = str(value or "")
        if text and text not in reasons:
            reasons.append(text)
    target["merge_reasons"] = reasons
    try:
        target["seen_count"] = max(int(target.get("seen_count") or 1), 1) + max(int(source.get("seen_count") or 1), 1)
    except (TypeError, ValueError):
        target["seen_count"] = 2


def _dedupe_refined_candidates(
    evidence: list[dict],
    cards: list[dict],
    crop_paths: list[Path],
) -> tuple[list[dict], list[dict], list[Path], int, int]:
    kept_evidence: list[dict] = []
    kept_cards: list[dict] = []
    kept_paths: list[Path] = []
    kept_hashes: list[int | None] = []
    duplicate_count = 0
    blank_count = 0

    for item, card, path in zip(evidence, cards, crop_paths):
        quality = _json_object(item.get("quality"))
        if _is_low_information_crop(quality):
            blank_count += 1
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        try:
            image_hash: int | None = _average_hash_path(path)
        except Exception:
            image_hash = None
        item["crop_hash"] = f"{image_hash:016x}" if image_hash is not None else ""
        side = _server_page_side(item.get("page_episode_id"))
        rect = _json_object(item.get("canonical_rect"))
        duplicate_index: int | None = None
        for index, existing in enumerate(kept_evidence):
            if side != _server_page_side(existing.get("page_episode_id")):
                continue
            existing_rect = _json_object(existing.get("canonical_rect"))
            hash_distance = _hash_distance(image_hash, kept_hashes[index])
            center_distance = abs(_rect_center_y(rect) - _rect_center_y(existing_rect))
            overlap = _rect_overlap_smaller(rect, existing_rect)
            if hash_distance <= 16 and (center_distance <= 0.12 or overlap >= 0.55):
                duplicate_index = index
                break
        if duplicate_index is not None:
            duplicate_count += 1
            _merge_source_frames(kept_evidence[duplicate_index], item)
            kept_cards[duplicate_index]["confidence"] = max(
                float(kept_cards[duplicate_index].get("confidence") or 0),
                float(card.get("confidence") or 0),
            )
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        kept_evidence.append(item)
        kept_cards.append(card)
        kept_paths.append(path)
        kept_hashes.append(image_hash)

    return kept_evidence, kept_cards, kept_paths, duplicate_count, blank_count


def _rotate_for_reading(image: Image.Image) -> tuple[Image.Image, int]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.width <= image.height:
        return image, 0
    # The iOS camera stream currently stores this experiment in landscape while
    # the page is easiest to inspect after a clockwise turn.
    return image.rotate(90, expand=True, fillcolor=(255, 255, 255)), 90


def _resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image.copy()
    out = image.copy()
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def _order_cv_points(points):
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def _perspective_warp(image: Image.Image, rect, max_side: int) -> Image.Image:
    tl, tr, br, bl = rect
    width = max(1, int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = max(1, int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(np.asarray(image), matrix, (width, height), borderValue=(255, 255, 255))
    return _resize_max_side(Image.fromarray(warped), max_side)


def _normalize_page_perspective(image: Image.Image, max_side: int = 1800) -> tuple[Image.Image, bool, float]:
    if cv2 is None or np is None:
        return image, False, 0.0
    resized = _resize_max_side(image.convert("RGB"), max_side)
    arr = np.asarray(resized)
    height, width = arr.shape[:2]
    if width <= 0 or height <= 0:
        return resized, False, 0.0
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 135)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(width * height)
    best_rect = None
    best_area = 0.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.18:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) == 4:
            best_rect = _order_cv_points(approx.reshape(4, 2).astype("float32"))
            best_area = area
            break
    if best_rect is None:
        return resized, False, 0.0
    confidence = min(1.0, best_area / max(image_area, 1.0))
    if confidence < 0.22:
        return resized, False, confidence
    try:
        return _perspective_warp(resized, best_rect, max_side), True, confidence
    except Exception:
        return resized, False, confidence


def _content_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    thumb = image.copy()
    thumb.thumbnail((700, 700), Image.Resampling.LANCZOS)
    scale_x = image.width / max(1, thumb.width)
    scale_y = image.height / max(1, thumb.height)
    hsv = thumb.convert("HSV")
    mask = Image.new("L", thumb.size)
    data = []
    for _h, saturation, value in hsv.getdata():
        # Workbook paper is usually bright and not very saturated, but printed
        # figures can be colorful. The high-value branch keeps those regions.
        data.append(255 if (value > 132 and saturation < 120) or value > 218 else 0)
    mask.putdata(data)
    mask = mask.filter(ImageFilter.MaxFilter(13)).filter(ImageFilter.MinFilter(7))
    bbox = mask.getbbox()
    if not bbox:
        return None
    x1, y1, x2, y2 = bbox
    pad_x = max(10, int((x2 - x1) * 0.03))
    pad_y = max(10, int((y2 - y1) * 0.025))
    return (
        max(0, int((x1 - pad_x) * scale_x)),
        max(0, int((y1 - pad_y) * scale_y)),
        min(image.width, int((x2 + pad_x) * scale_x)),
        min(image.height, int((y2 + pad_y) * scale_y)),
    )


def _projection_segments(values: list[float], *, min_gap: int, min_size: int, threshold: float) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    gap = 0
    for index, value in enumerate(values):
        active = value >= threshold
        if active:
            if start is None:
                start = index
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                end = index - gap + 1
                if end - start >= min_size:
                    segments.append((start, end))
                start = None
                gap = 0
    if start is not None and len(values) - start >= min_size:
        segments.append((start, len(values)))
    return segments


def _merge_close_segments(segments: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _detect_refined_regions(page: Image.Image, max_regions: int) -> list[dict]:
    gray = page.convert("L")
    small = gray.copy()
    small.thumbnail((900, 1200), Image.Resampling.LANCZOS)
    sx = page.width / max(1, small.width)
    sy = page.height / max(1, small.height)
    edges = ImageChops.difference(small, small.filter(ImageFilter.SMOOTH_MORE))
    stat = ImageStat.Stat(edges)
    threshold = max(8.0, float(stat.mean[0]) + float(stat.stddev[0]) * 0.55)
    mask = edges.point(lambda pixel: 255 if pixel >= threshold else 0)
    mask = mask.filter(ImageFilter.MaxFilter(5))

    width, height = mask.size
    pixels = mask.load()
    row_density = []
    for y in range(height):
        row_density.append(sum(1 for x in range(width) if pixels[x, y]) / max(1, width))
    average_row_density = sum(row_density) / max(1, len(row_density))
    active_rows = _projection_segments(
        row_density,
        min_gap=max(6, height // 80),
        min_size=max(16, height // 38),
        threshold=max(0.035, min(0.38, average_row_density * 1.15)),
    )
    active_rows = _merge_close_segments(active_rows, max(8, height // 60))

    regions: list[dict] = []
    for row_start, row_end in active_rows:
        col_density = []
        for x in range(width):
            col_density.append(sum(1 for y in range(row_start, row_end) if pixels[x, y]) / max(1, row_end - row_start))
        average_col_density = sum(col_density) / max(1, len(col_density))
        col_threshold = max(0.025, min(0.32, average_col_density * 1.10))
        col_segments = _projection_segments(
            col_density,
            min_gap=max(6, width // 60),
            min_size=max(22, width // 12),
            threshold=col_threshold,
        )
        col_segments = _merge_close_segments(col_segments, max(10, width // 45))
        if not col_segments:
            col_segments = [(0, width)]
        for col_start, col_end in col_segments:
            x1 = max(0, int((col_start - width * 0.025) * sx))
            y1 = max(0, int((row_start - height * 0.012) * sy))
            x2 = min(page.width, int((col_end + width * 0.025) * sx))
            y2 = min(page.height, int((row_end + height * 0.018) * sy))
            area = (x2 - x1) * (y2 - y1)
            if area < page.width * page.height * 0.006:
                continue
            if y2 - y1 < max(36, page.height * 0.025) or x2 - x1 < max(80, page.width * 0.10):
                continue
            rect = {
                "x": x1 / page.width,
                "y": y1 / page.height,
                "width": (x2 - x1) / page.width,
                "height": (y2 - y1) / page.height,
            }
            regions.append({"rect": rect, "area": area, "kind": "server_question"})

    regions.sort(key=lambda item: (item["rect"]["y"], item["rect"]["x"]))
    filtered: list[dict] = []
    for region in regions:
        rect = region["rect"]
        duplicate = False
        for existing in filtered:
            other = existing["rect"]
            ix1 = max(rect["x"], other["x"])
            iy1 = max(rect["y"], other["y"])
            ix2 = min(rect["x"] + rect["width"], other["x"] + other["width"])
            iy2 = min(rect["y"] + rect["height"], other["y"] + other["height"])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            smaller = min(rect["width"] * rect["height"], other["width"] * other["height"])
            if smaller > 0 and inter / smaller > 0.72:
                duplicate = True
                break
        if not duplicate:
            filtered.append(region)
        if len(filtered) >= max_regions:
            break
    if filtered:
        return filtered

    sections = []
    for index, (y, height_ratio) in enumerate([(0.04, 0.30), (0.35, 0.30), (0.66, 0.30)], start=1):
        sections.append(
            {
                "rect": {"x": 0.04, "y": y, "width": 0.92, "height": height_ratio},
                "area": page.width * page.height * 0.92 * height_ratio,
                "kind": "server_section",
                "fallback_index": index,
            }
        )
    return sections[:max_regions]


def _split_open_book_pages(image: Image.Image) -> list[tuple[str, Image.Image, dict]]:
    """Split a camera frame of an open workbook into page-like crops."""
    if image.width < 640 or image.height < 900:
        return [("page", image, {"x": 0, "y": 0, "width": 1, "height": 1})]
    gray = image.convert("L")
    small = gray.copy()
    small.thumbnail((520, 920), Image.Resampling.LANCZOS)
    width, height = small.size
    pixels = small.load()
    column_density: list[float] = []
    y1 = int(height * 0.05)
    y2 = int(height * 0.96)
    for x in range(width):
        dark = sum(1 for y in range(y1, y2) if pixels[x, y] < 175)
        column_density.append(dark / max(1, y2 - y1))
    smoothed = []
    for x in range(width):
        left = max(0, x - 4)
        right = min(width, x + 5)
        smoothed.append(sum(column_density[left:right]) / max(1, right - left))
    search_left = int(width * 0.38)
    search_right = int(width * 0.64)
    valley = min(range(search_left, search_right), key=lambda x: smoothed[x])
    split_x = int(valley / max(1, width) * image.width)
    if not (image.width * 0.35 <= split_x <= image.width * 0.68):
        return [("page", image, {"x": 0, "y": 0, "width": 1, "height": 1})]
    overlap = max(12, int(image.width * 0.025))
    left_box = (0, 0, min(image.width, split_x + overlap), image.height)
    right_box = (max(0, split_x - overlap), 0, image.width, image.height)
    pages: list[tuple[str, Image.Image, dict]] = []
    for name, box in [("left", left_box), ("right", right_box)]:
        crop = image.crop(box)
        if crop.width < image.width * 0.22 or crop.height < image.height * 0.45:
            continue
        pages.append(
            (
                name,
                crop,
                {
                    "x": box[0] / image.width,
                    "y": box[1] / image.height,
                    "width": (box[2] - box[0]) / image.width,
                    "height": (box[3] - box[1]) / image.height,
                    "split_x": split_x / image.width,
                },
            )
        )
    return pages or [("page", image, {"x": 0, "y": 0, "width": 1, "height": 1})]


def _detect_page_block_regions(page: Image.Image, max_regions: int) -> list[dict]:
    """Detect coarse, high-recall page blocks for later VLM question splitting."""
    gray = page.convert("L")
    small = gray.copy()
    small.thumbnail((900, 1200), Image.Resampling.LANCZOS)
    sx = page.width / max(1, small.width)
    sy = page.height / max(1, small.height)
    edges = ImageChops.difference(small, small.filter(ImageFilter.SMOOTH_MORE))
    stat = ImageStat.Stat(edges)
    threshold = max(8.0, float(stat.mean[0]) + float(stat.stddev[0]) * 0.55)
    mask = edges.point(lambda pixel: 255 if pixel >= threshold else 0).filter(ImageFilter.MaxFilter(5))
    width, height = mask.size
    pixels = mask.load()
    row_density = [
        sum(1 for x in range(width) if pixels[x, y]) / max(1, width)
        for y in range(height)
    ]
    average_row_density = sum(row_density) / max(1, len(row_density))
    active_rows = _projection_segments(
        row_density,
        min_gap=max(6, height // 80),
        min_size=max(12, height // 60),
        threshold=max(0.035, min(0.38, average_row_density * 1.15)),
    )
    active_rows = _merge_close_segments(active_rows, max(12, height // 28))
    raw_regions: list[tuple[int, int]] = []
    for row_start, row_end in active_rows:
        y1 = max(0, int((row_start - height * 0.01) * sy))
        y2 = min(page.height, int((row_end + height * 0.016) * sy))
        if y2 - y1 >= max(36, int(page.height * 0.025)):
            raw_regions.append((y1, y2))
    if not raw_regions:
        raw_regions = [(0, page.height)]

    # Small one-line fragments are not useful evidence by themselves. Merge them
    # into neighboring blocks so the next VLM pass gets enough context and does
    # not miss diagrams/answers that belong to the same question area.
    min_block = max(140, int(page.height * 0.13))
    gap_limit = max(70, int(page.height * 0.07))
    merged: list[tuple[int, int]] = []
    for y1, y2 in raw_regions:
        if merged and (y1 - merged[-1][1] <= gap_limit or merged[-1][1] - merged[-1][0] < min_block):
            merged[-1] = (merged[-1][0], max(merged[-1][1], y2))
        else:
            merged.append((y1, y2))
    compact: list[tuple[int, int]] = []
    for y1, y2 in merged:
        if compact and y2 - y1 < min_block:
            compact[-1] = (compact[-1][0], y2)
        else:
            compact.append((y1, y2))
    if compact and compact[-1][1] - compact[-1][0] < min_block and len(compact) > 1:
        last = compact.pop()
        compact[-1] = (compact[-1][0], last[1])

    if len(compact) > max_regions:
        # Fall back to evenly sized high-recall page bands when CV found too many
        # fragments. This is intentionally coarse: missing less beats fake precision.
        bands = min(max_regions, 4 if page.height > 1100 else 3)
        compact = []
        for index in range(bands):
            y1 = int(index * page.height / bands)
            y2 = int((index + 1) * page.height / bands)
            compact.append((max(0, y1 - 18), min(page.height, y2 + 18)))

    regions: list[dict] = []
    for y1, y2 in compact[:max_regions]:
        y1 = max(0, y1)
        y2 = min(page.height, max(y1 + 1, y2))
        rect = {"x": 0.0, "y": y1 / page.height, "width": 1.0, "height": (y2 - y1) / page.height}
        regions.append({"rect": rect, "area": page.width * (y2 - y1), "kind": "server_page_block"})
    return regions


def _crop_rect(image: Image.Image, rect: dict) -> Image.Image:
    x = max(0.0, min(1.0, float(rect.get("x") or 0.0)))
    y = max(0.0, min(1.0, float(rect.get("y") or 0.0)))
    w = max(0.01, min(1.0 - x, float(rect.get("width") or rect.get("w") or 0.0)))
    h = max(0.01, min(1.0 - y, float(rect.get("height") or rect.get("h") or 0.0)))
    left = int(x * image.width)
    top = int(y * image.height)
    right = max(left + 1, int((x + w) * image.width))
    bottom = max(top + 1, int((y + h) * image.height))
    return image.crop((left, top, min(image.width, right), min(image.height, bottom)))


def register_homework_ledger_routes(
    app,
    *,
    init_db: Callable[[], None],
    principal_from_request: Callable[[Request], dict],
    connect: Callable,
    utc_now: Callable[[], str],
    get_settings: Callable,
    resolve_student_profile: Callable[[str, str], str],
    clean_user_text: Callable,
    json_dumps: Callable[[object], str],
    emit_log: Callable,
) -> None:
    def ledger_root() -> Path:
        path = get_settings().data_dir / HOMEWORK_LEDGER_ASSET_ROOT
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run_dir(run_id: str) -> Path:
        safe_run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(run_id or ""))[:120]
        if not safe_run_id:
            raise HTTPException(400, "invalid run id")
        path = (ledger_root() / safe_run_id).resolve()
        root = ledger_root().resolve()
        if not path.is_relative_to(root):
            raise HTTPException(400, "invalid run id")
        path.mkdir(parents=True, exist_ok=True)
        (path / "frames").mkdir(exist_ok=True)
        (path / "crops").mkdir(exist_ok=True)
        return path

    def asset_file_path(run_id: str, rel_path: str) -> Path:
        clean_rel = str(rel_path or "").replace("\\", "/").strip().lstrip("/")
        if not clean_rel or ".." in Path(clean_rel).parts:
            raise HTTPException(400, "invalid asset path")
        base = run_dir(run_id).resolve()
        path = (base / clean_rel).resolve()
        if not path.is_relative_to(base):
            raise HTTPException(400, "invalid asset path")
        if not path.is_file():
            raise HTTPException(404, "asset not found")
        return path

    def asset_url(run_id: str, rel_path: str) -> str:
        clean_rel = str(rel_path or "").replace("\\", "/").strip().lstrip("/")
        if not clean_rel:
            return ""
        return f"/api/homework-ledger/runs/{quote(run_id, safe='')}/assets/{quote(clean_rel, safe='/')}"

    def asset_src(run_id: str, rel_path: str, *, inline: bool = False) -> str:
        raw = str(rel_path or "").strip()
        if not raw:
            return ""
        if raw.startswith("data:") or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", raw):
            return raw
        asset_prefix = f"/api/homework-ledger/runs/{quote(run_id, safe='')}/assets/"
        if raw.startswith(asset_prefix):
            raw = unquote(raw[len(asset_prefix) :])
        elif raw.startswith("/api/homework-ledger/runs/") and "/assets/" in raw:
            raw = unquote(raw.split("/assets/", 1)[1])
        clean_rel = raw.replace("\\", "/").strip().lstrip("/")
        if not inline:
            return asset_url(run_id, clean_rel)
        try:
            path = asset_file_path(run_id, clean_rel)
            suffix = path.suffix.lower()
            media_type = "image/jpeg"
            if suffix == ".png":
                media_type = "image/png"
            elif suffix == ".webp":
                media_type = "image/webp"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{media_type};base64,{encoded}"
        except Exception:
            return asset_url(run_id, clean_rel)

    async def save_upload(
        upload: UploadFile,
        *,
        run_id: str,
        rel_path: str,
        default_folder: str,
        fallback_prefix: str,
    ) -> tuple[str, int]:
        ext = Path(upload.filename or rel_path or "asset.jpg").suffix.lower() or ".jpg"
        fallback_name = f"{fallback_prefix}_{uuid.uuid4().hex[:10]}{ext}"
        safe_rel = _safe_asset_rel(rel_path, default_folder, fallback_name)
        target = run_dir(run_id) / safe_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            upload.file.seek(0)
        except Exception:
            pass
        with target.open("wb") as out:
            shutil.copyfileobj(upload.file, out)
        return safe_rel, target.stat().st_size

    def require_run(conn, run_id: str, principal: dict) -> dict:
        if principal.get("authenticated") or get_settings().auth_required:
            row = conn.execute(
                "SELECT * FROM homework_ledger_runs WHERE id=? AND account_id=?",
                (run_id, principal["account_id"]),
            ).fetchone()
        else:
            row = conn.execute("SELECT * FROM homework_ledger_runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise HTTPException(404, "homework ledger run not found")
        return dict(row)

    def card_row(row: dict, run_id: str) -> dict:
        item = dict(row)
        item["evidence_ids"] = _json_array(item.pop("evidence_ids", "[]"))
        item["figure_assets"] = _json_array(item.pop("figure_assets", "[]"))
        item["review_flags"] = _json_array(item.pop("review_flags", "[]"))
        item["payload"] = _json_object(item.get("payload"))
        for asset in item["figure_assets"]:
            if isinstance(asset, dict) and asset.get("filename") and not asset.get("url"):
                asset["url"] = asset_url(run_id, str(asset.get("filename") or ""))
        return item

    def payload_for(run_id: str, principal: dict) -> dict:
        with connect() as conn:
            run = require_run(conn, run_id, principal)
            frames = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM homework_ledger_frames WHERE run_id=? ORDER BY sequence_index, created_at",
                    (run_id,),
                )
            ]
            pages = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM homework_ledger_page_episodes WHERE run_id=? ORDER BY created_at",
                    (run_id,),
                )
            ]
            evidence = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM homework_ledger_evidence WHERE run_id=? ORDER BY created_at, id",
                    (run_id,),
                )
            ]
            cards = [
                card_row(dict(row), run_id)
                for row in conn.execute(
                    "SELECT * FROM homework_ledger_cards WHERE run_id=? ORDER BY created_at, id",
                    (run_id,),
                )
            ]
        run["metrics"] = _json_object(run.get("metrics"))
        run["client_summary"] = _json_object(run.get("client_summary"))
        for frame in frames:
            frame["accepted"] = bool(frame.get("accepted"))
            frame["frame_meta"] = _json_object(frame.get("frame_meta"))
            if frame.get("filename"):
                frame["url"] = asset_url(run_id, frame["filename"])
                frame["inline_url"] = asset_src(run_id, frame["filename"], inline=True)
        for page in pages:
            page["frame_ids"] = _json_array(page.get("frame_ids"))
            page["coverage_cells"] = _json_array(page.get("coverage_cells"))
            page["payload"] = _json_object(page.get("payload"))
        for item in evidence:
            item["canonical_rect"] = _json_object(item.get("canonical_rect"))
            item["quality"] = _json_object(item.get("quality"))
            item["source_frames"] = _json_array(item.get("source_frames"))
            item["merge_reasons"] = _json_array(item.get("merge_reasons"))
            item["payload"] = _json_object(item.get("payload"))
            if item.get("best_crop_filename"):
                item["best_crop_url"] = asset_url(run_id, item["best_crop_filename"])
                item["best_crop_inline_url"] = asset_src(run_id, item["best_crop_filename"], inline=True)
        return {"run": run, "frames": frames, "page_episodes": pages, "evidence": evidence, "cards": cards}

    def latest_run_id(principal: dict) -> str:
        with connect() as conn:
            if principal.get("authenticated") or get_settings().auth_required:
                row = conn.execute(
                    "SELECT id FROM homework_ledger_runs WHERE account_id=? ORDER BY created_at DESC LIMIT 1",
                    (principal["account_id"],),
                ).fetchone()
            else:
                row = conn.execute("SELECT id FROM homework_ledger_runs ORDER BY created_at DESC LIMIT 1").fetchone()
        if not row:
            raise HTTPException(404, "no homework ledger run found")
        return str(row["id"])

    def upsert_manifest(conn, run_id: str, manifest: dict, now: str) -> dict:
        frames = _manifest_list(manifest, "frames", "frame_records")
        pages = _manifest_list(manifest, "page_episodes", "pages", "episodes")
        evidence_items = _manifest_list(manifest, "question_evidence", "evidence", "questionEvidence")
        cards = _manifest_list(manifest, "question_cards", "cards", "questionCards")
        metrics = _json_object(manifest.get("metrics"))
        client_summary = {
            key: manifest.get(key)
            for key in ("source", "input", "out_dir", "simulator_version", "client_version")
            if manifest.get(key) not in (None, "", [], {})
        }
        if metrics:
            conn.execute(
                "UPDATE homework_ledger_runs SET metrics=?, client_summary=?, updated_at=? WHERE id=?",
                (json_dumps(metrics), json_dumps(client_summary), now, run_id),
            )
        for index, item in enumerate(frames, start=1):
            frame_id = _frame_id(item, index)
            filename = _truncate(
                _meta_text(item, "filename", "normalized_filename", "normalizedFilename", "normalized_path", "normalizedPath"),
                260,
            ).replace("\\", "/")
            if filename and "/" in filename:
                filename = _safe_asset_rel(filename, "frames", f"{frame_id}.jpg")
            conn.execute(
                """
                INSERT INTO homework_ledger_frames(
                    id, run_id, sequence_index, filename, original_name, width, height,
                    source_bytes, normalized_bytes, sharpness, brightness, contrast, phash,
                    accepted, reason, page_episode_id, frame_meta, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, id) DO UPDATE SET
                    sequence_index=excluded.sequence_index, filename=excluded.filename,
                    original_name=excluded.original_name, width=excluded.width, height=excluded.height,
                    source_bytes=excluded.source_bytes, normalized_bytes=excluded.normalized_bytes,
                    sharpness=excluded.sharpness, brightness=excluded.brightness, contrast=excluded.contrast,
                    phash=excluded.phash, accepted=excluded.accepted, reason=excluded.reason,
                    page_episode_id=excluded.page_episode_id, frame_meta=excluded.frame_meta
                """,
                (
                    frame_id,
                    run_id,
                    _int_value(item.get("sequence_index") or item.get("sequenceIndex") or item.get("index")) or index,
                    filename,
                    _truncate(_meta_text(item, "original_name", "originalName", "source_path", "sourcePath"), 260),
                    _int_value(item.get("width")) or 0,
                    _int_value(item.get("height")) or 0,
                    _int_value(item.get("source_bytes") or item.get("sourceBytes")) or 0,
                    _int_value(item.get("normalized_bytes") or item.get("normalizedBytes")) or 0,
                    float(item.get("sharpness") or 0),
                    float(item.get("brightness") or 0),
                    float(item.get("contrast") or 0),
                    _truncate(_meta_text(item, "phash", "visual_hash", "visualHash"), 80),
                    1 if item.get("accepted", True) else 0,
                    _truncate(_meta_text(item, "reason"), 160),
                    _truncate(_meta_text(item, "page_episode_id", "pageEpisodeId"), 80),
                    json_dumps(item),
                    now,
                ),
            )
        for index, item in enumerate(pages, start=1):
            page_id = _truncate(_meta_text(item, "id", "page_episode_id", "pageEpisodeId") or f"page_{index:04d}", 80)
            conn.execute(
                """
                INSERT INTO homework_ledger_page_episodes(
                    id, run_id, first_frame_id, last_frame_id, frame_ids, fingerprint,
                    coverage_cells, payload, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, id) DO UPDATE SET
                    first_frame_id=excluded.first_frame_id, last_frame_id=excluded.last_frame_id,
                    frame_ids=excluded.frame_ids, fingerprint=excluded.fingerprint,
                    coverage_cells=excluded.coverage_cells, payload=excluded.payload, updated_at=excluded.updated_at
                """,
                (
                    page_id,
                    run_id,
                    _truncate(_meta_text(item, "first_frame_id", "firstFrameId"), 80),
                    _truncate(_meta_text(item, "last_frame_id", "lastFrameId"), 80),
                    json_dumps(_json_array(item.get("frame_ids") or item.get("frameIds"))),
                    _truncate(_meta_text(item, "fingerprint"), 160),
                    json_dumps(_json_array(item.get("coverage_cells") or item.get("coverageCells"))),
                    json_dumps(item),
                    now,
                    now,
                ),
            )
        for index, item in enumerate(evidence_items, start=1):
            evidence_id = _evidence_id(item, index)
            crop_filename = _truncate(
                _meta_text(item, "best_crop_filename", "bestCropFilename", "crop_filename", "cropFilename"),
                260,
            ).replace("\\", "/")
            if crop_filename and "/" in crop_filename:
                crop_filename = _safe_asset_rel(crop_filename, "crops", f"{evidence_id}.jpg")
            conn.execute(
                """
                INSERT INTO homework_ledger_evidence(
                    id, run_id, page_episode_id, best_frame_id, best_crop_filename, crop_kind,
                    canonical_rect, crop_hash, layout_key, ocr_key, seen_count, status,
                    quality, source_frames, merge_reasons, payload, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, id) DO UPDATE SET
                    page_episode_id=excluded.page_episode_id, best_frame_id=excluded.best_frame_id,
                    best_crop_filename=excluded.best_crop_filename, crop_kind=excluded.crop_kind,
                    canonical_rect=excluded.canonical_rect, crop_hash=excluded.crop_hash,
                    layout_key=excluded.layout_key, ocr_key=excluded.ocr_key, seen_count=excluded.seen_count,
                    status=excluded.status, quality=excluded.quality, source_frames=excluded.source_frames,
                    merge_reasons=excluded.merge_reasons, payload=excluded.payload, updated_at=excluded.updated_at
                """,
                (
                    evidence_id,
                    run_id,
                    _truncate(_meta_text(item, "page_episode_id", "pageEpisodeId"), 80),
                    _truncate(_meta_text(item, "best_frame_id", "bestFrameId"), 80),
                    crop_filename,
                    _truncate(_meta_text(item, "crop_kind", "cropKind") or "question", 40),
                    json_dumps(_rect_payload(item)),
                    _truncate(_meta_text(item, "crop_hash", "cropHash"), 120),
                    _truncate(_meta_text(item, "layout_key", "layoutKey"), 160),
                    _truncate(_meta_text(item, "ocr_key", "ocrKey"), 160),
                    _int_value(item.get("seen_count") or item.get("seenCount")) or 1,
                    _truncate(_meta_text(item, "status") or "ready", 40),
                    json_dumps(_json_object(item.get("quality"))),
                    json_dumps(_json_array(item.get("source_frames") or item.get("sourceFrames"))),
                    json_dumps(_json_array(item.get("merge_reasons") or item.get("mergeReasons"))),
                    json_dumps(item),
                    now,
                    now,
                ),
            )
        for index, item in enumerate(cards, start=1):
            card_id = _card_id(item, index)
            evidence_ids = _json_array(item.get("evidence_ids") or item.get("evidenceIds"))
            figure_assets = _json_array(item.get("figure_assets") or item.get("figureAssets"))
            normalized_assets: list[dict] = []
            for asset in figure_assets:
                if not isinstance(asset, dict):
                    continue
                normalized = dict(asset)
                filename = str(normalized.get("filename") or "").replace("\\", "/").strip().lstrip("/")
                if filename and "/" in filename:
                    normalized["filename"] = _safe_asset_rel(filename, "crops", Path(filename).name)
                normalized_assets.append(normalized)
            conn.execute(
                """
                INSERT INTO homework_ledger_cards(
                    id, run_id, card_key, evidence_ids, number, subject, question_type,
                    stem_text, editable_html, figure_assets, confidence, review_flags,
                    payload, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, id) DO UPDATE SET
                    card_key=excluded.card_key, evidence_ids=excluded.evidence_ids, number=excluded.number,
                    subject=excluded.subject, question_type=excluded.question_type, stem_text=excluded.stem_text,
                    editable_html=excluded.editable_html, figure_assets=excluded.figure_assets,
                    confidence=excluded.confidence, review_flags=excluded.review_flags,
                    payload=excluded.payload, updated_at=excluded.updated_at
                """,
                (
                    card_id,
                    run_id,
                    _truncate(_meta_text(item, "card_key", "cardKey") or card_id, 160),
                    json_dumps(evidence_ids),
                    _truncate(_meta_text(item, "number"), 80),
                    _truncate(_meta_text(item, "subject"), 80),
                    _truncate(_meta_text(item, "question_type", "questionType", "qtype"), 80),
                    _truncate(_meta_text(item, "stem_text", "stemText", "stem"), 4000),
                    _truncate(_meta_text(item, "editable_html", "editableHtml"), HOMEWORK_LEDGER_HTML_LIMIT),
                    json_dumps(normalized_assets),
                    float(item.get("confidence") or 0),
                    json_dumps(_json_array(item.get("review_flags") or item.get("reviewFlags"))),
                    json_dumps(item),
                    now,
                    now,
                ),
            )
        return {"frames": len(frames), "page_episodes": len(pages), "evidence": len(evidence_items), "cards": len(cards)}

    def render_html(payload: dict) -> str:
        return render_compare_html(payload, inline_assets=True)
        run = payload["run"]
        metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
        cards = payload.get("cards") or []
        evidence_by_id = {item.get("id"): item for item in payload.get("evidence") or []}
        metric_bits = [
            f"frames {html.escape(str(metrics.get('frames_total', len(payload.get('frames') or []))))}",
            f"keyframes {html.escape(str(metrics.get('keyframes', '')))}",
            f"evidence {html.escape(str(metrics.get('question_evidence_count', len(payload.get('evidence') or []))))}",
            f"cards {html.escape(str(metrics.get('question_card_count', len(cards))))}",
            f"upload ratio {html.escape(str(metrics.get('estimated_upload_ratio', '')))}",
        ]
        card_html: list[str] = []
        for index, card in enumerate(cards, start=1):
            title = card.get("number") or f"#{index}"
            flags = " ".join(f"<span>{html.escape(str(flag))}</span>" for flag in card.get("review_flags", [])[:8])
            editable = card.get("editable_html") or f"<article><p>{html.escape(card.get('stem_text') or 'Needs recognition')}</p></article>"
            assets: list[str] = []
            for asset in card.get("figure_assets") or []:
                if not isinstance(asset, dict):
                    continue
                src = asset.get("url") or asset_url(run["id"], str(asset.get("filename") or ""))
                if src:
                    assets.append(
                        f'<figure><img src="{html.escape(src)}" alt="{html.escape(str(asset.get("source_evidence_id") or card.get("id") or ""))}">'
                        f'<figcaption>{html.escape(str(asset.get("type") or "raster"))}</figcaption></figure>'
                    )
            if not assets:
                for evidence_id in card.get("evidence_ids") or []:
                    evidence = evidence_by_id.get(evidence_id)
                    if evidence and evidence.get("best_crop_url"):
                        assets.append(
                            f'<figure><img src="{html.escape(evidence["best_crop_url"])}" alt="{html.escape(str(evidence_id))}">'
                            f'<figcaption>evidence {html.escape(str(evidence_id))}</figcaption></figure>'
                        )
            card_html.append(
                '<section class="card">'
                f'<header><div><strong>{html.escape(str(title))}</strong><small>{html.escape(str(card.get("subject") or "unknown"))} / {html.escape(str(card.get("question_type") or "evidence"))}</small></div>'
                f'<meter min="0" max="1" value="{max(0.0, min(1.0, float(card.get("confidence") or 0))):.3f}"></meter></header>'
                f'<div class="editable" contenteditable="true">{editable}</div>'
                f'<div class="assets">{"".join(assets)}</div>'
                f'<div class="flags">{flags}</div>'
                '</section>'
            )
        return (
            '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(run.get("title") or "Homework Evidence Ledger")}</title>'
            '<style>'
            'body{margin:0;background:#f7f7f4;color:#1d1d1b;font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}'
            'main{max-width:1040px;margin:0 auto;padding:18px}'
            'h1{font-size:22px;margin:0 0 8px}.summary{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 18px}.summary span{border:1px solid #d2d2ca;background:#fff;padding:5px 8px;border-radius:6px;font-size:12px}'
            '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}.card{background:#fff;border:1px solid #ddd9ce;border-radius:8px;padding:14px;min-width:0}'
            '.card header{display:flex;justify-content:space-between;gap:12px;align-items:start;border-bottom:1px solid #eee9dd;padding-bottom:8px;margin-bottom:10px}.card small{display:block;color:#6d6a61;margin-top:3px}'
            '.editable{outline:0;line-height:1.55;font-size:15px}.editable:focus{box-shadow:0 0 0 2px #2670ff33;border-radius:4px}.assets{display:grid;gap:8px;margin-top:10px}.assets img{max-width:100%;border:1px solid #dedbd2;border-radius:6px;background:#fafafa}.assets figure{margin:0}.assets figcaption{font-size:11px;color:#777;text-align:center;margin-top:3px}'
            '.flags{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}.flags span{background:#edf3ff;color:#22426f;border-radius:6px;padding:3px 6px;font-size:11px}meter{width:80px}'
            '</style></head><body><main>'
            f'<h1>{html.escape(run.get("title") or "Homework Evidence Ledger")}</h1>'
            f'<div class="summary">{"".join(f"<span>{bit}</span>" for bit in metric_bits if bit.strip())}</div>'
            f'<div class="grid">{"".join(card_html) or "<p>No question evidence yet.</p>"}</div>'
            '</main></body></html>'
        )

    def _metric_chip(label: str, value: object) -> str:
        if value in (None, "", [], {}):
            return ""
        return f'<span><strong>{html.escape(label)}</strong>{html.escape(str(value))}</span>'

    def _rect_style(rect: dict) -> str:
        try:
            x = max(0.0, min(1.0, float(rect.get("x") or 0.0)))
            y = max(0.0, min(1.0, float(rect.get("y") or 0.0)))
            width = max(0.0, min(1.0 - x, float(rect.get("width") or rect.get("w") or 0.0)))
            height = max(0.0, min(1.0 - y, float(rect.get("height") or rect.get("h") or 0.0)))
        except (TypeError, ValueError):
            return ""
        if width <= 0 or height <= 0:
            return ""
        return (
            f"left:{x * 100:.3f}%;top:{y * 100:.3f}%;"
            f"width:{width * 100:.3f}%;height:{height * 100:.3f}%"
        )

    def _digital_iframe(run_id: str, card: dict, *, inline_assets: bool = False) -> str:
        editable = card.get("editable_html") or f"<article><p>{html.escape(card.get('stem_text') or '待识别题目')}</p></article>"
        if inline_assets:
            for match in sorted(set(re.findall(r'(?:src|href)=["\']([^"\']+)["\']', editable)), key=len, reverse=True):
                if match.startswith("data:") or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", match):
                    continue
                editable = editable.replace(match, asset_src(run_id, match, inline=True))
        asset_base = f"/api/homework-ledger/runs/{quote(run_id, safe='')}/assets/"
        document = (
            '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            f'<base href="{html.escape(asset_base, quote=True)}">'
            '<style>'
            'body{margin:0;padding:12px;background:#fff;color:#191916;font:15px/1.55 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}'
            'article{margin:0}p{margin:0 0 10px}img{max-width:100%;height:auto;border:1px solid #d8d4c9;border-radius:6px}'
            'figure{margin:10px 0 0}table{border-collapse:collapse;max-width:100%}td,th{border:1px solid #ddd;padding:3px 5px}'
            '</style></head><body>'
            f'{editable}'
            '</body></html>'
        )
        return f'<iframe class="digital-frame" sandbox srcdoc="{html.escape(document, quote=True)}"></iframe>'

    def render_compare_html(payload: dict, *, inline_assets: bool = False) -> str:
        run = payload["run"]
        run_id = str(run.get("id") or "")
        metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
        frames = payload.get("frames") or []
        evidence_items = payload.get("evidence") or []
        cards = payload.get("cards") or []
        frames_by_id = {item.get("id"): item for item in frames}
        evidence_by_id = {item.get("id"): item for item in evidence_items}
        original_card_count = sum(1 for card in cards if not str(card.get("id") or "").startswith("srv_"))
        refined_card_count = sum(1 for card in cards if str(card.get("id") or "").startswith("srv_"))
        original_evidence_count = sum(1 for item in evidence_items if not str(item.get("id") or "").startswith("srv_"))
        refined_evidence_count = sum(1 for item in evidence_items if str(item.get("id") or "").startswith("srv_"))
        vlm_status = "已启用" if metrics.get("vlm_enabled") else "未启用"

        metric_chips = "".join(
            chip
            for chip in [
                _metric_chip("状态", run.get("status")),
                _metric_chip("总帧", metrics.get("frames_total", len(frames))),
                _metric_chip("关键帧", metrics.get("keyframes", len(frames))),
                _metric_chip("跳过", metrics.get("skipped_frames")),
                _metric_chip("端上证据", metrics.get("question_evidence_count", original_evidence_count)),
                _metric_chip("端上题卡", metrics.get("question_card_count", original_card_count)),
                _metric_chip("服务端证据", metrics.get("server_refined_evidence", refined_evidence_count)),
                _metric_chip("服务端候选", metrics.get("server_refined_cards", refined_card_count)),
                _metric_chip("合并重复", metrics.get("server_refined_duplicates_merged")),
                _metric_chip("丢弃低信息", metrics.get("server_refined_low_info_dropped")),
                _metric_chip("VLM", vlm_status),
                _metric_chip("上传比例", f"{float(metrics.get('estimated_upload_ratio')) * 100:.1f}%" if metrics.get("estimated_upload_ratio") not in (None, "") else ""),
            ]
            if chip
        )

        frame_strip = []
        for frame in frames:
            src = (frame.get("inline_url") if inline_assets else frame.get("url")) or ""
            if not src:
                continue
            frame_strip.append(
                '<figure class="frame-thumb">'
                f'<img src="{html.escape(src, quote=True)}" alt="{html.escape(str(frame.get("id") or ""))}">'
                f'<figcaption>{html.escape(str(frame.get("id") or ""))}</figcaption>'
                '</figure>'
            )

        rows: list[str] = []
        if cards:
            card_source = cards
        else:
            card_source = [
                {
                    "id": item.get("id"),
                    "evidence_ids": [item.get("id")],
                    "number": "",
                    "subject": "unknown",
                    "question_type": item.get("crop_kind") or "evidence",
                    "confidence": item.get("quality", {}).get("confidence") if isinstance(item.get("quality"), dict) else 0,
                    "review_flags": ["no_card"],
                    "stem_text": "待识别题目",
                    "editable_html": "",
                }
                for item in evidence_items
            ]
        original_cards = [card for card in card_source if not str(card.get("id") or "").startswith("srv_")]
        refined_cards = [card for card in card_source if str(card.get("id") or "").startswith("srv_")]
        ordered_card_groups = [("服务端二次处理候选", refined_cards), ("端上原始证据", original_cards)]

        row_index = 0
        for group_title, grouped_cards in ordered_card_groups:
            if not grouped_cards:
                continue
            rows.append(f'<h2 class="group-title">{html.escape(group_title)}</h2>')
            for card in grouped_cards:
                row_index += 1
                index = row_index
                evidence_ids = [str(eid) for eid in card.get("evidence_ids") or [] if eid]
                if not evidence_ids and card.get("id") in evidence_by_id:
                    evidence_ids = [str(card.get("id"))]
                linked_evidence = [evidence_by_id[eid] for eid in evidence_ids if eid in evidence_by_id]
                if not linked_evidence:
                    linked_evidence = [{}]
                evidence_blocks: list[str] = []
                for evidence in linked_evidence:
                    frame = frames_by_id.get(evidence.get("best_frame_id") or "")
                    frame_src = ((frame.get("inline_url") if inline_assets else frame.get("url")) if frame else "") or ""
                    crop_src = (evidence.get("best_crop_inline_url") if inline_assets else evidence.get("best_crop_url")) or ""
                    rect_style = _rect_style(evidence.get("canonical_rect") or {})
                    flags = list(card.get("review_flags") or []) + list(evidence.get("merge_reasons") or [])
                    flag_html = "".join(f"<span>{html.escape(str(flag))}</span>" for flag in flags[:10])
                    if frame_src:
                        original = (
                            '<div class="image-stage">'
                            f'<img src="{html.escape(frame_src, quote=True)}" alt="{html.escape(str(evidence.get("best_frame_id") or ""))}">'
                        )
                        if rect_style:
                            original += f'<i class="box" style="{html.escape(rect_style, quote=True)}"></i>'
                        original += '</div>'
                    else:
                        original = '<div class="empty">无源关键帧</div>'
                    crop = (
                        f'<img class="crop-img" src="{html.escape(crop_src, quote=True)}" alt="{html.escape(str(evidence.get("id") or ""))}">'
                        if crop_src
                        else '<div class="empty">无证据图</div>'
                    )
                    evidence_blocks.append(
                        '<div class="compare-grid">'
                        f'<section><h3>原图定位</h3>{original}<p class="meta">frame {html.escape(str(evidence.get("best_frame_id") or ""))}</p></section>'
                        f'<section><h3>证据裁剪</h3>{crop}<p class="meta">{html.escape(str(evidence.get("id") or ""))} · {html.escape(str(evidence.get("crop_kind") or ""))} · seen {html.escape(str(evidence.get("seen_count") or 0))}</p></section>'
                        f'<section><h3>数字 HTML / 识别结果</h3>{_digital_iframe(run_id, card, inline_assets=inline_assets)}<p class="meta">未接 VLM 时这里是可编辑占位，不代表已识别题干。confidence {float(card.get("confidence") or 0):.2f}</p></section>'
                        f'<div class="flags">{flag_html}</div>'
                        '</div>'
                    )
                title = card.get("number") or f"候选 {index}"
                rows.append(
                    '<article class="question-row">'
                    '<header>'
                    f'<div><h2>{html.escape(str(title))}</h2><p>{html.escape(str(card.get("subject") or "unknown"))} / {html.escape(str(card.get("question_type") or "evidence"))} / {html.escape(str(card.get("id") or ""))}</p></div>'
                    f'<a href="#top">回顶部</a>'
                    '</header>'
                    f'{"".join(evidence_blocks)}'
                    '</article>'
                )

        return (
            '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(str(run.get("title") or "作业账本对比"))}</title>'
            '<style>'
            ':root{color-scheme:light;--bg:#f4f4f0;--panel:#fff;--line:#d9d6cc;--text:#1d1c19;--muted:#666156;--accent:#176c8c}'
            '*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}'
            'main{max-width:1480px;margin:0 auto;padding:18px}a{color:var(--accent);text-decoration:none}h1{font-size:24px;margin:0}h2{font-size:18px;margin:0}h3{font-size:13px;margin:0 0 8px;color:var(--muted);font-weight:650}'
            '.topbar{position:sticky;top:0;z-index:5;background:rgba(244,244,240,.94);backdrop-filter:blur(10px);border-bottom:1px solid var(--line);padding:14px 0 12px;margin-bottom:14px}'
            '.topbar p{margin:4px 0 0;color:var(--muted);font-size:13px}.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}.chips span{display:inline-flex;gap:6px;align-items:center;border:1px solid var(--line);background:var(--panel);border-radius:6px;padding:5px 8px;font-size:12px}.chips strong{color:var(--muted);font-weight:600}'
            '.frames{display:flex;gap:10px;overflow:auto;padding:4px 2px 12px;margin-bottom:10px}.frame-thumb{margin:0;min-width:180px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px}.frame-thumb img{display:block;width:100%;height:110px;object-fit:contain;background:#eee}.frame-thumb figcaption{font-size:11px;color:var(--muted);margin-top:4px}'
            '.group-title{font-size:18px;margin:20px 0 10px;padding-top:10px;border-top:2px solid var(--line)}'
            '.question-row{background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:14px 0;padding:14px}.question-row>header{display:flex;justify-content:space-between;gap:12px;align-items:start;border-bottom:1px solid #ebe8df;padding-bottom:10px;margin-bottom:12px}.question-row header p{margin:4px 0 0;color:var(--muted);font-size:12px}'
            '.compare-grid{display:grid;grid-template-columns:minmax(260px,1fr) minmax(260px,1fr) minmax(320px,1fr);gap:12px;margin-top:12px}.compare-grid section{min-width:0}.image-stage{position:relative;background:#eee;border:1px solid var(--line);border-radius:6px;overflow:hidden}.image-stage img{display:block;width:100%;height:auto}.box{position:absolute;border:3px solid #ff2f00;background:rgba(255,47,0,.12);box-shadow:0 0 0 1px rgba(255,255,255,.9) inset}.crop-img{display:block;width:100%;height:auto;max-height:360px;object-fit:contain;background:#eee;border:1px solid var(--line);border-radius:6px}.digital-frame{width:100%;height:360px;border:1px solid var(--line);border-radius:6px;background:#fff}.meta{font-size:12px;color:var(--muted);margin:7px 0 0;word-break:break-all}.flags{grid-column:1/-1;display:flex;gap:6px;flex-wrap:wrap}.flags span{background:#edf6f8;color:#25515b;border:1px solid #cde3e8;border-radius:6px;padding:3px 6px;font-size:11px}.empty{display:grid;place-items:center;min-height:180px;background:#eee;border:1px solid var(--line);border-radius:6px;color:var(--muted)}'
            '@media (max-width:980px){.compare-grid{grid-template-columns:1fr}.digital-frame{height:300px}.frame-thumb{min-width:150px}}'
            '</style></head><body><main id="top">'
            '<div class="topbar">'
            f'<h1>{html.escape(str(run.get("title") or "作业账本对比"))}</h1>'
            f'<p>run {html.escape(run_id)} · {html.escape(str(run.get("created_at") or ""))}</p>'
            f'<div class="chips">{metric_chips}</div>'
            '</div>'
            f'<section class="frames">{"".join(frame_strip) or "<p>暂无关键帧</p>"}</section>'
            f'{"".join(rows) or "<p>暂无题卡或证据。</p>"}'
            '</main></body></html>'
        )

    def _parse_vlm_cards(raw_text: str) -> list[dict]:
        text = (raw_text or "").strip()
        if not text:
            return []
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, dict):
            for key in ("questions", "cards", "items"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
        if not isinstance(parsed, list):
            return []
        return [dict(item) for item in parsed if isinstance(item, dict)]

    async def maybe_extract_refined_cards(cards: list[dict], crop_paths: list[Path]) -> list[dict]:
        if not crop_paths:
            return cards
        settings = get_settings()
        if not (getattr(settings, "grading_llm_url", "") and getattr(settings, "grading_llm_key", "")):
            return cards
        try:
            from . import llm

            prompt = (
                "请把这些作业题裁剪图转成可编辑 HTML。只输出 JSON 数组；每个元素对应一张输入图，字段："
                "input_index, number, subject, question_type, stem_text, editable_html, confidence, review_flags。"
                "几何图或地图等非文字内容不要强行矢量化，editable_html 里保留 <figure><img src=\"...\"> 占位说明。"
            )
            raw = await llm.analyze_images_responses(
                settings,
                "你是作业题目数字化助手。只输出 JSON，不要输出解释。",
                prompt,
                crop_paths,
                getattr(settings, "grading_llm_seg_effort", "low") or "low",
            )
        except Exception as exc:
            for card in cards:
                flags = _json_array(card.get("review_flags"))
                flags.append(f"vlm_failed:{_truncate(str(exc), 120)}")
                card["review_flags"] = flags
            return cards
        extracted = _parse_vlm_cards(raw)
        by_index: dict[int, dict] = {}
        for item in extracted:
            index = _int_value(item.get("input_index") or item.get("inputIndex"))
            if index:
                by_index[index] = item
        for index, card in enumerate(cards, start=1):
            item = by_index.get(index)
            if not item:
                continue
            card["number"] = _truncate(_meta_text(item, "number"), 80)
            card["subject"] = _truncate(_meta_text(item, "subject") or card.get("subject"), 80)
            card["question_type"] = _truncate(_meta_text(item, "question_type", "questionType") or card.get("question_type"), 80)
            card["stem_text"] = _truncate(_meta_text(item, "stem_text", "stemText", "stem") or card.get("stem_text"), 4000)
            html_text = _meta_text(item, "editable_html", "editableHtml")
            if html_text:
                card["editable_html"] = _truncate(html_text, HOMEWORK_LEDGER_HTML_LIMIT)
            try:
                card["confidence"] = max(float(card.get("confidence") or 0), float(item.get("confidence") or 0))
            except (TypeError, ValueError):
                pass
            flags = set(_json_array(card.get("review_flags")))
            flags.update(str(flag) for flag in _json_array(item.get("review_flags") or item.get("reviewFlags")))
            flags.discard("needs_text_extraction")
            flags.add("server_vlm_extracted")
            card["review_flags"] = sorted(flags)
        return cards

    def build_refined_manifest(run_id: str, payload: dict, *, max_frames: int, max_regions_per_frame: int) -> tuple[dict, list[Path]]:
        frames = []
        for frame in payload.get("frames") or []:
            frame_id = str(frame.get("id") or "")
            filename = str(frame.get("filename") or "").replace("\\", "/")
            if frame_id.startswith("srv_") or filename.startswith("server_refined/"):
                continue
            frames.append(frame)
        selected_frames = frames[-max_frames:] if max_frames > 0 else frames
        now = utc_now()
        frame_records: list[dict] = []
        pages: list[dict] = []
        evidence: list[dict] = []
        cards: list[dict] = []
        crop_paths: list[Path] = []
        output_dir = run_dir(run_id)
        refined_dir = (output_dir / "server_refined").resolve()
        if refined_dir.exists() and refined_dir.is_relative_to(output_dir.resolve()):
            shutil.rmtree(refined_dir)
        sequence = 0

        for frame in selected_frames:
            rel = str(frame.get("filename") or "").replace("\\", "/")
            if not rel:
                continue
            source_path = asset_file_path(run_id, rel)
            with Image.open(source_path) as raw:
                rotated, rotation = _rotate_for_reading(raw)
            bbox = _content_bbox(rotated)
            if bbox:
                workbook_image = rotated.crop(bbox)
                workbook_bbox = {
                    "x": bbox[0] / rotated.width,
                    "y": bbox[1] / rotated.height,
                    "width": (bbox[2] - bbox[0]) / rotated.width,
                    "height": (bbox[3] - bbox[1]) / rotated.height,
                }
            else:
                workbook_image = rotated
                workbook_bbox = {"x": 0, "y": 0, "width": 1, "height": 1}

            for page_name, page_image, split_bbox in _split_open_book_pages(workbook_image):
                page_image, page_rectified, page_rectify_confidence = _normalize_page_perspective(page_image, max_side=1800)
                page_id = f"srv_page_{frame.get('id') or len(pages) + 1}_{page_name}"
                frame_id = f"srv_frame_{frame.get('id') or len(frame_records) + 1}_{page_name}"
                page_rel = f"server_refined/pages/{page_id}.jpg"
                page_bytes = _save_jpeg(page_image, output_dir / page_rel, quality=88)
                metrics = _image_metrics(page_image)
                frame_records.append(
                    {
                        "id": frame_id,
                        "sequence_index": len(frame_records) + 1,
                        "filename": page_rel,
                        "original_name": rel,
                        "width": page_image.width,
                        "height": page_image.height,
                        "normalized_bytes": page_bytes,
                        "accepted": True,
                        "reason": "server_refined_page_split",
                        "page_episode_id": page_id,
                        "frame_meta": {
                            "source_frame_id": frame.get("id"),
                            "rotation_degrees": rotation,
                            "workbook_bbox_after_rotation": workbook_bbox,
                            "page_split_bbox": split_bbox,
                            "page_name": page_name,
                            "page_rectified": page_rectified,
                            "page_rectify_confidence": round(float(page_rectify_confidence), 4),
                            **metrics,
                        },
                    }
                )
                pages.append(
                    {
                        "id": page_id,
                        "first_frame_id": frame_id,
                        "last_frame_id": frame_id,
                        "frame_ids": [frame_id],
                        "fingerprint": f"server_refined:{frame.get('id')}:{page_name}",
                        "coverage_cells": [],
                        "payload": {
                            "source_frame_id": frame.get("id"),
                            "rotation_degrees": rotation,
                            "workbook_bbox_after_rotation": workbook_bbox,
                            "page_split_bbox": split_bbox,
                            "page_name": page_name,
                            "page_rectified": page_rectified,
                            "page_rectify_confidence": round(float(page_rectify_confidence), 4),
                        },
                    }
                )

                regions = _detect_page_block_regions(page_image, max_regions=max(1, max_regions_per_frame))
                for region in regions:
                    rect = region["rect"]
                    crop = _crop_rect(page_image, rect)
                    paper_score = _paper_score(crop)
                    if paper_score < 0.10 and region.get("kind") == "server_page_block":
                        continue
                    sequence += 1
                    evidence_id = f"srv_qev_{sequence:04d}"
                    card_id = f"srv_card_{sequence:04d}"
                    crop_rel = f"server_refined/crops/{evidence_id}.jpg"
                    _save_jpeg(crop, output_dir / crop_rel, quality=92)
                    crop_paths.append(output_dir / crop_rel)
                    quality = _image_metrics(crop)
                    quality["paper_score"] = round(paper_score, 4)
                    confidence = 0.58 if region.get("kind") == "server_page_block" else 0.42
                    evidence.append(
                        {
                            "id": evidence_id,
                            "page_episode_id": page_id,
                            "best_frame_id": frame_id,
                            "best_crop_filename": crop_rel,
                            "crop_kind": region.get("kind") or "server_page_block",
                            "canonical_rect": rect,
                            "crop_hash": "",
                            "layout_key": f"server:{page_id}:{round(rect['y'], 2)}:{round(rect['height'], 2)}",
                            "ocr_key": "",
                            "seen_count": 1,
                            "status": "ready",
                            "quality": {**quality, "confidence": confidence, "area": round(rect["width"] * rect["height"], 5)},
                            "source_frames": [str(frame.get("id") or "")],
                            "merge_reasons": ["server_refined", "rotated_page" if rotation else "page_crop", "page_split", "coarse_question_block"],
                        }
                    )
                    cards.append(
                        {
                            "id": card_id,
                            "card_key": evidence_id,
                            "evidence_ids": [evidence_id],
                            "number": "",
                            "subject": "unknown",
                            "question_type": "server_page_block",
                            "stem_text": "Server refined page block. VLM extraction pending.",
                            "editable_html": (
                                "<article><p>Server refined page block. VLM extraction pending; edit here if needed.</p>"
                                f'<figure><img src="{html.escape(crop_rel)}" alt="{html.escape(evidence_id)}"></figure></article>'
                            ),
                            "figure_assets": [{"filename": crop_rel, "source_evidence_id": evidence_id, "type": "raster"}],
                            "confidence": confidence,
                            "review_flags": ["server_refined", "page_split", "coarse_question_block", "needs_text_extraction", "raster_figure_evidence"],
                        }
                    )

        evidence, cards, crop_paths, duplicate_count, blank_count = _dedupe_refined_candidates(evidence, cards, crop_paths)
        for index, (item, card, path) in enumerate(zip(evidence, cards, crop_paths), start=1):
            evidence_id = f"srv_qev_{index:04d}"
            card_id = f"srv_card_{index:04d}"
            target_rel = f"server_refined/crops/{evidence_id}.jpg"
            target_path = output_dir / target_rel
            if path.resolve() != target_path.resolve():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                path.replace(target_path)
            item["id"] = evidence_id
            item["best_crop_filename"] = target_rel
            item["layout_key"] = f"server:{_server_page_side(item.get('page_episode_id'))}:{round(_rect_center_y(_json_object(item.get('canonical_rect'))), 2)}"
            card["id"] = card_id
            card["card_key"] = evidence_id
            card["evidence_ids"] = [evidence_id]
            card["figure_assets"] = [{"filename": target_rel, "source_evidence_id": evidence_id, "type": "raster"}]
            card["editable_html"] = (
                "<article><p>Server refined page block. VLM extraction pending; edit here if needed.</p>"
                f'<figure><img src="{html.escape(target_rel)}" alt="{html.escape(evidence_id)}"></figure></article>'
            )
            crop_paths[index - 1] = target_path

        metrics = dict(payload.get("run", {}).get("metrics") or {})
        metrics.update(
            {
                "server_refined_at": now,
                "server_refined_frames": len(frame_records),
                "server_refined_evidence": len(evidence),
                "server_refined_cards": len(cards),
                "server_refined_duplicates_merged": duplicate_count,
                "server_refined_low_info_dropped": blank_count,
            }
        )
        return {
            "metrics": metrics,
            "frames": frame_records,
            "page_episodes": pages,
            "question_evidence": evidence,
            "question_cards": cards,
            "source": "server_refined_homework_ledger",
            "simulator_version": "server-refine-v0",
        }, crop_paths

    @app.post("/api/homework-ledger/runs")
    async def create_homework_ledger_run(request: Request) -> dict:
        init_db()
        principal = principal_from_request(request)
        body = await request.json()
        run_id = _truncate(str(body.get("run_id") or body.get("id") or uuid.uuid4().hex), 80)
        title = clean_user_text(body.get("title") or "Homework Evidence Ledger", 160)
        device_id = clean_user_text(body.get("device_id") or body.get("deviceId") or "iphone", 120)
        student_profile_id = resolve_student_profile(principal["account_id"], str(body.get("student_profile_id") or body.get("studentProfileId") or ""))
        now = utc_now()
        with connect() as conn:
            existing = conn.execute("SELECT * FROM homework_ledger_runs WHERE id=?", (run_id,)).fetchone()
            if existing:
                require_run(conn, run_id, principal)
                conn.execute(
                    """
                    UPDATE homework_ledger_runs
                    SET title=?, device_id=?, student_profile_id=?, updated_at=?
                    WHERE id=?
                    """,
                    (title, device_id, student_profile_id, now, run_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO homework_ledger_runs(
                        id, account_id, created_by_user_id, student_profile_id, device_id,
                        title, status, metrics, client_summary, created_at, updated_at
                    )
                    VALUES(?, ?, ?, ?, ?, ?, 'running', '{}', '{}', ?, ?)
                    """,
                    (run_id, principal["account_id"], principal.get("user_id", ""), student_profile_id, device_id, title, now, now),
                )
            run = require_run(conn, run_id, principal)
        run_dir(run_id)
        emit_log(f"created homework ledger: {title}", device_id=device_id, source="homework-ledger")
        return {"run_id": run_id, "run": dict(run)}

    @app.get("/api/homework-ledger/runs")
    def list_homework_ledger_runs(request: Request, limit: int = 50) -> dict:
        init_db()
        principal = principal_from_request(request)
        safe_limit = max(1, min(100, int(limit or 50)))
        with connect() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT
                        runs.*,
                        (SELECT COUNT(*) FROM homework_ledger_frames WHERE run_id=runs.id) AS frame_count,
                        (SELECT COUNT(*) FROM homework_ledger_evidence WHERE run_id=runs.id) AS evidence_count,
                        (SELECT COUNT(*) FROM homework_ledger_cards WHERE run_id=runs.id) AS card_count
                    FROM homework_ledger_runs runs
                    WHERE account_id=?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (principal["account_id"], safe_limit),
                )
            ]
        for row in rows:
            row["metrics"] = _json_object(row.get("metrics"))
            row["client_summary"] = _json_object(row.get("client_summary"))
        return {"runs": rows}

    @app.post("/api/homework-ledger/runs/{run_id}/sync")
    async def sync_homework_ledger_run(
        run_id: str,
        request: Request,
        frames: list[UploadFile] | None = File(None),
        crops: list[UploadFile] | None = File(None),
        manifest: str = Form("{}"),
    ) -> dict:
        init_db()
        principal = principal_from_request(request)
        payload = _parse_manifest(manifest)
        now = utc_now()
        with connect() as conn:
            require_run(conn, run_id, principal)
        frame_uploads = _file_map(frames)
        crop_uploads = _file_map(crops)
        saved_frames = 0
        saved_crops = 0
        for index, item in enumerate(_manifest_list(payload, "frames", "frame_records"), start=1):
            upload_ref = _upload_ref(
                item,
                "upload_ref",
                "uploadRef",
                "filename",
                "normalized_filename",
                "normalizedFilename",
                "normalized_path",
                "normalizedPath",
                fallback=str(index - 1),
            )
            upload = frame_uploads.get(upload_ref) or frame_uploads.get(Path(upload_ref).name)
            if upload is None:
                continue
            frame_id = _frame_id(item, index)
            rel_hint = _upload_ref(
                item,
                "filename",
                "normalized_filename",
                "normalizedFilename",
                "normalized_path",
                "normalizedPath",
                fallback=f"frames/{frame_id}.jpg",
            )
            saved_rel, saved_bytes = await save_upload(upload, run_id=run_id, rel_path=rel_hint, default_folder="frames", fallback_prefix=frame_id)
            item["filename"] = saved_rel
            item["normalized_bytes"] = item.get("normalized_bytes") or item.get("normalizedBytes") or saved_bytes
            saved_frames += 1
        for index, item in enumerate(_manifest_list(payload, "question_evidence", "evidence", "questionEvidence"), start=1):
            upload_ref = _upload_ref(
                item,
                "upload_ref",
                "uploadRef",
                "best_crop_filename",
                "bestCropFilename",
                "crop_filename",
                "cropFilename",
                fallback=str(index - 1),
            )
            upload = crop_uploads.get(upload_ref) or crop_uploads.get(Path(upload_ref).name)
            if upload is None:
                continue
            evidence_id = _evidence_id(item, index)
            rel_hint = _upload_ref(
                item,
                "best_crop_filename",
                "bestCropFilename",
                "crop_filename",
                "cropFilename",
                fallback=f"crops/{evidence_id}.jpg",
            )
            saved_rel, _ = await save_upload(upload, run_id=run_id, rel_path=rel_hint, default_folder="crops", fallback_prefix=evidence_id)
            item["best_crop_filename"] = saved_rel
            saved_crops += 1
        for card in _manifest_list(payload, "question_cards", "cards", "questionCards"):
            for asset in _json_array(card.get("figure_assets") or card.get("figureAssets")):
                if isinstance(asset, dict) and asset.get("filename"):
                    asset["filename"] = _safe_asset_rel(str(asset.get("filename")), "crops", Path(str(asset.get("filename"))).name)
        with connect() as conn:
            require_run(conn, run_id, principal)
            counts = upsert_manifest(conn, run_id, payload, now)
            run = require_run(conn, run_id, principal)
        return {"run_id": run_id, "saved_frames": saved_frames, "saved_crops": saved_crops, "manifest_counts": counts, "run": dict(run)}

    @app.post("/api/homework-ledger/runs/{run_id}/finish")
    async def finish_homework_ledger_run(run_id: str, request: Request) -> dict:
        init_db()
        principal = principal_from_request(request)
        body = await request.json()
        now = utc_now()
        metrics = _json_object(body.get("metrics"))
        with connect() as conn:
            require_run(conn, run_id, principal)
            if metrics:
                conn.execute(
                    "UPDATE homework_ledger_runs SET status='completed', metrics=?, finished_at=?, updated_at=? WHERE id=?",
                    (json_dumps(metrics), now, now, run_id),
                )
            else:
                conn.execute(
                    "UPDATE homework_ledger_runs SET status='completed', finished_at=?, updated_at=? WHERE id=?",
                    (now, now, run_id),
                )
            run = require_run(conn, run_id, principal)
        emit_log(f"finished homework ledger: {run_id}", device_id=run.get("device_id") or "", source="homework-ledger")
        return {"run_id": run_id, "status": run["status"], "run": dict(run)}

    @app.get("/api/homework-ledger/runs/{run_id}")
    def get_homework_ledger_run(run_id: str, request: Request) -> dict:
        init_db()
        principal = principal_from_request(request)
        return payload_for(run_id, principal)

    @app.get("/api/homework-ledger/runs/{run_id}/html", response_class=HTMLResponse)
    def get_homework_ledger_html(run_id: str, request: Request) -> str:
        init_db()
        principal = principal_from_request(request)
        return render_html(payload_for(run_id, principal))

    @app.get("/api/homework-ledger/runs/latest/compare", response_class=HTMLResponse)
    def get_latest_homework_ledger_compare(request: Request) -> str:
        init_db()
        principal = principal_from_request(request)
        return render_compare_html(payload_for(latest_run_id(principal), principal))

    @app.get("/api/homework-ledger/runs/{run_id}/compare", response_class=HTMLResponse)
    def get_homework_ledger_compare(run_id: str, request: Request) -> str:
        init_db()
        principal = principal_from_request(request)
        return render_compare_html(payload_for(run_id, principal))

    @app.post("/api/homework-ledger/runs/{run_id}/refine")
    async def refine_homework_ledger_run(run_id: str, request: Request, max_frames: int = 4, max_regions_per_frame: int = 6, use_vlm: bool = False) -> dict:
        init_db()
        principal = principal_from_request(request)
        payload = payload_for(run_id, principal)
        max_frames = max(1, min(12, int(max_frames or 4)))
        max_regions_per_frame = max(1, min(12, int(max_regions_per_frame or 6)))
        manifest, crop_paths = build_refined_manifest(run_id, payload, max_frames=max_frames, max_regions_per_frame=max_regions_per_frame)
        if use_vlm:
            manifest["question_cards"] = await maybe_extract_refined_cards(manifest["question_cards"], crop_paths)
        now = utc_now()
        with connect() as conn:
            require_run(conn, run_id, principal)
            conn.execute("DELETE FROM homework_ledger_cards WHERE run_id=? AND id LIKE 'srv_%'", (run_id,))
            conn.execute("DELETE FROM homework_ledger_evidence WHERE run_id=? AND id LIKE 'srv_%'", (run_id,))
            conn.execute("DELETE FROM homework_ledger_page_episodes WHERE run_id=? AND id LIKE 'srv_%'", (run_id,))
            conn.execute("DELETE FROM homework_ledger_frames WHERE run_id=? AND id LIKE 'srv_%'", (run_id,))
            counts = upsert_manifest(conn, run_id, manifest, now)
            run = require_run(conn, run_id, principal)
        emit_log(
            f"refined homework ledger: {run_id}",
            device_id=run.get("device_id") or "",
            source="homework-ledger-refine",
        )
        return {
            "run_id": run_id,
            "manifest_counts": counts,
            "server_refined_frames": len(manifest.get("frames") or []),
            "server_refined_evidence": len(manifest.get("question_evidence") or []),
            "server_refined_cards": len(manifest.get("question_cards") or []),
            "compare_url": f"/api/homework-ledger/runs/{quote(run_id, safe='')}/compare",
            "run": dict(run),
        }

    @app.get("/api/homework-ledger/runs/{run_id}/assets/{asset_path:path}")
    def get_homework_ledger_asset(run_id: str, asset_path: str, request: Request) -> FileResponse:
        init_db()
        principal = principal_from_request(request)
        with connect() as conn:
            require_run(conn, run_id, principal)
        path = asset_file_path(run_id, asset_path)
        media_type = "image/jpeg"
        if path.suffix.lower() == ".png":
            media_type = "image/png"
        elif path.suffix.lower() == ".webp":
            media_type = "image/webp"
        return FileResponse(path, media_type=media_type, headers={"Cache-Control": "public, max-age=31536000, immutable"})
