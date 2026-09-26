"""Replay question-crop payload strategies on historical images.

This script estimates the product tradeoff the app cares about:

- full keyframes only;
- keyframes plus crop JPEGs;
- keyframes plus rect-only manifests, with backend-side canonical crops;
- cross-frame dedupe before backend/VLM work.

It can read existing diagnostics, draft/approved prelabels, or exported detector
datasets. Optionally it reruns the lightweight layout prelabeler to measure local
image-processing latency on the same historical images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from question_crop_benchmark import (
    Rect,
    clamp_rect,
    crop_image,
    jpeg_upload_bytes,
    rect_only_v3_server_crop,
)
from question_detector_prelabel import draft_boxes_for_image, load_rgb


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ReplayBox:
    image_key: str
    image_path: Path
    source: str
    rect: Rect
    score: float
    question_key: str
    quality_flags: list[str]
    source_row: dict[str, Any]

    @property
    def area(self) -> float:
        return max(0.0, self.rect["width"]) * max(0.0, self.rect["height"])


@dataclass
class ReplayImage:
    image_key: str
    image_path: Path
    source: str
    width: int
    height: int
    split_key: str
    metadata: dict[str, Any]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def stable_id(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_average_hash(path: Path, hash_size: int = 8) -> str:
    try:
        with Image.open(path) as image:
            gray = ImageOps.exif_transpose(image).convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    except Exception:
        return ""
    pixels = list(gray.getdata())
    if not pixels:
        return ""
    mean = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | (1 if pixel >= mean else 0)
    return f"{value:0{hash_size * hash_size // 4}x}"


def resized_pixel_count(width: int, height: int, max_side: int) -> int:
    longest = max(width, height, 1)
    scale = min(1.0, max_side / longest)
    return max(1, int(width * scale)) * max(1, int(height * scale))


def box_iou(a: Rect, b: Rect) -> float:
    ax2 = a["x"] + a["width"]
    ay2 = a["y"] + a["height"]
    bx2 = b["x"] + b["width"]
    by2 = b["y"] + b["height"]
    overlap_w = max(0.0, min(ax2, bx2) - max(a["x"], b["x"]))
    overlap_h = max(0.0, min(ay2, by2) - max(a["y"], b["y"]))
    overlap = overlap_w * overlap_h
    if overlap <= 0:
        return 0.0
    union = max(0.000001, a["width"] * a["height"] + b["width"] * b["height"] - overlap)
    return overlap / union


def rect_to_px(rect: Rect, width: int, height: int) -> tuple[int, int, int, int]:
    x = max(0, min(width - 1, int(round(rect["x"] * width))))
    y = max(0, min(height - 1, int(round(rect["y"] * height))))
    w = max(1, min(width - x, int(round(rect["width"] * width))))
    h = max(1, min(height - y, int(round(rect["height"] * height))))
    return x, y, w, h


def px_to_rect(x: float, y: float, width: float, height: float, image_width: int, image_height: int) -> Rect:
    return clamp_rect(
        {
            "x": x / max(1, image_width),
            "y": y / max(1, image_height),
            "width": width / max(1, image_width),
            "height": height / max(1, image_height),
        }
    )


def resolve_image(root: Path, image_value: str) -> Path:
    raw = Path(str(image_value or ""))
    if raw.is_absolute():
        return raw
    candidates = [
        root / raw,
        root / "images" / raw.name,
        root.parent / raw,
        root.parent / "images" / raw.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).size
    except Exception:
        return None


def add_image(images: dict[str, ReplayImage], path: Path, source: str, split_key: str = "", metadata: dict[str, Any] | None = None) -> str | None:
    size = image_size(path)
    if size is None:
        return None
    try:
        key = file_sha1(path)
    except OSError:
        key = stable_id(str(path.resolve()))
    if key not in images:
        images[key] = ReplayImage(
            image_key=key,
            image_path=path,
            source=source,
            width=size[0],
            height=size[1],
            split_key=split_key,
            metadata=metadata or {},
        )
    return key


def load_prelabel_root(root: Path, include_quarantine: bool) -> tuple[dict[str, ReplayImage], list[ReplayBox], dict[str, Any]]:
    images: dict[str, ReplayImage] = {}
    boxes: list[ReplayBox] = []
    source_files = [root / "annotations" / "draft_boxes.jsonl"]
    if include_quarantine:
        source_files.append(root / "annotations" / "quarantine.jsonl")
    for source_file in source_files:
        for row in read_jsonl(source_file):
            image_path = resolve_image(root, str(row.get("image") or ""))
            split_key = str((row.get("candidate") or {}).get("session_id") or row.get("source_candidate") or "")
            image_key = add_image(images, image_path, f"prelabel:{source_file}", split_key, row.get("metadata") or {})
            if image_key is None:
                continue
            image = images[image_key]
            for index, box in enumerate(row.get("boxes") or []):
                bbox = box.get("bbox_px") or {}
                try:
                    rect = px_to_rect(
                        float(bbox.get("x")),
                        float(bbox.get("y")),
                        float(bbox.get("width")),
                        float(bbox.get("height")),
                        image.width,
                        image.height,
                    )
                except (TypeError, ValueError):
                    continue
                boxes.append(
                    ReplayBox(
                        image_key=image_key,
                        image_path=image_path,
                        source="prelabel_box",
                        rect=rect,
                        score=float(box.get("score") or 0.0),
                        question_key=str(box.get("question_key") or f"{image_key}:{index + 1}"),
                        quality_flags=[str(flag) for flag in box.get("quality_flags") or []],
                        source_row=row,
                    )
                )
    return images, boxes, {"input_kind": "prelabel", "root": str(root)}


def load_dataset_root(root: Path) -> tuple[dict[str, ReplayImage], list[ReplayBox], dict[str, Any]]:
    images: dict[str, ReplayImage] = {}
    boxes: list[ReplayBox] = []
    image_manifest = read_jsonl(root / "annotations" / "image_manifest.jsonl")
    image_key_by_rel: dict[str, str] = {}
    for row in image_manifest:
        image_path = resolve_image(root, str(row.get("image") or ""))
        image_key = add_image(
            images,
            image_path,
            str(row.get("origin") or "dataset"),
            str(row.get("split_key") or ""),
            row,
        )
        if image_key is not None:
            image_key_by_rel[str(row.get("image") or "")] = image_key
    for row in read_jsonl(root / "annotations" / "manifest.jsonl"):
        rel = str(row.get("image") or "")
        image_key = image_key_by_rel.get(rel)
        if image_key is None:
            image_path = resolve_image(root, rel)
            image_key = add_image(images, image_path, str(row.get("origin") or "dataset"), str(row.get("split_key") or ""), row)
        if image_key is None:
            continue
        rect = clamp_rect(row.get("bbox_norm") or {})
        boxes.append(
            ReplayBox(
                image_key=image_key,
                image_path=images[image_key].image_path,
                source="dataset_label",
                rect=rect,
                score=float(row.get("confidence") or 1.0),
                question_key=str(row.get("question_key") or f"{image_key}:{len(boxes) + 1}"),
                quality_flags=[str(flag) for flag in row.get("quality_flags") or []],
                source_row=row,
            )
        )
    return images, boxes, {"input_kind": "dataset", "root": str(root)}


def load_diagnostics_root(root: Path) -> tuple[dict[str, ReplayImage], list[ReplayBox], dict[str, Any]]:
    images: dict[str, ReplayImage] = {}
    boxes: list[ReplayBox] = []
    payload = read_json(root / "manifest.json")
    rows = payload.get("manifest") or payload.get("crops") or payload.get("items") or []
    images_dir = root / "images"
    for index, row in enumerate(rows):
        source_name = str(row.get("src_filename") or row.get("source_filename") or "")
        if not source_name:
            continue
        image_path = images_dir / Path(source_name).name
        image_key = add_image(images, image_path, f"diagnostics:{root}", str(row.get("session_id") or row.get("batch_id") or ""), row)
        if image_key is None:
            continue
        image = images[image_key]
        rect = rect_only_v3_server_crop(row, (image.width, image.height))
        boxes.append(
            ReplayBox(
                image_key=image_key,
                image_path=image_path,
                source="diagnostics_v3_rect",
                rect=rect,
                score=float(row.get("confidence") or 0.0),
                question_key=str(row.get("question_key") or f"{image_key}:{index + 1}"),
                quality_flags=[],
                source_row=row,
            )
        )
    return images, boxes, {"input_kind": "diagnostics", "root": str(root)}


def load_image_dir(root: Path) -> tuple[dict[str, ReplayImage], list[ReplayBox], dict[str, Any]]:
    images: dict[str, ReplayImage] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES and path.is_file():
            add_image(images, path, f"image_dir:{root}")
    return images, [], {"input_kind": "image_dir", "root": str(root)}


def merge_inputs(args: argparse.Namespace) -> tuple[dict[str, ReplayImage], list[ReplayBox], list[dict[str, Any]]]:
    merged_images: dict[str, ReplayImage] = {}
    merged_boxes: list[ReplayBox] = []
    input_summaries: list[dict[str, Any]] = []

    def merge(images: dict[str, ReplayImage], boxes: list[ReplayBox], summary: dict[str, Any]) -> None:
        merged_images.update(images)
        merged_boxes.extend(boxes)
        input_summaries.append({**summary, "images": len(images), "boxes": len(boxes)})

    for root in args.prelabel_root:
        merge(*load_prelabel_root(root, args.include_quarantine))
    for root in args.dataset_root:
        merge(*load_dataset_root(root))
    for root in args.diagnostics:
        merge(*load_diagnostics_root(root))
    for root in args.image_dir:
        merge(*load_image_dir(root))
    if args.max_images > 0:
        allowed = set(sorted(merged_images, key=lambda key: key)[: args.max_images])
        merged_images = {key: value for key, value in merged_images.items() if key in allowed}
        merged_boxes = [box for box in merged_boxes if box.image_key in allowed]
    return merged_images, merged_boxes, input_summaries


def run_layout_measurement(images: dict[str, ReplayImage], max_side: int) -> tuple[list[ReplayBox], dict[str, Any]]:
    measured_boxes: list[ReplayBox] = []
    latencies: list[float] = []
    failed = 0
    quarantined = 0
    for image in images.values():
        rgb = load_rgb(image.image_path)
        if rgb is None:
            failed += 1
            continue
        started = time.perf_counter()
        boxes, metadata = draft_boxes_for_image(rgb, max_side)
        latency = (time.perf_counter() - started) * 1000
        latencies.append(latency)
        if metadata.get("quarantine_reasons"):
            quarantined += 1
        for index, box in enumerate(boxes):
            rect = px_to_rect(box.x, box.y, box.width, box.height, image.width, image.height)
            measured_boxes.append(
                ReplayBox(
                    image_key=image.image_key,
                    image_path=image.image_path,
                    source="layout_replay_box",
                    rect=rect,
                    score=box.score,
                    question_key=f"layout:{image.image_key}:{index + 1}",
                    quality_flags=box.flags,
                    source_row={"metadata": metadata},
                )
            )
    return measured_boxes, {
        "attempted_images": len(images),
        "failed_images": failed,
        "quarantined_images": quarantined,
        "box_count": len(measured_boxes),
        "latency_ms": stats(latencies),
    }


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "total": 0, "mean": 0, "median": 0, "p90": 0, "p95": 0, "max": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "total": round(sum(values), 3),
        "mean": round(sum(values) / len(values), 3),
        "median": round(statistics.median(values), 3),
        "p90": round(percentile(ordered, 0.90), 3),
        "p95": round(percentile(ordered, 0.95), 3),
        "max": round(max(values), 3),
    }


def percentile(ordered: list[float], p: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[index]


def dedupe_within_image(boxes: list[ReplayBox], iou_threshold: float) -> list[ReplayBox]:
    selected: list[ReplayBox] = []
    grouped: dict[str, list[ReplayBox]] = defaultdict(list)
    for box in boxes:
        grouped[box.image_key].append(box)
    for group in grouped.values():
        for box in sorted(group, key=lambda item: (item.score, item.area), reverse=True):
            if any(box_iou(box.rect, existing.rect) >= iou_threshold for existing in selected if existing.image_key == box.image_key):
                continue
            selected.append(box)
    return sorted(selected, key=lambda item: (item.image_key, item.rect["y"], item.rect["x"]))


def image_hashes(images: dict[str, ReplayImage]) -> dict[str, str]:
    return {key: image_average_hash(image.image_path) for key, image in images.items()}


def hex_hamming(left: str, right: str) -> int:
    if not left or not right:
        return 999
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return 999


def cross_frame_dedupe(boxes: list[ReplayBox], images: dict[str, ReplayImage], ahash_threshold: int, box_iou_threshold: float) -> list[ReplayBox]:
    hashes = image_hashes(images)
    selected: list[ReplayBox] = []
    for box in sorted(boxes, key=lambda item: (item.score, item.area), reverse=True):
        duplicate = False
        for existing in selected:
            if box.question_key and existing.question_key and box.question_key == existing.question_key:
                duplicate = True
                break
            distance = hex_hamming(hashes.get(box.image_key, ""), hashes.get(existing.image_key, ""))
            if distance <= ahash_threshold and box_iou(box.rect, existing.rect) >= box_iou_threshold:
                duplicate = True
                break
        if not duplicate:
            selected.append(box)
    return sorted(selected, key=lambda item: (images[item.image_key].split_key, item.image_key, item.rect["y"], item.rect["x"]))


def rect_manifest_bytes(boxes: list[ReplayBox]) -> int:
    payload = {
        "version": 2,
        "source": "offline-observation-replay",
        "crops": [
            {
                "source_image_key": box.image_key[:16],
                "question_key": box.question_key,
                "crop_rect": {key: round(value, 6) for key, value in box.rect.items()},
                "crop_area": round(box.area, 6),
                "transfer_mode": "rect_only",
                "crop_prepared": False,
            }
            for box in boxes
        ],
    }
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def evaluate_payload(
    images: dict[str, ReplayImage],
    boxes: list[ReplayBox],
    in_image_boxes: list[ReplayBox],
    dedup_boxes: list[ReplayBox],
    full_max_side: int,
    crop_max_side: int,
    full_quality: int,
    crop_quality: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    full_jpeg_ms: list[float] = []
    crop_jpeg_ms: list[float] = []
    details: list[dict[str, Any]] = []
    full_bytes_by_image: dict[str, int] = {}
    full_pixels_by_image: dict[str, int] = {}
    crop_bytes_by_box: dict[int, int] = {}
    crop_pixels_by_box: dict[int, int] = {}

    opened: dict[str, Image.Image] = {}
    for key, image in images.items():
        with Image.open(image.image_path) as raw:
            pil = ImageOps.exif_transpose(raw).convert("RGB")
        opened[key] = pil
        started = time.perf_counter()
        full_bytes_by_image[key] = jpeg_upload_bytes(pil, max_side=full_max_side, quality=full_quality)
        full_jpeg_ms.append((time.perf_counter() - started) * 1000)
        full_pixels_by_image[key] = resized_pixel_count(image.width, image.height, full_max_side)

    for index, box in enumerate(boxes):
        image = images[box.image_key]
        pil = opened[box.image_key]
        crop = crop_image(pil, box.rect)
        started = time.perf_counter()
        crop_bytes = jpeg_upload_bytes(crop, max_side=crop_max_side, quality=crop_quality)
        crop_jpeg_ms.append((time.perf_counter() - started) * 1000)
        crop_bytes_by_box[index] = crop_bytes
        crop_pixels_by_box[index] = resized_pixel_count(crop.width, crop.height, crop_max_side)
        x, y, width, height = rect_to_px(box.rect, image.width, image.height)
        details.append(
            {
                "image": str(image.image_path),
                "image_key": box.image_key,
                "source": box.source,
                "bbox_px": {"x": x, "y": y, "width": width, "height": height},
                "bbox_norm": {key: round(value, 6) for key, value in box.rect.items()},
                "area": round(box.area, 6),
                "score": box.score,
                "question_key": box.question_key,
                "quality_flags": box.quality_flags,
                "crop_jpeg_bytes": crop_bytes,
                "crop_resized_pixels": crop_pixels_by_box[index],
            }
        )

    box_index = {id(box): index for index, box in enumerate(boxes)}

    def sum_crop_bytes(selected: list[ReplayBox]) -> int:
        return sum(crop_bytes_by_box.get(box_index.get(id(box), -1), 0) for box in selected)

    def sum_crop_pixels(selected: list[ReplayBox]) -> int:
        return sum(crop_pixels_by_box.get(box_index.get(id(box), -1), 0) for box in selected)

    full_total = sum(full_bytes_by_image.values())
    full_pixels_total = sum(full_pixels_by_image.values())
    raw_crop_total = sum_crop_bytes(boxes)
    in_image_crop_total = sum_crop_bytes(in_image_boxes)
    dedup_crop_total = sum_crop_bytes(dedup_boxes)
    raw_rect_bytes = rect_manifest_bytes(boxes)
    in_image_rect_bytes = rect_manifest_bytes(in_image_boxes)
    dedup_rect_bytes = rect_manifest_bytes(dedup_boxes)
    dedup_crop_pixels = sum_crop_pixels(dedup_boxes)
    summary = {
        "counts": {
            "images": len(images),
            "raw_boxes": len(boxes),
            "in_image_dedup_boxes": len(in_image_boxes),
            "cross_frame_dedup_boxes": len(dedup_boxes),
            "in_image_removed_boxes": len(boxes) - len(in_image_boxes),
            "cross_frame_removed_boxes": len(in_image_boxes) - len(dedup_boxes),
            "images_with_boxes": len({box.image_key for box in boxes}),
        },
        "upload_bytes": {
            "full_frame_only": full_total,
            "crop_jpeg_extra_raw": raw_crop_total,
            "crop_jpeg_extra_in_image_dedup": in_image_crop_total,
            "crop_jpeg_extra_cross_frame_dedup": dedup_crop_total,
            "rect_manifest_raw": raw_rect_bytes,
            "rect_manifest_in_image_dedup": in_image_rect_bytes,
            "rect_manifest_cross_frame_dedup": dedup_rect_bytes,
            "full_plus_raw_crop_vs_full": round((full_total + raw_crop_total) / max(1, full_total), 4),
            "full_plus_cross_frame_dedup_crop_vs_full": round((full_total + dedup_crop_total) / max(1, full_total), 4),
            "rect_only_raw_vs_full": round((full_total + raw_rect_bytes) / max(1, full_total), 4),
            "rect_only_cross_frame_dedup_vs_full": round((full_total + dedup_rect_bytes) / max(1, full_total), 4),
            "dedup_crop_extra_vs_full": round(dedup_crop_total / max(1, full_total), 4),
            "dedup_rect_extra_vs_full": round(dedup_rect_bytes / max(1, full_total), 6),
        },
        "vlm_proxy": {
            "full_frame_input_count": len(images),
            "crop_input_count_after_dedup": len(dedup_boxes),
            "full_frame_resized_pixels": full_pixels_total,
            "crop_resized_pixels_after_dedup": dedup_crop_pixels,
            "crop_pixels_vs_full_pixels": round(dedup_crop_pixels / max(1, full_pixels_total), 4),
        },
        "latency_ms": {
            "full_frame_jpeg": stats(full_jpeg_ms),
            "crop_jpeg": stats(crop_jpeg_ms),
        },
        "quality_proxy": {
            "near_full_page_boxes": sum(1 for box in boxes if box.area >= 0.72),
            "small_area_boxes": sum(1 for box in boxes if box.area < 0.025),
            "flag_counts": flag_counts(boxes),
        },
    }
    return summary, details


def flag_counts(boxes: list[ReplayBox]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for box in boxes:
        for flag in box.quality_flags:
            counts[flag] += 1
    return dict(sorted(counts.items()))


def replay_image_rows(images: dict[str, ReplayImage]) -> list[dict[str, Any]]:
    return [
        {
            "image": str(image.image_path),
            "image_key": image.image_key,
            "source": image.source,
            "width": image.width,
            "height": image.height,
            "split_key": image.split_key,
            "source_filename": image.image_path.name,
            "metadata": image.metadata,
        }
        for image in sorted(images.values(), key=lambda item: (item.split_key, item.image_key))
    ]


def build_contact_sheet(out_dir: Path, images: dict[str, ReplayImage], boxes: list[ReplayBox], filename: str, limit: int = 60) -> None:
    grouped: dict[str, list[ReplayBox]] = defaultdict(list)
    for box in boxes:
        grouped[box.image_key].append(box)
    tiles: list[Image.Image] = []
    colors = [(40, 170, 80), (225, 120, 40), (60, 120, 220), (190, 70, 170)]
    for image_key in list(grouped.keys())[:limit]:
        image = images[image_key]
        with Image.open(image.image_path) as raw:
            pil = ImageOps.exif_transpose(raw).convert("RGB")
        scale = min(260 / pil.width, 190 / pil.height)
        thumb = pil.resize((max(1, int(pil.width * scale)), max(1, int(pil.height * scale))), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(thumb)
        for index, box in enumerate(grouped[image_key]):
            x, y, width, height = rect_to_px(box.rect, image.width, image.height)
            color = colors[index % len(colors)]
            draw.rectangle([x * scale, y * scale, (x + width) * scale, (y + height) * scale], outline=color, width=3)
            draw.text((x * scale + 3, y * scale + 3), str(index + 1), fill=color)
        tile = Image.new("RGB", (280, 235), "white")
        tile.paste(thumb, ((280 - thumb.width) // 2, 8))
        label = f"boxes={len(grouped[image_key])} {image.image_path.name[:24]}"
        ImageDraw.Draw(tile).text((8, 205), label, fill=(0, 0, 0))
        tiles.append(tile)
    if not tiles:
        return
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 280, rows * 235), (245, 245, 245))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % cols) * 280, (index // cols) * 235))
    sheet.save(out_dir / filename, quality=88)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        import shutil

        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    images, boxes, input_summaries = merge_inputs(args)
    layout_summary: dict[str, Any] | None = None
    if args.measure_layout:
        layout_boxes, layout_summary = run_layout_measurement(images, args.layout_max_side)
        if args.use_measured_layout_boxes or not boxes:
            boxes = layout_boxes
    in_image_boxes = dedupe_within_image(boxes, args.in_image_iou_threshold)
    dedup_boxes = cross_frame_dedupe(in_image_boxes, images, args.ahash_threshold, args.cross_frame_iou_threshold)
    payload_summary, details = evaluate_payload(
        images,
        boxes,
        in_image_boxes,
        dedup_boxes,
        args.full_max_side,
        args.crop_max_side,
        args.full_quality,
        args.crop_quality,
    )
    summary = {
        "inputs": input_summaries,
        "settings": {
            "full_max_side": args.full_max_side,
            "crop_max_side": args.crop_max_side,
            "full_quality": args.full_quality,
            "crop_quality": args.crop_quality,
            "in_image_iou_threshold": args.in_image_iou_threshold,
            "ahash_threshold": args.ahash_threshold,
            "cross_frame_iou_threshold": args.cross_frame_iou_threshold,
            "measure_layout": bool(args.measure_layout),
            "use_measured_layout_boxes": bool(args.use_measured_layout_boxes),
        },
        **payload_summary,
        "layout_measurement": layout_summary,
        "interpretation": {
            "whole_frame_note": "Whole-frame upload is already required for observation keyframes, but using whole frames as VLM inputs makes the VLM redetect questions on every frame.",
            "crop_jpeg_note": "If keyframes are already uploaded, crop JPEGs are extra network payload.",
            "rect_only_note": "Rect-only upload keeps keyframes as the source of truth and adds only JSON rect manifests; backend crops canonical question images for the VLM.",
        },
        "outputs": {
            "images": "images.jsonl",
            "details": "details.jsonl",
            "raw_preview": "raw_boxes_contact_sheet.jpg",
            "dedup_preview": "dedup_boxes_contact_sheet.jpg",
        },
    }
    write_json(args.out / "summary.json", summary)
    write_jsonl(args.out / "images.jsonl", replay_image_rows(images))
    write_jsonl(args.out / "details.jsonl", details)
    build_contact_sheet(args.out, images, boxes, "raw_boxes_contact_sheet.jpg")
    build_contact_sheet(args.out, images, dedup_boxes, "dedup_boxes_contact_sheet.jpg")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay observation crop payload/latency strategies.")
    parser.add_argument("--prelabel-root", type=Path, action="append", default=[], help="Root produced by question_detector_prelabel.py.")
    parser.add_argument("--dataset-root", type=Path, action="append", default=[], help="Root produced by question_detector_dataset.py.")
    parser.add_argument("--diagnostics", type=Path, action="append", default=[], help="Diagnostics crop bundle containing manifest.json and images/.")
    parser.add_argument("--image-dir", type=Path, action="append", default=[], help="Raw image directory; use with --measure-layout.")
    parser.add_argument("--include-quarantine", action="store_true", help="Also include quarantined prelabel boxes as raw boxes.")
    parser.add_argument("--measure-layout", action="store_true", help="Rerun the lightweight layout prelabeler on every image and record latency.")
    parser.add_argument("--use-measured-layout-boxes", action="store_true", help="Use measured layout boxes instead of imported labels/boxes.")
    parser.add_argument("--layout-max-side", type=int, default=1400)
    parser.add_argument("--full-max-side", type=int, default=1600)
    parser.add_argument("--crop-max-side", type=int, default=1200)
    parser.add_argument("--full-quality", type=int, default=78)
    parser.add_argument("--crop-quality", type=int, default=82)
    parser.add_argument("--in-image-iou-threshold", type=float, default=0.82)
    parser.add_argument("--ahash-threshold", type=int, default=4)
    parser.add_argument("--cross-frame-iou-threshold", type=float, default=0.60)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-observation-replay"))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if not args.prelabel_root and not args.dataset_root and not args.diagnostics and not args.image_dir:
        parser.error("provide at least one input root")
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
