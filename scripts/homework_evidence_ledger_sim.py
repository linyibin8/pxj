"""Replay camera frames into a homework evidence ledger.

This Phase 0 simulator is intentionally independent from the existing
observation/question_set flow. It proves the lower-level product loop first:

1. scan a folder of camera-like frames;
2. keep only useful keyframes;
3. build page episodes and loose question evidence regions;
4. merge repeated sightings;
5. save a best crop for every evidence item;
6. render an editable HTML review notebook plus metrics.

No database, iOS bundle, or production question_set data is modified.
"""

from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import html
import json
import math
import re
import shutil
import time
from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_BASE_URL = "http://100.64.0.5:39000/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "evowit-agent27b"


@dataclass
class NormalizedRect:
    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0

    def clamped(self) -> "NormalizedRect":
        x1 = min(1.0, max(0.0, self.x))
        y1 = min(1.0, max(0.0, self.y))
        x2 = min(1.0, max(x1, self.x + self.w))
        y2 = min(1.0, max(y1, self.y + self.h))
        return NormalizedRect(x1, y1, x2 - x1, y2 - y1)

    def expanded(self, pad_x: float, pad_y: float) -> "NormalizedRect":
        return NormalizedRect(
            self.x - pad_x,
            self.y - pad_y,
            self.w + 2 * pad_x,
            self.h + 2 * pad_y,
        ).clamped()


@dataclass
class FrameRecord:
    frame_id: str
    index: int
    source_path: str
    normalized_path: str
    width: int
    height: int
    source_bytes: int
    normalized_bytes: int
    sharpness: float
    brightness: float
    contrast: float
    phash: str
    accepted: bool
    reason: str
    page_episode_id: str = ""
    hamming_from_previous_keyframe: int | None = None
    page_rectified: bool = False
    page_confidence: float = 0.0
    candidate_count: int = 0


@dataclass
class PageEpisode:
    id: str
    first_frame_id: str
    last_frame_id: str
    frame_ids: list[str] = field(default_factory=list)
    fingerprint: str = ""
    coverage_cells: list[str] = field(default_factory=list)


@dataclass
class CandidateRegion:
    rect: NormalizedRect
    kind: str
    confidence: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class QuestionEvidence:
    id: str
    page_episode_id: str
    canonical_rect: NormalizedRect
    best_frame_id: str
    best_crop_filename: str
    crop_hash: str
    layout_key: str
    ocr_key: str
    seen_count: int
    status: str
    quality: dict[str, float]
    source_frames: list[str] = field(default_factory=list)
    crop_kind: str = "question"
    merge_reasons: list[str] = field(default_factory=list)


@dataclass
class QuestionCard:
    id: str
    evidence_ids: list[str]
    number: str
    subject: str
    question_type: str
    stem_text: str
    editable_html: str
    figure_assets: list[dict[str, Any]]
    confidence: float
    review_flags: list[str]
    source_input_index: int = 0
    extraction_notes: str = ""


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, values: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def list_images(input_path: Path, limit: int | None) -> list[Path]:
    if input_path.is_file():
        images = [input_path]
    else:
        images = [
            path
            for path in sorted(input_path.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    return images[:limit] if limit is not None else images


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in {"RGBA", "LA"}:
            base = Image.new("RGB", image.size, (255, 255, 255))
            base.paste(image.convert("RGB"), mask=image.getchannel("A"))
            return base
        return image.convert("RGB")


def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image.copy()
    out = image.copy()
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def save_jpeg(image: Image.Image, path: Path, quality: int = 90) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=quality, optimize=True)
    return path.stat().st_size


def image_to_data_url(path: Path, max_side: int = 1400, quality: int = 88) -> str:
    image = resize_max_side(load_rgb(path), max_side)
    out = BytesIO()
    image.save(out, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(out.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def hamming_hex(left: str, right: str) -> int:
    if not left or not right:
        return 64
    li = int(left, 16)
    ri = int(right, 16)
    return (li ^ ri).bit_count() + abs(len(left) - len(right)) * 4


def perceptual_hash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    size = hash_size * highfreq_factor
    gray = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    dct = cv2.dct(pixels)
    low = dct[:hash_size, :hash_size]
    median = np.median(low[1:, :])
    bits = low > median
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return f"{value:0{hash_size * hash_size // 4}x}"


def image_metrics(image: Image.Image) -> tuple[float, float, float]:
    gray_image = image.convert("L")
    gray_image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    gray = np.asarray(gray_image, dtype=np.uint8)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return sharpness, float(gray.mean()), float(gray.std())


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def perspective_warp(image: Image.Image, rect: np.ndarray, max_side: int) -> Image.Image:
    tl, tr, br, bl = rect
    width = max(1, int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = max(1, int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(np.asarray(image), matrix, (width, height), borderValue=(255, 255, 255))
    return resize_max_side(Image.fromarray(warped), max_side)


def normalize_page(image: Image.Image, max_side: int) -> tuple[Image.Image, bool, float]:
    resized = resize_max_side(image, max_side)
    arr = np.asarray(resized)
    height, width = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 135)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(width * height)

    best_rect: np.ndarray | None = None
    best_area = 0.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:12]:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.18:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) == 4:
            best_rect = order_points(approx.reshape(4, 2).astype("float32"))
            best_area = area
            break
    if best_rect is None:
        return resized, False, 0.0

    confidence = min(1.0, best_area / max(image_area, 1.0))
    if confidence < 0.22:
        return resized, False, confidence
    try:
        return perspective_warp(resized, best_rect, max_side), True, confidence
    except cv2.error:
        return resized, False, confidence


def normalized_for_hash(image: Image.Image) -> Image.Image:
    out = image.convert("L")
    out.thumbnail((640, 640), Image.Resampling.LANCZOS)
    return out.convert("RGB")


def rect_iou(left: NormalizedRect, right: NormalizedRect) -> float:
    ax1, ay1, ax2, ay2 = left.x, left.y, left.x + left.w, left.y + left.h
    bx1, by1, bx2, by2 = right.x, right.y, right.x + right.w, right.y + right.h
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = left.area + right.area - inter
    return inter / union if union > 0 else 0.0


def rect_distance(left: NormalizedRect, right: NormalizedRect) -> float:
    ax, ay = left.center
    bx, by = right.center
    return math.hypot(ax - bx, ay - by)


def crop_image(image: Image.Image, rect: NormalizedRect) -> Image.Image:
    rect = rect.clamped()
    width, height = image.size
    x1 = max(0, min(width - 1, int(rect.x * width)))
    y1 = max(0, min(height - 1, int(rect.y * height)))
    x2 = max(x1 + 1, min(width, int((rect.x + rect.w) * width)))
    y2 = max(y1 + 1, min(height, int((rect.y + rect.h) * height)))
    return image.crop((x1, y1, x2, y2))


def crop_hash(image: Image.Image) -> str:
    return perceptual_hash(normalized_for_hash(image), hash_size=8, highfreq_factor=4)


def layout_key(rect: NormalizedRect) -> str:
    def q(value: float) -> int:
        return int(round(value * 20))

    return f"layout:{q(rect.x)}:{q(rect.y)}:{q(rect.w)}:{q(rect.h)}"


def quality_score(sharpness: float, rect: NormalizedRect, confidence: float) -> float:
    area = max(0.001, min(1.0, rect.area))
    area_bonus = 1.0 - abs(math.log(area / 0.12)) * 0.08
    return math.log1p(max(0.0, sharpness)) * 0.72 + max(0.0, area_bonus) + confidence * 0.8


def detect_candidate_regions(image: Image.Image, *, max_regions: int) -> list[CandidateRegion]:
    arr = np.asarray(image)
    height, width = arr.shape[:2]
    if width <= 0 or height <= 0:
        return []

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    block = max(31, ((min(width, height) // 40) | 1))
    binary = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        13,
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    # Connect nearby text, formula, and simple figure strokes into loose blocks.
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(16, width // 32), max(3, height // 260)))
    block_mask = cv2.dilate(binary, line_kernel, iterations=1)
    block_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(7, width // 95), max(10, height // 55)))
    block_mask = cv2.dilate(block_mask, block_kernel, iterations=1)
    contours, _ = cv2.findContours(block_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[CandidateRegion] = []
    image_area = width * height
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < image_area * 0.002:
            continue
        if w < width * 0.08 or h < height * 0.018:
            continue
        if area > image_area * 0.70:
            continue
        rect = NormalizedRect(x / width, y / height, w / width, h / height)
        reasons = []
        if rect.h < 0.06:
            reasons.append("thin")
            rect = rect.expanded(0.035, 0.045)
        elif rect.h < 0.11:
            rect = rect.expanded(0.03, 0.035)
        else:
            rect = rect.expanded(0.025, 0.025)
        if rect.w > 0.72:
            reasons.append("wide")
        candidates.append(
            CandidateRegion(
                rect=rect,
                kind="question",
                confidence=min(0.92, 0.42 + min(0.4, area / max(1, image_area) * 4.0)),
                reasons=reasons,
            )
        )

    merged = merge_regions(candidates)
    merged.sort(key=lambda item: (item.rect.y, item.rect.x))

    if len(merged) == 0:
        return fallback_section_regions(binary, width, height, max_sections=max(1, min(4, max_regions)))
    if len(merged) > max_regions:
        ranked = sorted(merged, key=lambda item: item.rect.area * item.confidence, reverse=True)[:max_regions]
        section = section_region_for([item.rect for item in merged])
        ranked.append(
            CandidateRegion(
                rect=section,
                kind="section",
                confidence=0.66,
                reasons=["candidate_cap_section"],
            )
        )
        return sorted(ranked, key=lambda item: (item.kind != "section", item.rect.y, item.rect.x))
    return merged


def content_rect_from_mask(mask: np.ndarray, width: int, height: int) -> NormalizedRect:
    ys, xs = np.where(mask > 0)
    if len(xs) < 20 or len(ys) < 20:
        return NormalizedRect(0.04, 0.04, 0.92, 0.92)
    x1 = float(np.percentile(xs, 1)) / width
    y1 = float(np.percentile(ys, 1)) / height
    x2 = float(np.percentile(xs, 99)) / width
    y2 = float(np.percentile(ys, 99)) / height
    return NormalizedRect(x1, y1, max(0.01, x2 - x1), max(0.01, y2 - y1)).expanded(0.04, 0.05)


def fallback_section_regions(mask: np.ndarray, width: int, height: int, max_sections: int) -> list[CandidateRegion]:
    content = content_rect_from_mask(mask, width, height)
    columns = 2 if content.w > 0.70 and width / max(1, height) > 1.15 and max_sections >= 2 else 1
    rows = 2 if content.h > 0.48 and max_sections >= columns * 2 else 1
    while columns * rows > max_sections and rows > 1:
        rows -= 1

    regions: list[CandidateRegion] = []
    for row in range(rows):
        for column in range(columns):
            rect = NormalizedRect(
                content.x + content.w * column / columns,
                content.y + content.h * row / rows,
                content.w / columns,
                content.h / rows,
            ).expanded(0.025, 0.03)
            regions.append(
                CandidateRegion(
                    rect=rect,
                    kind="section",
                    confidence=0.52,
                    reasons=["fallback_content_section", f"grid_{columns}x{rows}"],
                )
            )
    return regions or [
        CandidateRegion(
            rect=NormalizedRect(0.04, 0.04, 0.92, 0.92),
            kind="section",
            confidence=0.45,
            reasons=["fallback_full_page"],
        )
    ]


def merge_regions(candidates: list[CandidateRegion]) -> list[CandidateRegion]:
    output: list[CandidateRegion] = []
    for candidate in sorted(candidates, key=lambda item: item.rect.area, reverse=True):
        merged = False
        for existing in output:
            same_column = abs(candidate.rect.center[0] - existing.rect.center[0]) < 0.08
            if rect_iou(candidate.rect, existing.rect) > 0.38 or (
                same_column and rect_distance(candidate.rect, existing.rect) < 0.08
            ):
                union = union_rect([candidate.rect, existing.rect])
                existing.rect = union
                existing.confidence = max(existing.confidence, candidate.confidence)
                existing.reasons = sorted(set(existing.reasons + candidate.reasons + ["merged_local"]))
                merged = True
                break
        if not merged:
            output.append(candidate)
    return output


def union_rect(rects: list[NormalizedRect]) -> NormalizedRect:
    x1 = min(rect.x for rect in rects)
    y1 = min(rect.y for rect in rects)
    x2 = max(rect.x + rect.w for rect in rects)
    y2 = max(rect.y + rect.h for rect in rects)
    return NormalizedRect(x1, y1, x2 - x1, y2 - y1).clamped()


def section_region_for(rects: list[NormalizedRect]) -> NormalizedRect:
    if not rects:
        return NormalizedRect(0.04, 0.04, 0.92, 0.92)
    return union_rect(rects).expanded(0.045, 0.06)


def page_coverage_cells(regions: list[QuestionEvidence], grid_w: int = 6, grid_h: int = 8) -> list[str]:
    cells: set[str] = set()
    for evidence in regions:
        rect = evidence.canonical_rect
        x1 = max(0, int(math.floor(rect.x * grid_w)))
        y1 = max(0, int(math.floor(rect.y * grid_h)))
        x2 = min(grid_w - 1, int(math.floor((rect.x + rect.w) * grid_w)))
        y2 = min(grid_h - 1, int(math.floor((rect.y + rect.h) * grid_h)))
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                cells.add(f"{x}:{y}")
    return sorted(cells)


def next_id(prefix: str, count: int) -> str:
    return f"{prefix}_{count:04d}"


def source_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def frame_should_be_accepted(
    *,
    index: int,
    sharpness: float,
    phash: str,
    previous_key_hash: str,
    previous_key_index: int,
    previous_key_sharpness: float,
    args: argparse.Namespace,
) -> tuple[bool, str, int | None]:
    if sharpness < args.min_sharpness:
        return False, "too_blurry", None
    if not previous_key_hash:
        return True, "first_keyframe", None

    distance = hamming_hex(phash, previous_key_hash)
    if distance <= args.near_duplicate_hamming:
        if sharpness >= previous_key_sharpness * args.better_quality_ratio and index - previous_key_index >= 2:
            return True, f"better_quality_same_view_hamming_{distance}", distance
        return False, f"near_duplicate_hamming_{distance}", distance
    if distance >= args.new_content_hamming:
        return True, f"new_content_hamming_{distance}", distance
    if index - previous_key_index >= args.keepalive_interval:
        return True, f"keepalive_hamming_{distance}", distance
    return False, f"minor_change_hamming_{distance}", distance


def create_or_update_page_episode(
    episodes: list[PageEpisode],
    frame: FrameRecord,
    previous_episode_id: str,
    args: argparse.Namespace,
) -> str:
    if not previous_episode_id or not episodes:
        episode_id = next_id("page", len(episodes) + 1)
        episodes.append(
            PageEpisode(
                id=episode_id,
                first_frame_id=frame.frame_id,
                last_frame_id=frame.frame_id,
                frame_ids=[frame.frame_id],
                fingerprint=frame.phash,
            )
        )
        return episode_id

    current = episodes[-1]
    distance = hamming_hex(frame.phash, current.fingerprint)
    if distance >= args.page_turn_hamming:
        episode_id = next_id("page", len(episodes) + 1)
        episodes.append(
            PageEpisode(
                id=episode_id,
                first_frame_id=frame.frame_id,
                last_frame_id=frame.frame_id,
                frame_ids=[frame.frame_id],
                fingerprint=frame.phash,
            )
        )
        return episode_id

    current.last_frame_id = frame.frame_id
    current.frame_ids.append(frame.frame_id)
    return current.id


def evidence_match(
    evidence: QuestionEvidence,
    candidate_rect: NormalizedRect,
    candidate_hash: str,
    candidate_kind: str,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if evidence.crop_kind != candidate_kind:
        return False, "kind_mismatch"
    iou = rect_iou(evidence.canonical_rect, candidate_rect)
    if iou >= args.merge_iou:
        return True, f"iou_{iou:.2f}"
    distance = rect_distance(evidence.canonical_rect, candidate_rect)
    hash_distance = hamming_hex(evidence.crop_hash, candidate_hash) if evidence.crop_hash and candidate_hash else 64
    if hash_distance <= args.crop_hash_hamming and distance <= args.merge_center_distance:
        return True, f"crop_hash_{hash_distance}_dist_{distance:.2f}"
    if layout_key(evidence.canonical_rect) == layout_key(candidate_rect) and distance <= args.merge_center_distance:
        return True, "layout_key"
    return False, "no_match"


def update_episode_coverage(episodes: list[PageEpisode], evidence_items: list[QuestionEvidence]) -> None:
    by_episode: dict[str, list[QuestionEvidence]] = {}
    for evidence in evidence_items:
        by_episode.setdefault(evidence.page_episode_id, []).append(evidence)
    for episode in episodes:
        episode.coverage_cells = page_coverage_cells(by_episode.get(episode.id, []))


def build_ledger(args: argparse.Namespace, image_paths: list[Path], out_dir: Path) -> tuple[
    list[FrameRecord],
    list[PageEpisode],
    list[QuestionEvidence],
]:
    normalized_dir = out_dir / "normalized_frames"
    crop_dir = out_dir / "crops"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    frames: list[FrameRecord] = []
    episodes: list[PageEpisode] = []
    evidence_items: list[QuestionEvidence] = []

    previous_key_hash = ""
    previous_key_index = 0
    previous_key_sharpness = 0.0
    previous_episode_id = ""

    for index, source_path in enumerate(image_paths, start=1):
        frame_id = next_id("frame", index)
        source = load_rgb(source_path)
        normalized, rectified, page_confidence = normalize_page(source, args.max_side)
        sharpness, brightness, contrast = image_metrics(normalized)
        phash = perceptual_hash(normalized_for_hash(normalized))
        accepted, reason, hamming = frame_should_be_accepted(
            index=index,
            sharpness=sharpness,
            phash=phash,
            previous_key_hash=previous_key_hash,
            previous_key_index=previous_key_index,
            previous_key_sharpness=previous_key_sharpness,
            args=args,
        )
        normalized_path = normalized_dir / f"{frame_id}.jpg"
        normalized_bytes = save_jpeg(normalized, normalized_path, quality=args.keyframe_quality)

        frame = FrameRecord(
            frame_id=frame_id,
            index=index,
            source_path=str(source_path),
            normalized_path=str(normalized_path),
            width=normalized.width,
            height=normalized.height,
            source_bytes=source_size(source_path),
            normalized_bytes=normalized_bytes,
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            phash=phash,
            accepted=accepted,
            reason=reason,
            hamming_from_previous_keyframe=hamming,
            page_rectified=rectified,
            page_confidence=page_confidence,
        )

        if accepted:
            episode_id = create_or_update_page_episode(
                episodes,
                frame,
                previous_episode_id,
                args,
            )
            frame.page_episode_id = episode_id
            previous_episode_id = episode_id
            previous_key_hash = phash
            previous_key_index = index
            previous_key_sharpness = sharpness

            candidates = detect_candidate_regions(normalized, max_regions=args.max_regions_per_frame)
            frame.candidate_count = len(candidates)
            for candidate in candidates:
                crop = crop_image(normalized, candidate.rect)
                chash = crop_hash(crop)
                crop_sharpness, crop_brightness, crop_contrast = image_metrics(crop)
                score = quality_score(crop_sharpness, candidate.rect, candidate.confidence)
                matched: QuestionEvidence | None = None
                match_reason = ""
                for evidence in evidence_items:
                    if evidence.page_episode_id != episode_id:
                        continue
                    is_match, reason_text = evidence_match(
                        evidence,
                        candidate.rect,
                        chash,
                        candidate.kind,
                        args,
                    )
                    if is_match:
                        matched = evidence
                        match_reason = reason_text
                        break

                if matched is None:
                    evidence_id = next_id("qev", len(evidence_items) + 1)
                    crop_filename = f"{evidence_id}.jpg"
                    save_jpeg(crop, crop_dir / crop_filename, quality=args.crop_quality)
                    evidence_items.append(
                        QuestionEvidence(
                            id=evidence_id,
                            page_episode_id=episode_id,
                            canonical_rect=candidate.rect,
                            best_frame_id=frame_id,
                            best_crop_filename=f"crops/{crop_filename}",
                            crop_hash=chash,
                            layout_key=layout_key(candidate.rect),
                            ocr_key="",
                            seen_count=1,
                            status="ready",
                            quality={
                                "score": round(score, 4),
                                "sharpness": round(crop_sharpness, 4),
                                "brightness": round(crop_brightness, 4),
                                "contrast": round(crop_contrast, 4),
                                "confidence": round(candidate.confidence, 4),
                                "area": round(candidate.rect.area, 6),
                            },
                            source_frames=[frame_id],
                            crop_kind=candidate.kind,
                            merge_reasons=candidate.reasons,
                        )
                    )
                else:
                    matched.seen_count += 1
                    matched.source_frames.append(frame_id)
                    matched.merge_reasons.append(match_reason)
                    old_score = float(matched.quality.get("score") or 0.0)
                    if score > old_score * 1.05:
                        save_jpeg(crop, out_dir / matched.best_crop_filename, quality=args.crop_quality)
                        matched.best_frame_id = frame_id
                        matched.crop_hash = chash
                        matched.canonical_rect = candidate.rect
                        matched.layout_key = layout_key(candidate.rect)
                        matched.quality = {
                            "score": round(score, 4),
                            "sharpness": round(crop_sharpness, 4),
                            "brightness": round(crop_brightness, 4),
                            "contrast": round(crop_contrast, 4),
                            "confidence": round(candidate.confidence, 4),
                            "area": round(candidate.rect.area, 6),
                        }
        frames.append(frame)

    update_episode_coverage(episodes, evidence_items)
    return frames, episodes, evidence_items


def build_question_cards(evidence_items: list[QuestionEvidence]) -> list[QuestionCard]:
    cards: list[QuestionCard] = []
    for index, evidence in enumerate(evidence_items, start=1):
        flags = []
        if evidence.crop_kind == "section":
            flags.append("section_crop")
        flags.append("needs_text_extraction")
        flags.append("raster_figure_evidence")
        flags.append("vlm_not_run")
        flags.append("needs_review")
        title = f"Evidence {index}"
        body = (
            "<p>This card is generated from the best visual evidence crop. "
            "Text extraction and figure vectorization are intentionally separate later steps.</p>"
        )
        figure_html = (
            f'<figure><img src="{html.escape(evidence.best_crop_filename)}" alt="{html.escape(evidence.id)}">'
            f"<figcaption>{html.escape(evidence.id)} · seen {evidence.seen_count} times</figcaption></figure>"
        )
        cards.append(
            QuestionCard(
                id=f"q_{evidence.id}",
                evidence_ids=[evidence.id],
                number="",
                subject="unknown",
                question_type="evidence",
                stem_text=title,
                editable_html=f"<article><h2>{html.escape(title)}</h2>{body}{figure_html}</article>",
                figure_assets=[
                    {
                        "type": "raster",
                        "filename": evidence.best_crop_filename,
                        "source_evidence_id": evidence.id,
                    }
                ],
                confidence=float(evidence.quality.get("confidence") or 0.0),
                review_flags=flags,
                source_input_index=0,
                extraction_notes="fallback evidence card; VLM extraction not run",
            )
        )
    return cards


def relative(path: str | Path, base: Path) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def rect_style(rect: NormalizedRect) -> str:
    return (
        f"left:{rect.x * 100:.2f}%;top:{rect.y * 100:.2f}%;"
        f"width:{rect.w * 100:.2f}%;height:{rect.h * 100:.2f}%;"
    )


def render_html(
    out_dir: Path,
    frames: list[FrameRecord],
    episodes: list[PageEpisode],
    evidence_items: list[QuestionEvidence],
    cards: list[QuestionCard],
    metrics: dict[str, Any],
) -> str:
    frame_tiles = []
    for frame in frames:
        state = "accepted" if frame.accepted else "skipped"
        image_src = relative(frame.normalized_path, out_dir)
        frame_tiles.append(
            f"""
            <button class="frame-tile {state}" data-frame="{html.escape(frame.frame_id)}">
              <img src="{html.escape(image_src)}" alt="{html.escape(frame.frame_id)}">
              <strong>{html.escape(frame.frame_id)}</strong>
              <span>{html.escape(frame.reason)}</span>
              <small>sharp {frame.sharpness:.0f} · candidates {frame.candidate_count}</small>
            </button>
            """
        )

    evidence_by_frame: dict[str, list[QuestionEvidence]] = {}
    for evidence in evidence_items:
        evidence_by_frame.setdefault(evidence.best_frame_id, []).append(evidence)

    boards = []
    for frame in [item for item in frames if item.accepted]:
        image_src = relative(frame.normalized_path, out_dir)
        boxes = []
        for evidence in evidence_by_frame.get(frame.frame_id, []):
            boxes.append(
                f'<a class="box {html.escape(evidence.crop_kind)}" href="#{html.escape(evidence.id)}" '
                f'style="{rect_style(evidence.canonical_rect)}">{html.escape(evidence.id)}</a>'
            )
        boards.append(
            f"""
            <section class="board">
              <div class="board-head">
                <b>{html.escape(frame.frame_id)}</b>
                <span>{html.escape(frame.page_episode_id)}</span>
              </div>
              <div class="image-board">
                <img src="{html.escape(image_src)}" alt="{html.escape(frame.frame_id)}">
                {"".join(boxes)}
              </div>
            </section>
            """
        )

    evidence_cards = []
    for evidence in evidence_items:
        crop_src = relative(out_dir / evidence.best_crop_filename, out_dir)
        evidence_cards.append(
            f"""
            <section class="evidence-card" id="{html.escape(evidence.id)}">
              <div class="card-head">
                <div>
                  <b>{html.escape(evidence.id)}</b>
                  <span>{html.escape(evidence.crop_kind)}</span>
                  <span>{html.escape(evidence.status)}</span>
                  <span>seen {evidence.seen_count}</span>
                </div>
                <strong>{float(evidence.quality.get("score") or 0):.2f}</strong>
              </div>
              <div class="evidence-grid">
                <img src="{html.escape(crop_src)}" alt="{html.escape(evidence.id)}">
                <div>
                  <p><b>Best frame:</b> {html.escape(evidence.best_frame_id)}</p>
                  <p><b>Layout:</b> {html.escape(evidence.layout_key)}</p>
                  <p><b>Crop hash:</b> {html.escape(evidence.crop_hash[:16])}</p>
                  <p><b>Sources:</b> {html.escape(", ".join(evidence.source_frames))}</p>
                  <p><b>Merge:</b> {html.escape(", ".join(evidence.merge_reasons[-6:]))}</p>
                </div>
              </div>
            </section>
            """
        )

    question_cards = []
    for card in cards:
        flags = ", ".join(card.review_flags)
        question_cards.append(
            f"""
            <section class="question-card" id="{html.escape(card.id)}">
              <div class="card-head">
                <div>
                  <b>{html.escape(card.id)}</b>
                  <span>{html.escape(card.subject)}</span>
                  <span>{html.escape(card.question_type)}</span>
                  <span>{html.escape(flags)}</span>
                </div>
                <strong>{card.confidence:.2f}</strong>
              </div>
              <div class="editable" contenteditable="true" data-question-id="{html.escape(card.id)}">
                {card.editable_html}
              </div>
              <div class="card-foot">
                evidence {html.escape(", ".join(card.evidence_ids))} · input {card.source_input_index or "-"} · {html.escape(card.extraction_notes)}
              </div>
            </section>
            """
        )

    episode_rows = []
    for episode in episodes:
        episode_rows.append(
            f"""
            <tr>
              <td>{html.escape(episode.id)}</td>
              <td>{html.escape(episode.first_frame_id)} - {html.escape(episode.last_frame_id)}</td>
              <td>{len(episode.frame_ids)}</td>
              <td>{len(episode.coverage_cells)}</td>
            </tr>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Homework Evidence Ledger</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #16202a;
      --muted: #637083;
      --line: #d8e0ea;
      --paper: #f6f7f9;
      --panel: #ffffff;
      --blue: #2457c5;
      --green: #16794c;
      --amber: #9b6500;
      --red: #b42318;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--paper);
      color: var(--ink);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 4;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 14px;
      align-items: center;
      padding: 13px 16px;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }}
    h1 {{ margin: 0; font-size: 18px; letter-spacing: 0; }}
    .stats {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    .pill, .card-head span {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--muted);
      border-radius: 6px;
      padding: 5px 8px;
      font-size: 12px;
    }}
    main {{
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      min-height: calc(100vh - 60px);
    }}
    aside {{
      background: #fff;
      border-right: 1px solid var(--line);
      padding: 12px;
      overflow: auto;
      max-height: calc(100vh - 60px);
    }}
    .frame-list {{ display: grid; gap: 9px; }}
    .frame-tile {{
      width: 100%;
      display: grid;
      grid-template-columns: 72px minmax(0, 1fr);
      grid-template-rows: auto auto auto;
      gap: 2px 9px;
      border: 1px solid var(--line);
      border-left: 4px solid var(--muted);
      background: #fff;
      border-radius: 6px;
      padding: 8px;
      text-align: left;
    }}
    .frame-tile.accepted {{ border-left-color: var(--green); }}
    .frame-tile.skipped {{ opacity: 0.66; }}
    .frame-tile img {{
      grid-row: 1 / 4;
      width: 72px;
      height: 58px;
      object-fit: cover;
      border: 1px solid var(--line);
      border-radius: 4px;
    }}
    .frame-tile span, .frame-tile small {{
      color: var(--muted);
      overflow-wrap: anywhere;
    }}
    .workspace {{ padding: 16px; display: grid; gap: 16px; align-content: start; }}
    .section-title {{ margin: 2px 0 0; font-size: 16px; }}
    .episode-table {{
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .episode-table th, .episode-table td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      font-size: 13px;
    }}
    .boards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }}
    .board, .evidence-card, .question-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .board-head, .card-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      padding: 9px 11px;
      background: #fbfcfe;
      border-bottom: 1px solid var(--line);
    }}
    .card-head div {{ display: flex; gap: 7px; flex-wrap: wrap; align-items: center; }}
    .image-board {{ position: relative; background: #f0f2f5; }}
    .image-board img {{ display: block; width: 100%; height: auto; }}
    .box {{
      position: absolute;
      display: grid;
      place-items: start;
      border: 2px solid var(--blue);
      background: rgba(36, 87, 197, 0.08);
      color: var(--blue);
      font-size: 11px;
      font-weight: 700;
      padding: 2px;
      text-decoration: none;
      overflow: hidden;
    }}
    .box.section {{ border-color: var(--amber); background: rgba(155, 101, 0, 0.10); color: var(--amber); }}
    .evidence-grid {{
      display: grid;
      grid-template-columns: minmax(160px, 280px) minmax(0, 1fr);
      gap: 12px;
      padding: 12px;
      align-items: start;
    }}
    .evidence-grid img {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 5px;
      background: #fff;
    }}
    .evidence-grid p {{ margin: 0 0 7px; color: var(--muted); overflow-wrap: anywhere; }}
    .editable {{
      padding: 14px;
      min-height: 130px;
      line-height: 1.6;
      outline: none;
    }}
    .editable:focus {{ box-shadow: inset 0 0 0 2px rgba(36, 87, 197, 0.25); }}
    .editable h2 {{ margin: 0 0 8px; font-size: 17px; letter-spacing: 0; }}
    .editable figure {{ margin: 10px 0 0; }}
    .editable img {{ max-width: 100%; border: 1px solid var(--line); border-radius: 5px; }}
    .editable figcaption {{ color: var(--muted); font-size: 12px; margin-top: 5px; }}
    .card-foot {{
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      padding: 8px 12px;
      overflow-wrap: anywhere;
    }}
    button.action {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 6px 9px;
      cursor: pointer;
      font-weight: 650;
    }}
    @media (max-width: 880px) {{
      header, main {{ grid-template-columns: 1fr; }}
      aside {{ max-height: none; border-right: 0; border-bottom: 1px solid var(--line); }}
      .evidence-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Homework Evidence Ledger</h1>
    <div class="stats">
      <span class="pill">frames {metrics["frames_total"]}</span>
      <span class="pill">keyframes {metrics["keyframes"]}</span>
      <span class="pill">evidence {metrics["question_evidence_count"]}</span>
      <span class="pill">cards {metrics["question_card_count"]}</span>
      <span class="pill">vlm {html.escape(str(metrics.get("vlm_enabled", False)).lower())}</span>
      <span class="pill">upload ratio {metrics["estimated_upload_ratio"]:.3f}</span>
      <button class="action" id="download">Download edited cards</button>
    </div>
  </header>
  <main>
    <aside><div class="frame-list">{"".join(frame_tiles)}</div></aside>
    <section class="workspace">
      <h2 class="section-title">Page Episodes</h2>
      <table class="episode-table">
        <thead><tr><th>Episode</th><th>Frames</th><th>Keyframes</th><th>Coverage cells</th></tr></thead>
        <tbody>{"".join(episode_rows)}</tbody>
      </table>
      <h2 class="section-title">Accepted Keyframes</h2>
      <div class="boards">{"".join(boards)}</div>
      <h2 class="section-title">Question Evidence</h2>
      <div class="cards">{"".join(evidence_cards)}</div>
      <h2 class="section-title">Editable Question Cards</h2>
      <div class="cards">{"".join(question_cards)}</div>
    </section>
  </main>
  <script>
    document.querySelector('#download').addEventListener('click', () => {{
      const cards = Array.from(document.querySelectorAll('.editable')).map(node => ({{
        id: node.dataset.questionId,
        html: node.innerHTML
      }}));
      const blob = new Blob([JSON.stringify({{ cards }}, null, 2)], {{ type: 'application/json' }});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'edited_homework_cards.json';
      link.click();
      URL.revokeObjectURL(link.href);
    }});
  </script>
</body>
</html>
"""


def dataclass_to_json(value: Any) -> Any:
    if isinstance(value, NormalizedRect):
        return asdict(value)
    if isinstance(value, list):
        return [dataclass_to_json(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        data = asdict(value)
        return dataclass_to_json(data)
    if isinstance(value, dict):
        return {key: dataclass_to_json(item) for key, item in value.items()}
    return value


def parse_jsonish(raw: str) -> Any:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not starts:
        return {}
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def coerce_question_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        questions = payload.get("questions") or payload.get("items") or payload.get("question_cards")
        if isinstance(questions, list):
            return [item for item in questions if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def strip_scripts(fragment: str) -> str:
    return re.sub(r"<\s*script\b.*?<\s*/\s*script\s*>", "", fragment or "", flags=re.I | re.S)


def canonical_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value or "").lower()
    text = re.sub(r"\s+", "", text)
    return "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def card_similarity(left: QuestionCard, right: QuestionCard) -> float:
    left_text = canonical_text(left.stem_text or left.editable_html)
    right_text = canonical_text(right.stem_text or right.editable_html)
    if not left_text or not right_text:
        return 0.0
    score = difflib.SequenceMatcher(None, left_text, right_text).ratio()
    if left.number and right.number and left.number == right.number and score >= 0.68:
        score = max(score, 0.90)
    return score


def merge_question_card(cards: list[QuestionCard], candidate: QuestionCard, threshold: float) -> None:
    best: QuestionCard | None = None
    best_score = 0.0
    for existing in cards:
        score = card_similarity(existing, candidate)
        if score > best_score:
            best = existing
            best_score = score
    if best is None or best_score < threshold:
        cards.append(candidate)
        return
    best.evidence_ids = sorted(set(best.evidence_ids + candidate.evidence_ids))
    best.figure_assets.extend(
        asset for asset in candidate.figure_assets if asset not in best.figure_assets
    )
    best.review_flags = sorted(set(best.review_flags + candidate.review_flags + ["semantic_duplicate_merged"]))
    best.confidence = max(best.confidence, candidate.confidence)
    if len(candidate.stem_text) > len(best.stem_text):
        best.stem_text = candidate.stem_text
        best.editable_html = candidate.editable_html
        best.number = candidate.number or best.number
        best.subject = candidate.subject or best.subject
        best.question_type = candidate.question_type or best.question_type
        best.extraction_notes = candidate.extraction_notes or best.extraction_notes


def evidence_vlm_prompt(evidence_batch: list[QuestionEvidence]) -> str:
    lines = [
        "你在做一个作业摄像头的证据账本系统。",
        "输入图片不是整页定位任务，而是系统已经裁好的题目证据 crop 或 section crop。",
        "请只基于每张输入图里真实可见的内容生成可编辑题卡 JSON。",
        "",
        "重要边界：",
        "- 不要编答案，不要补图外信息。",
        "- 不要尝试猜原图中的绝对位置；来源由 input_index 和 evidence_id 绑定。",
        "- 几何图、坐标图、表格、插图优先保留为 raster evidence。只有非常简单且可靠时，才在 figure_assets 里提供 svg_draft。",
        "- 如果图片是 section crop，里面可能有多道题；请把每道可见题分别列出，并保留相同 input_index。",
        "- 若只能确认这是图形/题目证据但文字看不清，也返回一条低置信题卡，review_flags 包含 needs_review。",
        "",
        "返回严格 JSON：",
        "{",
        '  "questions": [',
        "    {",
        '      "input_index": 1,',
        '      "evidence_id": "qev_0001",',
        '      "number": "题号或空",',
        '      "subject": "math|chinese|english|science|unknown",',
        '      "question_type": "choice|fill_blank|calculation|geometry|reading|table|figure|unknown",',
        '      "stem_text": "题干纯文本，公式可写 LaTeX，看不清写 [未识别]",',
        '      "editable_html": "<article>...</article>",',
        '      "has_figure": true,',
        '      "figure_note": "含图/几何/表格时说明保留局部位图证据",',
        '      "svg_draft": "只有可靠时填写 SVG，否则空字符串",',
        '      "confidence": 0.0,',
        '      "review_flags": ["has_figure","needs_review"],',
        '      "notes": "不确定点"',
        "    }",
        "  ]",
        "}",
        "",
        "输入索引和 evidence：",
    ]
    for index, evidence in enumerate(evidence_batch, start=1):
        lines.append(
            f"{index}. evidence_id={evidence.id}, kind={evidence.crop_kind}, seen={evidence.seen_count}, "
            f"best_frame={evidence.best_frame_id}"
        )
    return "\n".join(lines)


def call_vision_model(
    *,
    evidence_batch: list[QuestionEvidence],
    image_paths: list[Path],
    base_url: str,
    api_key: str,
    model: str,
    timeout_seconds: float,
    max_tokens: int,
) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": evidence_vlm_prompt(evidence_batch)}]
    for index, path in enumerate(image_paths, start=1):
        content.append({"type": "text", "text": f"input_index={index}; filename={path.name}"})
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful OCR and visual evidence extraction assistant. Return valid JSON only.",
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    timeout = httpx.Timeout(timeout_seconds, connect=10)
    with httpx.Client(base_url=base_url, timeout=timeout, trust_env=False) as client:
        response = client.post("/chat/completions", json=payload, headers={"Authorization": f"Bearer {api_key}"})
        response.raise_for_status()
        data = response.json()
    message = data["choices"][0]["message"]
    return message.get("content") or message.get("reasoning_content") or json.dumps(data, ensure_ascii=False)


def normalize_vlm_question(
    item: dict[str, Any],
    evidence_by_input_index: dict[int, QuestionEvidence],
) -> QuestionCard | None:
    try:
        input_index = int(item.get("input_index") or item.get("inputIndex") or 0)
    except (TypeError, ValueError):
        input_index = 0
    evidence = evidence_by_input_index.get(input_index)
    if evidence is None:
        evidence_id = str(item.get("evidence_id") or item.get("evidenceId") or "").strip()
        evidence = next((value for value in evidence_by_input_index.values() if value.id == evidence_id), None)
    if evidence is None:
        return None

    confidence_raw = item.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    except (TypeError, ValueError):
        confidence = 0.0

    flags = item.get("review_flags") or item.get("reviewFlags") or []
    if not isinstance(flags, list):
        flags = [str(flags)]
    flags = [str(flag) for flag in flags if str(flag)]
    if item.get("has_figure") or item.get("hasFigure") or item.get("figure_note") or item.get("svg_draft"):
        flags.append("has_figure")
    if evidence.crop_kind == "section":
        flags.append("section_crop")
    if confidence < 0.55:
        flags.append("needs_review")

    figure_assets = [
        {
            "type": "raster",
            "filename": evidence.best_crop_filename,
            "source_evidence_id": evidence.id,
        }
    ]
    svg_draft = str(item.get("svg_draft") or item.get("svgDraft") or "").strip()
    if svg_draft:
        figure_assets.append(
            {
                "type": "svg_draft",
                "source_evidence_id": evidence.id,
                "svg": strip_scripts(svg_draft),
            }
        )
        flags.append("svg_draft_unverified")

    stem_text = str(item.get("stem_text") or item.get("stem") or item.get("question") or "[未识别]").strip()
    editable_html = str(item.get("editable_html") or item.get("html") or "").strip()
    if not editable_html or "<" not in editable_html:
        editable_html = (
            f"<article><h2>{html.escape(str(item.get('number') or '未编号题目'))}</h2>"
            f"<p>{html.escape(stem_text)}</p></article>"
        )
    editable_html = strip_scripts(editable_html)
    if evidence.best_crop_filename not in editable_html:
        editable_html += (
            f'<figure><img src="{html.escape(evidence.best_crop_filename)}" alt="{html.escape(evidence.id)}">'
            f"<figcaption>原始证据：{html.escape(evidence.id)}</figcaption></figure>"
        )

    card_seed = canonical_text(stem_text) or evidence.id
    card_id = "q_" + hashlib.sha1(card_seed.encode("utf-8")).hexdigest()[:12]
    return QuestionCard(
        id=card_id,
        evidence_ids=[evidence.id],
        number=str(item.get("number") or ""),
        subject=str(item.get("subject") or "unknown"),
        question_type=str(item.get("question_type") or item.get("type") or "unknown"),
        stem_text=stem_text,
        editable_html=editable_html,
        figure_assets=figure_assets,
        confidence=confidence,
        review_flags=sorted(set(flags)),
        source_input_index=input_index,
        extraction_notes=str(item.get("notes") or item.get("figure_note") or ""),
    )


def build_question_cards_with_vlm(args: argparse.Namespace, evidence_items: list[QuestionEvidence], out_dir: Path) -> list[QuestionCard]:
    raw_dir = out_dir / "vlm_extraction"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cards: list[QuestionCard] = []
    selected = evidence_items[: args.max_vlm_evidence] if args.max_vlm_evidence is not None else evidence_items
    for start in range(0, len(selected), args.vlm_batch_size):
        batch = selected[start : start + args.vlm_batch_size]
        if not batch:
            continue
        image_paths = [out_dir / evidence.best_crop_filename for evidence in batch]
        batch_id = f"batch_{start // args.vlm_batch_size + 1:04d}"
        print(f"[vlm] extracting {batch_id}: {', '.join(item.id for item in batch)}", flush=True)
        raw = call_vision_model(
            evidence_batch=batch,
            image_paths=image_paths,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout,
            max_tokens=args.max_tokens,
        )
        write_text(raw_dir / f"{batch_id}.raw.txt", raw)
        parsed = parse_jsonish(raw)
        write_json(raw_dir / f"{batch_id}.json", parsed)
        evidence_by_input_index = {index: evidence for index, evidence in enumerate(batch, start=1)}
        for item in coerce_question_payload(parsed):
            card = normalize_vlm_question(item, evidence_by_input_index)
            if card is not None:
                merge_question_card(cards, card, args.semantic_duplicate_threshold)

    extracted_evidence_ids = {evidence_id for card in cards for evidence_id in card.evidence_ids}
    for fallback_card in build_question_cards([item for item in evidence_items if item.id not in extracted_evidence_ids]):
        fallback_card.review_flags = sorted(set(fallback_card.review_flags + ["vlm_not_run"]))
        cards.append(fallback_card)
    return cards


def build_metrics(
    frames: list[FrameRecord],
    episodes: list[PageEpisode],
    evidence_items: list[QuestionEvidence],
    cards: list[QuestionCard],
    vlm_enabled: bool,
) -> dict[str, Any]:
    full_upload_bytes = sum(frame.source_bytes for frame in frames)
    keyframe_upload_bytes = sum(frame.normalized_bytes for frame in frames if frame.accepted)
    manifest_bytes = len(
        json.dumps(
            {
                "frames": [dataclass_to_json(frame) for frame in frames if frame.accepted],
                "evidence": [dataclass_to_json(item) for item in evidence_items],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    keyframes = sum(1 for frame in frames if frame.accepted)
    return {
        "frames_total": len(frames),
        "keyframes": keyframes,
        "skipped_frames": len(frames) - keyframes,
        "page_episode_count": len(episodes),
        "question_evidence_count": len(evidence_items),
        "question_card_count": len(cards),
        "vlm_enabled": vlm_enabled,
        "extracted_question_card_count": sum(1 for card in cards if "vlm_not_run" not in card.review_flags),
        "cards_with_raster_figure": sum(
            1 for card in cards if any(asset.get("type") == "raster" for asset in card.figure_assets)
        ),
        "cards_with_svg_draft": sum(
            1 for card in cards if any(asset.get("type") == "svg_draft" for asset in card.figure_assets)
        ),
        "needs_review_count": sum(1 for card in cards if "needs_review" in card.review_flags),
        "duplicate_sighting_count": sum(max(0, item.seen_count - 1) for item in evidence_items),
        "full_frame_upload_bytes_if_naive": full_upload_bytes,
        "keyframe_upload_bytes": keyframe_upload_bytes,
        "ledger_manifest_bytes": manifest_bytes,
        "estimated_upload_bytes": keyframe_upload_bytes + manifest_bytes,
        "estimated_upload_ratio": (
            (keyframe_upload_bytes + manifest_bytes) / full_upload_bytes if full_upload_bytes else 0.0
        ),
        "avg_keyframes_per_input_frame": keyframes / len(frames) if frames else 0.0,
        "crop_jpeg_upload_bytes": 0,
    }


def create_demo_burst(source_paths: list[Path], out_dir: Path, variants_per_source: int) -> Path:
    demo_dir = out_dir / "demo_camera_burst"
    clean_dir(demo_dir)
    rotations = [-0.8, 0.35, -0.25, 0.65, -0.45, 0.15]
    brightness = [0.96, 1.02, 1.0, 1.04, 0.98, 1.01]
    frame_index = 1
    for source_index, source_path in enumerate(source_paths, start=1):
        source = resize_max_side(load_rgb(source_path), 1600)
        for variant_index in range(variants_per_source):
            image = source.copy()
            image = image.rotate(
                rotations[(source_index + variant_index) % len(rotations)],
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=(248, 248, 248),
            )
            image = ImageEnhance.Brightness(image).enhance(
                brightness[(source_index + variant_index) % len(brightness)]
            )
            width, height = image.size
            jitter_x = int(width * (0.004 * ((variant_index % 3) - 1)))
            jitter_y = int(height * (0.004 * (((variant_index + 1) % 3) - 1)))
            crop = image.crop(
                (
                    max(0, jitter_x),
                    max(0, jitter_y),
                    min(width, width + jitter_x),
                    min(height, height + jitter_y),
                )
            )
            canvas = Image.new("RGB", (width, height), (248, 248, 248))
            canvas.paste(crop, (max(0, -jitter_x), max(0, -jitter_y)))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((8, 8, 96, 30), fill=(255, 255, 255))
            draw.text((12, 12), f"sim {frame_index:03d}", fill=(90, 90, 90))
            save_jpeg(canvas, demo_dir / f"frame_{frame_index:03d}_source_{source_index}.jpg", quality=90)
            frame_index += 1
    return demo_dir


def process(args: argparse.Namespace) -> Path:
    started = time.time()
    out_dir = args.out.resolve()
    if args.clean:
        clean_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input.resolve()
    if args.demo_from:
        input_path = create_demo_burst([path.resolve() for path in args.demo_from], out_dir, args.demo_variants)

    image_paths = list_images(input_path, args.limit)
    if not image_paths:
        raise SystemExit(f"No images found in {input_path}")

    frames, episodes, evidence_items = build_ledger(args, image_paths, out_dir)
    if args.use_vlm:
        cards = build_question_cards_with_vlm(args, evidence_items, out_dir)
    else:
        cards = build_question_cards(evidence_items)
    metrics = build_metrics(frames, episodes, evidence_items, cards, vlm_enabled=args.use_vlm)
    metrics["input"] = str(input_path)
    metrics["out_dir"] = str(out_dir)
    metrics["elapsed_seconds"] = round(time.time() - started, 3)

    append_jsonl(out_dir / "frames.jsonl", [dataclass_to_json(frame) for frame in frames])
    write_json(out_dir / "page_episodes.json", dataclass_to_json(episodes))
    write_json(out_dir / "question_evidence.json", dataclass_to_json(evidence_items))
    write_json(out_dir / "question_cards.json", dataclass_to_json(cards))
    write_json(out_dir / "metrics.json", metrics)
    write_text(out_dir / "index.html", render_html(out_dir, frames, episodes, evidence_items, cards, metrics))
    write_text(
        out_dir / "README.md",
        "\n".join(
            [
                "# Homework Evidence Ledger Replay",
                "",
                "Open `index.html` to review keyframes, question evidence crops, and editable cards.",
                "",
                "Important: crop JPEGs in this directory are saved for inspection; the simulated upload path sends keyframes plus a ledger manifest only.",
                "",
                "## Metrics",
                "",
                "```json",
                json.dumps(metrics, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        ),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay homework camera frames into an evidence ledger.")
    parser.add_argument("input", type=Path, nargs="?", default=Path("."), help="Image file or directory.")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/homework-evidence-ledger-sim"))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-side", type=int, default=1800)
    parser.add_argument("--keyframe-quality", type=int, default=88)
    parser.add_argument("--crop-quality", type=int, default=92)
    parser.add_argument("--min-sharpness", type=float, default=28.0)
    parser.add_argument("--near-duplicate-hamming", type=int, default=7)
    parser.add_argument("--new-content-hamming", type=int, default=14)
    parser.add_argument("--page-turn-hamming", type=int, default=22)
    parser.add_argument("--keepalive-interval", type=int, default=12)
    parser.add_argument("--better-quality-ratio", type=float, default=1.25)
    parser.add_argument("--max-regions-per-frame", type=int, default=10)
    parser.add_argument("--merge-iou", type=float, default=0.50)
    parser.add_argument("--merge-center-distance", type=float, default=0.10)
    parser.add_argument("--crop-hash-hamming", type=int, default=5)
    parser.add_argument("--use-vlm", action="store_true", help="Extract editable question cards from evidence crops.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=170.0)
    parser.add_argument("--max-tokens", type=int, default=2600)
    parser.add_argument("--vlm-batch-size", type=int, default=4)
    parser.add_argument("--max-vlm-evidence", type=int, default=None)
    parser.add_argument("--semantic-duplicate-threshold", type=float, default=0.88)
    parser.add_argument("--demo-from", type=Path, nargs="+", help="Create a synthetic burst from source images.")
    parser.add_argument("--demo-variants", type=int, default=4)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process(args)


if __name__ == "__main__":
    main()
