"""Tune observation question dedupe thresholds on replayed crop candidates.

The iOS observation path has two competing goals:

- keep every distinct question crop;
- suppress repeated crops before backend/VLM work.

This tool replays dedupe decisions offline against historical details.jsonl
files. It can evaluate the Swift signature-style rule used by
ContentView.observationQuestionSignatureSimilar and the image-hash/IoU rule used
by question_observation_replay.py. When reviewed detector labels are supplied,
it estimates whether a threshold would suppress a different reviewed question.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


Rect = dict[str, float]


@dataclass
class Candidate:
    candidate_id: str
    replay: str
    image_path: Path
    image_key: str
    question_key: str
    index: int
    rect: Rect
    area: float
    score: float
    crop_jpeg_bytes: int
    crop_resized_pixels: int
    quality_flags: list[str]
    gt_key: str = ""
    gt_iou: float = 0.0


@dataclass
class GTBox:
    gt_key: str
    image_id: str
    image_path: Path | None
    source_filename: str
    rect: Rect


@dataclass
class ReplayImageRef:
    image_path: Path
    image_key: str
    width: int = 0
    height: int = 0
    source_filename: str = ""


@dataclass
class Suppression:
    candidate: Candidate
    kept: Candidate
    reason: str
    overlap: float
    center_distance: float
    iou: float
    hash_distance: int | None = None


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def stable_id(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path, hash_size: int = 8) -> str:
    try:
        with Image.open(path) as image:
            gray = ImageOps.exif_transpose(image).convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    except Exception:
        return ""
    pixels = list(gray.getdata())
    if not pixels:
        return ""
    mean = sum(pixels) / len(pixels)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | (1 if pixel >= mean else 0)
    return f"{bits:0{hash_size * hash_size // 4}x}"


def hex_hamming(left: str, right: str) -> int:
    if not left or not right:
        return 999
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return 999


def to_rect(value: Any) -> Rect | None:
    if not isinstance(value, dict):
        return None
    try:
        x = float(value.get("x"))
        y = float(value.get("y"))
        width = float(value.get("width"))
        height = float(value.get("height"))
    except (TypeError, ValueError):
        return None
    x1 = max(0.0, min(1.0, x))
    y1 = max(0.0, min(1.0, y))
    x2 = max(0.0, min(1.0, x + width))
    y2 = max(0.0, min(1.0, y + height))
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if width <= 0 or height <= 0:
        return None
    return {"x": x1, "y": y1, "width": width, "height": height}


def rect_area(rect: Rect) -> float:
    return max(0.0, rect["width"]) * max(0.0, rect["height"])


def intersection(rect_a: Rect, rect_b: Rect) -> float:
    ax2 = rect_a["x"] + rect_a["width"]
    ay2 = rect_a["y"] + rect_a["height"]
    bx2 = rect_b["x"] + rect_b["width"]
    by2 = rect_b["y"] + rect_b["height"]
    width = max(0.0, min(ax2, bx2) - max(rect_a["x"], rect_b["x"]))
    height = max(0.0, min(ay2, by2) - max(rect_a["y"], rect_b["y"]))
    return width * height


def rect_iou(rect_a: Rect, rect_b: Rect) -> float:
    overlap = intersection(rect_a, rect_b)
    if overlap <= 0:
        return 0.0
    union = rect_area(rect_a) + rect_area(rect_b) - overlap
    return overlap / max(1e-9, union)


def overlap_smaller(rect_a: Rect, rect_b: Rect) -> float:
    overlap = intersection(rect_a, rect_b)
    if overlap <= 0:
        return 0.0
    return overlap / max(1e-9, min(rect_area(rect_a), rect_area(rect_b)))


def center_distance(rect_a: Rect, rect_b: Rect) -> float:
    ax = rect_a["x"] + rect_a["width"] / 2
    ay = rect_a["y"] + rect_a["height"] / 2
    bx = rect_b["x"] + rect_b["width"] / 2
    by = rect_b["y"] + rect_b["height"] / 2
    return math.hypot(ax - bx, ay - by)


def parse_index(question_key: str, fallback: int) -> int:
    tail = question_key.rsplit(":", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return fallback


def is_weak_key(key: str) -> bool:
    return key.startswith("layout:")


def weak_layout_text_part(key: str) -> str:
    if not is_weak_key(key):
        return ""
    parts = key.split(":")
    if len(parts) < 4:
        return ""
    text = parts[-1].strip().lower()
    if not text or text == "notext":
        return ""
    return text


def weak_layout_text_compatible(left: str, right: str) -> bool:
    left_text = weak_layout_text_part(left)
    right_text = weak_layout_text_part(right)
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True
    return near_key_similar(left_text, right_text)


def normalized_key(value: str) -> str:
    return "".join(ch.lower() for ch in value if not ch.isspace() and ch.isalnum())


def ngrams(value: str, size: int = 3) -> set[str]:
    chars = list(value)
    if len(chars) < size:
        return {value} if value else set()
    return {"".join(chars[index : index + size]) for index in range(len(chars) - size + 1)}


def trigram_jaccard(left: str, right: str) -> float:
    left_set = ngrams(left)
    right_set = ngrams(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def edit_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for i, left_ch in enumerate(left):
        current = [i + 1]
        for j, right_ch in enumerate(right):
            cost = 0 if left_ch == right_ch else 1
            current.append(min(previous[j + 1] + 1, current[j] + 1, previous[j] + cost))
        previous = current
    distance = previous[-1]
    return 1.0 - distance / max(len(left), len(right))


def near_key_similar(left: str, right: str) -> bool:
    left_norm = normalized_key(left)
    right_norm = normalized_key(right)
    if len(left_norm) < 8 or len(right_norm) < 8:
        return False
    return trigram_jaccard(left_norm, right_norm) >= 0.35 or edit_similarity(left_norm, right_norm) >= 0.62


def load_replay(replay_dir: Path) -> tuple[list[Candidate], dict[str, int]]:
    details_path = replay_dir / "details.jsonl"
    if not details_path.is_file():
        raise FileNotFoundError(f"replay details not found: {details_path}")
    candidates: list[Candidate] = []
    for row_index, row in enumerate(read_jsonl(details_path)):
        rect = to_rect(row.get("bbox_norm"))
        if rect is None:
            continue
        image_path = Path(str(row.get("image") or ""))
        question_key = str(row.get("question_key") or "").strip()
        candidates.append(
            Candidate(
                candidate_id=stable_id(f"{replay_dir}:{row_index}:{image_path}:{question_key}:{rect}"),
                replay=str(replay_dir),
                image_path=image_path,
                image_key=str(row.get("image_key") or image_path.stem),
                question_key=question_key,
                index=parse_index(question_key, row_index + 1),
                rect=rect,
                area=rect_area(rect),
                score=float(row.get("score") or 0.0),
                crop_jpeg_bytes=int(row.get("crop_jpeg_bytes") or 0),
                crop_resized_pixels=int(row.get("crop_resized_pixels") or 0),
                quality_flags=[str(flag) for flag in row.get("quality_flags") or []],
            )
        )
    summary: dict[str, int] = {}
    summary_path = replay_dir / "summary.json"
    if summary_path.is_file():
        payload = read_json(summary_path)
        upload = payload.get("upload_bytes") if isinstance(payload.get("upload_bytes"), dict) else {}
        vlm = payload.get("vlm_proxy") if isinstance(payload.get("vlm_proxy"), dict) else {}
        summary["full_frame_only_bytes"] = int(upload.get("full_frame_only") or 0)
        summary["full_frame_resized_pixels"] = int(vlm.get("full_frame_resized_pixels") or 0)
    return candidates, summary


def load_replay_images(replay_dir: Path) -> list[ReplayImageRef]:
    images_path = replay_dir / "images.jsonl"
    refs: dict[str, ReplayImageRef] = {}
    if images_path.is_file():
        for row in read_jsonl(images_path):
            image_path = Path(str(row.get("image") or ""))
            image_key = str(row.get("image_key") or image_path.stem)
            key = image_key or str(image_path)
            refs[key] = ReplayImageRef(
                image_path=image_path,
                image_key=image_key,
                width=int(row.get("width") or 0),
                height=int(row.get("height") or 0),
                source_filename=str(row.get("source_filename") or image_path.name),
            )
    if refs:
        return list(refs.values())
    details_path = replay_dir / "details.jsonl"
    if not details_path.is_file():
        return []
    for row in read_jsonl(details_path):
        image_path = Path(str(row.get("image") or ""))
        image_key = str(row.get("image_key") or image_path.stem)
        key = image_key or str(image_path)
        if key not in refs:
            refs[key] = ReplayImageRef(
                image_path=image_path,
                image_key=image_key,
                source_filename=str(row.get("source_filename") or image_path.name),
            )
    return list(refs.values())


def manifest_root(manifest_path: Path, explicit_root: Path | None) -> Path:
    if explicit_root is not None:
        return explicit_root
    if manifest_path.parent.name == "annotations":
        return manifest_path.parent.parent
    return manifest_path.parent


def row_image_path(root: Path, row: dict[str, Any]) -> Path | None:
    raw = str(row.get("image") or row.get("file_name") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    candidate = path if path.is_absolute() else root / path
    return candidate if candidate.is_file() else None


def load_gt(manifest_paths: list[Path], dataset_root: Path | None) -> list[GTBox]:
    boxes: list[GTBox] = []
    for manifest_path in manifest_paths:
        root = manifest_root(manifest_path, dataset_root)
        for row_index, row in enumerate(read_jsonl(manifest_path)):
            rect = to_rect(row.get("bbox_norm"))
            if rect is None:
                bbox_px = row.get("bbox_px")
                if isinstance(bbox_px, dict):
                    width = float(row.get("image_width") or row.get("width") or 0)
                    height = float(row.get("image_height") or row.get("height") or 0)
                    if width > 0 and height > 0:
                        rect = to_rect(
                            {
                                "x": float(bbox_px.get("x") or 0) / width,
                                "y": float(bbox_px.get("y") or 0) / height,
                                "width": float(bbox_px.get("width") or 0) / width,
                                "height": float(bbox_px.get("height") or 0) / height,
                            }
                        )
            if rect is None:
                continue
            image_path = row_image_path(root, row)
            source_filename = str(row.get("source_filename") or Path(str(row.get("image") or "")).name)
            image_id = source_filename or (image_path.name if image_path else f"{manifest_path}:{row_index}")
            gt_key = str(row.get("question_key") or f"gt:{image_id}:{row_index + 1}")
            boxes.append(GTBox(gt_key=gt_key, image_id=image_id, image_path=image_path, source_filename=source_filename, rect=rect))
    return boxes


def image_identity_keys(image_path: Path, image_key: str = "", source_filename: str = "") -> set[str]:
    keys = {image_path.name, image_path.stem, image_key, source_filename, Path(source_filename).name, Path(source_filename).stem}
    if image_path.is_file():
        try:
            keys.add(file_sha1(image_path))
        except OSError:
            pass
    return {key for key in keys if key}


def candidate_match_keys(candidate: Candidate) -> set[str]:
    return image_identity_keys(candidate.image_path, candidate.image_key)


def build_gt_indexes(gt_boxes: list[GTBox]) -> tuple[dict[str, list[GTBox]], dict[str, list[GTBox]], list[GTBox]]:
    by_key: dict[str, list[GTBox]] = {}
    by_hash: dict[str, list[GTBox]] = {}
    for box in gt_boxes:
        names = {box.source_filename, Path(box.source_filename).name, Path(box.source_filename).stem}
        if box.image_path is not None:
            names.update({box.image_path.name, box.image_path.stem})
            if box.image_path.is_file():
                try:
                    by_hash.setdefault(file_sha1(box.image_path), []).append(box)
                except OSError:
                    pass
        for name in names:
            if name:
                by_key.setdefault(name, []).append(box)
    return by_key, by_hash, gt_boxes


def find_gt_boxes_for_image_identity(
    image_path: Path,
    image_key: str,
    source_filename: str,
    by_key: dict[str, list[GTBox]],
    by_hash: dict[str, list[GTBox]],
    all_gt: list[GTBox],
) -> list[GTBox]:
    matched: dict[str, GTBox] = {}
    keys = image_identity_keys(image_path, image_key, source_filename)
    for key in keys:
        for box in by_key.get(key, []):
            matched[f"{box.image_id}:{box.gt_key}"] = box
        for box in by_hash.get(key, []):
            matched[f"{box.image_id}:{box.gt_key}"] = box
    if not matched:
        stem = image_path.stem
        for box in all_gt:
            source_stem = Path(box.source_filename).stem
            if stem and source_stem and (stem in source_stem or source_stem in stem):
                matched[f"{box.image_id}:{box.gt_key}"] = box
    return list(matched.values())


def find_gt_image_boxes(candidate: Candidate, by_key: dict[str, list[GTBox]], by_hash: dict[str, list[GTBox]], all_gt: list[GTBox]) -> list[GTBox]:
    return find_gt_boxes_for_image_identity(candidate.image_path, candidate.image_key, "", by_key, by_hash, all_gt)


def attach_gt(candidates: list[Candidate], gt_boxes: list[GTBox], match_iou: float) -> dict[str, Any]:
    if not gt_boxes:
        return {"gt_available": False, "represented_gt_key_count": 0, "matched_candidate_count": 0}
    by_key, by_hash, all_gt = build_gt_indexes(gt_boxes)
    represented_gt_keys: set[str] = set()
    matched_candidate_count = 0
    for candidate in candidates:
        image_gt = find_gt_image_boxes(candidate, by_key, by_hash, all_gt)
        for box in image_gt:
            represented_gt_keys.add(box.gt_key)
        best: tuple[float, str] = (0.0, "")
        for box in image_gt:
            iou = rect_iou(candidate.rect, box.rect)
            if iou > best[0]:
                best = (iou, box.gt_key)
        if best[0] >= match_iou:
            candidate.gt_iou = best[0]
            candidate.gt_key = best[1]
            matched_candidate_count += 1
    return {
        "gt_available": True,
        "gt_box_count": len(gt_boxes),
        "represented_gt_key_count": len(represented_gt_keys),
        "matched_candidate_count": matched_candidate_count,
        "represented_gt_keys": sorted(represented_gt_keys),
    }


def swift_duplicate_reason(candidate: Candidate, kept: Candidate, params: dict[str, Any]) -> tuple[bool, str, float, float]:
    overlap = overlap_smaller(candidate.rect, kept.rect)
    distance = center_distance(candidate.rect, kept.rect)
    same_index = candidate.index > 0 and candidate.index == kept.index
    left_weak = is_weak_key(candidate.question_key)
    right_weak = is_weak_key(kept.question_key)
    if left_weak or right_weak:
        if left_weak and right_weak and candidate.question_key == kept.question_key:
            if overlap >= params["exact_layout_overlap"] or distance <= params["exact_layout_center"]:
                return True, "same_layout_key", overlap, distance
        if left_weak and right_weak:
            if params.get("weak_text_guard") and not weak_layout_text_compatible(candidate.question_key, kept.question_key):
                return False, "", overlap, distance
            if overlap >= params["weak_overlap"]:
                return True, "weak_overlap", overlap, distance
            if distance <= params["weak_center"]:
                return True, "weak_center", overlap, distance
            if same_index and overlap >= params["same_index_overlap"] and distance <= params["same_index_center"]:
                return True, "weak_same_index", overlap, distance
            return False, "", overlap, distance
        if overlap >= params["mixed_overlap"] or (same_index and overlap >= params["mixed_same_index_overlap"] and distance <= params["mixed_same_index_center"]):
            return True, "mixed_key", overlap, distance
        return False, "", overlap, distance
    if candidate.question_key == kept.question_key or near_key_similar(candidate.question_key, kept.question_key):
        if overlap >= params["strong_overlap"] or distance <= params["strong_center"] or (same_index and distance <= params["strong_same_index_center"]):
            return True, "strong_key", overlap, distance
    return False, "", overlap, distance


def dedupe_swift(candidates: list[Candidate], params: dict[str, Any]) -> tuple[list[Candidate], list[Suppression]]:
    selected: list[Candidate] = []
    suppressions: list[Suppression] = []
    for candidate in sorted(candidates, key=lambda item: (item.score, item.area), reverse=True):
        duplicate = False
        for kept in selected:
            is_duplicate, reason, overlap, distance = swift_duplicate_reason(candidate, kept, params)
            if is_duplicate:
                suppressions.append(
                    Suppression(
                        candidate=candidate,
                        kept=kept,
                        reason=reason,
                        overlap=overlap,
                        center_distance=distance,
                        iou=rect_iou(candidate.rect, kept.rect),
                    )
                )
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
    return selected, suppressions


def dedupe_image_hash(candidates: list[Candidate], params: dict[str, Any], hashes: dict[Path, str]) -> tuple[list[Candidate], list[Suppression]]:
    selected: list[Candidate] = []
    suppressions: list[Suppression] = []
    for candidate in sorted(candidates, key=lambda item: (item.score, item.area), reverse=True):
        duplicate = False
        for kept in selected:
            overlap_iou = rect_iou(candidate.rect, kept.rect)
            if candidate.question_key and kept.question_key and candidate.question_key == kept.question_key:
                suppressions.append(
                    Suppression(candidate=candidate, kept=kept, reason="same_question_key", overlap=overlap_smaller(candidate.rect, kept.rect), center_distance=center_distance(candidate.rect, kept.rect), iou=overlap_iou)
                )
                duplicate = True
                break
            distance = hex_hamming(hashes.get(candidate.image_path, ""), hashes.get(kept.image_path, ""))
            if distance <= params["ahash_threshold"] and overlap_iou >= params["iou_threshold"]:
                suppressions.append(
                    Suppression(
                        candidate=candidate,
                        kept=kept,
                        reason="image_hash_iou",
                        overlap=overlap_smaller(candidate.rect, kept.rect),
                        center_distance=center_distance(candidate.rect, kept.rect),
                        iou=overlap_iou,
                        hash_distance=distance,
                    )
                )
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
    return selected, suppressions


def rect_manifest_bytes(candidates: list[Candidate]) -> int:
    payload = {
        "version": 2,
        "source": "offline-dedupe-tune",
        "crops": [
            {
                "source_image_key": candidate.image_key[:16],
                "question_key": candidate.question_key,
                "crop_rect": {key: round(value, 6) for key, value in candidate.rect.items()},
                "crop_area": round(candidate.area, 6),
                "transfer_mode": "rect_only",
                "crop_prepared": False,
            }
            for candidate in candidates
        ],
    }
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def summarize_run(
    strategy: str,
    params: dict[str, Any],
    candidates: list[Candidate],
    selected: list[Candidate],
    suppressions: list[Suppression],
    gt_info: dict[str, Any],
    full_frame_bytes: int,
    full_frame_pixels: int,
    baseline: bool,
) -> dict[str, Any]:
    raw_crop_bytes = sum(candidate.crop_jpeg_bytes for candidate in candidates)
    selected_crop_bytes = sum(candidate.crop_jpeg_bytes for candidate in selected)
    raw_pixels = sum(candidate.crop_resized_pixels for candidate in candidates)
    selected_pixels = sum(candidate.crop_resized_pixels for candidate in selected)
    selected_gt_keys = {candidate.gt_key for candidate in selected if candidate.gt_key}
    candidate_gt_keys = {candidate.gt_key for candidate in candidates if candidate.gt_key}
    represented_gt_keys = set(gt_info.get("represented_gt_keys") or [])
    denominator_keys = represented_gt_keys or candidate_gt_keys
    lost_gt_keys = sorted(denominator_keys - selected_gt_keys)
    false_merges = [
        item
        for item in suppressions
        if item.candidate.gt_key and item.kept.gt_key and item.candidate.gt_key != item.kept.gt_key
    ]
    unknown_gt_suppressions = [
        item
        for item in suppressions
        if item.candidate.gt_key and not item.kept.gt_key
    ]
    distinct_index_same_image = [
        item
        for item in suppressions
        if item.candidate.image_key == item.kept.image_key and item.candidate.index != item.kept.index
    ]
    gt_recall = len(selected_gt_keys & denominator_keys) / len(denominator_keys) if denominator_keys else None
    return {
        "strategy": strategy,
        "baseline": baseline,
        "params": params,
        "raw_count": len(candidates),
        "selected_count": len(selected),
        "removed_count": len(candidates) - len(selected),
        "removed_rate": round((len(candidates) - len(selected)) / max(1, len(candidates)), 6),
        "raw_crop_jpeg_bytes": raw_crop_bytes,
        "selected_crop_jpeg_bytes": selected_crop_bytes,
        "selected_crop_bytes_vs_raw": round(selected_crop_bytes / max(1, raw_crop_bytes), 6),
        "raw_crop_resized_pixels": raw_pixels,
        "selected_crop_resized_pixels": selected_pixels,
        "selected_crop_pixels_vs_raw": round(selected_pixels / max(1, raw_pixels), 6),
        "rect_manifest_bytes": rect_manifest_bytes(selected),
        "full_plus_selected_crop_vs_full": round((full_frame_bytes + selected_crop_bytes) / max(1, full_frame_bytes), 6) if full_frame_bytes else None,
        "rect_only_selected_vs_full": round((full_frame_bytes + rect_manifest_bytes(selected)) / max(1, full_frame_bytes), 6) if full_frame_bytes else None,
        "selected_crop_pixels_vs_full": round(selected_pixels / max(1, full_frame_pixels), 6) if full_frame_pixels else None,
        "gt_available": bool(gt_info.get("gt_available")),
        "represented_gt_key_count": len(denominator_keys),
        "selected_gt_key_count": len(selected_gt_keys & denominator_keys),
        "gt_recall": round(gt_recall, 6) if gt_recall is not None else None,
        "lost_gt_key_count": len(lost_gt_keys),
        "lost_gt_keys": lost_gt_keys[:30],
        "false_merge_count": len(false_merges),
        "unknown_gt_suppression_count": len(unknown_gt_suppressions),
        "same_image_distinct_index_suppression_count": len(distinct_index_same_image),
        "near_full_page_selected_count": sum(1 for item in selected if item.area >= 0.72 or "near_full_page" in item.quality_flags),
        "small_area_selected_count": sum(1 for item in selected if item.area < 0.025),
    }


def swift_param_grid() -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []
    for weak_text_guard in [False, True]:
        for weak_overlap in [0.46, 0.50, 0.52, 0.56, 0.60]:
            for weak_center in [0.06, 0.08, 0.10, 0.12]:
                for same_overlap in [0.14, 0.18, 0.22]:
                    for same_center in [0.14, 0.18, 0.22]:
                        params.append(
                            {
                                "exact_layout_overlap": 0.20,
                                "exact_layout_center": 0.20,
                                "weak_text_guard": weak_text_guard,
                                "weak_overlap": weak_overlap,
                                "weak_center": weak_center,
                                "same_index_overlap": same_overlap,
                                "same_index_center": same_center,
                                "mixed_overlap": 0.62,
                                "mixed_same_index_overlap": 0.24,
                                "mixed_same_index_center": 0.12,
                                "strong_overlap": 0.34,
                                "strong_center": 0.18,
                                "strong_same_index_center": 0.28,
                            }
                        )
    return params


def image_hash_param_grid() -> list[dict[str, Any]]:
    return [
        {"ahash_threshold": ahash, "iou_threshold": iou}
        for ahash in [0, 2, 4, 6, 8]
        for iou in [0.45, 0.55, 0.60, 0.70, 0.82]
    ]


def is_swift_baseline(params: dict[str, Any]) -> bool:
    return (
        params["weak_overlap"] == 0.52
        and params["weak_center"] == 0.08
        and params["same_index_overlap"] == 0.18
        and params["same_index_center"] == 0.18
    )


def is_hash_baseline(params: dict[str, Any]) -> bool:
    return params["ahash_threshold"] == 4 and params["iou_threshold"] == 0.60


def choose_recommendation(rows: list[dict[str, Any]], min_gt_recall: float, max_false_merges: int) -> dict[str, Any] | None:
    eligible: list[dict[str, Any]] = []
    for row in rows:
        recall = row.get("gt_recall")
        if recall is not None and recall < min_gt_recall:
            continue
        if int(row.get("false_merge_count") or 0) > max_false_merges:
            continue
        eligible.append(row)
    if not eligible:
        return None
    eligible.sort(
        key=lambda row: (
            row["selected_crop_pixels_vs_raw"],
            row["selected_crop_bytes_vs_raw"],
            row["selected_count"],
            -(row.get("gt_recall") if row.get("gt_recall") is not None else 1.0),
            row["same_image_distinct_index_suppression_count"],
        )
    )
    return eligible[0]


def suppression_row(item: Suppression) -> dict[str, Any]:
    return {
        "candidate_id": item.candidate.candidate_id,
        "kept_id": item.kept.candidate_id,
        "reason": item.reason,
        "overlap": round(item.overlap, 6),
        "center_distance": round(item.center_distance, 6),
        "iou": round(item.iou, 6),
        "hash_distance": item.hash_distance,
        "candidate_question_key": item.candidate.question_key,
        "kept_question_key": item.kept.question_key,
        "candidate_gt_key": item.candidate.gt_key,
        "kept_gt_key": item.kept.gt_key,
        "candidate_image": str(item.candidate.image_path),
        "kept_image": str(item.kept.image_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune observation question dedupe thresholds.")
    parser.add_argument("--replay", type=Path, action="append", default=[], help="Replay output dir containing details.jsonl.")
    parser.add_argument("--ground-truth-manifest", type=Path, action="append", default=[], help="Dataset annotations/manifest.jsonl with reviewed question boxes.")
    parser.add_argument("--dataset-root", type=Path, help="Dataset root for all --ground-truth-manifest paths.")
    parser.add_argument("--strategy", choices=["swift_signature", "image_hash_iou", "both"], default="both")
    parser.add_argument("--gt-match-iou", type=float, default=0.45)
    parser.add_argument("--min-gt-recall", type=float, default=0.999)
    parser.add_argument("--max-false-merges", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-observation-dedupe-tune"))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if not args.replay:
        parser.error("provide at least one --replay")
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    candidates: list[Candidate] = []
    full_frame_bytes = 0
    full_frame_pixels = 0
    replay_summaries: list[dict[str, Any]] = []
    for replay in args.replay:
        replay_candidates, replay_summary = load_replay(replay)
        candidates.extend(replay_candidates)
        full_frame_bytes += replay_summary.get("full_frame_only_bytes", 0)
        full_frame_pixels += replay_summary.get("full_frame_resized_pixels", 0)
        replay_summaries.append({"path": str(replay), "candidate_count": len(replay_candidates), **replay_summary})

    gt_boxes = load_gt(args.ground_truth_manifest, args.dataset_root) if args.ground_truth_manifest else []
    gt_info = attach_gt(candidates, gt_boxes, args.gt_match_iou)
    hashes = {candidate.image_path: average_hash(candidate.image_path) for candidate in candidates if candidate.image_path.is_file()}

    rows: list[dict[str, Any]] = []
    baseline_suppressions: dict[str, list[Suppression]] = {}
    if args.strategy in {"swift_signature", "both"}:
        for params in swift_param_grid():
            selected, suppressions = dedupe_swift(candidates, params)
            baseline = is_swift_baseline(params)
            rows.append(summarize_run("swift_signature", params, candidates, selected, suppressions, gt_info, full_frame_bytes, full_frame_pixels, baseline))
            if baseline:
                baseline_suppressions["swift_signature"] = suppressions
    if args.strategy in {"image_hash_iou", "both"}:
        for params in image_hash_param_grid():
            selected, suppressions = dedupe_image_hash(candidates, params, hashes)
            baseline = is_hash_baseline(params)
            rows.append(summarize_run("image_hash_iou", params, candidates, selected, suppressions, gt_info, full_frame_bytes, full_frame_pixels, baseline))
            if baseline:
                baseline_suppressions["image_hash_iou"] = suppressions

    recommendation = choose_recommendation(rows, args.min_gt_recall, args.max_false_merges)
    baselines = [row for row in rows if row.get("baseline")]
    rows_sorted = sorted(rows, key=lambda row: (row["strategy"], row["selected_crop_pixels_vs_raw"], row["selected_count"]))
    write_jsonl(args.out / "grid.jsonl", rows_sorted)
    for strategy, suppressions in baseline_suppressions.items():
        write_jsonl(args.out / f"{strategy}_baseline_suppressions.jsonl", [suppression_row(item) for item in suppressions])

    summary = {
        "inputs": {
            "replays": replay_summaries,
            "ground_truth_manifests": [str(path) for path in args.ground_truth_manifest],
            "dataset_root": str(args.dataset_root) if args.dataset_root else "",
        },
        "settings": {
            "strategy": args.strategy,
            "gt_match_iou": args.gt_match_iou,
            "min_gt_recall": args.min_gt_recall,
            "max_false_merges": args.max_false_merges,
        },
        "candidate_count": len(candidates),
        "gt": {key: value for key, value in gt_info.items() if key != "represented_gt_keys"},
        "baselines": baselines,
        "recommendation": recommendation,
        "outputs": {
            "grid": "grid.jsonl",
            "swift_baseline_suppressions": "swift_signature_baseline_suppressions.jsonl" if "swift_signature" in baseline_suppressions else "",
            "image_hash_baseline_suppressions": "image_hash_iou_baseline_suppressions.jsonl" if "image_hash_iou" in baseline_suppressions else "",
        },
    }
    write_json(args.out / "summary.json", summary)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "candidate_count": len(candidates),
                "gt_available": gt_info.get("gt_available", False),
                "baseline_count": len(baselines),
                "recommendation": {
                    "strategy": recommendation.get("strategy") if recommendation else "",
                    "params": recommendation.get("params") if recommendation else {},
                    "selected_count": recommendation.get("selected_count") if recommendation else None,
                    "selected_crop_pixels_vs_raw": recommendation.get("selected_crop_pixels_vs_raw") if recommendation else None,
                    "gt_recall": recommendation.get("gt_recall") if recommendation else None,
                    "false_merge_count": recommendation.get("false_merge_count") if recommendation else None,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
