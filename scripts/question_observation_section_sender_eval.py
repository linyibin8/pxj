"""Evaluate the live iOS dense section-crop sender policy.

This is narrower than question_observation_section_crop_sweep.py.  It mirrors
the Swift sender that adds section rect-only crops only when a frame is at
candidate-cap risk, then measures whether those section sources avoid backend
full-frame fallback without losing represented GT recall.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from question_observation_candidate_cap_sweep import (
    covered_gt_for_candidates,
    evaluate_cap,
    load_inputs,
    ranked_candidates,
)
from question_observation_dedupe_tune import Candidate, GTBox, intersection, is_weak_key, load_gt, rect_area
from question_observation_fallback_tune import current_backend_policy, rect_union_area


Rect = dict[str, float]


@dataclass
class SectionCrop:
    section_id: str
    image_key: str
    rect: Rect
    candidates: list[Candidate]
    reason: str = "limited_candidate_frame"
    strategy: str = "ios_page_dense_v1"

    @property
    def area(self) -> float:
        return rect_area(self.rect)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def image_size(path: Path, cache: dict[Path, tuple[int, int]]) -> tuple[int, int]:
    if path in cache:
        return cache[path]
    try:
        with Image.open(path) as raw:
            cache[path] = ImageOps.exif_transpose(raw).size
    except Exception:
        cache[path] = (0, 0)
    return cache[path]


def clamp_rect(rect: Rect) -> Rect:
    x = max(0.0, min(1.0, float(rect.get("x") or 0.0)))
    y = max(0.0, min(1.0, float(rect.get("y") or 0.0)))
    max_x = max(0.0, min(1.0, x + float(rect.get("width") or 0.0)))
    max_y = max(0.0, min(1.0, y + float(rect.get("height") or 0.0)))
    return {"x": x, "y": y, "width": max(0.0, max_x - x), "height": max(0.0, max_y - y)}


def union_rects(rects: list[Rect]) -> Rect:
    clamped = [clamp_rect(rect) for rect in rects]
    clamped = [rect for rect in clamped if rect["width"] > 0 and rect["height"] > 0]
    if not clamped:
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    x1 = min(rect["x"] for rect in clamped)
    y1 = min(rect["y"] for rect in clamped)
    x2 = max(rect["x"] + rect["width"] for rect in clamped)
    y2 = max(rect["y"] + rect["height"] for rect in clamped)
    return {"x": x1, "y": y1, "width": max(0.0, x2 - x1), "height": max(0.0, y2 - y1)}


def fit_interval(start: float, length: float) -> tuple[float, float]:
    fitted_length = max(0.0, min(1.0, length))
    fitted_start = max(0.0, min(1.0 - fitted_length, start))
    return fitted_start, min(1.0, fitted_start + fitted_length)


def section_crop_rect(input_rect: Rect, candidate_count: int, image_path: Path, size_cache: dict[Path, tuple[int, int]]) -> Rect:
    rect = clamp_rect(input_rect)
    if rect["width"] <= 0 or rect["height"] <= 0:
        return rect
    width, height = image_size(image_path, size_cache)
    wide_page = width >= height if width > 0 and height > 0 else False
    min_width = 0.76 if wide_page else 0.86
    horizontal_padding = 0.18 if candidate_count >= 5 else 0.12
    vertical_padding = 0.075 + min(0.06, candidate_count * 0.006)
    target_width = min(0.98, max(rect["width"] + horizontal_padding, min_width))
    max_height = 0.96 if candidate_count >= 12 else (0.72 if candidate_count >= 10 else 0.48)
    target_height = min(max_height, max(rect["height"] + vertical_padding, 0.18 if candidate_count >= 5 else 0.14))
    center_x = rect["x"] + rect["width"] * 0.5
    center_y = rect["y"] + rect["height"] * 0.5
    x1, x2 = fit_interval(center_x - target_width * 0.5, target_width)
    y1, y2 = fit_interval(center_y - target_height * 0.5, target_height)
    return {"x": x1, "y": y1, "width": max(0.0, x2 - x1), "height": max(0.0, y2 - y1)}


def empty_frame_section_rect(image_path: Path, size_cache: dict[Path, tuple[int, int]]) -> Rect:
    width, height = image_size(image_path, size_cache)
    wide_page = width >= height if width > 0 and height > 0 else False
    target_width = 0.895 if wide_page else 0.875
    target_height = 0.835 if wide_page else 0.855
    x1, x2 = fit_interval((1.0 - target_width) * 0.5, target_width)
    y1, y2 = fit_interval(0.095, target_height)
    return {"x": x1, "y": y1, "width": max(0.0, x2 - x1), "height": max(0.0, y2 - y1)}


def risk_frame_section_rect(input_rect: Rect, image_path: Path, size_cache: dict[Path, tuple[int, int]]) -> Rect:
    rect = clamp_rect(input_rect)
    width, height = image_size(image_path, size_cache)
    wide_page = width >= height if width > 0 and height > 0 else False
    target_width = 0.94 if wide_page else 0.88
    target_height = 0.82 if wide_page else 0.86
    center_x = rect["x"] + rect["width"] * 0.5
    x1, x2 = fit_interval(center_x - target_width * 0.5, target_width)
    y_start = 0.09 if wide_page else 0.075
    if rect["height"] > 0:
        y_start = min(y_start, max(0.0, rect["y"] - 0.08))
    y1, y2 = fit_interval(y_start, target_height)
    return {"x": x1, "y": y1, "width": max(0.0, x2 - x1), "height": max(0.0, y2 - y1)}


def rect_overlap_ratio(left: Rect, right: Rect) -> float:
    overlap = intersection(left, right)
    if overlap <= 0:
        return 0.0
    return overlap / max(1e-9, min(rect_area(left), rect_area(right)))


def build_ios_section_crops(
    image_key: str,
    image_path: Path,
    ranked: list[Candidate],
    skipped: list[Candidate],
    cap: int,
    limited_confidence_score: float,
    size_cache: dict[Path, tuple[int, int]],
) -> list[SectionCrop]:
    if len(ranked) <= cap:
        return []
    if not skipped:
        return []
    unique_candidates = sorted(
        ranked,
        key=lambda candidate: (
            candidate.rect["y"],
            candidate.rect["x"],
            candidate.index,
            candidate.candidate_id,
        ),
    )
    if len(unique_candidates) <= cap:
        return []
    section_count = 1
    sections: list[SectionCrop] = []
    for section_index in range(section_count):
        start = section_index * len(unique_candidates) // section_count
        end = (section_index + 1) * len(unique_candidates) // section_count
        if start >= end:
            continue
        chunk = unique_candidates[start:end]
        raw_rect = union_rects([candidate.rect for candidate in chunk])
        rect = section_crop_rect(raw_rect, len(chunk), image_path, size_cache)
        if rect["width"] <= 0 or rect["height"] <= 0:
            continue
        if any(rect_overlap_ratio(existing.rect, rect) >= 0.88 for existing in sections):
            continue
        sections.append(
            SectionCrop(
                section_id=f"ios_h{section_index + 1:02d}",
                image_key=image_key,
                rect=rect,
                candidates=chunk,
            )
        )
    return sections


def build_risk_frame_section_crop(
    image_key: str,
    image_path: Path,
    ranked: list[Candidate],
    skipped: list[Candidate],
    current_fallback_reason: str,
    low_confidence_score: float,
    size_cache: dict[Path, tuple[int, int]],
    max_candidates: int,
    min_section_area: float,
) -> list[SectionCrop]:
    if not ranked or skipped:
        return []
    if len(ranked) > max_candidates:
        return []
    weak_count = sum(1 for candidate in ranked if is_weak_key(candidate.question_key))
    low_conf_count = sum(1 for candidate in ranked if 0 < candidate.score < low_confidence_score)
    total_area = sum(candidate.area for candidate in ranked)
    max_area = max((candidate.area for candidate in ranked), default=0.0)
    low_coverage = total_area < 0.48 or max_area < 0.34
    weak_or_low_conf = weak_count == len(ranked) or low_conf_count == len(ranked)
    current_policy_risky = current_fallback_reason in {
        "single_low_coverage_crop",
        "sparse_low_coverage_crops",
        "tiny_crop_coverage",
        "no_large_question_crop",
        "all_weak_crop_keys",
        "low_confidence_crops",
    }
    if not (current_policy_risky or (weak_or_low_conf and low_coverage)):
        return []
    raw_rect = union_rects([candidate.rect for candidate in ranked])
    rect = risk_frame_section_rect(raw_rect, image_path, size_cache)
    if rect_area(rect) < min_section_area:
        return []
    section_id = f"section:{image_key}:risk:{rect['x']:.3f}_{rect['y']:.3f}_{rect['width']:.3f}_{rect['height']:.3f}"
    return [
        SectionCrop(
            section_id=section_id,
            image_key=image_key,
            rect=rect,
            candidates=ranked,
            reason=current_fallback_reason or "weak_low_coverage_frame",
            strategy="ios_page_risk_v1",
        )
    ]


def build_empty_frame_section_crop(
    image_key: str,
    image_path: Path,
    ranked: list[Candidate],
    size_cache: dict[Path, tuple[int, int]],
) -> list[SectionCrop]:
    if ranked:
        return []
    rect = empty_frame_section_rect(image_path, size_cache)
    if rect_area(rect) < 0.42:
        return []
    section_id = f"section:{image_key}:empty:{rect['x']:.3f}_{rect['y']:.3f}_{rect['width']:.3f}_{rect['height']:.3f}"
    return [
        SectionCrop(
            section_id=section_id,
            image_key=image_key,
            rect=rect,
            candidates=[],
            reason="no_candidate_study_frame",
            strategy="ios_page_empty_v1",
        )
    ]


def section_crop_pixels(rect: Rect, image_path: Path, max_side: int, size_cache: dict[Path, tuple[int, int]]) -> int:
    width, height = image_size(image_path, size_cache)
    if width <= 0 or height <= 0:
        return 0
    crop_width = max(1, min(width, int(round(rect["width"] * width))))
    crop_height = max(1, min(height, int(round(rect["height"] * height))))
    scale = min(1.0, max_side / max(crop_width, crop_height, 1))
    return max(1, int(crop_width * scale)) * max(1, int(crop_height * scale))


def section_covers_gt(section: SectionCrop, gt_box: GTBox, min_gt_coverage: float) -> bool:
    overlap = intersection(section.rect, gt_box.rect)
    return overlap > 0 and overlap / max(1e-9, rect_area(gt_box.rect)) >= min_gt_coverage


def section_aware_fallback_reason(stats: dict[str, Any], policy: dict[str, Any]) -> str:
    crop_count = int(stats.get("count") or 0)
    total_area = float(stats.get("area") or 0)
    max_area = float(stats.get("max_area") or 0)
    weak_count = int(stats.get("weak_count") or 0)
    low_conf_count = int(stats.get("low_conf_count") or 0)
    limited_count = int(stats.get("limited_count") or 0)
    limited_unique_count = int(stats.get("limited_unique_count") or 0)
    limited_strong_count = int(stats.get("limited_strong_count") or 0)
    limited_confident_count = int(stats.get("limited_confident_count") or 0)
    ranked_limit_telemetry_count = int(stats.get("ranked_limit_telemetry_count") or 0)
    section_count = int(stats.get("section_count") or 0)
    section_covered_question_count = int(stats.get("section_covered_question_count") or 0)
    section_union_area = float(stats.get("section_union_area") or 0)
    section_limited_protection = section_count > 0 and (
        section_covered_question_count >= max(1, limited_unique_count or limited_count)
        or section_union_area >= 0.42
    )
    section_fallback_protection = section_limited_protection
    if crop_count <= 0:
        return "no_crops"
    if limited_count > 0 and not section_limited_protection and (
        ranked_limit_telemetry_count <= 0 or limited_strong_count > 0 or limited_confident_count > 0
    ):
        return "limited_candidate_frame"
    if section_fallback_protection:
        return ""
    if crop_count <= 1 and total_area < float(policy["single_low_coverage_area"]):
        return "single_low_coverage_crop"
    if crop_count <= 2 and total_area < float(policy["sparse_low_coverage_area"]):
        return "sparse_low_coverage_crops"
    if total_area < float(policy["tiny_crop_coverage_area"]):
        return "tiny_crop_coverage"
    if max_area < float(policy["no_large_min_max_area"]) and total_area < float(policy["no_large_max_total_area"]):
        return "no_large_question_crop"
    if (
        weak_count >= crop_count
        and crop_count <= int(policy.get("all_weak_max_count") or 0)
        and total_area < float(policy.get("all_weak_max_total_area", 1.01))
        and max_area < float(policy.get("all_weak_max_max_area", 1.01))
    ):
        return "all_weak_crop_keys"
    if low_conf_count >= crop_count and crop_count <= int(policy.get("low_confidence_max_count") or 0):
        return "low_confidence_crops"
    return ""


def combined_stats(
    selected: list[Candidate],
    skipped: list[Candidate],
    sections: list[SectionCrop],
    low_confidence_score: float,
    limited_confidence_score: float,
) -> dict[str, Any]:
    rects = [candidate.rect for candidate in selected]
    section_rects = [section.rect for section in sections]
    all_rects = [*rects, *section_rects]
    total_area = sum(rect_area(rect) for rect in all_rects)
    union_area = rect_union_area(all_rects)
    section_union_area = rect_union_area(section_rects)
    return {
        "count": len(selected) + len(sections),
        "area": round(total_area, 6),
        "union_area": round(union_area, 6),
        "overlap_area": round(max(0.0, total_area - union_area), 6),
        "overlap_area_ratio": round(max(0.0, total_area - union_area) / max(1e-9, total_area), 6) if total_area else 0.0,
        "max_area": round(max((rect_area(rect) for rect in all_rects), default=0.0), 6),
        "weak_count": sum(1 for candidate in selected if is_weak_key(candidate.question_key)) + len(sections),
        "low_conf_count": sum(1 for candidate in selected if 0 < candidate.score < low_confidence_score),
        "selected_candidate_count": len(selected),
        "section_count": len(sections),
        "section_covered_question_count": sum(len(section.candidates) for section in sections),
        "section_union_area": round(section_union_area, 6),
        "limited_count": len(skipped),
        "limited_unique_count": len(skipped),
        "limited_strong_count": sum(1 for candidate in skipped if not is_weak_key(candidate.question_key)),
        "limited_confident_count": sum(1 for candidate in skipped if candidate.score >= limited_confidence_score),
        "limited_max_confidence": round(max((candidate.score for candidate in skipped), default=0.0), 6),
        "ranked_limit_telemetry_count": 1,
    }


def evaluate_sender(
    grouped: dict[str, list[Candidate]],
    gt_by_image: dict[str, dict[str, GTBox]],
    full_pixels_by_image: dict[str, int],
    image_paths_by_key: dict[str, Path],
    cap: int,
    section_max_side: int,
    min_gt_coverage: float,
    min_iou: float,
    low_confidence_score: float,
    limited_confidence_score: float,
    replace_selected_on_section: bool,
    selected_on_section_limit: int | None,
    enable_empty_frame_section: bool,
    enable_risk_frame_section: bool,
    risk_frame_max_candidates: int,
    risk_frame_min_section_area: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    represented_gt = {uid for values in gt_by_image.values() for uid in values}
    selected_gt_total: set[str] = set()
    section_gt_total: set[str] = set()
    combined_gt_total: set[str] = set()
    fallback_gt_total: set[str] = set()
    selected_pixels = 0
    section_pixels = 0
    fallback_pixels = 0
    cap_selected_count = 0
    uploaded_selected_count = 0
    skipped_count = 0
    section_count = 0
    section_frame_count = 0
    fallback_reason_counts: Counter[str] = Counter()
    policy = current_backend_policy()
    size_cache: dict[Path, tuple[int, int]] = {}
    per_image_rows: list[dict[str, Any]] = []
    section_rows: list[dict[str, Any]] = []

    for image_key, candidates in sorted(grouped.items()):
        image_path = image_paths_by_key.get(image_key, candidates[0].image_path if candidates else Path(""))
        image_gt = dict(gt_by_image.get(image_key) or {})
        ranked = ranked_candidates(candidates)
        selected = ranked[:cap] if cap > 0 else ranked
        skipped = ranked[cap:] if cap > 0 else []
        pre_section_stats = combined_stats(selected, skipped, [], low_confidence_score, limited_confidence_score)
        pre_section_reason = section_aware_fallback_reason(pre_section_stats, policy)
        sections = build_ios_section_crops(
            image_key,
            image_path,
            ranked,
            skipped,
            cap=cap,
            limited_confidence_score=limited_confidence_score,
            size_cache=size_cache,
        )
        if enable_risk_frame_section and not sections:
            sections = build_risk_frame_section_crop(
                image_key,
                image_path,
                ranked,
                skipped,
                current_fallback_reason=pre_section_reason,
                low_confidence_score=low_confidence_score,
                size_cache=size_cache,
                max_candidates=risk_frame_max_candidates,
                min_section_area=risk_frame_min_section_area,
            )
        if enable_empty_frame_section and not sections:
            sections = build_empty_frame_section_crop(image_key, image_path, ranked, size_cache)
        if sections and selected_on_section_limit is not None:
            upload_selected = selected[: max(0, selected_on_section_limit)]
        elif replace_selected_on_section and sections:
            upload_selected = []
        else:
            upload_selected = selected
        selected_gt = covered_gt_for_candidates(upload_selected, list(image_gt.values()), min_gt_coverage, min_iou, {})
        section_gt: set[str] = set()
        for section in sections:
            for gt_uid, gt in image_gt.items():
                if section_covers_gt(section, gt, min_gt_coverage):
                    section_gt.add(gt_uid)
        combined_gt = selected_gt | section_gt
        stats = combined_stats(upload_selected, skipped, sections, low_confidence_score, limited_confidence_score)
        reason = section_aware_fallback_reason(stats, policy)
        image_selected_pixels = sum(candidate.crop_resized_pixels for candidate in upload_selected)
        image_section_pixels = sum(section_crop_pixels(section.rect, image_path, section_max_side, size_cache) for section in sections)
        selected_pixels += image_selected_pixels
        section_pixels += image_section_pixels
        cap_selected_count += len(selected)
        uploaded_selected_count += len(upload_selected)
        skipped_count += len(skipped)
        section_count += len(sections)
        if sections:
            section_frame_count += 1
        if reason:
            fallback_reason_counts[reason] += 1
            fallback_pixels += full_pixels_by_image.get(image_key, 0)
            fallback_gt_total.update(image_gt)
        selected_gt_total.update(selected_gt)
        section_gt_total.update(section_gt)
        combined_gt_total.update(combined_gt)
        for section in sections:
            section_rows.append(
                {
                    "image_key": image_key,
                    "image": str(image_path),
                    "section_id": section.section_id,
                    "strategy": section.strategy,
                    "candidate_count": len(section.candidates),
                    "bbox_norm": {key: round(value, 6) for key, value in section.rect.items()},
                    "area": round(section.area, 6),
                    "candidate_ids": [candidate.candidate_id for candidate in section.candidates],
                    "question_keys": [candidate.question_key for candidate in section.candidates],
                }
            )
        per_image_rows.append(
            {
                "image_key": image_key,
                "image": str(image_path),
                "candidate_count": len(candidates),
                "cap_selected_count": len(selected),
                "uploaded_selected_count": len(upload_selected),
                "skipped_count": len(skipped),
                "section_count": len(sections),
                "gt_key_count": len(image_gt),
                "selected_gt_key_count": len(selected_gt),
                "section_gt_key_count": len(section_gt),
                "combined_gt_key_count": len(combined_gt),
                "missed_after_combined_keys": sorted(set(image_gt) - combined_gt),
                "fallback_reason": reason,
                "selected_pixels": image_selected_pixels,
                "section_pixels": image_section_pixels,
                "full_frame_pixels": full_pixels_by_image.get(image_key, 0),
                "stats": stats,
            }
        )

    policy_gt_total = combined_gt_total | fallback_gt_total
    full_pixels = sum(full_pixels_by_image.values())
    summary = {
        "candidate_cap": cap,
        "section_max_side": section_max_side,
        "replace_selected_on_section": replace_selected_on_section,
        "selected_on_section_limit": selected_on_section_limit,
        "image_count": len(grouped),
        "candidate_count": sum(len(items) for items in grouped.values()),
        "represented_gt_key_count": len(represented_gt),
        "cap_selected_candidate_count": cap_selected_count,
        "uploaded_selected_candidate_count": uploaded_selected_count,
        "skipped_candidate_count": skipped_count,
        "section_frame_count": section_frame_count,
        "section_count": section_count,
        "selected_covered_gt_key_count": len(selected_gt_total),
        "section_covered_gt_key_count": len(section_gt_total),
        "combined_crop_covered_gt_key_count": len(combined_gt_total),
        "policy_covered_gt_key_count": len(policy_gt_total),
        "selected_gt_recall": round(len(selected_gt_total) / max(1, len(represented_gt)), 6) if represented_gt else None,
        "section_only_gt_recall": round(len(section_gt_total) / max(1, len(represented_gt)), 6) if represented_gt else None,
        "combined_crop_gt_recall": round(len(combined_gt_total) / max(1, len(represented_gt)), 6) if represented_gt else None,
        "policy_gt_recall": round(len(policy_gt_total) / max(1, len(represented_gt)), 6) if represented_gt else None,
        "fallback_image_count": sum(fallback_reason_counts.values()),
        "fallback_image_ratio": round(sum(fallback_reason_counts.values()) / max(1, len(grouped)), 6),
        "fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
        "selected_crop_pixels": selected_pixels,
        "section_pixels": section_pixels,
        "fallback_full_frame_pixels": fallback_pixels,
        "total_vlm_pixels": selected_pixels + section_pixels + fallback_pixels,
        "full_frame_all_pixels": full_pixels,
        "selected_crop_pixels_vs_full": round(selected_pixels / max(1, full_pixels), 6),
        "section_pixels_vs_full": round(section_pixels / max(1, full_pixels), 6),
        "total_pixels_vs_full": round((selected_pixels + section_pixels + fallback_pixels) / max(1, full_pixels), 6),
    }
    return summary, per_image_rows, section_rows


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    sender = summary.get("sender") if isinstance(summary.get("sender"), dict) else {}
    cap_only = summary.get("cap_only") if isinstance(summary.get("cap_only"), dict) else {}
    lines = [
        "# Observation Section Sender Eval",
        "",
        f"- images: {summary.get('image_count')}",
        f"- candidates: {summary.get('candidate_count')}",
        f"- represented GT keys: {summary.get('represented_gt_key_count')}",
        "",
        "## Cap Only",
        "",
        f"- crop recall: {cap_only.get('selected_gt_recall')}",
        f"- policy recall: {cap_only.get('policy_gt_recall')}",
        f"- fallback images: {cap_only.get('fallback_image_count')}",
        f"- total pixels vs full: {cap_only.get('total_pixels_vs_full')}",
        "",
        "## iOS Section Sender",
        "",
        f"- selected crop recall: {sender.get('selected_gt_recall')}",
        f"- section-only recall: {sender.get('section_only_gt_recall')}",
        f"- combined crop recall: {sender.get('combined_crop_gt_recall')}",
        f"- policy recall: {sender.get('policy_gt_recall')}",
        f"- section frames: {sender.get('section_frame_count')}",
        f"- sections: {sender.get('section_count')}",
        f"- fallback images: {sender.get('fallback_image_count')}",
        f"- total pixels vs full: {sender.get('total_pixels_vs_full')}",
        f"- delta vs cap-only: {summary.get('total_pixels_delta_vs_cap_only')}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the live iOS dense section-crop sender policy.")
    parser.add_argument("--replay", type=Path, action="append", required=True)
    parser.add_argument("--ground-truth-manifest", type=Path, action="append", default=[])
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--candidate-cap", type=int, default=12)
    parser.add_argument("--section-max-side", type=int, default=1200)
    parser.add_argument("--min-gt-coverage", type=float, default=0.85)
    parser.add_argument("--min-iou", type=float, default=0.30)
    parser.add_argument("--low-confidence-score", type=float, default=0.42)
    parser.add_argument("--limited-confidence-score", type=float, default=0.55)
    parser.add_argument("--augment-selected-on-section", action="store_true", help="Diagnostic old behavior: send selected per-question crops even when section crops are present.")
    parser.add_argument("--selected-on-section-limit", type=int, default=None, help="When section crops are present, also send only the first N selected per-question crops. Defaults to 2 for the live iOS companion-anchor policy; use 0 to test section-only replacement.")
    parser.add_argument("--enable-empty-frame-section", action="store_true", help="Mirror the iOS no-candidate study-frame section fallback.")
    parser.add_argument("--enable-risk-frame-section", action="store_true", help="Evaluate a broad section companion for sparse weak/low-coverage frames that would otherwise rely on backend fallback or risky crop-only upload.")
    parser.add_argument("--risk-frame-max-candidates", type=int, default=4)
    parser.add_argument("--risk-frame-min-section-area", type=float, default=0.42)
    args = parser.parse_args()

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    gt_boxes = load_gt(args.ground_truth_manifest, args.dataset_root) if args.ground_truth_manifest else []
    grouped, gt_by_image, full_pixels_by_image, image_paths_by_key, input_summary = load_inputs(args.replay, gt_boxes)
    selected_on_section_limit = args.selected_on_section_limit
    if selected_on_section_limit is None and not args.augment_selected_on_section:
        selected_on_section_limit = 2
    replace_selected_on_section = not args.augment_selected_on_section
    cap_only, cap_risk_rows = evaluate_cap(
        args.candidate_cap,
        grouped,
        gt_boxes,
        gt_by_image,
        full_pixels_by_image,
        image_paths_by_key,
        min_gt_coverage=args.min_gt_coverage,
        min_iou=args.min_iou,
        low_confidence_score=args.low_confidence_score,
        limited_confidence_score=args.limited_confidence_score,
    )
    sender, per_image_rows, section_rows = evaluate_sender(
        grouped,
        gt_by_image,
        full_pixels_by_image,
        image_paths_by_key,
        cap=args.candidate_cap,
        section_max_side=args.section_max_side,
        min_gt_coverage=args.min_gt_coverage,
        min_iou=args.min_iou,
        low_confidence_score=args.low_confidence_score,
        limited_confidence_score=args.limited_confidence_score,
        replace_selected_on_section=replace_selected_on_section,
        selected_on_section_limit=selected_on_section_limit,
        enable_empty_frame_section=args.enable_empty_frame_section,
        enable_risk_frame_section=args.enable_risk_frame_section,
        risk_frame_max_candidates=args.risk_frame_max_candidates,
        risk_frame_min_section_area=args.risk_frame_min_section_area,
    )
    summary = {
        **input_summary,
        "settings": {
            "candidate_cap": args.candidate_cap,
            "section_max_side": args.section_max_side,
            "min_gt_coverage": args.min_gt_coverage,
            "min_iou": args.min_iou,
            "low_confidence_score": args.low_confidence_score,
            "limited_confidence_score": args.limited_confidence_score,
            "replace_selected_on_section": replace_selected_on_section,
            "selected_on_section_limit": selected_on_section_limit,
            "enable_empty_frame_section": args.enable_empty_frame_section,
            "enable_risk_frame_section": args.enable_risk_frame_section,
            "risk_frame_max_candidates": args.risk_frame_max_candidates,
            "risk_frame_min_section_area": args.risk_frame_min_section_area,
        },
        "cap_only": cap_only,
        "sender": sender,
        "total_pixels_delta_vs_cap_only": round(
            float(sender.get("total_pixels_vs_full") or 0) - float(cap_only.get("total_pixels_vs_full") or 0),
            6,
        ),
        "outputs": {
            "per_image": "per_image.jsonl",
            "sections": "sections.jsonl",
            "cap_risk": "cap_risk.jsonl",
            "summary_md": "summary.md",
        },
    }
    write_json(args.out / "summary.json", summary)
    write_jsonl(args.out / "per_image.jsonl", per_image_rows)
    write_jsonl(args.out / "sections.jsonl", section_rows)
    write_jsonl(args.out / "cap_risk.jsonl", cap_risk_rows)
    write_markdown(args.out / "summary.md", summary)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "images": summary.get("image_count"),
                "candidates": summary.get("candidate_count"),
                "represented_gt_keys": summary.get("represented_gt_key_count"),
                "cap_only": {
                    "recall": cap_only.get("policy_gt_recall"),
                    "total_pixels_vs_full": cap_only.get("total_pixels_vs_full"),
                    "fallback_images": cap_only.get("fallback_image_count"),
                },
                "sender": {
                    "combined_crop_recall": sender.get("combined_crop_gt_recall"),
                    "policy_recall": sender.get("policy_gt_recall"),
                    "total_pixels_vs_full": sender.get("total_pixels_vs_full"),
                    "fallback_images": sender.get("fallback_image_count"),
                    "sections": sender.get("section_count"),
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
