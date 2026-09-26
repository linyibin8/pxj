"""Tune full-frame fallback policy for observation question crops.

The crop pipeline has a product-level tradeoff:

- crop-only VLM inputs are faster and cheaper;
- full-frame fallback protects recall when local crops look suspicious.

This tool replays historical crop candidates and optional reviewed GT boxes,
then sweeps backend-like fallback thresholds to find policies that preserve
question recall with the least full-frame VLM cost.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from question_observation_dedupe_tune import (
    Candidate,
    GTBox,
    build_gt_indexes,
    find_gt_boxes_for_image_identity,
    find_gt_image_boxes,
    intersection,
    is_weak_key,
    load_replay_images,
    load_gt,
    load_replay,
    rect_area,
    rect_iou,
)


@dataclass
class ImageState:
    image_key: str
    image_path: Path
    candidates: list[Candidate]
    gt_boxes: list[GTBox]
    gt_uids: set[str]
    crop_covered_gt_uids: set[str]
    full_frame_pixels: int
    crop_pixels: int
    crop_bytes: int
    stats: dict[str, Any]


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


def parse_float_list(value: str) -> list[float]:
    result: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        result.append(float(item))
    return result


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        result.append(int(item))
    return result


def resized_pixel_count(path: Path, max_side: int) -> int:
    try:
        with Image.open(path) as image:
            width, height = ImageOps.exif_transpose(image).size
    except Exception:
        return 0
    longest = max(width, height, 1)
    scale = min(1.0, max_side / longest)
    return max(1, int(width * scale)) * max(1, int(height * scale))


def replay_full_max_side(replay_dir: Path, default: int) -> int:
    summary_path = replay_dir / "summary.json"
    if not summary_path.is_file():
        return default
    try:
        summary = read_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return default
    settings = summary.get("settings") if isinstance(summary.get("settings"), dict) else {}
    try:
        return int(settings.get("full_max_side") or default)
    except (TypeError, ValueError):
        return default


def gt_uid(box: GTBox) -> str:
    return f"{box.image_id}:{box.gt_key}"


def crop_covers_gt(candidate: Candidate, gt_box: GTBox, min_gt_coverage: float, min_iou: float) -> bool:
    overlap = intersection(candidate.rect, gt_box.rect)
    if overlap <= 0:
        return False
    coverage = overlap / max(1e-9, rect_area(gt_box.rect))
    return coverage >= min_gt_coverage or rect_iou(candidate.rect, gt_box.rect) >= min_iou


def rect_union_area(rects: list[dict[str, float]]) -> float:
    events: list[tuple[float, int, float, float]] = []
    for rect in rects:
        x1 = max(0.0, min(1.0, float(rect.get("x") or 0.0)))
        y1 = max(0.0, min(1.0, float(rect.get("y") or 0.0)))
        x2 = max(0.0, min(1.0, x1 + float(rect.get("width") or 0.0)))
        y2 = max(0.0, min(1.0, y1 + float(rect.get("height") or 0.0)))
        if x2 <= x1 or y2 <= y1:
            continue
        events.append((x1, 1, y1, y2))
        events.append((x2, -1, y1, y2))
    if not events:
        return 0.0
    events.sort(key=lambda item: item[0])
    active: list[tuple[float, float]] = []
    previous_x = events[0][0]
    area = 0.0

    def active_height() -> float:
        if not active:
            return 0.0
        merged = 0.0
        current_start, current_end = sorted(active)[0]
        for start, end in sorted(active)[1:]:
            if start > current_end:
                merged += current_end - current_start
                current_start, current_end = start, end
            else:
                current_end = max(current_end, end)
        merged += current_end - current_start
        return merged

    for x, kind, y1, y2 in events:
        width = max(0.0, x - previous_x)
        if width > 0:
            area += width * active_height()
        if kind > 0:
            active.append((y1, y2))
        else:
            try:
                active.remove((y1, y2))
            except ValueError:
                pass
        previous_x = x
    return max(0.0, min(1.0, area))


def ranked_upload_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            is_weak_key(candidate.question_key),
            -candidate.score,
            -candidate.area,
            candidate.index,
        ),
    )


def image_stats(
    candidates: list[Candidate],
    low_confidence_score: float,
    candidate_limit: int,
    limited_confidence_score: float,
) -> dict[str, Any]:
    count = len(candidates)
    total_area = sum(candidate.area for candidate in candidates)
    union_area = rect_union_area([candidate.rect for candidate in candidates])
    max_area = max((candidate.area for candidate in candidates), default=0.0)
    weak_count = sum(1 for candidate in candidates if is_weak_key(candidate.question_key))
    low_conf_count = sum(
        1
        for candidate in candidates
        if candidate.score > 0 and candidate.score < low_confidence_score
    )
    ranked = ranked_upload_candidates(candidates)
    selected_count = len(ranked[:candidate_limit]) if candidate_limit > 0 else len(ranked)
    limited_candidates = ranked[candidate_limit:] if candidate_limit > 0 else []
    limited_strong_count = sum(1 for candidate in limited_candidates if not is_weak_key(candidate.question_key))
    limited_confident_count = sum(1 for candidate in limited_candidates if candidate.score >= limited_confidence_score)
    limited_max_confidence = max((candidate.score for candidate in limited_candidates), default=0.0)
    quality_flags = Counter(flag for candidate in candidates for flag in candidate.quality_flags)
    return {
        "count": count,
        "area": round(total_area, 6),
        "union_area": round(union_area, 6),
        "overlap_area": round(max(0.0, total_area - union_area), 6),
        "overlap_area_ratio": round(max(0.0, total_area - union_area) / max(1e-9, total_area), 6) if total_area > 0 else 0.0,
        "max_area": round(max_area, 6),
        "weak_count": weak_count,
        "low_conf_count": low_conf_count,
        "selected_candidate_count": selected_count,
        "limited_count": len(limited_candidates),
        "limited_unique_count": len(limited_candidates),
        "limited_strong_count": limited_strong_count,
        "limited_confident_count": limited_confident_count,
        "limited_max_confidence": round(limited_max_confidence, 6),
        "ranked_limit_telemetry_count": 1,
        "quality_flags": dict(sorted(quality_flags.items())),
    }


def build_image_states(
    replay_dirs: list[Path],
    gt_boxes: list[GTBox],
    min_gt_coverage: float,
    min_iou: float,
    full_max_side: int,
    low_confidence_score: float,
    candidate_limit: int,
    limited_confidence_score: float,
) -> tuple[list[ImageState], dict[str, Any]]:
    candidates: list[Candidate] = []
    replay_summaries: list[dict[str, Any]] = []
    replay_images = []
    for replay_dir in replay_dirs:
        replay_candidates, replay_summary = load_replay(replay_dir)
        loaded_images = load_replay_images(replay_dir)
        candidates.extend(replay_candidates)
        replay_images.extend(loaded_images)
        replay_summaries.append(
            {
                "path": str(replay_dir),
                "candidate_count": len(replay_candidates),
                "image_count": len(loaded_images),
                **replay_summary,
            }
        )

    grouped: dict[str, list[Candidate]] = {}
    image_paths_by_key: dict[str, Path] = {}
    gt_by_image: dict[str, dict[str, GTBox]] = {}
    by_key, by_hash, all_gt = build_gt_indexes(gt_boxes)
    for image_ref in replay_images:
        key = image_ref.image_key or str(image_ref.image_path)
        grouped.setdefault(key, [])
        image_paths_by_key[key] = image_ref.image_path
        matched_gt: dict[str, GTBox] = {}
        for box in find_gt_boxes_for_image_identity(
            image_ref.image_path,
            image_ref.image_key,
            image_ref.source_filename,
            by_key,
            by_hash,
            all_gt,
        ):
            matched_gt[gt_uid(box)] = box
        gt_by_image[key] = matched_gt
    for candidate in candidates:
        key = candidate.image_key or str(candidate.image_path)
        grouped.setdefault(key, []).append(candidate)
        image_paths_by_key.setdefault(key, candidate.image_path)

    states: list[ImageState] = []
    represented_gt_uids: set[str] = set()
    crop_covered_gt_uids: set[str] = set()

    for image_key, image_candidates in sorted(grouped.items()):
        image_path = image_candidates[0].image_path if image_candidates else image_paths_by_key.get(image_key, Path(""))
        matched_gt: dict[str, GTBox] = dict(gt_by_image.get(image_key) or {})
        for candidate in image_candidates:
            for box in find_gt_image_boxes(candidate, by_key, by_hash, all_gt):
                matched_gt[gt_uid(box)] = box
        gt_for_image = list(matched_gt.values())
        image_gt_uids = set(matched_gt.keys())
        represented_gt_uids.update(image_gt_uids)

        covered: set[str] = set()
        for candidate in image_candidates:
            for box in gt_for_image:
                uid = gt_uid(box)
                if uid in covered:
                    continue
                if crop_covers_gt(candidate, box, min_gt_coverage=min_gt_coverage, min_iou=min_iou):
                    covered.add(uid)
        crop_covered_gt_uids.update(covered)

        states.append(
            ImageState(
                image_key=image_key,
                image_path=image_path,
                candidates=image_candidates,
                gt_boxes=gt_for_image,
                gt_uids=image_gt_uids,
                crop_covered_gt_uids=covered,
                full_frame_pixels=resized_pixel_count(image_path, max_side=full_max_side),
                crop_pixels=sum(candidate.crop_resized_pixels for candidate in image_candidates),
                crop_bytes=sum(candidate.crop_jpeg_bytes for candidate in image_candidates),
                stats=image_stats(
                    image_candidates,
                    low_confidence_score=low_confidence_score,
                    candidate_limit=candidate_limit,
                    limited_confidence_score=limited_confidence_score,
                ),
            )
        )

    summary = {
        "replays": replay_summaries,
        "candidate_count": len(candidates),
        "image_count": len(states),
        "gt_available": bool(gt_boxes),
        "gt_box_count": len(gt_boxes),
        "represented_gt_key_count": len(represented_gt_uids),
        "crop_only_covered_gt_key_count": len(crop_covered_gt_uids),
        "crop_only_gt_recall": round(len(crop_covered_gt_uids) / max(1, len(represented_gt_uids)), 6) if represented_gt_uids else None,
    }
    return states, summary


def current_backend_policy() -> dict[str, Any]:
    return {
        "policy_id": "backend_current",
        "single_low_coverage_area": 0.30,
        "sparse_low_coverage_area": 0.25,
        "tiny_crop_coverage_area": 0.12,
        "no_large_min_max_area": 0.12,
        "no_large_max_total_area": 0.40,
        "all_weak_max_count": 0,
        "all_weak_max_total_area": 0.40,
        "all_weak_max_max_area": 0.20,
        "low_confidence_max_count": 0,
        "low_confidence_score": 0.42,
    }


def legacy_backend_policy() -> dict[str, Any]:
    return {
        "policy_id": "backend_legacy_before_tuned",
        "single_low_coverage_area": 0.42,
        "sparse_low_coverage_area": 0.34,
        "tiny_crop_coverage_area": 0.18,
        "no_large_min_max_area": 0.16,
        "no_large_max_total_area": 0.50,
        "all_weak_max_count": 3,
        "all_weak_max_total_area": 0.50,
        "all_weak_max_max_area": 0.34,
        "low_confidence_max_count": 3,
        "low_confidence_score": 0.42,
    }


def current_backend_union_area_policy() -> dict[str, Any]:
    policy = current_backend_policy()
    policy["policy_id"] = "backend_current_union_area_shadow"
    policy["coverage_area"] = "union"
    return policy


def fallback_reason(stats: dict[str, Any], policy: dict[str, Any]) -> str:
    crop_count = int(stats.get("count") or 0)
    coverage_area_key = "union_area" if str(policy.get("coverage_area") or "sum") == "union" else "area"
    total_area = float(stats.get(coverage_area_key) or 0)
    max_area = float(stats.get("max_area") or 0)
    weak_count = int(stats.get("weak_count") or 0)
    low_conf_count = int(stats.get("low_conf_count") or 0)
    limited_count = int(stats.get("limited_count") or 0)
    ranked_limit_telemetry_count = int(stats.get("ranked_limit_telemetry_count") or 0)
    limited_strong_count = int(stats.get("limited_strong_count") or 0)
    limited_confident_count = int(stats.get("limited_confident_count") or 0)
    if crop_count <= 0:
        return "no_crops"
    if limited_count > 0 and (
        ranked_limit_telemetry_count <= 0
        or limited_strong_count > 0
        or limited_confident_count > 0
    ):
        return "limited_candidate_frame"
    if crop_count <= 1 and total_area < float(policy["single_low_coverage_area"]):
        return "single_low_coverage_crop"
    if crop_count <= 2 and total_area < float(policy["sparse_low_coverage_area"]):
        return "sparse_low_coverage_crops"
    if total_area < float(policy["tiny_crop_coverage_area"]):
        return "tiny_crop_coverage"
    if max_area < float(policy["no_large_min_max_area"]) and total_area < float(policy["no_large_max_total_area"]):
        return "no_large_question_crop"
    all_weak_limit = int(policy.get("all_weak_max_count") or 0)
    if (
        all_weak_limit > 0
        and weak_count >= crop_count
        and crop_count <= all_weak_limit
        and total_area < float(policy.get("all_weak_max_total_area", 1.01))
        and max_area < float(policy.get("all_weak_max_max_area", 1.01))
    ):
        return "all_weak_crop_keys"
    low_conf_limit = int(policy.get("low_confidence_max_count") or 0)
    if low_conf_limit > 0 and low_conf_count >= crop_count and crop_count <= low_conf_limit:
        return "low_confidence_crops"
    return ""


def policy_id(policy: dict[str, Any]) -> str:
    named_policy = str(policy.get("policy_id") or "")
    if named_policy in {"backend_current", "backend_current_union_area_shadow", "backend_legacy_before_tuned"}:
        return named_policy
    mode = "union" if str(policy.get("coverage_area") or "sum") == "union" else "sum"
    parts = [
        mode,
        f"s{policy['single_low_coverage_area']:.2f}",
        f"p{policy['sparse_low_coverage_area']:.2f}",
        f"t{policy['tiny_crop_coverage_area']:.2f}",
        f"m{policy['no_large_min_max_area']:.2f}",
        f"n{policy['no_large_max_total_area']:.2f}",
        f"w{policy['all_weak_max_count']}",
        f"wa{policy.get('all_weak_max_total_area', 1.01):.2f}",
        f"wm{policy.get('all_weak_max_max_area', 1.01):.2f}",
        f"l{policy['low_confidence_max_count']}",
    ]
    return "__".join(parts).replace(".", "p")


def evaluate_policy(states: list[ImageState], policy: dict[str, Any]) -> dict[str, Any]:
    represented_gt = {uid for state in states for uid in state.gt_uids}
    crop_covered_gt = {uid for state in states for uid in state.crop_covered_gt_uids}
    policy_covered_gt = set(crop_covered_gt)
    fallback_images: list[ImageState] = []
    reason_counts: Counter[str] = Counter()

    crop_pixels = sum(state.crop_pixels for state in states)
    crop_bytes = sum(state.crop_bytes for state in states)
    full_pixels = sum(state.full_frame_pixels for state in states)
    fallback_pixels = 0

    missed_without_fallback = 0
    protected_missed_gt = 0
    for state in states:
        missing_gt = state.gt_uids - state.crop_covered_gt_uids
        if missing_gt:
            missed_without_fallback += len(missing_gt)
        reason = fallback_reason(state.stats, policy)
        if not reason:
            continue
        fallback_images.append(state)
        reason_counts[reason] += 1
        fallback_pixels += state.full_frame_pixels
        policy_covered_gt.update(state.gt_uids)
        protected_missed_gt += len(missing_gt)

    gt_recall = (
        len(policy_covered_gt) / max(1, len(represented_gt))
        if represented_gt
        else None
    )
    crop_gt_recall = (
        len(crop_covered_gt) / max(1, len(represented_gt))
        if represented_gt
        else None
    )
    total_vlm_pixels = crop_pixels + fallback_pixels
    return {
        "policy_id": policy_id(policy),
        "policy": {key: value for key, value in policy.items() if key != "policy_id"},
        "image_count": len(states),
        "fallback_image_count": len(fallback_images),
        "fallback_image_ratio": round(len(fallback_images) / max(1, len(states)), 6),
        "fallback_reason_counts": dict(sorted(reason_counts.items())),
        "represented_gt_key_count": len(represented_gt),
        "crop_only_covered_gt_key_count": len(crop_covered_gt),
        "crop_only_gt_recall": round(crop_gt_recall, 6) if crop_gt_recall is not None else None,
        "policy_covered_gt_key_count": len(policy_covered_gt),
        "policy_gt_recall": round(gt_recall, 6) if gt_recall is not None else None,
        "crop_missed_gt_key_count": missed_without_fallback,
        "fallback_protected_missed_gt_key_count": protected_missed_gt,
        "crop_vlm_pixels": crop_pixels,
        "fallback_full_frame_pixels": fallback_pixels,
        "total_vlm_pixels": total_vlm_pixels,
        "full_frame_all_pixels": full_pixels,
        "crop_pixels_vs_full": round(crop_pixels / max(1, full_pixels), 6),
        "total_pixels_vs_full": round(total_vlm_pixels / max(1, full_pixels), 6),
        "crop_jpeg_bytes": crop_bytes,
    }


def iter_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = [current_backend_policy(), current_backend_union_area_policy(), legacy_backend_policy()]
    seen = {policy_id(policy) for policy in policies}
    for single_area in parse_float_list(args.single_low_coverage_areas):
        for sparse_area in parse_float_list(args.sparse_low_coverage_areas):
            for tiny_area in parse_float_list(args.tiny_crop_coverage_areas):
                for no_large_area in parse_float_list(args.no_large_min_max_areas):
                    for no_large_total in parse_float_list(args.no_large_max_total_areas):
                        for all_weak_limit in parse_int_list(args.all_weak_max_counts):
                            for all_weak_total in parse_float_list(args.all_weak_max_total_areas):
                                for all_weak_max_area in parse_float_list(args.all_weak_max_max_areas):
                                    for low_conf_limit in parse_int_list(args.low_confidence_max_counts):
                                        for coverage_area in ("sum", "union"):
                                            policy = {
                                                "coverage_area": coverage_area,
                                                "single_low_coverage_area": single_area,
                                                "sparse_low_coverage_area": sparse_area,
                                                "tiny_crop_coverage_area": tiny_area,
                                                "no_large_min_max_area": no_large_area,
                                                "no_large_max_total_area": no_large_total,
                                                "all_weak_max_count": all_weak_limit,
                                                "all_weak_max_total_area": all_weak_total,
                                                "all_weak_max_max_area": all_weak_max_area,
                                                "low_confidence_max_count": low_conf_limit,
                                                "low_confidence_score": args.low_confidence_score,
                                            }
                                            key = policy_id(policy)
                                            if key in seen:
                                                continue
                                            seen.add(key)
                                            policies.append(policy)
    return policies


def choose_recommendation(rows: list[dict[str, Any]], min_gt_recall: float) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {}
    rows_with_gt = [row for row in rows if row.get("policy_gt_recall") is not None]
    if not rows_with_gt:
        return next((row for row in rows if row.get("policy_id") == "backend_current"), rows[0])
    passing = [row for row in rows_with_gt if float(row.get("policy_gt_recall") or 0) >= min_gt_recall]
    if passing:
        return sorted(
            passing,
            key=lambda row: (
                float(row.get("total_pixels_vs_full") or 999),
                int(row.get("fallback_image_count") or 999999),
                -float(row.get("policy_gt_recall") or 0),
            ),
        )[0]
    return sorted(
        rows_with_gt,
        key=lambda row: (
            -float(row.get("policy_gt_recall") or 0),
            float(row.get("total_pixels_vs_full") or 999),
            int(row.get("fallback_image_count") or 999999),
        ),
    )[0]


def image_risk_rows(states: list[ImageState], baseline: dict[str, Any], recommended: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in sorted(states, key=lambda item: (len(item.gt_uids - item.crop_covered_gt_uids), item.stats.get("area", 0)), reverse=True):
        missing = sorted(state.gt_uids - state.crop_covered_gt_uids)
        rows.append(
            {
                "image": str(state.image_path),
                "image_key": state.image_key,
                "candidate_count": len(state.candidates),
                "gt_key_count": len(state.gt_uids),
                "crop_covered_gt_key_count": len(state.crop_covered_gt_uids),
                "crop_missed_gt_keys": missing,
                "stats": state.stats,
                "crop_pixels": state.crop_pixels,
                "full_frame_pixels": state.full_frame_pixels,
                "backend_current_fallback_reason": fallback_reason(state.stats, baseline),
                "backend_current_union_area_fallback_reason": fallback_reason(state.stats, current_backend_union_area_policy()),
                "recommended_fallback_reason": fallback_reason(state.stats, recommended) if recommended else "",
            }
        )
    return rows


def policy_delta_rows(states: list[ImageState], before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in states:
        before_reason = fallback_reason(state.stats, before)
        after_reason = fallback_reason(state.stats, after)
        if before_reason == after_reason:
            continue
        missing = sorted(state.gt_uids - state.crop_covered_gt_uids)
        if before_reason and not after_reason:
            delta_type = "fallback_removed"
        elif after_reason and not before_reason:
            delta_type = "fallback_added"
        else:
            delta_type = "fallback_reason_changed"
        rows.append(
            {
                "delta_type": delta_type,
                "image": str(state.image_path),
                "image_key": state.image_key,
                "before_reason": before_reason,
                "after_reason": after_reason,
                "candidate_count": len(state.candidates),
                "gt_key_count": len(state.gt_uids),
                "crop_covered_gt_key_count": len(state.crop_covered_gt_uids),
                "crop_missed_gt_keys": missing,
                "stats": state.stats,
                "crop_pixels": state.crop_pixels,
                "full_frame_pixels": state.full_frame_pixels,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["delta_type"] != "fallback_removed",
            -int(row.get("gt_key_count") or 0),
            int(row.get("candidate_count") or 0),
            float(row.get("stats", {}).get("area") or 0),
            row.get("image") or "",
        ),
    )


def summarize_policy_delta(rows: list[dict[str, Any]], before_eval: dict[str, Any], after_eval: dict[str, Any]) -> dict[str, Any]:
    removed = [row for row in rows if row.get("delta_type") == "fallback_removed"]
    added = [row for row in rows if row.get("delta_type") == "fallback_added"]
    changed = [row for row in rows if row.get("delta_type") == "fallback_reason_changed"]
    return {
        "before_policy_id": before_eval.get("policy_id"),
        "after_policy_id": after_eval.get("policy_id"),
        "fallback_removed_image_count": len(removed),
        "fallback_added_image_count": len(added),
        "fallback_reason_changed_image_count": len(changed),
        "fallback_removed_with_gt_count": sum(1 for row in removed if int(row.get("gt_key_count") or 0) > 0),
        "fallback_removed_with_crop_missed_gt_count": sum(1 for row in removed if row.get("crop_missed_gt_keys")),
        "before_fallback_image_count": before_eval.get("fallback_image_count"),
        "after_fallback_image_count": after_eval.get("fallback_image_count"),
        "before_total_pixels_vs_full": before_eval.get("total_pixels_vs_full"),
        "after_total_pixels_vs_full": after_eval.get("total_pixels_vs_full"),
    }


def draw_norm_rect(draw: ImageDraw.ImageDraw, rect: dict[str, Any], scale_x: float, scale_y: float, color: str, width: int = 2) -> None:
    try:
        x1 = float(rect.get("x") or 0.0) * scale_x
        y1 = float(rect.get("y") or 0.0) * scale_y
        x2 = (float(rect.get("x") or 0.0) + float(rect.get("width") or 0.0)) * scale_x
        y2 = (float(rect.get("y") or 0.0) + float(rect.get("height") or 0.0)) * scale_y
    except (TypeError, ValueError):
        return
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)


def write_policy_delta_contact_sheet(path: Path, rows: list[dict[str, Any]], states: list[ImageState], limit: int) -> None:
    selected = [row for row in rows if row.get("delta_type") == "fallback_removed"][:limit]
    if not selected:
        return
    states_by_key = {state.image_key: state for state in states}
    tile_w, tile_h = 360, 300
    cols = 3
    rows_count = (len(selected) + cols - 1) // cols
    sheet = Image.new("RGB", (tile_w * cols, tile_h * rows_count), "white")
    for index, row in enumerate(selected):
        state = states_by_key.get(str(row.get("image_key") or ""))
        if state is None:
            continue
        try:
            with Image.open(state.image_path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
        except Exception:
            continue
        image.thumbnail((tile_w, tile_h - 44))
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        tile.paste(image, ((tile_w - image.width) // 2, 28))
        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, 0, tile_w - 1, tile_h - 1], outline="#cccccc")
        label = (
            f"{index + 1}. {row.get('before_reason') or '-'} -> {row.get('after_reason') or '-'} "
            f"gt={row.get('gt_key_count')} miss={len(row.get('crop_missed_gt_keys') or [])}"
        )
        draw.text((6, 6), label[:58], fill="#111111")
        offset_x = (tile_w - image.width) // 2
        offset_y = 28
        overlay = ImageDraw.Draw(tile)
        for candidate in state.candidates:
            rect = dict(candidate.rect)
            shifted = {
                "x": (rect.get("x") or 0) + offset_x / max(1, image.width),
                "y": (rect.get("y") or 0) + offset_y / max(1, image.height),
                "width": rect.get("width") or 0,
                "height": rect.get("height") or 0,
            }
            draw_norm_rect(overlay, shifted, image.width, image.height, "#2f6fed", width=2)
        for gt in state.gt_boxes:
            rect = dict(gt.rect)
            shifted = {
                "x": (rect.get("x") or 0) + offset_x / max(1, image.width),
                "y": (rect.get("y") or 0) + offset_y / max(1, image.height),
                "width": rect.get("width") or 0,
                "height": rect.get("height") or 0,
            }
            draw_norm_rect(overlay, shifted, image.width, image.height, "#00a36c", width=3)
        sheet.paste(tile, ((index % cols) * tile_w, (index // cols) * tile_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=92)


def image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as raw:
            return ImageOps.exif_transpose(raw).size
    except Exception:
        return 0, 0


def candidate_bbox_px(candidate: Candidate, width: int, height: int) -> dict[str, int]:
    rect = candidate.rect
    x = max(0, min(width, round(float(rect.get("x") or 0.0) * width)))
    y = max(0, min(height, round(float(rect.get("y") or 0.0) * height)))
    w = max(1, min(width - x, round(float(rect.get("width") or 0.0) * width)))
    h = max(1, min(height - y, round(float(rect.get("height") or 0.0) * height)))
    return {"x": x, "y": y, "width": w, "height": h}


def write_policy_delta_review_queue(out: Path, rows: list[dict[str, Any]], states: list[ImageState], limit: int) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row.get("delta_type") == "fallback_removed" and int(row.get("gt_key_count") or 0) <= 0
    ][:limit]
    if out.exists():
        shutil.rmtree(out)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    states_by_key = {state.image_key: state for state in states}
    draft_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        state = states_by_key.get(str(row.get("image_key") or ""))
        if state is None:
            continue
        width, height = image_dimensions(state.image_path)
        if width <= 0 or height <= 0:
            continue
        image_name = f"{index:03d}_{state.image_path.name}"
        target_rel = Path("images") / image_name
        shutil.copy2(state.image_path, out / target_rel)
        boxes: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(state.candidates, start=1):
            boxes.append(
                {
                    "bbox_px": candidate_bbox_px(candidate, width, height),
                    "score": candidate.score,
                    "source": "fallback_delta_candidate",
                    "question_key": candidate.question_key,
                    "candidate_id": candidate.candidate_id,
                    "quality_flags": sorted(set([*candidate.quality_flags, "fallback_delta_removed", str(row.get("before_reason") or "")])),
                    "default_review_status": "pending",
                    "suggested_review_action": "Verify this crop covers every visible question; add or correct boxes if the removed full-frame fallback would have found more.",
                }
            )
        draft_rows.append(
            {
                "image": target_rel.as_posix(),
                "width": width,
                "height": height,
                "boxes": boxes,
                "metadata": {
                    "source": "policy_delta_legacy_to_current",
                    "before_reason": row.get("before_reason") or "",
                    "after_reason": row.get("after_reason") or "",
                    "candidate_count": row.get("candidate_count"),
                    "crop_area": row.get("stats", {}).get("area") if isinstance(row.get("stats"), dict) else None,
                    "max_crop_area": row.get("stats", {}).get("max_area") if isinstance(row.get("stats"), dict) else None,
                },
                "annotation_status": "draft_review_required",
            }
        )

    write_jsonl(out / "annotations" / "draft_boxes.jsonl", draft_rows)
    summary = {
        "images": len(draft_rows),
        "boxes": sum(len(row.get("boxes") or []) for row in draft_rows),
        "source": "fallback_removed_no_gt",
        "notes": [
            "This queue targets old-policy full-frame fallbacks removed by backend-tuned policy.",
            "Rows have no represented GT in the current reviewed manifest, so review is needed to harden fallback precision/recall evidence.",
        ],
        "outputs": {"draft_boxes": "annotations/draft_boxes.jsonl"},
    }
    write_json(out / "summary.json", summary)
    return summary


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    baseline = summary.get("baseline_backend_current") or {}
    legacy = summary.get("baseline_backend_legacy_before_tuned") or {}
    delta = summary.get("fallback_delta_legacy_to_current") or {}
    recommendation = summary.get("recommendation") or {}
    lines = [
        "# Observation Fallback Tune",
        "",
        f"- images: {summary.get('image_count', 0)}",
        f"- replay candidates: {summary.get('candidate_count', 0)}",
        f"- represented GT keys: {summary.get('represented_gt_key_count', 0)}",
        f"- crop-only GT recall: {summary.get('crop_only_gt_recall')}",
        "",
        "## Backend Current",
        "",
        f"- GT recall: {baseline.get('policy_gt_recall')}",
        f"- fallback images: {baseline.get('fallback_image_count')}",
        f"- total VLM pixels vs full frames: {baseline.get('total_pixels_vs_full')}",
        f"- reasons: {baseline.get('fallback_reason_counts')}",
        "",
        "## Legacy Before Backend-Tuned",
        "",
        f"- GT recall: {legacy.get('policy_gt_recall')}",
        f"- fallback images: {legacy.get('fallback_image_count')}",
        f"- total VLM pixels vs full frames: {legacy.get('total_pixels_vs_full')}",
        f"- reasons: {legacy.get('fallback_reason_counts')}",
        "",
        "## Legacy To Current Delta",
        "",
        f"- fallback removed images: {delta.get('fallback_removed_image_count')}",
        f"- fallback removed images with GT: {delta.get('fallback_removed_with_gt_count')}",
        f"- fallback removed images with crop-missed GT: {delta.get('fallback_removed_with_crop_missed_gt_count')}",
        f"- fallback removed no-GT review images: {summary.get('policy_delta_review_queue', {}).get('images')}",
        f"- total VLM pixels: {delta.get('before_total_pixels_vs_full')} -> {delta.get('after_total_pixels_vs_full')}",
        "",
        "## Backend Current With Union-Area Coverage",
        "",
        f"- GT recall: {summary.get('baseline_backend_current_union_area', {}).get('policy_gt_recall')}",
        f"- fallback images: {summary.get('baseline_backend_current_union_area', {}).get('fallback_image_count')}",
        f"- total VLM pixels vs full frames: {summary.get('baseline_backend_current_union_area', {}).get('total_pixels_vs_full')}",
        f"- reasons: {summary.get('baseline_backend_current_union_area', {}).get('fallback_reason_counts')}",
        "",
        "## Recommendation",
        "",
        f"- policy: {recommendation.get('policy_id')}",
        f"- GT recall: {recommendation.get('policy_gt_recall')}",
        f"- fallback images: {recommendation.get('fallback_image_count')}",
        f"- total VLM pixels vs full frames: {recommendation.get('total_pixels_vs_full')}",
        f"- reasons: {recommendation.get('fallback_reason_counts')}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune full-frame fallback policy for question observation crops.")
    parser.add_argument("--replay", type=Path, action="append", required=True, help="Replay output directory containing details.jsonl and summary.json.")
    parser.add_argument("--ground-truth-manifest", type=Path, action="append", default=[], help="Reviewed detector annotations/manifest.jsonl.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional root for all ground-truth manifests.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--full-max-side", type=int, default=1600)
    parser.add_argument("--min-gt-coverage", type=float, default=0.85, help="Crop covers GT when intersection / GT area reaches this value.")
    parser.add_argument("--min-iou", type=float, default=0.30, help="Crop also covers GT when IoU reaches this value.")
    parser.add_argument("--min-policy-gt-recall", type=float, default=0.999)
    parser.add_argument("--low-confidence-score", type=float, default=0.42)
    parser.add_argument("--candidate-limit-per-frame", type=int, default=12)
    parser.add_argument("--limited-confidence-score", type=float, default=0.55)
    parser.add_argument("--single-low-coverage-areas", default="0.30,0.42,0.55")
    parser.add_argument("--sparse-low-coverage-areas", default="0.25,0.34,0.45")
    parser.add_argument("--tiny-crop-coverage-areas", default="0.12,0.18,0.24")
    parser.add_argument("--no-large-min-max-areas", default="0.12,0.16,0.20")
    parser.add_argument("--no-large-max-total-areas", default="0.40,0.50,0.60")
    parser.add_argument("--all-weak-max-counts", default="0,3,5")
    parser.add_argument("--all-weak-max-total-areas", default="0.40,0.50,0.65")
    parser.add_argument("--all-weak-max-max-areas", default="0.20,0.34,0.50")
    parser.add_argument("--low-confidence-max-counts", default="0,3,5")
    parser.add_argument("--delta-contact-sheet-limit", type=int, default=36)
    parser.add_argument("--delta-review-limit", type=int, default=80)
    args = parser.parse_args()

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    full_max_side = args.full_max_side
    if args.replay:
        full_max_side = replay_full_max_side(args.replay[0], default=full_max_side)

    gt_boxes = load_gt(args.ground_truth_manifest, args.dataset_root) if args.ground_truth_manifest else []
    states, input_summary = build_image_states(
        replay_dirs=args.replay,
        gt_boxes=gt_boxes,
        min_gt_coverage=args.min_gt_coverage,
        min_iou=args.min_iou,
        full_max_side=full_max_side,
        low_confidence_score=args.low_confidence_score,
        candidate_limit=args.candidate_limit_per_frame,
        limited_confidence_score=args.limited_confidence_score,
    )

    policies = iter_grid(args)
    rows = [evaluate_policy(states, policy) for policy in policies]
    rows.sort(
        key=lambda row: (
            -float(row.get("policy_gt_recall") or 0),
            float(row.get("total_pixels_vs_full") or 999),
            int(row.get("fallback_image_count") or 999999),
        )
    )
    current_policy = current_backend_policy()
    legacy_policy = legacy_backend_policy()
    baseline = evaluate_policy(states, current_policy)
    baseline_union = evaluate_policy(states, current_backend_union_area_policy())
    legacy_baseline = evaluate_policy(states, legacy_policy)
    recommendation = choose_recommendation(rows, min_gt_recall=args.min_policy_gt_recall)
    recommended_policy = recommendation.get("policy") if isinstance(recommendation.get("policy"), dict) else {}
    delta_rows = policy_delta_rows(states, legacy_policy, current_policy)
    delta_summary = summarize_policy_delta(delta_rows, legacy_baseline, baseline)
    delta_review_summary = write_policy_delta_review_queue(args.out / "policy_delta_review_queue", delta_rows, states, limit=args.delta_review_limit)

    summary = {
        **input_summary,
        "ground_truth_manifests": [str(path) for path in args.ground_truth_manifest],
        "dataset_root": str(args.dataset_root) if args.dataset_root else "",
        "settings": {
            "full_max_side": full_max_side,
            "min_gt_coverage": args.min_gt_coverage,
            "min_iou": args.min_iou,
            "min_policy_gt_recall": args.min_policy_gt_recall,
            "low_confidence_score": args.low_confidence_score,
            "candidate_limit_per_frame": args.candidate_limit_per_frame,
            "limited_confidence_score": args.limited_confidence_score,
            "policy_count": len(policies),
        },
        "baseline_backend_current": baseline,
        "baseline_backend_current_union_area": baseline_union,
        "baseline_backend_legacy_before_tuned": legacy_baseline,
        "fallback_delta_legacy_to_current": delta_summary,
        "policy_delta_review_queue": delta_review_summary,
        "recommendation": recommendation,
        "outputs": {
            "grid": "grid.jsonl",
            "image_risk": "image_risk.jsonl",
            "policy_delta": "policy_delta.jsonl",
            "policy_delta_contact_sheet": "policy_delta_contact_sheet.jpg",
            "policy_delta_review_queue": "policy_delta_review_queue",
            "summary_md": "summary.md",
        },
    }

    write_json(args.out / "summary.json", summary)
    write_jsonl(args.out / "grid.jsonl", rows)
    write_jsonl(args.out / "image_risk.jsonl", image_risk_rows(states, current_policy, recommended_policy))
    write_jsonl(args.out / "policy_delta.jsonl", delta_rows)
    write_policy_delta_contact_sheet(args.out / "policy_delta_contact_sheet.jpg", delta_rows, states, limit=args.delta_contact_sheet_limit)
    write_markdown(args.out / "summary.md", summary)

    print(
        json.dumps(
            {
                "out": str(args.out),
                "images": len(states),
                "candidates": input_summary.get("candidate_count"),
                "represented_gt_keys": input_summary.get("represented_gt_key_count"),
                "crop_only_gt_recall": input_summary.get("crop_only_gt_recall"),
                "backend_current": {
                    "gt_recall": baseline.get("policy_gt_recall"),
                    "fallback_images": baseline.get("fallback_image_count"),
                    "total_pixels_vs_full": baseline.get("total_pixels_vs_full"),
                },
                "backend_current_union_area": {
                    "gt_recall": baseline_union.get("policy_gt_recall"),
                    "fallback_images": baseline_union.get("fallback_image_count"),
                    "total_pixels_vs_full": baseline_union.get("total_pixels_vs_full"),
                },
                "recommendation": {
                    "policy_id": recommendation.get("policy_id"),
                    "gt_recall": recommendation.get("policy_gt_recall"),
                    "fallback_images": recommendation.get("fallback_image_count"),
                    "total_pixels_vs_full": recommendation.get("total_pixels_vs_full"),
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
