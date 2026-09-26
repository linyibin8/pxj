"""Sweep iOS per-frame question-candidate caps on observation replay data.

The live app ranks question candidates before applying a per-frame cap. This
script simulates that cap offline, then measures crop-only recall, backend
fallback-protected recall, and VLM pixel cost against reviewed boxes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from question_observation_candidate_eval import gt_effective_uid, replay_full_max_side, resized_pixel_count
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
from question_observation_fallback_tune import current_backend_policy, fallback_reason, rect_union_area


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        result.append(max(0, int(item)))
    return sorted(set(result), key=lambda item: (item == 0, item))


def crop_covers_gt(candidate: Candidate, gt_box: GTBox, min_gt_coverage: float, min_iou: float) -> bool:
    overlap = intersection(candidate.rect, gt_box.rect)
    if overlap <= 0:
        return False
    gt_coverage = overlap / max(1e-9, rect_area(gt_box.rect))
    return gt_coverage >= min_gt_coverage or rect_iou(candidate.rect, gt_box.rect) >= min_iou


def candidate_risk_flags(candidate: Candidate) -> list[str]:
    return [
        flag
        for flag in candidate.quality_flags
        if not flag.startswith("page:") and flag not in {"stable", "reviewed"}
    ]


def ranked_candidates(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            is_weak_key(candidate.question_key),
            len(candidate_risk_flags(candidate)),
            -candidate.score,
            -candidate.area,
            candidate.index,
            candidate.candidate_id,
        ),
    )


def image_stats(selected: list[Candidate], skipped: list[Candidate], low_confidence_score: float, limited_confidence_score: float) -> dict[str, Any]:
    total_area = sum(candidate.area for candidate in selected)
    union_area = rect_union_area([candidate.rect for candidate in selected])
    max_area = max((candidate.area for candidate in selected), default=0.0)
    limited_strong_count = sum(1 for candidate in skipped if not is_weak_key(candidate.question_key))
    limited_confident_count = sum(1 for candidate in skipped if candidate.score >= limited_confidence_score)
    return {
        "count": len(selected),
        "area": round(total_area, 6),
        "union_area": round(union_area, 6),
        "overlap_area": round(max(0.0, total_area - union_area), 6),
        "overlap_area_ratio": round(max(0.0, total_area - union_area) / max(1e-9, total_area), 6) if total_area > 0 else 0.0,
        "max_area": round(max_area, 6),
        "weak_count": sum(1 for candidate in selected if is_weak_key(candidate.question_key)),
        "low_conf_count": sum(1 for candidate in selected if 0 < candidate.score < low_confidence_score),
        "selected_candidate_count": len(selected),
        "limited_count": len(skipped),
        "limited_unique_count": len(skipped),
        "limited_strong_count": limited_strong_count,
        "limited_confident_count": limited_confident_count,
        "limited_max_confidence": round(max((candidate.score for candidate in skipped), default=0.0), 6),
        "ranked_limit_telemetry_count": 1,
    }


def load_inputs(replay_dirs: list[Path], gt_boxes: list[GTBox]) -> tuple[dict[str, list[Candidate]], dict[str, dict[str, GTBox]], dict[str, int], dict[str, Path], dict[str, Any]]:
    candidates: list[Candidate] = []
    replay_summaries: list[dict[str, Any]] = []
    replay_images = []
    full_max_side = replay_full_max_side(replay_dirs[0], default=1600) if replay_dirs else 1600
    for replay_dir in replay_dirs:
        replay_candidates, replay_summary = load_replay(replay_dir)
        loaded_images = load_replay_images(replay_dir)
        candidates.extend(replay_candidates)
        replay_images.extend(loaded_images)
        replay_summaries.append({"path": str(replay_dir), "candidate_count": len(replay_candidates), "image_count": len(loaded_images), **replay_summary})

    by_key, by_hash, all_gt = build_gt_indexes(gt_boxes)
    hash_cache: dict[Path, str] = {}
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    gt_by_image: dict[str, dict[str, GTBox]] = defaultdict(dict)
    full_pixels_by_image: dict[str, int] = {}
    image_paths_by_key: dict[str, Path] = {}

    for image_ref in replay_images:
        image_key = image_ref.image_key or str(image_ref.image_path)
        grouped.setdefault(image_key, [])
        image_paths_by_key[image_key] = image_ref.image_path
        full_pixels_by_image.setdefault(image_key, resized_pixel_count(image_ref.image_path, max_side=full_max_side))
        for gt in find_gt_boxes_for_image_identity(
            image_ref.image_path,
            image_ref.image_key,
            image_ref.source_filename,
            by_key,
            by_hash,
            all_gt,
        ):
            gt_by_image[image_key][gt_effective_uid(gt, hash_cache)] = gt

    for candidate in candidates:
        image_key = candidate.image_key or str(candidate.image_path)
        grouped[image_key].append(candidate)
        image_paths_by_key.setdefault(image_key, candidate.image_path)
        full_pixels_by_image.setdefault(image_key, resized_pixel_count(candidate.image_path, max_side=full_max_side))
        for gt in find_gt_image_boxes(candidate, by_key, by_hash, all_gt):
            gt_by_image[image_key][gt_effective_uid(gt, hash_cache)] = gt

    summary = {
        "replays": replay_summaries,
        "candidate_count": len(candidates),
        "image_count": len(grouped),
        "gt_box_count": len(gt_boxes),
        "represented_gt_key_count": len({uid for values in gt_by_image.values() for uid in values}),
        "full_frame_all_pixels": sum(full_pixels_by_image.values()),
        "full_max_side": full_max_side,
    }
    return dict(grouped), dict(gt_by_image), full_pixels_by_image, image_paths_by_key, summary


def covered_gt_for_candidates(
    candidates: list[Candidate],
    gt_boxes: list[GTBox],
    min_gt_coverage: float,
    min_iou: float,
    hash_cache: dict[Path, str],
) -> set[str]:
    covered: set[str] = set()
    for candidate in candidates:
        for gt in gt_boxes:
            if crop_covers_gt(candidate, gt, min_gt_coverage=min_gt_coverage, min_iou=min_iou):
                covered.add(gt_effective_uid(gt, hash_cache))
    return covered


def evaluate_cap(
    cap: int,
    grouped: dict[str, list[Candidate]],
    gt_boxes: list[GTBox],
    gt_by_image: dict[str, dict[str, GTBox]],
    full_pixels_by_image: dict[str, int],
    image_paths_by_key: dict[str, Path],
    min_gt_coverage: float,
    min_iou: float,
    low_confidence_score: float,
    limited_confidence_score: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hash_cache: dict[Path, str] = {}
    represented_gt = {uid for values in gt_by_image.values() for uid in values}
    selected_covered_gt: set[str] = set()
    fallback_covered_gt: set[str] = set()
    selected_crop_pixels = 0
    selected_crop_bytes = 0
    fallback_pixels = 0
    selected_count = 0
    skipped_count = 0
    skipped_strong_count = 0
    skipped_confident_count = 0
    fallback_reason_counts: Counter[str] = Counter()
    risk_rows: list[dict[str, Any]] = []
    policy = current_backend_policy()

    for image_key, image_candidates in sorted(grouped.items()):
        ranked = ranked_candidates(image_candidates)
        selected = ranked if cap <= 0 else ranked[:cap]
        skipped = [] if cap <= 0 else ranked[cap:]
        image_gt_boxes: dict[str, GTBox] = dict(gt_by_image.get(image_key) or {})
        image_gt_set = set(image_gt_boxes)
        selected_gt = covered_gt_for_candidates(
            selected,
            list(image_gt_boxes.values()),
            min_gt_coverage=min_gt_coverage,
            min_iou=min_iou,
            hash_cache=hash_cache,
        )
        all_candidate_gt = covered_gt_for_candidates(
            image_candidates,
            list(image_gt_boxes.values()),
            min_gt_coverage=min_gt_coverage,
            min_iou=min_iou,
            hash_cache=hash_cache,
        )
        selected_covered_gt.update(selected_gt)
        selected_crop_pixels += sum(candidate.crop_resized_pixels for candidate in selected)
        selected_crop_bytes += sum(candidate.crop_jpeg_bytes for candidate in selected)
        selected_count += len(selected)
        skipped_count += len(skipped)
        skipped_strong_count += sum(1 for candidate in skipped if not is_weak_key(candidate.question_key))
        skipped_confident_count += sum(1 for candidate in skipped if candidate.score >= limited_confidence_score)
        stats = image_stats(
            selected,
            skipped,
            low_confidence_score=low_confidence_score,
            limited_confidence_score=limited_confidence_score,
        )
        reason = fallback_reason(stats, policy)
        if reason:
            fallback_reason_counts[reason] += 1
            fallback_pixels += full_pixels_by_image.get(image_key, 0)
            fallback_covered_gt.update(image_gt_set)
        if image_gt_set - selected_gt or skipped:
            risk_rows.append(
                {
                    "cap": cap,
                    "image_key": image_key,
                    "image": str(image_candidates[0].image_path if image_candidates else image_paths_by_key.get(image_key, "")),
                    "candidate_count": len(image_candidates),
                    "selected_count": len(selected),
                    "skipped_count": len(skipped),
                    "skipped_strong_count": sum(1 for candidate in skipped if not is_weak_key(candidate.question_key)),
                    "skipped_confident_count": sum(1 for candidate in skipped if candidate.score >= limited_confidence_score),
                    "gt_key_count": len(image_gt_set),
                    "selected_gt_key_count": len(selected_gt),
                    "all_candidate_gt_key_count": len(all_candidate_gt),
                    "selected_missed_gt_keys": sorted(image_gt_set - selected_gt),
                    "cap_lost_gt_keys": sorted(all_candidate_gt - selected_gt),
                    "fallback_reason": reason,
                    "stats": stats,
                }
            )

    policy_covered_gt = set(selected_covered_gt) | fallback_covered_gt
    full_pixels = sum(full_pixels_by_image.values())
    selected_recall = len(selected_covered_gt) / max(1, len(represented_gt)) if represented_gt else None
    policy_recall = len(policy_covered_gt) / max(1, len(represented_gt)) if represented_gt else None
    row = {
        "cap": cap,
        "cap_label": "no_cap" if cap <= 0 else str(cap),
        "image_count": len(grouped),
        "represented_gt_key_count": len(represented_gt),
        "selected_candidate_count": selected_count,
        "skipped_candidate_count": skipped_count,
        "skipped_strong_candidate_count": skipped_strong_count,
        "skipped_confident_candidate_count": skipped_confident_count,
        "selected_covered_gt_key_count": len(selected_covered_gt),
        "selected_gt_recall": round(selected_recall, 6) if selected_recall is not None else None,
        "policy_covered_gt_key_count": len(policy_covered_gt),
        "policy_gt_recall": round(policy_recall, 6) if policy_recall is not None else None,
        "fallback_image_count": sum(fallback_reason_counts.values()),
        "fallback_image_ratio": round(sum(fallback_reason_counts.values()) / max(1, len(grouped)), 6),
        "fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
        "selected_crop_pixels": selected_crop_pixels,
        "fallback_full_frame_pixels": fallback_pixels,
        "total_vlm_pixels": selected_crop_pixels + fallback_pixels,
        "full_frame_all_pixels": full_pixels,
        "selected_crop_pixels_vs_full": round(selected_crop_pixels / max(1, full_pixels), 6),
        "total_pixels_vs_full": round((selected_crop_pixels + fallback_pixels) / max(1, full_pixels), 6),
        "selected_crop_jpeg_bytes": selected_crop_bytes,
    }
    return row, risk_rows


def choose_recommendation(rows: list[dict[str, Any]], min_selected_gt_recall: float, min_policy_gt_recall: float) -> dict[str, Any]:
    passing = [
        row
        for row in rows
        if row.get("selected_gt_recall") is not None
        and float(row.get("selected_gt_recall") or 0) >= min_selected_gt_recall
        and float(row.get("policy_gt_recall") or 0) >= min_policy_gt_recall
    ]
    if not passing:
        return {}
    return sorted(
        passing,
        key=lambda row: (
            float(row.get("total_pixels_vs_full") or 999),
            int(row.get("fallback_image_count") or 999999),
            int(row.get("cap") or 999999),
        ),
    )[0]


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    current = summary.get("current_cap") if isinstance(summary.get("current_cap"), dict) else {}
    recommendation = summary.get("recommendation") if isinstance(summary.get("recommendation"), dict) else {}
    lines = [
        "# Observation Candidate Cap Sweep",
        "",
        f"- images: {summary.get('image_count')}",
        f"- candidates: {summary.get('candidate_count')}",
        f"- represented GT keys: {summary.get('represented_gt_key_count')}",
        "",
        "## Current Cap",
        "",
        f"- cap: {current.get('cap_label')}",
        f"- selected GT recall: {current.get('selected_gt_recall')}",
        f"- fallback-protected GT recall: {current.get('policy_gt_recall')}",
        f"- selected candidates: {current.get('selected_candidate_count')}",
        f"- fallback images: {current.get('fallback_image_count')}",
        f"- total VLM pixels vs full frames: {current.get('total_pixels_vs_full')}",
        "",
        "## Recommendation",
        "",
        f"- cap: {recommendation.get('cap_label')}",
        f"- selected GT recall: {recommendation.get('selected_gt_recall')}",
        f"- fallback-protected GT recall: {recommendation.get('policy_gt_recall')}",
        f"- selected candidates: {recommendation.get('selected_candidate_count')}",
        f"- fallback images: {recommendation.get('fallback_image_count')}",
        f"- total VLM pixels vs full frames: {recommendation.get('total_pixels_vs_full')}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep per-frame observation question candidate caps.")
    parser.add_argument("--replay", type=Path, action="append", required=True)
    parser.add_argument("--ground-truth-manifest", type=Path, action="append", default=[])
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--caps", default="1,2,3,4,6,8,10,12,16,24,0")
    parser.add_argument("--current-cap", type=int, default=12)
    parser.add_argument("--min-gt-coverage", type=float, default=0.85)
    parser.add_argument("--min-iou", type=float, default=0.30)
    parser.add_argument("--min-selected-gt-recall", type=float, default=0.999)
    parser.add_argument("--min-policy-gt-recall", type=float, default=0.999)
    parser.add_argument("--low-confidence-score", type=float, default=0.42)
    parser.add_argument("--limited-confidence-score", type=float, default=0.55)
    args = parser.parse_args()

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    gt_boxes = load_gt(args.ground_truth_manifest, args.dataset_root) if args.ground_truth_manifest else []
    grouped, gt_by_image, full_pixels_by_image, image_paths_by_key, input_summary = load_inputs(args.replay, gt_boxes)
    caps = parse_int_list(args.caps)
    if args.current_cap not in caps:
        caps.append(args.current_cap)
        caps = parse_int_list(",".join(str(item) for item in caps))

    rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    for cap in caps:
        row, cap_risks = evaluate_cap(
            cap,
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
        rows.append(row)
        risk_rows.extend(cap_risks)

    rows.sort(key=lambda row: (row["cap"] <= 0, row["cap"]))
    current = next((row for row in rows if row.get("cap") == args.current_cap), {})
    recommendation = choose_recommendation(
        rows,
        min_selected_gt_recall=args.min_selected_gt_recall,
        min_policy_gt_recall=args.min_policy_gt_recall,
    )
    summary = {
        **input_summary,
        "ground_truth_manifests": [str(path) for path in args.ground_truth_manifest],
        "dataset_root": str(args.dataset_root) if args.dataset_root else "",
        "settings": {
            "caps": caps,
            "current_cap": args.current_cap,
            "min_gt_coverage": args.min_gt_coverage,
            "min_iou": args.min_iou,
            "min_selected_gt_recall": args.min_selected_gt_recall,
            "min_policy_gt_recall": args.min_policy_gt_recall,
            "low_confidence_score": args.low_confidence_score,
            "limited_confidence_score": args.limited_confidence_score,
        },
        "current_cap": current,
        "recommendation": recommendation,
        "outputs": {
            "cap_sweep": "cap_sweep.jsonl",
            "cap_risk": "cap_risk.jsonl",
            "summary_md": "summary.md",
        },
    }
    write_json(args.out / "summary.json", summary)
    write_jsonl(args.out / "cap_sweep.jsonl", rows)
    write_jsonl(args.out / "cap_risk.jsonl", risk_rows)
    write_markdown(args.out / "summary.md", summary)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "images": input_summary.get("image_count"),
                "candidates": input_summary.get("candidate_count"),
                "represented_gt_keys": input_summary.get("represented_gt_key_count"),
                "current_cap": {
                    "cap": current.get("cap"),
                    "selected_gt_recall": current.get("selected_gt_recall"),
                    "policy_gt_recall": current.get("policy_gt_recall"),
                    "selected_candidates": current.get("selected_candidate_count"),
                    "total_pixels_vs_full": current.get("total_pixels_vs_full"),
                },
                "recommendation": {
                    "cap": recommendation.get("cap"),
                    "selected_gt_recall": recommendation.get("selected_gt_recall"),
                    "policy_gt_recall": recommendation.get("policy_gt_recall"),
                    "selected_candidates": recommendation.get("selected_candidate_count"),
                    "total_pixels_vs_full": recommendation.get("total_pixels_vs_full"),
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
