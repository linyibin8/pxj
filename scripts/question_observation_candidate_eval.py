"""Evaluate observation question crop candidates against reviewed boxes.

This sits between low-level detector IoU and product question-set eval. It asks:

- did the local crop candidates cover every reviewed question on represented images?
- how many crop candidates were needed?
- how much VLM pixel work do those crop candidates cost versus full frames?
- which images/questions need review because candidates missed, over-expanded, or
  were produced on images without reviewed GT?
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from question_observation_dedupe_tune import (
    Candidate,
    GTBox,
    build_gt_indexes,
    file_sha1,
    find_gt_boxes_for_image_identity,
    find_gt_image_boxes,
    intersection,
    load_replay_images,
    load_gt,
    load_replay,
    rect_area,
    rect_iou,
)


@dataclass
class CandidateMatch:
    candidate: Candidate
    gt: GTBox | None
    iou: float
    gt_coverage: float
    candidate_coverage: float
    overcrop_ratio: float
    matched: bool


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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


def resized_pixel_count(path: Path, max_side: int) -> int:
    try:
        with Image.open(path) as image:
            width, height = ImageOps.exif_transpose(image).size
    except Exception:
        return 0
    longest = max(width, height, 1)
    scale = min(1.0, max_side / longest)
    return max(1, int(width * scale)) * max(1, int(height * scale))


def gt_uid(box: GTBox) -> str:
    return f"{box.image_id}:{box.gt_key}"


def gt_effective_uid(box: GTBox, hash_cache: dict[Path, str]) -> str:
    image_key = box.source_filename
    if box.image_path is not None and box.image_path.is_file():
        try:
            image_key = hash_cache.setdefault(box.image_path, file_sha1(box.image_path))
        except OSError:
            image_key = box.source_filename
    rect_key = ",".join(f"{box.rect[key]:.5f}" for key in ("x", "y", "width", "height"))
    return f"{image_key}:{rect_key}"


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    rank = (len(ordered) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(ordered[lower], 6)
    weight = rank - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def best_match(candidate: Candidate, gt_boxes: list[GTBox], min_gt_coverage: float, min_iou: float) -> CandidateMatch:
    best_gt: GTBox | None = None
    best_iou = 0.0
    best_gt_coverage = 0.0
    best_candidate_coverage = 0.0
    best_score = -1.0
    for gt in gt_boxes:
        overlap = intersection(candidate.rect, gt.rect)
        gt_area = rect_area(gt.rect)
        candidate_area = rect_area(candidate.rect)
        gt_coverage = overlap / max(1e-9, gt_area)
        candidate_coverage = overlap / max(1e-9, candidate_area)
        iou = rect_iou(candidate.rect, gt.rect)
        score = max(gt_coverage, iou) + candidate_coverage * 0.05
        if score > best_score:
            best_gt = gt
            best_iou = iou
            best_gt_coverage = gt_coverage
            best_candidate_coverage = candidate_coverage
            best_score = score
    overcrop_ratio = rect_area(candidate.rect) / max(1e-9, rect_area(best_gt.rect) if best_gt else 0.0)
    matched = best_gt is not None and (best_gt_coverage >= min_gt_coverage or best_iou >= min_iou)
    return CandidateMatch(
        candidate=candidate,
        gt=best_gt,
        iou=best_iou,
        gt_coverage=best_gt_coverage,
        candidate_coverage=best_candidate_coverage,
        overcrop_ratio=overcrop_ratio if best_gt is not None else 0.0,
        matched=matched,
    )


def row_for_candidate_match(match: CandidateMatch) -> dict[str, Any]:
    candidate = match.candidate
    row = {
        "candidate_id": candidate.candidate_id,
        "image": str(candidate.image_path),
        "image_key": candidate.image_key,
        "question_key": candidate.question_key,
        "bbox_norm": candidate.rect,
        "area": round(candidate.area, 6),
        "score": candidate.score,
        "crop_resized_pixels": candidate.crop_resized_pixels,
        "crop_jpeg_bytes": candidate.crop_jpeg_bytes,
        "quality_flags": candidate.quality_flags,
        "matched": match.matched,
        "iou": round(match.iou, 6),
        "gt_coverage": round(match.gt_coverage, 6),
        "candidate_coverage": round(match.candidate_coverage, 6),
        "overcrop_ratio": round(match.overcrop_ratio, 6),
    }
    if match.gt is not None:
        row.update(
            {
                "gt_key": match.gt.gt_key,
                "gt_uid": gt_uid(match.gt),
                "gt_source_filename": match.gt.source_filename,
                "gt_bbox_norm": match.gt.rect,
            }
        )
    return row


def evaluate(
    replay_dirs: list[Path],
    gt_boxes: list[GTBox],
    min_gt_coverage: float,
    min_iou: float,
    max_overcrop_ratio: float,
    full_max_side: int,
) -> dict[str, Any]:
    candidates: list[Candidate] = []
    replay_summaries: list[dict[str, Any]] = []
    replay_images = []
    for replay_dir in replay_dirs:
        loaded_candidates, replay_summary = load_replay(replay_dir)
        loaded_images = load_replay_images(replay_dir)
        candidates.extend(loaded_candidates)
        replay_images.extend(loaded_images)
        replay_summaries.append({"path": str(replay_dir), "candidate_count": len(loaded_candidates), "image_count": len(loaded_images), **replay_summary})

    by_key, by_hash, all_gt = build_gt_indexes(gt_boxes)
    matches: list[CandidateMatch] = []
    gt_by_uid: dict[str, GTBox] = {}
    gt_by_image: dict[str, dict[str, GTBox]] = defaultdict(dict)
    hash_cache: dict[Path, str] = {}
    candidates_by_image: dict[str, list[Candidate]] = defaultdict(list)
    full_pixels_by_image: dict[str, int] = {}
    image_paths_by_key: dict[str, Path] = {}

    for image_ref in replay_images:
        image_key = image_ref.image_key or str(image_ref.image_path)
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
            uid = gt_effective_uid(gt, hash_cache)
            gt_by_uid[gt_uid(gt)] = gt
            gt_by_image[image_key][uid] = gt

    for candidate in candidates:
        image_key = candidate.image_key or str(candidate.image_path)
        candidates_by_image[image_key].append(candidate)
        image_paths_by_key.setdefault(image_key, candidate.image_path)
        if image_key not in full_pixels_by_image:
            full_pixels_by_image[image_key] = resized_pixel_count(candidate.image_path, max_side=full_max_side)
        image_gt = find_gt_image_boxes(candidate, by_key, by_hash, all_gt)
        for gt in image_gt:
            gt_by_uid[gt_uid(gt)] = gt
            gt_by_image[image_key][gt_effective_uid(gt, hash_cache)] = gt
        matches.append(best_match(candidate, image_gt, min_gt_coverage=min_gt_coverage, min_iou=min_iou))

    covered_gt: dict[str, CandidateMatch] = {}
    gt_units: dict[str, list[GTBox]] = defaultdict(list)
    for gt in gt_by_uid.values():
        gt_units[gt_effective_uid(gt, hash_cache)].append(gt)

    for match in matches:
        if not match.matched or match.gt is None:
            continue
        uid = gt_effective_uid(match.gt, hash_cache)
        previous = covered_gt.get(uid)
        if previous is None or (match.gt_coverage, match.iou, -match.overcrop_ratio) > (previous.gt_coverage, previous.iou, -previous.overcrop_ratio):
            covered_gt[uid] = match

    represented_gt_uids = set(gt_units)
    missed_gt_uids = sorted(represented_gt_uids - set(covered_gt))
    matched_candidates = [match for match in matches if match.matched]
    reviewed_image_candidates = [match for match in matches if match.gt is not None]
    unmatched_reviewed_candidates = [match for match in reviewed_image_candidates if not match.matched]
    unverified_candidates = [match for match in matches if match.gt is None]
    overcrop_matches = [match for match in matched_candidates if match.overcrop_ratio > max_overcrop_ratio]
    overcrop_values = [match.overcrop_ratio for match in matched_candidates if match.overcrop_ratio > 0]

    crop_pixels = sum(candidate.crop_resized_pixels for candidate in candidates)
    crop_bytes = sum(candidate.crop_jpeg_bytes for candidate in candidates)
    full_pixels = sum(full_pixels_by_image.values())
    quality_flags = Counter(flag for candidate in candidates for flag in candidate.quality_flags)
    missed_rows = []
    for uid in missed_gt_uids:
        unit_boxes = gt_units.get(uid) or []
        gt = unit_boxes[0]
        image_candidates = [
            row_for_candidate_match(match)
            for match in matches
            if match.gt is not None and gt_effective_uid(match.gt, hash_cache) == uid
        ]
        missed_rows.append(
            {
                "gt_uid": uid,
                "raw_gt_uids": [gt_uid(item) for item in unit_boxes],
                "raw_gt_keys": [item.gt_key for item in unit_boxes],
                "source_filenames": [item.source_filename for item in unit_boxes],
                "duplicate_source_count": len(unit_boxes),
                "gt_key": gt.gt_key,
                "source_filename": gt.source_filename,
                "bbox_norm": gt.rect,
                "candidate_count_on_image": len(image_candidates),
                "best_candidates": sorted(image_candidates, key=lambda row: (row["gt_coverage"], row["iou"]), reverse=True)[:5],
            }
        )

    per_image_rows: list[dict[str, Any]] = []
    all_image_keys = sorted(set(full_pixels_by_image) | set(candidates_by_image) | set(gt_by_image))
    for image_key in all_image_keys:
        image_candidates = candidates_by_image.get(image_key, [])
        image_matches = [match for match in matches if match.candidate.image_key == image_key]
        image_gt = dict(gt_by_image.get(image_key) or {})
        for match in image_matches:
            if match.gt is not None:
                image_gt[gt_effective_uid(match.gt, hash_cache)] = match.gt
        image_covered = {gt_effective_uid(match.gt, hash_cache) for match in image_matches if match.matched and match.gt is not None}
        per_image_rows.append(
            {
                "image_key": image_key,
                "image": str(image_candidates[0].image_path if image_candidates else image_paths_by_key.get(image_key, "")),
                "candidate_count": len(image_candidates),
                "gt_key_count": len(image_gt),
                "covered_gt_key_count": len(image_covered),
                "missed_gt_key_count": max(0, len(image_gt) - len(image_covered)),
                "crop_resized_pixels": sum(candidate.crop_resized_pixels for candidate in image_candidates),
                "full_frame_pixels": full_pixels_by_image.get(image_key, 0),
                "quality_flags": dict(sorted(Counter(flag for candidate in image_candidates for flag in candidate.quality_flags).items())),
            }
        )

    summary = {
        "replays": replay_summaries,
        "settings": {
            "min_gt_coverage": min_gt_coverage,
            "min_iou": min_iou,
            "max_overcrop_ratio": max_overcrop_ratio,
            "full_max_side": full_max_side,
        },
        "counts": {
            "images": len(all_image_keys),
            "images_with_candidates": len(candidates_by_image),
            "candidate_count": len(candidates),
            "gt_box_count": len(gt_boxes),
            "raw_represented_gt_key_count": len(gt_by_uid),
            "represented_gt_key_count": len(represented_gt_uids),
            "duplicate_represented_gt_key_count": max(0, len(gt_by_uid) - len(represented_gt_uids)),
            "covered_gt_key_count": len(covered_gt),
            "missed_gt_key_count": len(missed_gt_uids),
            "matched_candidate_count": len(matched_candidates),
            "reviewed_image_candidate_count": len(reviewed_image_candidates),
            "unmatched_reviewed_candidate_count": len(unmatched_reviewed_candidates),
            "unverified_candidate_count": len(unverified_candidates),
            "overcrop_match_count": len(overcrop_matches),
        },
        "metrics": {
            "represented_gt_recall": round(len(covered_gt) / max(1, len(represented_gt_uids)), 6) if represented_gt_uids else None,
            "reviewed_candidate_match_rate": round(len(matched_candidates) / max(1, len(reviewed_image_candidates)), 6) if reviewed_image_candidates else None,
            "crop_pixels_vs_full_frame_pixels": round(crop_pixels / max(1, full_pixels), 6),
            "crop_resized_pixels": crop_pixels,
            "full_frame_resized_pixels": full_pixels,
            "crop_jpeg_bytes": crop_bytes,
            "overcrop_ratio_median": percentile(overcrop_values, 0.5),
            "overcrop_ratio_p90": percentile(overcrop_values, 0.9),
            "overcrop_ratio_max": percentile(overcrop_values, 1.0),
        },
        "quality_flags": dict(sorted(quality_flags.items())),
        "outputs": {
            "matches": "matches.jsonl",
            "missed": "missed_gt.jsonl",
            "unmatched_reviewed_candidates": "unmatched_reviewed_candidates.jsonl",
            "unverified_candidates": "unverified_candidates.jsonl",
            "overcrop": "overcrop_candidates.jsonl",
            "per_image": "per_image.jsonl",
            "summary_md": "summary.md",
        },
    }
    return {
        "summary": summary,
        "matches": [row_for_candidate_match(match) for match in matches],
        "missed": missed_rows,
        "unmatched_reviewed_candidates": [row_for_candidate_match(match) for match in unmatched_reviewed_candidates],
        "unverified_candidates": [row_for_candidate_match(match) for match in unverified_candidates],
        "overcrop": [row_for_candidate_match(match) for match in overcrop_matches],
        "per_image": per_image_rows,
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    counts = summary.get("counts") or {}
    metrics = summary.get("metrics") or {}
    lines = [
        "# Observation Candidate Eval",
        "",
        f"- images: {counts.get('images')}",
        f"- images with candidates: {counts.get('images_with_candidates')}",
        f"- candidates: {counts.get('candidate_count')}",
        f"- represented GT keys: {counts.get('represented_gt_key_count')}",
        f"- covered GT keys: {counts.get('covered_gt_key_count')}",
        f"- represented GT recall: {metrics.get('represented_gt_recall')}",
        f"- crop pixels vs full frames: {metrics.get('crop_pixels_vs_full_frame_pixels')}",
        f"- reviewed candidate match rate: {metrics.get('reviewed_candidate_match_rate')}",
        f"- overcrop p90: {metrics.get('overcrop_ratio_p90')}",
        "",
        "## Outputs",
        "",
    ]
    for name, rel in (summary.get("outputs") or {}).items():
        lines.append(f"- {name}: `{rel}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate observation crop candidates against reviewed question boxes.")
    parser.add_argument("--replay", type=Path, action="append", required=True, help="Replay output directory containing details.jsonl.")
    parser.add_argument("--ground-truth-manifest", type=Path, action="append", required=True, help="Reviewed detector annotations/manifest.jsonl.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Optional root for all ground-truth manifests.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--full-max-side", type=int, default=1600)
    parser.add_argument("--min-gt-coverage", type=float, default=0.85)
    parser.add_argument("--min-iou", type=float, default=0.30)
    parser.add_argument("--max-overcrop-ratio", type=float, default=3.0)
    args = parser.parse_args()

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    full_max_side = replay_full_max_side(args.replay[0], args.full_max_side) if args.replay else args.full_max_side
    gt_boxes = load_gt(args.ground_truth_manifest, args.dataset_root)
    result = evaluate(
        replay_dirs=args.replay,
        gt_boxes=gt_boxes,
        min_gt_coverage=args.min_gt_coverage,
        min_iou=args.min_iou,
        max_overcrop_ratio=args.max_overcrop_ratio,
        full_max_side=full_max_side,
    )
    result["summary"]["ground_truth_manifests"] = [str(path) for path in args.ground_truth_manifest]
    result["summary"]["dataset_root"] = str(args.dataset_root) if args.dataset_root else ""

    write_json(args.out / "summary.json", result["summary"])
    write_jsonl(args.out / "matches.jsonl", result["matches"])
    write_jsonl(args.out / "missed_gt.jsonl", result["missed"])
    write_jsonl(args.out / "unmatched_reviewed_candidates.jsonl", result["unmatched_reviewed_candidates"])
    write_jsonl(args.out / "unverified_candidates.jsonl", result["unverified_candidates"])
    write_jsonl(args.out / "overcrop_candidates.jsonl", result["overcrop"])
    write_jsonl(args.out / "per_image.jsonl", result["per_image"])
    write_markdown(args.out / "summary.md", result["summary"])

    counts = result["summary"]["counts"]
    metrics = result["summary"]["metrics"]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "candidates": counts["candidate_count"],
                "represented_gt_keys": counts["represented_gt_key_count"],
                "covered_gt_keys": counts["covered_gt_key_count"],
                "represented_gt_recall": metrics["represented_gt_recall"],
                "crop_pixels_vs_full_frame_pixels": metrics["crop_pixels_vs_full_frame_pixels"],
                "overcrop_match_count": counts["overcrop_match_count"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
