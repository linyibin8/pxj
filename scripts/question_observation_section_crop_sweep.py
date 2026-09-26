"""Sweep section/group crop strategies for dense observation pages.

Dense pages can make per-question crops more expensive than full-frame VLM
inputs, especially when the iOS per-frame cap forces full-frame fallback. This
script simulates larger section crops that cover multiple question candidates
while keeping the existing replay/GT evidence model.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from question_observation_candidate_cap_sweep import load_inputs, ranked_candidates
from question_observation_candidate_eval import gt_effective_uid, replay_full_max_side
from question_observation_dedupe_tune import (
    Candidate,
    GTBox,
    intersection,
    load_gt,
    rect_area,
)
from question_observation_fallback_tune import current_backend_policy, fallback_reason, rect_union_area


Rect = dict[str, float]


@dataclass
class SectionCrop:
    section_id: str
    image_key: str
    rect: Rect
    candidate_count: int
    source_candidate_ids: list[str]
    source_question_keys: list[str]

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


def parse_int_list(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(max(1, int(item)))
    return sorted(set(result))


def image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as raw:
            return ImageOps.exif_transpose(raw).size
    except Exception:
        return 0, 0


def clamp_rect(rect: Rect) -> Rect:
    x = max(0.0, min(1.0, float(rect.get("x") or 0.0)))
    y = max(0.0, min(1.0, float(rect.get("y") or 0.0)))
    width = max(0.0, min(1.0 - x, float(rect.get("width") or 0.0)))
    height = max(0.0, min(1.0 - y, float(rect.get("height") or 0.0)))
    return {"x": x, "y": y, "width": width, "height": height}


def expand_rect(rect: Rect, pad_x: float, pad_y: float) -> Rect:
    rect = clamp_rect(rect)
    x1 = max(0.0, rect["x"] - pad_x)
    y1 = max(0.0, rect["y"] - pad_y)
    x2 = min(1.0, rect["x"] + rect["width"] + pad_x)
    y2 = min(1.0, rect["y"] + rect["height"] + pad_y)
    return {"x": x1, "y": y1, "width": max(0.0, x2 - x1), "height": max(0.0, y2 - y1)}


def union_rect(rects: list[Rect], pad_x: float, pad_y: float) -> Rect:
    if not rects:
        return {"x": 0.0, "y": 0.0, "width": 0.0, "height": 0.0}
    x1 = min(rect["x"] for rect in rects)
    y1 = min(rect["y"] for rect in rects)
    x2 = max(rect["x"] + rect["width"] for rect in rects)
    y2 = max(rect["y"] + rect["height"] for rect in rects)
    return expand_rect({"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}, pad_x, pad_y)


def section_from_candidates(
    image_key: str,
    section_id: str,
    candidates: list[Candidate],
    pad_x: float,
    pad_y: float,
) -> SectionCrop:
    rect = union_rect([candidate.rect for candidate in candidates], pad_x=pad_x, pad_y=pad_y)
    return SectionCrop(
        section_id=section_id,
        image_key=image_key,
        rect=rect,
        candidate_count=len(candidates),
        source_candidate_ids=[candidate.candidate_id for candidate in candidates],
        source_question_keys=[candidate.question_key for candidate in candidates],
    )


def column_clusters(candidates: list[Candidate], gap_threshold: float) -> list[list[Candidate]]:
    ordered = sorted(candidates, key=lambda item: (item.rect["x"] + item.rect["width"] / 2, item.rect["y"]))
    clusters: list[list[Candidate]] = []
    cluster: list[Candidate] = []
    previous_center: float | None = None
    for candidate in ordered:
        center = candidate.rect["x"] + candidate.rect["width"] / 2
        if cluster and previous_center is not None and center - previous_center > gap_threshold:
            clusters.append(cluster)
            cluster = [candidate]
        else:
            cluster.append(candidate)
        previous_center = center
    if cluster:
        clusters.append(cluster)
    return [sorted(group, key=lambda item: (item.rect["y"], item.rect["x"])) for group in clusters]


def chunked(items: list[Candidate], max_per_section: int) -> list[list[Candidate]]:
    return [items[index : index + max_per_section] for index in range(0, len(items), max_per_section)]


def merge_sections(left: SectionCrop, right: SectionCrop, pad_x: float, pad_y: float) -> SectionCrop:
    rect = union_rect([left.rect, right.rect], pad_x=pad_x, pad_y=pad_y)
    return SectionCrop(
        section_id=f"{left.section_id}+{right.section_id}",
        image_key=left.image_key,
        rect=rect,
        candidate_count=left.candidate_count + right.candidate_count,
        source_candidate_ids=[*left.source_candidate_ids, *right.source_candidate_ids],
        source_question_keys=[*left.source_question_keys, *right.source_question_keys],
    )


def coarsen_to_limit(sections: list[SectionCrop], limit: int, pad_x: float, pad_y: float) -> list[SectionCrop]:
    if limit <= 0:
        return sections
    sections = sorted(sections, key=lambda item: (item.rect["y"], item.rect["x"]))
    while len(sections) > limit:
        best_index = 0
        best_cost = float("inf")
        for index in range(len(sections) - 1):
            merged = merge_sections(sections[index], sections[index + 1], pad_x=pad_x, pad_y=pad_y)
            cost = merged.area - sections[index].area - sections[index + 1].area
            if cost < best_cost:
                best_cost = cost
                best_index = index
        merged = merge_sections(sections[best_index], sections[best_index + 1], pad_x=pad_x, pad_y=pad_y)
        sections = [*sections[:best_index], merged, *sections[best_index + 2 :]]
    return sections


def build_sections(
    image_key: str,
    candidates: list[Candidate],
    mode: str,
    max_per_section: int,
    max_sections: int,
    pad_x: float,
    pad_y: float,
    column_gap_threshold: float,
) -> list[SectionCrop]:
    if not candidates:
        return []
    ranked = ranked_candidates(candidates)
    if mode == "individual":
        sections = [
            section_from_candidates(image_key, f"q{index + 1:02d}", [candidate], pad_x=0.0, pad_y=0.0)
            for index, candidate in enumerate(ranked)
        ]
        return sections
    if mode == "page":
        return [section_from_candidates(image_key, "page", ranked, pad_x=pad_x, pad_y=pad_y)]
    sections: list[SectionCrop] = []
    if mode == "horizontal":
        ordered = sorted(ranked, key=lambda item: (item.rect["y"], item.rect["x"]))
        for chunk_index, group in enumerate(chunked(ordered, max_per_section), start=1):
            sections.append(section_from_candidates(image_key, f"h{chunk_index:02d}", group, pad_x=pad_x, pad_y=pad_y))
    elif mode == "columns":
        for col_index, cluster in enumerate(column_clusters(ranked, column_gap_threshold), start=1):
            for chunk_index, group in enumerate(chunked(cluster, max_per_section), start=1):
                sections.append(section_from_candidates(image_key, f"c{col_index:02d}_{chunk_index:02d}", group, pad_x=pad_x, pad_y=pad_y))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return coarsen_to_limit(sections, max_sections, pad_x=pad_x, pad_y=pad_y)


def section_crop_pixels(rect: Rect, image_path: Path, max_side: int, cache: dict[Path, tuple[int, int]]) -> int:
    width, height = cache.setdefault(image_path, image_size(image_path))
    if width <= 0 or height <= 0:
        return 0
    crop_width = max(1, min(width, int(round(rect["width"] * width))))
    crop_height = max(1, min(height, int(round(rect["height"] * height))))
    longest = max(crop_width, crop_height, 1)
    scale = min(1.0, max_side / longest)
    return max(1, int(crop_width * scale)) * max(1, int(crop_height * scale))


def section_covers_gt(section: SectionCrop, gt_box: GTBox, min_gt_coverage: float) -> bool:
    overlap = intersection(section.rect, gt_box.rect)
    if overlap <= 0:
        return False
    gt_coverage = overlap / max(1e-9, rect_area(gt_box.rect))
    return gt_coverage >= min_gt_coverage


def stats_for_sections(sections: list[SectionCrop]) -> dict[str, Any]:
    total_area = sum(section.area for section in sections)
    union_area = rect_union_area([section.rect for section in sections])
    return {
        "count": len(sections),
        "area": round(total_area, 6),
        "union_area": round(union_area, 6),
        "overlap_area": round(max(0.0, total_area - union_area), 6),
        "overlap_area_ratio": round(max(0.0, total_area - union_area) / max(1e-9, total_area), 6) if total_area else 0.0,
        "max_area": round(max((section.area for section in sections), default=0.0), 6),
        "weak_count": 0,
        "low_conf_count": 0,
        "selected_candidate_count": len(sections),
        "limited_count": 0,
        "limited_unique_count": 0,
        "limited_strong_count": 0,
        "limited_confident_count": 0,
        "limited_max_confidence": 0.0,
        "ranked_limit_telemetry_count": 1,
    }


def evaluate_strategy(
    strategy_id: str,
    mode: str,
    max_per_section: int,
    section_max_side: int,
    grouped: dict[str, list[Candidate]],
    gt_by_image: dict[str, dict[str, GTBox]],
    full_pixels_by_image: dict[str, int],
    image_paths_by_key: dict[str, Path],
    min_gt_coverage: float,
    max_sections: int,
    pad_x: float,
    pad_y: float,
    column_gap_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    size_cache: dict[Path, tuple[int, int]] = {}
    policy = current_backend_policy()
    represented_gt = {uid for values in gt_by_image.values() for uid in values}
    covered_gt: set[str] = set()
    fallback_covered_gt: set[str] = set()
    total_section_pixels = 0
    fallback_pixels = 0
    section_count = 0
    fallback_reason_counts: dict[str, int] = {}
    image_rows: list[dict[str, Any]] = []

    for image_key, image_candidates in sorted(grouped.items()):
        image_path = image_paths_by_key.get(image_key, image_candidates[0].image_path if image_candidates else Path(""))
        image_gt = dict(gt_by_image.get(image_key) or {})
        sections = build_sections(
            image_key,
            image_candidates,
            mode=mode,
            max_per_section=max_per_section,
            max_sections=max_sections,
            pad_x=pad_x,
            pad_y=pad_y,
            column_gap_threshold=column_gap_threshold,
        )
        image_covered: set[str] = set()
        for section in sections:
            for gt_uid, gt in image_gt.items():
                if section_covers_gt(section, gt, min_gt_coverage=min_gt_coverage):
                    image_covered.add(gt_uid)
        covered_gt.update(image_covered)
        image_section_pixels = sum(section_crop_pixels(section.rect, image_path, section_max_side, size_cache) for section in sections)
        total_section_pixels += image_section_pixels
        section_count += len(sections)
        stats = stats_for_sections(sections)
        reason = fallback_reason(stats, policy)
        if reason:
            fallback_reason_counts[reason] = fallback_reason_counts.get(reason, 0) + 1
            fallback_pixels += full_pixels_by_image.get(image_key, 0)
            fallback_covered_gt.update(image_gt)
        image_rows.append(
            {
                "strategy_id": strategy_id,
                "image_key": image_key,
                "image": str(image_path),
                "candidate_count": len(image_candidates),
                "section_count": len(sections),
                "gt_key_count": len(image_gt),
                "covered_gt_key_count": len(image_covered),
                "missed_gt_key_count": max(0, len(image_gt) - len(image_covered)),
                "section_pixels": image_section_pixels,
                "full_frame_pixels": full_pixels_by_image.get(image_key, 0),
                "fallback_reason": reason,
                "sections": [
                    {
                        "section_id": section.section_id,
                        "candidate_count": section.candidate_count,
                        "bbox_norm": {key: round(value, 6) for key, value in section.rect.items()},
                        "area": round(section.area, 6),
                    }
                    for section in sections
                ],
            }
        )

    policy_covered_gt = covered_gt | fallback_covered_gt
    full_pixels = sum(full_pixels_by_image.values())
    gt_image_full_pixels = sum(
        full_pixels_by_image.get(image_key, 0)
        for image_key, image_gt in gt_by_image.items()
        if image_gt
    )
    row = {
        "strategy_id": strategy_id,
        "mode": mode,
        "max_per_section": max_per_section,
        "max_sections_per_image": max_sections,
        "section_max_side": section_max_side,
        "image_count": len(grouped),
        "candidate_count": sum(len(items) for items in grouped.values()),
        "section_count": section_count,
        "represented_gt_key_count": len(represented_gt),
        "section_covered_gt_key_count": len(covered_gt),
        "section_gt_recall": round(len(covered_gt) / max(1, len(represented_gt)), 6) if represented_gt else None,
        "policy_covered_gt_key_count": len(policy_covered_gt),
        "policy_gt_recall": round(len(policy_covered_gt) / max(1, len(represented_gt)), 6) if represented_gt else None,
        "fallback_image_count": sum(fallback_reason_counts.values()),
        "fallback_reason_counts": dict(sorted(fallback_reason_counts.items())),
        "section_pixels": total_section_pixels,
        "fallback_full_frame_pixels": fallback_pixels,
        "total_vlm_pixels": total_section_pixels + fallback_pixels,
        "full_frame_all_pixels": full_pixels,
        "full_frame_gt_image_pixels": gt_image_full_pixels,
        "section_pixels_vs_full": round(total_section_pixels / max(1, full_pixels), 6),
        "total_pixels_vs_full": round((total_section_pixels + fallback_pixels) / max(1, full_pixels), 6),
        "total_pixels_vs_full_gt_images": round((total_section_pixels + fallback_pixels) / max(1, gt_image_full_pixels), 6),
        "pixels_per_represented_gt": round((total_section_pixels + fallback_pixels) / max(1, len(represented_gt)), 3),
        "pixels_per_section_covered_gt": round(total_section_pixels / max(1, len(covered_gt)), 3),
    }
    return row, image_rows


def choose_recommendation(
    rows: list[dict[str, Any]],
    min_section_recall: float,
    min_policy_recall: float,
    max_total_pixels_vs_full: float,
    min_recommendation_section_max_side: int,
) -> dict[str, Any]:
    passing = [
        row
        for row in rows
        if row.get("section_gt_recall") is not None
        and float(row.get("section_gt_recall") or 0) >= min_section_recall
        and float(row.get("policy_gt_recall") or 0) >= min_policy_recall
        and float(row.get("total_pixels_vs_full") or 999) <= max_total_pixels_vs_full
        and int(row.get("section_max_side") or 0) >= min_recommendation_section_max_side
    ]
    if not passing:
        return {}
    return sorted(
        passing,
        key=lambda row: (
            float(row.get("total_pixels_vs_full") or 999),
            int(row.get("fallback_image_count") or 999999),
            int(row.get("section_count") or 999999),
        ),
    )[0]


def markdown(path: Path, summary: dict[str, Any]) -> None:
    recommendation = summary.get("recommendation") if isinstance(summary.get("recommendation"), dict) else {}
    lines = [
        "# Observation Section Crop Sweep",
        "",
        f"- images: {summary.get('image_count')}",
        f"- candidates: {summary.get('candidate_count')}",
        f"- represented GT keys: {summary.get('represented_gt_key_count')}",
        "",
        "## Recommendation",
        "",
        f"- strategy: {recommendation.get('strategy_id')}",
        f"- section recall: {recommendation.get('section_gt_recall')}",
        f"- fallback-protected recall: {recommendation.get('policy_gt_recall')}",
        f"- sections: {recommendation.get('section_count')}",
        f"- fallback images: {recommendation.get('fallback_image_count')}",
        f"- total VLM pixels vs full frames: {recommendation.get('total_pixels_vs_full')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep dense-page section/group crop strategies.")
    parser.add_argument("--replay", type=Path, action="append", required=True)
    parser.add_argument("--ground-truth-manifest", type=Path, action="append", default=[])
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--max-per-section", default="4,6,8,12")
    parser.add_argument("--section-max-side", default="900,1000,1200")
    parser.add_argument("--max-sections-per-image", type=int, default=12)
    parser.add_argument("--section-pad-x", type=float, default=0.01)
    parser.add_argument("--section-pad-y", type=float, default=0.006)
    parser.add_argument("--column-gap-threshold", type=float, default=0.18)
    parser.add_argument("--min-gt-coverage", type=float, default=0.85)
    parser.add_argument("--min-section-gt-recall", type=float, default=0.999)
    parser.add_argument("--min-policy-gt-recall", type=float, default=0.999)
    parser.add_argument("--max-total-pixels-vs-full", type=float, default=1.0)
    parser.add_argument("--min-recommendation-section-max-side", type=int, default=1200)
    args = parser.parse_args()

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    gt_boxes = load_gt(args.ground_truth_manifest, args.dataset_root) if args.ground_truth_manifest else []
    grouped, gt_by_image, full_pixels_by_image, image_paths_by_key, input_summary = load_inputs(args.replay, gt_boxes)

    rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    max_per_values = parse_int_list(args.max_per_section)
    section_max_sides = parse_int_list(args.section_max_side)
    for section_max_side in section_max_sides:
        for mode in ["individual", "page"]:
            row, per_image = evaluate_strategy(
                f"{mode}__side{section_max_side}",
                mode,
                max_per_section=999999,
                section_max_side=section_max_side,
                grouped=grouped,
                gt_by_image=gt_by_image,
                full_pixels_by_image=full_pixels_by_image,
                image_paths_by_key=image_paths_by_key,
                min_gt_coverage=args.min_gt_coverage,
                max_sections=args.max_sections_per_image,
                pad_x=args.section_pad_x,
                pad_y=args.section_pad_y,
                column_gap_threshold=args.column_gap_threshold,
            )
            rows.append(row)
            image_rows.extend(per_image)
        for max_per_section in max_per_values:
            for mode in ["horizontal", "columns"]:
                row, per_image = evaluate_strategy(
                    f"{mode}__n{max_per_section}__side{section_max_side}",
                    mode,
                    max_per_section=max_per_section,
                    section_max_side=section_max_side,
                    grouped=grouped,
                    gt_by_image=gt_by_image,
                    full_pixels_by_image=full_pixels_by_image,
                    image_paths_by_key=image_paths_by_key,
                    min_gt_coverage=args.min_gt_coverage,
                    max_sections=args.max_sections_per_image,
                    pad_x=args.section_pad_x,
                    pad_y=args.section_pad_y,
                    column_gap_threshold=args.column_gap_threshold,
                )
                rows.append(row)
                image_rows.extend(per_image)

    recommendation = choose_recommendation(
        rows,
        args.min_section_gt_recall,
        args.min_policy_gt_recall,
        args.max_total_pixels_vs_full,
        args.min_recommendation_section_max_side,
    )
    summary = {
        **input_summary,
        "settings": {
            "max_per_section": max_per_values,
            "section_max_side": section_max_sides,
            "max_sections_per_image": args.max_sections_per_image,
            "section_pad_x": args.section_pad_x,
            "section_pad_y": args.section_pad_y,
            "column_gap_threshold": args.column_gap_threshold,
            "min_gt_coverage": args.min_gt_coverage,
            "min_section_gt_recall": args.min_section_gt_recall,
            "min_policy_gt_recall": args.min_policy_gt_recall,
            "max_total_pixels_vs_full": args.max_total_pixels_vs_full,
            "min_recommendation_section_max_side": args.min_recommendation_section_max_side,
        },
        "recommendation": recommendation,
        "outputs": {
            "strategies": "strategies.jsonl",
            "per_image": "per_image.jsonl",
            "summary_md": "summary.md",
        },
    }
    write_json(args.out / "summary.json", summary)
    write_jsonl(args.out / "strategies.jsonl", rows)
    write_jsonl(args.out / "per_image.jsonl", image_rows)
    markdown(args.out / "summary.md", summary)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "images": summary.get("image_count"),
                "candidates": summary.get("candidate_count"),
                "represented_gt_keys": summary.get("represented_gt_key_count"),
                "recommendation": {
                    "strategy_id": recommendation.get("strategy_id"),
                    "section_gt_recall": recommendation.get("section_gt_recall"),
                    "policy_gt_recall": recommendation.get("policy_gt_recall"),
                    "section_count": recommendation.get("section_count"),
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
