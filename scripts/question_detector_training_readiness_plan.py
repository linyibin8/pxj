"""Plan the next reviewed-data loop for the question detector.

This is a planning/reporting tool. It does not approve labels, train a model,
or count synthetic/pseudo boxes as production evidence. It turns the current
dataset audit, review queues, release blockers, and strategy-selection evidence
into a concrete next-data plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
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


def resolve_file(path: Path, default_name: str) -> Path:
    return path / default_name if path.is_dir() else path


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def dataset_plan(path: Path) -> dict[str, Any]:
    audit_path = resolve_file(path, "audit.json")
    audit = read_json(audit_path)
    readiness = audit.get("readiness") if isinstance(audit.get("readiness"), dict) else {}
    counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
    minimums = audit.get("minimums") if isinstance(audit.get("minimums"), dict) else {}
    targets = {
        "source_images": to_int(minimums.get("production_images"), 300),
        "annotations": to_int(minimums.get("production_annotations"), 1000),
        "split_groups": to_int(minimums.get("production_split_groups"), 50),
        "negative_images": to_int(minimums.get("production_negative_images"), 50),
    }
    current = {
        "source_images": to_int(counts.get("source_images") or counts.get("images")),
        "positive_images": to_int(counts.get("positive_images")),
        "negative_images": to_int(counts.get("negative_images")),
        "annotations": to_int(counts.get("annotations")),
        "split_groups": to_int(counts.get("split_groups")),
    }
    gaps = {key: max(0, targets[key] - current.get(key, 0)) for key in targets}
    return {
        "path": str(audit_path),
        "readiness": {
            "pilot_eval_ready": bool(readiness.get("pilot_eval_ready")),
            "model_training_ready": bool(readiness.get("model_training_ready")),
            "has_error": bool(readiness.get("has_error")),
        },
        "current": current,
        "targets": targets,
        "gaps": gaps,
        "warnings": audit.get("warnings") if isinstance(audit.get("warnings"), list) else [],
    }


def group_key_for_item(item: dict[str, Any]) -> str:
    candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    for key in ("session_id", "batch_id", "image_id", "qa_event_id"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return f"{key}:{value}"
    for key in ("split_key", "source_key", "source_filename", "backlog_source_review_id"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return f"{key}:{value}"
    value = str(item.get("source_candidate") or item.get("image") or "").strip()
    return value or "unknown"


def review_workbench_plan(path: Path) -> dict[str, Any]:
    root = path if path.is_dir() else path.parent
    summary_path = resolve_file(path, "summary.json")
    summary = read_json(summary_path) if summary_path.is_file() else {}
    review_data_path = root / "review_data.json"
    review_data = read_json(review_data_path) if review_data_path.is_file() else {}
    items = review_data.get("items") if isinstance(review_data.get("items"), list) else []
    box_count = to_int(summary.get("boxes")) or sum(len(item.get("boxes") or []) for item in items if isinstance(item, dict))
    image_count = to_int(summary.get("images")) or len(items)
    approved_exports = list(root.glob("approved_boxes*.jsonl"))
    decision_files = [path for path in root.glob("*decisions*.json") if path.name != "review_data.json"]
    source_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    group_keys: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        backlog = item.get("backlog") if isinstance(item.get("backlog"), dict) else {}
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        source_kind = str(backlog.get("source_kind") or metadata.get("backlog_source_kind") or "unknown")
        source_counts[source_kind] += 1
        group_keys.add(group_key_for_item(item))
        for reason in backlog.get("reasons") or []:
            reason_counts[str(reason)] += 1
        for box in item.get("boxes") or []:
            if not isinstance(box, dict):
                continue
            for flag in box.get("quality_flags") or []:
                flag_counts[str(flag)] += 1
    return {
        "path": str(root),
        "summary_path": str(summary_path) if summary_path.is_file() else "",
        "images": image_count,
        "boxes": box_count,
        "estimated_split_groups": len(group_keys),
        "has_decisions": bool(decision_files),
        "has_approved_export": bool(approved_exports),
        "pending_boxes": 0 if approved_exports else box_count,
        "source_counts": dict(sorted(source_counts.items())),
        "reason_counts": dict(reason_counts.most_common(12)),
        "quality_flag_counts": dict(flag_counts.most_common(12)),
    }


def backlog_plan(path: Path) -> dict[str, Any]:
    summary_path = resolve_file(path, "summary.json")
    if not summary_path.is_file():
        return {}
    summary = read_json(summary_path)
    ranked_path = summary_path.parent / "annotations" / "ranked_backlog.jsonl"
    ranked = read_jsonl(ranked_path)
    return {
        "path": str(summary_path.parent),
        "summary_path": str(summary_path),
        "candidate_images": to_int(summary.get("candidate_images")),
        "selected_images": to_int(summary.get("selected_images")),
        "selected_boxes": to_int(summary.get("selected_boxes")),
        "selection": summary.get("selection") if isinstance(summary.get("selection"), dict) else {},
        "ranked_rows": len(ranked),
        "skipped_rows": sum(1 for row in ranked if row.get("skip_reason")),
    }


def release_blocker_plan(path: Path) -> dict[str, Any]:
    report_path = resolve_file(path, "release_check.json")
    report = read_json(report_path)
    hard = [item for item in report.get("hard_failures") or [] if isinstance(item, dict)]
    warnings = [item for item in report.get("warnings") or [] if isinstance(item, dict)]
    categories = Counter(str(item.get("category") or "unknown") for item in hard)
    return {
        "path": str(report_path),
        "release_ready": bool(report.get("release_ready")),
        "hard_failure_count": len(hard),
        "warning_count": len(warnings),
        "hard_failure_categories": dict(sorted(categories.items())),
        "top_hard_failures": [
            {
                "category": item.get("category"),
                "name": item.get("name"),
                "detail": item.get("detail"),
            }
            for item in hard[:8]
        ],
    }


def wave_selected_box_keys(waves: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for wave in waves:
        if not isinstance(wave, dict):
            continue
        root_value = wave.get("root")
        if not root_value:
            continue
        root = Path(str(root_value))
        draft_path = root / "annotations" / "draft_boxes.jsonl"
        for row in read_jsonl(draft_path):
            image_path = root / str(row.get("image") or "")
            if image_path.is_file():
                try:
                    image_key = hashlib.sha256(image_path.read_bytes()).hexdigest()
                except OSError:
                    image_key = str(row.get("image") or "")
            else:
                image_key = str(row.get("image") or row.get("source_candidate") or "")
            for box in row.get("boxes") or []:
                if not isinstance(box, dict):
                    continue
                bbox = box.get("bbox_px") if isinstance(box.get("bbox_px"), dict) else {}
                bbox_key = (
                    to_int(bbox.get("x")),
                    to_int(bbox.get("y")),
                    to_int(bbox.get("width")),
                    to_int(bbox.get("height")),
                )
                keys.append(f"{image_key}:{bbox_key[0]}:{bbox_key[1]}:{bbox_key[2]}:{bbox_key[3]}")
    return keys


def review_wave_plan(path: Path) -> dict[str, Any]:
    report_path = resolve_file(path, "review_wave_plan.json")
    report = read_json(report_path)
    waves = [item for item in (report.get("waves") or []) if isinstance(item, dict)]
    selected_box_keys = sorted(set(wave_selected_box_keys(waves)))
    return {
        "path": str(report_path),
        "prelabel_root": report.get("prelabel_root"),
        "exhausted": bool(report.get("exhausted")),
        "total_selected_images": to_int(report.get("total_selected_images")),
        "total_selected_boxes": to_int(report.get("total_selected_boxes")),
        "unique_selected_boxes": len(selected_box_keys) if selected_box_keys else to_int(report.get("total_selected_boxes")),
        "selected_box_keys": selected_box_keys[:200],
        "wave_count": len(waves),
        "waves": [
            {
                "wave": item.get("wave"),
                "root": item.get("root"),
                "candidate_count": item.get("candidate_count"),
                "selected_images": item.get("selected_images"),
                "selected_boxes": item.get("selected_boxes"),
                "workbench": item.get("workbench"),
            }
            for item in waves[:8]
        ],
    }


def strategy_plan(path: Path) -> dict[str, Any]:
    report_path = resolve_file(path, "strategy_selection.json")
    report = read_json(report_path)
    best = (report.get("best_reviewed_live_by_cost") or [{}])[0]
    propagated = next(
        (
            row
            for row in report.get("best_reviewed_live_by_cost") or []
            if isinstance(row, dict) and row.get("evidence_kind") == "reviewed_propagated"
        ),
        {},
    )
    return {
        "path": str(report_path),
        "row_count": to_int(report.get("row_count")),
        "best_reviewed_live_strategy": {
            "strategy_id": best.get("strategy_id"),
            "family": best.get("family"),
            "replay": best.get("replay"),
            "recall": to_float(best.get("policy_gt_recall") if best.get("policy_gt_recall") is not None else best.get("crop_gt_recall")),
            "fallback_image_ratio": to_float(best.get("fallback_image_ratio")),
            "total_pixels_vs_full": to_float(best.get("total_pixels_vs_full")),
        },
        "best_propagated_live_strategy": {
            "strategy_id": propagated.get("strategy_id"),
            "family": propagated.get("family"),
            "replay": propagated.get("replay"),
            "recall": to_float(propagated.get("policy_gt_recall") if propagated.get("policy_gt_recall") is not None else propagated.get("crop_gt_recall")),
            "fallback_image_ratio": to_float(propagated.get("fallback_image_ratio")),
            "total_pixels_vs_full": to_float(propagated.get("total_pixels_vs_full")),
        } if propagated else {},
    }


def build_actions(
    dataset: dict[str, Any],
    workbenches: list[dict[str, Any]],
    backlog: dict[str, Any],
    wave_plans: list[dict[str, Any]],
    release: dict[str, Any],
    strategy: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    pending_boxes = sum(to_int(workbench.get("pending_boxes")) for workbench in workbenches)
    pending_images = sum(to_int(workbench.get("images")) for workbench in workbenches if to_int(workbench.get("pending_boxes")))
    pending_groups = sum(to_int(workbench.get("estimated_split_groups")) for workbench in workbenches if to_int(workbench.get("pending_boxes")))
    current = dataset.get("current") or {}
    gaps = dataset.get("gaps") or {}
    after_pending = {
        "source_images": to_int(current.get("source_images")) + pending_images,
        "annotations": to_int(current.get("annotations")) + pending_boxes,
        "split_groups": to_int(current.get("split_groups")) + pending_groups,
        "negative_images": to_int(current.get("negative_images")),
    }
    remaining_after_pending = {
        key: max(0, to_int((dataset.get("targets") or {}).get(key)) - value)
        for key, value in after_pending.items()
    }
    exhausted_low_yield = [
        plan for plan in wave_plans
        if plan.get("exhausted") and to_int(plan.get("total_selected_boxes")) < args.review_boxes_per_wave
    ]
    active_capacity_plans = [
        plan for plan in wave_plans
        if (not plan.get("exhausted")) or to_int(plan.get("unique_selected_boxes") or plan.get("total_selected_boxes")) >= args.review_boxes_per_wave
    ]
    exhausted_box_keys = {
        str(key)
        for plan in exhausted_low_yield
        for key in (plan.get("selected_box_keys") or [])
        if key
    }
    current_pool_capacity = (
        len(exhausted_box_keys)
        if exhausted_box_keys
        else sum(to_int(plan.get("unique_selected_boxes") or plan.get("total_selected_boxes")) for plan in exhausted_low_yield)
    )
    wave_size = pending_boxes or args.review_boxes_per_wave
    more_waves = math.ceil(remaining_after_pending["annotations"] / max(1, wave_size))
    if pending_boxes:
        if remaining_after_pending["annotations"] > 0:
            review_detail = (
                f"Review/export the current {pending_boxes} pending boxes before training; "
                "even if all pass, production annotations remain short."
            )
        else:
            review_detail = (
                f"Review/export the current {pending_boxes} pending boxes before training; "
                "if approval and dedupe hold, this backlog can cover the production annotation target."
            )
        actions.append(
            {
                "priority": 1,
                "code": "finish_current_review_backlog",
                "detail": review_detail,
                "evidence": {"pending_boxes": pending_boxes, "pending_images": pending_images, "pending_groups": pending_groups},
            }
        )
    if remaining_after_pending["annotations"] > 0:
        if active_capacity_plans:
            detail = (
                f"Continue active batches from the non-exhausted expanded candidate pool; "
                f"about {remaining_after_pending['annotations']} reviewed annotations are still needed."
            )
        elif exhausted_low_yield:
            detail = (
                f"After expanding the candidate pool, create more active batches; "
                f"about {remaining_after_pending['annotations']} reviewed annotations are still needed."
            )
        else:
            detail = f"After the current backlog, about {remaining_after_pending['annotations']} reviewed annotations are still needed."
        actions.append(
            {
                "priority": 2,
                "code": "create_more_active_batches",
                "detail": detail,
                "evidence": {
                    "similar_review_waves_needed": more_waves,
                    "assumed_boxes_per_wave": wave_size,
                    "current_pool_capacity": current_pool_capacity,
                    "wave_estimate_valid_after_pool_expansion": bool(exhausted_low_yield),
                    "remaining_after_pending": remaining_after_pending,
                },
            }
        )
    if exhausted_low_yield and (not active_capacity_plans or remaining_after_pending["annotations"] > 0):
        actions.append(
            {
                "priority": 6 if active_capacity_plans else 2,
                "code": "expand_candidate_pool",
                "detail": (
                    "Older prelabel pools are exhausted, but a newer expanded pool still has review capacity."
                    if active_capacity_plans
                    else "The current prelabel pool is exhausted; mine broader historical SQLite/image pools before expecting more active batches."
                ),
                "evidence": {
                    "active_capacity_plans": [
                        {
                            "path": plan.get("path"),
                            "prelabel_root": plan.get("prelabel_root"),
                            "total_selected_boxes": plan.get("total_selected_boxes"),
                            "exhausted": plan.get("exhausted"),
                        }
                        for plan in active_capacity_plans
                    ],
                    "exhausted_wave_plans": [
                        {
                            "path": plan.get("path"),
                            "prelabel_root": plan.get("prelabel_root"),
                            "total_selected_boxes": plan.get("total_selected_boxes"),
                            "unique_selected_boxes": plan.get("unique_selected_boxes"),
                        }
                        for plan in exhausted_low_yield
                    ]
                },
            }
        )
    if remaining_after_pending["split_groups"] > 0:
        actions.append(
            {
                "priority": 3,
                "code": "diversify_capture_sessions",
                "detail": f"Need about {remaining_after_pending['split_groups']} more split/session groups after the current backlog.",
                "evidence": {"remaining_split_groups": remaining_after_pending["split_groups"]},
            }
        )
    elif to_int(gaps.get("split_groups")) > 0 and pending_groups:
        actions.append(
            {
                "priority": 3,
                "code": "verify_split_group_gain_after_merge",
                "detail": "Pending review may close the split-group gap only if those groups survive approval/dedup; rerun dataset audit after merge.",
                "evidence": {
                    "current_split_group_gap": gaps.get("split_groups"),
                    "optimistic_pending_groups": pending_groups,
                    "projected_split_groups": after_pending["split_groups"],
                },
            }
        )
    if release and not release.get("release_ready"):
        actions.append(
            {
                "priority": 4,
                "code": "respect_release_blockers",
                "detail": "Do not promote a Core ML detector until release blockers clear.",
                "evidence": release.get("hard_failure_categories") or {},
            }
        )
    best_strategy = strategy.get("best_reviewed_live_strategy") if isinstance(strategy.get("best_reviewed_live_strategy"), dict) else {}
    if best_strategy:
        actions.append(
            {
                "priority": 5,
                "code": "keep_current_observation_strategy",
                "detail": "Keep the current reviewed-live observation strategy while the training dataset grows.",
                "evidence": best_strategy,
            }
        )
    if backlog:
        actions.append(
            {
                "priority": 6,
                "code": "preserve_review_provenance",
                "detail": "Backlog outputs are review queues only; merge only approved/corrected/verified exports into training.",
                "evidence": {"backlog": backlog.get("path"), "selected_boxes": backlog.get("selected_boxes")},
            }
        )
    return sorted(actions, key=lambda item: item["priority"])


def markdown(report: dict[str, Any]) -> str:
    dataset = report.get("dataset") or {}
    current = dataset.get("current") or {}
    targets = dataset.get("targets") or {}
    gaps = dataset.get("gaps") or {}
    pending = report.get("pending_review") or {}
    projection = report.get("projection") or {}
    lines = [
        "# Question Detector Training Readiness Plan",
        "",
        f"- Generated: {report.get('generated_at')}",
        f"- Dataset ready: `{str((dataset.get('readiness') or {}).get('model_training_ready')).lower()}`",
        f"- Current annotations: {current.get('annotations')} / {targets.get('annotations')} (gap {gaps.get('annotations')})",
        f"- Pending review boxes: {pending.get('boxes')} across {pending.get('images')} images",
        f"- Optimistic annotations after pending review: {projection.get('annotations_after_pending')} / {targets.get('annotations')}",
        f"- Optimistic split groups after pending review: {projection.get('split_groups_after_pending')} / {targets.get('split_groups')} (verify after approved merge)",
        "",
        "## Actions",
        "",
    ]
    for action in report.get("actions") or []:
        lines.append(f"- P{action['priority']} `{action['code']}`: {action['detail']}")
    lines.extend(["", "## Strategy", ""])
    strategy = (report.get("strategy") or {}).get("best_reviewed_live_strategy") or {}
    if strategy:
        lines.append(
            f"- Keep `{strategy.get('strategy_id')}` while labels grow: recall={strategy.get('recall')} "
            f"fallback={strategy.get('fallback_image_ratio')} pixels/full={strategy.get('total_pixels_vs_full')}"
        )
    lines.extend(["", "## Review Sources", ""])
    for workbench in report.get("review_workbenches") or []:
        lines.append(
            f"- `{workbench.get('path')}` boxes={workbench.get('boxes')} pending={workbench.get('pending_boxes')} "
            f"groups~={workbench.get('estimated_split_groups')} sources={workbench.get('source_counts')}"
        )
    if report.get("review_wave_plans"):
        lines.extend(["", "## Review Wave Capacity", ""])
        for plan in report.get("review_wave_plans") or []:
            lines.append(
                f"- `{plan.get('path')}` exhausted={plan.get('exhausted')} "
                f"selected={plan.get('total_selected_images')} images / {plan.get('total_selected_boxes')} boxes "
                f"unique_boxes={plan.get('unique_selected_boxes')}"
            )
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = dataset_plan(args.dataset)
    workbenches = [review_workbench_plan(path) for path in args.review_workbench]
    backlog = backlog_plan(args.review_backlog) if args.review_backlog else {}
    wave_plans = [review_wave_plan(path) for path in args.review_wave_plan]
    release = release_blocker_plan(args.release_check) if args.release_check else {}
    strategy = strategy_plan(args.strategy_selection) if args.strategy_selection else {}
    pending_boxes = sum(to_int(item.get("pending_boxes")) for item in workbenches)
    pending_images = sum(to_int(item.get("images")) for item in workbenches if to_int(item.get("pending_boxes")))
    pending_groups = sum(to_int(item.get("estimated_split_groups")) for item in workbenches if to_int(item.get("pending_boxes")))
    current = dataset.get("current") or {}
    targets = dataset.get("targets") or {}
    annotations_after_pending = to_int(current.get("annotations")) + pending_boxes
    images_after_pending = to_int(current.get("source_images")) + pending_images
    groups_after_pending = to_int(current.get("split_groups")) + pending_groups
    projection = {
        "annotations_after_pending": annotations_after_pending,
        "source_images_after_pending": images_after_pending,
        "split_groups_after_pending": groups_after_pending,
        "remaining_annotation_gap_after_pending": max(0, to_int(targets.get("annotations")) - annotations_after_pending),
        "remaining_image_gap_after_pending": max(0, to_int(targets.get("source_images")) - images_after_pending),
        "remaining_split_group_gap_after_pending": max(0, to_int(targets.get("split_groups")) - groups_after_pending),
    }
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dataset": str(args.dataset),
            "review_workbenches": [str(path) for path in args.review_workbench],
            "review_backlog": str(args.review_backlog) if args.review_backlog else "",
            "review_wave_plans": [str(path) for path in args.review_wave_plan],
            "release_check": str(args.release_check) if args.release_check else "",
            "strategy_selection": str(args.strategy_selection) if args.strategy_selection else "",
        },
        "dataset": dataset,
        "review_workbenches": workbenches,
        "review_backlog": backlog,
        "review_wave_plans": wave_plans,
        "release_check": release,
        "strategy": strategy,
        "pending_review": {
            "images": pending_images,
            "boxes": pending_boxes,
            "estimated_split_groups": pending_groups,
        },
        "projection": projection,
        "projection_notes": [
            "Projection assumes every pending review box is approved and every estimated pending group survives dataset dedupe.",
            "Only a fresh dataset audit after approved-box merge can prove model_training_ready.",
        ],
    }
    report["actions"] = build_actions(dataset, workbenches, backlog, wave_plans, release, strategy, args)
    write_json(args.out / "training_readiness_plan.json", report)
    (args.out / "training_readiness_plan.md").write_text(markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan the next reviewed-data loop for detector training readiness.")
    parser.add_argument("--dataset", type=Path, required=True, help="Detector dataset root or audit.json.")
    parser.add_argument("--review-workbench", type=Path, action="append", default=[])
    parser.add_argument("--review-backlog", type=Path)
    parser.add_argument("--review-wave-plan", type=Path, action="append", default=[])
    parser.add_argument("--release-check", type=Path)
    parser.add_argument("--strategy-selection", type=Path)
    parser.add_argument("--review-boxes-per-wave", type=int, default=80)
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-training-readiness-plan"))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "model_training_ready": report["dataset"]["readiness"]["model_training_ready"],
                "pending_boxes": report["pending_review"]["boxes"],
                "annotations_after_pending": report["projection"]["annotations_after_pending"],
                "remaining_annotation_gap_after_pending": report["projection"]["remaining_annotation_gap_after_pending"],
                "actions": [item["code"] for item in report["actions"]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
