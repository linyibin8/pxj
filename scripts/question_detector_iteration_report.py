"""Summarize a question-detector iteration and recommend next actions.

This report sits above the individual gates. It does not decide release
readiness itself; question_detector_release_check.py remains the hard gate.
Instead, it answers: what did this iteration learn, what is blocking progress,
and what should the next loop do?
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_file(path: Path, default_name: str) -> Path:
    return path / default_name if path.is_dir() else path


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


def to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def dataset_summary(path: Path) -> dict[str, Any]:
    audit_path = resolve_file(path, "audit.json")
    audit = read_json(audit_path)
    readiness = audit.get("readiness") if isinstance(audit.get("readiness"), dict) else {}
    counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
    minimums = audit.get("minimums") if isinstance(audit.get("minimums"), dict) else {}
    bootstrap = audit.get("bootstrap") if isinstance(audit.get("bootstrap"), dict) else {}
    targets = {
        "source_images": to_int(minimums.get("production_images") or 300),
        "annotations": to_int(minimums.get("production_annotations") or 1000),
        "split_groups": to_int(minimums.get("production_split_groups") or 50),
        "negative_images": to_int(minimums.get("production_negative_images") or 1),
    }
    values = {
        "source_images": to_int(counts.get("source_images") or counts.get("images")),
        "positive_images": to_int(counts.get("positive_images")),
        "negative_images": to_int(counts.get("negative_images")),
        "annotations": to_int(counts.get("annotations")),
        "human_reviewed_annotations": to_int(counts.get("human_reviewed_annotations")),
        "pseudo_annotations": to_int(counts.get("pseudo_annotations")),
        "split_groups": to_int(counts.get("split_groups")),
    }
    if not values["human_reviewed_annotations"] and not values["pseudo_annotations"]:
        values["human_reviewed_annotations"] = values["annotations"]
    gaps = {
        key: max(0, targets[key] - values.get(key, 0))
        for key in targets
    }
    pseudo_ratio = values["pseudo_annotations"] / max(1, values["annotations"])
    return {
        "path": str(audit_path),
        "pilot_eval_ready": bool(readiness.get("pilot_eval_ready")),
        "model_training_ready": bool(readiness.get("model_training_ready")),
        "has_error": bool(readiness.get("has_error")),
        "counts": values,
        "targets": targets,
        "gaps": gaps,
        "pseudo_ratio": round(pseudo_ratio, 4),
        "is_bootstrap": bool(bootstrap),
        "warning_count": len(audit.get("warnings") or []),
    }


def manifest_eval_summary(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    eval_summary = manifest.get("eval") if isinstance(manifest.get("eval"), dict) else {}
    primary_key = f"{to_float(eval_summary.get('primary_threshold'), 0.5):.2f}"
    primary = {}
    if isinstance(eval_summary.get("thresholds"), dict):
        primary = eval_summary["thresholds"].get(primary_key) or {}
    return {
        "path": str(path),
        "status": manifest.get("status") or "",
        "model": manifest.get("model") or "",
        "imgsz": (manifest.get("train") or {}).get("imgsz") if isinstance(manifest.get("train"), dict) else None,
        "dataset": manifest.get("dataset") or "",
        "coreml_artifact": manifest.get("coreml_artifact") or "",
        "eval_gate_passed": bool((manifest.get("eval_gate_result") or {}).get("passed")) if isinstance(manifest.get("eval_gate_result"), dict) else None,
        "primary_threshold": primary_key,
        "gt_count": to_int(primary.get("gt_count")),
        "pred_count": to_int(primary.get("pred_count")),
        "precision": to_float(primary.get("precision")),
        "recall": to_float(primary.get("recall")),
        "f1": to_float(primary.get("f1")),
        "fp_per_image": to_float(primary.get("fp_per_image")),
        "missed_question_rate": to_float(primary.get("missed_question_rate"), 1.0),
    }


def experiment_summary(path: Path) -> dict[str, Any]:
    report_path = resolve_file(path, "experiment_report.json")
    if path.is_file() and path.name == "train_manifest.json":
        best = manifest_eval_summary(path)
        return {
            "path": str(path),
            "status": best.get("status") or "",
            "run_count": 1,
            "executed_run_count": 1,
            "selection_summary": "",
            "best_manifest": best,
            "manifest_status_counts": {str(best.get("status") or ""): 1},
        }
    report = read_json(report_path)
    manifests: list[dict[str, Any]] = []
    for item in report.get("executed_runs") or []:
        if not isinstance(item, dict):
            continue
        manifest_path = Path(str(item.get("manifest") or ""))
        if manifest_path.is_file():
            try:
                manifests.append(manifest_eval_summary(manifest_path))
            except (OSError, json.JSONDecodeError):
                pass
    best = None
    if manifests:
        best = sorted(
            manifests,
            key=lambda row: (
                row.get("eval_gate_passed") is not True,
                -to_float(row.get("recall")),
                -to_float(row.get("precision")),
                to_float(row.get("fp_per_image")),
                -to_float(row.get("f1")),
            ),
        )[0]
    return {
        "path": str(report_path),
        "status": report.get("status") or "",
        "run_count": to_int(report.get("run_count")),
        "executed_run_count": len(report.get("executed_runs") or []),
        "selection_summary": (report.get("selection") or {}).get("summary") if isinstance(report.get("selection"), dict) else "",
        "best_manifest": best,
        "manifest_status_counts": dict(Counter(row.get("status") or "" for row in manifests)),
    }


def release_summary(path: Path) -> dict[str, Any]:
    report_path = resolve_file(path, "release_check.json")
    report = read_json(report_path)
    hard = report.get("hard_failures") if isinstance(report.get("hard_failures"), list) else []
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    categories = Counter(str(item.get("category") or "unknown") for item in hard if isinstance(item, dict))
    return {
        "path": str(report_path),
        "release_ready": bool(report.get("release_ready")),
        "hard_failure_count": len(hard),
        "warning_count": len(warnings),
        "hard_failure_categories": dict(categories),
        "top_hard_failures": [
            {
                "category": item.get("category"),
                "name": item.get("name"),
                "detail": item.get("detail"),
            }
            for item in hard[:8]
            if isinstance(item, dict)
        ],
    }


def replay_summary(path: Path) -> dict[str, Any]:
    summary_path = resolve_file(path, "summary.json")
    report = read_json(summary_path)
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    upload = report.get("upload_bytes") if isinstance(report.get("upload_bytes"), dict) else {}
    vlm = report.get("vlm_proxy") if isinstance(report.get("vlm_proxy"), dict) else {}
    layout = report.get("layout_measurement") if isinstance(report.get("layout_measurement"), dict) else {}
    latency = layout.get("latency_ms") if isinstance(layout.get("latency_ms"), dict) else {}
    return {
        "path": str(summary_path),
        "images": to_int(counts.get("images")),
        "cross_frame_dedup_boxes": to_int(counts.get("cross_frame_dedup_boxes")),
        "rect_only_vs_full": to_float(upload.get("rect_only_cross_frame_dedup_vs_full"), math.inf),
        "full_plus_crop_vs_full": to_float(upload.get("full_plus_cross_frame_dedup_crop_vs_full"), math.inf),
        "crop_pixels_vs_full": to_float(vlm.get("crop_pixels_vs_full_pixels"), math.inf),
        "layout_median_ms": to_float(latency.get("median"), math.inf),
        "layout_p95_ms": to_float(latency.get("p95"), math.inf),
    }


def strategy_selection_summary(path: Path) -> dict[str, Any]:
    summary_path = resolve_file(path, "strategy_selection.json")
    report = read_json(summary_path)
    best_reviewed = [row for row in (report.get("best_reviewed_live_by_cost") or []) if isinstance(row, dict)]
    synthetic_rows = [row for row in (report.get("synthetic_stress_rows") or []) if isinstance(row, dict)]
    rows = [row for row in (report.get("rows") or []) if isinstance(row, dict)]
    best = best_reviewed[0] if best_reviewed else {}
    propagated = next((row for row in best_reviewed if row.get("evidence_kind") == "reviewed_propagated"), {})
    return {
        "path": str(summary_path),
        "row_count": to_int(report.get("row_count")) or len(rows),
        "best_reviewed_live_strategy": {
            "strategy_id": best.get("strategy_id"),
            "family": best.get("family"),
            "replay": best.get("replay"),
            "recall": to_float(best.get("policy_gt_recall") if best.get("policy_gt_recall") is not None else best.get("crop_gt_recall")),
            "fallback_image_ratio": to_float(best.get("fallback_image_ratio"), math.inf),
            "total_pixels_vs_full": to_float(best.get("total_pixels_vs_full"), math.inf),
        },
        "best_propagated_live_strategy": {
            "strategy_id": propagated.get("strategy_id"),
            "family": propagated.get("family"),
            "replay": propagated.get("replay"),
            "recall": to_float(propagated.get("policy_gt_recall") if propagated.get("policy_gt_recall") is not None else propagated.get("crop_gt_recall")),
            "fallback_image_ratio": to_float(propagated.get("fallback_image_ratio"), math.inf),
            "total_pixels_vs_full": to_float(propagated.get("total_pixels_vs_full"), math.inf),
        } if propagated else {},
        "synthetic_stress_row_count": len(synthetic_rows),
    }


def training_plan_summary(path: Path) -> dict[str, Any]:
    summary_path = resolve_file(path, "training_readiness_plan.json")
    report = read_json(summary_path)
    projection = report.get("projection") if isinstance(report.get("projection"), dict) else {}
    pending = report.get("pending_review") if isinstance(report.get("pending_review"), dict) else {}
    dataset = report.get("dataset") if isinstance(report.get("dataset"), dict) else {}
    readiness = dataset.get("readiness") if isinstance(dataset.get("readiness"), dict) else {}
    return {
        "path": str(summary_path),
        "model_training_ready": bool(readiness.get("model_training_ready")),
        "pending_boxes": to_int(pending.get("boxes")),
        "pending_images": to_int(pending.get("images")),
        "annotations_after_pending": to_int(projection.get("annotations_after_pending")),
        "remaining_annotation_gap_after_pending": to_int(projection.get("remaining_annotation_gap_after_pending")),
        "remaining_image_gap_after_pending": to_int(projection.get("remaining_image_gap_after_pending")),
        "remaining_split_group_gap_after_pending": to_int(projection.get("remaining_split_group_gap_after_pending")),
        "actions": [
            {
                "code": item.get("code"),
                "detail": item.get("detail"),
                "priority": item.get("priority"),
            }
            for item in (report.get("actions") or [])[:8]
            if isinstance(item, dict)
        ],
    }


def review_queue_summary(path: Path) -> dict[str, Any]:
    manifest_path = resolve_file(path, "review_queue_manifest.json")
    report = read_json(manifest_path)
    index_html = manifest_path.parent / "index.html"
    return {
        "path": str(manifest_path),
        "index_html": str(index_html) if index_html.is_file() else "",
        "workbench_count": to_int(report.get("workbench_count")),
        "total_boxes": to_int(report.get("total_boxes")),
        "pending_boxes": to_int(report.get("pending_boxes")),
        "approved_boxes": to_int(report.get("approved_boxes")),
        "needs_review_count": to_int(report.get("needs_review_count")),
        "needs_export_count": to_int(report.get("needs_export_count")),
        "merge_command": report.get("merge_command") or "",
    }


def error_mining_summary(path: Path) -> dict[str, Any]:
    summary_path = resolve_file(path, "summary.json")
    report = read_json(summary_path)
    stats = report.get("stats") if isinstance(report.get("stats"), dict) else {}
    return {
        "path": str(summary_path),
        "threshold": report.get("threshold") or "",
        "images": to_int(stats.get("images")),
        "missed_boxes": to_int(stats.get("missed:boxes")),
        "false_positive_boxes": to_int(stats.get("false_positive:boxes")),
        "stats": stats,
    }


def approved_jsonl_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    box_count = 0
    for row in rows:
        boxes = row.get("boxes") if isinstance(row.get("boxes"), list) else []
        box_count += len(boxes)
    summary_path = path.with_suffix(".summary.json")
    summary = read_json(summary_path) if summary_path.is_file() else {}
    return {
        "path": str(path),
        "summary_path": str(summary_path) if summary_path.is_file() else "",
        "images": to_int(summary.get("images")) or len(rows),
        "boxes": to_int(summary.get("boxes")) or box_count,
    }


def decisions_summary(path: Path, approved_statuses: set[str]) -> dict[str, Any]:
    payload = read_json(path)
    decisions = payload.get("decisions") if isinstance(payload, dict) else {}
    if not isinstance(decisions, dict):
        decisions = {}
    status_counts: Counter[str] = Counter()
    box_count = 0
    approved_count = 0
    image_count = 0
    for decision in decisions.values():
        if not isinstance(decision, dict):
            continue
        image_count += 1
        boxes = decision.get("boxes") if isinstance(decision.get("boxes"), list) else []
        for box in boxes:
            if not isinstance(box, dict):
                continue
            status = str(box.get("status") or "pending").strip().lower() or "pending"
            status_counts[status] += 1
            box_count += 1
            if status in approved_statuses:
                approved_count += 1
    return {
        "path": str(path),
        "images_with_decisions": image_count,
        "box_decisions": box_count,
        "approved_box_decisions": approved_count,
        "status_counts": dict(sorted(status_counts.items())),
    }


def review_workbench_summary(path: Path, approved_statuses: set[str]) -> dict[str, Any]:
    root = path if path.is_dir() else path.parent
    summary_path = resolve_file(path, "summary.json")
    summary = read_json(summary_path) if summary_path.is_file() else {}
    review_data_path = root / "review_data.json"
    review_data = read_json(review_data_path) if review_data_path.is_file() else {}
    review_items = review_data.get("items") if isinstance(review_data.get("items"), list) else []
    item_count = to_int(summary.get("images")) or len(review_items)
    box_count = to_int(summary.get("boxes")) or sum(len(item.get("boxes") or []) for item in review_items if isinstance(item, dict))
    prelabel_root = str(summary.get("prelabel_root") or review_data.get("source") or "").strip()
    include_quarantine = bool(summary.get("include_quarantine"))

    decision_candidates = [
        root / "question_box_decisions.json",
        root / "decisions.json",
    ]
    decision_candidates.extend(sorted(root.glob("*decisions*.json")))
    decision_paths: list[Path] = []
    seen_decisions: set[str] = set()
    for candidate in decision_candidates:
        if not candidate.is_file() or candidate.name == "review_data.json":
            continue
        key = str(candidate.resolve())
        if key in seen_decisions:
            continue
        seen_decisions.add(key)
        decision_paths.append(candidate)
    decisions = []
    for decision_path in decision_paths:
        try:
            decisions.append(decisions_summary(decision_path, approved_statuses))
        except (OSError, json.JSONDecodeError):
            decisions.append({"path": str(decision_path), "error": "unreadable_decisions"})

    approved_paths = sorted(root.glob("approved_boxes*.jsonl"))
    approved_paths.extend(sorted((root / "annotations").glob("approved_boxes*.jsonl")) if (root / "annotations").is_dir() else [])
    approved: list[dict[str, Any]] = []
    seen_approved: set[str] = set()
    for approved_path in approved_paths:
        key = str(approved_path.resolve())
        if key in seen_approved:
            continue
        seen_approved.add(key)
        approved.append(approved_jsonl_summary(approved_path))

    approved_box_count = sum(to_int(item.get("boxes")) for item in approved)
    decision_box_count = sum(to_int(item.get("box_decisions")) for item in decisions)
    pending_box_count = max(0, box_count - decision_box_count) if decisions else box_count
    export_command = ""
    if decisions and not approved and prelabel_root:
        decision_path = decisions[0].get("path") or ""
        export_parts = [
            "python scripts\\question_detector_review_workbench.py",
            f"--prelabel-root {prelabel_root}",
            f"--decisions {decision_path}",
            f"--approved-jsonl {root}\\approved_boxes.jsonl",
        ]
        if include_quarantine:
            export_parts.append("--include-quarantine")
        export_command = " ".join(export_parts)
    return {
        "path": str(root),
        "summary_path": str(summary_path) if summary_path.is_file() else "",
        "prelabel_root": prelabel_root,
        "index_html": str(root / "index.html") if (root / "index.html").is_file() else "",
        "review_data": str(review_data_path) if review_data_path.is_file() else "",
        "images": item_count,
        "boxes": box_count,
        "decisions": decisions,
        "approved_exports": approved,
        "decision_file_count": len(decisions),
        "approved_export_count": len(approved),
        "approved_box_count": approved_box_count,
        "pending_box_count": pending_box_count,
        "has_decisions": bool(decisions),
        "has_approved_export": bool(approved),
        "export_command": export_command,
        "approved_inputs": [item["path"] for item in approved if item.get("path")],
        "needs_review": not decisions,
        "needs_export": bool(decisions) and not approved,
    }


def add_action(actions: list[dict[str, Any]], priority: int, code: str, detail: str, evidence: dict[str, Any] | None = None) -> None:
    actions.append({"priority": priority, "code": code, "detail": detail, "evidence": evidence or {}})


def recommend(report: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    datasets = report.get("datasets") or []
    experiments = report.get("experiments") or []
    releases = report.get("release_checks") or []
    errors = report.get("error_mining") or []
    replays = report.get("replays") or []
    strategies = report.get("strategy_selections") or []
    training_plans = report.get("training_plans") or []
    review_queues = report.get("review_queues") or []
    workbenches = report.get("review_workbenches") or []
    projected_annotation_gap_closed = any(
        to_int(plan.get("pending_boxes")) > 0 and to_int(plan.get("remaining_annotation_gap_after_pending")) <= 0
        for plan in training_plans
    )
    projected_split_gap_closed = any(
        to_int(plan.get("pending_boxes")) > 0 and to_int(plan.get("remaining_split_group_gap_after_pending")) <= 0
        for plan in training_plans
    )

    workbench_pending_review_boxes = sum(to_int(item.get("pending_box_count")) for item in workbenches if item.get("needs_review"))
    queue_pending_review_boxes = sum(to_int(item.get("pending_boxes")) for item in review_queues if to_int(item.get("pending_boxes")) > 0)
    pending_review_boxes = workbench_pending_review_boxes or queue_pending_review_boxes
    pending_export_boxes = sum(to_int(item.get("boxes")) for item in workbenches if item.get("needs_export"))
    approved_boxes = sum(to_int(item.get("approved_box_count")) for item in workbenches)
    if pending_review_boxes:
        add_action(
            actions,
            8,
            "complete_review_workbenches",
            f"Review workbenches still have about {pending_review_boxes} boxes without a decisions export.",
            {
                "pending_boxes": pending_review_boxes,
                "review_queues": [
                    {"path": item.get("path"), "index_html": item.get("index_html"), "pending_boxes": item.get("pending_boxes")}
                    for item in review_queues
                ][:4],
                "workbenches": [
                    {"path": item.get("path"), "index_html": item.get("index_html"), "boxes": item.get("boxes")}
                    for item in workbenches
                    if item.get("needs_review")
                ][:8],
            },
        )
    if pending_export_boxes:
        add_action(
            actions,
            9,
            "export_approved_review_boxes",
            "Review decisions exist but approved_boxes.jsonl has not been exported for at least one workbench.",
            {
                "pending_export_boxes": pending_export_boxes,
                "commands": [item.get("export_command") for item in workbenches if item.get("needs_export") and item.get("export_command")][:8],
            },
        )
    if approved_boxes:
        approved_inputs: list[str] = []
        for workbench in workbenches:
            approved_inputs.extend(str(path) for path in (workbench.get("approved_inputs") or []) if path)
        merge_command = ""
        if approved_inputs:
            merge_command = f"python scripts\\question_detector_merge_reviewed.py {' '.join(approved_inputs)} --out diagnostics\\question-detector-merged-reviewed-next --clean"
        add_action(
            actions,
            18,
            "merge_approved_review_boxes",
            f"{approved_boxes} approved reviewed boxes are available; merge them into the next detector dataset build.",
            {
                "approved_boxes": approved_boxes,
                "approved_inputs": approved_inputs,
                "command": merge_command,
            },
        )

    for dataset in datasets:
        gaps = dataset.get("gaps") or {}
        counts = dataset.get("counts") or {}
        if dataset.get("is_bootstrap") or to_float(dataset.get("pseudo_ratio")) > 0:
            add_action(
                actions,
                10,
                "review_pseudo_labels",
                "Bootstrap/pseudo labels are present; approve or correct hard examples before treating this data as training evidence.",
                {"pseudo_annotations": counts.get("pseudo_annotations"), "pseudo_ratio": dataset.get("pseudo_ratio")},
            )
        if not dataset.get("model_training_ready"):
            add_action(
                actions,
                20,
                "grow_reviewed_dataset",
                "Dataset is not model-training-ready; expand reviewed labels and split groups before release training.",
                {"gaps": gaps, "counts": counts},
            )
        if to_int(gaps.get("annotations")) > 0 and not projected_annotation_gap_closed:
            add_action(
                actions,
                30,
                "annotate_more_boxes",
                f"Need about {gaps['annotations']} more reviewed annotations for the default production target.",
                {"target": dataset.get("targets", {}).get("annotations"), "current": counts.get("annotations")},
            )
        if to_int(gaps.get("split_groups")) > 0 and not projected_split_gap_closed:
            add_action(
                actions,
                35,
                "diversify_sessions",
                f"Need about {gaps['split_groups']} more split groups to make held-out evaluation reliable.",
                {"target": dataset.get("targets", {}).get("split_groups"), "current": counts.get("split_groups")},
            )

    for experiment in experiments:
        best = experiment.get("best_manifest") or {}
        if experiment.get("status") == "failed_eval_gate" or best.get("eval_gate_passed") is False:
            add_action(
                actions,
                15,
                "mine_and_review_model_errors",
                "Latest detector failed the eval gate; mine missed/false-positive images and review them before the next matrix run.",
                {"recall": best.get("recall"), "precision": best.get("precision"), "missed_question_rate": best.get("missed_question_rate")},
            )
        elif experiment.get("status") == "passed":
            add_action(
                actions,
                70,
                "run_release_gate",
                "Experiment matrix passed; run the release readiness gate with replay and iOS artifact evidence.",
                {"selection_summary": experiment.get("selection_summary")},
            )

    for error in errors:
        if to_int(error.get("missed_boxes")) or to_int(error.get("false_positive_boxes")):
            add_action(
                actions,
                12,
                "open_error_workbench",
                "Detector errors have been converted to reviewable hard examples.",
                {"images": error.get("images"), "missed_boxes": error.get("missed_boxes"), "false_positive_boxes": error.get("false_positive_boxes")},
            )

    for release in releases:
        if not release.get("release_ready"):
            add_action(
                actions,
                5,
                "respect_release_blockers",
                "Do not bundle this detector into iOS until hard release failures are cleared.",
                {"hard_failure_categories": release.get("hard_failure_categories"), "hard_failure_count": release.get("hard_failure_count")},
            )
        else:
            add_action(actions, 90, "prepare_ios_release", "Release gate passed; bundle Core ML artifact, sync thresholds, and run Mac/Xcode validation.", {})

    for replay in replays:
        if to_float(replay.get("rect_only_vs_full"), math.inf) <= 1.005 and to_float(replay.get("crop_pixels_vs_full"), math.inf) <= 0.75:
            add_action(
                actions,
                80,
                "keep_rect_only_upload",
                "Replay still supports rect-only upload: low network overhead and lower VLM pixel load than full-frame VLM analysis.",
                {"rect_only_vs_full": replay.get("rect_only_vs_full"), "crop_pixels_vs_full": replay.get("crop_pixels_vs_full"), "layout_median_ms": replay.get("layout_median_ms")},
            )
        else:
            add_action(
                actions,
                25,
                "recheck_payload_strategy",
                "Replay payload or pixel savings did not meet target; inspect dedupe/crop sizing before release.",
                replay,
            )

    for strategy in strategies:
        best = strategy.get("best_reviewed_live_strategy") or {}
        if not best.get("strategy_id"):
            add_action(
                actions,
                24,
                "recheck_observation_strategy",
                "No reviewed live crop strategy is available in the strategy-selection report.",
                {"path": strategy.get("path")},
            )
            continue
        if (
            to_float(best.get("recall")) >= 0.999
            and to_float(best.get("total_pixels_vs_full"), math.inf) <= 0.90
            and to_float(best.get("fallback_image_ratio"), math.inf) <= 0.25
        ):
            add_action(
                actions,
                82,
                "keep_best_observation_strategy",
                "Strategy selection supports the current reviewed live observation crop strategy.",
                best,
            )

    for plan in training_plans:
        plan_actions = [item for item in (plan.get("actions") or []) if isinstance(item, dict)]
        expand_pool_actions = [item for item in plan_actions if item.get("code") == "expand_candidate_pool"]
        if expand_pool_actions:
            add_action(
                actions,
                21,
                "expand_candidate_pool",
                expand_pool_actions[0].get("detail")
                or "Current active-learning candidate pools are exhausted; mine broader historical image/SQLite pools.",
                {
                    "training_plan": plan.get("path"),
                    "source_action": expand_pool_actions[0],
                },
            )
        if to_int(plan.get("remaining_annotation_gap_after_pending")) > 0:
            add_action(
                actions,
                22,
                "follow_training_readiness_plan",
                "Training readiness plan shows the current review backlog is not enough for production model training.",
                {
                    "pending_boxes": plan.get("pending_boxes"),
                    "annotations_after_pending": plan.get("annotations_after_pending"),
                    "remaining_annotation_gap_after_pending": plan.get("remaining_annotation_gap_after_pending"),
                },
            )

    unique: dict[str, dict[str, Any]] = {}
    for action in sorted(actions, key=lambda row: row["priority"]):
        key = action["code"]
        if key not in unique:
            unique[key] = action
    return list(unique.values())


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Question Detector Iteration Report",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Stage: `{report['stage']}`",
        "",
        "## Next Actions",
        "",
    ]
    for action in report.get("next_actions") or []:
        lines.append(f"- P{action['priority']} `{action['code']}`: {action['detail']}")
    lines.extend(["", "## Dataset", ""])
    for dataset in report.get("datasets") or []:
        counts = dataset.get("counts") or {}
        gaps = dataset.get("gaps") or {}
        lines.append(
            f"- `{dataset['path']}` ready={dataset['model_training_ready']} annotations={counts.get('annotations')} "
            f"human={counts.get('human_reviewed_annotations')} pseudo={counts.get('pseudo_annotations')} "
            f"images={counts.get('source_images')} groups={counts.get('split_groups')} gaps={gaps}"
        )
    lines.extend(["", "## Experiments", ""])
    for experiment in report.get("experiments") or []:
        best = experiment.get("best_manifest") or {}
        lines.append(
            f"- `{experiment['path']}` status={experiment['status']} best={best.get('model')} img={best.get('imgsz')} "
            f"recall={best.get('recall')} precision={best.get('precision')} fp/img={best.get('fp_per_image')}"
        )
    lines.extend(["", "## Release Gates", ""])
    for release in report.get("release_checks") or []:
        lines.append(
            f"- `{release['path']}` ready={release['release_ready']} hard={release['hard_failure_count']} categories={release['hard_failure_categories']}"
        )
    lines.extend(["", "## Replay", ""])
    for replay in report.get("replays") or []:
        lines.append(
            f"- `{replay['path']}` rect_only={replay['rect_only_vs_full']} crop_pixels={replay['crop_pixels_vs_full']} "
            f"layout_median_ms={replay['layout_median_ms']}"
        )
    lines.extend(["", "## Strategy Selection", ""])
    for strategy in report.get("strategy_selections") or []:
        best = strategy.get("best_reviewed_live_strategy") or {}
        propagated = strategy.get("best_propagated_live_strategy") or {}
        lines.append(
            f"- `{strategy['path']}` rows={strategy['row_count']} best={best.get('strategy_id')} "
            f"recall={best.get('recall')} fallback={best.get('fallback_image_ratio')} pixels={best.get('total_pixels_vs_full')}"
        )
        if propagated:
            lines.append(
                f"  propagated: `{propagated.get('strategy_id')}` recall={propagated.get('recall')} "
                f"fallback={propagated.get('fallback_image_ratio')} pixels={propagated.get('total_pixels_vs_full')}"
            )
    lines.extend(["", "## Training Readiness", ""])
    for plan in report.get("training_plans") or []:
        lines.append(
            f"- `{plan['path']}` pending_boxes={plan['pending_boxes']} "
            f"annotations_after_pending={plan['annotations_after_pending']} "
            f"remaining_annotation_gap={plan['remaining_annotation_gap_after_pending']}"
        )
    lines.extend(["", "## Review Queues", ""])
    for queue in report.get("review_queues") or []:
        lines.append(
            f"- `{queue['path']}` workbenches={queue['workbench_count']} boxes={queue['total_boxes']} "
            f"pending={queue['pending_boxes']} approved={queue['approved_boxes']} html=`{queue['index_html']}`"
        )
    lines.extend(["", "## Error Mining", ""])
    for error in report.get("error_mining") or []:
        lines.append(
            f"- `{error['path']}` images={error['images']} missed={error['missed_boxes']} false_positive={error['false_positive_boxes']}"
        )
    lines.extend(["", "## Review Workbenches", ""])
    for workbench in report.get("review_workbenches") or []:
        lines.append(
            f"- `{workbench['path']}` boxes={workbench['boxes']} decisions={workbench['decision_file_count']} "
            f"approved_exports={workbench['approved_export_count']} approved_boxes={workbench['approved_box_count']} "
            f"pending_boxes={workbench['pending_box_count']}"
        )
        if workbench.get("export_command"):
            lines.append(f"  export: `{workbench['export_command']}`")
        if workbench.get("approved_inputs"):
            lines.append(f"  approved inputs: `{', '.join(workbench['approved_inputs'])}`")
    lines.append("")
    return "\n".join(lines)


def stage_for(report: dict[str, Any]) -> str:
    if any(item.get("release_ready") for item in report.get("release_checks") or []):
        return "release_candidate"
    if any(item.get("status") == "passed" for item in report.get("experiments") or []):
        return "model_selection_passed"
    if any(item.get("missed_boxes") or item.get("false_positive_boxes") for item in report.get("error_mining") or []):
        return "hard_example_review"
    if any(item.get("needs_review") or item.get("needs_export") for item in report.get("review_workbenches") or []):
        return "review_closure"
    if any(item.get("is_bootstrap") for item in report.get("datasets") or []):
        return "bootstrap_exploration"
    return "data_building"


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    datasets = [dataset_summary(path) for path in args.dataset]
    experiments = [experiment_summary(path) for path in args.experiment]
    releases = [release_summary(path) for path in args.release_check]
    replays = [replay_summary(path) for path in args.replay]
    strategies = [strategy_selection_summary(path) for path in args.strategy_selection]
    training_plans = [training_plan_summary(path) for path in args.training_plan]
    review_queues = [review_queue_summary(path) for path in args.review_queue]
    errors = [error_mining_summary(path) for path in args.error_mining]
    approved_statuses = {part.strip().lower() for part in args.approved_statuses.split(",") if part.strip()}
    workbenches = [review_workbench_summary(path, approved_statuses) for path in args.review_workbench]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "datasets": [str(path) for path in args.dataset],
            "experiments": [str(path) for path in args.experiment],
            "release_checks": [str(path) for path in args.release_check],
            "replays": [str(path) for path in args.replay],
            "strategy_selections": [str(path) for path in args.strategy_selection],
            "training_plans": [str(path) for path in args.training_plan],
            "review_queues": [str(path) for path in args.review_queue],
            "error_mining": [str(path) for path in args.error_mining],
            "review_workbenches": [str(path) for path in args.review_workbench],
        },
        "datasets": datasets,
        "experiments": experiments,
        "release_checks": releases,
        "replays": replays,
        "strategy_selections": strategies,
        "training_plans": training_plans,
        "review_queues": review_queues,
        "error_mining": errors,
        "review_workbenches": workbenches,
    }
    report["stage"] = stage_for(report)
    report["next_actions"] = recommend(report)
    write_json(args.out / "iteration_report.json", report)
    (args.out / "iteration_report.md").write_text(markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize one question-detector iteration.")
    parser.add_argument("--dataset", type=Path, action="append", default=[], help="Dataset root or audit.json.")
    parser.add_argument("--experiment", type=Path, action="append", default=[], help="Experiment root or experiment_report.json.")
    parser.add_argument("--release-check", type=Path, action="append", default=[], help="Release-check root or release_check.json.")
    parser.add_argument("--replay", type=Path, action="append", default=[], help="Replay root or summary.json.")
    parser.add_argument("--strategy-selection", type=Path, action="append", default=[], help="Strategy selector root or strategy_selection.json.")
    parser.add_argument("--training-plan", type=Path, action="append", default=[], help="Training readiness plan root or training_readiness_plan.json.")
    parser.add_argument("--review-queue", type=Path, action="append", default=[], help="Review queue manifest root or review_queue_manifest.json.")
    parser.add_argument("--error-mining", type=Path, action="append", default=[], help="Error-mining root or summary.json.")
    parser.add_argument("--review-workbench", type=Path, action="append", default=[], help="Review workbench root produced by question_detector_review_workbench.py.")
    parser.add_argument("--approved-statuses", default="approved,accepted,corrected,verified")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-iteration-report"))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    report = build_report(args)
    print(
        json.dumps(
            {
                "stage": report["stage"],
                "next_actions": [item["code"] for item in report["next_actions"][:6]],
                "out": str(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
