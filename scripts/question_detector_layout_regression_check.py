"""Check that dense layout changes pass both real and synthetic evidence.

Synthetic dense fixtures catch dense-page misses. Real reviewed smoke replays
catch over-triggered dense splitting on historical captures. This gate keeps
both in the loop so a synthetic-only improvement cannot silently regress real
question crops.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_file(path: Path, default_name: str) -> Path:
    return path / default_name if path.is_dir() else path


def metric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return default


def candidate_eval_summary(path: Path) -> dict[str, Any]:
    payload = read_json(resolve_file(path, "summary.json"))
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    return {
        "path": str(resolve_file(path, "summary.json")),
        "candidate_count": int(counts.get("candidate_count") or 0),
        "represented_gt_key_count": int(counts.get("represented_gt_key_count") or 0),
        "covered_gt_key_count": int(counts.get("covered_gt_key_count") or 0),
        "missed_gt_key_count": int(counts.get("missed_gt_key_count") or 0),
        "unmatched_reviewed_candidate_count": int(counts.get("unmatched_reviewed_candidate_count") or 0),
        "unverified_candidate_count": int(counts.get("unverified_candidate_count") or 0),
        "overcrop_match_count": int(counts.get("overcrop_match_count") or 0),
        "represented_gt_recall": metrics.get("represented_gt_recall"),
        "crop_pixels_vs_full_frame_pixels": metrics.get("crop_pixels_vs_full_frame_pixels"),
        "overcrop_ratio_p90": metrics.get("overcrop_ratio_p90"),
        "overcrop_ratio_max": metrics.get("overcrop_ratio_max"),
    }


def replay_summary(path: Path) -> dict[str, Any]:
    payload = read_json(resolve_file(path, "summary.json"))
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    vlm = payload.get("vlm_proxy") if isinstance(payload.get("vlm_proxy"), dict) else {}
    quality = payload.get("quality_proxy") if isinstance(payload.get("quality_proxy"), dict) else {}
    layout = payload.get("layout_measurement") if isinstance(payload.get("layout_measurement"), dict) else {}
    latency = layout.get("latency_ms") if isinstance(layout.get("latency_ms"), dict) else {}
    return {
        "path": str(resolve_file(path, "summary.json")),
        "images": int(counts.get("images") or 0),
        "raw_boxes": int(counts.get("raw_boxes") or 0),
        "cross_frame_dedup_boxes": int(counts.get("cross_frame_dedup_boxes") or 0),
        "images_with_boxes": int(counts.get("images_with_boxes") or 0),
        "crop_pixels_vs_full_pixels": vlm.get("crop_pixels_vs_full_pixels"),
        "small_area_boxes": int(quality.get("small_area_boxes") or 0),
        "near_full_page_boxes": int(quality.get("near_full_page_boxes") or 0),
        "flag_counts": quality.get("flag_counts") if isinstance(quality.get("flag_counts"), dict) else {},
        "layout_latency_ms": latency,
    }


def stress_summary(path: Path) -> dict[str, Any]:
    payload = read_json(resolve_file(path, "stress_check.json"))
    return {
        "path": str(resolve_file(path, "stress_check.json")),
        "stress_ready": bool(payload.get("stress_ready")),
        "transport_ready": bool(payload.get("transport_ready")),
        "detector_ready": bool(payload.get("detector_ready")),
        "hard_failure_count": len(payload.get("hard_failures") or []),
        "warning_count": len(payload.get("warnings") or []),
    }


def fallback_tune_summary(path: Path) -> dict[str, Any]:
    payload = read_json(resolve_file(path, "summary.json"))
    baseline = payload.get("baseline_backend_current") if isinstance(payload.get("baseline_backend_current"), dict) else {}
    legacy = payload.get("baseline_backend_legacy_before_tuned") if isinstance(payload.get("baseline_backend_legacy_before_tuned"), dict) else {}
    delta = payload.get("fallback_delta_legacy_to_current") if isinstance(payload.get("fallback_delta_legacy_to_current"), dict) else {}
    review_queue = payload.get("policy_delta_review_queue") if isinstance(payload.get("policy_delta_review_queue"), dict) else {}
    return {
        "path": str(resolve_file(path, "summary.json")),
        "image_count": int(payload.get("image_count") or 0),
        "represented_gt_key_count": int(payload.get("represented_gt_key_count") or 0),
        "crop_only_gt_recall": payload.get("crop_only_gt_recall"),
        "policy_id": baseline.get("policy_id"),
        "policy_gt_recall": baseline.get("policy_gt_recall"),
        "fallback_image_count": baseline.get("fallback_image_count"),
        "fallback_image_ratio": baseline.get("fallback_image_ratio"),
        "total_pixels_vs_full": baseline.get("total_pixels_vs_full"),
        "fallback_reason_counts": baseline.get("fallback_reason_counts") if isinstance(baseline.get("fallback_reason_counts"), dict) else {},
        "legacy_fallback_image_count": legacy.get("fallback_image_count"),
        "legacy_total_pixels_vs_full": legacy.get("total_pixels_vs_full"),
        "delta_fallback_removed_image_count": delta.get("fallback_removed_image_count"),
        "delta_fallback_added_image_count": delta.get("fallback_added_image_count"),
        "delta_fallback_removed_with_gt_count": delta.get("fallback_removed_with_gt_count"),
        "delta_fallback_removed_with_crop_missed_gt_count": delta.get("fallback_removed_with_crop_missed_gt_count"),
        "delta_review_queue_images": review_queue.get("images"),
        "delta_review_queue_boxes": review_queue.get("boxes"),
    }


def add_check(
    checks: list[dict[str, Any]],
    category: str,
    name: str,
    passed: bool,
    detail: str,
    values: dict[str, Any],
) -> None:
    checks.append(
        {
            "category": category,
            "name": name,
            "passed": bool(passed),
            "detail": detail,
            "values": values,
        }
    )


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Layout Regression Check",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Passed: `{str(report['passed']).lower()}`",
        f"- Failures: {len(report['failures'])}",
        "",
        "## Key Metrics",
        "",
        f"- real recall: `{report['real_candidate_eval'].get('represented_gt_recall')}`",
        f"- real candidates: `{report['real_candidate_eval'].get('candidate_count')}`",
        f"- real raw boxes: `{report['real_replay'].get('raw_boxes')}`",
        f"- real crop pixels vs full: `{report['real_replay'].get('crop_pixels_vs_full_pixels')}`",
        f"- synthetic recall: `{report['synthetic_candidate_eval'].get('represented_gt_recall')}`",
        f"- synthetic candidates: `{report['synthetic_candidate_eval'].get('candidate_count')}`",
        f"- synthetic stress ready: `{str(report['synthetic_stress'].get('stress_ready')).lower()}`",
        f"- KPAI fallback pixels vs full: `{report['real_fallback_tune'].get('total_pixels_vs_full')}`",
        f"- KPAI removed-fallback known GT misses: `{report['real_fallback_tune'].get('delta_fallback_removed_with_crop_missed_gt_count')}`",
        f"- KPAI removed-fallback review queue: `{report['real_fallback_tune'].get('delta_review_queue_images')}` images",
        f"- propagated fallback pixels vs full: `{report['propagated_fallback_tune'].get('total_pixels_vs_full')}`",
        f"- propagated removed-fallback known GT misses: `{report['propagated_fallback_tune'].get('delta_fallback_removed_with_crop_missed_gt_count')}`",
        f"- propagated removed-fallback review queue: `{report['propagated_fallback_tune'].get('delta_review_queue_images')}` images",
        "",
    ]
    if report["failures"]:
        lines.extend(["## Failures", ""])
        for item in report["failures"]:
            lines.append(f"- [{item['category']}] {item['name']}: {item['detail']}")
        lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)

    real_eval = candidate_eval_summary(args.real_candidate_eval)
    real_replay = replay_summary(args.real_replay)
    baseline_replay = replay_summary(args.real_baseline_replay) if args.real_baseline_replay else {}
    synthetic_eval = candidate_eval_summary(args.synthetic_candidate_eval)
    synthetic_stress = stress_summary(args.synthetic_stress)
    real_fallback = fallback_tune_summary(args.real_fallback_tune)
    propagated_fallback = fallback_tune_summary(args.propagated_fallback_tune)

    checks: list[dict[str, Any]] = []
    real_box_limit = args.max_real_boxes
    baseline_raw_boxes = int(baseline_replay.get("raw_boxes") or 0)
    if baseline_raw_boxes > 0:
        real_box_limit = max(real_box_limit, int(round(baseline_raw_boxes * args.max_real_box_multiplier)))

    add_check(
        checks,
        "real",
        "real reviewed recall is preserved",
        metric(real_eval, "represented_gt_recall") >= args.min_real_recall,
        f"Real reviewed replay recall must stay >= {args.min_real_recall}.",
        real_eval,
    )
    add_check(
        checks,
        "real",
        "real candidate count is not inflated",
        int(real_eval.get("candidate_count") or 0) <= real_box_limit,
        f"Real replay candidate count must stay <= {real_box_limit}; dense over-triggering caused a 403-box regression before gating.",
        {"candidate_count": real_eval.get("candidate_count"), "limit": real_box_limit, "baseline_raw_boxes": baseline_raw_boxes},
    )
    add_check(
        checks,
        "real",
        "real crop pixel cost is bounded",
        metric(real_replay, "crop_pixels_vs_full_pixels") <= args.max_real_crop_pixels_vs_full,
        f"Real replay crop pixels should stay <= {args.max_real_crop_pixels_vs_full}x full-frame pixels.",
        real_replay,
    )
    add_check(
        checks,
        "real",
        "real overcrop hard cases are absent",
        int(real_eval.get("overcrop_match_count") or 0) <= args.max_real_overcrop_matches,
        f"Real reviewed replay should have <= {args.max_real_overcrop_matches} overcrop hard cases.",
        real_eval,
    )
    add_check(
        checks,
        "synthetic",
        "synthetic dense recall is preserved",
        metric(synthetic_eval, "represented_gt_recall") >= args.min_synthetic_recall,
        f"Synthetic dense recall must stay >= {args.min_synthetic_recall}.",
        synthetic_eval,
    )
    add_check(
        checks,
        "synthetic",
        "synthetic dense stress gate passes",
        bool(synthetic_stress.get("stress_ready")),
        "Synthetic dense stress check must be ready; warnings are allowed when section sender evidence makes cap-only diagnostics non-blocking.",
        synthetic_stress,
    )
    add_check(
        checks,
        "fallback",
        "KPAI backend fallback recall is preserved",
        metric(real_fallback, "policy_gt_recall") >= args.min_fallback_recall,
        f"KPAI backend fallback policy recall must stay >= {args.min_fallback_recall}.",
        real_fallback,
    )
    add_check(
        checks,
        "fallback",
        "KPAI backend fallback cost is bounded",
        metric(real_fallback, "total_pixels_vs_full", 999.0) <= args.max_real_fallback_total_pixels_vs_full,
        f"KPAI backend fallback total pixels must stay <= {args.max_real_fallback_total_pixels_vs_full}x full-frame pixels.",
        real_fallback,
    )
    add_check(
        checks,
        "fallback",
        "KPAI fallback delta has no known GT miss",
        int(real_fallback.get("delta_fallback_removed_with_crop_missed_gt_count") or 0) == 0,
        "Images that no longer receive legacy full-frame fallback must not include represented GT missed by crops.",
        real_fallback,
    )
    add_check(
        checks,
        "fallback",
        "propagated backend fallback recall is preserved",
        metric(propagated_fallback, "policy_gt_recall") >= args.min_fallback_recall,
        f"Propagated backend fallback policy recall must stay >= {args.min_fallback_recall}.",
        propagated_fallback,
    )
    add_check(
        checks,
        "fallback",
        "propagated backend fallback cost is bounded",
        metric(propagated_fallback, "total_pixels_vs_full", 999.0) <= args.max_propagated_fallback_total_pixels_vs_full,
        f"Propagated backend fallback total pixels must stay <= {args.max_propagated_fallback_total_pixels_vs_full}x full-frame pixels.",
        propagated_fallback,
    )
    add_check(
        checks,
        "fallback",
        "propagated fallback delta has no known GT miss",
        int(propagated_fallback.get("delta_fallback_removed_with_crop_missed_gt_count") or 0) == 0,
        "Images that no longer receive legacy full-frame fallback must not include represented GT missed by crops.",
        propagated_fallback,
    )

    failures = [check for check in checks if not check["passed"]]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "thresholds": {
            "min_real_recall": args.min_real_recall,
            "max_real_boxes": args.max_real_boxes,
            "max_real_box_multiplier": args.max_real_box_multiplier,
            "max_real_crop_pixels_vs_full": args.max_real_crop_pixels_vs_full,
            "max_real_overcrop_matches": args.max_real_overcrop_matches,
            "min_synthetic_recall": args.min_synthetic_recall,
            "min_fallback_recall": args.min_fallback_recall,
            "max_real_fallback_total_pixels_vs_full": args.max_real_fallback_total_pixels_vs_full,
            "max_propagated_fallback_total_pixels_vs_full": args.max_propagated_fallback_total_pixels_vs_full,
        },
        "real_candidate_eval": real_eval,
        "real_replay": real_replay,
        "real_baseline_replay": baseline_replay,
        "synthetic_candidate_eval": synthetic_eval,
        "synthetic_stress": synthetic_stress,
        "real_fallback_tune": real_fallback,
        "propagated_fallback_tune": propagated_fallback,
        "checks": checks,
        "failures": failures,
    }
    write_json(args.out / "layout_regression_check.json", report)
    (args.out / "layout_regression_check.md").write_text(markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check real + synthetic layout regression evidence.")
    parser.add_argument("--real-replay", type=Path, default=Path("diagnostics/question-observation-replay-kpai-layout-measured-v5-dense-gated"))
    parser.add_argument("--real-baseline-replay", type=Path, default=Path("diagnostics/question-observation-replay-kpai-layout-measured"))
    parser.add_argument("--real-candidate-eval", type=Path, default=Path("diagnostics/question-observation-candidate-eval-kpai-layout-v5-dense-gated"))
    parser.add_argument("--synthetic-candidate-eval", type=Path, default=Path("diagnostics/question-observation-candidate-eval-synthetic-dense-layout-dense-prelabel-v5-gated"))
    parser.add_argument("--synthetic-stress", type=Path, default=Path("diagnostics/question-detector-synthetic-dense-stress-check-dense-prelabel-v5-gated-backend-tuned"))
    parser.add_argument("--real-fallback-tune", type=Path, default=Path("diagnostics/question-observation-fallback-tune-kpai-layout-v5-dense-gated-backend-tuned"))
    parser.add_argument("--propagated-fallback-tune", type=Path, default=Path("diagnostics/question-observation-fallback-tune-reviewed-propagated-layout-v5-gated-backend-tuned"))
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-layout-regression-check-v5-gated-backend-tuned"))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--min-real-recall", type=float, default=0.999)
    parser.add_argument("--max-real-boxes", type=int, default=80)
    parser.add_argument("--max-real-box-multiplier", type=float, default=1.20)
    parser.add_argument("--max-real-crop-pixels-vs-full", type=float, default=0.70)
    parser.add_argument("--max-real-overcrop-matches", type=int, default=0)
    parser.add_argument("--min-synthetic-recall", type=float, default=0.999)
    parser.add_argument("--min-fallback-recall", type=float, default=0.999)
    parser.add_argument("--max-real-fallback-total-pixels-vs-full", type=float, default=0.85)
    parser.add_argument("--max-propagated-fallback-total-pixels-vs-full", type=float, default=0.90)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "failures": len(report["failures"]),
                "out": str(args.out),
            },
            ensure_ascii=False,
        )
    )
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
