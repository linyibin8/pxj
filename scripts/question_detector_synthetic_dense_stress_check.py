"""Summarize synthetic dense observation stress evidence.

This is a negative gate only: passing synthetic stress does not prove release
readiness, but failing it exposes dense-page/candidate-cap/fallback risks that
should block promotion until understood.
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


def add_check(
    checks: list[dict[str, Any]],
    category: str,
    name: str,
    passed: bool,
    detail: str,
    severity: str = "hard",
    values: dict[str, Any] | None = None,
) -> None:
    checks.append(
        {
            "category": category,
            "name": name,
            "passed": bool(passed),
            "severity": severity,
            "detail": detail,
            "values": values or {},
        }
    )


def candidate_eval_summary(path: Path) -> dict[str, Any]:
    payload = read_json(resolve_file(path, "summary.json"))
    counts = payload.get("counts") if isinstance(payload.get("counts"), dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    return {
        "path": str(resolve_file(path, "summary.json")),
        "images": int(counts.get("images") or 0),
        "images_with_candidates": int(counts.get("images_with_candidates") or 0),
        "candidate_count": int(counts.get("candidate_count") or 0),
        "represented_gt_key_count": int(counts.get("represented_gt_key_count") or 0),
        "covered_gt_key_count": int(counts.get("covered_gt_key_count") or 0),
        "missed_gt_key_count": int(counts.get("missed_gt_key_count") or 0),
        "represented_gt_recall": metrics.get("represented_gt_recall"),
        "crop_pixels_vs_full_frame_pixels": metrics.get("crop_pixels_vs_full_frame_pixels"),
    }


def cap_sweep_summary(path: Path) -> dict[str, Any]:
    payload = read_json(resolve_file(path, "summary.json"))
    current = payload.get("current_cap") if isinstance(payload.get("current_cap"), dict) else {}
    recommendation = payload.get("recommendation") if isinstance(payload.get("recommendation"), dict) else {}
    return {
        "path": str(resolve_file(path, "summary.json")),
        "image_count": int(payload.get("image_count") or 0),
        "candidate_count": int(payload.get("candidate_count") or 0),
        "represented_gt_key_count": int(payload.get("represented_gt_key_count") or 0),
        "current_cap": current,
        "recommendation": recommendation,
    }


def fallback_summary(path: Path) -> dict[str, Any]:
    payload = read_json(resolve_file(path, "summary.json"))
    current = payload.get("baseline_backend_current") if isinstance(payload.get("baseline_backend_current"), dict) else {}
    union = payload.get("baseline_backend_current_union_area") if isinstance(payload.get("baseline_backend_current_union_area"), dict) else {}
    return {
        "path": str(resolve_file(path, "summary.json")),
        "image_count": int(payload.get("image_count") or 0),
        "candidate_count": int(payload.get("candidate_count") or 0),
        "represented_gt_key_count": int(payload.get("represented_gt_key_count") or 0),
        "crop_only_gt_recall": payload.get("crop_only_gt_recall"),
        "backend_current": current,
        "backend_current_union_area": union,
    }


def section_summary(path: Path) -> dict[str, Any]:
    payload = read_json(resolve_file(path, "summary.json"))
    recommendation = payload.get("recommendation") if isinstance(payload.get("recommendation"), dict) else {}
    return {
        "path": str(resolve_file(path, "summary.json")),
        "image_count": int(payload.get("image_count") or 0),
        "candidate_count": int(payload.get("candidate_count") or 0),
        "represented_gt_key_count": int(payload.get("represented_gt_key_count") or 0),
        "recommendation": recommendation,
    }


def section_sender_summary(path: Path) -> dict[str, Any]:
    payload = read_json(resolve_file(path, "summary.json"))
    sender = payload.get("sender") if isinstance(payload.get("sender"), dict) else {}
    return {
        "path": str(resolve_file(path, "summary.json")),
        "image_count": int(sender.get("image_count") or payload.get("image_count") or 0),
        "candidate_count": int(sender.get("candidate_count") or payload.get("candidate_count") or 0),
        "represented_gt_keys": int(sender.get("represented_gt_key_count") or payload.get("represented_gt_key_count") or 0),
        "combined_crop_gt_recall": sender.get("combined_crop_gt_recall"),
        "policy_gt_recall": sender.get("policy_gt_recall"),
        "total_pixels_vs_full": sender.get("total_pixels_vs_full"),
        "fallback_image_ratio": sender.get("fallback_image_ratio"),
        "sender": sender,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Synthetic Dense Stress Check",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Stress ready: `{str(report['stress_ready']).lower()}`",
        f"- Transport ready: `{str(report.get('transport_ready')).lower()}`",
        f"- Detector ready: `{str(report.get('detector_ready')).lower()}`",
        f"- Hard failures: {len(report['hard_failures'])}",
        f"- Warnings: {len(report['warnings'])}",
        "",
    ]
    if report["hard_failures"]:
        lines.extend(["## Hard Failures", ""])
        for item in report["hard_failures"]:
            lines.append(f"- [{item['category']}] {item['name']}: {item['detail']}")
        lines.append("")
    if report["warnings"]:
        lines.extend(["## Warnings", ""])
        for item in report["warnings"]:
            lines.append(f"- [{item['category']}] {item['name']}: {item['detail']}")
        lines.append("")
    lines.extend(["## Key Metrics", ""])
    oracle_cap = report.get("oracle_cap_sweep", {}).get("current_cap", {})
    section = report.get("section_crop_sweep", {}).get("recommendation", {})
    sender = report.get("section_sender_eval", {})
    empty_sender = report.get("empty_section_sender_eval", {})
    layout_eval = report.get("layout_candidate_eval", {})
    empty_layout_fallback = report.get("empty_layout_fallback", {}).get("backend_current", {})
    lines.extend(
        [
            f"- oracle cap-12 crop recall: `{oracle_cap.get('selected_gt_recall')}`",
            f"- oracle cap-12 fallback-protected recall: `{oracle_cap.get('policy_gt_recall')}`",
            f"- oracle cap-12 total pixels vs full: `{oracle_cap.get('total_pixels_vs_full')}`",
            f"- section recommendation: `{section.get('strategy_id')}`",
            f"- section total pixels vs full: `{section.get('total_pixels_vs_full')}`",
            f"- section sender represented GT keys: `{sender.get('represented_gt_keys')}`",
            f"- section sender combined crop recall: `{sender.get('combined_crop_gt_recall')}`",
            f"- section sender policy recall: `{sender.get('policy_gt_recall')}`",
            f"- section sender total pixels vs full: `{sender.get('total_pixels_vs_full')}`",
            f"- section sender fallback image ratio: `{sender.get('fallback_image_ratio')}`",
            f"- empty-frame section represented GT keys: `{empty_sender.get('represented_gt_keys')}`",
            f"- empty-frame section combined crop recall: `{empty_sender.get('combined_crop_gt_recall')}`",
            f"- empty-frame section total pixels vs full: `{empty_sender.get('total_pixels_vs_full')}`",
            f"- empty-frame section fallback image ratio: `{empty_sender.get('fallback_image_ratio')}`",
            f"- layout crop recall: `{layout_eval.get('represented_gt_recall')}`",
            f"- empty-layout fallback images: `{empty_layout_fallback.get('fallback_image_count')}`",
            f"- empty-layout fallback total pixels vs full: `{empty_layout_fallback.get('total_pixels_vs_full')}`",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    checks: list[dict[str, Any]] = []

    audit = read_json(resolve_file(args.fixture, "audit.json"))
    readiness = audit.get("readiness") if isinstance(audit.get("readiness"), dict) else {}
    counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
    add_check(
        checks,
        "fixture",
        "synthetic fixture is diagnostic only",
        readiness.get("model_training_ready") is False,
        "Synthetic dense fixture must not be usable as model_training_ready release data.",
        values={"readiness": readiness},
    )
    add_check(
        checks,
        "fixture",
        "synthetic fixture has dense GT volume",
        int(counts.get("annotations") or 0) >= args.min_gt_keys,
        f"Synthetic dense stress should cover at least {args.min_gt_keys} expected question boxes.",
        values={"annotations": counts.get("annotations")},
    )

    oracle_eval = candidate_eval_summary(args.oracle_candidate_eval)
    layout_eval = candidate_eval_summary(args.layout_candidate_eval)
    oracle_cap = cap_sweep_summary(args.oracle_cap_sweep)
    layout_cap = cap_sweep_summary(args.layout_cap_sweep)
    empty_layout_cap = cap_sweep_summary(args.empty_layout_cap_sweep)
    oracle_fallback = fallback_summary(args.oracle_fallback)
    layout_fallback = fallback_summary(args.layout_fallback)
    empty_layout_fallback = fallback_summary(args.empty_layout_fallback)
    section = section_summary(args.section_sweep)
    section_sender = section_sender_summary(args.section_sender_eval)
    empty_section_sender = section_sender_summary(args.empty_section_sender_eval)
    min_section_sender_gt_keys = (
        args.min_section_sender_gt_keys if args.min_section_sender_gt_keys is not None else args.min_gt_keys
    )
    min_empty_section_sender_gt_keys = (
        args.min_empty_section_sender_gt_keys if args.min_empty_section_sender_gt_keys is not None else args.min_gt_keys
    )
    empty_section_sender_passed = (
        int(empty_section_sender.get("represented_gt_keys") or 0) >= min_empty_section_sender_gt_keys
        and metric(empty_section_sender, "combined_crop_gt_recall") >= args.min_section_sender_combined_crop_recall
        and metric(empty_section_sender, "policy_gt_recall") >= args.min_section_sender_policy_recall
        and metric(empty_section_sender, "total_pixels_vs_full", 999.0) <= args.max_section_sender_total_pixels_vs_full
        and metric(empty_section_sender, "fallback_image_ratio", 999.0) <= args.max_section_sender_fallback_image_ratio
    )
    layout_candidates_passed = metric(layout_eval, "represented_gt_recall") >= args.min_layout_recall

    add_check(
        checks,
        "oracle",
        "oracle candidates cover dense GT",
        metric(oracle_eval, "represented_gt_recall") >= args.min_oracle_recall,
        f"Oracle fixture candidates must cover dense expected boxes at recall >= {args.min_oracle_recall}.",
        values=oracle_eval,
    )
    add_check(
        checks,
        "layout",
        "current layout candidates cover dense GT",
        layout_candidates_passed,
        f"Current local layout/detector path must cover dense expected boxes at recall >= {args.min_layout_recall}. "
        "Empty-frame section sender may protect transport cost, but it does not prove local question-box precision.",
        values=layout_eval,
    )
    current_cap = oracle_cap.get("current_cap") if isinstance(oracle_cap.get("current_cap"), dict) else {}
    section_sender_passed = (
        int(section_sender.get("represented_gt_keys") or 0) >= min_section_sender_gt_keys
        and metric(section_sender, "combined_crop_gt_recall") >= args.min_section_sender_combined_crop_recall
        and metric(section_sender, "policy_gt_recall") >= args.min_section_sender_policy_recall
        and metric(section_sender, "total_pixels_vs_full", 999.0) <= args.max_section_sender_total_pixels_vs_full
        and metric(section_sender, "fallback_image_ratio", 999.0) <= args.max_section_sender_fallback_image_ratio
    )
    add_check(
        checks,
        "section_sender",
        "section sender covers dense cap overflow",
        section_sender_passed,
        "Section sender eval must prove dense cap overflow is represented by uploaded section crops without whole-frame fallback pressure.",
        values={
            **section_sender,
            "thresholds": {
                "min_represented_gt_keys": min_section_sender_gt_keys,
                "min_combined_crop_gt_recall": args.min_section_sender_combined_crop_recall,
                "min_policy_gt_recall": args.min_section_sender_policy_recall,
                "max_total_pixels_vs_full": args.max_section_sender_total_pixels_vs_full,
                "max_fallback_image_ratio": args.max_section_sender_fallback_image_ratio,
            },
        },
    )
    add_check(
        checks,
        "section_sender",
        "empty-frame section sender covers dense zero-candidate layout",
        empty_section_sender_passed,
        "Empty-frame section sender eval must prove zero-candidate dense pages are represented by uploaded section crops without whole-frame fallback pressure.",
        values={
            **empty_section_sender,
            "thresholds": {
                "min_represented_gt_keys": min_empty_section_sender_gt_keys,
                "min_combined_crop_gt_recall": args.min_section_sender_combined_crop_recall,
                "min_policy_gt_recall": args.min_section_sender_policy_recall,
                "max_total_pixels_vs_full": args.max_section_sender_total_pixels_vs_full,
                "max_fallback_image_ratio": args.max_section_sender_fallback_image_ratio,
            },
        },
    )
    cap_only_severity = "warning" if section_sender_passed else "hard"
    cap_only_detail_suffix = (
        " Section sender evidence passes, so this cap-only diagnostic is no longer release-blocking."
        if section_sender_passed
        else ""
    )
    add_check(
        checks,
        "candidate_cap",
        "current cap preserves dense crop-only recall",
        metric(current_cap, "selected_gt_recall") >= args.min_current_cap_crop_recall,
        f"Production cap must not drop dense-page crop-only recall below {args.min_current_cap_crop_recall}."
        f"{cap_only_detail_suffix}",
        severity=cap_only_severity,
        values=current_cap,
    )
    section_rec = section.get("recommendation") if isinstance(section.get("recommendation"), dict) else {}
    add_check(
        checks,
        "section_crop",
        "section crop alternative covers dense GT under full-frame cost",
        bool(section_rec)
        and metric(section_rec, "section_gt_recall") >= args.min_oracle_recall
        and metric(section_rec, "policy_gt_recall") >= args.min_policy_recall
        and metric(section_rec, "total_pixels_vs_full", 999.0) <= args.max_section_total_pixels_vs_full,
        "Dense section/group crop sweep should find at least one diagnostic alternative with full crop recall and total VLM pixels below full-frame.",
        values=section_rec,
    )
    add_check(
        checks,
        "candidate_cap",
        "current cap dense fallback-protected recall",
        metric(current_cap, "policy_gt_recall") >= args.min_policy_recall,
        f"Backend fallback must preserve dense-page recall >= {args.min_policy_recall}."
        f"{cap_only_detail_suffix}",
        severity=cap_only_severity,
        values=current_cap,
    )
    add_check(
        checks,
        "candidate_cap",
        "current cap dense VLM pixel cost",
        metric(current_cap, "total_pixels_vs_full", 999.0) <= args.max_current_cap_total_pixels_vs_full,
        f"Dense crop+fallback VLM pixels should stay <= {args.max_current_cap_total_pixels_vs_full}x full-frame."
        f"{cap_only_detail_suffix}",
        severity=cap_only_severity,
        values=current_cap,
    )
    empty_layout_current = empty_layout_cap.get("current_cap") if isinstance(empty_layout_cap.get("current_cap"), dict) else {}
    add_check(
        checks,
        "fallback",
        "layout zero-candidate fallback is explicit",
        metric(empty_layout_current, "policy_gt_recall") >= args.min_policy_recall
        and int(empty_layout_current.get("fallback_image_count") or 0) == int(empty_layout_cap.get("image_count") or -1),
        "When local layout emits no dense crops, the stress report must show full-frame fallback protecting recall and cost.",
        values=empty_layout_current,
    )
    empty_layout_backend = (
        empty_layout_fallback.get("backend_current") if isinstance(empty_layout_fallback.get("backend_current"), dict) else {}
    )
    add_check(
        checks,
        "fallback",
        "layout fallback cost is measured",
        metric(empty_layout_backend, "total_pixels_vs_full", 999.0) <= args.max_layout_fallback_pixels_vs_full,
        f"Whole-frame fallback cost for zero-candidate dense layout should stay <= {args.max_layout_fallback_pixels_vs_full}x full-frame.",
        values=empty_layout_backend,
    )

    hard_failures = [check for check in checks if check["severity"] == "hard" and not check["passed"]]
    warnings = [check for check in checks if check["severity"] == "warning" and not check["passed"]]
    transport_blocking_categories = {"fixture", "oracle", "section_sender", "section_crop", "candidate_cap", "fallback"}
    transport_failures = [
        check
        for check in hard_failures
        if check.get("category") in transport_blocking_categories
    ]
    detector_failures = [
        check
        for check in hard_failures
        if check.get("category") == "layout"
    ]
    transport_ready = not transport_failures
    detector_ready = not detector_failures
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stress_ready": transport_ready and detector_ready,
        "transport_ready": transport_ready,
        "detector_ready": detector_ready,
        "thresholds": {
            "min_gt_keys": args.min_gt_keys,
            "min_oracle_recall": args.min_oracle_recall,
            "min_layout_recall": args.min_layout_recall,
            "min_current_cap_crop_recall": args.min_current_cap_crop_recall,
            "min_policy_recall": args.min_policy_recall,
            "min_section_sender_gt_keys": min_section_sender_gt_keys,
            "min_empty_section_sender_gt_keys": min_empty_section_sender_gt_keys,
            "min_section_sender_combined_crop_recall": args.min_section_sender_combined_crop_recall,
            "min_section_sender_policy_recall": args.min_section_sender_policy_recall,
            "max_section_sender_total_pixels_vs_full": args.max_section_sender_total_pixels_vs_full,
            "max_section_sender_fallback_image_ratio": args.max_section_sender_fallback_image_ratio,
            "max_current_cap_total_pixels_vs_full": args.max_current_cap_total_pixels_vs_full,
            "max_layout_fallback_pixels_vs_full": args.max_layout_fallback_pixels_vs_full,
            "max_section_total_pixels_vs_full": args.max_section_total_pixels_vs_full,
        },
        "fixture_audit": {"path": str(resolve_file(args.fixture, "audit.json")), "readiness": readiness, "counts": counts},
        "oracle_candidate_eval": oracle_eval,
        "layout_candidate_eval": layout_eval,
        "oracle_cap_sweep": oracle_cap,
        "layout_cap_sweep": layout_cap,
        "empty_layout_cap_sweep": empty_layout_cap,
        "section_crop_sweep": section,
        "section_sender_eval": section_sender,
        "empty_section_sender_eval": empty_section_sender,
        "oracle_fallback": oracle_fallback,
        "layout_fallback": layout_fallback,
        "empty_layout_fallback": empty_layout_fallback,
        "checks": checks,
        "hard_failures": hard_failures,
        "transport_failures": transport_failures,
        "detector_failures": detector_failures,
        "warnings": warnings,
    }
    write_json(args.out / "stress_check.json", report)
    (args.out / "stress_check.md").write_text(markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check synthetic dense observation stress reports.")
    parser.add_argument("--fixture", type=Path, default=Path("diagnostics/question-detector-synthetic-dense"))
    parser.add_argument("--oracle-candidate-eval", type=Path, default=Path("diagnostics/question-observation-candidate-eval-synthetic-dense-oracle"))
    parser.add_argument("--layout-candidate-eval", type=Path, default=Path("diagnostics/question-observation-candidate-eval-synthetic-dense-layout"))
    parser.add_argument("--oracle-cap-sweep", type=Path, default=Path("diagnostics/question-observation-candidate-cap-sweep-synthetic-dense-oracle"))
    parser.add_argument("--layout-cap-sweep", type=Path, default=Path("diagnostics/question-observation-candidate-cap-sweep-synthetic-dense-layout"))
    parser.add_argument(
        "--empty-layout-cap-sweep",
        type=Path,
        default=Path("diagnostics/question-observation-candidate-cap-sweep-synthetic-dense-layout"),
        help="Zero-candidate layout cap/fallback evidence; kept separate from current layout candidates.",
    )
    parser.add_argument("--oracle-fallback", type=Path, default=Path("diagnostics/question-observation-fallback-tune-synthetic-dense-oracle"))
    parser.add_argument("--layout-fallback", type=Path, default=Path("diagnostics/question-observation-fallback-tune-synthetic-dense-layout"))
    parser.add_argument(
        "--empty-layout-fallback",
        type=Path,
        default=Path("diagnostics/question-observation-fallback-tune-synthetic-dense-layout"),
        help="Zero-candidate layout fallback evidence; kept separate from current layout fallback tuning.",
    )
    parser.add_argument("--section-sweep", type=Path, default=Path("diagnostics/question-observation-section-crop-sweep-synthetic-dense-oracle"))
    parser.add_argument("--section-sender-eval", type=Path, default=Path("diagnostics/question-observation-section-sender-eval-synthetic-dense-oracle"))
    parser.add_argument("--empty-section-sender-eval", type=Path, default=Path("diagnostics/question-observation-section-sender-eval-synthetic-dense-layout-empty-section"))
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-synthetic-dense-stress-check"))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--min-gt-keys", type=int, default=200)
    parser.add_argument("--min-oracle-recall", type=float, default=0.999)
    parser.add_argument("--min-layout-recall", type=float, default=0.999)
    parser.add_argument("--min-current-cap-crop-recall", type=float, default=0.999)
    parser.add_argument("--min-policy-recall", type=float, default=0.999)
    parser.add_argument("--min-section-sender-gt-keys", type=int, default=None)
    parser.add_argument("--min-empty-section-sender-gt-keys", type=int, default=None)
    parser.add_argument("--min-section-sender-combined-crop-recall", type=float, default=0.999)
    parser.add_argument("--min-section-sender-policy-recall", type=float, default=0.999)
    parser.add_argument("--max-section-sender-total-pixels-vs-full", type=float, default=0.75)
    parser.add_argument("--max-section-sender-fallback-image-ratio", type=float, default=0.05)
    parser.add_argument("--max-current-cap-total-pixels-vs-full", type=float, default=1.05)
    parser.add_argument("--max-layout-fallback-pixels-vs-full", type=float, default=1.05)
    parser.add_argument("--max-section-total-pixels-vs-full", type=float, default=1.0)
    parser.add_argument("--allow-failed-gate", action="store_true", help="Write the report but exit 0 even when stress_ready=false.")
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "stress_ready": report["stress_ready"],
                "transport_ready": report["transport_ready"],
                "detector_ready": report["detector_ready"],
                "hard_failures": len(report["hard_failures"]),
                "warnings": len(report["warnings"]),
                "report": str(args.out / "stress_check.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["stress_ready"] and not args.allow_failed_gate:
        sys.exit(1)


if __name__ == "__main__":
    main()
