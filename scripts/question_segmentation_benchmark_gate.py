"""Gate question segmentation benchmark results.

The evaluator computes metrics; this script turns them into an explicit pass or
fail decision so Codex/detector comparisons cannot silently pass with missing
images, unmatched input rows, or weak cohort performance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def metric(metrics: dict[str, Any], key: str) -> float:
    try:
        return float(metrics.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str, data: dict[str, Any] | None = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail, "data": data or {}})


def evaluate_gate(args: argparse.Namespace) -> dict[str, Any]:
    summary = read_json(args.summary)
    checks: list[dict[str, Any]] = []
    audit = summary.get("input_audit") if isinstance(summary.get("input_audit"), dict) else {}
    for side in ("reference", "predictions"):
        item = audit.get(side) if isinstance(audit.get(side), dict) else {}
        missing = item.get("missing_images") if isinstance(item.get("missing_images"), list) else []
        unmatched = item.get("unmatched_rows") if isinstance(item.get("unmatched_rows"), list) else []
        add_check(
            checks,
            f"{side} image coverage",
            len(missing) <= args.max_missing_images,
            f"{side} missing images <= {args.max_missing_images}",
            {"missing_image_count": len(missing), "matched_images": item.get("matched_images"), "rows": item.get("rows")},
        )
        add_check(
            checks,
            f"{side} unmatched rows",
            len(unmatched) <= args.max_unmatched_rows,
            f"{side} unmatched rows <= {args.max_unmatched_rows}",
            {"unmatched_row_count": len(unmatched)},
        )

    overall = summary.get("overall") if isinstance(summary.get("overall"), dict) else {}
    add_check(
        checks,
        "overall recall",
        metric(overall, "recall") >= args.min_recall,
        f"overall recall >= {args.min_recall}",
        {"actual": metric(overall, "recall")},
    )
    add_check(
        checks,
        "overall precision",
        metric(overall, "precision") >= args.min_precision,
        f"overall precision >= {args.min_precision}",
        {"actual": metric(overall, "precision")},
    )
    add_check(
        checks,
        "overall f1",
        metric(overall, "f1") >= args.min_f1,
        f"overall f1 >= {args.min_f1}",
        {"actual": metric(overall, "f1")},
    )
    add_check(
        checks,
        "missed boxes",
        int(overall.get("missed_boxes") or 0) <= args.max_missed_boxes,
        f"missed boxes <= {args.max_missed_boxes}",
        {"actual": int(overall.get("missed_boxes") or 0)},
    )
    add_check(
        checks,
        "false positive boxes",
        int(overall.get("false_positive_boxes") or 0) <= args.max_false_positive_boxes,
        f"false positive boxes <= {args.max_false_positive_boxes}",
        {"actual": int(overall.get("false_positive_boxes") or 0)},
    )

    cohort_checks: list[dict[str, Any]] = []
    by_cohort = summary.get("by_cohort") if isinstance(summary.get("by_cohort"), dict) else {}
    for cohort, metrics in sorted(by_cohort.items()):
        if not isinstance(metrics, dict):
            continue
        reference_boxes = int(metrics.get("reference_boxes") or 0)
        if reference_boxes < args.min_cohort_reference_boxes:
            continue
        recall = metric(metrics, "recall")
        precision = metric(metrics, "precision")
        passed = recall >= args.min_cohort_recall and precision >= args.min_cohort_precision
        cohort_checks.append(
            {
                "cohort": cohort,
                "passed": passed,
                "detail": f"recall >= {args.min_cohort_recall} and precision >= {args.min_cohort_precision}",
                "data": {
                    "reference_boxes": reference_boxes,
                    "prediction_boxes": int(metrics.get("prediction_boxes") or 0),
                    "recall": recall,
                    "precision": precision,
                    "f1": metric(metrics, "f1"),
                    "missed_boxes": int(metrics.get("missed_boxes") or 0),
                    "false_positive_boxes": int(metrics.get("false_positive_boxes") or 0),
                },
            }
        )
    if cohort_checks:
        add_check(
            checks,
            "cohort gates",
            all(item["passed"] for item in cohort_checks),
            "all sufficiently represented cohorts pass recall/precision gates",
            {"cohorts": cohort_checks},
        )
    else:
        add_check(checks, "cohort gates", False, "no sufficiently represented cohorts found", {})

    passed = all(item["passed"] for item in checks)
    report = {
        "gate_passed": passed,
        "summary": str(args.summary),
        "thresholds": {
            "min_recall": args.min_recall,
            "min_precision": args.min_precision,
            "min_f1": args.min_f1,
            "min_cohort_recall": args.min_cohort_recall,
            "min_cohort_precision": args.min_cohort_precision,
            "max_missing_images": args.max_missing_images,
            "max_unmatched_rows": args.max_unmatched_rows,
            "max_missed_boxes": args.max_missed_boxes,
            "max_false_positive_boxes": args.max_false_positive_boxes,
        },
        "checks": checks,
    }
    write_json(args.out / "gate.json", report)
    (args.out / "gate.md").write_text(make_markdown(report), encoding="utf-8")
    return report


def make_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Question Segmentation Benchmark Gate",
        "",
        f"- Gate passed: `{str(report['gate_passed']).lower()}`",
        f"- Summary: `{report['summary']}`",
        "",
        "## Checks",
        "",
    ]
    for check in report.get("checks") or []:
        mark = "PASS" if check.get("passed") else "FAIL"
        lines.append(f"- `{mark}` {check.get('name')}: {check.get('detail')}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate question segmentation benchmark metrics.")
    parser.add_argument("--summary", type=Path, required=True, help="summary.json from question_segmentation_benchmark_eval.py.")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-segmentation-benchmark-gate"))
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--min-f1", type=float, default=0.95)
    parser.add_argument("--min-cohort-recall", type=float, default=0.90)
    parser.add_argument("--min-cohort-precision", type=float, default=0.90)
    parser.add_argument("--min-cohort-reference-boxes", type=int, default=1)
    parser.add_argument("--max-missing-images", type=int, default=0)
    parser.add_argument("--max-unmatched-rows", type=int, default=0)
    parser.add_argument("--max-missed-boxes", type=int, default=0)
    parser.add_argument("--max-false-positive-boxes", type=int, default=0)
    parser.add_argument("--allow-failed-gate", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report = evaluate_gate(args)
    print(json.dumps({"gate_passed": report["gate_passed"], "out": str(args.out)}, ensure_ascii=False, indent=2))
    if not report["gate_passed"] and not args.allow_failed_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
