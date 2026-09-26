"""Run segmentation benchmark eval, error mining, and review workbench.

This is a thin orchestration wrapper for the Codex-vs-reviewed comparison loop:
question_segmentation_benchmark_eval.py -> question_segmentation_benchmark_mine.py
-> question_detector_review_workbench.py.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = SCRIPT_DIR / "question_segmentation_benchmark_eval.py"
MINE_SCRIPT = SCRIPT_DIR / "question_segmentation_benchmark_mine.py"
GATE_SCRIPT = SCRIPT_DIR / "question_segmentation_benchmark_gate.py"
WORKBENCH_SCRIPT = SCRIPT_DIR / "question_detector_review_workbench.py"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_step(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Codex/question segmentation benchmark eval and error review queue.")
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--label", default="codex_vs_reviewed")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-segmentation-benchmark-loop"))
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--min-f1", type=float, default=0.95)
    parser.add_argument("--min-cohort-recall", type=float, default=0.90)
    parser.add_argument("--min-cohort-precision", type=float, default=0.90)
    parser.add_argument("--max-missing-images", type=int, default=0)
    parser.add_argument("--max-unmatched-rows", type=int, default=0)
    parser.add_argument("--max-missed-boxes", type=int, default=0)
    parser.add_argument("--max-false-positive-boxes", type=int, default=0)
    parser.add_argument("--strict-gate", action="store_true", help="Exit nonzero when the benchmark gate fails.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    eval_out = args.out / "eval"
    gate_out = args.out / "gate"
    error_out = args.out / "errors"
    workbench_out = args.out / "error_workbench"
    steps = [
        [
            sys.executable,
            str(EVAL_SCRIPT),
            "--benchmark-manifest",
            str(args.benchmark_manifest),
            "--reference-jsonl",
            str(args.reference_jsonl),
            "--predictions-jsonl",
            str(args.predictions_jsonl),
            "--label",
            args.label,
            "--iou-threshold",
            str(args.iou_threshold),
            "--out",
            str(eval_out),
            *(["--clean"] if args.clean else []),
        ],
        [
            sys.executable,
            str(GATE_SCRIPT),
            "--summary",
            str(eval_out / "summary.json"),
            "--out",
            str(gate_out),
            "--min-recall",
            str(args.min_recall),
            "--min-precision",
            str(args.min_precision),
            "--min-f1",
            str(args.min_f1),
            "--min-cohort-recall",
            str(args.min_cohort_recall),
            "--min-cohort-precision",
            str(args.min_cohort_precision),
            "--max-missing-images",
            str(args.max_missing_images),
            "--max-unmatched-rows",
            str(args.max_unmatched_rows),
            "--max-missed-boxes",
            str(args.max_missed_boxes),
            "--max-false-positive-boxes",
            str(args.max_false_positive_boxes),
            "--allow-failed-gate",
        ],
        [
            sys.executable,
            str(MINE_SCRIPT),
            "--eval",
            str(eval_out),
            "--out",
            str(error_out),
            *(["--clean"] if args.clean else []),
        ],
        [
            sys.executable,
            str(WORKBENCH_SCRIPT),
            "--prelabel-root",
            str(error_out),
            "--out",
            str(workbench_out),
            *(["--clean"] if args.clean else []),
        ],
    ]

    step_reports: list[dict[str, Any]] = []
    failed = False
    for command in steps:
        report = run_step(command)
        step_reports.append(report)
        if report["returncode"] != 0:
            failed = True
            break

    gate_report = read_json(gate_out / "gate.json") if (gate_out / "gate.json").is_file() else {}
    gate_passed = bool(gate_report.get("gate_passed")) if gate_report else False
    status = "failed" if failed else ("passed" if gate_passed else "failed_gate")

    summary: dict[str, Any] = {
        "status": status,
        "outputs": {
            "eval": str(eval_out),
            "gate": str(gate_out),
            "errors": str(error_out),
            "error_workbench": str(workbench_out),
            "eval_summary": str(eval_out / "summary.json"),
            "gate_report": str(gate_out / "gate.json"),
            "error_review_root": str(error_out / "annotations" / "draft_boxes.jsonl"),
            "error_workbench_html": str(workbench_out / "index.html"),
        },
        "steps": step_reports,
    }
    if (eval_out / "summary.json").is_file():
        summary["eval_summary"] = read_json(eval_out / "summary.json")
    if gate_report:
        summary["gate_report"] = gate_report
    if (error_out / "summary.json").is_file():
        summary["error_summary"] = read_json(error_out / "summary.json")
    if (workbench_out / "summary.json").is_file():
        summary["workbench_summary"] = read_json(workbench_out / "summary.json")
    write_json(args.out / "loop_report.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "outputs")}, ensure_ascii=False, indent=2))
    if failed or (args.strict_gate and not gate_passed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
