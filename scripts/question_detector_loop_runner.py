"""Run one repeatable question-detector iteration.

This script ties together the local tools without weakening their gates:

1. optional dataset export
2. optional grouped CV export
3. experiment matrix plan/execute
4. release readiness check when a selection exists
5. error mining from the most informative failed/passed eval
6. review workbench for mined hard examples
7. iteration report

By default it plans the matrix but does not train. Use --execute-training for a
real run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_SCRIPT = SCRIPT_DIR / "question_detector_dataset.py"
MERGE_REVIEWED_SCRIPT = SCRIPT_DIR / "question_detector_merge_reviewed.py"
CV_SCRIPT = SCRIPT_DIR / "question_detector_cv.py"
MATRIX_SCRIPT = SCRIPT_DIR / "question_detector_experiment_matrix.py"
RELEASE_SCRIPT = SCRIPT_DIR / "question_detector_release_check.py"
ERROR_MINING_SCRIPT = SCRIPT_DIR / "question_detector_error_mining.py"
WORKBENCH_SCRIPT = SCRIPT_DIR / "question_detector_review_workbench.py"
ITERATION_SCRIPT = SCRIPT_DIR / "question_detector_iteration_report.py"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def maybe_clean(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_command(command: list[str], cwd: Path, log_dir: Path, name: str, execute: bool = True) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "name": name,
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": "",
        "returncode": None,
        "status": "planned",
        "stdout": str(log_dir / f"{name}.stdout.txt"),
        "stderr": str(log_dir / f"{name}.stderr.txt"),
    }
    if not execute:
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        return result
    with Path(result["stdout"]).open("w", encoding="utf-8") as stdout, Path(result["stderr"]).open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=str(cwd), stdout=stdout, stderr=stderr, text=True)
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    result["returncode"] = completed.returncode
    result["status"] = "passed" if completed.returncode == 0 else "failed"
    return result


def build_merge_reviewed_command(args: argparse.Namespace, merged_out: Path) -> list[str]:
    command = [sys.executable, str(MERGE_REVIEWED_SCRIPT)]
    command.extend(str(path) for path in args.merge_reviewed)
    command.extend(["--out", str(merged_out), "--clean"])
    return command


def build_dataset_command(args: argparse.Namespace, dataset_out: Path, reviewed_prelabels: list[Path]) -> list[str]:
    command = [sys.executable, str(DATASET_SCRIPT), "--out", str(dataset_out), "--split-scope", args.split_scope, "--clean"]
    for path in reviewed_prelabels:
        command.extend(["--reviewed-prelabels", str(path)])
    for path in args.negative_images_dir:
        command.extend(["--negative-images-dir", str(path)])
    for path in args.diagnostics:
        command.extend(["--diagnostics", str(path)])
    if args.diagnostics_root:
        command.extend(["--diagnostics-root", str(args.diagnostics_root)])
    if args.sqlite:
        command.extend(["--sqlite", str(args.sqlite)])
    if args.data_dir:
        command.extend(["--data-dir", str(args.data_dir)])
    if args.include_sqlite_empty_pages:
        command.append("--include-sqlite-empty-pages")
        command.extend(["--sqlite-empty-page-mode", args.sqlite_empty_page_mode])
    if args.min_production_negative_images is not None:
        command.extend(["--min-production-negative-images", str(args.min_production_negative_images)])
    return command


def build_cv_command(args: argparse.Namespace, dataset: Path, cv_out: Path) -> list[str]:
    return [
        sys.executable,
        str(CV_SCRIPT),
        "--dataset",
        str(dataset),
        "--out",
        str(cv_out),
        "--folds",
        str(args.folds),
        "--clean",
    ]


def build_matrix_command(args: argparse.Namespace, dataset: Path | None, cv_root: Path | None, experiment_out: Path) -> list[str]:
    command = [
        sys.executable,
        str(MATRIX_SCRIPT),
        "--out",
        str(experiment_out),
        "--model",
        args.models,
        "--imgsz",
        args.imgsz,
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--eval-score-thresholds",
        args.eval_score_thresholds,
        "--clean",
    ]
    if cv_root is not None:
        command.extend(["--cv-root", str(cv_root)])
    elif dataset is not None:
        command.extend(["--dataset", str(dataset)])
    if args.device:
        command.extend(["--device", args.device])
    if args.allow_unready:
        command.append("--allow-unready")
    if args.allow_failed_eval:
        command.append("--allow-failed-eval")
    if args.train_dry_run:
        command.append("--train-dry-run")
    if args.execute_training:
        command.append("--execute")
    return command


def build_release_command(args: argparse.Namespace, dataset: Path, experiment_out: Path, release_out: Path) -> list[str] | None:
    selection = experiment_out / "selection"
    selection_summary = selection / "selection_summary.json"
    if not selection_summary.is_file():
        return None
    command = [
        sys.executable,
        str(RELEASE_SCRIPT),
        "--dataset",
        str(dataset),
        "--selection",
        str(selection),
        "--ios-root",
        str(args.ios_root),
        "--backend-main",
        str(args.backend_main),
        "--out",
        str(release_out),
        "--clean",
    ]
    for manifest in sorted((experiment_out / "runs").glob("*/train_manifest.json")):
        command.extend(["--train-manifest", str(manifest)])
    for replay in args.replay:
        command.extend(["--replay", str(replay)])
    for dedupe_tune in args.dedupe_tune:
        command.extend(["--dedupe-tune", str(dedupe_tune)])
    for fallback_tune in args.fallback_tune:
        command.extend(["--fallback-tune", str(fallback_tune)])
    for candidate_eval in args.candidate_eval:
        command.extend(["--candidate-eval", str(candidate_eval)])
    for candidate_cap_sweep in args.candidate_cap_sweep:
        command.extend(["--candidate-cap-sweep", str(candidate_cap_sweep)])
    for section_sender_eval in args.section_sender_eval:
        command.extend(["--section-sender-eval", str(section_sender_eval)])
    for strategy_selection in args.strategy_selection:
        command.extend(["--strategy-selection", str(strategy_selection)])
    for runtime_perf in args.runtime_perf:
        command.extend(["--runtime-perf", str(runtime_perf)])
    if args.allow_runtime_fixture:
        command.append("--allow-runtime-fixture")
    if args.xcode_build_log:
        command.extend(["--xcode-build-log", str(args.xcode_build_log)])
    if not args.strict_release_gate:
        command.append("--allow-failed-gate")
    return command


def metric_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def choose_eval_summary(experiment_out: Path) -> Path | None:
    candidates: list[dict[str, Any]] = []
    for manifest_path in sorted((experiment_out / "runs").glob("*/train_manifest.json")):
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            continue
        eval_summary = manifest.get("eval") if isinstance(manifest.get("eval"), dict) else {}
        threshold = f"{metric_float(eval_summary, 'primary_threshold', 0.5):.2f}"
        primary = (eval_summary.get("thresholds") or {}).get(threshold) if isinstance(eval_summary.get("thresholds"), dict) else {}
        summary_path = manifest_path.parent / "eval" / "summary.json"
        if not summary_path.is_file():
            continue
        candidates.append(
            {
                "path": summary_path,
                "recall": metric_float(primary or {}, "recall"),
                "precision": metric_float(primary or {}, "precision"),
                "problem_count": int((primary or {}).get("problem_image_count") or 0),
                "status": manifest.get("status") or "",
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda row: (-row["problem_count"], row["recall"], row["precision"]))
    return Path(candidates[0]["path"])


def training_plan_review_workbenches(paths: list[Path]) -> list[Path]:
    workbenches: list[Path] = []
    for path in paths:
        plan_path = path if path.is_file() else path / "training_readiness_plan.json"
        if not plan_path.is_file():
            continue
        try:
            report = read_json(plan_path)
        except (OSError, json.JSONDecodeError):
            continue
        inputs = report.get("inputs") if isinstance(report.get("inputs"), dict) else {}
        for workbench in inputs.get("review_workbenches") or []:
            if workbench:
                workbenches.append(Path(str(workbench)))
    return workbenches


def build_iteration_command(
    args: argparse.Namespace,
    dataset: Path,
    experiment_out: Path,
    release_out: Path | None,
    error_out: Path | None,
    iteration_out: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(ITERATION_SCRIPT),
        "--dataset",
        str(dataset),
        "--experiment",
        str(experiment_out),
        "--out",
        str(iteration_out),
        "--clean",
    ]
    if release_out is not None and (release_out / "release_check.json").is_file():
        command.extend(["--release-check", str(release_out)])
    for replay in args.replay:
        command.extend(["--replay", str(replay)])
    for strategy_selection in args.strategy_selection:
        command.extend(["--strategy-selection", str(strategy_selection)])
    for training_plan in args.training_plan:
        command.extend(["--training-plan", str(training_plan)])
    for review_queue in args.review_queue:
        command.extend(["--review-queue", str(review_queue)])
    if error_out is not None and (error_out / "summary.json").is_file():
        command.extend(["--error-mining", str(error_out)])
    review_workbenches = list(args.review_workbench)
    review_workbenches.extend(training_plan_review_workbenches(list(args.training_plan)))
    generated_workbench = args.out / "error_workbench"
    if generated_workbench.exists():
        review_workbenches.append(generated_workbench)
    seen_workbenches: set[str] = set()
    for workbench in review_workbenches:
        key = str(workbench.resolve()) if workbench.exists() else str(workbench)
        if key in seen_workbenches:
            continue
        seen_workbenches.add(key)
        command.extend(["--review-workbench", str(workbench)])
    return command


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    maybe_clean(args.out, args.clean)
    logs = args.out / "logs"
    steps: list[dict[str, Any]] = []
    dataset = args.dataset or (args.out / "dataset")
    merged_reviewed: Path | None = None
    reviewed_prelabels = list(args.reviewed_prelabels)

    if args.merge_reviewed:
        merged_reviewed = args.out / "merged_reviewed"
        command = build_merge_reviewed_command(args, merged_reviewed)
        steps.append(run_command(command, Path.cwd(), logs, "merge_reviewed", execute=True))
        if steps[-1]["returncode"] != 0:
            return finish_report(args, steps, dataset, merged_reviewed, None, None, None, None, "failed_merge_reviewed")
        reviewed_prelabels.append(merged_reviewed)

    if args.dataset is None:
        command = build_dataset_command(args, dataset, reviewed_prelabels)
        steps.append(run_command(command, Path.cwd(), logs, "dataset", execute=True))
        if steps[-1]["returncode"] != 0:
            return finish_report(args, steps, dataset, merged_reviewed, None, None, None, None, "failed_dataset")

    cv_root: Path | None = None
    if args.folds > 1:
        cv_root = args.out / "cv"
        command = build_cv_command(args, dataset, cv_root)
        steps.append(run_command(command, Path.cwd(), logs, "cv", execute=True))
        if steps[-1]["returncode"] != 0:
            return finish_report(args, steps, dataset, merged_reviewed, cv_root, None, None, None, "failed_cv")

    experiment_out = args.out / "experiment"
    command = build_matrix_command(args, dataset, cv_root, experiment_out)
    steps.append(run_command(command, Path.cwd(), logs, "experiment", execute=True))
    if steps[-1]["returncode"] != 0:
        return finish_report(args, steps, dataset, merged_reviewed, cv_root, experiment_out, None, None, "failed_experiment")

    release_out: Path | None = None
    release_command = build_release_command(args, dataset, experiment_out, args.out / "release_check")
    if release_command is not None:
        release_out = args.out / "release_check"
        steps.append(run_command(release_command, Path.cwd(), logs, "release_check", execute=True))

    error_out: Path | None = None
    eval_summary = choose_eval_summary(experiment_out)
    if eval_summary is not None:
        error_out = args.out / "error_mining"
        error_command = [
            sys.executable,
            str(ERROR_MINING_SCRIPT),
            "--eval-summary",
            str(eval_summary),
            "--dataset-root",
            str(dataset),
            "--out",
            str(error_out),
            "--clean",
        ]
        steps.append(run_command(error_command, Path.cwd(), logs, "error_mining", execute=True))
        if (error_out / "annotations" / "draft_boxes.jsonl").is_file():
            workbench_out = args.out / "error_workbench"
            workbench_command = [
                sys.executable,
                str(WORKBENCH_SCRIPT),
                "--prelabel-root",
                str(error_out),
                "--out",
                str(workbench_out),
                "--clean",
            ]
            steps.append(run_command(workbench_command, Path.cwd(), logs, "error_workbench", execute=True))

    iteration_out = args.out / "iteration_report"
    iteration_command = build_iteration_command(args, dataset, experiment_out, release_out, error_out, iteration_out)
    steps.append(run_command(iteration_command, Path.cwd(), logs, "iteration_report", execute=True))

    status = "passed"
    if any(step.get("status") == "failed" for step in steps):
        status = "failed"
    elif experiment_out.is_dir():
        experiment_report = experiment_out / "experiment_report.json"
        if experiment_report.is_file():
            try:
                experiment = read_json(experiment_report)
                if experiment.get("status") == "failed_eval_gate":
                    status = "failed_eval_gate"
            except (OSError, json.JSONDecodeError):
                pass
    return finish_report(args, steps, dataset, merged_reviewed, cv_root, experiment_out, release_out, error_out, status)


def finish_report(
    args: argparse.Namespace,
    steps: list[dict[str, Any]],
    dataset: Path | None,
    merged_reviewed: Path | None,
    cv_root: Path | None,
    experiment_out: Path | None,
    release_out: Path | None,
    error_out: Path | None,
    status: str,
) -> dict[str, Any]:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "out": str(args.out),
        "execute_training": bool(args.execute_training),
        "train_dry_run": bool(args.train_dry_run),
        "artifacts": {
            "merged_reviewed": str(merged_reviewed) if merged_reviewed else "",
            "dataset": str(dataset) if dataset else "",
            "cv": str(cv_root) if cv_root else "",
            "experiment": str(experiment_out) if experiment_out else "",
            "release_check": str(release_out) if release_out else "",
            "error_mining": str(error_out) if error_out else "",
            "error_workbench": str(args.out / "error_workbench") if (args.out / "error_workbench").exists() else "",
            "iteration_report": str(args.out / "iteration_report"),
        },
        "steps": steps,
    }
    write_json(args.out / "loop_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one repeatable question-detector loop.")
    parser.add_argument("--dataset", type=Path, help="Existing detector dataset root. If omitted, dataset export is run.")
    parser.add_argument("--merge-reviewed", type=Path, action="append", default=[], help="Approved review exports or workbench/package dirs to merge before dataset export.")
    parser.add_argument("--reviewed-prelabels", type=Path, action="append", default=[])
    parser.add_argument("--negative-images-dir", type=Path, action="append", default=[])
    parser.add_argument("--diagnostics", type=Path, action="append", default=[])
    parser.add_argument("--diagnostics-root", type=Path)
    parser.add_argument("--sqlite", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--include-sqlite-empty-pages", action="store_true")
    parser.add_argument("--sqlite-empty-page-mode", choices=["textless", "all_observations"], default="textless")
    parser.add_argument("--split-scope", choices=["batch", "session", "source"], default="session")
    parser.add_argument("--min-production-negative-images", type=int)
    parser.add_argument("--folds", type=int, default=0, help="Run grouped CV when > 1.")
    parser.add_argument("--models", default="yolo11n.pt")
    parser.add_argument("--imgsz", default="768")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--eval-score-thresholds", default="0.05,0.10,0.20,0.25,0.35,0.50")
    parser.add_argument("--allow-unready", action="store_true", help="Smoke/exploration only.")
    parser.add_argument("--allow-failed-eval", action="store_true", help="Keep failed eval manifests for mining.")
    parser.add_argument("--train-dry-run", action="store_true", help="Validate training commands without YOLO training.")
    parser.add_argument("--execute-training", action="store_true", help="Actually run experiment matrix training jobs.")
    parser.add_argument("--replay", type=Path, action="append", default=[])
    parser.add_argument("--dedupe-tune", type=Path, action="append", default=[], help="question_observation_dedupe_tune root or summary.json to pass into the release gate.")
    parser.add_argument("--fallback-tune", type=Path, action="append", default=[], help="question_observation_fallback_tune root or summary.json to pass into the release gate.")
    parser.add_argument("--candidate-eval", type=Path, action="append", default=[], help="question_observation_candidate_eval root or summary.json to pass into the release gate.")
    parser.add_argument("--candidate-cap-sweep", type=Path, action="append", default=[], help="question_observation_candidate_cap_sweep root or summary.json to pass into the release gate.")
    parser.add_argument("--section-sender-eval", type=Path, action="append", default=[], help="question_observation_section_sender_eval root or summary.json to pass into the release gate.")
    parser.add_argument("--strategy-selection", type=Path, action="append", default=[], help="question_observation_strategy_selector root or strategy_selection.json to pass into release and iteration reports.")
    parser.add_argument("--runtime-perf", type=Path, action="append", default=[], help="question_observation_runtime_perf_check root or summary.json to pass into the release gate.")
    parser.add_argument("--allow-runtime-fixture", action="store_true", help="Smoke only: pass --allow-runtime-fixture to the release gate.")
    parser.add_argument("--training-plan", type=Path, action="append", default=[], help="question_detector_training_readiness_plan root or training_readiness_plan.json to pass into iteration reports.")
    parser.add_argument("--review-queue", type=Path, action="append", default=[], help="question_detector_review_queue_manifest root or review_queue_manifest.json to pass into iteration reports.")
    parser.add_argument("--review-workbench", type=Path, action="append", default=[], help="Review workbench roots to include in the iteration report.")
    parser.add_argument("--ios-root", type=Path, default=Path("ios/PXJ/App"))
    parser.add_argument("--backend-main", type=Path, default=Path("backend/app/main.py"), help="Backend main.py used by the release gate static crop-source checks.")
    parser.add_argument("--xcode-build-log", type=Path)
    parser.add_argument("--strict-release-gate", action="store_true", help="Let release_ready=false make the release-check step fail instead of only reporting blockers.")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-loop"))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.dataset is not None and args.merge_reviewed:
        parser.error("--merge-reviewed can only be used when this loop exports a fresh dataset; omit --dataset")
    report = run_loop(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "out": str(args.out),
                "artifacts": report["artifacts"],
                "steps": [{"name": step["name"], "status": step["status"], "returncode": step["returncode"]} for step in report["steps"]],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
