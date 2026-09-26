"""Plan and run cross-validated question detector experiment matrices.

This is the orchestration layer above:

- question_detector_cv.py
- question_detector_train.py
- question_detector_select.py

It lets us compare multiple model checkpoints, input sizes, and seeds across
the same grouped folds, then feed completed runs into the selector. By default
it only writes a plan and a PowerShell runner. Use --execute to launch jobs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "question_detector_train.py"
SELECT_SCRIPT = SCRIPT_DIR / "question_detector_select.py"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_clean(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_int_list(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return default
    parsed: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                parsed.append(int(part))
    return parsed


def split_values(values: list[str] | None, default: list[str]) -> list[str]:
    if not values:
        return default
    parsed: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                parsed.append(part)
    return parsed


def slug(value: str, max_len: int = 80) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    return (safe or "run")[:max_len]


def ps_quote(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_./:\\-]+", value):
        return value
    return "'" + value.replace("'", "''") + "'"


def command_to_ps(command: list[str]) -> str:
    return " ".join(ps_quote(part) for part in command)


def discover_datasets(cv_root: Path | None, explicit: list[Path]) -> list[Path]:
    datasets: list[Path] = []
    if cv_root is not None:
        if not cv_root.is_dir():
            raise SystemExit(f"--cv-root does not exist or is not a directory: {cv_root}")
        datasets.extend(path for path in sorted(cv_root.glob("fold-*")) if path.is_dir())
    datasets.extend(explicit)
    unique: dict[str, Path] = {}
    for dataset in datasets:
        if not dataset.is_dir():
            raise SystemExit(f"dataset does not exist or is not a directory: {dataset}")
        if not (dataset / "yolo_dataset.yaml").is_file():
            raise SystemExit(f"dataset is missing yolo_dataset.yaml: {dataset}")
        if not (dataset / "audit.json").is_file():
            raise SystemExit(f"dataset is missing audit.json: {dataset}")
        unique[str(dataset.resolve())] = dataset
    if not unique:
        raise SystemExit("No datasets supplied. Use --cv-root or --dataset.")
    return sorted(unique.values())


def dataset_summary(path: Path) -> dict[str, Any]:
    audit = read_json(path / "audit.json")
    readiness = audit.get("readiness") if isinstance(audit.get("readiness"), dict) else {}
    counts = audit.get("counts") if isinstance(audit.get("counts"), dict) else {}
    split_counts = audit.get("split_counts") if isinstance(audit.get("split_counts"), dict) else {}
    split_positive_images = sum(
        int(value.get("positive_images") or 0)
        for value in split_counts.values()
        if isinstance(value, dict)
    )
    return {
        "path": str(path),
        "pilot_eval_ready": bool(readiness.get("pilot_eval_ready")),
        "model_training_ready": bool(readiness.get("model_training_ready")),
        "source_images": int(counts.get("source_images") or counts.get("images") or 0),
        "positive_images": int(counts.get("positive_images") or split_positive_images or 0),
        "negative_images": int(counts.get("negative_images") or 0),
        "annotations": int(counts.get("annotations") or 0),
        "split_groups": int(counts.get("split_groups") or 0),
    }


@dataclass
class MatrixRun:
    run_id: str
    dataset: Path
    out: Path
    model: str
    imgsz: int
    seed: int
    epochs: int
    command: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset": str(self.dataset),
            "out": str(self.out),
            "model": self.model,
            "imgsz": self.imgsz,
            "seed": self.seed,
            "epochs": self.epochs,
            "command": self.command,
            "powershell": command_to_ps(self.command),
        }


def build_run_command(args: argparse.Namespace, dataset: Path, run_out: Path, model: str, imgsz: int, seed: int, epochs: int) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset",
        str(dataset),
        "--out",
        str(run_out),
        "--model",
        model,
        "--name",
        args.name,
        "--epochs",
        str(epochs),
        "--imgsz",
        str(imgsz),
        "--batch",
        str(args.batch),
        "--workers",
        str(args.workers),
        "--seed",
        str(seed),
        "--conf",
        str(args.conf),
        "--eval-thresholds",
        args.eval_thresholds,
        "--min-eval-recall",
        str(args.min_eval_recall),
        "--min-eval-precision",
        str(args.min_eval_precision),
        "--max-eval-fp-per-image",
        str(args.max_eval_fp_per_image),
        "--eval-score-thresholds",
        args.eval_score_thresholds,
        "--clean",
    ]
    if args.device:
        command.extend(["--device", args.device])
    if args.optimizer:
        command.extend(["--optimizer", args.optimizer])
    if args.predict_test:
        command.append("--predict-test")
    if args.export_coreml:
        command.append("--export-coreml")
        command.extend(["--coreml-name", args.coreml_name])
    if args.allow_unready:
        command.append("--allow-unready")
    if args.allow_failed_eval:
        command.append("--allow-failed-eval")
    if args.train_dry_run:
        command.append("--dry-run")
    return command


def build_runs(args: argparse.Namespace, datasets: list[Path]) -> list[MatrixRun]:
    models = split_values(args.model, ["yolo11n.pt"])
    imgszs = parse_int_list(args.imgsz, [768])
    seeds = parse_int_list(args.seed, [42])
    epochs_values = parse_int_list(args.epochs, [80])
    runs: list[MatrixRun] = []
    for dataset, model, imgsz, seed, epochs in itertools.product(datasets, models, imgszs, seeds, epochs_values):
        fold_name = slug(dataset.name)
        model_name = slug(Path(model).stem)
        run_id = f"{fold_name}__{model_name}__img{imgsz}__seed{seed}__ep{epochs}"
        run_out = args.out / "runs" / run_id
        command = build_run_command(args, dataset, run_out, model, imgsz, seed, epochs)
        runs.append(MatrixRun(run_id, dataset, run_out, model, imgsz, seed, epochs, command))
    return runs


def write_plan(args: argparse.Namespace, datasets: list[Path], runs: list[MatrixRun]) -> dict[str, Any]:
    plan = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execute": bool(args.execute),
        "train_dry_run": bool(args.train_dry_run),
        "dataset_count": len(datasets),
        "run_count": len(runs),
        "datasets": [dataset_summary(path) for path in datasets],
        "runs": [run.to_json() for run in runs],
        "selection_out": str(args.out / "selection"),
    }
    write_json(args.out / "experiment_plan.json", plan)
    lines = [
        "$ErrorActionPreference = 'Stop'",
        f"# Generated {plan['generated_at']}",
        f"# Runs: {len(runs)}",
        "",
    ]
    for run in runs:
        lines.append(command_to_ps(run.command))
    if runs and args.predict_test and not args.train_dry_run:
        select_command = [
            sys.executable,
            str(SELECT_SCRIPT),
            *[str(run.out) for run in runs],
            "--out",
            str(args.out / "selection"),
        ]
        lines.extend(["", "# Select best candidate after all runs finish.", command_to_ps(select_command)])
    (args.out / "run_experiment_matrix.ps1").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan


def execute_run(run: MatrixRun, cwd: Path) -> dict[str, Any]:
    matrix_root = run.out.parent.parent
    log_dir = matrix_root / "logs" / run.run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    stdout_path = log_dir / "stdout.txt"
    stderr_path = log_dir / "stderr.txt"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(run.command, cwd=str(cwd), stdout=stdout, stderr=stderr, text=True)
    finished = datetime.now(timezone.utc).isoformat()
    manifest_path = run.out / "train_manifest.json"
    manifest_status = ""
    eval_gate_passed: bool | None = None
    if manifest_path.is_file():
        try:
            manifest = read_json(manifest_path)
            if isinstance(manifest, dict):
                manifest_status = str(manifest.get("status") or "")
                gate = manifest.get("eval_gate_result") if isinstance(manifest.get("eval_gate_result"), dict) else {}
                if gate:
                    eval_gate_passed = bool(gate.get("passed"))
        except (OSError, json.JSONDecodeError):
            manifest_status = "unreadable"
    return {
        "run_id": run.run_id,
        "out": str(run.out),
        "started_at": started,
        "finished_at": finished,
        "returncode": completed.returncode,
        "status": "passed" if completed.returncode == 0 else "failed",
        "manifest_status": manifest_status,
        "eval_gate_passed": eval_gate_passed,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "manifest": str(manifest_path),
    }


def execute_selection(args: argparse.Namespace, runs: list[MatrixRun]) -> dict[str, Any] | None:
    if args.skip_select or args.train_dry_run or not args.predict_test:
        return None
    successful_run_dirs = [run.out for run in runs if (run.out / "train_manifest.json").is_file()]
    if not successful_run_dirs:
        return None
    selection_out = args.out / "selection"
    command = [
        sys.executable,
        str(SELECT_SCRIPT),
        *[str(path) for path in successful_run_dirs],
        "--out",
        str(selection_out),
    ]
    log_dir = selection_out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.txt"
    stderr_path = log_dir / "stderr.txt"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=str(Path.cwd()), stdout=stdout, stderr=stderr, text=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "status": "passed" if completed.returncode == 0 else "failed",
        "out": str(selection_out),
        "summary": str(selection_out / "selection_summary.json"),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    maybe_clean(args.out, args.clean)
    datasets = discover_datasets(args.cv_root, args.dataset)
    runs = build_runs(args, datasets)
    plan = write_plan(args, datasets, runs)
    results: list[dict[str, Any]] = []
    stopped_early = False
    if args.execute:
        for run in runs:
            result = execute_run(run, Path.cwd())
            results.append(result)
            if result["returncode"] != 0 and not args.keep_going:
                stopped_early = True
                break
    selection = execute_selection(args, runs) if args.execute and not stopped_early else None
    process_failed = stopped_early or any(item["returncode"] != 0 for item in results)
    eval_failed = any(
        item.get("manifest_status") == "failed_eval_gate" or item.get("eval_gate_passed") is False
        for item in results
    )
    status = "planned"
    if args.execute:
        if process_failed:
            status = "failed"
        elif eval_failed:
            status = "failed_eval_gate"
        else:
            status = "passed"
    report = {
        **plan,
        "executed_runs": results,
        "stopped_early": stopped_early,
        "selection": selection,
        "status": status,
    }
    write_json(args.out / "experiment_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run a question detector experiment matrix.")
    parser.add_argument("--cv-root", type=Path, help="Root containing fold-* dataset directories from question_detector_cv.py.")
    parser.add_argument("--dataset", type=Path, action="append", default=[], help="Dataset root to train directly; may be repeated.")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-experiment-matrix"))
    parser.add_argument("--model", action="append", help="Model checkpoint/name; comma-separated values are accepted.")
    parser.add_argument("--imgsz", action="append", help="Input sizes; comma-separated values are accepted.")
    parser.add_argument("--seed", action="append", help="Seeds; comma-separated values are accepted.")
    parser.add_argument("--epochs", action="append", help="Epoch counts; comma-separated values are accepted.")
    parser.add_argument("--name", default="question-block")
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", default="")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--eval-thresholds", default="0.5,0.75")
    parser.add_argument("--eval-score-thresholds", default="0.05,0.10,0.20,0.25,0.35,0.50")
    parser.add_argument("--min-eval-recall", type=float, default=0.95)
    parser.add_argument("--min-eval-precision", type=float, default=0.90)
    parser.add_argument("--max-eval-fp-per-image", type=float, default=0.05)
    parser.add_argument("--no-predict-test", dest="predict_test", action="store_false", help="Do not export/evaluate test predictions.")
    parser.set_defaults(predict_test=True)
    parser.add_argument("--export-coreml", action="store_true")
    parser.add_argument("--coreml-name", default="QuestionRegionDetector")
    parser.add_argument("--allow-unready", action="store_true", help="Forward to question_detector_train.py for smoke tests only.")
    parser.add_argument("--allow-failed-eval", action="store_true")
    parser.add_argument("--train-dry-run", action="store_true", help="Forward --dry-run to train wrapper, useful for validating matrix commands.")
    parser.add_argument("--execute", action="store_true", help="Actually launch the planned train commands.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after failed train commands.")
    parser.add_argument("--skip-select", action="store_true", help="Do not run question_detector_select.py after execution.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    report = run_matrix(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_count": report["run_count"],
                "executed": bool(args.execute),
                "executed_run_count": len(report.get("executed_runs") or []),
                "out": str(args.out),
                "plan": str(args.out / "experiment_plan.json"),
                "report": str(args.out / "experiment_report.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
