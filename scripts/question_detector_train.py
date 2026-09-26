"""Train and export a question-block detector with audit-gated inputs.

This script intentionally delegates model training to Ultralytics YOLO while
keeping PXJ-specific gates, outputs, and evaluation stable. It refuses to train
on a dataset whose audit does not mark it model-training-ready unless
--allow-unready is passed for pipeline smoke tests.
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


COREML_SUFFIXES = (".mlpackage", ".mlmodel", ".mlmodelc")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_float_list(value: str) -> list[float]:
    result: list[float] = []
    for item in str(value or "").split(","):
        text = item.strip()
        if not text:
            continue
        result.append(float(text))
    return result


def effective_prediction_conf(args: argparse.Namespace) -> float:
    score_thresholds = parse_float_list(args.eval_score_thresholds)
    if not score_thresholds:
        return float(args.conf)
    return min(float(args.conf), min(score_thresholds))


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise SystemExit(f"{label} not found: {path}")
    return path


def load_dataset_audit(dataset: Path) -> dict[str, Any]:
    audit_path = require_file(dataset / "audit.json", "dataset audit")
    audit = load_json(audit_path)
    if not isinstance(audit, dict):
        raise SystemExit(f"dataset audit is not a JSON object: {audit_path}")
    return audit


def assert_audit_ready(audit: dict[str, Any], *, allow_unready: bool) -> None:
    readiness = audit.get("readiness") if isinstance(audit.get("readiness"), dict) else {}
    if readiness.get("model_training_ready"):
        return
    if allow_unready:
        return
    warnings = audit.get("warnings") if isinstance(audit.get("warnings"), list) else []
    warning_text = "\n".join(
        f"- {item.get('severity', 'warning')} {item.get('code', 'unknown')}: {item.get('detail', '')}"
        for item in warnings
        if isinstance(item, dict)
    )
    raise SystemExit(
        "dataset audit is not model-training-ready. "
        "Use --allow-unready only for pipeline smoke tests.\n"
        f"{warning_text}"
    )


def import_yolo():
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is required for training/export. Install it in this "
            "environment, or rerun with --dry-run to validate the command plan."
        ) from exc
    return YOLO


def maybe_clean(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def yolo_train_kwargs(args: argparse.Namespace, dataset_yaml: Path, runs_dir: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "data": str(dataset_yaml.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(runs_dir.resolve()),
        "name": args.name,
        "patience": args.patience,
        "workers": args.workers,
        "seed": args.seed,
        "exist_ok": True,
    }
    if args.device:
        kwargs["device"] = args.device
    if args.optimizer:
        kwargs["optimizer"] = args.optimizer
    return kwargs


def find_best_weights(save_dir: Path) -> Path:
    candidates = [
        save_dir / "weights" / "best.pt",
        save_dir / "best.pt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(save_dir.rglob("best.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    raise SystemExit(f"could not locate best.pt under training output: {save_dir}")


def latest_coreml_artifact(search_root: Path) -> Path | None:
    candidates = [
        path
        for path in search_root.rglob("*")
        if path.suffix.lower() in COREML_SUFFIXES and (path.is_file() or path.is_dir())
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def copy_coreml_artifact(source: Path, target_dir: Path, name: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{name}{source.suffix}"
    if target.exists():
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return target


def run_eval(dataset: Path, labels_dir: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    script = Path(__file__).resolve().parent / "question_detector_eval.py"
    cmd = [
        sys.executable,
        str(script),
        "--ground-truth",
        str(dataset / "annotations" / "coco_test.json"),
        "--predictions-yolo-dir",
        str(labels_dir),
        "--image-root",
        str(dataset),
        "--out",
        str(out_dir),
        "--thresholds",
        args.eval_thresholds,
        "--min-recall",
        str(args.min_eval_recall),
        "--min-precision",
        str(args.min_eval_precision),
        "--max-fp-per-image",
        str(args.max_eval_fp_per_image),
    ]
    if args.eval_score_thresholds:
        cmd.extend(["--score-thresholds", args.eval_score_thresholds])
    subprocess.run(cmd, check=True)
    summary_path = out_dir / "summary.json"
    return load_json(summary_path) if summary_path.is_file() else {}


def eval_gate_result(summary: dict[str, Any]) -> dict[str, Any]:
    gate = summary.get("gate") if isinstance(summary.get("gate"), dict) else {}
    checks = gate.get("checks") if isinstance(gate.get("checks"), dict) else {}
    failed = [
        key
        for key, value in checks.items()
        if isinstance(value, dict) and not bool(value.get("passed"))
    ]
    if gate:
        return {
            "passed": bool(gate.get("passed")),
            "failed_checks": failed,
            "gate": gate,
        }
    passed = bool(summary.get("pass_recall_95")) and bool(summary.get("pass_precision_90"))
    failed = []
    if not summary.get("pass_recall_95"):
        failed.append("recall")
    if not summary.get("pass_precision_90"):
        failed.append("precision")
    return {
        "passed": passed,
        "failed_checks": failed,
        "gate": {
            "passed": passed,
            "checks": {
                "pass_recall_95": {"passed": bool(summary.get("pass_recall_95"))},
                "pass_precision_90": {"passed": bool(summary.get("pass_precision_90"))},
            },
        },
    }


def prediction_labels_dir(predict_save_dir: Path) -> Path:
    labels = predict_save_dir / "labels"
    if labels.is_dir():
        return labels
    matches = [path for path in predict_save_dir.rglob("labels") if path.is_dir()]
    if matches:
        return matches[0]
    raise SystemExit(f"prediction labels were not written under: {predict_save_dir}")


def planned_manifest(args: argparse.Namespace, dataset: Path, audit: dict[str, Any]) -> dict[str, Any]:
    dataset_yaml = dataset / "yolo_dataset.yaml"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "dataset_yaml": str(dataset_yaml),
        "audit_readiness": audit.get("readiness", {}),
        "audit_warnings": audit.get("warnings", []),
        "model": args.model,
        "name": args.name,
        "train": {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "patience": args.patience,
            "workers": args.workers,
            "seed": args.seed,
            "optimizer": args.optimizer,
        },
        "export_coreml": bool(args.export_coreml),
        "predict_test": bool(args.predict_test),
        "prediction_conf": effective_prediction_conf(args),
        "prediction": {
            "requested_conf": args.conf,
            "effective_conf": effective_prediction_conf(args),
            "adjusted_for_score_sweep": effective_prediction_conf(args) < float(args.conf),
        },
        "allow_unready": bool(args.allow_unready),
        "eval_gate": {
            "thresholds": args.eval_thresholds,
            "min_recall": args.min_eval_recall,
            "min_precision": args.min_eval_precision,
            "max_fp_per_image": args.max_eval_fp_per_image,
            "score_thresholds": args.eval_score_thresholds,
            "allow_failed_eval": bool(args.allow_failed_eval),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit-gated question detector training/export.")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root from question_detector_dataset.py.")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-train"))
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model checkpoint/name to train from.")
    parser.add_argument("--name", default="question-block")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default="", help="Ultralytics device string, e.g. 0, cpu, or mps.")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer", default="", help="Optional Ultralytics optimizer override.")
    parser.add_argument("--conf", type=float, default=0.25, help="Prediction confidence for test export.")
    parser.add_argument("--export-coreml", action="store_true", help="Export best.pt to Core ML with NMS for VNCoreMLRequest.")
    parser.add_argument("--coreml-name", default="QuestionRegionDetector")
    parser.add_argument("--predict-test", action="store_true", help="Run predictions on images/test and evaluate them.")
    parser.add_argument("--eval-thresholds", default="0.5,0.75", help="Comma-separated IoU thresholds for question_detector_eval.py.")
    parser.add_argument("--min-eval-recall", type=float, default=0.95, help="Minimum primary-threshold recall required after --predict-test.")
    parser.add_argument("--min-eval-precision", type=float, default=0.90, help="Minimum primary-threshold precision required after --predict-test.")
    parser.add_argument("--max-eval-fp-per-image", type=float, default=0.05, help="Maximum primary-threshold false positives per image after --predict-test.")
    parser.add_argument("--eval-score-thresholds", default="", help="Optional comma-separated detector score thresholds to sweep during --predict-test eval.")
    parser.add_argument("--allow-failed-eval", action="store_true", help="Write failed eval gate to manifest but do not exit nonzero.")
    parser.add_argument("--allow-unready", action="store_true", help="Allow training on audit-unready data for smoke tests only.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and write a plan without importing Ultralytics.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    dataset = require_dir(args.dataset, "dataset").resolve()
    args.out = args.out.resolve()
    dataset_yaml = require_file(dataset / "yolo_dataset.yaml", "YOLO dataset yaml").resolve()
    require_file(dataset / "audit.json", "dataset audit")
    require_file(dataset / "annotations" / "coco_test.json", "COCO test annotations")
    if args.predict_test:
        require_dir(dataset / "images" / "test", "test images")
    try:
        effective_conf = effective_prediction_conf(args)
    except ValueError as exc:
        raise SystemExit(f"invalid --eval-score-thresholds: {args.eval_score_thresholds}") from exc

    audit = load_dataset_audit(dataset)
    assert_audit_ready(audit, allow_unready=args.allow_unready)

    maybe_clean(args.out, args.clean)
    manifest = planned_manifest(args, dataset, audit)
    manifest["status"] = "planned" if args.dry_run else "running"
    write_json(args.out / "train_manifest.json", manifest)

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    YOLO = import_yolo()
    runs_dir = args.out / "runs"
    train_kwargs = yolo_train_kwargs(args, dataset_yaml, runs_dir)
    model = YOLO(args.model)
    train_result = model.train(**train_kwargs)
    save_dir = Path(getattr(train_result, "save_dir", runs_dir / args.name))
    best_pt = find_best_weights(save_dir)

    manifest["status"] = "trained"
    manifest["train_save_dir"] = str(save_dir)
    manifest["best_pt"] = str(best_pt)

    if args.export_coreml:
        export_model = YOLO(str(best_pt))
        export_model.export(format="coreml", imgsz=args.imgsz, nms=True)
        artifact = latest_coreml_artifact(best_pt.parent.parent)
        if artifact is None:
            artifact = latest_coreml_artifact(best_pt.parent)
        if artifact is None:
            raise SystemExit("Core ML export finished but no .mlpackage/.mlmodel/.mlmodelc artifact was found.")
        copied = copy_coreml_artifact(artifact, args.out / "coreml", args.coreml_name)
        manifest["coreml_artifact"] = str(copied)

    if args.predict_test:
        pred_project = args.out / "predict"
        pred_model = YOLO(str(best_pt))
        pred_result = pred_model.predict(
            source=str((dataset / "images" / "test").resolve()),
            imgsz=args.imgsz,
            conf=effective_conf,
            save_txt=True,
            save_conf=True,
            project=str(pred_project.resolve()),
            name=f"{args.name}-test",
            exist_ok=True,
            device=args.device or None,
        )
        pred_save_dir = Path(getattr(pred_result[0], "save_dir", pred_project / f"{args.name}-test")) if pred_result else pred_project / f"{args.name}-test"
        labels_dir = prediction_labels_dir(pred_save_dir)
        eval_summary = run_eval(dataset, labels_dir, args.out / "eval", args)
        manifest["prediction_labels_dir"] = str(labels_dir)
        manifest["prediction_conf"] = effective_conf
        manifest["prediction"] = {
            "requested_conf": args.conf,
            "effective_conf": effective_conf,
            "adjusted_for_score_sweep": effective_conf < float(args.conf),
        }
        manifest["eval"] = eval_summary
        gate = eval_gate_result(eval_summary)
        manifest["eval_gate_result"] = gate
        if not gate["passed"]:
            manifest["status"] = "failed_eval_gate"
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(args.out / "train_manifest.json", manifest)
            if not args.allow_failed_eval:
                failed = ", ".join(gate.get("failed_checks") or ["unknown"])
                raise SystemExit(f"question detector eval gate failed: {failed}")

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(args.out / "train_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
