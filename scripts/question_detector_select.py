"""Select the best question detector run and iOS score threshold.

The selector consumes train_manifest.json files produced by
question_detector_train.py, including score_sweep results from
question_detector_eval.py. It ranks candidate model/threshold combinations
across one or more folds and writes a compact selection report.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_manifests(paths: list[Path]) -> list[Path]:
    manifests: list[Path] = []
    for path in paths:
        if path.is_file() and path.name == "train_manifest.json":
            manifests.append(path)
        elif path.is_dir():
            direct = path / "train_manifest.json"
            if direct.is_file():
                manifests.append(direct)
            manifests.extend(sorted(path.rglob("train_manifest.json")))
    unique: dict[str, Path] = {}
    for path in manifests:
        unique[str(path.resolve())] = path
    return sorted(unique.values())


def resolve_manifest_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        (manifest_path.parent / path),
        (Path.cwd() / path),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def path_key(path: Path | str) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return str(path)


def load_dataset_metadata(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    dataset = str(manifest.get("dataset") or "").strip()
    if not dataset:
        return {}
    metadata_path = resolve_manifest_path(dataset, manifest_path) / "metadata.json"
    if not metadata_path.is_file():
        return {}
    try:
        loaded = load_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def dataset_selection_context(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    dataset = str(manifest.get("dataset") or "").strip()
    dataset_path = resolve_manifest_path(dataset, manifest_path) if dataset else Path("")
    metadata = load_dataset_metadata(manifest, manifest_path)
    source_dataset = str(metadata.get("source_dataset") or "").strip()
    if source_dataset:
        dataset_family = path_key(resolve_manifest_path(source_dataset, manifest_path))
    elif dataset:
        dataset_family = path_key(dataset_path)
    else:
        dataset_family = ""
    return {
        "dataset_family": dataset_family,
        "fold_index": metadata.get("fold_index"),
        "folds": metadata.get("folds"),
        "group_field": metadata.get("group_field") or "",
    }


def candidate_group_key(
    *,
    dataset_family: str,
    model: str,
    name: str,
    train: dict[str, Any],
    prediction_conf: float,
    primary_threshold: float,
    score: float,
) -> str:
    return "|".join(
        [
            f"dataset={dataset_family}",
            f"model={model}",
            f"name={name}",
            f"imgsz={int(train.get('imgsz') or 0)}",
            f"epochs={int(train.get('epochs') or 0)}",
            f"seed={int(train.get('seed') or 0)}",
            f"batch={train.get('batch', '')}",
            f"optimizer={train.get('optimizer') or ''}",
            f"prediction_conf={prediction_conf:.4f}",
            f"primary_iou={primary_threshold:.2f}",
            f"score={score:.4f}",
        ]
    )


def fold_identity(candidate: dict[str, Any]) -> str:
    fold_index = candidate.get("fold_index")
    if fold_index is not None:
        return f"fold:{fold_index}"
    return f"manifest:{candidate.get('manifest') or ''}"


def primary_metrics(eval_summary: dict[str, Any]) -> dict[str, Any]:
    primary_threshold = float(eval_summary.get("primary_threshold") or 0.5)
    threshold_key = f"{primary_threshold:.2f}"
    thresholds = eval_summary.get("thresholds") if isinstance(eval_summary.get("thresholds"), dict) else {}
    metrics = thresholds.get(threshold_key) if isinstance(thresholds.get(threshold_key), dict) else {}
    return {
        "min_score": None,
        "primary_threshold": primary_threshold,
        "pred_count": metrics.get("pred_count", 0),
        "precision": metrics.get("precision", 0),
        "recall": metrics.get("recall", 0),
        "f1": metrics.get("f1", 0),
        "missed_question_rate": metrics.get("missed_question_rate", 1),
        "fp_per_image": metrics.get("fp_per_image", 0),
        "duplicate_prediction_count": metrics.get("duplicate_prediction_count", 0),
        "pass_eval_gate": bool(eval_summary.get("pass_eval_gate")),
        "mean_matched_iou": metrics.get("mean_matched_iou", 0),
    }


def metric_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def extract_candidates(manifest_path: Path) -> list[dict[str, Any]]:
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        return []
    eval_summary = manifest.get("eval") if isinstance(manifest.get("eval"), dict) else {}
    if not eval_summary:
        eval_path = manifest_path.parent / "eval" / "summary.json"
        if eval_path.is_file():
            eval_summary = load_json(eval_path)
    if not isinstance(eval_summary, dict) or not eval_summary:
        return []

    train = manifest.get("train") if isinstance(manifest.get("train"), dict) else {}
    prediction = manifest.get("prediction") if isinstance(manifest.get("prediction"), dict) else {}
    dataset_context = dataset_selection_context(manifest, manifest_path)
    fallback_score = metric_float(
        {"value": manifest.get("prediction_conf", prediction.get("effective_conf", prediction.get("requested_conf")))},
        "value",
        0.25,
    )
    score_rows = eval_summary.get("score_sweep") if isinstance(eval_summary.get("score_sweep"), list) else []
    if not score_rows:
        score_rows = [primary_metrics(eval_summary)]
    candidates: list[dict[str, Any]] = []
    for row in score_rows:
        if not isinstance(row, dict):
            continue
        min_score = row.get("min_score")
        try:
            normalized_score = float(min_score) if min_score is not None else fallback_score
        except (TypeError, ValueError):
            normalized_score = fallback_score
        model = str(manifest.get("model") or "")
        name = str(manifest.get("name") or "")
        imgsz = int(train.get("imgsz") or 0)
        primary_threshold = metric_float(row, "primary_threshold", metric_float(eval_summary, "primary_threshold", 0.5))
        candidate_key = candidate_group_key(
            dataset_family=str(dataset_context.get("dataset_family") or ""),
            model=model,
            name=name,
            train=train,
            prediction_conf=fallback_score,
            primary_threshold=primary_threshold,
            score=normalized_score,
        )
        candidates.append(
            {
                "candidate_key": candidate_key,
                "manifest": str(manifest_path),
                "run_dir": str(manifest_path.parent),
                "dataset": manifest.get("dataset") or "",
                "dataset_family": dataset_context.get("dataset_family") or "",
                "fold_index": dataset_context.get("fold_index"),
                "folds": dataset_context.get("folds"),
                "group_field": dataset_context.get("group_field") or "",
                "model": model,
                "name": name,
                "imgsz": imgsz,
                "epochs": int(train.get("epochs") or 0),
                "seed": int(train.get("seed") or 0),
                "batch": train.get("batch", ""),
                "optimizer": train.get("optimizer") or "",
                "best_pt": manifest.get("best_pt") or "",
                "coreml_artifact": manifest.get("coreml_artifact") or "",
                "status": manifest.get("status") or "",
                "min_score": normalized_score,
                "prediction_conf": fallback_score,
                "primary_threshold": primary_threshold,
                "pred_count": metric_float(row, "pred_count"),
                "precision": metric_float(row, "precision"),
                "recall": metric_float(row, "recall"),
                "f1": metric_float(row, "f1"),
                "missed_question_rate": metric_float(row, "missed_question_rate", 1.0),
                "fp_per_image": metric_float(row, "fp_per_image"),
                "duplicate_prediction_count": metric_float(row, "duplicate_prediction_count"),
                "pass_eval_gate": bool(row.get("pass_eval_gate")),
                "mean_matched_iou": metric_float(row, "mean_matched_iou"),
            }
        )
    return candidates


def aggregate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["candidate_key"])].append(candidate)
    rows: list[dict[str, Any]] = []
    for key, items in grouped.items():
        recalls = [metric_float(item, "recall") for item in items]
        precisions = [metric_float(item, "precision") for item in items]
        f1s = [metric_float(item, "f1") for item in items]
        fps = [metric_float(item, "fp_per_image") for item in items]
        pred_counts = [metric_float(item, "pred_count") for item in items]
        missed = [metric_float(item, "missed_question_rate", 1.0) for item in items]
        duplicates = [metric_float(item, "duplicate_prediction_count") for item in items]
        gate_pass_count = sum(1 for item in items if item.get("pass_eval_gate"))
        fold_ids = [fold_identity(item) for item in items]
        unique_fold_ids = sorted(set(fold_ids))
        duplicate_fold_count = max(0, len(fold_ids) - len(unique_fold_ids))
        first = items[0]
        rows.append(
            {
                "candidate_key": key,
                "model": first["model"],
                "name": first["name"],
                "imgsz": first["imgsz"],
                "dataset_family": first.get("dataset_family") or "",
                "epochs": first.get("epochs") or 0,
                "seed": first.get("seed") or 0,
                "prediction_conf": first.get("prediction_conf"),
                "primary_threshold": first.get("primary_threshold"),
                "min_score": first["min_score"],
                "fold_count": len(unique_fold_ids),
                "run_count": len(items),
                "fold_ids": unique_fold_ids,
                "duplicate_fold_count": duplicate_fold_count,
                "gate_pass_count": gate_pass_count,
                "all_folds_pass_eval_gate": duplicate_fold_count == 0 and gate_pass_count == len(items),
                "mean_recall": round(mean(recalls), 4) if recalls else 0,
                "min_recall": round(min(recalls), 4) if recalls else 0,
                "std_recall": round(pstdev(recalls), 4) if len(recalls) > 1 else 0,
                "mean_precision": round(mean(precisions), 4) if precisions else 0,
                "mean_f1": round(mean(f1s), 4) if f1s else 0,
                "mean_pred_count": round(mean(pred_counts), 4) if pred_counts else 0,
                "mean_missed_question_rate": round(mean(missed), 4) if missed else 1,
                "max_missed_question_rate": round(max(missed), 4) if missed else 1,
                "mean_fp_per_image": round(mean(fps), 4) if fps else 0,
                "max_fp_per_image": round(max(fps), 4) if fps else 0,
                "mean_duplicate_prediction_count": round(mean(duplicates), 4) if duplicates else 0,
                "runs": items,
            }
        )
    rows.sort(key=rank_key)
    return rows


def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        not bool(row.get("all_folds_pass_eval_gate")),
        -int(row.get("gate_pass_count") or 0),
        -metric_float(row, "min_recall"),
        -metric_float(row, "mean_recall"),
        metric_float(row, "max_fp_per_image"),
        metric_float(row, "mean_fp_per_image"),
        metric_float(row, "mean_pred_count"),
        -metric_float(row, "mean_precision"),
        -metric_float(row, "mean_f1"),
        metric_float(row, "mean_duplicate_prediction_count"),
        -metric_float(row, "min_score"),
    )


def summarize_ios_recommendation(best: dict[str, Any] | None) -> dict[str, Any]:
    if not best:
        return {}
    score = metric_float(best, "min_score", 0.25)
    return {
        "question_region_detector_fast_min_confidence": round(max(0.01, score), 4),
        "question_region_detector_accurate_min_confidence": round(max(0.01, score), 4),
        "note": "Use the selected score as the VNRecognizedObjectObservation confidence floor, then verify live fallback and latency telemetry before release.",
    }


def select(args: argparse.Namespace) -> dict[str, Any]:
    manifests = discover_manifests(args.inputs)
    candidates: list[dict[str, Any]] = []
    for manifest_path in manifests:
        candidates.extend(extract_candidates(manifest_path))
    aggregated = aggregate_candidates(candidates)
    best = aggregated[0] if aggregated else None
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(path) for path in args.inputs],
        "manifest_count": len(manifests),
        "candidate_count": len(candidates),
        "aggregated_candidate_count": len(aggregated),
        "best": best,
        "ios_recommendation": summarize_ios_recommendation(best),
        "ranked": aggregated[: args.limit],
    }
    write_json(args.out / "selection_summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best question detector model and iOS score threshold.")
    parser.add_argument("inputs", type=Path, nargs="+", help="train_manifest.json files or directories containing them.")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-selection"))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    report = select(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
