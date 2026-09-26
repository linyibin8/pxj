"""Tune lightweight detector post-processing on held-out predictions.

This is diagnostic only: it checks whether score/shape/NMS/top-k filtering can
turn a noisy detector into a usable on-device candidate before spending more
training time or exporting Core ML.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path
from typing import Any

import question_detector_eval as qeval


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value or "").split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value or "").split(",") if item.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def box_shape(box: qeval.Box, image: dict[str, Any]) -> dict[str, float]:
    _, _, width, height = box.bbox
    image_width = float(image.get("width") or 1)
    image_height = float(image.get("height") or 1)
    area_ratio = max(0.0, width * height / max(1.0, image_width * image_height))
    width_ratio = max(0.0, width / max(1.0, image_width))
    height_ratio = max(0.0, height / max(1.0, image_height))
    aspect = width / max(1.0, height)
    return {
        "area_ratio": area_ratio,
        "width_ratio": width_ratio,
        "height_ratio": height_ratio,
        "aspect": aspect,
    }


def nms(boxes: list[qeval.Box], threshold: float) -> list[qeval.Box]:
    if threshold <= 0 or threshold >= 1:
        return boxes
    kept: list[qeval.Box] = []
    for box in sorted(boxes, key=lambda item: item.score, reverse=True):
        if all(qeval.box_iou(box.bbox, existing.bbox) < threshold for existing in kept):
            kept.append(box)
    return kept


def filter_predictions(
    pred: dict[int, list[qeval.Box]],
    images: dict[int, dict[str, Any]],
    params: dict[str, Any],
) -> dict[int, list[qeval.Box]]:
    filtered: dict[int, list[qeval.Box]] = {}
    for image_id, boxes in pred.items():
        image = images.get(image_id, {})
        kept: list[qeval.Box] = []
        for box in boxes:
            shape = box_shape(box, image)
            if box.score < params["min_score"]:
                continue
            if shape["area_ratio"] < params["min_area_ratio"] or shape["area_ratio"] > params["max_area_ratio"]:
                continue
            if shape["width_ratio"] > params["max_width_ratio"]:
                continue
            if shape["height_ratio"] > params["max_height_ratio"]:
                continue
            if shape["aspect"] < params["min_aspect"] or shape["aspect"] > params["max_aspect"]:
                continue
            kept.append(box)
        kept = nms(kept, params["nms_iou"])
        kept.sort(key=lambda item: item.score, reverse=True)
        if params["top_k"] > 0:
            kept = kept[: params["top_k"]]
        filtered[image_id] = kept
    return filtered


def evaluate_row(
    images: dict[int, dict[str, Any]],
    gt: dict[int, list[qeval.Box]],
    pred: dict[int, list[qeval.Box]],
    params: dict[str, Any],
    thresholds: list[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    filtered = filter_predictions(pred, images, params)
    summary = qeval.evaluate(
        images,
        gt,
        filtered,
        thresholds,
        min_recall=args.min_eval_recall,
        min_precision=args.min_eval_precision,
        max_fp_per_image=args.max_eval_fp_per_image,
        score_thresholds=None,
        include_problem_images=False,
    )
    primary_key = f"{thresholds[0]:.2f}"
    metrics = summary["thresholds"][primary_key]
    return {
        **params,
        "pred_count": metrics["pred_count"],
        "matched": metrics["matched"],
        "missed": metrics["missed"],
        "false_positive": metrics["false_positive"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "fp_per_image": metrics["fp_per_image"],
        "negative_false_positive": metrics["negative_false_positive"],
        "duplicate_prediction_count": metrics["duplicate_prediction_count"],
        "pass_eval_gate": summary["pass_eval_gate"],
    }


def row_sort_key(row: dict[str, Any], target_recall: float) -> tuple[Any, ...]:
    recall = float(row.get("recall") or 0.0)
    precision = float(row.get("precision") or 0.0)
    fp_per_image = float(row.get("fp_per_image") or 999999.0)
    pred_count = int(row.get("pred_count") or 0)
    meets_target = recall >= target_recall
    return (
        not meets_target,
        -recall if not meets_target else fp_per_image,
        fp_per_image if not meets_target else -precision,
        pred_count,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    thresholds = parse_floats(args.thresholds) or [0.5]
    images, gt = qeval.load_ground_truth(args.ground_truth)
    pred = qeval.load_yolo_predictions(args.predictions_yolo_dir, images)

    rows: list[dict[str, Any]] = []
    grid = itertools.product(
        parse_floats(args.min_scores),
        parse_floats(args.min_area_ratios),
        parse_floats(args.max_area_ratios),
        parse_floats(args.max_width_ratios),
        parse_floats(args.max_height_ratios),
        parse_floats(args.min_aspects),
        parse_floats(args.max_aspects),
        parse_floats(args.nms_ious),
        parse_ints(args.top_ks),
    )
    for (
        min_score,
        min_area_ratio,
        max_area_ratio,
        max_width_ratio,
        max_height_ratio,
        min_aspect,
        max_aspect,
        nms_iou,
        top_k,
    ) in grid:
        if min_area_ratio >= max_area_ratio:
            continue
        if min_aspect >= max_aspect:
            continue
        params = {
            "min_score": min_score,
            "min_area_ratio": min_area_ratio,
            "max_area_ratio": max_area_ratio,
            "max_width_ratio": max_width_ratio,
            "max_height_ratio": max_height_ratio,
            "min_aspect": min_aspect,
            "max_aspect": max_aspect,
            "nms_iou": nms_iou,
            "top_k": top_k,
        }
        rows.append(evaluate_row(images, gt, pred, params, thresholds, args))

    rows.sort(key=lambda row: row_sort_key(row, args.target_recall))
    passing = [row for row in rows if row.get("pass_eval_gate")]
    target_recall_rows = [row for row in rows if float(row.get("recall") or 0) >= args.target_recall]
    summary = {
        "ground_truth": str(args.ground_truth),
        "predictions_yolo_dir": str(args.predictions_yolo_dir),
        "thresholds": thresholds,
        "grid_count": len(rows),
        "target_recall": args.target_recall,
        "best": rows[0] if rows else {},
        "best_target_recall": target_recall_rows[0] if target_recall_rows else {},
        "pass_eval_gate_count": len(passing),
        "top_rows": rows[: args.keep_top],
        "settings": {
            "min_scores": parse_floats(args.min_scores),
            "min_area_ratios": parse_floats(args.min_area_ratios),
            "max_area_ratios": parse_floats(args.max_area_ratios),
            "max_width_ratios": parse_floats(args.max_width_ratios),
            "max_height_ratios": parse_floats(args.max_height_ratios),
            "min_aspects": parse_floats(args.min_aspects),
            "max_aspects": parse_floats(args.max_aspects),
            "nms_ious": parse_floats(args.nms_ious),
            "top_ks": parse_ints(args.top_ks),
        },
    }
    write_json(args.out / "summary.json", summary)
    write_jsonl(args.out / "sweep.jsonl", rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep lightweight detector post-processing filters.")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions-yolo-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.5")
    parser.add_argument("--min-eval-recall", type=float, default=0.95)
    parser.add_argument("--min-eval-precision", type=float, default=0.90)
    parser.add_argument("--max-eval-fp-per-image", type=float, default=0.05)
    parser.add_argument("--target-recall", type=float, default=0.90)
    parser.add_argument("--min-scores", default="0.001,0.005,0.01,0.02")
    parser.add_argument("--min-area-ratios", default="0.002")
    parser.add_argument("--max-area-ratios", default="0.12,0.18,0.25,0.35,0.50")
    parser.add_argument("--max-width-ratios", default="0.60,0.80,1.0")
    parser.add_argument("--max-height-ratios", default="0.40,0.60,0.80")
    parser.add_argument("--min-aspects", default="0.25")
    parser.add_argument("--max-aspects", default="2.5,4.0")
    parser.add_argument("--nms-ious", default="0.50")
    parser.add_argument("--top-ks", default="1,2,3,5,0")
    parser.add_argument("--keep-top", type=int, default=50)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "grid_count": summary["grid_count"],
                "best": summary.get("best"),
                "best_target_recall": summary.get("best_target_recall"),
                "pass_eval_gate_count": summary.get("pass_eval_gate_count"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
