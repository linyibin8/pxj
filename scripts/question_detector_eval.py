"""Evaluate question-region detector outputs against COCO ground truth.

The script is model-agnostic: future YOLO/Core ML/heuristic predictions only
need to be exported as COCO detection results or as the dataset JSONL format.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


@dataclass
class Box:
    image_id: int
    bbox: tuple[float, float, float, float]
    score: float
    source: dict[str, Any]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    overlap_w = max(0.0, min(ax2, bx2) - max(ax, bx))
    overlap_h = max(0.0, min(ay2, by2) - max(ay, by))
    overlap = overlap_w * overlap_h
    if overlap <= 0:
        return 0.0
    union = max(1.0, aw * ah + bw * bh - overlap)
    return overlap / union


def normalize_file_name(value: str) -> str:
    return str(value or "").replace("\\", "/").strip()


def load_ground_truth(path: Path) -> tuple[dict[int, dict[str, Any]], dict[int, list[Box]]]:
    coco = load_json(path)
    images = {int(item["id"]): item for item in coco.get("images", [])}
    boxes: dict[int, list[Box]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        image_id = int(ann.get("image_id") or 0)
        if image_id not in images:
            continue
        bbox = ann.get("bbox") or []
        if len(bbox) != 4:
            continue
        boxes[image_id].append(Box(image_id=image_id, bbox=tuple(float(x) for x in bbox), score=1.0, source=ann))
    return images, boxes


def image_lookup(images: dict[int, dict[str, Any]]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for image_id, image in images.items():
        file_name = normalize_file_name(image.get("file_name") or "")
        lookup[file_name] = image_id
        lookup[Path(file_name).name] = image_id
        lookup[Path(file_name).stem] = image_id
    return lookup


def load_coco_predictions(path: Path, images: dict[int, dict[str, Any]]) -> dict[int, list[Box]]:
    payload = load_json(path)
    if isinstance(payload, dict):
        rows = payload.get("annotations") or payload.get("predictions") or payload.get("detections") or []
    else:
        rows = payload
    lookup = image_lookup(images)
    boxes: dict[int, list[Box]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            continue
        image_id = row.get("image_id")
        if image_id is None:
            image_name = normalize_file_name(row.get("image") or row.get("file_name") or row.get("filename") or "")
            image_id = lookup.get(image_name) or lookup.get(Path(image_name).name)
        if image_id is None:
            continue
        image_id = int(image_id)
        bbox = row.get("bbox") or row.get("bbox_px")
        if isinstance(bbox, dict):
            bbox = [bbox.get("x"), bbox.get("y"), bbox.get("width"), bbox.get("height")]
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        score = float(row.get("score", row.get("confidence", 1.0)) or 0.0)
        boxes[image_id].append(Box(image_id=image_id, bbox=tuple(float(x) for x in bbox), score=score, source=row))
    return boxes


def load_jsonl_predictions(path: Path, images: dict[int, dict[str, Any]]) -> dict[int, list[Box]]:
    lookup = image_lookup(images)
    boxes: dict[int, list[Box]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_name = normalize_file_name(row.get("image") or row.get("file_name") or row.get("filename") or "")
            image_id = lookup.get(image_name) or lookup.get(Path(image_name).name)
            if image_id is None:
                continue
            bbox = row.get("bbox_px") or row.get("bbox")
            if isinstance(bbox, dict):
                bbox = [bbox.get("x"), bbox.get("y"), bbox.get("width"), bbox.get("height")]
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            score = float(row.get("score", row.get("confidence", 1.0)) or 0.0)
            boxes[image_id].append(Box(image_id=image_id, bbox=tuple(float(x) for x in bbox), score=score, source=row))
    return boxes


def load_yolo_predictions(labels_dir: Path, images: dict[int, dict[str, Any]]) -> dict[int, list[Box]]:
    lookup = image_lookup(images)
    boxes: dict[int, list[Box]] = defaultdict(list)
    for label_path in sorted(labels_dir.rglob("*.txt")):
        image_id = lookup.get(label_path.stem)
        if image_id is None:
            continue
        image = images[image_id]
        image_width = float(image.get("width") or 1)
        image_height = float(image.get("height") or 1)
        for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            try:
                _class_id = int(float(parts[0]))
                cx, cy, width, height = [float(value) for value in parts[1:5]]
                score = float(parts[5]) if len(parts) >= 6 else 1.0
            except ValueError:
                continue
            px_width = width * image_width
            px_height = height * image_height
            x = (cx * image_width) - px_width / 2
            y = (cy * image_height) - px_height / 2
            boxes[image_id].append(
                Box(
                    image_id=image_id,
                    bbox=(x, y, px_width, px_height),
                    score=score,
                    source={"file": str(label_path), "line": line_number, "format": "yolo"},
                )
            )
    return boxes


def load_predictions(args: argparse.Namespace, images: dict[int, dict[str, Any]], gt: dict[int, list[Box]]) -> dict[int, list[Box]]:
    if args.predictions_jsonl:
        return load_jsonl_predictions(args.predictions_jsonl, images)
    if args.predictions_yolo_dir:
        return load_yolo_predictions(args.predictions_yolo_dir, images)
    if args.predictions:
        return load_coco_predictions(args.predictions, images)
    return {
        image_id: [
            Box(image_id=image_id, bbox=box.bbox, score=1.0, source={"source": "ground_truth_sanity"})
            for box in boxes
        ]
        for image_id, boxes in gt.items()
    }


def match_image(gt_boxes: list[Box], pred_boxes: list[Box], threshold: float) -> dict[str, Any]:
    predictions = sorted(pred_boxes, key=lambda box: box.score, reverse=True)
    matched_gt: set[int] = set()
    matches: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    for pred_index, pred in enumerate(predictions):
        best_index = None
        best_iou = 0.0
        best_any_index = None
        best_any_iou = 0.0
        for gt_index, gt in enumerate(gt_boxes):
            iou = box_iou(gt.bbox, pred.bbox)
            if iou > best_any_iou:
                best_any_iou = iou
                best_any_index = gt_index
        for gt_index, gt in enumerate(gt_boxes):
            if gt_index in matched_gt:
                continue
            iou = box_iou(gt.bbox, pred.bbox)
            if iou > best_iou:
                best_iou = iou
                best_index = gt_index
        if best_index is not None and best_iou >= threshold:
            matched_gt.add(best_index)
            matches.append(
                {
                    "gt_index": best_index,
                    "pred_index": pred_index,
                    "iou": best_iou,
                    "score": pred.score,
                    "gt_bbox": list(gt_boxes[best_index].bbox),
                    "pred_bbox": list(pred.bbox),
                }
            )
        else:
            false_positives.append(
                {
                    "pred_index": pred_index,
                    "score": pred.score,
                    "bbox": list(pred.bbox),
                    "best_iou": best_iou,
                    "best_any_iou": best_any_iou,
                    "best_any_gt_index": best_any_index,
                }
            )
    missed = [
        {"gt_index": index, "bbox": list(box.bbox), "question_key": box.source.get("question_key") or ""}
        for index, box in enumerate(gt_boxes)
        if index not in matched_gt
    ]
    return {"matches": matches, "false_positives": false_positives, "missed": missed}


def area_bucket(box: Box, image: dict[str, Any]) -> str:
    image_area = max(1.0, float(image.get("width") or 1) * float(image.get("height") or 1))
    _, _, width, height = box.bbox
    ratio = max(0.0, width * height / image_area)
    if ratio < 0.04:
        return "small"
    if ratio < 0.18:
        return "medium"
    if ratio < 0.45:
        return "large"
    return "near_full_page"


def confidence_bucket(box: Box) -> str:
    value = box.source.get("confidence")
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if confidence < 0.35:
        return "low"
    if confidence < 0.7:
        return "medium"
    return "high"


def empty_positive_stats() -> dict[str, int]:
    return {"gt_count": 0, "matched": 0, "missed": 0}


def empty_negative_stats() -> dict[str, int]:
    return {"images": 0, "false_positive": 0, "images_with_false_positive": 0}


def finalize_positive_strata(stats: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, value in sorted(stats.items()):
        gt_count = int(value.get("gt_count") or 0)
        matched = int(value.get("matched") or 0)
        missed = int(value.get("missed") or 0)
        result[key] = {
            "gt_count": gt_count,
            "matched": matched,
            "missed": missed,
            "recall": round(matched / max(1, gt_count), 4),
            "missed_question_rate": round(missed / max(1, gt_count), 4),
        }
    return result


def finalize_negative_strata(stats: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, value in sorted(stats.items()):
        images_count = int(value.get("images") or 0)
        false_positive = int(value.get("false_positive") or 0)
        fp_images = int(value.get("images_with_false_positive") or 0)
        result[key] = {
            "images": images_count,
            "false_positive": false_positive,
            "images_with_false_positive": fp_images,
            "fp_per_image": round(false_positive / max(1, images_count), 4),
            "image_fp_rate": round(fp_images / max(1, images_count), 4),
        }
    return result


def filter_predictions_by_score(pred: dict[int, list[Box]], min_score: float) -> dict[int, list[Box]]:
    return {
        image_id: [box for box in boxes if box.score >= min_score]
        for image_id, boxes in pred.items()
    }


def compact_threshold_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    thresholds = summary.get("thresholds") if isinstance(summary.get("thresholds"), dict) else {}
    compact: dict[str, Any] = {}
    for threshold_key, metrics in thresholds.items():
        if not isinstance(metrics, dict):
            continue
        compact[threshold_key] = {
            key: value
            for key, value in metrics.items()
            if key not in {"problem_images", "strata"}
        }
    return compact


def evaluate(
    images: dict[int, dict[str, Any]],
    gt: dict[int, list[Box]],
    pred: dict[int, list[Box]],
    thresholds: list[float],
    min_recall: float,
    min_precision: float,
    max_fp_per_image: float,
    score_thresholds: list[float] | None = None,
    include_problem_images: bool = True,
) -> dict[str, Any]:
    per_threshold: dict[str, Any] = {}
    for threshold in thresholds:
        details: dict[str, Any] = {}
        matched = 0
        false_positive = 0
        missed = 0
        images_with_missed_count = 0
        images_with_false_positive_count = 0
        negative_image_count = 0
        negative_image_false_positive_count = 0
        negative_false_positive = 0
        duplicate_prediction_count = 0
        ious: list[float] = []
        quality_stats: dict[str, dict[str, int]] = defaultdict(empty_positive_stats)
        area_stats: dict[str, dict[str, int]] = defaultdict(empty_positive_stats)
        origin_stats: dict[str, dict[str, int]] = defaultdict(empty_positive_stats)
        confidence_stats: dict[str, dict[str, int]] = defaultdict(empty_positive_stats)
        negative_kind_stats: dict[str, dict[str, int]] = defaultdict(empty_negative_stats)
        for image_id, image in images.items():
            gt_boxes = gt.get(image_id, [])
            pred_boxes = pred.get(image_id, [])
            result = match_image(gt_boxes, pred_boxes, threshold)
            matched += len(result["matches"])
            false_positive += len(result["false_positives"])
            missed += len(result["missed"])
            ious.extend(item["iou"] for item in result["matches"])
            if result["missed"]:
                images_with_missed_count += 1
            if result["false_positives"]:
                images_with_false_positive_count += 1
            duplicate_prediction_count += sum(1 for item in result["false_positives"] if float(item.get("best_any_iou") or 0) >= threshold)
            matched_gt = {int(item["gt_index"]) for item in result["matches"]}
            for gt_index, box in enumerate(gt_boxes):
                is_matched = gt_index in matched_gt
                flags = box.source.get("quality_flags") if isinstance(box.source.get("quality_flags"), list) else []
                if not flags:
                    flags = ["clean"]
                for flag in flags:
                    bucket = quality_stats[str(flag)]
                    bucket["gt_count"] += 1
                    bucket["matched"] += 1 if is_matched else 0
                    bucket["missed"] += 0 if is_matched else 1
                for stats, key in (
                    (area_stats, area_bucket(box, image)),
                    (origin_stats, str(box.source.get("origin") or image.get("origin") or "unknown")),
                    (confidence_stats, confidence_bucket(box)),
                ):
                    bucket = stats[key]
                    bucket["gt_count"] += 1
                    bucket["matched"] += 1 if is_matched else 0
                    bucket["missed"] += 0 if is_matched else 1
            if not gt_boxes:
                negative_image_count += 1
                negative_kind = str(image.get("negative_kind") or ("negative" if image.get("is_negative") else "unlabeled_no_gt"))
                bucket = negative_kind_stats[negative_kind]
                bucket["images"] += 1
                bucket["false_positive"] += len(result["false_positives"])
                if result["false_positives"]:
                    bucket["images_with_false_positive"] += 1
                    negative_image_false_positive_count += 1
                    negative_false_positive += len(result["false_positives"])
            if result["false_positives"] or result["missed"]:
                details[str(image_id)] = {
                    "file_name": image.get("file_name") or "",
                    "is_negative": bool(image.get("is_negative")),
                    "negative_kind": image.get("negative_kind") or "",
                    **result,
                }
        gt_count = sum(len(items) for items in gt.values())
        pred_count = sum(len(items) for items in pred.values())
        image_count = len(images)
        images_with_gt_count = sum(1 for image_id in images if gt.get(image_id))
        images_with_predictions_count = sum(1 for image_id in images if pred.get(image_id))
        precision = matched / max(1, matched + false_positive)
        recall = matched / max(1, gt_count)
        f1 = (2 * precision * recall / max(0.000001, precision + recall)) if precision + recall else 0.0
        per_threshold[f"{threshold:.2f}"] = {
            "image_count": image_count,
            "images_with_gt_count": images_with_gt_count,
            "negative_image_count": negative_image_count,
            "images_with_predictions_count": images_with_predictions_count,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "matched": matched,
            "missed": missed,
            "false_positive": false_positive,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "mean_matched_iou": round(sum(ious) / len(ious), 4) if ious else 0,
            "missed_question_rate": round(missed / max(1, gt_count), 4),
            "fp_per_image": round(false_positive / max(1, image_count), 4),
            "pred_per_gt": round(pred_count / max(1, gt_count), 4),
            "problem_image_count": len(details),
            "problem_image_rate": round(len(details) / max(1, image_count), 4),
            "images_with_missed_count": images_with_missed_count,
            "images_with_false_positive_count": images_with_false_positive_count,
            "negative_image_false_positive_count": negative_image_false_positive_count,
            "negative_false_positive": negative_false_positive,
            "duplicate_prediction_count": duplicate_prediction_count,
            "strata": {
                "quality_flags": finalize_positive_strata(quality_stats),
                "area_buckets": finalize_positive_strata(area_stats),
                "origins": finalize_positive_strata(origin_stats),
                "confidence_buckets": finalize_positive_strata(confidence_stats),
                "negative_kinds": finalize_negative_strata(negative_kind_stats),
            },
        }
        if include_problem_images:
            per_threshold[f"{threshold:.2f}"]["problem_images"] = details
    primary = per_threshold[f"{thresholds[0]:.2f}"]
    recall_passed = primary["recall"] >= min_recall
    precision_passed = primary["precision"] >= min_precision
    fp_passed = primary["fp_per_image"] <= max_fp_per_image
    gate = {
        "passed": bool(recall_passed and precision_passed and fp_passed),
        "primary_threshold": thresholds[0],
        "min_recall": min_recall,
        "min_precision": min_precision,
        "max_fp_per_image": max_fp_per_image,
        "checks": {
            "recall": {"actual": primary["recall"], "passed": recall_passed},
            "precision": {"actual": primary["precision"], "passed": precision_passed},
            "fp_per_image": {"actual": primary["fp_per_image"], "passed": fp_passed},
        },
    }
    summary = {
        "primary_threshold": thresholds[0],
        "thresholds": per_threshold,
        "pass_recall_95": primary["recall"] >= 0.95,
        "pass_precision_90": primary["precision"] >= 0.90,
        "pass_eval_gate": gate["passed"],
        "gate": gate,
    }
    if score_thresholds:
        sweep: list[dict[str, Any]] = []
        for min_score in sorted(set(score_thresholds)):
            filtered = filter_predictions_by_score(pred, min_score)
            filtered_summary = evaluate(
                images,
                gt,
                filtered,
                thresholds,
                min_recall=min_recall,
                min_precision=min_precision,
                max_fp_per_image=max_fp_per_image,
                score_thresholds=None,
                include_problem_images=False,
            )
            primary_key = f"{thresholds[0]:.2f}"
            primary_metrics = filtered_summary["thresholds"][primary_key]
            sweep.append(
                {
                    "min_score": min_score,
                    "primary_threshold": thresholds[0],
                    "pred_count": primary_metrics["pred_count"],
                    "precision": primary_metrics["precision"],
                    "recall": primary_metrics["recall"],
                    "f1": primary_metrics["f1"],
                    "missed_question_rate": primary_metrics["missed_question_rate"],
                    "fp_per_image": primary_metrics["fp_per_image"],
                    "duplicate_prediction_count": primary_metrics["duplicate_prediction_count"],
                    "pass_eval_gate": filtered_summary["pass_eval_gate"],
                    "gate": filtered_summary["gate"],
                    "thresholds": compact_threshold_metrics(filtered_summary),
                }
            )
        summary["score_sweep"] = sweep
    return summary


def draw_box(draw: ImageDraw.ImageDraw, bbox: tuple[float, float, float, float], color: tuple[int, int, int], width: int = 3) -> None:
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)


def build_error_sheet(
    images: dict[int, dict[str, Any]],
    gt: dict[int, list[Box]],
    pred: dict[int, list[Box]],
    summary: dict[str, Any],
    image_root: Path,
    out_path: Path,
    limit: int = 40,
) -> None:
    threshold_key = f"{summary['primary_threshold']:.2f}"
    problem_images = summary["thresholds"][threshold_key]["problem_images"]
    if not problem_images:
        return
    tiles: list[Image.Image] = []
    for key in list(problem_images.keys())[:limit]:
        image_id = int(key)
        image = images[image_id]
        source = image_root / normalize_file_name(image.get("file_name") or "")
        if not source.is_file():
            continue
        with Image.open(source) as raw:
            canvas = ImageOps.exif_transpose(raw).convert("RGB")
        scale = min(260 / canvas.width, 190 / canvas.height)
        canvas = canvas.resize((max(1, int(canvas.width * scale)), max(1, int(canvas.height * scale))), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(canvas)
        for box in gt.get(image_id, []):
            draw_box(draw, tuple(value * scale for value in box.bbox), (40, 180, 70), width=2)
        for box in pred.get(image_id, []):
            draw_box(draw, tuple(value * scale for value in box.bbox), (220, 60, 60), width=2)
        tile = Image.new("RGB", (280, 235), "white")
        tile.paste(canvas, ((280 - canvas.width) // 2, 8))
        text = f"miss={len(problem_images[key]['missed'])} fp={len(problem_images[key]['false_positives'])}"
        ImageDraw.Draw(tile).text((8, 205), text, fill=(0, 0, 0))
        tiles.append(tile)
    if not tiles:
        return
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 280, rows * 235), (245, 245, 245))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % cols) * 280, (index // cols) * 235))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate question detector predictions.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="COCO ground-truth annotation file.")
    parser.add_argument("--predictions", type=Path, default=None, help="COCO detection results JSON. If omitted, uses ground truth as a sanity oracle.")
    parser.add_argument("--predictions-jsonl", type=Path, default=None, help="Dataset manifest JSONL predictions with image + bbox_px.")
    parser.add_argument("--predictions-yolo-dir", type=Path, default=None, help="YOLO prediction label directory with class cx cy w h [score] rows.")
    parser.add_argument("--image-root", type=Path, default=None, help="Dataset root for drawing error contact sheet.")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-eval"))
    parser.add_argument("--thresholds", default="0.5,0.75", help="Comma-separated IoU thresholds.")
    parser.add_argument("--min-recall", type=float, default=0.95, help="Primary-threshold recall required for pass_eval_gate.")
    parser.add_argument("--min-precision", type=float, default=0.90, help="Primary-threshold precision required for pass_eval_gate.")
    parser.add_argument("--max-fp-per-image", type=float, default=0.05, help="Maximum primary-threshold false positives per image for pass_eval_gate.")
    parser.add_argument("--score-thresholds", default="", help="Optional comma-separated confidence thresholds for PR/latency tuning, e.g. 0.05,0.1,0.25,0.4.")
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before writing eval results.")
    args = parser.parse_args()

    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    if not thresholds:
        thresholds = [0.5]
    score_thresholds = [float(item.strip()) for item in args.score_thresholds.split(",") if item.strip()]
    images, gt = load_ground_truth(args.ground_truth)
    pred = load_predictions(args, images, gt)
    summary = evaluate(
        images,
        gt,
        pred,
        thresholds,
        min_recall=args.min_recall,
        min_precision=args.min_precision,
        max_fp_per_image=args.max_fp_per_image,
        score_thresholds=score_thresholds,
    )
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "summary.json", summary)
    if args.image_root:
        build_error_sheet(images, gt, pred, summary, args.image_root, args.out / "errors_contact_sheet.jpg")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
