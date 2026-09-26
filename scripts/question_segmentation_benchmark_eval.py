"""Evaluate question segmentation outputs on a shared benchmark manifest.

The benchmark manifest is produced by question_detector_review_priority.py. It
pins the image set used for both human review and Codex-style comparison. This
script only evaluates supplied boxes; it does not call any model.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


@dataclass
class BenchImage:
    review_id: str
    row: dict[str, Any]
    width: int
    height: int


@dataclass
class Box:
    review_id: str
    bbox: tuple[float, float, float, float]
    score: float
    source: dict[str, Any]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise SystemExit(f"{path}:{line_number} is not a JSON object")
            rows.append(item)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_key(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip()
    return text.lower()


def alias_keys(row: dict[str, Any]) -> list[str]:
    values: list[str] = []
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    merge_metadata = row.get("merge_metadata") if isinstance(row.get("merge_metadata"), dict) else {}
    for key in (
        "review_id",
        "benchmark_review_id",
        "source_review_id",
        "image_path",
        "image",
        "file_name",
        "filename",
        "source_image",
        "source_image_path",
        "source_candidate",
    ):
        value = str(row.get(key) or "").strip()
        if value:
            values.append(value)
            path = Path(value)
            values.append(path.name)
            values.append(path.stem)
        meta_value = str(metadata.get(key) or merge_metadata.get(key) or "").strip()
        if meta_value:
            values.append(meta_value)
            path = Path(meta_value)
            values.append(path.name)
            values.append(path.stem)
    return [norm_key(value) for value in values if value]


def image_size(row: dict[str, Any]) -> tuple[int, int]:
    width = int(row.get("width") or row.get("image_width") or 0)
    height = int(row.get("height") or row.get("image_height") or 0)
    if width > 0 and height > 0:
        return width, height
    image_path = Path(str(row.get("image_path") or ""))
    if not image_path.is_file():
        raise SystemExit(f"benchmark image is missing and size was not provided: {image_path}")
    with Image.open(image_path) as image:
        normalized = ImageOps.exif_transpose(image)
        return normalized.size


def load_benchmark(path: Path) -> tuple[list[BenchImage], dict[str, str], dict[str, BenchImage]]:
    images: list[BenchImage] = []
    aliases: dict[str, str] = {}
    by_review_id: dict[str, BenchImage] = {}
    for row in read_jsonl(path):
        review_id = str(row.get("review_id") or "").strip()
        if not review_id:
            raise SystemExit(f"benchmark row is missing review_id: {row}")
        width, height = image_size(row)
        image = BenchImage(review_id=review_id, row=row, width=width, height=height)
        images.append(image)
        by_review_id[review_id] = image
        for key in alias_keys(row):
            aliases.setdefault(key, review_id)
    return images, aliases, by_review_id


def resolve_review_id(row: dict[str, Any], aliases: dict[str, str]) -> str | None:
    for key in alias_keys(row):
        review_id = aliases.get(key)
        if review_id:
            return review_id
    return None


def to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def raw_bbox_values(raw: Any) -> tuple[float, float, float, float] | None:
    if isinstance(raw, dict):
        if {"x", "y", "width", "height"}.issubset(raw.keys()):
            values = [raw.get("x"), raw.get("y"), raw.get("width"), raw.get("height")]
        elif {"left", "top", "right", "bottom"}.issubset(raw.keys()):
            left, top, right, bottom = [to_float(raw.get(key)) for key in ("left", "top", "right", "bottom")]
            if None in (left, top, right, bottom):
                return None
            return left, top, right - left, bottom - top
        else:
            return None
    elif isinstance(raw, list) and len(raw) >= 4:
        values = raw[:4]
    else:
        return None
    parsed = [to_float(value) for value in values]
    if any(value is None for value in parsed):
        return None
    x, y, width, height = [float(value) for value in parsed]
    return x, y, width, height


def normalize_bbox(box: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float] | None:
    raw = box.get("bbox_px")
    is_normalized = False
    if raw is None:
        raw = box.get("bbox_norm")
        is_normalized = raw is not None
    if raw is None:
        raw = box.get("bbox") or box.get("box") or box.get("rect")
    parsed = raw_bbox_values(raw)
    if parsed is None:
        nested = box.get("box")
        if isinstance(nested, dict) and nested is not box:
            return normalize_bbox(nested, width, height)
        return None
    x, y, box_width, box_height = parsed
    if is_normalized or (0 <= x <= 1.5 and 0 <= y <= 1.5 and 0 <= box_width <= 1.5 and 0 <= box_height <= 1.5):
        x *= width
        y *= height
        box_width *= width
        box_height *= height
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + box_width))
    y2 = max(0.0, min(float(height), y + box_height))
    out = (x1, y1, x2 - x1, y2 - y1)
    if out[2] <= 1 or out[3] <= 1:
        return None
    return out


def row_box_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("boxes", "regions", "predictions", "detections", "questions"):
        value = row.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if any(key in row for key in ("bbox_px", "bbox_norm", "bbox", "rect")):
        return [row]
    return []


def boxes_from_row(row: dict[str, Any], review_id: str, bench: BenchImage) -> list[Box]:
    boxes: list[Box] = []
    for index, item in enumerate(row_box_items(row)):
        source_item = item.get("box") if isinstance(item.get("box"), dict) else item
        bbox = normalize_bbox(source_item, bench.width, bench.height)
        if bbox is None:
            continue
        score = to_float(source_item.get("score") or source_item.get("confidence"))
        boxes.append(
            Box(
                review_id=review_id,
                bbox=bbox,
                score=1.0 if score is None else score,
                source={"row": row, "box_index": index},
            )
        )
    return boxes


def load_jsonl_boxes(path: Path, aliases: dict[str, str], by_review_id: dict[str, BenchImage]) -> tuple[dict[str, list[Box]], dict[str, Any]]:
    out: dict[str, list[Box]] = defaultdict(list)
    unmatched_rows: list[dict[str, Any]] = []
    rows = read_jsonl(path)
    for row_index, row in enumerate(rows, start=1):
        review_id = resolve_review_id(row, aliases)
        if not review_id:
            unmatched_rows.append(
                {
                    "row_index": row_index,
                    "image": row.get("image") or row.get("image_path") or row.get("source_candidate") or "",
                    "review_id": row.get("review_id") or row.get("benchmark_review_id") or "",
                }
            )
            continue
        bench = by_review_id.get(review_id)
        if bench is None:
            unmatched_rows.append({"row_index": row_index, "review_id": review_id, "reason": "review_id_not_in_benchmark"})
            continue
        out[review_id].extend(boxes_from_row(row, review_id, bench))
    return out, {"path": str(path), "rows": len(rows), "unmatched_rows": unmatched_rows}


def load_priority_boxes(path: Path, by_review_id: dict[str, BenchImage]) -> tuple[dict[str, list[Box]], dict[str, Any]]:
    payload = read_json(path)
    out: dict[str, list[Box]] = defaultdict(list)
    unmatched_rows: list[dict[str, Any]] = []
    rows = payload.get("recommended_images") or []
    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        review_id = str(row.get("review_id") or "").strip()
        bench = by_review_id.get(review_id)
        if not review_id or bench is None:
            unmatched_rows.append({"row_index": row_index, "review_id": review_id, "reason": "review_id_not_in_benchmark"})
            continue
        out[review_id].extend(boxes_from_row(row, review_id, bench))
    return out, {"path": str(path), "rows": len(rows), "unmatched_rows": unmatched_rows}


def load_boxes(
    *,
    jsonl_path: Path | None,
    priority_path: Path | None,
    aliases: dict[str, str],
    by_review_id: dict[str, BenchImage],
) -> tuple[dict[str, list[Box]], dict[str, Any]]:
    if jsonl_path and priority_path:
        raise SystemExit("choose either JSONL or priority JSON for one side, not both")
    if jsonl_path:
        return load_jsonl_boxes(jsonl_path, aliases, by_review_id)
    if priority_path:
        return load_priority_boxes(priority_path, by_review_id)
    raise SystemExit("missing required box input")


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
    return overlap / max(1.0, aw * ah + bw * bh - overlap)


def match_image(reference: list[Box], predictions: list[Box], threshold: float) -> dict[str, Any]:
    pred_sorted = sorted(predictions, key=lambda item: item.score, reverse=True)
    matched_reference: set[int] = set()
    matches: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    for pred_index, pred in enumerate(pred_sorted):
        best_index = None
        best_iou = 0.0
        best_any_iou = 0.0
        for ref_index, ref in enumerate(reference):
            iou = box_iou(ref.bbox, pred.bbox)
            best_any_iou = max(best_any_iou, iou)
            if ref_index not in matched_reference and iou > best_iou:
                best_iou = iou
                best_index = ref_index
        if best_index is not None and best_iou >= threshold:
            matched_reference.add(best_index)
            matches.append(
                {
                    "reference_index": best_index,
                    "prediction_index": pred_index,
                    "iou": round(best_iou, 4),
                    "score": pred.score,
                    "reference_bbox": list(reference[best_index].bbox),
                    "prediction_bbox": list(pred.bbox),
                }
            )
        else:
            false_positives.append(
                {
                    "prediction_index": pred_index,
                    "score": pred.score,
                    "bbox": list(pred.bbox),
                    "best_any_iou": round(best_any_iou, 4),
                }
            )
    missed = [
        {"reference_index": index, "bbox": list(box.bbox)}
        for index, box in enumerate(reference)
        if index not in matched_reference
    ]
    return {"matches": matches, "missed": missed, "false_positives": false_positives}


def add_counts(counter: Counter[str], *, reference: int, predictions: int, matches: int, missed: int, false_positives: int) -> None:
    counter["reference_boxes"] += reference
    counter["prediction_boxes"] += predictions
    counter["matched_boxes"] += matches
    counter["missed_boxes"] += missed
    counter["false_positive_boxes"] += false_positives


def metrics_from_counts(counts: Counter[str]) -> dict[str, Any]:
    reference = counts["reference_boxes"]
    predictions = counts["prediction_boxes"]
    matches = counts["matched_boxes"]
    precision = matches / predictions if predictions else 1.0 if reference == 0 else 0.0
    recall = matches / reference if reference else 1.0 if predictions == 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "reference_boxes": reference,
        "prediction_boxes": predictions,
        "matched_boxes": matches,
        "missed_boxes": counts["missed_boxes"],
        "false_positive_boxes": counts["false_positive_boxes"],
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "count_delta": predictions - reference,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    benchmark, aliases, by_review_id = load_benchmark(args.benchmark_manifest)
    reference, reference_input = load_boxes(
        jsonl_path=args.reference_jsonl,
        priority_path=args.reference_priority_json,
        aliases=aliases,
        by_review_id=by_review_id,
    )
    predictions, prediction_input = load_boxes(
        jsonl_path=args.predictions_jsonl,
        priority_path=args.predictions_priority_json,
        aliases=aliases,
        by_review_id=by_review_id,
    )

    overall: Counter[str] = Counter()
    by_cohort: dict[str, Counter[str]] = defaultdict(Counter)
    per_image: list[dict[str, Any]] = []
    matches_out: list[dict[str, Any]] = []
    missed_out: list[dict[str, Any]] = []
    false_positive_out: list[dict[str, Any]] = []
    for image in benchmark:
        ref_boxes = reference.get(image.review_id, [])
        pred_boxes = predictions.get(image.review_id, [])
        result = match_image(ref_boxes, pred_boxes, args.iou_threshold)
        ref_count = len(ref_boxes)
        pred_count = len(pred_boxes)
        match_count = len(result["matches"])
        missed_count = len(result["missed"])
        fp_count = len(result["false_positives"])
        cohort = str(image.row.get("cohort") or "unknown")
        add_counts(overall, reference=ref_count, predictions=pred_count, matches=match_count, missed=missed_count, false_positives=fp_count)
        add_counts(by_cohort[cohort], reference=ref_count, predictions=pred_count, matches=match_count, missed=missed_count, false_positives=fp_count)
        per_image.append(
            {
                "review_id": image.review_id,
                "cohort": cohort,
                "source_kind": image.row.get("source_kind"),
                "reference_boxes": ref_count,
                "prediction_boxes": pred_count,
                "matched_boxes": match_count,
                "missed_boxes": missed_count,
                "false_positive_boxes": fp_count,
            }
        )
        for row in result["matches"]:
            matches_out.append({"review_id": image.review_id, "cohort": cohort, **row})
        for row in result["missed"]:
            missed_out.append({"review_id": image.review_id, "cohort": cohort, **row})
        for row in result["false_positives"]:
            false_positive_out.append({"review_id": image.review_id, "cohort": cohort, **row})

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "benchmark_manifest": str(args.benchmark_manifest),
        "iou_threshold": args.iou_threshold,
        "image_count": len(benchmark),
        "input_audit": {
            "reference": {
                **reference_input,
                "matched_images": sum(1 for image in benchmark if reference.get(image.review_id)),
                "missing_images": [image.review_id for image in benchmark if not reference.get(image.review_id)],
            },
            "predictions": {
                **prediction_input,
                "matched_images": sum(1 for image in benchmark if predictions.get(image.review_id)),
                "missing_images": [image.review_id for image in benchmark if not predictions.get(image.review_id)],
            },
        },
        "overall": metrics_from_counts(overall),
        "by_cohort": {key: metrics_from_counts(counts) for key, counts in sorted(by_cohort.items())},
    }
    write_json(args.out / "summary.json", summary)
    write_jsonl(args.out / "per_image.jsonl", per_image)
    write_jsonl(args.out / "matches.jsonl", matches_out)
    write_jsonl(args.out / "missed_reference.jsonl", missed_out)
    write_jsonl(args.out / "false_positive_predictions.jsonl", false_positive_out)
    (args.out / "summary.md").write_text(make_markdown(summary), encoding="utf-8")
    return summary


def make_markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# Question Segmentation Benchmark Eval",
        "",
        f"- Generated: {summary['generated_at']}",
        f"- Label: `{summary['label']}`",
        f"- Benchmark manifest: `{summary['benchmark_manifest']}`",
        f"- IoU threshold: {summary['iou_threshold']}",
        f"- Images: {summary['image_count']}",
        f"- Reference matched/missing images: {summary['input_audit']['reference']['matched_images']} / {len(summary['input_audit']['reference']['missing_images'])}",
        f"- Prediction matched/missing images: {summary['input_audit']['predictions']['matched_images']} / {len(summary['input_audit']['predictions']['missing_images'])}",
        "",
        "## Overall",
        "",
        f"- Precision: {overall['precision']}",
        f"- Recall: {overall['recall']}",
        f"- F1: {overall['f1']}",
        f"- Matched/reference/predicted: {overall['matched_boxes']} / {overall['reference_boxes']} / {overall['prediction_boxes']}",
        f"- Missed: {overall['missed_boxes']}",
        f"- False positives: {overall['false_positive_boxes']}",
        "",
        "## By Cohort",
        "",
    ]
    for cohort, metrics in summary["by_cohort"].items():
        lines.append(
            f"- `{cohort}`: P={metrics['precision']} R={metrics['recall']} F1={metrics['f1']} "
            f"matched/ref/pred={metrics['matched_boxes']}/{metrics['reference_boxes']}/{metrics['prediction_boxes']}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Codex or detector segmentation boxes on a shared benchmark manifest.")
    parser.add_argument("--benchmark-manifest", type=Path, required=True, help="codex_benchmark_manifest.jsonl from question_detector_review_priority.py.")
    parser.add_argument("--reference-jsonl", type=Path, help="Human-reviewed reference JSONL, one row per image or box.")
    parser.add_argument("--reference-priority-json", type=Path, help="Draft-only sanity reference from review_priority.json.")
    parser.add_argument("--predictions-jsonl", type=Path, help="Candidate/Codex predictions JSONL, one row per image or box.")
    parser.add_argument("--predictions-priority-json", type=Path, help="Draft-only sanity predictions from review_priority.json.")
    parser.add_argument("--label", default="codex", help="Label shown in reports.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-segmentation-benchmark-eval"))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    summary = evaluate(args)
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
