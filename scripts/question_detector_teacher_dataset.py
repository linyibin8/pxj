"""Export AI-teacher packages directly to YOLO/COCO without weak-label dedupe."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


CATEGORY_ID = 1
CATEGORY_NAME = "question_block"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                item = json.loads(raw)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def stable_id(value: str, length: int = 14) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def split_for_key(key: str, val_ratio: float, test_ratio: float) -> str:
    value = int(stable_id(key, 8), 16) / 0xFFFFFFFF
    if value < test_ratio:
        return "test"
    if value < test_ratio + val_ratio:
        return "val"
    return "train"


def resolve_input(package: Path) -> Path:
    candidates = [
        package / "annotations" / "approved_boxes.jsonl",
        package / "teacher_reference_boxes.jsonl",
        package,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(f"teacher package has no approved boxes jsonl: {package}")


def resolve_image(package: Path, image_value: str) -> Path | None:
    raw = Path(str(image_value or ""))
    if raw.is_absolute() and raw.is_file():
        return raw
    candidates = [
        package / raw,
        package / "images" / raw.name,
        package.parent / raw,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def norm_box(raw: Any, width: int, height: int) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    try:
        x = float(raw.get("x"))
        y = float(raw.get("y"))
        w = float(raw.get("width"))
        h = float(raw.get("height"))
    except (TypeError, ValueError):
        return None
    x1 = max(0.0, min(float(width), x))
    y1 = max(0.0, min(float(height), y))
    x2 = max(0.0, min(float(width), x + w))
    y2 = max(0.0, min(float(height), y + h))
    out = {
        "x": int(round(x1)),
        "y": int(round(y1)),
        "width": int(round(x2 - x1)),
        "height": int(round(y2 - y1)),
    }
    if out["width"] <= 3 or out["height"] <= 3:
        return None
    return out


def copy_image(source: Path, target: Path) -> tuple[int, int]:
    with Image.open(source) as image:
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        width, height = normalized.size
        target.parent.mkdir(parents=True, exist_ok=True)
        normalized.save(target, format="JPEG", quality=90, optimize=True)
        return width, height


def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    stats: Counter[str] = Counter()
    image_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    source_rows = read_jsonl(resolve_input(args.teacher_package))

    for row_index, row in enumerate(source_rows, start=1):
        source = resolve_image(args.teacher_package, str(row.get("image") or ""))
        if source is None:
            source = Path(str(row.get("source_candidate") or ""))
        if not source.is_file():
            stats["missing_image"] += 1
            continue
        review_id = str(row.get("review_id") or source.stem)
        split = split_for_key(review_id, args.val_ratio, args.test_ratio)
        image_key = stable_id(f"{review_id}:{source}")
        target_rel = f"images/{split}/{image_key}_{source.stem[:80]}.jpg"
        width, height = copy_image(source, args.out / target_rel)
        image_id = len(image_rows) + 1
        image_rows.append(
            {
                "id": image_id,
                "file_name": target_rel,
                "width": width,
                "height": height,
                "split": split,
                "review_id": review_id,
                "source_image": str(source),
                "label_count": 0,
            }
        )
        for box_index, box_row in enumerate(row.get("boxes") or [], start=1):
            if not isinstance(box_row, dict):
                continue
            box = norm_box(box_row.get("bbox_px") or box_row.get("bbox"), width, height)
            if box is None:
                stats["bad_box"] += 1
                continue
            flags = [str(flag) for flag in box_row.get("quality_flags") or []]
            if "exclude_from_training" in flags or str(box_row.get("annotation_status")) == "needs_review":
                stats["excluded_box"] += 1
                continue
            label_rows.append(
                {
                    "image_id": image_id,
                    "image": target_rel,
                    "split": split,
                    "bbox_px": box,
                    "question_label": box_row.get("question_label") or box_index,
                    "score": float(box_row.get("score") or 1.0),
                    "quality_flags": flags + ["ai_teacher"],
                    "review_id": review_id,
                    "box_index": box_index,
                }
            )
            image_rows[-1]["label_count"] += 1
            stats["annotations"] += 1
        stats["images"] += 1

    write_yolo(args.out, image_rows, label_rows)
    write_coco(args.out, image_rows, label_rows)
    write_manifests(args.out, image_rows, label_rows)
    write_preview(args.out, image_rows, label_rows)
    audit = audit_summary(args, image_rows, label_rows, stats)
    write_json(args.out / "audit.json", audit)
    write_json(
        args.out / "yolo_dataset.yaml",
        {
            "path": str(args.out.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: CATEGORY_NAME},
        },
    )
    return audit


def write_yolo(out: Path, image_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> None:
    labels_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in label_rows:
        labels_by_image[int(row["image_id"])].append(row)
    for image in image_rows:
        rel = Path(str(image["file_name"]))
        label_path = out / "labels" / str(image["split"]) / f"{rel.stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        width, height = int(image["width"]), int(image["height"])
        for label in labels_by_image.get(int(image["id"]), []):
            box = label["bbox_px"]
            cx = (box["x"] + box["width"] / 2) / width
            cy = (box["y"] + box["height"] / 2) / height
            bw = box["width"] / width
            bh = box["height"] / height
            lines.append(f"0 {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def coco_for_split(image_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]], split: str | None) -> dict[str, Any]:
    images = [row for row in image_rows if split is None or row["split"] == split]
    ids = {int(row["id"]) for row in images}
    annotations: list[dict[str, Any]] = []
    for label in label_rows:
        image_id = int(label["image_id"])
        if image_id not in ids:
            continue
        box = label["bbox_px"]
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": CATEGORY_ID,
                "bbox": [box["x"], box["y"], box["width"], box["height"]],
                "area": box["width"] * box["height"],
                "iscrowd": 0,
                "question_label": label.get("question_label"),
                "quality_flags": label.get("quality_flags") or [],
            }
        )
    return {"images": images, "annotations": annotations, "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME}]}


def write_coco(out: Path, image_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> None:
    for split in ("train", "val", "test"):
        write_json(out / "annotations" / f"coco_{split}.json", coco_for_split(image_rows, label_rows, split))
    write_json(out / "annotations" / "coco_all.json", coco_for_split(image_rows, label_rows, None))


def write_manifests(out: Path, image_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> None:
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    with (out / "annotations" / "image_manifest.jsonl").open("w", encoding="utf-8") as fh:
        for row in image_rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with (out / "annotations" / "manifest.jsonl").open("w", encoding="utf-8") as fh:
        for row in label_rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_preview(out: Path, image_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]], limit: int = 40) -> None:
    labels_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in label_rows:
        labels_by_image[int(row["image_id"])].append(row)
    tiles = []
    for image in image_rows[:limit]:
        image_path = out / str(image["file_name"])
        with Image.open(image_path) as source:
            img = source.convert("RGB")
        scale = min(260 / img.width, 190 / img.height)
        thumb = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(thumb)
        for label in labels_by_image.get(int(image["id"]), []):
            box = label["bbox_px"]
            draw.rectangle(
                [
                    int(box["x"] * scale),
                    int(box["y"] * scale),
                    int((box["x"] + box["width"]) * scale),
                    int((box["y"] + box["height"]) * scale),
                ],
                outline=(0, 190, 80),
                width=2,
            )
        tile = Image.new("RGB", (280, 235), "white")
        tile.paste(thumb, ((280 - thumb.width) // 2, 8))
        ImageDraw.Draw(tile).text((8, 205), f"{image['split']} boxes={image['label_count']}", fill=(0, 0, 0))
        tiles.append(tile)
    if not tiles:
        return
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 280, rows * 235), "white")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % cols) * 280, (index // cols) * 235))
    sheet.save(out / "preview_contact_sheet.jpg", quality=90)


def audit_summary(args: argparse.Namespace, image_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]], stats: Counter[str]) -> dict[str, Any]:
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_images = [row for row in image_rows if row["split"] == split]
        split_labels = [row for row in label_rows if row["split"] == split]
        split_counts[split] = {"images": len(split_images), "annotations": len(split_labels)}
    warnings = []
    if len(image_rows) < args.min_images_ready:
        warnings.append({"code": "too_few_images", "severity": "warning", "detail": f"{len(image_rows)} images; target >= {args.min_images_ready}."})
    if len(label_rows) < args.min_annotations_ready:
        warnings.append({"code": "too_few_annotations", "severity": "warning", "detail": f"{len(label_rows)} annotations; target >= {args.min_annotations_ready}."})
    for split, counts in split_counts.items():
        if counts["images"] == 0 or counts["annotations"] == 0:
            warnings.append({"code": "empty_split", "severity": "error", "detail": f"{split} has {counts['images']} images and {counts['annotations']} annotations."})
    has_error = any(item.get("severity") == "error" for item in warnings)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "teacher_package": str(args.teacher_package),
        "out": str(args.out),
        "counts": {"images": len(image_rows), "annotations": len(label_rows), **dict(stats)},
        "splits": split_counts,
        "readiness": {
            "pilot_eval_ready": len(image_rows) >= 20 and len(label_rows) >= 60 and not has_error,
            "model_training_ready": len(image_rows) >= args.min_images_ready and len(label_rows) >= args.min_annotations_ready and not has_error,
            "has_error": has_error,
        },
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--min-images-ready", type=int, default=300)
    parser.add_argument("--min-annotations-ready", type=int, default=1000)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_dataset(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
