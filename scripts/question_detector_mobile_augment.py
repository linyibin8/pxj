"""Create mobile-capture augmentations for reviewed question detector datasets.

The output keeps the existing dataset contract: images/{split}, labels/{split},
annotations/manifest.jsonl, annotations/image_manifest.jsonl, COCO exports, and
yolo_dataset.yaml. Augmentation is for robustness, not for faking data scale:
the exported audit only becomes model_training_ready when the source dataset was
already model_training_ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


CATEGORY_NAME = "question_block"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stable_id(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def image_path_for_row(dataset_root: Path, row: dict[str, Any]) -> Path:
    image = Path(str(row.get("image") or ""))
    if image.is_absolute():
        return image
    return dataset_root / image


def normalize_image(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def bbox_norm_to_corners(bbox: dict[str, Any], width: int, height: int) -> list[tuple[float, float]]:
    x = float(bbox.get("x") or 0.0) * width
    y = float(bbox.get("y") or 0.0) * height
    w = float(bbox.get("width") or 0.0) * width
    h = float(bbox.get("height") or 0.0) * height
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def clamp_bbox(x: float, y: float, width: float, height: float, image_width: int, image_height: int) -> dict[str, float] | None:
    x1 = max(0.0, min(float(image_width), x))
    y1 = max(0.0, min(float(image_height), y))
    x2 = max(0.0, min(float(image_width), x + width))
    y2 = max(0.0, min(float(image_height), y + height))
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w < 4 or clipped_h < 4:
        return None
    return {"x": x1, "y": y1, "width": clipped_w, "height": clipped_h}


def corners_to_bbox(corners: list[tuple[float, float]], image_width: int, image_height: int) -> dict[str, float] | None:
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return clamp_bbox(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), image_width, image_height)


def crop_transform_points(
    corners: list[tuple[float, float]],
    crop_left: float,
    crop_top: float,
    crop_width: float,
    crop_height: float,
    out_width: int,
    out_height: int,
) -> list[tuple[float, float]]:
    return [
        ((x - crop_left) * out_width / crop_width, (y - crop_top) * out_height / crop_height)
        for x, y in corners
    ]


def rotate_points(corners: list[tuple[float, float]], angle_degrees: float, width: int, height: int) -> list[tuple[float, float]]:
    radians = math.radians(angle_degrees)
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    center_x = width / 2.0
    center_y = height / 2.0
    rotated: list[tuple[float, float]] = []
    for x, y in corners:
        dx = x - center_x
        dy = y - center_y
        rotated.append((center_x + cos_a * dx - sin_a * dy, center_y + sin_a * dx + cos_a * dy))
    return rotated


def bbox_area(box: dict[str, float]) -> float:
    return max(0.0, float(box.get("width") or 0.0)) * max(0.0, float(box.get("height") or 0.0))


def transformed_label(
    row: dict[str, Any],
    width: int,
    height: int,
    params: dict[str, float],
    min_area_ratio: float,
) -> dict[str, Any] | None:
    bbox_norm = row.get("bbox_norm") if isinstance(row.get("bbox_norm"), dict) else {}
    original_px = {
        "x": float(bbox_norm.get("x") or 0.0) * width,
        "y": float(bbox_norm.get("y") or 0.0) * height,
        "width": float(bbox_norm.get("width") or 0.0) * width,
        "height": float(bbox_norm.get("height") or 0.0) * height,
    }
    original_area = max(1.0, bbox_area(original_px))
    corners = bbox_norm_to_corners(bbox_norm, width, height)
    corners = crop_transform_points(
        corners,
        params["crop_left"],
        params["crop_top"],
        params["crop_width"],
        params["crop_height"],
        width,
        height,
    )
    corners = rotate_points(corners, params["angle"], width, height)
    clipped = corners_to_bbox(corners, width, height)
    if clipped is None or bbox_area(clipped) / original_area < min_area_ratio:
        return None
    bbox_px = {
        "x": int(round(clipped["x"])),
        "y": int(round(clipped["y"])),
        "width": int(round(clipped["width"])),
        "height": int(round(clipped["height"])),
    }
    if bbox_px["width"] <= 3 or bbox_px["height"] <= 3:
        return None
    bbox_norm_out = {
        "x": bbox_px["x"] / width,
        "y": bbox_px["y"] / height,
        "width": bbox_px["width"] / width,
        "height": bbox_px["height"] / height,
    }
    out = dict(row)
    out["bbox_px"] = bbox_px
    out["bbox_norm"] = bbox_norm_out
    flags = list(dict.fromkeys([str(flag) for flag in row.get("quality_flags") or []] + params["quality_flags"]))
    out["quality_flags"] = flags
    return out


def random_params(rng: random.Random, width: int, height: int, max_crop_fraction: float) -> dict[str, Any]:
    crop_left = rng.uniform(0.0, max_crop_fraction) * width
    crop_right = rng.uniform(0.0, max_crop_fraction) * width
    crop_top = rng.uniform(0.0, max_crop_fraction) * height
    crop_bottom = rng.uniform(0.0, max_crop_fraction) * height
    crop_width = max(1.0, width - crop_left - crop_right)
    crop_height = max(1.0, height - crop_top - crop_bottom)
    blur_radius = rng.choice([0.0, 0.0, 0.15, 0.3])
    downscale = rng.choice([1.0, 1.0, rng.uniform(0.68, 0.92)])
    params = {
        "angle": rng.uniform(-2.8, 2.8),
        "crop_left": crop_left,
        "crop_top": crop_top,
        "crop_width": crop_width,
        "crop_height": crop_height,
        "brightness": rng.uniform(0.82, 1.18),
        "contrast": rng.uniform(0.86, 1.22),
        "blur_radius": blur_radius,
        "downscale": downscale,
        "jpeg_quality": int(rng.randint(62, 88)),
    }
    flags = ["augmented", "aug:mobile_capture", "aug:crop_resize", "aug:rotate", "aug:exposure"]
    if blur_radius > 0:
        flags.append("aug:slight_blur")
    if downscale < 1:
        flags.append("aug:resample")
    params["quality_flags"] = flags
    return params


def apply_image_transform(image: Image.Image, params: dict[str, Any]) -> Image.Image:
    width, height = image.size
    left = int(round(params["crop_left"]))
    top = int(round(params["crop_top"]))
    right = int(round(params["crop_left"] + params["crop_width"]))
    bottom = int(round(params["crop_top"] + params["crop_height"]))
    transformed = image.crop((left, top, right, bottom)).resize((width, height), Image.Resampling.BICUBIC)
    transformed = transformed.rotate(float(params["angle"]), resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(255, 255, 255))
    transformed = ImageEnhance.Brightness(transformed).enhance(float(params["brightness"]))
    transformed = ImageEnhance.Contrast(transformed).enhance(float(params["contrast"]))
    if float(params["downscale"]) < 0.999:
        scale = float(params["downscale"])
        small_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        transformed = transformed.resize(small_size, Image.Resampling.BILINEAR).resize((width, height), Image.Resampling.BICUBIC)
    if float(params["blur_radius"]) > 0:
        transformed = transformed.filter(ImageFilter.GaussianBlur(float(params["blur_radius"])))
    return transformed


def save_jpeg(image: Image.Image, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=quality, optimize=True)


def copy_or_normalize_image(source: Path, target: Path) -> tuple[int, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        normalized = normalize_image(image)
        width, height = normalized.size
        save_jpeg(normalized, target.with_suffix(".jpg"), 88)
    return width, height


def label_with_image(
    row: dict[str, Any],
    image_rel: str,
    split: str,
    source_key: str,
    variant_id: str,
) -> dict[str, Any]:
    out = dict(row)
    out["image"] = image_rel
    out["split"] = split
    out["question_key"] = f"{row.get('question_key') or source_key}:{variant_id}"
    out["source_key"] = source_key
    out["origin"] = f"{row.get('origin') or 'source'}|mobile_augment"
    return out


def yolo_line(row: dict[str, Any]) -> str:
    box = row.get("bbox_norm") if isinstance(row.get("bbox_norm"), dict) else {}
    x = float(box.get("x") or 0.0)
    y = float(box.get("y") or 0.0)
    w = float(box.get("width") or 0.0)
    h = float(box.get("height") or 0.0)
    cx = x + w / 2.0
    cy = y + h / 2.0
    return f"0 {cx:.8f} {cy:.8f} {w:.8f} {h:.8f}"


def write_yolo(root: Path, image_rows: list[dict[str, Any]], labels_by_image: dict[str, list[dict[str, Any]]]) -> None:
    for image_row in image_rows:
        image_rel = str(image_row["image"])
        split = str(image_row["split"])
        label_path = root / "labels" / split / f"{Path(image_rel).stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        rows = labels_by_image.get(image_rel, [])
        label_path.write_text("\n".join(yolo_line(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    write_json(
        root / "yolo_dataset.yaml",
        {
            "path": str(root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {"0": CATEGORY_NAME},
        },
    )


def coco_for_split(image_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]], split: str | None) -> dict[str, Any]:
    selected_images = [row for row in image_rows if split is None or row.get("split") == split]
    image_id_by_rel = {str(row["image"]): index + 1 for index, row in enumerate(selected_images)}
    images = [
        {
            "id": image_id_by_rel[str(row["image"])],
            "file_name": str(row["image"]),
            "width": int(row.get("width") or 0),
            "height": int(row.get("height") or 0),
        }
        for row in selected_images
    ]
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    for row in label_rows:
        image_rel = str(row.get("image") or "")
        if image_rel not in image_id_by_rel:
            continue
        box = row.get("bbox_px") if isinstance(row.get("bbox_px"), dict) else {}
        x = float(box.get("x") or 0.0)
        y = float(box.get("y") or 0.0)
        width = float(box.get("width") or 0.0)
        height = float(box.get("height") or 0.0)
        if width <= 0 or height <= 0:
            continue
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id_by_rel[image_rel],
                "category_id": 1,
                "bbox": [round(x, 3), round(y, 3), round(width, 3), round(height, 3)],
                "area": round(width * height, 3),
                "iscrowd": 0,
            }
        )
        ann_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": CATEGORY_NAME}],
    }


def write_coco(root: Path, image_rows: list[dict[str, Any]], label_rows: list[dict[str, Any]]) -> None:
    for split in ("train", "val", "test"):
        write_json(root / "annotations" / f"coco_{split}.json", coco_for_split(image_rows, label_rows, split))
    write_json(root / "annotations" / "coco_all.json", coco_for_split(image_rows, label_rows, None))


def build_preview(root: Path, image_rows: list[dict[str, Any]], labels_by_image: dict[str, list[dict[str, Any]]], limit: int = 48) -> None:
    positives = [row for row in image_rows if labels_by_image.get(str(row["image"]))]
    if not positives:
        return
    rows = positives[:limit]
    thumb_w = 320
    thumb_h = 220
    columns = 4
    sheet = Image.new("RGB", (columns * thumb_w, math.ceil(len(rows) / columns) * thumb_h), "white")
    for index, row in enumerate(rows):
        path = root / str(row["image"])
        try:
            with Image.open(path) as image:
                preview = normalize_image(image)
        except Exception:
            continue
        original_w, original_h = preview.size
        preview.thumbnail((thumb_w, thumb_h - 22), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
        offset_x = (thumb_w - preview.width) // 2
        offset_y = 18
        canvas.paste(preview, (offset_x, offset_y))
        draw = ImageDraw.Draw(canvas)
        scale_x = preview.width / max(1, original_w)
        scale_y = preview.height / max(1, original_h)
        for label in labels_by_image.get(str(row["image"]), []):
            box = label.get("bbox_px") if isinstance(label.get("bbox_px"), dict) else {}
            x = offset_x + float(box.get("x") or 0.0) * scale_x
            y = offset_y + float(box.get("y") or 0.0) * scale_y
            w = float(box.get("width") or 0.0) * scale_x
            h = float(box.get("height") or 0.0) * scale_y
            draw.rectangle((x, y, x + w, y + h), outline=(229, 57, 53), width=2)
        draw.text((6, 4), f"{row.get('split')} {Path(str(row['image'])).name[:34]}", fill=(20, 20, 20))
        sheet.paste(canvas, ((index % columns) * thumb_w, (index // columns) * thumb_h))
    save_jpeg(sheet, root / "preview_contact_sheet.jpg", 90)


def source_audit(root: Path) -> dict[str, Any]:
    audit_path = root / "audit.json"
    if not audit_path.is_file():
        return {}
    loaded = read_json(audit_path)
    return loaded if isinstance(loaded, dict) else {}


def build_audit(
    source_root: Path,
    source_image_count: int,
    image_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    skipped_variants: int,
) -> dict[str, Any]:
    audit = source_audit(source_root)
    readiness = audit.get("readiness") if isinstance(audit.get("readiness"), dict) else {}
    source_ready = bool(readiness.get("model_training_ready"))
    split_counts: dict[str, dict[str, int]] = {}
    labels_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in label_rows:
        labels_by_image[str(label.get("image") or "")].append(label)
    for split in ("train", "val", "test"):
        images = [row for row in image_rows if row.get("split") == split]
        split_labels = [row for row in label_rows if row.get("split") == split]
        split_counts[split] = {
            "images": len(images),
            "positive_images": sum(1 for row in images if labels_by_image.get(str(row["image"]))),
            "negative_images": sum(1 for row in images if not labels_by_image.get(str(row["image"]))),
            "annotations": len(split_labels),
            "groups": len({str(row.get("split_key") or row.get("source_key") or row.get("image")) for row in images}),
        }
    warnings: list[dict[str, Any]] = []
    if not source_ready:
        warnings.append(
            {
                "code": "source_dataset_not_model_ready",
                "severity": "warning",
                "detail": "Augmentation output inherits model_training_ready=false because the source dataset was not model_training_ready.",
            }
        )
    if skipped_variants:
        warnings.append(
            {
                "code": "skipped_clipped_variants",
                "severity": "warning",
                "detail": f"{skipped_variants} augmented positive variants were skipped because one or more boxes were clipped too much.",
            }
        )
    warnings.append(
        {
            "code": "augmented_dataset",
            "severity": "warning",
            "detail": "Augmented images improve robustness but do not replace source-image diversity or human-reviewed labels.",
        }
    )
    positive_images = sum(1 for row in image_rows if labels_by_image.get(str(row["image"])))
    split_groups = len({str(row.get("split_key") or row.get("source_key") or row.get("image")) for row in image_rows})
    return {
        "readiness": {
            "pilot_eval_ready": bool(label_rows),
            "model_training_ready": source_ready,
            "has_error": False,
        },
        "counts": {
            "source_images_before_augmentation": source_image_count,
            "source_images": len(image_rows),
            "images": len(image_rows),
            "positive_images": positive_images,
            "negative_images": len(image_rows) - positive_images,
            "annotations": len(label_rows),
            "split_groups": split_groups,
            "augmented_images": max(0, len(image_rows) - source_image_count),
            "skipped_variants": skipped_variants,
        },
        "split_counts": split_counts,
        "warnings": warnings,
    }


def augment_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    image_manifest = read_jsonl(args.dataset / "annotations" / "image_manifest.jsonl")
    label_manifest = read_jsonl(args.dataset / "annotations" / "manifest.jsonl")
    if args.max_images:
        image_manifest = image_manifest[: args.max_images]
    selected_images = {str(row.get("image") or "") for row in image_manifest}
    labels_by_source_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in label_manifest:
        image_rel = str(label.get("image") or "")
        if image_rel in selected_images:
            labels_by_source_image[image_rel].append(label)

    output_images: list[dict[str, Any]] = []
    output_labels: list[dict[str, Any]] = []
    skipped_variants = 0

    for image_row in image_manifest:
        split = str(image_row.get("split") or "train")
        source_rel = str(image_row.get("image") or "")
        source_path = image_path_for_row(args.dataset, image_row)
        if not source_path.is_file() or source_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        source_labels = labels_by_source_image.get(source_rel, [])
        stem = Path(source_rel).stem
        original_rel = f"images/{split}/{stem}.jpg"
        width, height = copy_or_normalize_image(source_path, args.out / original_rel)
        original_source_key = str(image_row.get("source_key") or source_rel)
        original_image_row = dict(image_row)
        original_image_row.update(
            {
                "image": original_rel,
                "width": width,
                "height": height,
                "label_count": len(source_labels),
                "augmentation": "original",
            }
        )
        output_images.append(original_image_row)
        for index, label in enumerate(source_labels):
            output_labels.append(label_with_image(label, original_rel, split, original_source_key, f"orig:{index + 1}"))

        with Image.open(source_path) as image:
            source_image = normalize_image(image)
        for variant_index in range(args.variants_per_image):
            variant_id = f"aug{variant_index + 1:02d}"
            variant_rel = f"images/{split}/{stem}__{variant_id}.jpg"
            variant_source_key = f"{original_source_key}:{variant_id}"
            transformed_labels: list[dict[str, Any]] = []
            transformed_image: Image.Image | None = None
            params: dict[str, Any] | None = None
            for _attempt in range(args.max_attempts):
                params = random_params(rng, width, height, args.max_crop_fraction)
                labels: list[dict[str, Any]] = []
                for label in source_labels:
                    transformed = transformed_label(label, width, height, params, args.min_box_area_ratio)
                    if transformed is None:
                        labels = []
                        break
                    labels.append(transformed)
                if len(labels) == len(source_labels):
                    transformed_labels = labels
                    transformed_image = apply_image_transform(source_image, params)
                    break
            if transformed_image is None or params is None:
                skipped_variants += 1
                continue
            save_jpeg(transformed_image, args.out / variant_rel, int(params["jpeg_quality"]))
            variant_row = dict(image_row)
            variant_row.update(
                {
                    "image": variant_rel,
                    "width": width,
                    "height": height,
                    "label_count": len(transformed_labels),
                    "source_key": variant_source_key,
                    "origin": f"{image_row.get('origin') or 'source'}|mobile_augment",
                    "augmentation": "mobile_capture",
                    "augmentation_params": {key: value for key, value in params.items() if key != "quality_flags"},
                }
            )
            output_images.append(variant_row)
            for label_index, label in enumerate(transformed_labels):
                output_labels.append(label_with_image(label, variant_rel, split, variant_source_key, f"{variant_id}:{label_index + 1}"))

    labels_by_output_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in output_labels:
        labels_by_output_image[str(label["image"])].append(label)

    write_jsonl(args.out / "annotations" / "image_manifest.jsonl", output_images)
    write_jsonl(args.out / "annotations" / "manifest.jsonl", output_labels)
    (args.out / "annotations" / "review.jsonl").parent.mkdir(parents=True, exist_ok=True)
    (args.out / "annotations" / "review.jsonl").write_text("", encoding="utf-8")
    write_yolo(args.out, output_images, labels_by_output_image)
    write_coco(args.out, output_images, output_labels)
    build_preview(args.out, output_images, labels_by_output_image)
    audit = build_audit(args.dataset, len(image_manifest), output_images, output_labels, skipped_variants)
    write_json(args.out / "audit.json", audit)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(args.dataset),
        "seed": args.seed,
        "variants_per_image": args.variants_per_image,
        "max_crop_fraction": args.max_crop_fraction,
        "min_box_area_ratio": args.min_box_area_ratio,
        "audit": audit,
    }
    write_json(args.out / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mobile-capture augmentations for question detector datasets.")
    parser.add_argument("--dataset", type=Path, required=True, help="Source dataset root exported by question_detector_dataset.py.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variants-per-image", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0, help="Optional smoke-test cap over image_manifest rows.")
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--max-crop-fraction", type=float, default=0.035)
    parser.add_argument("--min-box-area-ratio", type=float, default=0.55)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    summary = augment_dataset(args)
    print(
        json.dumps(
            {
                "images": summary["audit"]["counts"]["images"],
                "annotations": summary["audit"]["counts"]["annotations"],
                "augmented_images": summary["audit"]["counts"]["augmented_images"],
                "model_training_ready": summary["audit"]["readiness"]["model_training_ready"],
                "out": str(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
