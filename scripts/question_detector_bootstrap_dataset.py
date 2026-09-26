"""Build a non-release bootstrap detector dataset from reviewed and draft boxes.

This dataset is useful for early detector training and error discovery when the
human-reviewed set is still too small. It deliberately remains
model_training_ready=false, even when it contains many pseudo labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


CATEGORY_NAME = "question_block"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def stable_id(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def normalize_image(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def resolve_image(root: Path, image_value: str) -> Path | None:
    raw = Path(str(image_value or ""))
    candidates = [raw] if raw.is_absolute() else []
    if not raw.is_absolute():
        candidates.extend([root / raw, root / "images" / raw.name, root.parent / raw, root.parent / "images" / raw.name])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def split_for_group(group_key: str, val_ratio: float, test_ratio: float) -> str:
    bucket = int(hashlib.sha1(group_key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


def clamp_box(x: float, y: float, width: float, height: float, image_width: int, image_height: int) -> dict[str, int] | None:
    x1 = max(0.0, min(float(image_width), x))
    y1 = max(0.0, min(float(image_height), y))
    x2 = max(0.0, min(float(image_width), x + width))
    y2 = max(0.0, min(float(image_height), y + height))
    w = int(round(x2 - x1))
    h = int(round(y2 - y1))
    if w <= 3 or h <= 3:
        return None
    return {"x": int(round(x1)), "y": int(round(y1)), "width": w, "height": h}


def box_area_ratio(box: dict[str, int], width: int, height: int) -> float:
    return (box["width"] * box["height"]) / max(1.0, float(width * height))


def acceptable_box(
    box: dict[str, int],
    width: int,
    height: int,
    flags: list[str],
    score: float,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if score < args.min_draft_score:
        return False, "low_score"
    area = box_area_ratio(box, width, height)
    if area < args.min_area_ratio:
        return False, "small_area"
    if area > args.max_area_ratio:
        return False, "large_area"
    if box["width"] < args.min_box_side_px or box["height"] < args.min_box_side_px:
        return False, "tiny_side"
    aspect = box["width"] / max(1, box["height"])
    if aspect > args.max_aspect_ratio or aspect < 1.0 / args.max_aspect_ratio:
        return False, "extreme_aspect"
    rejected = set(flags) & args.reject_quality_flags
    if rejected:
        return False, f"rejected_flag:{sorted(rejected)[0]}"
    return True, "accepted"


@dataclass
class SourceImage:
    source_path: Path
    source_key: str
    split_key: str
    split: str
    width: int
    height: int
    origin: str
    negative_kind: str = ""


@dataclass
class SourceLabel:
    image_key: str
    bbox_px: dict[str, int]
    confidence: float
    origin: str
    quality_flags: list[str]
    annotation_status: str
    question_key: str


def add_source_image(
    images: dict[str, SourceImage],
    path: Path,
    split_key: str,
    origin: str,
    args: argparse.Namespace,
    negative_kind: str = "",
) -> str | None:
    try:
        with Image.open(path) as image:
            normalized = normalize_image(image)
            width, height = normalized.size
    except Exception:
        return None
    source_key = f"{origin}:{path.name}:{stable_id(str(path.resolve()))}"
    image_key = stable_id(source_key)
    if image_key not in images:
        images[image_key] = SourceImage(
            source_path=path,
            source_key=source_key,
            split_key=split_key or image_key,
            split=split_for_group(split_key or image_key, args.val_ratio, args.test_ratio),
            width=width,
            height=height,
            origin=origin,
            negative_kind=negative_kind,
        )
    return image_key


def approved_image_names(paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            names.add(Path(str(row.get("image") or "")).name)
    return names


def load_approved(paths: list[Path], images: dict[str, SourceImage], labels: list[SourceLabel], args: argparse.Namespace) -> Counter[str]:
    stats: Counter[str] = Counter()
    for path in paths:
        root = path.parent
        for row in read_jsonl(path):
            image_path = resolve_image(root, str(row.get("image") or ""))
            if image_path is None:
                stats["missing_image"] += 1
                continue
            candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
            split_key = str(candidate.get("session_id") or row.get("source_candidate") or Path(str(row.get("image") or "")).stem)
            image_key = add_source_image(images, image_path, split_key, f"human_reviewed:{path}", args)
            if image_key is None:
                stats["bad_image"] += 1
                continue
            image = images[image_key]
            for index, box_row in enumerate(row.get("boxes") or []):
                status = str(box_row.get("annotation_status") or row.get("annotation_status") or "")
                if status not in args.approved_statuses:
                    stats["not_approved"] += 1
                    continue
                bbox = box_row.get("bbox_px") if isinstance(box_row.get("bbox_px"), dict) else {}
                try:
                    box = clamp_box(float(bbox.get("x")), float(bbox.get("y")), float(bbox.get("width")), float(bbox.get("height")), image.width, image.height)
                except (TypeError, ValueError):
                    box = None
                if box is None:
                    stats["bad_box"] += 1
                    continue
                flags = list(dict.fromkeys([str(flag) for flag in box_row.get("quality_flags") or []] + ["human_reviewed"]))
                labels.append(
                    SourceLabel(
                        image_key=image_key,
                        bbox_px=box,
                        confidence=float(box_row.get("score") or 1.0),
                        origin=f"human_reviewed:{path}",
                        quality_flags=flags,
                        annotation_status=status,
                        question_key=f"reviewed:{stable_id(image.source_key)}:{index + 1}",
                    )
                )
                stats["accepted_labels"] += 1
    return stats


def load_draft_prelabels(
    prelabel_roots: list[Path],
    approved_names: set[str],
    images: dict[str, SourceImage],
    labels: list[SourceLabel],
    args: argparse.Namespace,
) -> Counter[str]:
    stats: Counter[str] = Counter()
    for root in prelabel_roots:
        for row in read_jsonl(root / "annotations" / "draft_boxes.jsonl"):
            image_name = Path(str(row.get("image") or "")).name
            if args.skip_draft_images_with_approved and image_name in approved_names:
                stats["skipped_approved_image"] += 1
                continue
            image_path = resolve_image(root, str(row.get("image") or ""))
            if image_path is None:
                stats["missing_image"] += 1
                continue
            candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
            split_key = str(candidate.get("session_id") or row.get("source_candidate") or image_name)
            image_key = add_source_image(images, image_path, split_key, f"pseudo_draft:{root}", args)
            if image_key is None:
                stats["bad_image"] += 1
                continue
            image = images[image_key]
            accepted_for_image = 0
            for index, box_row in enumerate(row.get("boxes") or []):
                bbox = box_row.get("bbox_px") if isinstance(box_row.get("bbox_px"), dict) else {}
                try:
                    box = clamp_box(float(bbox.get("x")), float(bbox.get("y")), float(bbox.get("width")), float(bbox.get("height")), image.width, image.height)
                except (TypeError, ValueError):
                    box = None
                if box is None:
                    stats["bad_box"] += 1
                    continue
                flags = [str(flag) for flag in box_row.get("quality_flags") or []]
                score = float(box_row.get("score") or 0.0)
                ok, reason = acceptable_box(box, image.width, image.height, flags, score, args)
                if not ok:
                    stats[f"rejected:{reason}"] += 1
                    continue
                out_flags = list(dict.fromkeys(flags + ["pseudo_label", "bootstrap_draft"]))
                labels.append(
                    SourceLabel(
                        image_key=image_key,
                        bbox_px=box,
                        confidence=score,
                        origin=f"pseudo_draft:{root}",
                        quality_flags=out_flags,
                        annotation_status="pseudo_bootstrap",
                        question_key=f"pseudo:{stable_id(image.source_key)}:{index + 1}",
                    )
                )
                accepted_for_image += 1
                stats["accepted_labels"] += 1
            if accepted_for_image:
                stats["accepted_images"] += 1
    return stats


def load_negatives(dirs: list[Path], images: dict[str, SourceImage], args: argparse.Namespace) -> Counter[str]:
    stats: Counter[str] = Counter()
    for directory in dirs:
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            if args.negative_limit and stats["accepted_images"] >= args.negative_limit:
                break
            split_key = path.stem.split("_")[0] if "_" in path.stem else path.stem
            if add_source_image(images, path, split_key, f"negative_dir:{directory}", args, negative_kind="empty_page"):
                stats["accepted_images"] += 1
    return stats


def output_image_rel(image_key: str, image: SourceImage) -> str:
    return f"images/{image.split}/{image_key}_{image.source_path.stem[:80]}.jpg"


def copy_images(out: Path, images: dict[str, SourceImage]) -> dict[str, str]:
    rel_by_key: dict[str, str] = {}
    for image_key, image in images.items():
        rel = output_image_rel(image_key, image)
        rel_by_key[image_key] = rel
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image.source_path) as source:
            normalized = normalize_image(source)
            normalized.save(target, format="JPEG", quality=88, optimize=True)
    return rel_by_key


def label_rows(images: dict[str, SourceImage], labels: list[SourceLabel], rel_by_key: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels, start=1):
        image = images[label.image_key]
        rel = rel_by_key[label.image_key]
        box = label.bbox_px
        rows.append(
            {
                "image": rel,
                "split": image.split,
                "label": CATEGORY_NAME,
                "bbox_norm": {
                    "x": box["x"] / image.width,
                    "y": box["y"] / image.height,
                    "width": box["width"] / image.width,
                    "height": box["height"] / image.height,
                },
                "bbox_px": box,
                "question_key": label.question_key,
                "question_index": index,
                "confidence": round(label.confidence, 4),
                "quality_flags": label.quality_flags,
                "annotation_status": label.annotation_status,
                "split_key": image.split_key,
                "source_key": image.source_key,
                "source_filename": image.source_path.name,
                "origin": label.origin,
            }
        )
    return rows


def image_rows(images: dict[str, SourceImage], labels: list[SourceLabel], rel_by_key: dict[str, str]) -> list[dict[str, Any]]:
    label_counts = Counter(label.image_key for label in labels)
    rows: list[dict[str, Any]] = []
    for image_key, image in images.items():
        count = label_counts[image_key]
        rows.append(
            {
                "image": rel_by_key[image_key],
                "split": image.split,
                "split_key": image.split_key,
                "source_key": image.source_key,
                "source_filename": image.source_path.name,
                "origin": image.origin,
                "width": image.width,
                "height": image.height,
                "label_count": count,
                "is_negative": count == 0,
                "negative_kind": image.negative_kind if count == 0 else "",
                "quality_flags": ["negative_sample", "negative:empty_page"] if count == 0 else ["bootstrap_positive"],
            }
        )
    return sorted(rows, key=lambda row: (row["split"], row["image"]))


def write_yolo(out: Path, images_rows: list[dict[str, Any]], labels_rows: list[dict[str, Any]]) -> None:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels_rows:
        by_image[str(row["image"])].append(row)
    for image in images_rows:
        rel = str(image["image"])
        split = str(image["split"])
        target = out / "labels" / split / f"{Path(rel).stem}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for label in by_image.get(rel, []):
            box = label["bbox_norm"]
            cx = float(box["x"]) + float(box["width"]) / 2.0
            cy = float(box["y"]) + float(box["height"]) / 2.0
            lines.append(f"0 {cx:.8f} {cy:.8f} {float(box['width']):.8f} {float(box['height']):.8f}")
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    write_json(
        out / "yolo_dataset.yaml",
        {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "test": "images/test", "names": {"0": CATEGORY_NAME}},
    )


def coco_for_split(images_rows: list[dict[str, Any]], labels_rows: list[dict[str, Any]], split: str | None) -> dict[str, Any]:
    selected = [row for row in images_rows if split is None or row["split"] == split]
    image_id = {str(row["image"]): index + 1 for index, row in enumerate(selected)}
    images = [
        {"id": image_id[str(row["image"])], "file_name": row["image"], "width": row["width"], "height": row["height"]}
        for row in selected
    ]
    annotations: list[dict[str, Any]] = []
    ann_id = 1
    for label in labels_rows:
        rel = str(label["image"])
        if rel not in image_id:
            continue
        box = label["bbox_px"]
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id[rel],
                "category_id": 1,
                "bbox": [box["x"], box["y"], box["width"], box["height"]],
                "area": box["width"] * box["height"],
                "iscrowd": 0,
            }
        )
        ann_id += 1
    return {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": CATEGORY_NAME}]}


def write_coco(out: Path, images_rows: list[dict[str, Any]], labels_rows: list[dict[str, Any]]) -> None:
    for split in ("train", "val", "test"):
        write_json(out / "annotations" / f"coco_{split}.json", coco_for_split(images_rows, labels_rows, split))
    write_json(out / "annotations" / "coco_all.json", coco_for_split(images_rows, labels_rows, None))


def build_preview(out: Path, images_rows: list[dict[str, Any]], labels_rows: list[dict[str, Any]], limit: int = 48) -> None:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in labels_rows:
        by_image[str(label["image"])].append(label)
    positives = [row for row in images_rows if by_image.get(str(row["image"]))]
    if not positives:
        return
    thumb_w, thumb_h, columns = 320, 220, 4
    selected = positives[:limit]
    sheet = Image.new("RGB", (columns * thumb_w, math.ceil(len(selected) / columns) * thumb_h), "white")
    for index, row in enumerate(selected):
        path = out / str(row["image"])
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
        sx = preview.width / max(1, original_w)
        sy = preview.height / max(1, original_h)
        for label in by_image.get(str(row["image"]), []):
            box = label["bbox_px"]
            x = offset_x + box["x"] * sx
            y = offset_y + box["y"] * sy
            draw.rectangle((x, y, x + box["width"] * sx, y + box["height"] * sy), outline=(229, 57, 53), width=2)
        draw.text((6, 4), f"{row['split']} {Path(str(row['image'])).name[:34]}", fill=(20, 20, 20))
        sheet.paste(canvas, ((index % columns) * thumb_w, (index // columns) * thumb_h))
    sheet.save(out / "preview_contact_sheet.jpg", format="JPEG", quality=90, optimize=True)


def build_audit(images_rows: list[dict[str, Any]], labels_rows: list[dict[str, Any]], stats: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_images = [row for row in images_rows if row["split"] == split]
        split_labels = [row for row in labels_rows if row["split"] == split]
        split_counts[split] = {
            "images": len(split_images),
            "positive_images": sum(1 for row in split_images if int(row["label_count"]) > 0),
            "negative_images": sum(1 for row in split_images if int(row["label_count"]) == 0),
            "annotations": len(split_labels),
            "groups": len({str(row["split_key"]) for row in split_images}),
        }
    pseudo_count = sum(1 for row in labels_rows if row.get("annotation_status") == "pseudo_bootstrap")
    reviewed_count = len(labels_rows) - pseudo_count
    warnings = [
        {
            "code": "bootstrap_dataset_not_release_ready",
            "severity": "warning",
            "detail": "This dataset contains pseudo labels and is forced to model_training_ready=false. Use it for exploration and active learning only.",
        }
    ]
    return {
        "readiness": {"pilot_eval_ready": bool(labels_rows), "model_training_ready": False, "has_error": False},
        "counts": {
            "source_images": len(images_rows),
            "positive_images": sum(1 for row in images_rows if int(row["label_count"]) > 0),
            "negative_images": sum(1 for row in images_rows if int(row["label_count"]) == 0),
            "annotations": len(labels_rows),
            "human_reviewed_annotations": reviewed_count,
            "pseudo_annotations": pseudo_count,
            "split_groups": len({str(row["split_key"]) for row in images_rows}),
        },
        "split_counts": split_counts,
        "bootstrap": {
            "min_draft_score": args.min_draft_score,
            "min_area_ratio": args.min_area_ratio,
            "max_area_ratio": args.max_area_ratio,
            "reject_quality_flags": sorted(args.reject_quality_flags),
            "stats": stats,
        },
        "warnings": warnings,
    }


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    args.approved_statuses = set(split_csv(args.approved_statuses))
    args.reject_quality_flags = set(split_csv(args.reject_quality_flags))

    images: dict[str, SourceImage] = {}
    labels: list[SourceLabel] = []
    stats: dict[str, Any] = {}
    approved_names = approved_image_names(args.approved_jsonl)
    stats["approved"] = dict(load_approved(args.approved_jsonl, images, labels, args))
    stats["draft"] = dict(load_draft_prelabels(args.prelabel_root, approved_names, images, labels, args))
    stats["negative"] = dict(load_negatives(args.negative_images_dir, images, args))

    rel_by_key = copy_images(args.out, images)
    labels_rows = label_rows(images, labels, rel_by_key)
    images_rows = image_rows(images, labels, rel_by_key)
    write_jsonl(args.out / "annotations" / "manifest.jsonl", labels_rows)
    write_jsonl(args.out / "annotations" / "image_manifest.jsonl", images_rows)
    (args.out / "annotations" / "review.jsonl").write_text("", encoding="utf-8")
    write_yolo(args.out, images_rows, labels_rows)
    write_coco(args.out, images_rows, labels_rows)
    build_preview(args.out, images_rows, labels_rows)
    audit = build_audit(images_rows, labels_rows, stats, args)
    write_json(args.out / "audit.json", audit)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "prelabel_roots": [str(path) for path in args.prelabel_root],
            "approved_jsonl": [str(path) for path in args.approved_jsonl],
            "negative_images_dir": [str(path) for path in args.negative_images_dir],
        },
        "audit": audit,
    }
    write_json(args.out / "metadata.json", metadata)
    return metadata


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a non-release bootstrap question detector dataset.")
    parser.add_argument("--prelabel-root", type=Path, action="append", default=[], help="Root containing annotations/draft_boxes.jsonl.")
    parser.add_argument("--approved-jsonl", type=Path, action="append", default=[], help="Approved review JSONL to include as human-reviewed labels.")
    parser.add_argument("--negative-images-dir", type=Path, action="append", default=[], help="Directory of textless/empty-page negative images.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-draft-score", type=float, default=0.70)
    parser.add_argument("--min-area-ratio", type=float, default=0.002)
    parser.add_argument("--max-area-ratio", type=float, default=0.92)
    parser.add_argument("--min-box-side-px", type=int, default=16)
    parser.add_argument("--max-aspect-ratio", type=float, default=24.0)
    parser.add_argument("--reject-quality-flags", default="few_text_lines,short_block")
    parser.add_argument("--approved-statuses", default="approved,accepted,corrected,verified")
    parser.add_argument("--skip-draft-images-with-approved", action="store_true", default=True)
    parser.add_argument("--negative-limit", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    report = build_dataset(args)
    audit = report["audit"]
    print(
        json.dumps(
            {
                "out": str(args.out),
                "model_training_ready": audit["readiness"]["model_training_ready"],
                "source_images": audit["counts"]["source_images"],
                "annotations": audit["counts"]["annotations"],
                "human_reviewed_annotations": audit["counts"]["human_reviewed_annotations"],
                "pseudo_annotations": audit["counts"]["pseudo_annotations"],
                "negative_images": audit["counts"]["negative_images"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
