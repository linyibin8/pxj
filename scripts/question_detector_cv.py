"""Build group-preserving cross-validation folds for question detector datasets.

Input is a dataset exported by question_detector_dataset.py. Output is a set of
fold directories that keep session/batch groups together while preserving COCO,
YOLO, Create ML, and image-level manifests for each fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


CATEGORY_ID = 1
CATEGORY_NAME = "question_block"


@dataclass
class FoldImage:
    original_id: int
    record: dict[str, Any]
    split: str


def stable_id(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def maybe_clean(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_average_hash(path: Path, hash_size: int = 8) -> str:
    try:
        with Image.open(path) as image:
            gray = ImageOps.exif_transpose(image).convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    except Exception:
        return ""
    pixels = list(gray.getdata())
    if not pixels:
        return ""
    mean = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | (1 if pixel >= mean else 0)
    return f"{value:0{hash_size * hash_size // 4}x}"


def hex_hamming(left: str, right: str) -> int:
    if not left or not right:
        return 999
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except ValueError:
        return 999


def group_key_for_image(image: dict[str, Any], field: str) -> str:
    value = str(image.get(field) or "").strip()
    if value:
        return value
    for fallback in ("split_key", "source_key", "source_filename", "file_name"):
        value = str(image.get(fallback) or "").strip()
        if value:
            return value
    return f"image:{image.get('id')}"


def effective_group_key(image: dict[str, Any], field: str, group_aliases: dict[str, str] | None = None) -> str:
    key = group_key_for_image(image, field)
    return (group_aliases or {}).get(key, key)


def load_dataset(dataset: Path) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    coco_path = dataset / "annotations" / "coco_all.json"
    if not coco_path.is_file():
        raise SystemExit(f"COCO all annotations not found: {coco_path}")
    coco = read_json(coco_path)
    images = {int(item["id"]): item for item in coco.get("images", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        image_id = int(ann.get("image_id") or 0)
        if image_id in images:
            annotations_by_image[image_id].append(ann)
    categories = coco.get("categories") or [{"id": CATEGORY_ID, "name": CATEGORY_NAME}]
    return images, annotations_by_image, categories


def duplicate_group_aliases(
    dataset: Path,
    images: dict[int, dict[str, Any]],
    group_field: str,
    args: argparse.Namespace,
) -> tuple[dict[str, str], dict[str, Any]]:
    groups = {group_key_for_image(image, group_field) for image in images.values()}
    parent = {key: key for key in groups}

    def find(key: str) -> str:
        root = parent.setdefault(key, key)
        while parent[root] != root:
            root = parent[root]
        while parent[key] != key:
            next_key = parent[key]
            parent[key] = root
            key = next_key
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        canonical = min(left_root, right_root)
        other = right_root if canonical == left_root else left_root
        parent[other] = canonical

    records: list[dict[str, Any]] = []
    for image_id, image in images.items():
        image_path = dataset / str(image.get("file_name") or "")
        if not image_path.is_file():
            continue
        records.append(
            {
                "image_id": image_id,
                "image": str(image.get("file_name") or ""),
                "group": group_key_for_image(image, group_field),
                "sha1": file_sha1(image_path),
                "ahash": image_average_hash(image_path),
            }
        )

    exact_links: list[dict[str, Any]] = []
    by_sha1: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sha1[str(record["sha1"])].append(record)
    for digest, matches in by_sha1.items():
        groups_for_digest = sorted({str(record["group"]) for record in matches})
        if len(groups_for_digest) <= 1:
            continue
        for group in groups_for_digest[1:]:
            union(groups_for_digest[0], group)
        exact_links.append(
            {
                "sha1": digest,
                "groups": groups_for_digest[:10],
                "images": [str(record["image"]) for record in matches[:10]],
            }
        )

    near_links: list[dict[str, Any]] = []
    threshold = max(0, int(args.near_duplicate_threshold))
    max_recorded = max(0, int(args.max_near_duplicate_pairs))
    for left_index, left in enumerate(records):
        for right in records[left_index + 1:]:
            if left["group"] == right["group"]:
                continue
            distance = hex_hamming(str(left.get("ahash") or ""), str(right.get("ahash") or ""))
            if distance <= threshold:
                union(str(left["group"]), str(right["group"]))
                if not max_recorded or len(near_links) < max_recorded:
                    near_links.append(
                        {
                            "distance": distance,
                            "left": str(left["image"]),
                            "left_group": str(left["group"]),
                            "right": str(right["image"]),
                            "right_group": str(right["group"]),
                        }
                    )

    clusters: dict[str, list[str]] = defaultdict(list)
    for group in sorted(groups):
        clusters[find(group)].append(group)
    aliases: dict[str, str] = {}
    merged_clusters: list[dict[str, Any]] = []
    for canonical, members in sorted(clusters.items()):
        for member in members:
            aliases[member] = canonical
        if len(members) > 1:
            merged_clusters.append({"canonical": canonical, "groups": members})

    summary = {
        "enabled": bool(args.cluster_near_duplicates),
        "group_field": group_field,
        "original_group_count": len(groups),
        "clustered_group_count": len(clusters),
        "merged_group_count": sum(1 for members in clusters.values() if len(members) > 1),
        "near_duplicate_threshold": threshold,
        "exact_duplicate_group_links": exact_links[:max_recorded] if max_recorded else exact_links,
        "near_duplicate_group_links": near_links,
        "merged_clusters": merged_clusters[:max_recorded] if max_recorded else merged_clusters,
    }
    return aliases, summary


def assign_groups_to_folds(
    images: dict[int, dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    group_field: str,
    folds: int,
    group_aliases: dict[str, str] | None = None,
) -> dict[str, int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for image_id, image in images.items():
        grouped[effective_group_key(image, group_field, group_aliases)].append(image_id)
    if not grouped:
        return {}
    fold_count = max(1, min(folds, len(grouped)))
    fold_weights = [0.0 for _ in range(fold_count)]
    assignments: dict[str, int] = {}
    weighted_groups: list[tuple[float, str]] = []
    for key, image_ids in grouped.items():
        annotation_count = sum(len(annotations_by_image.get(image_id, [])) for image_id in image_ids)
        negative_count = sum(1 for image_id in image_ids if not annotations_by_image.get(image_id))
        weight = float(annotation_count) + 0.1 * len(image_ids) + 0.05 * negative_count
        weighted_groups.append((weight, key))
    for _weight, key in sorted(weighted_groups, key=lambda item: (-item[0], stable_id(item[1], 40))):
        fold_index = min(range(fold_count), key=lambda index: (fold_weights[index], index))
        assignments[key] = fold_index
        fold_weights[fold_index] += _weight
    return assignments


def group_annotation_counts(
    images: dict[int, dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    group_field: str,
    group_aliases: dict[str, str] | None = None,
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"images": 0, "positive_images": 0, "annotations": 0})
    for image_id, image in images.items():
        key = effective_group_key(image, group_field, group_aliases)
        annotation_count = len(annotations_by_image.get(image_id, []))
        counts[key]["images"] += 1
        counts[key]["annotations"] += annotation_count
        if annotation_count > 0:
            counts[key]["positive_images"] += 1
    return counts


def val_group_keys(
    train_group_keys: list[str],
    val_ratio: float,
    fold_index: int,
    group_counts: dict[str, dict[str, int]],
) -> set[str]:
    if not train_group_keys or val_ratio <= 0:
        return set()
    count = max(1 if len(train_group_keys) >= 2 else 0, int(round(len(train_group_keys) * val_ratio)))
    count = min(count, max(0, len(train_group_keys) - 1))
    if count <= 0:
        return set()

    positive_keys = [
        key
        for key in train_group_keys
        if group_counts.get(key, {}).get("annotations", 0) > 0
    ]
    ordered_positive = sorted(positive_keys, key=lambda key: stable_id(f"{fold_index}:positive:{key}", 40))
    ordered_all = sorted(train_group_keys, key=lambda key: stable_id(f"{fold_index}:{key}", 40))

    selected: list[str] = []
    total_positive_groups = len(positive_keys)
    if total_positive_groups >= 2:
        positive_target = max(1, int(round(total_positive_groups * val_ratio)))
        positive_target = min(positive_target, total_positive_groups - 1, count)
        selected.extend(ordered_positive[:positive_target])

    for key in ordered_all:
        if len(selected) >= count:
            break
        if key not in selected:
            selected.append(key)
    return set(selected)


def split_images_for_fold(
    images: dict[int, dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    group_assignments: dict[str, int],
    group_field: str,
    fold_index: int,
    val_ratio: float,
    group_aliases: dict[str, str] | None = None,
) -> list[FoldImage]:
    train_groups = [key for key, assigned_fold in group_assignments.items() if assigned_fold != fold_index]
    val_groups = val_group_keys(
        train_groups,
        val_ratio,
        fold_index,
        group_annotation_counts(images, annotations_by_image, group_field, group_aliases),
    )
    result: list[FoldImage] = []
    for image_id, image in sorted(images.items()):
        group_key = effective_group_key(image, group_field, group_aliases)
        if group_assignments[group_key] == fold_index:
            split = "test"
        elif group_key in val_groups:
            split = "val"
        else:
            split = "train"
        result.append(FoldImage(original_id=image_id, record=image, split=split))
    return result


def copy_image(dataset: Path, fold_dir: Path, image: FoldImage, new_file_name: str) -> None:
    source = dataset / str(image.record.get("file_name") or "")
    if not source.is_file():
        raise SystemExit(f"source image missing for fold export: {source}")
    target = fold_dir / new_file_name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def yolo_line(ann: dict[str, Any], image_record: dict[str, Any]) -> str:
    width = max(1.0, float(image_record.get("width") or 1))
    height = max(1.0, float(image_record.get("height") or 1))
    x, y, box_width, box_height = [float(value) for value in ann.get("bbox") or [0, 0, 0, 0]]
    x = max(0.0, min(width - 1.0, x))
    y = max(0.0, min(height - 1.0, y))
    box_width = max(1.0, min(width - x, box_width))
    box_height = max(1.0, min(height - y, box_height))
    cx = max(0.0, min(1.0, (x + box_width / 2) / width))
    cy = max(0.0, min(1.0, (y + box_height / 2) / height))
    norm_width = max(0.0, min(1.0, box_width / width))
    norm_height = max(0.0, min(1.0, box_height / height))
    return f"0 {cx:.6f} {cy:.6f} {norm_width:.6f} {norm_height:.6f}"


def create_ml_annotation(ann: dict[str, Any]) -> dict[str, Any]:
    x, y, width, height = [float(value) for value in ann.get("bbox") or [0, 0, 0, 0]]
    return {
        "label": CATEGORY_NAME,
        "coordinates": {
            "x": x + width / 2,
            "y": y + height / 2,
            "width": width,
            "height": height,
        },
    }


def write_fold(
    dataset: Path,
    out_dir: Path,
    fold_index: int,
    fold_images: list[FoldImage],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    categories: list[dict[str, Any]],
    args: argparse.Namespace,
    group_aliases: dict[str, str] | None = None,
    duplicate_grouping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fold_dir = out_dir / f"fold-{fold_index:02d}"
    maybe_clean(fold_dir, clean=True)
    for split in ("train", "val", "test"):
        (fold_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (fold_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    coco_by_split: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: {"images": [], "annotations": []}
        for split in ("train", "val", "test")
    }
    createml_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    image_manifest_rows: list[dict[str, Any]] = []
    original_to_new_id: dict[int, int] = {}

    for new_id, fold_image in enumerate(fold_images, start=1):
        original = fold_image.record
        original_to_new_id[fold_image.original_id] = new_id
        original_name = Path(str(original.get("file_name") or "")).name
        export_name = f"{stable_id(f'fold:{fold_index}:{fold_image.original_id}')}__{original_name}"
        rel_name = (Path("images") / fold_image.split / export_name).as_posix()
        copy_image(dataset, fold_dir, fold_image, rel_name)
        anns = annotations_by_image.get(fold_image.original_id, [])
        label_path = fold_dir / "labels" / fold_image.split / f"{Path(export_name).stem}.txt"
        label_path.write_text("\n".join(yolo_line(ann, original) for ann in anns) + ("\n" if anns else ""), encoding="utf-8")

        image_record = {
            **original,
            "id": new_id,
            "file_name": rel_name,
            "split": fold_image.split,
            "label_count": len(anns),
            "is_negative": len(anns) == 0,
        }
        coco_by_split[fold_image.split]["images"].append(image_record)
        createml_by_split[fold_image.split].append(
            {
                "image": rel_name,
                "annotations": [create_ml_annotation(ann) for ann in anns],
            }
        )
        image_manifest_rows.append(
            {
                "image": rel_name,
                "split": fold_image.split,
                "split_key": original.get("split_key") or "",
                "effective_split_key": effective_group_key(original, args.group_field, group_aliases),
                "source_key": original.get("source_key") or "",
                "source_filename": original.get("source_filename") or original_name,
                "origin": original.get("origin") or "",
                "width": int(original.get("width") or 0),
                "height": int(original.get("height") or 0),
                "label_count": len(anns),
                "is_negative": len(anns) == 0,
                "negative_kind": original.get("negative_kind") or "",
                "quality_flags": original.get("quality_flags") or [],
            }
        )

    new_annotation_id = 1
    for fold_image in fold_images:
        split = fold_image.split
        for ann in annotations_by_image.get(fold_image.original_id, []):
            copied = {
                **ann,
                "id": new_annotation_id,
                "image_id": original_to_new_id[fold_image.original_id],
            }
            coco_by_split[split]["annotations"].append(copied)
            new_annotation_id += 1

    for split, payload in coco_by_split.items():
        write_json(
            fold_dir / "annotations" / f"coco_{split}.json",
            {
                "info": {
                    "description": f"PXJ question detector CV fold {fold_index} {split}",
                    "version": "cv1",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "source_dataset": str(dataset),
                },
                "images": payload["images"],
                "annotations": payload["annotations"],
                "categories": categories,
            },
        )
        write_json(fold_dir / "annotations" / f"createml_{split}.json", createml_by_split.get(split, []))

    all_images = [image for split in ("train", "val", "test") for image in coco_by_split[split]["images"]]
    all_annotations = [ann for split in ("train", "val", "test") for ann in coco_by_split[split]["annotations"]]
    write_json(
        fold_dir / "annotations" / "coco_all.json",
        {
            "info": {
                "description": f"PXJ question detector CV fold {fold_index} all",
                "version": "cv1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source_dataset": str(dataset),
            },
            "images": all_images,
            "annotations": all_annotations,
            "categories": categories,
        },
    )
    write_json(
        fold_dir / "annotations" / "createml_all.json",
        [item for split in ("train", "val", "test") for item in createml_by_split.get(split, [])],
    )
    write_jsonl(fold_dir / "annotations" / "image_manifest.jsonl", image_manifest_rows)
    write_jsonl(
        fold_dir / "annotations" / "manifest.jsonl",
        [
            {
                "image": next(image["file_name"] for image in all_images if image["id"] == ann["image_id"]),
                "split": next(image["split"] for image in all_images if image["id"] == ann["image_id"]),
                "label": CATEGORY_NAME,
                "bbox_px": {
                    "x": ann["bbox"][0],
                    "y": ann["bbox"][1],
                    "width": ann["bbox"][2],
                    "height": ann["bbox"][3],
                },
                "question_key": ann.get("question_key") or "",
                "question_index": int(ann.get("question_index") or 0),
                "confidence": ann.get("confidence"),
                "quality_flags": ann.get("quality_flags") or [],
                "split_key": ann.get("split_key") or "",
                "source_key": ann.get("source_key") or "",
                "origin": ann.get("origin") or "",
            }
            for ann in all_annotations
        ],
    )
    write_json(
        fold_dir / "yolo_dataset.yaml",
        {
            "path": str(fold_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {0: CATEGORY_NAME},
        },
    )
    audit = build_fold_audit(dataset, fold_images, annotations_by_image, args, group_aliases, duplicate_grouping)
    write_json(fold_dir / "audit.json", audit)
    write_json(
        fold_dir / "metadata.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_dataset": str(dataset),
            "fold_index": fold_index,
            "folds": args.folds,
            "group_field": args.group_field,
            "val_ratio": args.val_ratio,
            "duplicate_grouping": duplicate_grouping or {},
            "audit": audit,
            "outputs": {
                "coco": "annotations/coco_*.json",
                "yolo": "images/{split}, labels/{split}, yolo_dataset.yaml",
                "createml": "annotations/createml_*.json",
                "image_manifest": "annotations/image_manifest.jsonl",
                "jsonl": "annotations/manifest.jsonl",
            },
        },
    )
    return {
        "fold": fold_index,
        "path": str(fold_dir),
        "audit": audit,
    }


def build_fold_audit(
    dataset: Path,
    fold_images: list[FoldImage],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    args: argparse.Namespace,
    group_aliases: dict[str, str] | None = None,
    duplicate_grouping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    split_counts: dict[str, dict[str, int]] = {}
    group_splits: dict[str, set[str]] = defaultdict(set)
    for split in ("train", "val", "test"):
        selected = [image for image in fold_images if image.split == split]
        annotations = sum(len(annotations_by_image.get(image.original_id, [])) for image in selected)
        negative_images = sum(1 for image in selected if not annotations_by_image.get(image.original_id))
        groups = {
            effective_group_key(image.record, args.group_field, group_aliases)
            for image in selected
        }
        for group in groups:
            group_splits[group].add(split)
        split_counts[split] = {
            "images": len(selected),
            "positive_images": len(selected) - negative_images,
            "negative_images": negative_images,
            "annotations": annotations,
            "groups": len(groups),
        }

    warnings: list[dict[str, str]] = []

    def warn(code: str, detail: str, severity: str = "warning") -> None:
        warnings.append({"code": code, "severity": severity, "detail": detail})

    group_overlaps = {
        group: sorted(splits)
        for group, splits in sorted(group_splits.items())
        if len(splits) > 1
    }
    if group_overlaps:
        warn("split_group_overlap", f"{len(group_overlaps)} groups appear in more than one split.", "error")
    for split, counts in split_counts.items():
        if counts["images"] == 0 or counts["annotations"] == 0:
            warn("empty_split", f"{split} has {counts['images']} images and {counts['annotations']} annotations.", "error")

    image_hash_records: list[dict[str, Any]] = []
    for image in fold_images:
        image_path = dataset / str(image.record.get("file_name") or "")
        if not image_path.is_file():
            continue
        image_hash_records.append(
            {
                "image": str(image.record.get("file_name") or ""),
                "split": image.split,
                "split_key": effective_group_key(image.record, args.group_field, group_aliases),
                "raw_split_key": group_key_for_image(image.record, args.group_field),
                "sha1": file_sha1(image_path),
                "ahash": image_average_hash(image_path),
            }
        )

    exact_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in image_hash_records:
        exact_by_hash[str(record["sha1"])].append(record)
    exact_cross_split: list[dict[str, Any]] = []
    for digest, records in exact_by_hash.items():
        splits = sorted({str(record["split"]) for record in records})
        if len(splits) > 1:
            exact_cross_split.append(
                {
                    "sha1": digest,
                    "splits": splits,
                    "images": [str(record["image"]) for record in records[:10]],
                }
            )

    near_pairs: list[dict[str, Any]] = []
    threshold = max(0, int(args.near_duplicate_threshold))
    max_pairs = max(0, int(args.max_near_duplicate_pairs))
    for left_index, left in enumerate(image_hash_records):
        for right in image_hash_records[left_index + 1:]:
            if left["split"] == right["split"]:
                continue
            distance = hex_hamming(str(left.get("ahash") or ""), str(right.get("ahash") or ""))
            if distance <= threshold:
                near_pairs.append(
                    {
                        "distance": distance,
                        "left": str(left["image"]),
                        "left_split": str(left["split"]),
                        "right": str(right["image"]),
                        "right_split": str(right["split"]),
                    }
                )
                if max_pairs and len(near_pairs) >= max_pairs:
                    break
        if max_pairs and len(near_pairs) >= max_pairs:
            break

    if exact_cross_split:
        warn("exact_image_leakage", f"{len(exact_cross_split)} exact image hashes appear across splits.", "error")
    if near_pairs:
        warn(
            "near_duplicate_cross_split",
            f"{len(near_pairs)} cross-split image pairs have aHash distance <= {threshold}.",
            args.near_duplicate_severity,
        )

    source_count = len(fold_images)
    annotation_count = sum(len(annotations_by_image.get(image.original_id, [])) for image in fold_images)
    split_group_count = len(group_splits)
    negative_count = sum(1 for image in fold_images if not annotations_by_image.get(image.original_id))
    if source_count < args.min_production_images:
        warn("too_few_images", f"{source_count} source images; target >= {args.min_production_images}.")
    if annotation_count < args.min_production_annotations:
        warn("too_few_annotations", f"{annotation_count} annotations; target >= {args.min_production_annotations}.")
    if split_group_count < args.min_production_groups:
        warn("too_few_split_groups", f"{split_group_count} groups; target >= {args.min_production_groups}.")
    if negative_count < args.min_production_negative_images:
        warn("too_few_negative_images", f"{negative_count} negative images; target >= {args.min_production_negative_images}.")
    has_error = any(item["severity"] == "error" for item in warnings)
    meets_scale = (
        source_count >= args.min_production_images
        and annotation_count >= args.min_production_annotations
        and split_group_count >= args.min_production_groups
        and negative_count >= args.min_production_negative_images
    )
    return {
        "readiness": {
            "pilot_eval_ready": annotation_count > 0 and not any(item["code"] == "empty_split" for item in warnings),
            "model_training_ready": (not has_error) and meets_scale,
            "has_error": has_error,
        },
        "counts": {
            "source_images": source_count,
            "annotations": annotation_count,
            "negative_images": negative_count,
            "split_groups": split_group_count,
        },
        "split_counts": split_counts,
        "leakage": {
            "split_group_overlaps": group_overlaps,
            "exact_cross_split_images": exact_cross_split,
            "near_duplicate_threshold": threshold,
            "near_duplicate_severity": args.near_duplicate_severity,
            "near_duplicate_cross_split_pairs": near_pairs,
            "duplicate_grouping": duplicate_grouping or {},
        },
        "minimums": {
            "production_images": args.min_production_images,
            "production_annotations": args.min_production_annotations,
            "production_negative_images": args.min_production_negative_images,
            "production_split_groups": args.min_production_groups,
        },
        "warnings": warnings,
    }


def export_cv(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset
    images, annotations_by_image, categories = load_dataset(dataset)
    if args.cluster_near_duplicates:
        group_aliases, duplicate_grouping = duplicate_group_aliases(dataset, images, args.group_field, args)
    else:
        group_aliases = {}
        original_groups = {group_key_for_image(image, args.group_field) for image in images.values()}
        duplicate_grouping = {
            "enabled": False,
            "group_field": args.group_field,
            "original_group_count": len(original_groups),
            "clustered_group_count": len(original_groups),
            "merged_group_count": 0,
            "near_duplicate_threshold": max(0, int(args.near_duplicate_threshold)),
            "exact_duplicate_group_links": [],
            "near_duplicate_group_links": [],
            "merged_clusters": [],
        }
    group_assignments = assign_groups_to_folds(images, annotations_by_image, args.group_field, args.folds, group_aliases)
    if not group_assignments:
        raise SystemExit("no images found for cross-validation export")
    maybe_clean(args.out, args.clean)
    summaries: list[dict[str, Any]] = []
    fold_count = max(group_assignments.values()) + 1
    for fold_index in range(fold_count):
        fold_images = split_images_for_fold(
            images,
            annotations_by_image,
            group_assignments,
            args.group_field,
            fold_index,
            args.val_ratio,
            group_aliases,
        )
        summaries.append(write_fold(dataset, args.out, fold_index, fold_images, annotations_by_image, categories, args, group_aliases, duplicate_grouping))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(dataset),
        "out": str(args.out),
        "requested_folds": args.folds,
        "folds": fold_count,
        "group_field": args.group_field,
        "group_count": len(group_assignments),
        "duplicate_grouping": duplicate_grouping,
        "folds_detail": summaries,
    }
    write_json(args.out / "cv_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export grouped K-fold datasets for question detector training.")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root from question_detector_dataset.py.")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-cv"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--group-field", default="split_key", help="COCO image field used to keep related images in the same held-out fold.")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation group ratio inside the non-held-out groups for each fold.")
    parser.add_argument("--min-production-images", type=int, default=300)
    parser.add_argument("--min-production-annotations", type=int, default=1000)
    parser.add_argument("--min-production-negative-images", type=int, default=0)
    parser.add_argument("--min-production-groups", type=int, default=50)
    parser.add_argument("--near-duplicate-threshold", type=int, default=4, help="aHash Hamming threshold for cross-split near-duplicate checks and fold grouping.")
    parser.add_argument("--max-near-duplicate-pairs", type=int, default=200, help="Maximum cross-split near-duplicate pairs to record in audit.json; 0 means unlimited.")
    parser.add_argument("--near-duplicate-severity", choices=["warning", "error"], default="error", help="Severity for cross-split aHash near duplicates; default is error for CV model-selection gates.")
    parser.add_argument("--no-cluster-near-duplicates", dest="cluster_near_duplicates", action="store_false", help="Disable default exact/aHash duplicate group clustering before assigning CV folds.")
    parser.set_defaults(cluster_near_duplicates=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be >= 2")
    summary = export_cv(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
