"""Propagate reviewed question boxes to duplicate historical images.

This is a conservative data-growth helper. It does not infer new boxes from a
model; it copies approved boxes only to exact duplicate target images with
matching dimensions and draft-box overlap evidence. Near-duplicate transfers
must stay in review-candidate mode until a human approves or corrects them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


APPROVED_STATUSES = {"approved", "accepted", "corrected", "verified"}
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def stable_id(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


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


def bbox_px(raw: Any, width: int, height: int) -> dict[str, int] | None:
    if not isinstance(raw, dict):
        return None
    try:
        x = float(raw.get("x"))
        y = float(raw.get("y"))
        w = float(raw.get("width"))
        h = float(raw.get("height"))
    except (TypeError, ValueError):
        return None
    x1 = max(0, min(width, int(round(x))))
    y1 = max(0, min(height, int(round(y))))
    x2 = max(0, min(width, int(round(x + w))))
    y2 = max(0, min(height, int(round(y + h))))
    out = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
    if out["width"] <= 3 or out["height"] <= 3:
        return None
    return out


def scale_box(box: dict[str, int], src_size: tuple[int, int], dst_size: tuple[int, int]) -> dict[str, int] | None:
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return None
    scaled = {
        "x": round(box["x"] * dst_w / src_w),
        "y": round(box["y"] * dst_h / src_h),
        "width": round(box["width"] * dst_w / src_w),
        "height": round(box["height"] * dst_h / src_h),
    }
    return bbox_px(scaled, dst_w, dst_h)


def box_iou(a: dict[str, int], b: dict[str, int]) -> float:
    ax2 = a["x"] + a["width"]
    ay2 = a["y"] + a["height"]
    bx2 = b["x"] + b["width"]
    by2 = b["y"] + b["height"]
    overlap_w = max(0, min(ax2, bx2) - max(a["x"], b["x"]))
    overlap_h = max(0, min(ay2, by2) - max(a["y"], b["y"]))
    overlap = overlap_w * overlap_h
    if overlap <= 0:
        return 0.0
    union = a["width"] * a["height"] + b["width"] * b["height"] - overlap
    return overlap / max(1, union)


def row_status(row: dict[str, Any]) -> str:
    return str(row.get("annotation_status") or row.get("status") or "").strip().lower()


def discover_approved(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
            continue
        candidates = [
            path / "approved_boxes.jsonl",
            path / "annotations" / "approved_boxes.jsonl",
        ]
        files.extend(candidate for candidate in candidates if candidate.is_file())
        files.extend(sorted(path.rglob("approved_boxes*.jsonl")) if path.is_dir() else [])
    return sorted({str(path.resolve()): path for path in files}.values())


def discover_targets(paths: list[Path], include_quarantine: bool) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
            continue
        candidates = [
            path / "annotations" / "draft_boxes.jsonl",
            path / "review_data.json",
        ]
        if include_quarantine:
            candidates.append(path / "annotations" / "quarantine.jsonl")
        files.extend(candidate for candidate in candidates if candidate.is_file())
    return sorted({str(path.resolve()): path for path in files}.values())


def resolve_image(row: dict[str, Any], manifest_path: Path) -> Path | None:
    raw_text = str(row.get("image") or row.get("file_name") or "").strip()
    if not raw_text:
        return None
    raw = Path(raw_text)
    if raw.is_absolute() and raw.is_file():
        return raw
    base = manifest_path.parent
    if manifest_path.name == "review_data.json":
        payload_source = row.get("_review_source_root")
        if payload_source:
            base = Path(str(payload_source))
    roots = [base, base.parent]
    if base.name == "annotations":
        roots.append(base.parent)
    for root in roots:
        for candidate in (
            root / raw,
            root / "images" / raw.name,
            root.parent / raw,
            root.parent / "images" / raw.name,
        ):
            if candidate.is_file():
                return candidate
    return None


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).size
    except Exception:
        return None


def load_target_rows(path: Path) -> list[dict[str, Any]]:
    if path.name == "review_data.json":
        payload = read_json(path)
        root = Path(str(payload.get("source") or path.parent))
        rows = payload.get("items") if isinstance(payload.get("items"), list) else []
        return [{**row, "_review_source_root": str(root)} for row in rows if isinstance(row, dict)]
    return read_jsonl(path)


def copy_image(source: Path, target: Path) -> tuple[int, int] | None:
    try:
        with Image.open(source) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            size = normalized.size
            target.parent.mkdir(parents=True, exist_ok=True)
            normalized.save(target, format="JPEG", quality=88, optimize=True)
            return size
    except Exception:
        return None


def load_seeds(paths: list[Path], approved_statuses: set[str]) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    for manifest in discover_approved(paths):
        for row_index, row in enumerate(read_jsonl(manifest)):
            image_path = resolve_image(row, manifest)
            if image_path is None:
                continue
            size = image_size(image_path)
            if size is None:
                continue
            width, height = size
            boxes: list[dict[str, Any]] = []
            row_status_value = row_status(row)
            for box_index, box in enumerate(row.get("boxes") or []):
                if not isinstance(box, dict):
                    continue
                status = row_status(box) or row_status_value
                if status not in approved_statuses:
                    continue
                bbox = bbox_px(box.get("bbox_px") or box.get("bbox"), width, height)
                if bbox is None:
                    continue
                boxes.append({"box": box, "bbox": bbox, "box_index": box_index})
            if not boxes:
                continue
            seeds.append(
                {
                    "manifest": str(manifest),
                    "row_index": row_index,
                    "row": row,
                    "image_path": image_path,
                    "width": width,
                    "height": height,
                    "sha1": file_sha1(image_path),
                    "ahash": image_average_hash(image_path),
                    "boxes": boxes,
                }
            )
    return seeds


def best_seed_for_target(
    target: dict[str, Any],
    seeds: list[dict[str, Any]],
    min_ahash_distance: int,
    max_ahash_distance: int,
    require_same_size: bool,
) -> tuple[dict[str, Any] | None, int]:
    best: dict[str, Any] | None = None
    best_distance = 999
    for seed in seeds:
        if require_same_size and (target["width"], target["height"]) != (seed["width"], seed["height"]):
            continue
        if target["sha1"] == seed["sha1"]:
            if min_ahash_distance > 0:
                continue
            return seed, 0
        distance = hex_hamming(str(target.get("ahash") or ""), str(seed.get("ahash") or ""))
        if min_ahash_distance <= distance <= max_ahash_distance and distance < best_distance:
            best = seed
            best_distance = distance
    return best, best_distance


def draft_boxes_for_row(row: dict[str, Any], width: int, height: int) -> list[dict[str, int]]:
    result: list[dict[str, int]] = []
    for box in row.get("boxes") or []:
        if isinstance(box, dict):
            bbox = bbox_px(box.get("bbox_px") or box.get("bbox"), width, height)
            if bbox:
                result.append(bbox)
    return result


def build_contact_sheet(rows: list[dict[str, Any]], out: Path, limit: int = 40) -> None:
    tiles: list[Image.Image] = []
    for row in rows[:limit]:
        image_path = out / str(row.get("image") or "")
        if not image_path.is_file():
            continue
        try:
            with Image.open(image_path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
        except Exception:
            continue
        scale = min(220 / image.width, 170 / image.height)
        thumb = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(thumb)
        for box in row.get("boxes") or []:
            bbox = box.get("bbox_px") if isinstance(box, dict) else None
            if not isinstance(bbox, dict):
                continue
            x = int(float(bbox.get("x") or 0) * scale)
            y = int(float(bbox.get("y") or 0) * scale)
            w = int(float(bbox.get("width") or 0) * scale)
            h = int(float(bbox.get("height") or 0) * scale)
            draw.rectangle([x, y, x + w, y + h], outline=(255, 40, 40), width=2)
        tile = Image.new("RGB", (240, 220), "white")
        tile.paste(thumb, ((240 - thumb.width) // 2, 8))
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.text((8, 184), f"boxes={len(row.get('boxes') or [])}", fill=(0, 0, 0))
        transfer = row.get("label_transfer") if isinstance(row.get("label_transfer"), dict) else {}
        tile_draw.text((8, 202), f"d={transfer.get('ahash_distance')}", fill=(0, 0, 0))
        tiles.append(tile)
    if not tiles:
        return
    cols = 4
    rows_count = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 240, rows_count * 220), (245, 245, 245))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % cols) * 240, (index // cols) * 220))
    sheet.save(out / "propagated_contact_sheet.jpg", quality=88)


def propagate(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "annotations").mkdir(parents=True, exist_ok=True)

    approved_statuses = {part.strip().lower() for part in args.approved_statuses.split(",") if part.strip()} or APPROVED_STATUSES
    seeds = load_seeds(args.approved, approved_statuses)
    stats: Counter[str] = Counter({"seed_rows": len(seeds)})
    exported: list[dict[str, Any]] = []
    matches: list[dict[str, Any]] = []
    seen_targets: set[str] = set()
    target_files = discover_targets(args.target, include_quarantine=args.include_quarantine)
    now = datetime.now(timezone.utc).isoformat()

    for manifest in target_files:
        for row_index, row in enumerate(load_target_rows(manifest)):
            image_path = resolve_image(row, manifest)
            if image_path is None:
                stats["missing_target_image"] += 1
                continue
            size = image_size(image_path)
            if size is None:
                stats["bad_target_image"] += 1
                continue
            width, height = size
            try:
                target_sha1 = file_sha1(image_path)
            except OSError:
                stats["bad_target_image"] += 1
                continue
            target = {
                "image_path": image_path,
                "width": width,
                "height": height,
                "sha1": target_sha1,
                "ahash": image_average_hash(image_path),
            }
            if target_sha1 in seen_targets:
                stats["duplicate_target_hash"] += 1
                continue
            seed, distance = best_seed_for_target(
                target,
                seeds,
                min_ahash_distance=args.min_ahash_distance,
                max_ahash_distance=args.max_ahash_distance,
                require_same_size=args.require_same_size,
            )
            if seed is None:
                stats["no_duplicate_seed"] += 1
                continue
            if args.skip_exact_hash and target_sha1 == seed["sha1"]:
                stats["exact_seed_hash_skipped"] += 1
                continue

            draft_boxes = draft_boxes_for_row(row, width, height)
            if not args.review_candidates and target_sha1 != seed["sha1"] and not draft_boxes:
                stats["missing_draft_overlap_evidence"] += 1
                continue
            propagated_boxes: list[dict[str, Any]] = []
            max_overlap = 0.0
            for seed_box in seed["boxes"]:
                scaled = scale_box(seed_box["bbox"], (seed["width"], seed["height"]), (width, height))
                if scaled is None:
                    continue
                if target_sha1 != seed["sha1"] and draft_boxes:
                    overlap = max((box_iou(scaled, draft) for draft in draft_boxes), default=0.0)
                    max_overlap = max(max_overlap, overlap)
                    if overlap < args.min_draft_iou:
                        stats["box_failed_draft_overlap"] += 1
                        continue
                source_box = seed_box["box"]
                box_status = "draft_review_required" if args.review_candidates else "accepted"
                row_status = "draft_review_required" if args.review_candidates else "approved"
                transfer_source = "label_transfer_review_candidate" if args.review_candidates else "label_transfer_duplicate"
                propagated_boxes.append(
                    {
                        **source_box,
                        "bbox_px": scaled,
                        "score": float(source_box.get("score") or 1.0),
                        "source": transfer_source,
                        "quality_flags": sorted(
                            set(
                                [
                                    *(source_box.get("quality_flags") or []),
                                    "label_transfer",
                                    *("review_candidate" for _ in [0] if args.review_candidates),
                                ]
                            )
                        ),
                        "annotation_status": box_status,
                        "review_note": f"auto-transferred from reviewed duplicate; ahash_distance={distance}",
                        "reviewed_at": now,
                        "label_transfer": {
                            "source_manifest": seed["manifest"],
                            "source_row_index": seed["row_index"],
                            "source_box_index": seed_box["box_index"],
                            "source_image": str(seed["image_path"]),
                            "source_sha1": seed["sha1"],
                            "source_ahash": seed["ahash"],
                            "target_sha1": target_sha1,
                            "target_ahash": target["ahash"],
                            "ahash_distance": distance,
                            "max_draft_iou": round(max_overlap, 6),
                            "method": "exact_or_ahash_duplicate_box_transfer_v1",
                        },
                    }
                )
            if not propagated_boxes:
                stats["no_boxes_propagated"] += 1
                continue

            target_name = f"{stable_id(target_sha1)}_{image_path.name}"
            copied_size = copy_image(image_path, args.out / "images" / target_name)
            if copied_size is None:
                stats["copy_failed"] += 1
                continue
            target_row = {
                **row,
                "image": f"images/{target_name}",
                "width": copied_size[0],
                "height": copied_size[1],
                "boxes": propagated_boxes,
                "annotation_status": row_status,
                "review_note": "review transferred boxes before promotion" if args.review_candidates else "auto-transferred from reviewed duplicate image",
                "review_id": f"propagated:{stable_id(str(manifest) + ':' + str(row_index) + ':' + target_sha1)}",
                "label_transfer": {
                    "target_manifest": str(manifest),
                    "target_row_index": row_index,
                    "target_source_image": str(image_path),
                    "source_manifest": seed["manifest"],
                    "source_row_index": seed["row_index"],
                    "source_image": str(seed["image_path"]),
                    "ahash_distance": distance,
                    "max_draft_iou": round(max_overlap, 6),
                },
            }
            exported.append(target_row)
            matches.append(
                {
                    "target_manifest": str(manifest),
                    "target_row_index": row_index,
                    "target_image": str(image_path),
                    "source_manifest": seed["manifest"],
                    "source_row_index": seed["row_index"],
                    "source_image": str(seed["image_path"]),
                    "ahash_distance": distance,
                    "max_draft_iou": round(max_overlap, 6),
                    "boxes": len(propagated_boxes),
                }
            )
            seen_targets.add(target_sha1)
            stats["propagated_images"] += 1
            stats["propagated_boxes"] += len(propagated_boxes)

    if args.review_candidates:
        write_jsonl(args.out / "annotations" / "draft_boxes.jsonl", exported)
        write_jsonl(args.out / "draft_boxes.jsonl", exported)
    else:
        write_jsonl(args.out / "annotations" / "approved_boxes.jsonl", exported)
        write_jsonl(args.out / "approved_boxes.jsonl", exported)
    write_jsonl(args.out / "propagation_matches.jsonl", matches)
    build_contact_sheet(exported, args.out)
    summary = {
        "generated_at": now,
        "approved_inputs": [str(path) for path in args.approved],
        "target_inputs": [str(path) for path in args.target],
        "target_files": [str(path) for path in target_files],
        "settings": {
            "max_ahash_distance": args.max_ahash_distance,
            "min_ahash_distance": args.min_ahash_distance,
            "require_same_size": bool(args.require_same_size),
            "skip_exact_hash": bool(args.skip_exact_hash),
            "min_draft_iou": args.min_draft_iou,
            "include_quarantine": bool(args.include_quarantine),
            "review_candidates": bool(args.review_candidates),
        },
        "stats": dict(sorted(stats.items())),
        "outputs": {
            "approved_boxes": "" if args.review_candidates else "annotations/approved_boxes.jsonl",
            "root_approved_boxes": "" if args.review_candidates else "approved_boxes.jsonl",
            "draft_boxes": "annotations/draft_boxes.jsonl" if args.review_candidates else "",
            "root_draft_boxes": "draft_boxes.jsonl" if args.review_candidates else "",
            "matches": "propagation_matches.jsonl",
            "contact_sheet": "propagated_contact_sheet.jpg",
        },
    }
    write_json(args.out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Propagate reviewed boxes to duplicate historical images.")
    parser.add_argument("--approved", type=Path, action="append", required=True, help="Approved JSONL/root to use as seed labels.")
    parser.add_argument("--target", type=Path, action="append", required=True, help="Draft prelabel/workbench roots or JSONL files to receive propagated labels.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--approved-statuses", default="approved,accepted,corrected,verified")
    parser.add_argument("--max-ahash-distance", type=int, default=0)
    parser.add_argument("--min-ahash-distance", type=int, default=0)
    parser.add_argument("--min-draft-iou", type=float, default=0.45)
    parser.add_argument("--include-quarantine", action="store_true")
    parser.add_argument("--review-candidates", action="store_true", help="Write draft_boxes.jsonl instead of approved_boxes.jsonl.")
    parser.add_argument("--allow-resize", dest="require_same_size", action="store_false")
    parser.set_defaults(require_same_size=True)
    parser.add_argument("--include-exact-hash", dest="skip_exact_hash", action="store_false")
    parser.set_defaults(skip_exact_hash=True)
    args = parser.parse_args()
    if not args.review_candidates and (args.max_ahash_distance > 0 or args.min_ahash_distance > 0):
        parser.error(
            "non-exact aHash propagation can only be emitted with --review-candidates; "
            "automatic approved propagation is limited to exact aHash matches"
        )
    if not args.review_candidates and not args.require_same_size:
        parser.error("--allow-resize can only be used with --review-candidates")
    summary = propagate(args)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "propagated_images": summary["stats"].get("propagated_images", 0),
                "propagated_boxes": summary["stats"].get("propagated_boxes", 0),
                "stats": summary["stats"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
