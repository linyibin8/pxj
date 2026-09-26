"""Merge approved question-box review exports into one audited package.

Multiple review workbenches can overlap. This helper copies referenced images,
groups identical images by file hash, deduplicates highly-overlapping approved
boxes, and writes annotations/approved_boxes.jsonl for question_detector_dataset.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


APPROVED_STATUSES = {"approved", "accepted", "corrected", "verified"}


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def discover_inputs(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            candidates = [
                path / "approved_boxes.jsonl",
                path / "annotations" / "approved_boxes.jsonl",
                path / "annotations" / "reviewed_boxes.jsonl",
            ]
            for candidate in candidates:
                if candidate.is_file():
                    files.append(candidate)
            files.extend(sorted(path.rglob("approved_boxes*.jsonl")))
    unique = {str(path.resolve()): path for path in files}
    return sorted(unique.values())


def resolve_image(row: dict[str, Any], manifest_path: Path) -> Path | None:
    raw = Path(str(row.get("image") or ""))
    if raw.is_absolute() and raw.is_file():
        return raw
    base = manifest_path.parent
    roots = [base, base.parent]
    if base.name == "annotations":
        roots.append(base.parent)
    for root in roots:
        candidates = [
            root / raw,
            root / "images" / raw.name,
            root.parent / raw,
            root.parent / "images" / raw.name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    return None


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


def normalize_box(raw: Any, width: int, height: int) -> dict[str, int] | None:
    if isinstance(raw, dict):
        try:
            x = float(raw.get("x"))
            y = float(raw.get("y"))
            w = float(raw.get("width"))
            h = float(raw.get("height"))
        except (TypeError, ValueError):
            return None
    elif isinstance(raw, list) and len(raw) >= 4:
        try:
            x, y, w, h = map(float, raw[:4])
        except (TypeError, ValueError):
            return None
    else:
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


def approved_status(value: Any, approved: set[str]) -> bool:
    return str(value or "").strip().lower() in approved


def copy_image(source: Path, target: Path) -> tuple[int, int] | None:
    try:
        with Image.open(source) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            width, height = normalized.size
            target.parent.mkdir(parents=True, exist_ok=True)
            normalized.save(target, format="JPEG", quality=88, optimize=True)
            return width, height
    except Exception:
        return None


def merge_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inputs = discover_inputs(args.inputs)
    approved = {part.strip().lower() for part in args.approved_statuses.split(",") if part.strip()} or APPROVED_STATUSES
    stats: Counter[str] = Counter()
    grouped: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for manifest in inputs:
        for row_index, row in enumerate(read_jsonl(manifest)):
            image_path = resolve_image(row, manifest)
            if image_path is None:
                stats["missing_image"] += 1
                continue
            try:
                image_hash = file_sha1(image_path)
            except OSError:
                stats["bad_image"] += 1
                continue
            if image_hash not in grouped:
                grouped[image_hash] = {
                    "source_image": image_path,
                    "source_rows": [],
                    "boxes": [],
                    "row": row,
                }
            item = grouped[image_hash]
            item["source_rows"].append({"manifest": str(manifest), "row_index": row_index, "image": row.get("image")})
            width = int(row.get("width") or 0)
            height = int(row.get("height") or 0)
            if width <= 0 or height <= 0:
                try:
                    with Image.open(image_path) as image:
                        width, height = ImageOps.exif_transpose(image).size
                except Exception:
                    stats["bad_image_size"] += 1
                    continue
            row_status = str(row.get("annotation_status") or "").strip().lower()
            for box_index, box in enumerate(row.get("boxes") or []):
                if not isinstance(box, dict):
                    continue
                status = str(box.get("annotation_status") or row_status).strip().lower()
                if not approved_status(status, approved):
                    stats["skipped_not_approved"] += 1
                    continue
                bbox = normalize_box(box.get("bbox_px") or box.get("bbox"), width, height)
                if bbox is None:
                    stats["bad_box"] += 1
                    continue
                candidate = dict(box)
                candidate["bbox_px"] = bbox
                candidate["annotation_status"] = status
                candidate.setdefault("review_note", row.get("review_note") or "")
                candidate.setdefault("quality_flags", [])
                candidate["merge_source"] = {"manifest": str(manifest), "row_index": row_index, "box_index": box_index}
                duplicate = False
                for existing in item["boxes"]:
                    iou = box_iou(existing["bbox_px"], bbox)
                    if iou >= args.dedupe_iou:
                        duplicate = True
                        stats["deduped_boxes"] += 1
                        break
                    if iou >= args.conflict_iou:
                        conflicts.append(
                            {
                                "image_hash": image_hash,
                                "iou": round(iou, 4),
                                "existing": existing.get("merge_source"),
                                "candidate": candidate.get("merge_source"),
                            }
                        )
                if duplicate:
                    continue
                item["boxes"].append(candidate)
                stats["accepted_boxes"] += 1
    output_rows: list[dict[str, Any]] = []
    for image_hash, item in sorted(grouped.items()):
        if not item["boxes"]:
            continue
        source = Path(item["source_image"])
        target_name = f"{image_hash[:16]}_{source.stem[:80]}.jpg"
        image_rel = f"images/{target_name}"
        size = copy_image(source, args.out / image_rel)
        if size is None:
            stats["copy_failed"] += 1
            continue
        width, height = size
        base = dict(item["row"])
        base.update(
            {
                "image": image_rel,
                "source_candidate": base.get("source_candidate") or str(source),
                "width": width,
                "height": height,
                "boxes": item["boxes"],
                "annotation_status": "approved",
                "review_id": f"merged:{image_hash[:16]}",
                "review_note": "merged reviewed labels",
                "merge_metadata": {
                    "image_hash": image_hash,
                    "source_rows": item["source_rows"],
                    "box_count": len(item["boxes"]),
                },
            }
        )
        output_rows.append(base)
        stats["images"] += 1
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(path) for path in inputs],
        "approved_statuses": sorted(approved),
        "dedupe_iou": args.dedupe_iou,
        "conflict_iou": args.conflict_iou,
        "stats": dict(stats),
        "conflict_count": len(conflicts),
        "conflicts": conflicts[: args.max_conflicts],
        "outputs": {
            "approved_boxes": "annotations/approved_boxes.jsonl",
            "root_approved_boxes": "approved_boxes.jsonl",
            "contact_sheet": "preview_contact_sheet.jpg",
        },
    }
    return output_rows, summary


def build_contact_sheet(out: Path, rows: list[dict[str, Any]], limit: int = 48) -> None:
    if not rows:
        return
    thumb_w, thumb_h, columns = 320, 220, 4
    selected = rows[:limit]
    sheet = Image.new("RGB", (columns * thumb_w, math.ceil(len(selected) / columns) * thumb_h), "white")
    for index, row in enumerate(selected):
        path = out / str(row["image"])
        try:
            with Image.open(path) as image:
                preview = ImageOps.exif_transpose(image).convert("RGB")
        except Exception:
            continue
        original_w, original_h = preview.size
        preview.thumbnail((thumb_w, thumb_h - 24), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, thumb_h), "white")
        offset_x = (thumb_w - preview.width) // 2
        offset_y = 20
        canvas.paste(preview, (offset_x, offset_y))
        draw = ImageDraw.Draw(canvas)
        sx = preview.width / max(1, original_w)
        sy = preview.height / max(1, original_h)
        for box in row.get("boxes") or []:
            b = box["bbox_px"]
            x = offset_x + b["x"] * sx
            y = offset_y + b["y"] * sy
            draw.rectangle((x, y, x + b["width"] * sx, y + b["height"] * sy), outline=(40, 180, 70), width=2)
        draw.text((6, 4), f"{len(row.get('boxes') or [])} boxes", fill=(20, 20, 20))
        sheet.paste(canvas, ((index % columns) * thumb_w, (index // columns) * thumb_h))
    sheet.save(out / "preview_contact_sheet.jpg", format="JPEG", quality=90, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge approved question detector review exports.")
    parser.add_argument("inputs", type=Path, nargs="+", help="Approved JSONL files or workbench/package directories.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--approved-statuses", default="approved,accepted,corrected,verified")
    parser.add_argument("--dedupe-iou", type=float, default=0.95)
    parser.add_argument("--conflict-iou", type=float, default=0.45)
    parser.add_argument("--max-conflicts", type=int, default=100)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, summary = merge_rows(args)
    write_jsonl(args.out / "annotations" / "approved_boxes.jsonl", rows)
    write_jsonl(args.out / "approved_boxes.jsonl", rows)
    write_json(args.out / "summary.json", summary)
    build_contact_sheet(args.out, rows)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "images": len(rows),
                "boxes": sum(len(row.get("boxes") or []) for row in rows),
                "deduped_boxes": summary["stats"].get("deduped_boxes", 0),
                "conflict_count": summary["conflict_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
