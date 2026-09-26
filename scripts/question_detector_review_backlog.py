"""Build one prioritized review backlog from multiple workbenches.

This does not create ground truth. It copies pending review items from existing
workbenches into a single prelabel-style root, ranks them by training value, and
keeps source metadata so approved exports can be traced back to the original
queue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


SOURCE_WEIGHTS = {
    "fallback_delta": 1.25,
    "active": 1.0,
    "propagation": 0.75,
    "error_mining": 0.55,
    "other": 0.45,
}

FLAG_WEIGHTS = {
    "fallback_delta_removed": 0.55,
    "quarantine": 0.45,
    "few_text_lines": 0.28,
    "short_block": 0.22,
    "tall_block": 0.22,
    "near_full_page": 0.22,
    "page:bright_page": 0.08,
}


@dataclass
class BacklogItem:
    row: dict[str, Any]
    image_path: Path
    source_root: Path
    source_kind: str
    review_id: str
    sha1: str
    ahash: str
    score: float
    reasons: list[str]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_average_hash(path: Path, hash_size: int = 8) -> str:
    try:
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    except Exception:
        return ""
    pixels = list(image.getdata())
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


def infer_source_kind(root: Path, review_data: dict[str, Any]) -> str:
    text = f"{root} {review_data.get('source') or ''}".lower()
    if "fallback-delta" in text or "fallback_delta" in text or "policy_delta" in text:
        return "fallback_delta"
    if "active-batch" in text or "active_batch" in text:
        return "active"
    if "propagation" in text:
        return "propagation"
    if "error-mining" in text or "error_mining" in text:
        return "error_mining"
    return "other"


def box_area(box: dict[str, Any], width: int, height: int) -> float:
    bbox = box.get("bbox_px") if isinstance(box.get("bbox_px"), dict) else {}
    try:
        return float(bbox.get("width")) * float(bbox.get("height")) / max(1, width * height)
    except (TypeError, ValueError):
        return 0.0


def score_item(item: dict[str, Any], source_kind: str) -> tuple[float, list[str]]:
    boxes = [box for box in item.get("boxes") or [] if isinstance(box, dict)]
    width = max(1, int(item.get("width") or 0))
    height = max(1, int(item.get("height") or 0))
    reasons = [f"source:{source_kind}"]
    score = SOURCE_WEIGHTS.get(source_kind, SOURCE_WEIGHTS["other"])
    score += min(1.0, len(boxes) * 0.12)
    for box in boxes:
        status = str(box.get("default_review_status") or "pending").lower()
        if status == "rejected":
            score -= 0.18
            reasons.append("default_rejected")
        try:
            confidence = float(box.get("score") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if 0 < confidence < 0.45:
            score += 0.18
            reasons.append("low_confidence")
        area = box_area(box, width, height)
        if area >= 0.30:
            score += 0.20
            reasons.append("large_candidate")
        elif 0 < area < 0.04:
            score += 0.16
            reasons.append("small_candidate")
        for flag in box.get("quality_flags") or []:
            flag = str(flag)
            if flag in FLAG_WEIGHTS:
                score += FLAG_WEIGHTS[flag]
                reasons.append(f"flag:{flag}")
    return round(score, 6), sorted(set(reasons))


def resolve_workbench_image(root: Path, image: str) -> Path | None:
    raw = Path(image)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    candidates.extend(
        [
            root / raw,
            root / "images" / raw.name,
            root.parent / raw,
            root.parent / "images" / raw.name,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_workbench(root: Path) -> tuple[list[BacklogItem], dict[str, int]]:
    skipped: dict[str, int] = defaultdict(int)
    review_path = root / "review_data.json"
    if not review_path.is_file():
        skipped["missing_review_data"] += 1
        return [], dict(skipped)
    review_data = read_json(review_path)
    source_kind = infer_source_kind(root, review_data if isinstance(review_data, dict) else {})
    items = review_data.get("items") if isinstance(review_data.get("items"), list) else []
    out: list[BacklogItem] = []
    for item in items:
        if not isinstance(item, dict):
            skipped["invalid_item"] += 1
            continue
        boxes = [box for box in item.get("boxes") or [] if isinstance(box, dict)]
        if not boxes:
            skipped["no_boxes"] += 1
            continue
        image = str(item.get("image") or "")
        image_path = resolve_workbench_image(root, image)
        if image_path is None:
            skipped["missing_image"] += 1
            continue
        try:
            sha1 = file_sha1(image_path)
            ahash = image_average_hash(image_path)
        except OSError:
            skipped["unreadable_image"] += 1
            continue
        score, reasons = score_item(item, source_kind)
        out.append(
            BacklogItem(
                row=item,
                image_path=image_path,
                source_root=root,
                source_kind=source_kind,
                review_id=str(item.get("review_id") or image_path.name),
                sha1=sha1,
                ahash=ahash,
                score=score,
                reasons=reasons,
            )
        )
    return out, dict(skipped)


def select_items(items: list[BacklogItem], limit: int, ahash_threshold: int, allow_exact_duplicates: bool) -> tuple[list[BacklogItem], list[dict[str, Any]]]:
    selected: list[BacklogItem] = []
    ranked: list[dict[str, Any]] = []
    seen_sha1: set[str] = set()
    for item in sorted(items, key=lambda row: (-row.score, row.source_kind, row.image_path.name)):
        skip_reason = ""
        if not allow_exact_duplicates and item.sha1 in seen_sha1:
            skip_reason = "exact_duplicate_image"
        else:
            for kept in selected:
                if hex_hamming(item.ahash, kept.ahash) <= ahash_threshold:
                    skip_reason = "near_duplicate_image"
                    break
        if not skip_reason and len(selected) >= limit:
            skip_reason = "over_limit"
        is_selected = not skip_reason
        ranked.append(
            {
                "image": item.row.get("image") or "",
                "source_workbench": str(item.source_root),
                "source_kind": item.source_kind,
                "source_review_id": item.review_id,
                "score": item.score,
                "selected": is_selected,
                "skip_reason": skip_reason,
                "reasons": item.reasons,
                "box_count": len(item.row.get("boxes") or []),
            }
        )
        if not is_selected:
            continue
        selected.append(item)
        seen_sha1.add(item.sha1)
    return selected, ranked


def copy_selected(out: Path, selected: list[BacklogItem]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for priority, item in enumerate(selected, start=1):
        image_name = f"{priority:03d}_{item.source_kind}_{item.image_path.name}"
        image_rel = (Path("images") / image_name).as_posix()
        target = out / image_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.image_path, target)
        rows.append(
            {
                **item.row,
                "image": image_rel,
                "review_priority": priority,
                "backlog": {
                    "score": item.score,
                    "reasons": item.reasons,
                    "source_kind": item.source_kind,
                    "source_workbench": str(item.source_root),
                    "source_review_id": item.review_id,
                },
                "metadata": {
                    **(item.row.get("metadata") if isinstance(item.row.get("metadata"), dict) else {}),
                    "backlog_source_kind": item.source_kind,
                    "backlog_source_workbench": str(item.source_root),
                    "backlog_source_review_id": item.review_id,
                },
            }
        )
    return rows


def build_contact_sheet(out: Path, rows: list[dict[str, Any]], filename: str = "review_backlog_contact_sheet.jpg") -> None:
    tiles: list[Image.Image] = []
    colors = [(40, 150, 90), (40, 110, 220), (220, 130, 30), (190, 70, 170)]
    for row in rows[:96]:
        image_path = out / str(row.get("image") or "")
        if not image_path.is_file():
            continue
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        scale = min(300 / image.width, 210 / image.height)
        thumb = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(thumb)
        for index, box in enumerate(row.get("boxes") or []):
            bbox = box.get("bbox_px") if isinstance(box.get("bbox_px"), dict) else {}
            try:
                x = float(bbox.get("x")) * scale
                y = float(bbox.get("y")) * scale
                w = float(bbox.get("width")) * scale
                h = float(bbox.get("height")) * scale
            except (TypeError, ValueError):
                continue
            color = colors[index % len(colors)]
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            draw.text((x + 4, y + 4), str(index + 1), fill=color)
        tile = Image.new("RGB", (320, 260), "white")
        tile.paste(thumb, ((320 - thumb.width) // 2, 8))
        meta = row.get("backlog") if isinstance(row.get("backlog"), dict) else {}
        label = f"p{row.get('review_priority')} {meta.get('source_kind')} score={float(meta.get('score') or 0):.2f} boxes={len(row.get('boxes') or [])}"
        ImageDraw.Draw(tile).text((8, 224), label[:58], fill=(0, 0, 0))
        tiles.append(tile)
    if not tiles:
        return
    cols = 4
    rows_count = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 320, rows_count * 260), (245, 245, 245))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % cols) * 320, (index // cols) * 260))
    sheet.save(out / filename, quality=88)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    all_items: list[BacklogItem] = []
    skipped_by_root: dict[str, dict[str, int]] = {}
    for root in args.workbench:
        items, skipped = load_workbench(root)
        all_items.extend(items)
        skipped_by_root[str(root)] = skipped
    selected, ranked = select_items(all_items, args.limit, args.ahash_threshold, args.allow_exact_duplicates)
    rows = copy_selected(args.out, selected)
    write_jsonl(args.out / "annotations" / "draft_boxes.jsonl", rows)
    write_jsonl(args.out / "annotations" / "ranked_backlog.jsonl", ranked)
    build_contact_sheet(args.out, rows)
    candidate_source_counts = Counter(item.source_kind for item in all_items)
    source_counts = Counter(item.source_kind for item in selected)
    reason_counts = Counter(reason for item in selected for reason in item.reasons)
    skip_reason_counts = Counter(str(row.get("skip_reason") or "selected") for row in ranked)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "out": str(args.out),
        "workbenches": [str(path) for path in args.workbench],
        "candidate_images": len(all_items),
        "selected_images": len(rows),
        "selected_boxes": sum(len(row.get("boxes") or []) for row in rows),
        "limit": args.limit,
        "selection": {
            "ahash_threshold": args.ahash_threshold,
            "candidate_source_counts": dict(sorted(candidate_source_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "skip_reason_counts": dict(sorted(skip_reason_counts.items())),
        },
        "skipped_by_root": skipped_by_root,
        "outputs": {
            "draft_boxes": "annotations/draft_boxes.jsonl",
            "ranked_backlog": "annotations/ranked_backlog.jsonl",
            "contact_sheet": "review_backlog_contact_sheet.jpg",
        },
        "notes": [
            "This is a prioritized review queue, not approved training data.",
            "Build a workbench from this root, review boxes, then export approved JSONL.",
        ],
    }
    write_json(args.out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge pending workbenches into one prioritized review backlog.")
    parser.add_argument("--workbench", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-review-backlog"))
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--ahash-threshold", type=int, default=2)
    parser.add_argument("--allow-exact-duplicates", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
