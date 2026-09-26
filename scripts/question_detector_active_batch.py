"""Select high-value review batches from draft question-box prelabels.

The goal is to spend human review time where it most improves the detector:
diverse sessions, uncertain/risky boxes, and low visual duplication. The output
is another prelabel-style root that can be opened directly with
question_detector_review_workbench.py.
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
from statistics import median
from typing import Any

from PIL import Image, ImageDraw, ImageOps


RISK_FLAGS = {
    "near_full_page": 0.35,
    "tall_block": 0.25,
    "short_block": 0.20,
    "few_text_lines": 0.18,
    "single_union_fallback": 0.18,
    "page:full_image_fallback": 0.35,
}


@dataclass
class Candidate:
    row: dict[str, Any]
    image_path: Path
    manifest_name: str
    session_id: str
    batch_id: str
    image_id: str
    sha1: str
    ahash: str
    score: float
    reasons: list[str]


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


def normalized_image_name(value: str) -> str:
    return Path(str(value or "")).name


def approved_image_names(paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            image = str(row.get("image") or "")
            if image:
                names.add(normalized_image_name(image))
            candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
            source = str(candidate.get("image") or row.get("source_candidate") or "")
            if source:
                names.add(normalized_image_name(source))
    return names


def queued_review_image_names(paths: list[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        files: list[Path] = []
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            candidates = [
                path / "annotations" / "draft_boxes.jsonl",
                path / "draft_boxes.jsonl",
                path / "annotations" / "ranked_candidates.jsonl",
            ]
            files.extend(candidate for candidate in candidates if candidate.is_file())
        for file_path in files:
            for row in read_jsonl(file_path):
                if row.get("selected") is False:
                    continue
                image = str(row.get("image") or "")
                if image:
                    names.add(normalized_image_name(image))
                candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
                source = str(candidate.get("image") or row.get("source_candidate") or "")
                if source:
                    names.add(normalized_image_name(source))
    return names


def row_ids(row: dict[str, Any]) -> tuple[str, str, str]:
    candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
    session_id = str(row.get("session_id") or candidate.get("session_id") or "").strip()
    batch_id = str(row.get("batch_id") or candidate.get("batch_id") or "").strip()
    image_id = str(row.get("source_image_id") or candidate.get("image_id") or normalized_image_name(str(row.get("image") or ""))).strip()
    return session_id, batch_id, image_id


def box_area(box: dict[str, Any], width: int, height: int) -> float:
    bbox = box.get("bbox_px") if isinstance(box.get("bbox_px"), dict) else {}
    try:
        return float(bbox.get("width")) * float(bbox.get("height")) / max(1, width * height)
    except (TypeError, ValueError):
        return 0.0


def score_row(row: dict[str, Any], manifest_name: str) -> tuple[float, list[str]]:
    boxes = [box for box in row.get("boxes") or [] if isinstance(box, dict)]
    width = max(1, int(row.get("width") or 0))
    height = max(1, int(row.get("height") or 0))
    scores = []
    flags: list[str] = []
    for box in boxes:
        try:
            scores.append(float(box.get("score") or 0.0))
        except (TypeError, ValueError):
            pass
        flags.extend(str(flag) for flag in box.get("quality_flags") or [])
    areas = [box_area(box, width, height) for box in boxes]
    reasons: list[str] = []
    uncertainty = 1.0 - (median(scores) if scores else 0.5)
    risk = 0.0
    for flag in flags:
        for key, value in RISK_FLAGS.items():
            if flag == key or flag.startswith(key):
                risk += value
                reasons.append(f"flag:{flag}")
    if len(boxes) >= 3:
        risk += 0.25
        reasons.append("multi_box_page")
    if any(area >= 0.65 for area in areas):
        risk += 0.18
        reasons.append("large_box")
    if any(0 < area <= 0.04 for area in areas):
        risk += 0.18
        reasons.append("small_box")
    if manifest_name == "quarantine.jsonl":
        risk += 0.55
        reasons.append("quarantine")
    line_count = int((row.get("metadata") or {}).get("line_box_count") or 0)
    if line_count >= 90:
        risk += 0.12
        reasons.append("dense_text")
    score = len(boxes) * 0.12 + uncertainty * 0.35 + min(1.4, risk)
    return round(score, 6), sorted(set(reasons))


def load_candidates(
    prelabel_root: Path,
    include_quarantine: bool,
    approved_names: set[str],
    queued_names: set[str],
) -> tuple[list[Candidate], dict[str, int]]:
    manifests = ["draft_boxes.jsonl"]
    if include_quarantine:
        manifests.append("quarantine.jsonl")
    candidates: list[Candidate] = []
    skipped: dict[str, int] = defaultdict(int)
    hash_cache: dict[Path, tuple[str, str]] = {}
    for manifest_name in manifests:
        path = prelabel_root / "annotations" / manifest_name
        for row in read_jsonl(path):
            if not row.get("boxes"):
                skipped["no_boxes"] += 1
                continue
            image_rel = str(row.get("image") or "")
            image_name = normalized_image_name(image_rel)
            if image_name in approved_names:
                skipped["already_approved_image"] += 1
                continue
            if image_name in queued_names:
                skipped["already_queued_image"] += 1
                continue
            image_path = prelabel_root / image_rel
            if not image_path.is_file():
                skipped["missing_image"] += 1
                continue
            if image_path not in hash_cache:
                try:
                    hash_cache[image_path] = (file_sha1(image_path), image_average_hash(image_path))
                except OSError:
                    skipped["unreadable_image"] += 1
                    continue
            sha1, ahash = hash_cache[image_path]
            session_id, batch_id, image_id = row_ids(row)
            score, reasons = score_row(row, manifest_name)
            candidates.append(
                Candidate(
                    row={**row, "active_learning": {"score": score, "reasons": reasons, "source_manifest": manifest_name}},
                    image_path=image_path,
                    manifest_name=manifest_name,
                    session_id=session_id,
                    batch_id=batch_id,
                    image_id=image_id,
                    sha1=sha1,
                    ahash=ahash,
                    score=score,
                    reasons=reasons,
                )
            )
    return candidates, dict(sorted(skipped.items()))


def select_candidates(args: argparse.Namespace, candidates: list[Candidate]) -> tuple[list[Candidate], list[dict[str, Any]]]:
    selected: list[Candidate] = []
    ranked: list[dict[str, Any]] = []
    by_session: dict[str, int] = defaultdict(int)
    seen_sha1: set[str] = set()
    selected_box_count = 0
    for candidate in sorted(candidates, key=lambda item: (-item.score, item.session_id, item.image_id)):
        skip_reason = ""
        session_key = candidate.session_id or "unknown"
        candidate_box_count = len(candidate.row.get("boxes") or [])
        if not args.allow_exact_duplicates and candidate.sha1 in seen_sha1:
            skip_reason = "exact_duplicate_image"
        elif args.max_per_session > 0 and by_session[session_key] >= args.max_per_session:
            skip_reason = "session_quota"
        else:
            for existing in selected:
                if hex_hamming(candidate.ahash, existing.ahash) <= args.ahash_threshold:
                    skip_reason = "near_duplicate_image"
                    break
        if not skip_reason and len(selected) >= args.limit:
            skip_reason = "image_quota"
        if (
            not skip_reason
            and args.max_boxes_total > 0
            and selected
            and selected_box_count + candidate_box_count > args.max_boxes_total
        ):
            skip_reason = "box_quota"
        is_selected = not skip_reason
        ranked.append(
            {
                "image": candidate.row.get("image") or "",
                "score": candidate.score,
                "selected": is_selected,
                "skip_reason": skip_reason,
                "session_id": candidate.session_id,
                "batch_id": candidate.batch_id,
                "image_id": candidate.image_id,
                "source_manifest": candidate.manifest_name,
                "reasons": candidate.reasons,
                "box_count": candidate_box_count,
                "selected_box_count_before": selected_box_count,
            }
        )
        if skip_reason:
            continue
        selected.append(candidate)
        selected_box_count += candidate_box_count
        by_session[session_key] += 1
        seen_sha1.add(candidate.sha1)
    return selected, ranked


def copy_selected(prelabel_root: Path, out: Path, selected: list[Candidate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected, start=1):
        target_rel = (Path("images") / candidate.image_path.name).as_posix()
        target = out / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate.image_path, target)
        rows.append(
            {
                **candidate.row,
                "image": target_rel,
                "review_priority": index,
            }
        )
    return rows


def build_contact_sheet(out: Path, rows: list[dict[str, Any]], filename: str = "active_batch_contact_sheet.jpg") -> None:
    tiles: list[Image.Image] = []
    colors = [(40, 170, 80), (225, 120, 40), (60, 120, 220), (190, 70, 170)]
    for row in rows[:80]:
        path = out / str(row.get("image") or "")
        if not path.is_file():
            continue
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        scale = min(260 / image.width, 190 / image.height)
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
            draw.text((x + 3, y + 3), str(index + 1), fill=color)
        tile = Image.new("RGB", (280, 235), "white")
        tile.paste(thumb, ((280 - thumb.width) // 2, 8))
        meta = row.get("active_learning") if isinstance(row.get("active_learning"), dict) else {}
        label = f"p{row.get('review_priority')} score={float(meta.get('score') or 0):.2f} boxes={len(row.get('boxes') or [])}"
        ImageDraw.Draw(tile).text((8, 205), label, fill=(0, 0, 0))
        tiles.append(tile)
    if not tiles:
        return
    cols = 4
    rows_count = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 280, rows_count * 235), (245, 245, 245))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % cols) * 280, (index // cols) * 235))
    sheet.save(out / filename, quality=88)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    approved_names = approved_image_names(args.exclude_approved_jsonl)
    queued_names = queued_review_image_names(args.exclude_queued_root)
    candidates, skipped = load_candidates(args.prelabel_root, args.include_quarantine, approved_names, queued_names)
    selected, ranked = select_candidates(args, candidates)
    selected_rows = copy_selected(args.prelabel_root, args.out, selected)
    write_jsonl(args.out / "annotations" / "draft_boxes.jsonl", selected_rows)
    write_jsonl(args.out / "annotations" / "ranked_candidates.jsonl", ranked)
    build_contact_sheet(args.out, selected_rows)
    session_counts: dict[str, int] = defaultdict(int)
    for item in selected:
        session_counts[item.session_id or "unknown"] += 1
    reason_counts: dict[str, int] = defaultdict(int)
    for item in selected:
        for reason in item.reasons:
            reason_counts[reason] += 1
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prelabel_root": str(args.prelabel_root),
        "out": str(args.out),
        "candidate_count": len(candidates),
        "selected_images": len(selected_rows),
        "selected_boxes": sum(len(row.get("boxes") or []) for row in selected_rows),
        "limit": args.limit,
        "include_quarantine": bool(args.include_quarantine),
        "excluded_approved_images": len(approved_names),
        "excluded_queued_images": len(queued_names),
        "skipped": skipped,
        "selection": {
            "max_per_session": args.max_per_session,
            "max_boxes_total": args.max_boxes_total,
            "ahash_threshold": args.ahash_threshold,
            "session_counts": dict(sorted(session_counts.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
        "outputs": {
            "draft_boxes": "annotations/draft_boxes.jsonl",
            "ranked_candidates": "annotations/ranked_candidates.jsonl",
            "contact_sheet": "active_batch_contact_sheet.jpg",
        },
        "notes": [
            "This is a review-prioritization batch, not ground truth.",
            "Open this root with question_detector_review_workbench.py, then export approved boxes.",
        ],
    }
    write_json(args.out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Select a diverse active-learning review batch from prelabels.")
    parser.add_argument("--prelabel-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-active-batch"))
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--max-per-session", type=int, default=4)
    parser.add_argument("--max-boxes-total", type=int, default=0, help="Soft cap on selected draft boxes; 0 disables the cap.")
    parser.add_argument("--ahash-threshold", type=int, default=4)
    parser.add_argument("--include-quarantine", action="store_true")
    parser.add_argument("--allow-exact-duplicates", action="store_true")
    parser.add_argument("--exclude-approved-jsonl", type=Path, action="append", default=[], help="Approved JSONL files whose images should be skipped.")
    parser.add_argument("--exclude-queued-root", type=Path, action="append", default=[], help="Previously generated review batch/workbench roots or JSONL files whose images should be skipped while they are awaiting review.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
