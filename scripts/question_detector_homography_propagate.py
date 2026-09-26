"""Propagate verified teacher boxes to visually similar photos with homography.

This grows review candidates from a tiny set of trusted teacher boxes. It uses
ORB feature matching and RANSAC homography to map each verified box corner from
seed image to target image. Outputs are always candidate/draft rows, never
approved gold labels.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class Seed:
    row: dict[str, Any]
    image_path: Path
    width: int
    height: int
    boxes: list[dict[str, Any]]
    gray: np.ndarray
    keypoints: list[Any]
    descriptors: np.ndarray | None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def clean_out(path: Path, clean: bool) -> None:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def image_gray(path: Path, max_side: int) -> tuple[np.ndarray, tuple[int, int], float]:
    image = open_rgb(path)
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return gray, (width, height), scale


def orb_features(gray: np.ndarray, nfeatures: int) -> tuple[list[Any], np.ndarray | None]:
    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=7, edgeThreshold=15)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints or [], descriptors


def resolve_image(raw: Any, base: Path) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([base / path, base.parent / path, base / "images" / path.name, base.parent / "images" / path.name])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def bbox(raw: Any, width: int, height: int) -> dict[str, int] | None:
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


def load_seeds(package: Path, max_side: int, nfeatures: int) -> list[Seed]:
    seed_path = package / "teacher_reference_boxes.jsonl"
    if not seed_path.is_file():
        seed_path = package / "annotations" / "approved_boxes.jsonl"
    if not seed_path.is_file():
        raise SystemExit(f"teacher package has no reference boxes: {package}")
    seeds: list[Seed] = []
    for row in read_jsonl(seed_path):
        image_path = resolve_image(row.get("image") or row.get("image_path") or row.get("source_candidate"), package)
        if image_path is None:
            continue
        gray, (width, height), _ = image_gray(image_path, max_side)
        keypoints, descriptors = orb_features(gray, nfeatures)
        boxes: list[dict[str, Any]] = []
        for box in row.get("boxes") or []:
            if not isinstance(box, dict):
                continue
            b = bbox(box.get("bbox_px") or box.get("bbox"), width, height)
            if b:
                boxes.append({**box, "bbox_px": b})
        if boxes and descriptors is not None and len(keypoints) >= 20:
            seeds.append(Seed(row=row, image_path=image_path, width=width, height=height, boxes=boxes, gray=gray, keypoints=keypoints, descriptors=descriptors))
    return seeds


def target_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED)


def filtered_target_images(image_dir: Path, includes: list[str], excludes: list[str]) -> list[Path]:
    paths = target_images(image_dir)
    if includes:
        paths = [path for path in paths if any(token in path.name for token in includes)]
    if excludes:
        paths = [path for path in paths if not any(token in path.name for token in excludes)]
    return paths


def match_homography(seed: Seed, target_gray: np.ndarray, target_scale: float, args: argparse.Namespace) -> dict[str, Any] | None:
    target_keypoints, target_descriptors = orb_features(target_gray, args.nfeatures)
    if target_descriptors is None or seed.descriptors is None or len(target_keypoints) < args.min_keypoints:
        return None
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = matcher.knnMatch(seed.descriptors, target_descriptors, k=2)
    good = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < args.ratio * second.distance:
            good.append(first)
    if len(good) < args.min_matches:
        return None

    src = np.float32([seed.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([target_keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, args.ransac_reproj_threshold)
    if homography is None or mask is None:
        return None
    inliers = int(mask.ravel().sum())
    inlier_ratio = inliers / max(1, len(good))
    if inliers < args.min_inliers or inlier_ratio < args.min_inlier_ratio:
        return None
    return {
        "homography": homography,
        "matches": len(good),
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "target_keypoints": len(target_keypoints),
        "target_scale": target_scale,
    }


def transform_box(box: dict[str, int], homography: np.ndarray, target_size: tuple[int, int], seed_scale: float, target_scale: float) -> dict[str, int] | None:
    target_w, target_h = target_size
    x, y, w, h = box["x"], box["y"], box["width"], box["height"]
    pts = np.float32(
        [
            [x * seed_scale, y * seed_scale],
            [(x + w) * seed_scale, y * seed_scale],
            [(x + w) * seed_scale, (y + h) * seed_scale],
            [x * seed_scale, (y + h) * seed_scale],
        ]
    ).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
    warped = warped / max(1e-6, target_scale)
    min_x = float(np.min(warped[:, 0]))
    min_y = float(np.min(warped[:, 1]))
    max_x = float(np.max(warped[:, 0]))
    max_y = float(np.max(warped[:, 1]))
    out = {
        "x": int(round(max(0, min(target_w, min_x)))),
        "y": int(round(max(0, min(target_h, min_y)))),
        "width": int(round(max(0, min(target_w, max_x) - max(0, min(target_w, min_x))))),
        "height": int(round(max(0, min(target_h, max_y) - max(0, min(target_h, min_y))))),
    }
    area = out["width"] * out["height"] / max(1, target_w * target_h)
    if out["width"] <= 8 or out["height"] <= 8 or area < 0.006 or area > 0.55:
        return None
    if out["x"] <= 2 or out["y"] <= 2 or out["x"] + out["width"] >= target_w - 2 or out["y"] + out["height"] >= target_h - 2:
        return None
    return out


def draw_preview(row: dict[str, Any], target: Path) -> None:
    image = open_rgb(Path(row["source_candidate"]))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", max(18, image.width // 130))
    except Exception:
        font = ImageFont.load_default()
    for index, box_row in enumerate(row.get("boxes") or [], start=1):
        box = box_row["bbox_px"]
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        color = (255, 140, 0)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=max(3, image.width // 900))
        label = str(box_row.get("question_label") or index)
        draw.rectangle([x, max(0, y - 34), x + 96, y], fill=color)
        draw.text((x + 8, max(0, y - 30)), label, fill=(255, 255, 255), font=font)
    image.thumbnail((1500, 1500))
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, quality=90)


def contact_sheet(previews: list[Path], target: Path, limit: int) -> None:
    previews = previews[:limit]
    if not previews:
        return
    thumbs: list[Image.Image] = []
    labels: list[str] = []
    for preview in previews:
        with Image.open(preview) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((760, 570))
            thumbs.append(thumb.copy())
            labels.append(preview.name[:78])
    columns = 2
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * 780, rows * 610), "white")
    draw = ImageDraw.Draw(sheet)
    for index, thumb in enumerate(thumbs):
        x = (index % columns) * 780 + 10
        y = (index // columns) * 610 + 10
        sheet.paste(thumb, (x, y))
        draw.text((x, y + 575), labels[index], fill=(0, 0, 0))
    sheet.save(target, quality=90)


def propagate(args: argparse.Namespace) -> dict[str, Any]:
    clean_out(args.out, args.clean)
    seeds = load_seeds(args.teacher_package, args.max_side, args.nfeatures)
    stats: Counter[str] = Counter({"seeds": len(seeds)})
    rows: list[dict[str, Any]] = []
    previews: list[Path] = []
    seed_paths = {str(seed.image_path.resolve()).lower() for seed in seeds}
    seed_stems = {seed.image_path.stem for seed in seeds}
    for seed in seeds:
        source_candidate = str(seed.row.get("source_candidate") or "").strip()
        if source_candidate:
            seed_stems.add(Path(source_candidate).stem)

    for target in filtered_target_images(args.image_dir, args.include_name, args.exclude_name):
        is_seed_source = str(target.resolve()).lower() in seed_paths or target.stem in seed_stems
        if is_seed_source and args.skip_seed_images:
            stats["skipped_seed_image"] += 1
            continue
        target_gray, target_size, target_scale = image_gray(target, args.max_side)
        best: tuple[Seed, dict[str, Any]] | None = None
        for seed in seeds:
            seed_scale = min(1.0, args.max_side / max(seed.width, seed.height))
            match = match_homography(seed, target_gray, target_scale, args)
            if match is None:
                continue
            score = match["inliers"] * match["inlier_ratio"]
            if best is None or score > best[1]["inliers"] * best[1]["inlier_ratio"]:
                match["seed_scale"] = seed_scale
                best = (seed, match)
        if best is None:
            stats["no_homography"] += 1
            continue
        seed, match = best
        if match["inlier_ratio"] < args.promote_min_inlier_ratio or match["inliers"] < args.promote_min_inliers:
            stats["weak_homography_rejected"] += 1
            continue
        mapped_boxes: list[dict[str, Any]] = []
        for box in seed.boxes:
            mapped = transform_box(box["bbox_px"], match["homography"], target_size, match["seed_scale"], target_scale)
            if mapped is None:
                stats["box_transform_rejected"] += 1
                continue
            mapped_boxes.append(
                {
                    **box,
                    "bbox_px": mapped,
                    "annotation_status": "candidate",
                    "default_review_status": "pending",
                    "score": round(min(0.88, 0.45 + match["inlier_ratio"] * 0.35), 3),
                    "quality_flags": sorted(set([*(box.get("quality_flags") or []), "homography_transfer_candidate"])),
                    "teacher": "codex_homography_transfer",
                    "teacher_note": f"Transferred from {seed.image_path.name}; inliers={match['inliers']}; ratio={match['inlier_ratio']:.3f}",
                }
            )
        if not mapped_boxes:
            stats["no_boxes_mapped"] += 1
            continue
        row = {
            "review_id": f"homography:{target.stem}",
            "image": str(target),
            "source_candidate": str(target),
            "width": target_size[0],
            "height": target_size[1],
            "annotation_status": "draft_review_required",
            "teacher": "codex_homography_transfer",
            "cohort": "homography_transfer_candidate",
            "boxes": mapped_boxes,
            "metadata": {
                "source_seed": str(seed.image_path),
                "matches": match["matches"],
                "inliers": match["inliers"],
                "inlier_ratio": round(match["inlier_ratio"], 4),
            },
        }
        rows.append(row)
        stats["candidate_images"] += 1
        stats["candidate_boxes"] += len(mapped_boxes)
        if len(previews) < args.preview_limit:
            preview = args.out / "previews" / f"{target.stem}.jpg"
            draw_preview(row, preview)
            previews.append(preview)
        if args.limit > 0 and len(rows) >= args.limit:
            break

    write_jsonl(args.out / "candidate_boxes.jsonl", rows)
    annotations = args.out / "annotations"
    images = args.out / "images"
    review_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        src = Path(row["image"])
        target_name = f"{index:04d}_{src.name}"
        images.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, images / target_name)
        review_rows.append({**row, "image": f"images/{target_name}", "source_candidate": str(src)})
    write_jsonl(annotations / "draft_boxes.jsonl", review_rows)
    contact_sheet(previews, args.out / "homography_transfer_preview.jpg", args.preview_limit)
    summary = {
        "teacher_package": str(args.teacher_package),
        "image_dir": str(args.image_dir),
        "out": str(args.out),
        "settings": {
            "max_side": args.max_side,
            "nfeatures": args.nfeatures,
            "ratio": args.ratio,
            "min_matches": args.min_matches,
            "min_inliers": args.min_inliers,
            "min_inlier_ratio": args.min_inlier_ratio,
        },
        "stats": dict(sorted(stats.items())),
        "outputs": {
            "candidate_boxes": "candidate_boxes.jsonl",
            "draft_boxes": "annotations/draft_boxes.jsonl",
            "preview": "homography_transfer_preview.jpg",
        },
        "notes": ["Homography transfers are review candidates only; do not train before visual approval."],
    }
    write_json(args.out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-package", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--nfeatures", type=int, default=6000)
    parser.add_argument("--ratio", type=float, default=0.76)
    parser.add_argument("--min-keypoints", type=int, default=80)
    parser.add_argument("--min-matches", type=int, default=45)
    parser.add_argument("--min-inliers", type=int, default=32)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.28)
    parser.add_argument("--promote-min-inliers", type=int, default=1200)
    parser.add_argument("--promote-min-inlier-ratio", type=float, default=0.90)
    parser.add_argument("--ransac-reproj-threshold", type=float, default=5.0)
    parser.add_argument("--preview-limit", type=int, default=48)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-seed-images", action="store_true")
    parser.add_argument("--include-name", action="append", default=[], help="Only scan images whose filename contains this token.")
    parser.add_argument("--exclude-name", action="append", default=[], help="Skip images whose filename contains this token.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    print(json.dumps(propagate(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
