"""Benchmark question-crop rectangle strategies on historical diagnostic data.

The script is intentionally local/offline: it replays manifest rectangles against
stored source images, estimates upload bytes, and flags crop shapes that are
likely too narrow or too small for reliable VLM extraction.
"""

from __future__ import annotations

import argparse
import difflib
import io
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageOps


Rect = dict[str, float]

QUESTION_CROP_MIN_WIDTH_PX = 420
QUESTION_CROP_MIN_HEIGHT_PX = 240
QUESTION_CROP_MIN_AREA_PX = 160_000
QUESTION_CROP_THIN_ASPECT_RATIO = 6.0
QUESTION_CROP_EXPAND_MIN_WIDTH_RATIO = 0.72
QUESTION_CROP_EXPAND_MIN_HEIGHT_RATIO = 0.28
QUESTION_CROP_EXPAND_MAX_WIDTH_RATIO = 0.94
QUESTION_CROP_EXPAND_MAX_HEIGHT_RATIO = 0.52


def load_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def clamp_rect(rect: dict[str, Any]) -> Rect:
    x = max(0.0, min(1.0, float(rect.get("x", 0))))
    y = max(0.0, min(1.0, float(rect.get("y", 0))))
    w = max(0.0, min(1.0 - x, float(rect.get("width", rect.get("w", 0)))))
    h = max(0.0, min(1.0 - y, float(rect.get("height", rect.get("h", 0)))))
    return {"x": x, "y": y, "width": w, "height": h}


def fit_interval(start: float, length: float) -> tuple[float, float]:
    length = max(0.0, min(1.0, length))
    start = max(0.0, min(1.0 - length, start))
    return start, min(1.0, start + length)


def centered_expand(rect: Rect, *, min_width: float, min_height: float, pad_x: float, pad_y: float) -> Rect:
    r = clamp_rect(rect)
    cx = r["x"] + r["width"] / 2
    cy = r["y"] + r["height"] / 2
    w = min(1.0, max(r["width"] + pad_x * 2, min_width))
    h = min(1.0, max(r["height"] + pad_y * 2, min_height))
    x0, x1 = fit_interval(cx - w / 2, w)
    y0, y1 = fit_interval(cy - h / 2, h)
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def old_client_padding(item: dict[str, Any], source_size: tuple[int, int]) -> Rect:
    r = clamp_rect(load_json(item.get("normalized_rect")) or load_json(item.get("crop_rect")))
    return centered_expand(
        r,
        min_width=r["width"],
        min_height=r["height"],
        pad_x=r["width"] * 0.03,
        pad_y=r["height"] * 0.12,
    )


def adaptive_client_rect(item: dict[str, Any], source_size: tuple[int, int]) -> Rect:
    r = clamp_rect(load_json(item.get("normalized_rect")) or load_json(item.get("crop_rect")))
    if r["width"] <= 0 or r["height"] <= 0:
        return r

    source_width = max(source_size[0], 1)
    source_height = max(source_size[1], 1)
    source_min_width = min(0.55, max(0.38, 420 / source_width))
    source_min_height = min(0.24, max(0.14, 220 / source_height))

    aspect = r["width"] / max(r["height"], 0.001)
    area = r["width"] * r["height"]
    is_narrow = r["width"] < 0.32
    is_short = r["height"] < 0.11
    is_thin = aspect > 4.8

    target_width = r["width"] + max(0.014, r["width"] * 0.05) * 2
    target_height = r["height"] + max(0.014, r["height"] * 0.24) * 2
    if is_narrow:
        target_width = max(target_width, 0.58 if (r["x"] + r["width"] / 2) < 0.42 or (r["x"] + r["width"] / 2) > 0.58 else 0.72)
    if is_short:
        target_height = max(target_height, source_min_height)
    if is_thin:
        target_width = max(target_width, min(0.72, max(r["width"] * 1.65, source_min_width)))
        target_height = max(target_height, min(0.30, max(r["height"] * 3.2, source_min_height)))
    if area < 0.06:
        target_height = max(target_height, min(0.30, 0.070 / max(target_width, 0.01)))
    if is_thin or is_short:
        source_aspect = source_width / max(source_height, 1)
        target_height = max(target_height, min(0.30, target_width * source_aspect / 4.6))

    max_width = 1.0 if r["width"] >= 0.94 else 0.94
    max_height = 1.0 if r["height"] >= 0.58 else 0.58
    target_width = min(max(target_width, r["width"]), max_width)
    target_height = min(max(target_height, r["height"]), max_height)

    extra_width = max(0, target_width - r["width"])
    extra_height = max(0, target_height - r["height"])
    x0, x1 = fit_interval(r["x"] - extra_width * 0.5, target_width)
    top_bias = 0.22 if r["y"] < 0.08 else 0.35
    y0, y1 = fit_interval(r["y"] - extra_height * top_bias, target_height)
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def question_crop_expansion_reasons(rect: Rect, source_size: tuple[int, int]) -> list[str]:
    source_width, source_height = source_size
    crop_width = max(1, int(rect["width"] * source_width))
    crop_height = max(1, int(rect["height"] * source_height))
    min_width = min(max(QUESTION_CROP_MIN_WIDTH_PX, int(source_width * 0.32)), max(1, int(source_width * 0.55)))
    min_height = min(QUESTION_CROP_MIN_HEIGHT_PX, max(1, int(source_height * 0.22)))
    min_area = min(QUESTION_CROP_MIN_AREA_PX, max(1, int(source_width * source_height * 0.10)))
    reasons: list[str] = []
    if crop_width < min_width:
        reasons.append("too_narrow")
    if crop_height < min_height:
        reasons.append("too_short")
    if crop_width * crop_height < min_area:
        reasons.append("area_too_small")
    aspect_ratio = max(crop_width / max(1, crop_height), crop_height / max(1, crop_width))
    if aspect_ratio >= QUESTION_CROP_THIN_ASPECT_RATIO:
        reasons.append("thin_strip")
    return reasons


def server_expand_from_rect(r: Rect, source_size: tuple[int, int]) -> Rect:
    reasons = set(question_crop_expansion_reasons(r, source_size))
    if not reasons:
        return r
    target_width = r["width"]
    target_height = r["height"]
    if reasons & {"too_narrow", "area_too_small", "thin_strip", "unreadable_client_crop"}:
        target_width = max(target_width, r["width"] * 1.8)
    if "too_narrow" in reasons:
        target_width = max(target_width, QUESTION_CROP_EXPAND_MIN_WIDTH_RATIO)
    if "thin_strip" in reasons and r["width"] < 0.55:
        target_width = max(target_width, QUESTION_CROP_EXPAND_MIN_WIDTH_RATIO)
    if reasons & {"too_short", "area_too_small", "thin_strip", "unreadable_client_crop"}:
        target_height = max(target_height, r["height"] * 3.2, QUESTION_CROP_EXPAND_MIN_HEIGHT_RATIO)
    if "too_short" in reasons:
        target_height = max(target_height, QUESTION_CROP_EXPAND_MIN_HEIGHT_RATIO)
    target_width = min(target_width, QUESTION_CROP_EXPAND_MAX_WIDTH_RATIO)
    target_height = min(target_height, QUESTION_CROP_EXPAND_MAX_HEIGHT_RATIO)
    x0, x1 = fit_interval(r["x"] - (target_width - r["width"]) * 0.5, target_width)
    y0, y1 = fit_interval(r["y"] - (target_height - r["height"]) * 0.35, target_height)
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def server_expand_rect(item: dict[str, Any], source_size: tuple[int, int]) -> Rect:
    return server_expand_from_rect(old_client_padding(item, source_size), source_size)


def rect_only_v3_server_crop(item: dict[str, Any], source_size: tuple[int, int]) -> Rect:
    return server_expand_from_rect(adaptive_client_rect(item, source_size), source_size)


STRATEGIES: dict[str, Callable[[dict[str, Any], tuple[int, int]], Rect]] = {
    "historical_crop": lambda item, _: clamp_rect(load_json(item.get("crop_rect"))),
    "old_client_padding": old_client_padding,
    "adaptive_client_v2": adaptive_client_rect,
    "server_safety_expand": server_expand_rect,
    "rect_only_v3_server_crop": rect_only_v3_server_crop,
}


def crop_image(image: Image.Image, rect: Rect) -> Image.Image:
    w, h = image.size
    x0 = max(0, min(w - 1, int(round(rect["x"] * w))))
    y0 = max(0, min(h - 1, int(round(rect["y"] * h))))
    x1 = max(x0 + 1, min(w, int(round((rect["x"] + rect["width"]) * w))))
    y1 = max(y0 + 1, min(h, int(round((rect["y"] + rect["height"]) * h))))
    return image.crop((x0, y0, x1, y1))


def jpeg_upload_bytes(image: Image.Image, *, max_side: int, quality: int) -> int:
    img = image.convert("RGB")
    ratio = min(1.0, max_side / max(img.size))
    if ratio < 1.0:
        img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=quality, optimize=True)
    return len(out.getvalue())


def is_bad(width: int, height: int, rect: Rect) -> bool:
    ratio = width / height if height else 999
    area = rect["width"] * rect["height"]
    return height < 140 or ratio > 4.8 or rect["height"] < 0.09 or rect["width"] < 0.38 or area < 0.025


def normalize_key(text: str) -> str:
    return re.sub(r"\W+", "", (text or "").lower())


def ngrams(text: str, n: int = 3) -> set[str]:
    normalized = normalize_key(text)
    return {normalized[index : index + n] for index in range(max(0, len(normalized) - n + 1))}


def near_duplicate_question_key(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 8 and shorter in longer:
        return True
    left = ngrams(a)
    right = ngrams(b)
    jaccard = len(left & right) / max(1, len(left | right))
    sequence = difflib.SequenceMatcher(None, normalize_key(a), normalize_key(b)).ratio()
    return jaccard >= 0.35 or sequence >= 0.62


def rect_overlap_ratio(lhs: dict[str, Any], rhs: dict[str, Any]) -> float:
    lx0, ly0 = float(lhs.get("x", 0)), float(lhs.get("y", 0))
    lx1, ly1 = lx0 + float(lhs.get("width", 0)), ly0 + float(lhs.get("height", 0))
    rx0, ry0 = float(rhs.get("x", 0)), float(rhs.get("y", 0))
    rx1, ry1 = rx0 + float(rhs.get("width", 0)), ry0 + float(rhs.get("height", 0))
    overlap_w = max(0.0, min(lx1, rx1) - max(lx0, rx0))
    overlap_h = max(0.0, min(ly1, ry1) - max(ly0, ry0))
    overlap = overlap_w * overlap_h
    lhs_area = max(0.0001, (lx1 - lx0) * (ly1 - ly0))
    rhs_area = max(0.0001, (rx1 - rx0) * (ry1 - ry0))
    return overlap / min(lhs_area, rhs_area)


def row_signature_similar(a: dict[str, Any], b: dict[str, Any]) -> bool:
    key_a = str(a.get("question_key") or "")
    key_b = str(b.get("question_key") or "")
    if key_a == key_b:
        return True
    shorter, longer = (key_a, key_b) if len(key_a) <= len(key_b) else (key_b, key_a)
    if len(shorter) >= 8 and shorter in longer:
        return True
    if not near_duplicate_question_key(key_a, key_b):
        return False
    rect_a = a.get("rect") or {}
    rect_b = b.get("rect") or {}
    ax = float(rect_a.get("x", 0)) + float(rect_a.get("width", 0)) / 2
    ay = float(rect_a.get("y", 0)) + float(rect_a.get("height", 0)) / 2
    bx = float(rect_b.get("x", 0)) + float(rect_b.get("width", 0)) / 2
    by = float(rect_b.get("y", 0)) + float(rect_b.get("height", 0)) / 2
    center_distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    same_index = int(a.get("question_index") or 0) > 0 and int(a.get("question_index") or 0) == int(b.get("question_index") or 0)
    return rect_overlap_ratio(rect_a, rect_b) >= 0.34 or center_distance <= 0.18 or (same_index and center_distance <= 0.28)


def cluster_rows_by_question_key(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    clusters: list[list[dict[str, Any]]] = []
    for item in items:
        for cluster in clusters:
            if any(row_signature_similar(item, existing) for existing in cluster):
                cluster.append(item)
                break
        else:
            clusters.append([item])
    return clusters


def rect_manifest_bytes(items: list[dict[str, Any]]) -> int:
    payload = {
        "version": 2,
        "source": "ios-observation-rect",
        "crops": [
            {
                "source_sequence_index": item.get("sequence_index"),
                "question_index": item.get("question_index"),
                "question_key": item.get("question_key") or "",
                "preview_text": item.get("preview") or "",
                "crop_rect": item.get("rect") or {},
                "crop_area": item.get("area") or 0,
                "transfer_mode": "rect_only",
                "crop_prepared": False,
            }
            for item in items
        ],
    }
    return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def summarize(items: list[dict[str, Any]], full_frame_bytes_total: int) -> dict[str, Any]:
    heights = [x["height"] for x in items]
    ratios = [x["ratio"] for x in items]
    areas = [x["area"] for x in items]
    bytes_values = [x["upload_bytes"] for x in items]
    total_bytes = sum(bytes_values)
    clusters = cluster_rows_by_question_key(items)
    dedup_bytes = sum(max(member["upload_bytes"] for member in cluster) for cluster in clusters)
    dedup_items = [max(cluster, key=lambda member: member["upload_bytes"]) for cluster in clusters]
    manifest_bytes = rect_manifest_bytes(items)
    dedup_manifest_bytes = rect_manifest_bytes(dedup_items)
    return {
        "count": len(items),
        "bad": sum(1 for x in items if x["bad"]),
        "near_full_page": sum(1 for x in items if x["area"] >= 0.72),
        "near_dedupe_cluster_count": len(clusters),
        "near_dedupe_crop_reduction": len(items) - len(clusters),
        "height_median": statistics.median(heights) if heights else 0,
        "ratio_max": max(ratios) if ratios else 0,
        "area_median": round(statistics.median(areas), 4) if areas else 0,
        "area_p90": round(statistics.quantiles(areas, n=10)[8], 4) if len(areas) >= 10 else 0,
        "upload_bytes_total": total_bytes,
        "upload_kb_median": round((statistics.median(bytes_values) if bytes_values else 0) / 1024, 1),
        "vs_full_frame_bytes": round(total_bytes / max(1, full_frame_bytes_total), 3),
        "v2_total_upload_bytes_with_full_frames": full_frame_bytes_total + total_bytes,
        "v2_total_vs_full_frame_bytes": round((full_frame_bytes_total + total_bytes) / max(1, full_frame_bytes_total), 3),
        "near_dedupe_upload_bytes_total": dedup_bytes,
        "near_dedupe_vs_full_frame_bytes": round(dedup_bytes / max(1, full_frame_bytes_total), 3),
        "v2_near_dedupe_total_upload_bytes_with_full_frames": full_frame_bytes_total + dedup_bytes,
        "v2_near_dedupe_total_vs_full_frame_bytes": round((full_frame_bytes_total + dedup_bytes) / max(1, full_frame_bytes_total), 3),
        "v3_rect_manifest_bytes_total": manifest_bytes,
        "v3_rect_extra_vs_full_frame_bytes": round(manifest_bytes / max(1, full_frame_bytes_total), 4),
        "v3_near_dedupe_rect_manifest_bytes_total": dedup_manifest_bytes,
        "v3_near_dedupe_total_vs_full_frame_bytes": round((full_frame_bytes_total + dedup_manifest_bytes) / max(1, full_frame_bytes_total), 4),
        "rect_ms_total": round(sum(x["rect_ms"] for x in items), 2),
    }


def build_contact_sheet(out_dir: Path, rows_by_strategy: dict[str, list[dict[str, Any]]], crops: dict[tuple[str, int], Image.Image]) -> None:
    thumbs: list[Image.Image] = []
    for strategy, rows in rows_by_strategy.items():
        for item in rows[:20]:
            im = crops[(strategy, item["idx"])].copy()
            im.thumbnail((220, 150))
            tile = Image.new("RGB", (240, 190), "white")
            tile.paste(im.convert("RGB"), ((240 - im.width) // 2, 8))
            draw = ImageDraw.Draw(tile)
            draw.text((8, 158), f"{strategy[:18]}", fill=(0, 0, 0))
            draw.text((8, 174), f"#{item['idx']} {item['width']}x{item['height']} {item['upload_bytes']//1024}KB", fill=(0, 0, 0))
            thumbs.append(tile)
    cols = 4
    rows_count = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 240, max(1, rows_count) * 190), (245, 245, 245))
    for i, tile in enumerate(thumbs):
        sheet.paste(tile, ((i % cols) * 240, (i // cols) * 190))
    sheet.save(out_dir / "contact_sheet.jpg", quality=88)


def run(data_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "images"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["manifest"]
    source_cache: dict[str, Image.Image] = {}
    full_frame_bytes: dict[str, int] = {}
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATEGIES}
    crops: dict[tuple[str, int], Image.Image] = {}

    for idx, item in enumerate(rows, start=1):
        source_name = item.get("src_filename")
        if not source_name:
            continue
        if source_name not in source_cache:
            source_cache[source_name] = ImageOps.exif_transpose(Image.open(images_dir / source_name)).convert("RGB")
            full_frame_bytes[source_name] = jpeg_upload_bytes(source_cache[source_name], max_side=1600, quality=78)
        source = source_cache[source_name]
        source_size = source.size

        for strategy, strategy_fn in STRATEGIES.items():
            t0 = time.perf_counter()
            rect = strategy_fn(item, source_size)
            rect_ms = (time.perf_counter() - t0) * 1000
            crop = crop_image(source, rect)
            upload_bytes = jpeg_upload_bytes(crop, max_side=1200, quality=82)
            width, height = crop.size
            results[strategy].append(
                {
                    "idx": idx,
                    "sequence_index": item.get("sequence_index"),
                    "question_index": item.get("question_index"),
                    "width": width,
                    "height": height,
                    "ratio": round(width / height, 2) if height else 0,
                    "area": round(rect["width"] * rect["height"], 4),
                    "upload_bytes": upload_bytes,
                    "rect_ms": rect_ms,
                    "bad": is_bad(width, height, rect),
                    "rect": {key: round(value, 4) for key, value in rect.items()},
                    "question_key": item.get("question_key") or "",
                    "preview": (item.get("preview_text") or "")[:80],
                }
            )
            crops[(strategy, idx)] = crop

    full_frame_total = sum(full_frame_bytes.values())
    summary = {
        "fixture": str(data_dir),
        "source_frame_count": len(full_frame_bytes),
        "full_frame_upload_bytes_total": full_frame_total,
        "strategies": {name: summarize(items, full_frame_total) for name, items in results.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "details.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    build_contact_sheet(out_dir, results, crops)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark question crop strategies.")
    parser.add_argument("--data", type=Path, default=Path("diagnostics/crop-f472/data"))
    parser.add_argument("--out", type=Path, default=Path("diagnostics/crop-f472/benchmark-v2"))
    args = parser.parse_args()
    summary = run(args.data, args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
