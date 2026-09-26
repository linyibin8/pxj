import json
import statistics
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
IMAGES = DATA / "images"
OUT = ROOT / "out"


def load_json(value):
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def clamp_rect(rect):
    x = max(0.0, min(1.0, float(rect.get("x", 0))))
    y = max(0.0, min(1.0, float(rect.get("y", 0))))
    w = max(0.0, min(1.0 - x, float(rect.get("width", rect.get("w", 0)))))
    h = max(0.0, min(1.0 - y, float(rect.get("height", rect.get("h", 0)))))
    return {"x": x, "y": y, "width": w, "height": h}


def centered_expand(rect, *, min_width, min_height, pad_x, pad_y):
    r = clamp_rect(rect)
    cx = r["x"] + r["width"] / 2
    cy = r["y"] + r["height"] / 2
    w = max(r["width"] + pad_x * 2, min_width)
    h = max(r["height"] + pad_y * 2, min_height)
    x = max(0.0, min(1.0 - w, cx - w / 2))
    y = max(0.0, min(1.0 - h, cy - h / 2))
    return clamp_rect({"x": x, "y": y, "width": min(1.0, w), "height": min(1.0, h)})


def scheme_minbox(rect):
    r = clamp_rect(rect)
    return centered_expand(
        r,
        min_width=0.52 if r["width"] < 0.45 else r["width"],
        min_height=0.14 if r["height"] < 0.11 else r["height"],
        pad_x=max(0.02, r["width"] * 0.08),
        pad_y=max(0.025, r["height"] * 0.35),
    )


def scheme_column(rect):
    r = clamp_rect(rect)
    cx = r["x"] + r["width"] / 2
    if cx < 0.42:
        x, w = 0.02, 0.58
    elif cx > 0.58:
        x, w = 0.40, 0.58
    else:
        x, w = 0.06, 0.88
    y_pad_top = max(0.025, r["height"] * 0.35)
    y_pad_bottom = max(0.055, r["height"] * 0.85)
    y = max(0.0, r["y"] - y_pad_top)
    bottom = min(1.0, r["y"] + r["height"] + y_pad_bottom)
    h = max(bottom - y, 0.16)
    if y + h > 1.0:
        y = max(0.0, 1.0 - h)
    return clamp_rect({"x": x, "y": y, "width": w, "height": min(1.0, h)})


def scheme_pageband(rect):
    r = clamp_rect(rect)
    return centered_expand(
        {"x": 0.04, "y": r["y"], "width": 0.92, "height": r["height"]},
        min_width=0.92,
        min_height=max(0.18, r["height"] * 2.2),
        pad_x=0,
        pad_y=max(0.035, r["height"] * 0.45),
    )


SCHEMES = {
    "baseline": lambda r: clamp_rect(r),
    "minbox": scheme_minbox,
    "column": scheme_column,
    "pageband": scheme_pageband,
}


def crop_image(image, rect):
    w, h = image.size
    x0 = int(round(rect["x"] * w))
    y0 = int(round(rect["y"] * h))
    x1 = int(round((rect["x"] + rect["width"]) * w))
    y1 = int(round((rect["y"] + rect["height"]) * h))
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    return image.crop((x0, y0, x1, y1))


def is_bad(width, height, rect):
    ratio = width / height if height else 999
    area = rect["width"] * rect["height"]
    return height < 140 or ratio > 4.8 or rect["height"] < 0.09 or rect["width"] < 0.38 or area < 0.025


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["manifest"]
    source_cache = {}
    results = {name: [] for name in SCHEMES}
    timings = {name: 0.0 for name in SCHEMES}

    for idx, item in enumerate(rows, start=1):
        source_name = item.get("src_filename")
        if not source_name:
            continue
        if source_name not in source_cache:
            source_cache[source_name] = ImageOps.exif_transpose(Image.open(IMAGES / source_name)).convert("RGB")
        source = source_cache[source_name]
        raw_rect = load_json(item.get("crop_rect"))
        for name, fn in SCHEMES.items():
            t0 = time.perf_counter()
            rect = fn(raw_rect)
            crop = crop_image(source, rect)
            elapsed = time.perf_counter() - t0
            timings[name] += elapsed
            if name != "baseline":
                crop.save(OUT / f"{idx:02d}-{name}.jpg", quality=86, optimize=True)
            width, height = crop.size
            results[name].append(
                {
                    "idx": idx,
                    "seq": item.get("sequence_index"),
                    "qidx": item.get("question_index"),
                    "width": width,
                    "height": height,
                    "ratio": round(width / height, 2) if height else 0,
                    "area": round(rect["width"] * rect["height"], 4),
                    "rect": {k: round(v, 3) for k, v in rect.items()},
                    "bad": is_bad(width, height, rect),
                    "preview": (item.get("preview_text") or "")[:60],
                }
            )

    summary = {}
    for name, items in results.items():
        heights = [x["height"] for x in items]
        ratios = [x["ratio"] for x in items]
        areas = [x["area"] for x in items]
        summary[name] = {
            "count": len(items),
            "bad": sum(1 for x in items if x["bad"]),
            "height_min": min(heights) if heights else 0,
            "height_median": statistics.median(heights) if heights else 0,
            "height_max": max(heights) if heights else 0,
            "ratio_median": statistics.median(ratios) if ratios else 0,
            "ratio_max": max(ratios) if ratios else 0,
            "area_median": statistics.median(areas) if areas else 0,
            "crop_ms_total": round(timings[name] * 1000, 2),
            "crop_ms_each": round((timings[name] / max(1, len(items))) * 1000, 3),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "details.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    thumbs = []
    for name in ("baseline", "minbox", "column", "pageband"):
        for item in results[name][:20]:
            crop_file = OUT / f"{item['idx']:02d}-{name}.jpg"
            if name == "baseline":
                source_crop = rows[item["idx"] - 1].get("crop_filename")
                crop_file = IMAGES / source_crop
            if crop_file.exists():
                im = ImageOps.exif_transpose(Image.open(crop_file)).convert("RGB")
                im.thumbnail((220, 150))
                tile = Image.new("RGB", (240, 190), "white")
                tile.paste(im, ((240 - im.width) // 2, 8))
                draw = ImageDraw.Draw(tile)
                draw.text((8, 162), f"{name} #{item['idx']} {item['width']}x{item['height']}", fill=(0, 0, 0))
                thumbs.append(tile)
    cols = 4
    rows_count = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 240, rows_count * 190), (245, 245, 245))
    for i, tile in enumerate(thumbs):
        sheet.paste(tile, ((i % cols) * 240, (i // cols) * 190))
    sheet.save(OUT / "contact_sheet.jpg", quality=88)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("out", OUT)


if __name__ == "__main__":
    main()
