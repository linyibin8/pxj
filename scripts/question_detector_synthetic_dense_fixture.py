"""Generate synthetic dense question-page fixtures for observation crop stress tests.

The output is dataset-shaped so existing replay/eval tools can consume it, but
it is diagnostic evidence only. These pages are useful for cap/fallback/network
stress and for catching heuristic layout failures; they must not be treated as
reviewed production training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


CATEGORY_ID = 1
CATEGORY_NAME = "question_block"


@dataclass(frozen=True)
class Profile:
    name: str
    width: int
    height: int
    columns: int
    question_count: int
    lines_min: int
    lines_max: int
    weak_keys: bool = False
    table_every: int = 0
    figure_every: int = 0
    tight: bool = False


@dataclass
class QuestionBox:
    index: int
    bbox_px: dict[str, int]
    bbox_norm: dict[str, float]
    question_key: str
    confidence: float
    quality_flags: list[str]


PROFILES = [
    Profile("phone_dense_18", 1170, 2532, 1, 18, 2, 3, table_every=6, figure_every=5),
    Profile("phone_cap_overflow_24", 1170, 2532, 1, 24, 1, 2, tight=True),
    Profile("ipad_two_column_30", 2048, 2732, 2, 30, 2, 4, table_every=5, figure_every=7),
    Profile("ipad_landscape_three_column_36", 2732, 2048, 3, 36, 1, 3, table_every=6, figure_every=8, tight=True),
    Profile("ipad_weak_textless_20", 2048, 2732, 2, 20, 1, 2, weak_keys=True, figure_every=4),
]


def stable_id(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def font_path() -> str | None:
    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    if bold:
        bold_candidates = [
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/calibrib.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ]
        for candidate in bold_candidates:
            if candidate.is_file():
                return ImageFont.truetype(str(candidate), size=size)
    path = font_path()
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def norm_box(box: dict[str, int], width: int, height: int) -> dict[str, float]:
    return {
        "x": round(box["x"] / width, 6),
        "y": round(box["y"] / height, 6),
        "width": round(box["width"] / width, 6),
        "height": round(box["height"] / height, 6),
    }


def rect_key(box: dict[str, float]) -> str:
    return ",".join(str(int(round(float(box[key]) * 1000))) for key in ("x", "y", "width", "height"))


def draw_text_line(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> int:
    draw.text(xy, text, font=font, fill=fill)
    bbox = draw.textbbox(xy, text, font=font)
    return max(1, bbox[3] - bbox[1])


def draw_table(draw: ImageDraw.ImageDraw, box: dict[str, int], rng: random.Random) -> None:
    x = box["x"] + int(box["width"] * 0.08)
    y = box["y"] + int(box["height"] * 0.56)
    w = int(box["width"] * rng.uniform(0.36, 0.62))
    h = max(22, int(box["height"] * 0.23))
    rows = 3
    cols = 3
    color = (80, 86, 96)
    draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
    for row in range(1, rows):
        yy = y + int(h * row / rows)
        draw.line([x, yy, x + w, yy], fill=color, width=1)
    for col in range(1, cols):
        xx = x + int(w * col / cols)
        draw.line([xx, y, xx, y + h], fill=color, width=1)


def draw_figure(draw: ImageDraw.ImageDraw, box: dict[str, int], rng: random.Random) -> None:
    w = int(box["width"] * rng.uniform(0.20, 0.33))
    h = max(24, int(box["height"] * rng.uniform(0.24, 0.36)))
    x = box["x"] + box["width"] - w - int(box["width"] * 0.08)
    y = box["y"] + int(box["height"] * 0.48)
    color = (92, 94, 104)
    draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
    draw.line([x + 8, y + h - 8, x + int(w * 0.45), y + 8, x + w - 8, y + h - 10], fill=color, width=2)
    if rng.random() < 0.55:
        draw.ellipse([x + int(w * 0.58), y + 8, x + w - 10, y + int(h * 0.45)], outline=color, width=2)


def question_lines(index: int, line_count: int, profile: Profile, rng: random.Random) -> list[str]:
    verbs = ["solve", "compare", "explain", "choose", "estimate", "prove", "fill"]
    topics = ["fraction", "graph", "angle", "sequence", "ratio", "function", "table", "word problem"]
    lines = [f"{index}. {rng.choice(verbs).title()} the {rng.choice(topics)}. Show the key step."]
    snippets = [
        "A. 12    B. 18    C. 24    D. 30",
        "Given x + y = 7 and 2x - y = 5.",
        "Use the figure/table if it is provided.",
        "Write one clear answer in the blank.",
        "Check whether the statement is always true.",
        "Find the missing value and justify it.",
    ]
    if profile.tight:
        snippets = [
            "A  B  C  D",
            "x + y = 7",
            "see table",
            "fill blank",
            "give reason",
        ]
    while len(lines) < line_count:
        lines.append(rng.choice(snippets))
    return lines


def page_rect_for(profile: Profile, rng: random.Random) -> tuple[int, int, int, int]:
    margin_x = int(profile.width * rng.uniform(0.040, 0.065))
    margin_y = int(profile.height * rng.uniform(0.030, 0.050))
    return (
        margin_x,
        margin_y,
        profile.width - margin_x * 2,
        profile.height - margin_y * 2,
    )


def render_profile_page(profile: Profile, variant: int, rng: random.Random) -> tuple[Image.Image, list[QuestionBox], dict[str, Any]]:
    image = Image.new("RGB", (profile.width, profile.height), (221, 224, 229))
    draw = ImageDraw.Draw(image)
    page_x, page_y, page_w, page_h = page_rect_for(profile, rng)
    draw.rounded_rectangle([page_x, page_y, page_x + page_w, page_y + page_h], radius=10, fill=(252, 252, 249), outline=(203, 207, 214), width=2)

    title_font = load_font(max(24, int(profile.width * 0.017)), bold=True)
    body_font = load_font(max(18, int(profile.width * (0.012 if profile.columns > 1 else 0.017))))
    small_font = load_font(max(15, int(profile.width * 0.010)))
    title_h = draw_text_line(
        draw,
        (page_x + int(page_w * 0.04), page_y + int(page_h * 0.022)),
        f"Synthetic Diagnostic Sheet - {profile.name}",
        title_font,
        (31, 35, 43),
    )
    header_bottom = page_y + int(page_h * 0.040) + title_h
    draw.line([page_x + int(page_w * 0.035), header_bottom + 12, page_x + page_w - int(page_w * 0.035), header_bottom + 12], fill=(195, 198, 204), width=2)

    content_top = header_bottom + int(page_h * 0.035)
    content_bottom = page_y + page_h - int(page_h * 0.035)
    gutter = int(page_w * (0.030 if profile.columns > 1 else 0.0))
    col_w = int((page_w - int(page_w * 0.07) - gutter * (profile.columns - 1)) / profile.columns)
    col_x0 = page_x + int(page_w * 0.035)
    per_col = (profile.question_count + profile.columns - 1) // profile.columns
    usable_h = content_bottom - content_top
    gap = max(4, int(usable_h * (0.008 if profile.tight else 0.012)))
    cell_h = max(24, int((usable_h - gap * (per_col - 1)) / per_col))

    boxes: list[QuestionBox] = []
    for q in range(profile.question_count):
        col = min(profile.columns - 1, q // per_col)
        row = q % per_col
        x = col_x0 + col * (col_w + gutter)
        y = content_top + row * (cell_h + gap)
        jitter_x = rng.randint(-3, 3)
        jitter_y = rng.randint(-3, 3)
        h = clamp(cell_h + rng.randint(-4, 5), 22, max(24, content_bottom - y))
        box = {
            "x": clamp(x + jitter_x, page_x, page_x + page_w - 10),
            "y": clamp(y + jitter_y, page_y, page_y + page_h - 10),
            "width": clamp(col_w + rng.randint(-5, 6), 30, page_x + page_w - x),
            "height": h,
        }
        draw.rounded_rectangle(
            [box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]],
            radius=5,
            fill=(255, 255, 253),
            outline=(225, 226, 230),
            width=1,
        )
        line_count = rng.randint(profile.lines_min, profile.lines_max)
        lines = question_lines(q + 1, line_count, profile, rng)
        text_x = box["x"] + max(8, int(box["width"] * 0.025))
        text_y = box["y"] + max(5, int(box["height"] * 0.070))
        line_step = max(18, int((body_font.size if hasattr(body_font, "size") else 18) * 1.18))
        for line_index, line in enumerate(lines):
            font = body_font if line_index == 0 else small_font
            draw_text_line(draw, (text_x, text_y + line_index * line_step), line, font, (28, 31, 37))
        if profile.table_every and (q + 1) % profile.table_every == 0:
            draw_table(draw, box, rng)
        if profile.figure_every and (q + 1) % profile.figure_every == 0:
            draw_figure(draw, box, rng)

        bbox_norm = norm_box(box, profile.width, profile.height)
        question_key = (
            f"layout:{q + 1}:{rect_key(bbox_norm)}:notext"
            if profile.weak_keys
            else f"synthetic:{profile.name}:{variant}:{q + 1:02d}"
        )
        flags = ["synthetic_gt", "diagnostic_only", f"profile:{profile.name}"]
        if profile.tight:
            flags.append("dense_tight_spacing")
        if profile.weak_keys:
            flags.append("weak_layout_key")
        if profile.table_every and (q + 1) % profile.table_every == 0:
            flags.append("contains_table")
        if profile.figure_every and (q + 1) % profile.figure_every == 0:
            flags.append("contains_figure")
        boxes.append(
            QuestionBox(
                index=q + 1,
                bbox_px=box,
                bbox_norm=bbox_norm,
                question_key=question_key,
                confidence=0.38 if profile.weak_keys else round(rng.uniform(0.78, 0.96), 4),
                quality_flags=flags,
            )
        )
    metadata = {
        "profile": profile.name,
        "variant": variant,
        "page_rect": {"x": page_x, "y": page_y, "width": page_w, "height": page_h},
        "columns": profile.columns,
        "question_count": profile.question_count,
        "diagnostic_only": True,
    }
    return image, boxes, metadata


def save_yolo_label(path: Path, boxes: list[QuestionBox]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box in boxes:
        rect = box.bbox_norm
        cx = rect["x"] + rect["width"] / 2
        cy = rect["y"] + rect["height"] / 2
        lines.append(f"0 {cx:.6f} {cy:.6f} {rect['width']:.6f} {rect['height']:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_contact_sheet(out: Path, rows: list[dict[str, Any]], limit: int = 24) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("image") or ""), []).append(row)
    tiles: list[Image.Image] = []
    colors = [(38, 164, 94), (220, 118, 40), (54, 112, 214), (185, 68, 160)]
    for image_rel, anns in list(grouped.items())[:limit]:
        path = out / image_rel
        if not path.is_file():
            continue
        with Image.open(path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        scale = min(310 / image.width, 220 / image.height)
        thumb = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(thumb)
        for index, ann in enumerate(anns):
            box = ann["bbox_px"]
            color = colors[index % len(colors)]
            x = box["x"] * scale
            y = box["y"] * scale
            w = box["width"] * scale
            h = box["height"] * scale
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            draw.text((x + 2, y + 1), str(index + 1), fill=color)
        tile = Image.new("RGB", (330, 260), "white")
        tile.paste(thumb, ((330 - thumb.width) // 2, 8))
        ImageDraw.Draw(tile).text((8, 235), f"{Path(image_rel).name[:28]} boxes={len(anns)}", fill=(0, 0, 0))
        tiles.append(tile)
    if not tiles:
        return
    cols = 3
    rows_count = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 330, rows_count * 260), (244, 245, 247))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % cols) * 330, (index // cols) * 260))
    sheet.save(out / "annotations" / "synthetic_dense_contact_sheet.jpg", quality=90)


def write_coco(out: Path, image_rows: list[dict[str, Any]], ann_rows: list[dict[str, Any]]) -> None:
    image_id_by_rel = {str(row["image"]): index + 1 for index, row in enumerate(image_rows)}
    images = []
    annotations = []
    ann_id = 1
    for row in image_rows:
        images.append(
            {
                "id": image_id_by_rel[str(row["image"])],
                "file_name": row["image"],
                "width": row["width"],
                "height": row["height"],
                "source_filename": row["source_filename"],
                "synthetic": True,
            }
        )
    for row in ann_rows:
        box = row["bbox_px"]
        annotations.append(
            {
                "id": ann_id,
                "image_id": image_id_by_rel[str(row["image"])],
                "category_id": CATEGORY_ID,
                "bbox": [box["x"], box["y"], box["width"], box["height"]],
                "area": box["width"] * box["height"],
                "iscrowd": 0,
                "question_key": row["question_key"],
                "source_filename": row["source_filename"],
            }
        )
        ann_id += 1
    write_json(
        out / "annotations" / "coco_all.json",
        {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME}],
            "info": {"description": "Synthetic dense diagnostic fixture; not production training data."},
        },
    )


def generate(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "images" / "diagnostic").mkdir(parents=True, exist_ok=True)
    (args.out / "labels" / "diagnostic").mkdir(parents=True, exist_ok=True)
    (args.out / "annotations").mkdir(parents=True, exist_ok=True)

    selected_profiles = [profile for profile in PROFILES if not args.profile or profile.name in set(args.profile)]
    if not selected_profiles:
        raise ValueError("No profiles selected.")

    image_rows: list[dict[str, Any]] = []
    ann_rows: list[dict[str, Any]] = []
    draft_rows: list[dict[str, Any]] = []
    profile_counts: dict[str, dict[str, int]] = {}

    for profile in selected_profiles:
        profile_counts[profile.name] = {"images": 0, "annotations": 0}
        for variant in range(1, args.variants_per_profile + 1):
            seed_key = f"{args.seed}:{profile.name}:{variant}"
            local_rng = random.Random(seed_key)
            image, boxes, metadata = render_profile_page(profile, variant, local_rng)
            stem = f"{profile.name}_{variant:02d}_{stable_id(seed_key, 8)}"
            image_rel = f"images/diagnostic/{stem}.jpg"
            image_path = args.out / image_rel
            image.save(image_path, format="JPEG", quality=args.jpeg_quality, optimize=True)
            label_rel = f"labels/diagnostic/{stem}.txt"
            save_yolo_label(args.out / label_rel, boxes)
            image_row = {
                "image": image_rel,
                "width": profile.width,
                "height": profile.height,
                "split": "diagnostic",
                "split_key": f"synthetic:{profile.name}",
                "source_key": f"synthetic_dense:{profile.name}:{variant}",
                "source_filename": Path(image_rel).name,
                "origin": "synthetic_dense_fixture",
                "diagnostic_only": True,
                "synthetic_profile": profile.name,
                "metadata": metadata,
            }
            image_rows.append(image_row)
            draft_rows.append(
                {
                    "image": image_rel,
                    "source_candidate": image_row["source_key"],
                    "candidate": {
                        "session_id": f"synthetic:{profile.name}",
                        "candidate_kind": "synthetic_dense",
                    },
                    "width": profile.width,
                    "height": profile.height,
                    "boxes": [
                        {
                            "bbox_px": box.bbox_px,
                            "score": box.confidence,
                            "question_key": box.question_key,
                            "quality_flags": box.quality_flags,
                            "annotation_status": "synthetic_candidate",
                        }
                        for box in boxes
                    ],
                    "metadata": metadata,
                }
            )
            profile_counts[profile.name]["images"] += 1
            profile_counts[profile.name]["annotations"] += len(boxes)
            for box in boxes:
                ann_rows.append(
                    {
                        "image": image_rel,
                        "width": profile.width,
                        "height": profile.height,
                        "image_width": profile.width,
                        "image_height": profile.height,
                        "split": "diagnostic",
                        "split_key": f"synthetic:{profile.name}",
                        "source_key": image_row["source_key"],
                        "source_filename": image_row["source_filename"],
                        "source_image_id": stem,
                        "origin": "synthetic_dense_fixture",
                        "category": CATEGORY_NAME,
                        "category_id": CATEGORY_ID,
                        "bbox_px": box.bbox_px,
                        "bbox_norm": box.bbox_norm,
                        "question_key": box.question_key,
                        "question_key_strength": "synthetic_diagnostic",
                        "confidence": box.confidence,
                        "quality_flags": box.quality_flags,
                        "diagnostic_only": True,
                        "synthetic_profile": profile.name,
                        "question_index": box.index,
                    }
                )

    write_jsonl(args.out / "annotations" / "image_manifest.jsonl", image_rows)
    write_jsonl(args.out / "annotations" / "manifest.jsonl", ann_rows)
    write_jsonl(args.out / "annotations" / "draft_boxes.jsonl", draft_rows)
    write_coco(args.out, image_rows, ann_rows)
    build_contact_sheet(args.out, ann_rows)
    write_json(
        args.out / "yolo_dataset.yaml",
        {
            "path": str(args.out),
            "train": "images/diagnostic",
            "val": "images/diagnostic",
            "test": "images/diagnostic",
            "names": {0: CATEGORY_NAME},
            "diagnostic_only": True,
        },
    )
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "readiness": {
            "pilot_eval_ready": False,
            "model_training_ready": False,
            "has_error": False,
        },
        "counts": {
            "source_images": len(image_rows),
            "positive_images": len(image_rows),
            "negative_images": 0,
            "annotations": len(ann_rows),
            "split_groups": len(selected_profiles),
        },
        "warnings": [
            "synthetic_dense_fixture is diagnostic-only and must not be used to satisfy reviewed-data or release training readiness.",
            "Use this fixture for cap/fallback stress, VLM pixel-cost estimates, and heuristic/Core ML regression tests.",
        ],
        "profile_counts": profile_counts,
    }
    write_json(args.out / "audit.json", audit)
    summary = {
        "generated_at": audit["generated_at"],
        "out": str(args.out),
        "seed": args.seed,
        "variants_per_profile": args.variants_per_profile,
        "image_count": len(image_rows),
        "annotation_count": len(ann_rows),
        "profile_counts": profile_counts,
        "diagnostic_only": True,
        "outputs": {
            "image_manifest": "annotations/image_manifest.jsonl",
            "manifest": "annotations/manifest.jsonl",
            "draft_boxes": "annotations/draft_boxes.jsonl",
            "coco": "annotations/coco_all.json",
            "contact_sheet": "annotations/synthetic_dense_contact_sheet.jpg",
            "audit": "audit.json",
        },
        "recommended_commands": {
            "oracle_replay": f"python scripts\\question_observation_replay.py --prelabel-root {args.out} --out diagnostics\\question-observation-replay-synthetic-dense-oracle --clean",
            "layout_replay": f"python scripts\\question_observation_replay.py --image-dir {args.out / 'images'} --measure-layout --use-measured-layout-boxes --out diagnostics\\question-observation-replay-synthetic-dense-layout --clean",
            "cap_sweep": f"python scripts\\question_observation_candidate_cap_sweep.py --replay diagnostics\\question-observation-replay-synthetic-dense-oracle --ground-truth-manifest {args.out / 'annotations' / 'manifest.jsonl'} --out diagnostics\\question-observation-candidate-cap-sweep-synthetic-dense --clean",
            "candidate_eval": f"python scripts\\question_observation_candidate_eval.py --replay diagnostics\\question-observation-replay-synthetic-dense-layout --ground-truth-manifest {args.out / 'annotations' / 'manifest.jsonl'} --out diagnostics\\question-observation-candidate-eval-synthetic-dense-layout --clean",
        },
    }
    write_json(args.out / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic dense question pages for observation crop diagnostics.")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-synthetic-dense"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants-per-profile", type=int, default=2)
    parser.add_argument("--profile", action="append", default=[], help="Profile name to include. Defaults to all profiles.")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    summary = generate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
