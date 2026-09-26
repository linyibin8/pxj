"""Build a portable Codex comparison pack for question segmentation.

Input is the shared benchmark manifest written by
question_detector_review_priority.py. The pack copies the selected images and
writes a prediction template, optional draft reference boxes, a prompt, and the
exact evaluation commands needed to compare Codex outputs with reviewed labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise SystemExit(f"{path}:{line_number} is not a JSON object")
            rows.append(item)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(value: str, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def safe_stem(path: Path, max_chars: int = 56) -> str:
    raw = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in path.stem)
    return (raw or "image")[:max_chars]


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        normalized = ImageOps.exif_transpose(image)
        return normalized.size


def priority_reference_rows(priority_path: Path, image_map: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    payload = read_json(priority_path)
    rows: list[dict[str, Any]] = []
    for image in payload.get("recommended_images") or []:
        if not isinstance(image, dict):
            continue
        review_id = str(image.get("review_id") or "")
        mapped = image_map.get(review_id)
        if not mapped:
            continue
        boxes: list[dict[str, Any]] = []
        for row in image.get("boxes") or []:
            box = row.get("box") if isinstance(row, dict) else None
            if not isinstance(box, dict):
                continue
            bbox = box.get("bbox_px") or box.get("bbox")
            if not bbox:
                continue
            boxes.append(
                {
                    "bbox_px": bbox,
                    "score": box.get("score", row.get("net_score") if isinstance(row, dict) else 1.0),
                    "source": box.get("source") or "draft_priority",
                    "quality_flags": box.get("quality_flags") or [],
                }
            )
        rows.append(
            {
                "review_id": review_id,
                "image": mapped["image"],
                "image_path": mapped["image_path"],
                "width": mapped["width"],
                "height": mapped["height"],
                "cohort": mapped.get("cohort"),
                "source_kind": mapped.get("source_kind"),
                "reference_status": "draft_pending_human_review",
                "boxes": boxes,
            }
        )
    return rows


def make_prompt() -> str:
    return """# Codex Question Segmentation Benchmark

For each image listed in `benchmark_manifest.jsonl`, segment the visible worksheet into individual printed-question regions.

Return `codex_predictions.jsonl` with one JSON object per image. Use the exact `review_id` from the manifest. Use either pixel boxes or normalized boxes:

```json
{"review_id":"...","image":"images/001_example.jpg","boxes":[{"bbox_px":{"x":0,"y":0,"width":100,"height":100},"score":0.95}]}
```

Rules:

- Detect question regions, not answer correctness, not solution steps, and not decorative containers.
- A multi-line stem with its answer blanks/options belongs to one question region.
- Split adjacent numbered questions even when they share a dense two-column or three-column layout.
- Prefer tight boxes around the printed question content; include diagrams/options that belong to that question.
- Do not output full-page fallback boxes unless the whole page is genuinely one question.
- Preserve image order, and write an empty `boxes` list only when no printed question is visible.
"""


def make_readme(args: argparse.Namespace, summary: dict[str, Any]) -> str:
    return f"""# Question Segmentation Codex Benchmark Pack

Generated: {summary['generated_at']}

This pack contains {summary['image_count']} images selected from the current PXJ question-detector review queue. It is meant to compare Codex-style visual segmentation with the same human-reviewed reference boxes used for detector training.

## Files

- `benchmark_manifest.jsonl`: image list, review IDs, cohort labels, and copied image paths.
- `images/`: copied benchmark images.
- `codex_prompt.md`: task prompt and output schema.
- `codex_predictions_template.jsonl`: empty rows to fill with Codex predictions.
- `draft_reference_boxes.jsonl`: draft detector boxes for sanity checks only; not ground truth.
- `eval_commands.md`: exact commands for draft self-check and reviewed-reference evaluation.

## Required Prediction Shape

Each `codex_predictions.jsonl` row should look like:

```json
{{"review_id":"...","image":"images/001_example.jpg","boxes":[{{"bbox_px":{{"x":0,"y":0,"width":100,"height":100}},"score":0.95}}]}}
```

After human review export exists, evaluate Codex predictions against it with `question_segmentation_benchmark_eval.py`. Until then, the draft self-check only verifies wiring and should not be treated as model quality.

Always inspect `summary.json.input_audit`: reference and prediction `missing_images` should be empty for a valid comparison. A high F1 over a partially matched prediction file is not a usable benchmark result.
"""


def make_eval_commands(args: argparse.Namespace) -> str:
    out = args.out
    benchmark = out / "benchmark_manifest.jsonl"
    draft_reference = out / "draft_reference_boxes.jsonl"
    prediction_template = out / "codex_predictions_template.jsonl"
    return f"""# Evaluation Commands

Draft wiring self-check:

```powershell
python scripts\\question_segmentation_benchmark_eval.py `
  --benchmark-manifest {benchmark} `
  --reference-jsonl {draft_reference} `
  --predictions-jsonl {draft_reference} `
  --label draft_pack_self_check `
  --out diagnostics\\question-segmentation-benchmark-pack-self-check `
  --clean
```

After Codex predictions are written to `codex_predictions.jsonl`:

```powershell
python scripts\\question_segmentation_benchmark_eval.py `
  --benchmark-manifest {benchmark} `
  --reference-jsonl PATH_TO_HUMAN_APPROVED_BOXES.jsonl `
  --predictions-jsonl {out / "codex_predictions.jsonl"} `
  --label codex_vs_reviewed `
  --out diagnostics\\question-segmentation-benchmark-codex-vs-reviewed `
  --clean
```

Template file to start from:

```text
{prediction_template}
```
"""


def build_pack(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.benchmark_manifest)
    output_manifest: list[dict[str, Any]] = []
    prediction_template: list[dict[str, Any]] = []
    image_map: dict[str, dict[str, Any]] = {}
    missing_images: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        review_id = str(row.get("review_id") or "").strip()
        source_path = Path(str(row.get("image_path") or ""))
        if not review_id:
            raise SystemExit(f"benchmark row {index} is missing review_id")
        if not source_path.is_file():
            missing_images.append({"order": index, "review_id": review_id, "image_path": str(source_path)})
            continue
        suffix = source_path.suffix.lower() if source_path.suffix else ".jpg"
        target_name = f"{index:03d}_{stable_id(review_id)}_{safe_stem(source_path)}{suffix}"
        target_rel = f"images/{target_name}"
        target = args.out / target_rel
        shutil.copy2(source_path, target)
        width, height = image_size(target)
        packed = {
            "order": index,
            "review_id": review_id,
            "cohort": row.get("cohort"),
            "source_kind": row.get("source_kind"),
            "dominant_box_source": row.get("dominant_box_source"),
            "box_count_on_image": row.get("box_count_on_image"),
            "selected_box_count": row.get("selected_box_count"),
            "image": target_rel,
            "image_path": str(target.resolve()),
            "source_image_path": str(source_path),
            "width": width,
            "height": height,
            "task": row.get("task"),
            "reference_status": row.get("reference_status") or "pending_human_review",
        }
        output_manifest.append(packed)
        image_map[review_id] = packed
        prediction_template.append(
            {
                "review_id": review_id,
                "image": target_rel,
                "image_path": str(target.resolve()),
                "width": width,
                "height": height,
                "cohort": row.get("cohort"),
                "boxes": [],
            }
        )

    if missing_images:
        write_json(args.out / "missing_images.json", missing_images)
        if not args.allow_missing:
            raise SystemExit(f"{len(missing_images)} benchmark images are missing; see {args.out / 'missing_images.json'}")

    draft_reference = priority_reference_rows(args.review_priority, image_map) if args.review_priority else []
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_manifest": str(args.benchmark_manifest),
        "review_priority": str(args.review_priority) if args.review_priority else "",
        "image_count": len(output_manifest),
        "draft_reference_boxes": sum(len(row.get("boxes") or []) for row in draft_reference),
        "missing_images": len(missing_images),
        "outputs": {
            "benchmark_manifest": "benchmark_manifest.jsonl",
            "codex_prompt": "codex_prompt.md",
            "prediction_template": "codex_predictions_template.jsonl",
            "draft_reference": "draft_reference_boxes.jsonl",
            "readme": "README.md",
            "eval_commands": "eval_commands.md",
        },
    }
    write_jsonl(args.out / "benchmark_manifest.jsonl", output_manifest)
    write_jsonl(args.out / "codex_predictions_template.jsonl", prediction_template)
    write_jsonl(args.out / "draft_reference_boxes.jsonl", draft_reference)
    write_json(args.out / "summary.json", summary)
    (args.out / "codex_prompt.md").write_text(make_prompt(), encoding="utf-8")
    (args.out / "README.md").write_text(make_readme(args, summary), encoding="utf-8")
    (args.out / "eval_commands.md").write_text(make_eval_commands(args), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a portable Codex comparison pack for question segmentation.")
    parser.add_argument("--benchmark-manifest", type=Path, required=True, help="codex_benchmark_manifest.jsonl from question_detector_review_priority.py.")
    parser.add_argument("--review-priority", type=Path, help="review_priority.json used to export draft sanity reference boxes.")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-segmentation-codex-benchmark-pack"))
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    summary = build_pack(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
