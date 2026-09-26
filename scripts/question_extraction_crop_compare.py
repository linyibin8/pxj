"""Compare full-image question extraction with per-question crop extraction.

This diagnostic tool calls the configured OpenAI-compatible vision endpoint
directly. It does not write to the app database and does not modify training
datasets.
"""

from __future__ import annotations

import argparse
import base64
import difflib
import json
import mimetypes
import re
import shutil
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw, ImageOps


DEFAULT_BASE_URL = "http://100.64.0.5:39000/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "evowit-agent27b"


def image_bytes(path: Path, max_side: int = 1800, quality: int = 88) -> tuple[str, bytes]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            alpha = image.getchannel("A")
            background.paste(image.convert("RGB"), mask=alpha)
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        out = BytesIO()
        image.save(out, format="JPEG", quality=quality, optimize=True)
    return "image/jpeg", out.getvalue()


def image_part(path: Path) -> dict[str, Any]:
    try:
        mime, data = image_bytes(path)
    except Exception:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def call_vision(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_paths: list[Path],
    max_tokens: int,
    timeout_seconds: float,
) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for index, path in enumerate(image_paths, start=1):
        content.append({"type": "text", "text": f"Image {index}: filename={path.name}"})
        content.append(image_part(path))
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful vision extraction assistant. Use only visible evidence. "
                    "If text is unclear, write 'unrecognized'. Return valid JSON only."
                ),
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    timeout = httpx.Timeout(timeout_seconds, connect=10)
    with httpx.Client(base_url=base_url, timeout=timeout, trust_env=False) as client:
        response = client.post(
            "/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        data = response.json()
    message = data["choices"][0]["message"]
    return message.get("content") or message.get("reasoning_content") or json.dumps(data, ensure_ascii=False)


def parse_json(raw: str) -> Any:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not starts:
        return {}
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def questions_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        value = payload.get("questions") or payload.get("items")
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def bbox_from_question(question: dict[str, Any]) -> dict[str, float] | None:
    raw = question.get("bbox") or question.get("box") or question.get("rect")
    if not isinstance(raw, dict):
        return None
    x = raw.get("x", raw.get("left"))
    y = raw.get("y", raw.get("top"))
    w = raw.get("w", raw.get("width"))
    h = raw.get("h", raw.get("height"))
    try:
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    # Treat normalized bboxes as the expected format. If pixel-like values slip
    # through, the cropper will reject them later after clamping.
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w = max(0.0, min(1.0 - x, w))
    h = max(0.0, min(1.0 - y, h))
    if w <= 0.01 or h <= 0.01:
        return None
    return {"x": x, "y": y, "w": w, "h": h}


def crop_for_bbox(source: Path, bbox: dict[str, float], out: Path, pad: float = 0.018) -> dict[str, Any]:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        x = max(0.0, bbox["x"] - pad)
        y = max(0.0, bbox["y"] - pad)
        right = min(1.0, bbox["x"] + bbox["w"] + pad)
        bottom = min(1.0, bbox["y"] + bbox["h"] + pad)
        left_px = int(round(x * width))
        top_px = int(round(y * height))
        right_px = int(round(right * width))
        bottom_px = int(round(bottom * height))
        cropped = image.crop((left_px, top_px, max(left_px + 1, right_px), max(top_px + 1, bottom_px)))
        out.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(out, quality=92)
    return {
        "x": x,
        "y": y,
        "w": right - x,
        "h": bottom - y,
        "px": [left_px, top_px, max(left_px + 1, right_px), max(top_px + 1, bottom_px)],
    }


def draw_preview(source: Path, questions: list[dict[str, Any]], out: Path) -> None:
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_width = max(3, int(min(width, height) / 260))
    for idx, question in enumerate(questions, start=1):
        bbox = question.get("bbox_norm") or bbox_from_question(question) or question.get("bbox")
        if not isinstance(bbox, dict):
            continue
        x1 = bbox["x"] * width
        y1 = bbox["y"] * height
        x2 = (bbox["x"] + bbox.get("w", bbox.get("width", 0))) * width
        y2 = (bbox["y"] + bbox.get("h", bbox.get("height", 0))) * height
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=line_width)
        label = str(question.get("number") or question.get("index") or idx)
        draw.text((x1 + 5, y1 + 5), label, fill=(255, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    image.save(out, quality=92)


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    return "".join(ch for ch in text if ch.isalnum())


def question_text(question: dict[str, Any]) -> str:
    parts = [
        question.get("number") or "",
        question.get("text") or question.get("question_text") or question.get("stem") or question.get("title") or "",
        question.get("student_answer") or "",
    ]
    return " ".join(str(part) for part in parts if part).strip()


def best_match(row: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[float, dict[str, Any] | None]:
    source = normalize_text(question_text(row))
    best_score = 0.0
    best_row = None
    for candidate in candidates:
        score = difflib.SequenceMatcher(None, source, normalize_text(question_text(candidate))).ratio()
        if score > best_score:
            best_score = score
            best_row = candidate
    return best_score, best_row


def clipped(value: Any, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


FULL_PROMPT = """
Extract every visible study question from the whole image.
Return strict JSON only:
{
  "questions": [
    {
      "index": 1,
      "number": "visible question number or empty",
      "text": "visible printed question text, key numbers and figure notes; use Chinese where visible",
      "student_answer": "visible handwritten answer/work, or empty",
      "uncertain": false
    }
  ]
}
Rules:
- Include all visible questions, including partially visible ones.
- Do not solve the questions.
- Do not invent hidden text. Write "unrecognized" for unclear text.
"""


SEGMENT_PROMPT = """
Find question regions in the image and extract a short visible text summary.
Return strict JSON only:
{
  "questions": [
    {
      "index": 1,
      "number": "visible question number or empty",
      "bbox": {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0},
      "text": "short visible question text summary",
      "uncertain": false
    }
  ]
}
Rules:
- bbox uses normalized coordinates in [0,1], origin at top-left.
- Each bbox should cover one complete visible question block: prompt, diagrams, options, and student work that belongs to it.
- Avoid desk, keyboard, page margins and unrelated background.
- If a crop/region may contain several very small subquestions that belong to one numbered block, keep them together.
- Include partially visible questions but mark uncertain=true.
- Do not solve the questions.
"""


CROP_PROMPT = """
This image is a crop of one question region, or sometimes a section with several questions.
Extract all visible questions in this crop.
Return strict JSON only:
{
  "questions": [
    {
      "index": 1,
      "number": "visible question number or empty",
      "text": "visible printed question text, key numbers and figure notes",
      "student_answer": "visible handwritten answer/work, or empty",
      "uncertain": false
    }
  ]
}
Rules:
- Do not solve the question.
- Do not invent hidden text. Write "unrecognized" for unclear text.
- If the crop is not a question, return {"questions":[]}.
"""


def build_report(
    image_path: Path,
    full_questions: list[dict[str, Any]],
    segment_questions: list[dict[str, Any]],
    crop_questions: list[dict[str, Any]],
    crop_records: list[dict[str, Any]],
) -> str:
    lines = [
        "# Question Extraction Crop Compare",
        "",
        f"- source image: `{image_path}`",
        f"- full-image question count: {len(full_questions)}",
        f"- segmented crop count: {len(crop_records)}",
        f"- crop-extracted question count: {len(crop_questions)}",
        "",
        "## Full Image Questions",
        "",
    ]
    for idx, question in enumerate(full_questions, start=1):
        lines.append(f"{idx}. number={clipped(question.get('number'), 30)} text={clipped(question_text(question), 220)}")
    lines.extend(["", "## Crop Questions", ""])
    for idx, question in enumerate(crop_questions, start=1):
        crop_name = question.get("crop_file") or ""
        lines.append(
            f"{idx}. crop={crop_name} number={clipped(question.get('number'), 30)} text={clipped(question_text(question), 220)}"
        )
    lines.extend(["", "## Similarity: Crop -> Full", ""])
    for idx, question in enumerate(crop_questions, start=1):
        score, match = best_match(question, full_questions)
        lines.append(
            f"- crop {idx}: best_full_score={score:.3f}; crop={clipped(question_text(question), 120)}; "
            f"full={clipped(question_text(match or {}), 120)}"
        )
    lines.extend(["", "## Similarity: Full -> Crop", ""])
    for idx, question in enumerate(full_questions, start=1):
        score, match = best_match(question, crop_questions)
        lines.append(
            f"- full {idx}: best_crop_score={score:.3f}; full={clipped(question_text(question), 120)}; "
            f"crop={clipped(question_text(match or {}), 120)}"
        )
    lines.extend(["", "## Segmented Boxes", ""])
    for record in crop_records:
        lines.append(
            f"- {record['crop_file']}: number={clipped(record.get('number'), 30)} "
            f"bbox={record.get('bbox_norm')} text={clipped(record.get('text'), 140)}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full-image and per-crop question extraction.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=140.0)
    parser.add_argument("--max-crops", type=int, default=14)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    source_copy = args.out / "source.jpg"
    shutil.copy2(args.image, source_copy)

    print("full image extraction...", flush=True)
    full_raw = call_vision(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        prompt=FULL_PROMPT,
        image_paths=[args.image],
        max_tokens=2600,
        timeout_seconds=args.timeout,
    )
    write_text(args.out / "full_image_raw.txt", full_raw)
    full_payload = parse_json(full_raw)
    write_json(args.out / "full_image.json", full_payload)
    full_questions = questions_from_payload(full_payload)
    print(f"full questions: {len(full_questions)}", flush=True)

    print("segmentation...", flush=True)
    segment_raw = call_vision(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        prompt=SEGMENT_PROMPT,
        image_paths=[args.image],
        max_tokens=2600,
        timeout_seconds=args.timeout,
    )
    write_text(args.out / "segmentation_raw.txt", segment_raw)
    segment_payload = parse_json(segment_raw)
    write_json(args.out / "segmentation.json", segment_payload)
    segment_questions = questions_from_payload(segment_payload)
    print(f"segmented questions: {len(segment_questions)}", flush=True)

    crop_dir = args.out / "crops"
    crop_records: list[dict[str, Any]] = []
    crop_questions: list[dict[str, Any]] = []
    usable_segments = []
    for question in segment_questions:
        bbox = bbox_from_question(question)
        if bbox:
            usable_segments.append((question, bbox))
    usable_segments = usable_segments[: max(0, args.max_crops)]
    for index, (question, bbox) in enumerate(usable_segments, start=1):
        crop_file = crop_dir / f"crop_{index:02d}.jpg"
        padded = crop_for_bbox(args.image, bbox, crop_file)
        record = {
            "index": index,
            "crop_file": crop_file.name,
            "number": question.get("number") or "",
            "text": question.get("text") or "",
            "bbox_norm": bbox,
            "padded_bbox_norm": padded,
        }
        crop_records.append(record)
        print(f"crop extraction {index}/{len(usable_segments)}: {crop_file.name}", flush=True)
        crop_raw = call_vision(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt=CROP_PROMPT,
            image_paths=[crop_file],
            max_tokens=1400,
            timeout_seconds=args.timeout,
        )
        write_text(args.out / "crop_raw" / f"crop_{index:02d}.txt", crop_raw)
        crop_payload = parse_json(crop_raw)
        write_json(args.out / "crop_json" / f"crop_{index:02d}.json", crop_payload)
        for item in questions_from_payload(crop_payload):
            item = dict(item)
            item["crop_file"] = crop_file.name
            item["source_segment_index"] = index
            crop_questions.append(item)

    write_json(args.out / "crops_manifest.json", crop_records)
    write_json(args.out / "crop_questions_combined.json", {"questions": crop_questions})
    draw_preview(args.image, [{**row, "bbox_norm": row["bbox_norm"]} for row in crop_records], args.out / "segmentation_preview.jpg")
    report = build_report(args.image, full_questions, segment_questions, crop_questions, crop_records)
    write_text(args.out / "comparison_report.md", report)
    write_json(
        args.out / "summary.json",
        {
            "generated_at_unix": time.time(),
            "source_image": str(args.image),
            "full_question_count": len(full_questions),
            "segment_question_count": len(segment_questions),
            "crop_count": len(crop_records),
            "crop_question_count": len(crop_questions),
            "outputs": {
                "source": "source.jpg",
                "full_image": "full_image.json",
                "segmentation": "segmentation.json",
                "segmentation_preview": "segmentation_preview.jpg",
                "crops_manifest": "crops_manifest.json",
                "crop_questions_combined": "crop_questions_combined.json",
                "comparison_report": "comparison_report.md",
            },
        },
    )
    print(f"done: {args.out}", flush=True)


if __name__ == "__main__":
    main()
