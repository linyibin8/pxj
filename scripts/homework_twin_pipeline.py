"""Prototype a continuous-camera homework digital-twin pipeline.

This is intentionally independent from the YOLO detector experiments.  It
focuses on the product flow we actually need:

1. Read a stream/folder of camera frames.
2. Keep only sharp, meaningfully changed keyframes.
3. Normalize the page as much as cheap CV allows.
4. Ask a vision model for structured questions, not tight pixel crops.
5. Deduplicate repeated sightings into editable HTML question cards.

The script writes diagnostics only; it does not touch the app database,
training data, or iOS bundle.
"""

from __future__ import annotations

import argparse
import base64
import difflib
import hashlib
import html
import json
import math
import mimetypes
import re
import shutil
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps


DEFAULT_BASE_URL = "http://100.64.0.5:39000/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_MODEL = "evowit-agent27b"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class FrameDecision:
    frame_id: str
    index: int
    source_path: str
    normalized_path: str
    width: int
    height: int
    blur_score: float
    brightness: float
    contrast: float
    phash: str
    accepted: bool
    reason: str
    hamming_from_previous_keyframe: int | None = None
    page_rectified: bool = False
    page_confidence: float = 0.0
    question_count: int = 0
    raw_vlm_path: str | None = None
    parsed_vlm_path: str | None = None


@dataclass
class QuestionRecord:
    question_id: str
    canonical_text: str
    number: str = ""
    subject: str = ""
    question_type: str = ""
    stem_markdown: str = ""
    choices: list[str] = field(default_factory=list)
    answer_area: str = ""
    diagram: dict[str, Any] = field(default_factory=dict)
    html_fragment: str = ""
    anchors: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    uncertain: list[str] = field(default_factory=list)
    source_frames: list[dict[str, Any]] = field(default_factory=list)
    duplicate_count: int = 0


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode in ("RGBA", "LA"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image.convert("RGB"), mask=image.getchannel("A"))
            return background
        return image.convert("RGB")


def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image
    output = image.copy()
    output.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return output


def save_jpeg(image: Image.Image, path: Path, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="JPEG", quality=quality, optimize=True)


def image_to_data_url(path: Path, max_side: int = 1800, quality: int = 88) -> str:
    image = resize_max_side(load_rgb(path), max_side)
    out = BytesIO()
    image.save(out, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(out.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def list_images(input_path: Path, limit: int | None = None) -> list[Path]:
    if input_path.is_file():
        images = [input_path]
    else:
        images = [
            path
            for path in sorted(input_path.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    if limit is not None:
        images = images[:limit]
    return images


def hamming_hex(left: str, right: str) -> int:
    if not left or not right:
        return 64
    width = max(len(left), len(right))
    li = int(left, 16)
    ri = int(right, 16)
    return (li ^ ri).bit_count() + abs(len(left) - len(right)) * 4 if len(left) != len(right) else (li ^ ri).bit_count()


def perceptual_hash(image: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    size = hash_size * highfreq_factor
    gray = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    dct = cv2.dct(pixels)
    low = dct[:hash_size, :hash_size]
    median = np.median(low[1:, :])
    bits = low > median
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bool(bit))
    return f"{value:0{hash_size * hash_size // 4}x}"


def image_metrics(image: Image.Image) -> tuple[float, float, float]:
    gray_image = image.convert("L")
    gray_image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
    gray = np.asarray(gray_image, dtype=np.uint8)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return blur_score, float(gray.mean()), float(gray.std())


def order_points(points: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    rect[0] = points[np.argmin(sums)]
    rect[2] = points[np.argmax(sums)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    return rect


def perspective_warp(image: Image.Image, rect: np.ndarray, max_side: int) -> Image.Image:
    tl, tr, br, bl = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_width = max(1, int(max(width_a, width_b)))
    max_height = max(1, int(max(height_a, height_b)))

    dst = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(np.asarray(image), matrix, (max_width, max_height), borderValue=(255, 255, 255))
    return resize_max_side(Image.fromarray(warped), max_side)


def normalize_page(image: Image.Image, max_side: int = 1800) -> tuple[Image.Image, bool, float]:
    """Try to rectify a sheet/page. Fall back to resized original if unsure."""
    resized = resize_max_side(image, max_side)
    arr = np.asarray(resized)
    height, width = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 135)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(width * height)

    best_rect: np.ndarray | None = None
    best_area = 0.0
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.18:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) == 4:
            best_rect = order_points(approx.reshape(4, 2).astype("float32"))
            best_area = area
            break

    if best_rect is None:
        return resized, False, 0.0

    confidence = min(1.0, best_area / max(image_area, 1.0))
    if confidence < 0.22:
        return resized, False, confidence
    try:
        return perspective_warp(resized, best_rect, max_side), True, confidence
    except cv2.error:
        return resized, False, confidence


def normalize_for_keyframe(image: Image.Image) -> Image.Image:
    normalized = image.convert("L")
    normalized.thumbnail((640, 640), Image.Resampling.LANCZOS)
    return normalized.convert("RGB")


def canonicalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", "", text)
    return "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def question_canonical_text(question: dict[str, Any]) -> str:
    parts: list[str] = [
        str(question.get("number") or ""),
        str(question.get("stem_markdown") or question.get("stem") or question.get("question") or ""),
        str(question.get("answer_area") or ""),
    ]
    choices = question.get("choices")
    if isinstance(choices, list):
        parts.extend(str(choice) for choice in choices)
    diagram = question.get("diagram")
    if isinstance(diagram, dict):
        parts.append(str(diagram.get("description") or ""))
    elif diagram:
        parts.append(str(diagram))
    return canonicalize_text(" ".join(parts))


def question_id_for(canonical_text: str) -> str:
    if not canonical_text:
        canonical_text = f"empty-{time.time_ns()}"
    return hashlib.sha1(canonical_text.encode("utf-8")).hexdigest()[:12]


def parse_jsonish(raw: str) -> Any:
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


def coerce_questions(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        questions = payload.get("questions") or payload.get("items")
        if isinstance(questions, list):
            return [item for item in questions if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def vision_prompt() -> str:
    return """
你在做一个“摄像头连续拍作业”的题目数字孪生系统。

任务：从整页图片中提取可收藏、可编辑的题目对象。不要把重点放在精确像素裁剪框；优先利用题号、文本行、选项、图形说明和页面位置来还原题目结构。

只返回 JSON，不要返回解释。格式：
{
  "page": {
    "quality": "good|ok|poor",
    "visible_question_numbers": ["1", "2"],
    "notes": "页面倾斜/遮挡/手写等观察"
  },
  "questions": [
    {
      "number": "题号，识别不到则空字符串",
      "subject": "math|chinese|english|science|unknown",
      "question_type": "choice|fill_blank|calculation|geometry|reading|unknown",
      "stem_markdown": "题干，用中文，数学公式用 LaTeX，无法看清写 [未识别]",
      "choices": ["A. ...", "B. ..."],
      "answer_area": "题目中给学生作答的空格/横线/图中标记，无法判断则空",
      "diagram": {
        "present": true,
        "description": "如果有几何图/坐标图/表格，描述可见元素、点名、线段、角、已知量",
        "svg": "只有在能可靠还原简单图形时输出 SVG；否则空字符串"
      },
      "html": "<article class=\"question-card\" contenteditable=\"true\">...</article>",
      "anchors": {
        "page_side": "left|right|full|unknown",
        "vertical_band": "top|upper-middle|middle|lower-middle|bottom|unknown",
        "near_question_number": "题号或空"
      },
      "confidence": 0.0,
      "uncertain": ["不确定点"]
    }
  ]
}

要求：
- 不要编答案，不要补图片外没有的信息。
- 如果一道大题含多个小题，可以保留在同一个 question 里，并在 stem_markdown 里列出。
- 几何图尽量转成 SVG 或结构化描述；看不清时保留文字说明，不要硬猜。
- html 字段要是可直接放到页面里的片段，里面不要写 script。
""".strip()


def call_vision_model(
    *,
    image_path: Path,
    base_url: str,
    api_key: str,
    model: str,
    timeout_seconds: float,
    max_tokens: int,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a careful vision extraction assistant. "
                    "Use only visible evidence and return valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt()},
                    {"type": "text", "text": f"Image filename: {image_path.name}"},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    timeout = httpx.Timeout(timeout_seconds, connect=10)
    with httpx.Client(base_url=base_url, timeout=timeout, trust_env=False) as client:
        response = client.post("/chat/completions", json=payload, headers={"Authorization": f"Bearer {api_key}"})
        response.raise_for_status()
        data = response.json()
    message = data["choices"][0]["message"]
    return message.get("content") or message.get("reasoning_content") or json.dumps(data, ensure_ascii=False)


def normalize_question(question: dict[str, Any], frame: FrameDecision) -> QuestionRecord:
    canonical = question_canonical_text(question)
    qid = question_id_for(canonical)
    choices = question.get("choices")
    if not isinstance(choices, list):
        choices = []
    diagram = question.get("diagram")
    if not isinstance(diagram, dict):
        diagram = {"present": bool(diagram), "description": str(diagram or ""), "svg": ""}
    anchors = question.get("anchors")
    if not isinstance(anchors, dict):
        anchors = {}
    uncertain = question.get("uncertain")
    if not isinstance(uncertain, list):
        uncertain = []
    confidence = question.get("confidence", 0.0)
    try:
        confidence_float = float(confidence)
    except (TypeError, ValueError):
        confidence_float = 0.0
    return QuestionRecord(
        question_id=qid,
        canonical_text=canonical,
        number=str(question.get("number") or ""),
        subject=str(question.get("subject") or "unknown"),
        question_type=str(question.get("question_type") or question.get("type") or "unknown"),
        stem_markdown=str(question.get("stem_markdown") or question.get("stem") or question.get("question") or ""),
        choices=[str(choice) for choice in choices],
        answer_area=str(question.get("answer_area") or ""),
        diagram=diagram,
        html_fragment=str(question.get("html") or ""),
        anchors=anchors,
        confidence=max(0.0, min(1.0, confidence_float)),
        uncertain=[str(item) for item in uncertain],
        source_frames=[
            {
                "frame_id": frame.frame_id,
                "source_path": frame.source_path,
                "normalized_path": frame.normalized_path,
            }
        ],
    )


def find_duplicate(
    candidate: QuestionRecord,
    existing: list[QuestionRecord],
    threshold: float,
) -> tuple[QuestionRecord | None, float]:
    if not candidate.canonical_text:
        return None, 0.0
    best: QuestionRecord | None = None
    best_score = 0.0
    for item in existing:
        score = difflib.SequenceMatcher(None, candidate.canonical_text, item.canonical_text).ratio()
        same_number = candidate.number and item.number and candidate.number == item.number
        if same_number and score >= 0.72:
            score = max(score, 0.92)
        if score > best_score:
            best = item
            best_score = score
    if best and best_score >= threshold:
        return best, best_score
    return None, best_score


def record_question(
    records: list[QuestionRecord],
    candidate: QuestionRecord,
    duplicate_threshold: float,
) -> None:
    duplicate, _ = find_duplicate(candidate, records, duplicate_threshold)
    if duplicate is None:
        records.append(candidate)
        return
    duplicate.duplicate_count += 1
    duplicate.source_frames.extend(candidate.source_frames)
    duplicate.confidence = max(duplicate.confidence, candidate.confidence)
    if len(candidate.canonical_text) > len(duplicate.canonical_text):
        duplicate.canonical_text = candidate.canonical_text
        duplicate.stem_markdown = candidate.stem_markdown or duplicate.stem_markdown
        duplicate.choices = candidate.choices or duplicate.choices
        duplicate.answer_area = candidate.answer_area or duplicate.answer_area
        duplicate.diagram = candidate.diagram or duplicate.diagram
        duplicate.html_fragment = candidate.html_fragment or duplicate.html_fragment


def relative_url(path: str | Path, base: Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def markdownish_to_html(text: str) -> str:
    escaped = html.escape(text or "[未识别]")
    escaped = escaped.replace("\n", "<br>")
    return f"<p>{escaped}</p>"


def fallback_question_html(question: QuestionRecord) -> str:
    parts = []
    title = f"第 {question.number} 题" if question.number else "未编号题目"
    parts.append(f"<h2>{html.escape(title)}</h2>")
    parts.append(markdownish_to_html(question.stem_markdown))
    if question.choices:
        parts.append("<ol class=\"choices\">")
        for choice in question.choices:
            parts.append(f"<li>{html.escape(choice)}</li>")
        parts.append("</ol>")
    diagram_desc = ""
    if question.diagram:
        diagram_desc = str(question.diagram.get("description") or "")
    if diagram_desc:
        parts.append(f"<p class=\"diagram-desc\">图形：{html.escape(diagram_desc)}</p>")
    return "\n".join(parts)


def safe_html_fragment(question: QuestionRecord) -> str:
    fragment = question.html_fragment.strip()
    if not fragment or "<" not in fragment:
        return fallback_question_html(question)
    fragment = re.sub(r"<\s*script\b.*?<\s*/\s*script\s*>", "", fragment, flags=re.I | re.S)
    return fragment


def question_to_dict(question: QuestionRecord) -> dict[str, Any]:
    return {
        "question_id": question.question_id,
        "number": question.number,
        "subject": question.subject,
        "question_type": question.question_type,
        "stem_markdown": question.stem_markdown,
        "choices": question.choices,
        "answer_area": question.answer_area,
        "diagram": question.diagram,
        "html_fragment": question.html_fragment,
        "anchors": question.anchors,
        "confidence": question.confidence,
        "uncertain": question.uncertain,
        "source_frames": question.source_frames,
        "duplicate_count": question.duplicate_count,
    }


def frame_to_dict(frame: FrameDecision) -> dict[str, Any]:
    return {
        "frame_id": frame.frame_id,
        "index": frame.index,
        "source_path": frame.source_path,
        "normalized_path": frame.normalized_path,
        "width": frame.width,
        "height": frame.height,
        "blur_score": frame.blur_score,
        "brightness": frame.brightness,
        "contrast": frame.contrast,
        "phash": frame.phash,
        "accepted": frame.accepted,
        "reason": frame.reason,
        "hamming_from_previous_keyframe": frame.hamming_from_previous_keyframe,
        "page_rectified": frame.page_rectified,
        "page_confidence": frame.page_confidence,
        "question_count": frame.question_count,
        "raw_vlm_path": frame.raw_vlm_path,
        "parsed_vlm_path": frame.parsed_vlm_path,
    }


def render_html(out_dir: Path, frames: list[FrameDecision], questions: list[QuestionRecord]) -> str:
    accepted = [frame for frame in frames if frame.accepted]
    skipped = [frame for frame in frames if not frame.accepted]
    frame_rows = []
    for frame in frames:
        status = "accepted" if frame.accepted else "skipped"
        thumb = relative_url(frame.normalized_path, out_dir)
        frame_rows.append(
            f"""
            <button class="frame-tile {status}" data-frame="{html.escape(frame.frame_id)}">
              <img src="{html.escape(thumb)}" alt="{html.escape(frame.frame_id)}">
              <span>{html.escape(frame.frame_id)}</span>
              <small>{html.escape(frame.reason)} · blur {frame.blur_score:.0f}</small>
            </button>
            """
        )

    question_cards = []
    for idx, question in enumerate(questions, start=1):
        sources = []
        for source in question.source_frames[:4]:
            src = relative_url(source["normalized_path"], out_dir)
            sources.append(f"<img src=\"{html.escape(src)}\" alt=\"{html.escape(source['frame_id'])}\">")
        uncertain = ", ".join(question.uncertain) if question.uncertain else "无"
        source_ids = ", ".join(source["frame_id"] for source in question.source_frames)
        question_cards.append(
            f"""
            <section class="question-shell" id="q-{html.escape(question.question_id)}">
              <div class="question-meta">
                <div>
                  <b>#{idx}</b>
                  <span>题号 {html.escape(question.number or "未识别")}</span>
                  <span>{html.escape(question.subject)}</span>
                  <span>{html.escape(question.question_type)}</span>
                </div>
                <div class="confidence">{question.confidence:.2f}</div>
              </div>
              <div class="editable" contenteditable="true" data-question-id="{html.escape(question.question_id)}">
                {safe_html_fragment(question)}
              </div>
              <details>
                <summary>证据与不确定点</summary>
                <p>来源帧：{html.escape(source_ids)}</p>
                <p>重复命中：{question.duplicate_count}</p>
                <p>锚点：{html.escape(json.dumps(question.anchors, ensure_ascii=False))}</p>
                <p>不确定：{html.escape(uncertain)}</p>
                <div class="source-strip">{"".join(sources)}</div>
              </details>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Homework Twin Prototype</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #64748b;
      --line: #d6dde6;
      --paper: #f7f8fb;
      --panel: #ffffff;
      --green: #16794c;
      --red: #b42318;
      --blue: #2457c5;
      --amber: #986100;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--paper);
      color: var(--ink);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: center;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(12px);
    }}
    h1 {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }}
    .stats span, .question-meta span, .confidence {{
      border: 1px solid var(--line);
      background: #fff;
      padding: 5px 8px;
      border-radius: 6px;
      font-size: 12px;
      color: var(--muted);
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(220px, 320px) minmax(0, 1fr);
      min-height: calc(100vh - 62px);
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: #fff;
      padding: 12px;
      overflow: auto;
      max-height: calc(100vh - 62px);
    }}
    .frames {{
      display: grid;
      gap: 10px;
    }}
    .frame-tile {{
      width: 100%;
      display: grid;
      grid-template-columns: 72px minmax(0, 1fr);
      grid-template-rows: auto auto;
      column-gap: 10px;
      align-items: center;
      border: 1px solid var(--line);
      background: #fff;
      padding: 8px;
      border-radius: 6px;
      text-align: left;
      cursor: pointer;
    }}
    .frame-tile.accepted {{ border-left: 4px solid var(--green); }}
    .frame-tile.skipped {{ border-left: 4px solid var(--muted); opacity: 0.72; }}
    .frame-tile img {{
      grid-row: 1 / 3;
      width: 72px;
      height: 56px;
      object-fit: cover;
      border-radius: 4px;
      border: 1px solid var(--line);
    }}
    .frame-tile span {{
      font-weight: 650;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .frame-tile small {{
      color: var(--muted);
      line-height: 1.35;
    }}
    .questions {{
      padding: 18px;
      display: grid;
      gap: 16px;
      align-content: start;
    }}
    .question-shell {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .question-meta {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }}
    .question-meta div:first-child {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      align-items: center;
    }}
    .question-meta b {{ color: var(--blue); }}
    .editable {{
      padding: 14px 16px;
      min-height: 92px;
      line-height: 1.65;
      outline: none;
      background: #fff;
    }}
    .editable:focus {{
      box-shadow: inset 0 0 0 2px rgba(36, 87, 197, 0.28);
    }}
    .editable h1, .editable h2, .editable h3 {{
      font-size: 17px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }}
    .editable p {{ margin: 6px 0; }}
    .editable svg {{
      max-width: 100%;
      height: auto;
      border: 1px solid var(--line);
      background: #fff;
    }}
    details {{
      padding: 10px 12px 12px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 13px;
    }}
    summary {{
      color: var(--ink);
      cursor: pointer;
      margin-bottom: 8px;
    }}
    .source-strip {{
      display: flex;
      gap: 8px;
      overflow-x: auto;
      padding-top: 4px;
    }}
    .source-strip img {{
      width: 160px;
      height: 112px;
      object-fit: cover;
      border: 1px solid var(--line);
      border-radius: 5px;
      flex: 0 0 auto;
    }}
    .actions {{
      display: flex;
      gap: 8px;
      justify-content: flex-end;
    }}
    button.action {{
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 7px 10px;
      font-weight: 650;
      cursor: pointer;
    }}
    @media (max-width: 860px) {{
      header, main {{ grid-template-columns: 1fr; }}
      aside {{ max-height: none; border-right: 0; border-bottom: 1px solid var(--line); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Homework Twin Prototype</h1>
    <div class="stats">
      <span>frames {len(frames)}</span>
      <span>keyframes {len(accepted)}</span>
      <span>skipped {len(skipped)}</span>
      <span>questions {len(questions)}</span>
      <button class="action" id="download">下载编辑结果 JSON</button>
    </div>
  </header>
  <main>
    <aside>
      <div class="frames">
        {"".join(frame_rows)}
      </div>
    </aside>
    <section class="questions">
      {"".join(question_cards) if question_cards else '<p>没有生成题目。可以降低阈值或启用 VLM。</p>'}
    </section>
  </main>
  <script>
    const download = document.querySelector('#download');
    download.addEventListener('click', () => {{
      const edited = Array.from(document.querySelectorAll('.editable')).map(node => ({{
        question_id: node.dataset.questionId,
        html: node.innerHTML
      }}));
      const blob = new Blob([JSON.stringify({{ edited_questions: edited }}, null, 2)], {{ type: 'application/json' }});
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = 'edited_questions.json';
      link.click();
      URL.revokeObjectURL(link.href);
    }});
  </script>
</body>
</html>
"""


def render_readme(
    out_dir: Path,
    frames: list[FrameDecision],
    questions: list[QuestionRecord],
    started_at: float,
    args: argparse.Namespace,
) -> str:
    accepted = [frame for frame in frames if frame.accepted]
    elapsed = time.time() - started_at
    return f"""# Homework Twin Prototype

This directory was generated by `scripts/homework_twin_pipeline.py`.

## Summary

- Input: `{args.input}`
- Frames scanned: {len(frames)}
- Keyframes accepted: {len(accepted)}
- Unique questions: {len(questions)}
- VLM enabled: {not args.no_vlm}
- Elapsed seconds: {elapsed:.1f}

## Files

- `index.html`: editable digital-twin question notebook.
- `frames_manifest.json`: frame quality, page normalization, and keyframe decisions.
- `questions.json`: deduped structured questions.
- `normalized_frames/`: cheap page-normalized evidence frames.
- `raw_vlm/`: raw and parsed model responses for accepted keyframes.

## Design Notes

This prototype deliberately avoids making tight per-question crop boxes the
primary truth. It uses page-level normalization, question structure, and
duplicate matching first. Crops can still be used later as evidence, but they
should be loose and self-checked against recognized question numbers.
"""


def create_demo_burst(source_paths: list[Path], out_dir: Path, variants_per_source: int = 4) -> Path:
    demo_dir = out_dir / "demo_camera_burst"
    clean_dir(demo_dir)
    frame_index = 1
    rotations = [-1.0, 0.4, -0.3, 0.8, -0.6, 0.2]
    brightness = [0.96, 1.02, 1.0, 1.04, 0.98, 1.01]
    for source_index, source_path in enumerate(source_paths, start=1):
        source = resize_max_side(load_rgb(source_path), 1500)
        for variant_index in range(variants_per_source):
            image = source.copy()
            rotation = rotations[(source_index + variant_index) % len(rotations)]
            image = image.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(248, 248, 248))
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness[(source_index + variant_index) % len(brightness)])

            # A small crop/pad jitter mimics a hand-adjusted camera without
            # changing the actual page content.
            width, height = image.size
            jitter_x = int(width * (0.004 * ((variant_index % 3) - 1)))
            jitter_y = int(height * (0.004 * (((variant_index + 1) % 3) - 1)))
            crop = image.crop(
                (
                    max(0, jitter_x),
                    max(0, jitter_y),
                    min(width, width + jitter_x),
                    min(height, height + jitter_y),
                )
            )
            canvas = Image.new("RGB", (width, height), (248, 248, 248))
            canvas.paste(crop, (max(0, -jitter_x), max(0, -jitter_y)))
            save_jpeg(canvas, demo_dir / f"frame_{frame_index:03d}_source_{source_index}.jpg", quality=90)
            frame_index += 1
    return demo_dir


def scan_frames(args: argparse.Namespace, image_paths: list[Path], out_dir: Path) -> tuple[list[FrameDecision], list[Path]]:
    normalized_dir = out_dir / "normalized_frames"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    frames: list[FrameDecision] = []
    accepted_paths: list[Path] = []
    previous_key_hash: str | None = None

    for index, image_path in enumerate(image_paths, start=1):
        frame_id = f"frame_{index:04d}"
        source = load_rgb(image_path)
        normalized, rectified, confidence = normalize_page(source, max_side=args.max_side)
        blur, brightness, contrast = image_metrics(normalized)
        phash = perceptual_hash(normalize_for_keyframe(normalized))
        normalized_path = normalized_dir / f"{frame_id}.jpg"
        save_jpeg(normalized, normalized_path)

        hamming: int | None = None
        accepted = True
        reason = "new_keyframe"
        if blur < args.min_blur:
            accepted = False
            reason = "too_blurry"
        elif previous_key_hash is not None:
            hamming = hamming_hex(phash, previous_key_hash)
            if hamming <= args.near_duplicate_hamming:
                accepted = False
                reason = f"near_duplicate_hamming_{hamming}"
            elif hamming < args.new_content_hamming:
                accepted = False
                reason = f"minor_change_hamming_{hamming}"

        if accepted:
            previous_key_hash = phash
            accepted_paths.append(normalized_path)

        frames.append(
            FrameDecision(
                frame_id=frame_id,
                index=index,
                source_path=str(image_path),
                normalized_path=str(normalized_path),
                width=normalized.width,
                height=normalized.height,
                blur_score=blur,
                brightness=brightness,
                contrast=contrast,
                phash=phash,
                accepted=accepted,
                reason=reason,
                hamming_from_previous_keyframe=hamming,
                page_rectified=rectified,
                page_confidence=confidence,
            )
        )
    return frames, accepted_paths


def run_vlm_extraction(args: argparse.Namespace, frames: list[FrameDecision], out_dir: Path) -> list[QuestionRecord]:
    raw_dir = out_dir / "raw_vlm"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records: list[QuestionRecord] = []
    processed = 0

    for frame in frames:
        if not frame.accepted:
            continue
        if args.max_vlm_frames is not None and processed >= args.max_vlm_frames:
            frame.reason = f"{frame.reason};vlm_limit_not_called"
            continue
        processed += 1
        image_path = Path(frame.normalized_path)
        raw_path = raw_dir / f"{frame.frame_id}.raw.txt"
        parsed_path = raw_dir / f"{frame.frame_id}.json"
        print(f"[vlm] extracting {frame.frame_id} from {image_path.name}", flush=True)
        raw = call_vision_model(
            image_path=image_path,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout_seconds=args.timeout,
            max_tokens=args.max_tokens,
        )
        write_text(raw_path, raw)
        payload = parse_jsonish(raw)
        write_json(parsed_path, payload)
        questions = coerce_questions(payload)
        frame.question_count = len(questions)
        frame.raw_vlm_path = str(raw_path)
        frame.parsed_vlm_path = str(parsed_path)
        for item in questions:
            record_question(records, normalize_question(item, frame), args.duplicate_threshold)
    return records


def process(args: argparse.Namespace) -> Path:
    out_dir = args.out.resolve()
    started_at = time.time()
    if args.clean:
        clean_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    input_path = args.input.resolve()
    if args.demo_from:
        demo_sources = [path.resolve() for path in args.demo_from]
        input_path = create_demo_burst(demo_sources, out_dir, variants_per_source=args.demo_variants)
        args.input = input_path

    image_paths = list_images(input_path, limit=args.limit)
    if not image_paths:
        raise SystemExit(f"No images found in {input_path}")

    frames, _ = scan_frames(args, image_paths, out_dir)
    questions: list[QuestionRecord] = []
    if not args.no_vlm:
        questions = run_vlm_extraction(args, frames, out_dir)

    write_json(out_dir / "frames_manifest.json", [frame_to_dict(frame) for frame in frames])
    write_json(out_dir / "questions.json", [question_to_dict(question) for question in questions])
    write_text(out_dir / "index.html", render_html(out_dir, frames, questions))
    write_text(out_dir / "README.md", render_readme(out_dir, frames, questions, started_at, args))

    summary = {
        "input": str(input_path),
        "out_dir": str(out_dir),
        "frames": len(frames),
        "accepted_keyframes": sum(1 for frame in frames if frame.accepted),
        "skipped_frames": sum(1 for frame in frames if not frame.accepted),
        "unique_questions": len(questions),
        "vlm_enabled": not args.no_vlm,
        "elapsed_seconds": round(time.time() - started_at, 2),
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuous homework camera to editable HTML digital-twin prototype.")
    parser.add_argument("input", type=Path, nargs="?", default=Path("."), help="Image file or directory of camera frames.")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/homework-twin-prototype"), help="Output directory.")
    parser.add_argument("--clean", action="store_true", help="Clear output directory before running.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum frames to scan.")
    parser.add_argument("--min-blur", type=float, default=35.0, help="Reject frames below this Laplacian variance.")
    parser.add_argument("--near-duplicate-hamming", type=int, default=7, help="pHash distance treated as same frame.")
    parser.add_argument("--new-content-hamming", type=int, default=14, help="pHash distance required for a new keyframe.")
    parser.add_argument("--max-side", type=int, default=1800, help="Max side for normalized evidence images.")
    parser.add_argument("--no-vlm", action="store_true", help="Only run frame filtering/page normalization; skip model extraction.")
    parser.add_argument("--max-vlm-frames", type=int, default=3, help="Maximum accepted keyframes sent to the VLM.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=170.0)
    parser.add_argument("--max-tokens", type=int, default=2600)
    parser.add_argument("--duplicate-threshold", type=float, default=0.88)
    parser.add_argument(
        "--demo-from",
        type=Path,
        nargs="+",
        help="Create a synthetic continuous-camera burst from these source images, then process that burst.",
    )
    parser.add_argument("--demo-variants", type=int, default=4, help="Synthetic burst frames per demo source.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    process(args)


if __name__ == "__main__":
    main()
