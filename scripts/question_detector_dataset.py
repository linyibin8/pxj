"""Export weak question-region labels for an iOS/Core ML detector.

The exporter turns existing question-crop manifests or session_question_crops
rows into object-detection datasets. It intentionally exports safe V3 question
rectangles, not the historical thin client crop rectangles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from question_crop_benchmark import (
    Rect,
    clamp_rect,
    is_bad,
    load_json,
    rect_only_v3_server_crop,
    row_signature_similar,
)


CATEGORY_ID = 1
CATEGORY_NAME = "question_block"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class SourceRow:
    source_key: str
    split_key: str
    source_path: Path
    source_filename: str
    item: dict[str, Any]
    origin: str


@dataclass
class Label:
    source_key: str
    split_key: str
    source_path: Path
    source_filename: str
    export_filename: str
    split: str
    image_width: int
    image_height: int
    rect: Rect
    item: dict[str, Any]
    origin: str
    quality_flags: list[str]

    @property
    def bbox_px(self) -> tuple[int, int, int, int]:
        x = int(round(self.rect["x"] * self.image_width))
        y = int(round(self.rect["y"] * self.image_height))
        w = int(round(self.rect["width"] * self.image_width))
        h = int(round(self.rect["height"] * self.image_height))
        x = max(0, min(self.image_width - 1, x))
        y = max(0, min(self.image_height - 1, y))
        w = max(1, min(self.image_width - x, w))
        h = max(1, min(self.image_height - y, h))
        return x, y, w, h


@dataclass
class DatasetImage:
    source_key: str
    split_key: str
    source_path: Path
    source_filename: str
    export_filename: str
    split: str
    image_width: int
    image_height: int
    origin: str
    label_count: int
    is_negative: bool
    negative_kind: str
    quality_flags: list[str]
    item: dict[str, Any]


def stable_id(value: str, length: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
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
    payload = read_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        rows = payload.get("empty_pages") or payload.get("negative_images") or payload.get("images") or payload.get("items") or []
        return [item for item in rows if isinstance(item, dict)]
    return []


def parse_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def normalized_source_name(item: dict[str, Any]) -> str:
    return str(
        item.get("src_filename")
        or item.get("source_filename")
        or item.get("sourceFilename")
        or item.get("image_filename")
        or item.get("image")
        or item.get("filename")
        or ""
    ).strip()


def normalized_image_path(item: dict[str, Any]) -> str:
    return str(
        item.get("source_path")
        or item.get("image_path")
        or item.get("path")
        or item.get("file")
        or ""
    ).strip()


def infer_split_key(item: dict[str, Any], source_name: str, origin: str, split_scope: str = "batch") -> str:
    session_id = str(item.get("session_id") or "").strip()
    batch_id = str(item.get("batch_id") or "").strip()
    parts = Path(source_name).stem.split("_")
    filename_session = parts[0] if parts and len(parts[0]) >= 8 else ""
    filename_batch = parts[1] if len(parts) >= 2 and len(parts[1]) >= 8 else ""
    if split_scope == "source":
        return f"{origin}:{source_name}"
    if split_scope == "session":
        if session_id:
            return session_id
        if filename_session:
            return filename_session
    if session_id and batch_id:
        return f"{session_id}:{batch_id}"
    if filename_session and filename_batch:
        return f"{filename_session}:{filename_batch}"
    if session_id:
        return session_id
    if filename_session:
        return filename_session
    return f"{origin}:{source_name}"


def is_source_image(path: Path) -> bool:
    if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        return False
    return path.is_file()


def is_probable_crop_image(path: Path) -> bool:
    stem = path.stem.lower()
    return (
        stem.startswith("qcrop")
        or stem.startswith("crop_")
        or stem.startswith("crop-")
        or "_qcrop_" in stem
        or "-qcrop-" in stem
    )


def load_diagnostics_rows(root: Path, split_scope: str) -> list[SourceRow]:
    manifest_path = root / "manifest.json"
    images_dir = root / "images"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"diagnostics manifest not found: {manifest_path}")
    payload = read_json(manifest_path)
    rows = payload.get("manifest") or payload.get("crops") or payload.get("items") or []
    session_id = str(payload.get("session_id") or "").strip()
    result: list[SourceRow] = []
    for index, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        item = {**item}
        if session_id and not item.get("session_id"):
            item["session_id"] = session_id
        source_name = normalized_source_name(item)
        if not source_name:
            continue
        source_path = images_dir / Path(source_name).name
        origin = f"diagnostics:{root}"
        result.append(
                SourceRow(
                    source_key=f"diagnostics:{root}:{source_name}",
                    split_key=infer_split_key(item, source_name, origin, split_scope),
                source_path=source_path,
                source_filename=Path(source_name).name,
                item={**item, "_row_index": index},
                origin=origin,
            )
        )
    return result


def resolve_manifest_image_path(manifest_path: Path, item: dict[str, Any]) -> Path:
    raw_path = normalized_image_path(item)
    source_name = normalized_source_name(item)
    candidates: list[Path] = []
    if raw_path:
        raw = Path(raw_path)
        candidates.append(raw if raw.is_absolute() else manifest_path.parent / raw)
    if source_name:
        source = Path(source_name)
        candidates.append(source if source.is_absolute() else manifest_path.parent / source)
        candidates.append(manifest_path.parent / "images" / source.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if candidates:
        return candidates[0]
    return manifest_path.parent / source_name


def load_empty_page_manifest_rows(manifest_path: Path, split_scope: str) -> list[SourceRow]:
    rows = read_json_or_jsonl(manifest_path)
    result: list[SourceRow] = []
    origin = f"empty_manifest:{manifest_path}"
    for index, item in enumerate(rows):
        item = {**item}
        source_path = resolve_manifest_image_path(manifest_path, item)
        source_name = normalized_source_name(item) or source_path.name
        if not source_name:
            continue
        if not item.get("negative_kind"):
            item["negative_kind"] = "empty_page"
        result.append(
            SourceRow(
                source_key=str(item.get("source_key") or f"{origin}:{source_path}"),
                split_key=infer_split_key(item, source_name, origin, split_scope),
                source_path=source_path,
                source_filename=Path(source_name).name,
                item={**item, "_row_index": index, "negative_source": "empty_page_manifest"},
                origin=origin,
            )
        )
    return result


def load_negative_image_dir_rows(root: Path, split_scope: str) -> list[SourceRow]:
    if not root.is_dir():
        raise FileNotFoundError(f"negative image dir not found: {root}")
    result: list[SourceRow] = []
    origin = f"negative_dir:{root}"
    for index, source_path in enumerate(sorted(root.rglob("*"))):
        if not is_source_image(source_path) or is_probable_crop_image(source_path):
            continue
        source_name = source_path.name
        item = {
            "source_filename": source_name,
            "negative_kind": "empty_page",
            "negative_source": "negative_images_dir",
            "_row_index": index,
        }
        result.append(
            SourceRow(
                source_key=f"{origin}:{source_path}",
                split_key=infer_split_key(item, source_name, origin, split_scope),
                source_path=source_path,
                source_filename=source_name,
                item=item,
                origin=origin,
            )
        )
    return result


def approved_statuses_from_csv(value: str) -> set[str]:
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def label_status(item: dict[str, Any]) -> str:
    for key in ("annotation_status", "review_status", "status", "label_status"):
        value = str(item.get(key) or "").strip().lower()
        if value:
            return value
    return ""


def bbox_px_to_norm(bbox: dict[str, Any] | list[Any], image_width: int, image_height: int) -> Rect | None:
    try:
        if isinstance(bbox, dict):
            x = float(bbox.get("x"))
            y = float(bbox.get("y"))
            width = float(bbox.get("width"))
            height = float(bbox.get("height"))
        elif isinstance(bbox, list) and len(bbox) >= 4:
            x, y, width, height = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        else:
            return None
    except (TypeError, ValueError):
        return None
    if image_width <= 0 or image_height <= 0 or width <= 0 or height <= 0:
        return None
    return clamp_rect({"x": x / image_width, "y": y / image_height, "width": width / image_width, "height": height / image_height})


def resolve_reviewed_image_path(base_dir: Path, image_value: str) -> Path:
    text = str(image_value or "").strip()
    if not text:
        return base_dir
    if "://" in text:
        text = text.rstrip("/").split("/")[-1]
    rel = Path(text)
    if rel.is_absolute():
        return rel
    candidates = [
        base_dir / rel,
        base_dir.parent / rel,
        base_dir / "images" / rel.name,
        base_dir.parent / "images" / rel.name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def reviewed_source_row(
    manifest_path: Path,
    base_dir: Path,
    image_value: str,
    image_width: int,
    image_height: int,
    bbox_norm: Rect,
    item: dict[str, Any],
    row_index: int,
    box_index: int,
    split_scope: str,
) -> SourceRow | None:
    source_path = resolve_reviewed_image_path(base_dir, image_value)
    source_name = source_path.name
    if not source_name:
        return None
    session_id = str(item.get("session_id") or "").strip()
    batch_id = str(item.get("batch_id") or "").strip()
    source_image_id = str(item.get("source_image_id") or item.get("image_id") or Path(source_name).stem).strip()
    origin = f"reviewed_prelabel:{manifest_path}"
    question_key = str(item.get("question_key") or f"reviewed:{source_image_id}:{box_index + 1}").strip()
    row_item = {
        **item,
        "source_filename": source_name,
        "source_path": str(source_path),
        "session_id": session_id,
        "batch_id": batch_id,
        "source_image_id": source_image_id,
        "question_key": question_key,
        "question_key_strength": "human_reviewed",
        "question_index": int(item.get("question_index") or box_index + 1),
        "confidence": float(item.get("confidence") or 1.0),
        "bbox_norm": bbox_norm,
        "image_width": image_width,
        "image_height": image_height,
        "_row_index": row_index,
        "_box_index": box_index,
        "_detector_bbox_source": "reviewed_prelabel",
        "_human_reviewed": True,
    }
    return SourceRow(
        source_key=f"reviewed_prelabel:{source_image_id or source_path}",
        split_key=infer_split_key(row_item, source_name, origin, split_scope),
        source_path=source_path,
        source_filename=source_name,
        item=row_item,
        origin=origin,
    )


def load_reviewed_prelabel_jsonl(manifest_path: Path, approved_statuses: set[str], split_scope: str) -> list[SourceRow]:
    rows = read_jsonl_dicts(manifest_path)
    result: list[SourceRow] = []
    base_dir = manifest_path.parent.parent if manifest_path.parent.name == "annotations" else manifest_path.parent
    for row_index, row in enumerate(rows):
        image_value = str(row.get("image") or row.get("file_name") or "").strip()
        image_width = int(row.get("width") or 0)
        image_height = int(row.get("height") or 0)
        row_status = label_status(row)
        candidate = row.get("candidate") if isinstance(row.get("candidate"), dict) else {}
        base_item = {
            "session_id": row.get("session_id") or candidate.get("session_id") or "",
            "batch_id": row.get("batch_id") or candidate.get("batch_id") or "",
            "source_image_id": row.get("source_image_id") or candidate.get("image_id") or "",
            "candidate_kind": row.get("candidate_kind") or candidate.get("candidate_kind") or "",
            "review_source": "draft_boxes_jsonl",
        }
        for box_index, box in enumerate(row.get("boxes") or []):
            if not isinstance(box, dict):
                continue
            status = label_status(box) or row_status
            if status not in approved_statuses:
                continue
            bbox_norm = bbox_px_to_norm(box.get("bbox_px") or box.get("bbox") or {}, image_width, image_height)
            if bbox_norm is None:
                continue
            source_row = reviewed_source_row(
                manifest_path,
                base_dir,
                image_value,
                image_width,
                image_height,
                bbox_norm,
                {**base_item, "annotation_status": status, "prelabel_source": box.get("source") or ""},
                row_index,
                box_index,
                split_scope,
            )
            if source_row is not None:
                result.append(source_row)
    return result


def load_reviewed_prelabel_coco(manifest_path: Path, approved_statuses: set[str], split_scope: str) -> list[SourceRow]:
    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        return []
    images = {int(image["id"]): image for image in payload.get("images") or [] if isinstance(image, dict) and image.get("id") is not None}
    result: list[SourceRow] = []
    base_dir = manifest_path.parent.parent if manifest_path.parent.name == "annotations" else manifest_path.parent
    for ann_index, ann in enumerate(payload.get("annotations") or []):
        if not isinstance(ann, dict):
            continue
        try:
            image = images.get(int(ann.get("image_id") or 0))
        except (TypeError, ValueError):
            image = None
        if not image:
            continue
        status = label_status(ann) or label_status(image)
        if status not in approved_statuses:
            continue
        image_width = int(image.get("width") or 0)
        image_height = int(image.get("height") or 0)
        bbox_norm = bbox_px_to_norm(ann.get("bbox") or [], image_width, image_height)
        if bbox_norm is None:
            continue
        source_row = reviewed_source_row(
            manifest_path,
            base_dir,
            str(image.get("file_name") or ""),
            image_width,
            image_height,
            bbox_norm,
            {
                "session_id": image.get("session_id") or "",
                "batch_id": image.get("batch_id") or "",
                "source_image_id": image.get("source_image_id") or image.get("id") or "",
                "candidate_kind": image.get("candidate_kind") or "",
                "annotation_status": status,
                "review_source": "coco",
            },
            ann_index,
            ann_index,
            split_scope,
        )
        if source_row is not None:
            result.append(source_row)
    return result


def load_reviewed_prelabel_label_studio(manifest_path: Path, split_scope: str) -> list[SourceRow]:
    payload = read_json(manifest_path)
    if not isinstance(payload, list):
        return []
    result: list[SourceRow] = []
    base_dir = manifest_path.parent.parent if manifest_path.parent.name == "annotations" else manifest_path.parent
    for task_index, task in enumerate(payload):
        if not isinstance(task, dict):
            continue
        data = task.get("data") if isinstance(task.get("data"), dict) else {}
        image_value = str(data.get("image") or data.get("Image") or "").strip()
        if not image_value:
            continue
        source_path = resolve_reviewed_image_path(base_dir, image_value)
        try:
            with Image.open(source_path) as image:
                image_width, image_height = ImageOps.exif_transpose(image).size
        except Exception:
            continue
        annotations = task.get("annotations") if isinstance(task.get("annotations"), list) else []
        for annotation_index, annotation in enumerate(annotations):
            if not isinstance(annotation, dict) or annotation.get("was_cancelled"):
                continue
            for result_index, result_item in enumerate(annotation.get("result") or []):
                if not isinstance(result_item, dict):
                    continue
                value = result_item.get("value") if isinstance(result_item.get("value"), dict) else {}
                labels = value.get("rectanglelabels") or []
                if CATEGORY_NAME not in labels:
                    continue
                try:
                    bbox_norm = clamp_rect(
                        {
                            "x": float(value.get("x")) / 100.0,
                            "y": float(value.get("y")) / 100.0,
                            "width": float(value.get("width")) / 100.0,
                            "height": float(value.get("height")) / 100.0,
                        }
                    )
                except (TypeError, ValueError):
                    continue
                source_row = reviewed_source_row(
                    manifest_path,
                    base_dir,
                    image_value,
                    image_width,
                    image_height,
                    bbox_norm,
                    {
                        "annotation_status": "label_studio_annotation",
                        "review_source": "label_studio",
                    },
                    task_index,
                    annotation_index * 1000 + result_index,
                    split_scope,
                )
                if source_row is not None:
                    result.append(source_row)
    return result


def read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
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


def load_reviewed_prelabel_rows(path: Path, approved_statuses: set[str], split_scope: str) -> list[SourceRow]:
    if path.is_dir():
        candidates = [
            path / "annotations" / "reviewed_boxes.jsonl",
            path / "annotations" / "approved_boxes.jsonl",
            path / "annotations" / "draft_boxes.jsonl",
            path / "annotations" / "coco_reviewed.json",
            path / "annotations" / "coco_draft.json",
            path / "annotations" / "label_studio_reviewed_tasks.json",
            path / "annotations" / "label_studio_tasks.json",
        ]
        rows: list[SourceRow] = []
        for candidate in candidates:
            if candidate.is_file():
                rows.extend(load_reviewed_prelabel_rows(candidate, approved_statuses, split_scope))
        return rows
    if not path.is_file():
        raise FileNotFoundError(f"reviewed prelabel file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        return load_reviewed_prelabel_jsonl(path, approved_statuses, split_scope)
    payload = read_json(path)
    if isinstance(payload, dict) and "annotations" in payload and "images" in payload:
        return load_reviewed_prelabel_coco(path, approved_statuses, split_scope)
    if isinstance(payload, list):
        return load_reviewed_prelabel_label_studio(path, split_scope)
    return []


def load_sqlite_rows(db_path: Path, data_dir: Path, split_scope: str) -> list[SourceRow]:
    if not db_path.is_file():
        raise FileNotFoundError(f"sqlite database not found: {db_path}")
    images_dir = data_dir / "images"
    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"session_question_crops", "images"}.issubset(tables):
        return []
    query = """
        SELECT c.*,
               i.filename AS source_filename,
               i.capture_meta AS source_capture_meta,
               COALESCE(obs.novelty_status, 'unknown') AS frame_novelty_status,
               obs.duplicate_of_image_id,
               obs.visual_hash,
               obs.text_hash AS frame_text_hash,
               obs.visual_distance,
               obs.text_distance,
               COALESCE(obs.signal_summary, '') AS frame_signal_summary
        FROM session_question_crops c
        LEFT JOIN images i ON i.id = c.image_id
        LEFT JOIN session_observations obs ON obs.image_id = c.image_id
        WHERE COALESCE(c.status, 'ready') = 'ready'
          AND COALESCE(c.crop_filename, '') != ''
          AND COALESCE(obs.novelty_status, 'unknown') NOT IN ('invalid', 'duplicate')
        ORDER BY c.session_id, c.sequence_index, c.manifest_index
    """
    result: list[SourceRow] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for index, row in enumerate(conn.execute(query).fetchall()):
            item = dict(row)
            source_name = normalized_source_name(item)
            if not source_name:
                continue
            origin = f"sqlite:{db_path}"
            result.append(
                SourceRow(
                    source_key=f"sqlite:{db_path}:{item.get('image_id') or source_name}",
                    split_key=infer_split_key(item, source_name, origin, split_scope),
                    source_path=images_dir / Path(source_name).name,
                    source_filename=Path(source_name).name,
                    item={**item, "_row_index": index},
                    origin=origin,
                )
            )
    return result


def sqlite_row_is_safe_textless_negative(item: dict[str, Any]) -> bool:
    if str(item.get("qa_event_id") or "").strip():
        return False
    kind = str(item.get("source_kind") or item.get("kind") or "").strip().lower()
    if kind in {"qa", "extract", "grade"}:
        return False
    tokens = parse_json_value(item.get("text_tokens"))
    if isinstance(tokens, list) and tokens:
        return False
    if isinstance(tokens, dict) and tokens:
        return False
    text_hash = str(item.get("text_hash") or "").strip()
    if text_hash:
        return False
    signal = str(item.get("signal_summary") or "").strip().lower()
    if signal and not any(marker in signal for marker in ("文字0", "text0", "text 0", "0 text")):
        return False
    return True


def load_sqlite_empty_page_rows(db_path: Path, data_dir: Path, split_scope: str, mode: str) -> list[SourceRow]:
    if not db_path.is_file():
        raise FileNotFoundError(f"sqlite database not found: {db_path}")
    images_dir = data_dir / "images"
    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"session_observations", "images"}.issubset(tables):
        return []
    crop_join = ""
    crop_filter = ""
    if "session_question_crops" in tables:
        crop_join = """
        LEFT JOIN session_question_crops c
          ON c.image_id = obs.image_id
         AND COALESCE(c.status, 'ready') = 'ready'
         AND COALESCE(c.crop_filename, '') != ''
        """
        crop_filter = "AND c.id IS NULL"
    qa_join = ""
    qa_select = "'' AS qa_event_id"
    if "qa_events" in tables:
        qa_select = "qa.id AS qa_event_id"
        qa_join = """
        LEFT JOIN (
            SELECT image_id, MIN(id) AS id
            FROM qa_events
            GROUP BY image_id
        ) qa ON qa.image_id = obs.image_id
        """
    query = """
        SELECT obs.*,
               i.filename AS source_filename,
               i.kind AS source_kind,
               i.page_hint AS source_page_hint,
               i.question_hint AS source_question_hint,
               i.capture_meta AS source_capture_meta,
               {qa_select}
        FROM session_observations obs
        LEFT JOIN images i ON i.id = obs.image_id
        {crop_join}
        {qa_join}
        WHERE 1=1
          {crop_filter}
          AND COALESCE(obs.novelty_status, 'unknown') NOT IN ('invalid', 'duplicate')
        ORDER BY obs.session_id, obs.sequence_index
    """.format(qa_select=qa_select, crop_join=crop_join, qa_join=qa_join, crop_filter=crop_filter)
    result: list[SourceRow] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            records = conn.execute(query).fetchall()
        except sqlite3.Error as exc:
            raise RuntimeError(f"could not load sqlite empty pages from {db_path}: {exc}") from exc
        for index, row in enumerate(records):
            item = dict(row)
            if mode == "textless" and not sqlite_row_is_safe_textless_negative(item):
                continue
            source_name = normalized_source_name(item)
            if not source_name:
                continue
            item["negative_kind"] = "empty_page"
            item["negative_source"] = f"sqlite_empty_page:{mode}"
            origin = f"sqlite_empty:{db_path}"
            result.append(
                SourceRow(
                    source_key=f"{origin}:{item.get('image_id') or source_name}",
                    split_key=infer_split_key(item, source_name, origin, split_scope),
                    source_path=images_dir / Path(source_name).name,
                    source_filename=Path(source_name).name,
                    item={**item, "_row_index": index},
                    origin=origin,
                )
            )
    return result


def discover_sqlite_dbs(data_dir: Path) -> list[Path]:
    accounts = data_dir / "accounts"
    if not accounts.is_dir():
        return []
    candidates = sorted(accounts.glob("*/*.sqlite3"))
    return [path for path in candidates if path.is_file()]


def discover_diagnostics_roots(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if (root / "manifest.json").is_file() and (root / "images").is_dir():
        return [root]
    return sorted(path.parent for path in root.rglob("manifest.json") if (path.parent / "images").is_dir())


def rect_from_item(item: dict[str, Any], source_size: tuple[int, int], strategy: str) -> Rect:
    if item.get("_detector_bbox_source") == "reviewed_prelabel":
        return clamp_rect(item.get("bbox_norm") or {})
    if strategy == "stored_crop":
        return clamp_rect(load_json(item.get("crop_rect")) or load_json(item.get("normalized_rect")))
    if strategy == "stored_text":
        return clamp_rect(load_json(item.get("normalized_rect")) or load_json(item.get("crop_rect")))
    if strategy == "v3_safe":
        source = str(item.get("source") or "").strip()
        if source in {"server_rect_crop", "server_rect_expanded", "server_expanded"}:
            return clamp_rect(load_json(item.get("crop_rect")) or load_json(item.get("normalized_rect")))
        return rect_only_v3_server_crop(item, source_size)
    raise ValueError(f"unsupported label strategy: {strategy}")


def rect_to_signature_item(label: Label) -> dict[str, Any]:
    return {
        "question_key": str(label.item.get("question_key") or ""),
        "question_index": int(label.item.get("question_index") or 0),
        "rect": label.rect,
    }


def source_split_map(source_keys: list[str], train_ratio: float, val_ratio: float) -> dict[str, str]:
    unique = sorted(set(source_keys), key=lambda key: stable_id(key, 40))
    count = len(unique)
    if count == 0:
        return {}
    if count == 1:
        return {unique[0]: "train"}
    test_count = max(1 if count >= 3 else 0, int(round(count * max(0.0, 1.0 - train_ratio - val_ratio))))
    val_count = max(1 if count >= 2 else 0, int(round(count * val_ratio)))
    if val_count + test_count >= count:
        val_count = 1 if count >= 2 else 0
        test_count = 1 if count >= 3 else 0
    train_count = max(1, count - val_count - test_count)
    split: dict[str, str] = {}
    for index, key in enumerate(unique):
        if index < train_count:
            split[key] = "train"
        elif index < train_count + val_count:
            split[key] = "val"
        else:
            split[key] = "test"
    return split


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


def duplicate_split_group_aliases(rows: list[SourceRow], args: argparse.Namespace) -> tuple[dict[str, str], dict[str, Any]]:
    groups = {str(row.split_key or "") for row in rows if str(row.split_key or "")}
    parent = {key: key for key in groups}

    def find(key: str) -> str:
        root = parent.setdefault(key, key)
        while parent[root] != root:
            root = parent[root]
        while parent[key] != key:
            next_key = parent[key]
            parent[key] = root
            key = next_key
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        canonical = min(left_root, right_root)
        other = right_root if canonical == left_root else left_root
        parent[other] = canonical

    threshold = max(0, int(args.near_duplicate_threshold))
    max_recorded = max(0, int(args.max_near_duplicate_pairs))
    if not getattr(args, "cluster_near_duplicates", True):
        return (
            {group: group for group in groups},
            {
                "enabled": False,
                "original_group_count": len(groups),
                "clustered_group_count": len(groups),
                "merged_group_count": 0,
                "near_duplicate_threshold": threshold,
                "exact_duplicate_group_links": [],
                "near_duplicate_group_links": [],
                "merged_clusters": [],
            },
        )

    records: list[dict[str, Any]] = []
    seen_records: set[tuple[str, str]] = set()
    for row in rows:
        group = str(row.split_key or "")
        if not group or not is_source_image(row.source_path):
            continue
        try:
            image_identity = str(row.source_path.resolve())
        except OSError:
            image_identity = str(row.source_path)
        record_key = (group, image_identity)
        if record_key in seen_records:
            continue
        seen_records.add(record_key)
        records.append(
            {
                "group": group,
                "image": str(row.source_path),
                "source_filename": row.source_filename,
                "sha1": file_sha1(row.source_path),
                "ahash": image_average_hash(row.source_path),
            }
        )

    exact_links: list[dict[str, Any]] = []
    by_sha1: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_sha1[str(record["sha1"])].append(record)
    for digest, matches in by_sha1.items():
        groups_for_digest = sorted({str(record["group"]) for record in matches})
        if len(groups_for_digest) <= 1:
            continue
        for group in groups_for_digest[1:]:
            union(groups_for_digest[0], group)
        exact_links.append(
            {
                "sha1": digest,
                "groups": groups_for_digest[:10],
                "images": [str(record["source_filename"] or record["image"]) for record in matches[:10]],
            }
        )

    near_links: list[dict[str, Any]] = []
    for left_index, left in enumerate(records):
        for right in records[left_index + 1:]:
            if left["group"] == right["group"]:
                continue
            distance = hex_hamming(str(left.get("ahash") or ""), str(right.get("ahash") or ""))
            if distance <= threshold:
                union(str(left["group"]), str(right["group"]))
                if not max_recorded or len(near_links) < max_recorded:
                    near_links.append(
                        {
                            "distance": distance,
                            "left": str(left["source_filename"] or left["image"]),
                            "left_group": str(left["group"]),
                            "right": str(right["source_filename"] or right["image"]),
                            "right_group": str(right["group"]),
                        }
                    )

    clusters: dict[str, list[str]] = defaultdict(list)
    for group in sorted(groups):
        clusters[find(group)].append(group)
    aliases: dict[str, str] = {}
    merged_clusters: list[dict[str, Any]] = []
    for canonical, members in sorted(clusters.items()):
        for member in members:
            aliases[member] = canonical
        if len(members) > 1:
            merged_clusters.append({"canonical": canonical, "groups": members})

    summary = {
        "enabled": True,
        "original_group_count": len(groups),
        "clustered_group_count": len(clusters),
        "merged_group_count": sum(1 for members in clusters.values() if len(members) > 1),
        "near_duplicate_threshold": threshold,
        "exact_duplicate_group_links": exact_links[:max_recorded] if max_recorded else exact_links,
        "near_duplicate_group_links": near_links,
        "merged_clusters": merged_clusters[:max_recorded] if max_recorded else merged_clusters,
    }
    return aliases, summary


def safe_export_filename(source: SourceRow) -> str:
    suffix = source.source_path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        suffix = ".jpg"
    stem = Path(source.source_filename).stem[:80]
    return f"{stable_id(source.source_key)}_{stem}{suffix}"


def quality_flags_for_rect(rect: Rect, source_size: tuple[int, int], item: dict[str, Any]) -> list[str]:
    width = max(1, int(rect["width"] * source_size[0]))
    height = max(1, int(rect["height"] * source_size[1]))
    human_reviewed = bool(item.get("_human_reviewed"))
    flags: list[str] = []
    if human_reviewed:
        flags.append("human_reviewed")
    if is_bad(width, height, rect):
        flags.append("shape_risk_human_reviewed" if human_reviewed else "shape_risk")
    confidence = item.get("confidence")
    try:
        if confidence is not None and float(confidence) < 0.35:
            flags.append("low_ocr_confidence")
    except (TypeError, ValueError):
        flags.append("unknown_confidence")
    if rect["width"] * rect["height"] > 0.72:
        flags.append("near_full_page_human_reviewed" if human_reviewed else "near_full_page")
    question_key = str(item.get("question_key") or "").strip()
    key_strength = str(item.get("question_key_strength") or item.get("questionKeyStrength") or "").strip().lower()
    if not human_reviewed and (question_key.lower().startswith("layout:") or key_strength.startswith("weak")):
        flags.append("weak_question_key")
    if not human_reviewed and not question_key:
        flags.append("missing_question_key")
    return flags


def review_record(row: SourceRow, rect: Rect, source_size: tuple[int, int], flags: list[str], reason: str) -> dict[str, Any]:
    x = int(round(rect["x"] * source_size[0]))
    y = int(round(rect["y"] * source_size[1]))
    w = int(round(rect["width"] * source_size[0]))
    h = int(round(rect["height"] * source_size[1]))
    return {
        "reason": reason,
        "quality_flags": flags,
        "source_filename": row.source_filename,
        "source_path": str(row.source_path),
        "origin": row.origin,
        "question_key": row.item.get("question_key") or "",
        "question_index": int(row.item.get("question_index") or 0),
        "confidence": row.item.get("confidence"),
        "bbox_norm": rect,
        "bbox_px": {"x": x, "y": y, "width": w, "height": h},
    }


def build_labels(
    rows: list[SourceRow],
    strategy: str,
    min_confidence: float,
    include_review_labels: bool,
    split_by_group: dict[str, str],
) -> tuple[list[Label], dict[str, int], list[dict[str, Any]]]:
    skipped: dict[str, int] = defaultdict(int)
    review: list[dict[str, Any]] = []
    labels: list[Label] = []
    image_size_cache: dict[Path, tuple[int, int]] = {}
    seen_by_source: dict[str, list[Label]] = defaultdict(list)

    for row in rows:
        if not is_source_image(row.source_path):
            skipped["missing_source_image"] += 1
            continue
        try:
            confidence = row.item.get("confidence")
            if confidence is not None and float(confidence) < min_confidence:
                skipped["below_min_confidence"] += 1
                continue
        except (TypeError, ValueError):
            pass
        if row.source_path not in image_size_cache:
            try:
                with Image.open(row.source_path) as image:
                    image_size_cache[row.source_path] = ImageOps.exif_transpose(image).size
            except Exception:
                skipped["unreadable_source_image"] += 1
                continue
        source_size = image_size_cache[row.source_path]
        rect = rect_from_item(row.item, source_size, strategy)
        if rect["width"] <= 0 or rect["height"] <= 0:
            skipped["empty_rect"] += 1
            continue
        flags = quality_flags_for_rect(rect, source_size, row.item)
        if "shape_risk" in flags:
            skipped["shape_risk"] += 1
            review.append(review_record(row, rect, source_size, flags, "shape_risk"))
            continue
        review_flags = {"low_ocr_confidence", "near_full_page", "missing_question_key", "weak_question_key"}
        if set(flags) & review_flags and not include_review_labels:
            skipped["review_risk"] += 1
            review.append(review_record(row, rect, source_size, flags, "review_risk"))
            continue
        label = Label(
            source_key=row.source_key,
            split_key=row.split_key,
            source_path=row.source_path,
            source_filename=row.source_filename,
            export_filename=safe_export_filename(row),
            split=split_by_group.get(row.split_key, "train"),
            image_width=source_size[0],
            image_height=source_size[1],
            rect=rect,
            item=row.item,
            origin=row.origin,
            quality_flags=flags,
        )
        existing = seen_by_source[label.source_key]
        if any(row_signature_similar(rect_to_signature_item(label), rect_to_signature_item(other)) for other in existing):
            skipped["duplicate_label"] += 1
            review.append(review_record(row, rect, source_size, flags, "duplicate_label"))
            continue
        existing.append(label)
        labels.append(label)

    return labels, dict(sorted(skipped.items())), review


def build_negative_images(
    rows: list[SourceRow],
    split_by_group: dict[str, str],
    positive_source_keys: set[str],
    positive_source_paths: set[Path],
) -> tuple[list[DatasetImage], dict[str, int]]:
    skipped: dict[str, int] = defaultdict(int)
    images: list[DatasetImage] = []
    seen: set[str] = set()
    for row in rows:
        if row.source_key in positive_source_keys:
            skipped["negative_source_has_positive_label"] += 1
            continue
        try:
            resolved_path = row.source_path.resolve()
        except OSError:
            resolved_path = row.source_path
        if resolved_path in positive_source_paths:
            skipped["negative_path_has_positive_label"] += 1
            continue
        if row.source_key in seen:
            skipped["duplicate_negative_source"] += 1
            continue
        if not is_source_image(row.source_path):
            skipped["missing_negative_image"] += 1
            continue
        try:
            with Image.open(row.source_path) as image:
                image_size = ImageOps.exif_transpose(image).size
        except Exception:
            skipped["unreadable_negative_image"] += 1
            continue
        negative_kind = str(row.item.get("negative_kind") or "empty_page").strip() or "empty_page"
        quality_flags = ["negative_sample", f"negative:{negative_kind}"]
        images.append(
            DatasetImage(
                source_key=row.source_key,
                split_key=row.split_key,
                source_path=row.source_path,
                source_filename=row.source_filename,
                export_filename=safe_export_filename(row),
                split=split_by_group.get(row.split_key, "train"),
                image_width=image_size[0],
                image_height=image_size[1],
                origin=row.origin,
                label_count=0,
                is_negative=True,
                negative_kind=negative_kind,
                quality_flags=quality_flags,
                item=row.item,
            )
        )
        seen.add(row.source_key)
    return images, dict(sorted(skipped.items()))


def build_dataset_images(labels: list[Label], negative_images: list[DatasetImage]) -> list[DatasetImage]:
    grouped: dict[str, list[Label]] = defaultdict(list)
    for label in labels:
        grouped[label.source_key].append(label)
    images: list[DatasetImage] = []
    for source_key, image_labels in grouped.items():
        first = image_labels[0]
        flags = sorted({flag for label in image_labels for flag in label.quality_flags})
        images.append(
            DatasetImage(
                source_key=source_key,
                split_key=first.split_key,
                source_path=first.source_path,
                source_filename=first.source_filename,
                export_filename=first.export_filename,
                split=first.split,
                image_width=first.image_width,
                image_height=first.image_height,
                origin=first.origin,
                label_count=len(image_labels),
                is_negative=False,
                negative_kind="",
                quality_flags=flags,
                item=first.item,
            )
        )
    images.extend(negative_images)
    return sorted(images, key=lambda image: (image.split, stable_id(image.source_key, 40), image.source_filename))


def copy_images(images: list[DatasetImage], out_dir: Path) -> dict[str, dict[str, Any]]:
    image_records: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    for image in images:
        if image.source_key in image_records:
            continue
        target = out_dir / "images" / image.split / image.export_filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image.source_path, target)
        rel = target.relative_to(out_dir).as_posix()
        image_records[image.source_key] = {
            "id": len(image_records) + 1,
            "file_name": rel,
            "width": image.image_width,
            "height": image.image_height,
            "split": image.split,
            "split_key": image.split_key,
            "source_key": image.source_key,
            "source_filename": image.source_filename,
            "origin": image.origin,
            "label_count": image.label_count,
            "is_negative": image.is_negative,
            "negative_kind": image.negative_kind,
            "quality_flags": image.quality_flags,
        }
    return image_records


def write_yolo(labels: list[Label], out_dir: Path, image_records: dict[str, dict[str, Any]]) -> None:
    labels_by_image: dict[str, list[Label]] = defaultdict(list)
    for label in labels:
        labels_by_image[label.source_key].append(label)
    for source_key, image_record in sorted(image_records.items(), key=lambda item: int(item[1]["id"])):
        image_labels = labels_by_image.get(source_key, [])
        image_path = Path(image_record["file_name"])
        target = out_dir / "labels" / image_record["split"] / f"{image_path.stem}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for label in image_labels:
            x = label.rect["x"] + label.rect["width"] / 2
            y = label.rect["y"] + label.rect["height"] / 2
            lines.append(f"0 {x:.6f} {y:.6f} {label.rect['width']:.6f} {label.rect['height']:.6f}")
        target.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    json_dump(out_dir / "yolo_dataset.yaml", {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: CATEGORY_NAME},
    })


def coco_for_split(labels: list[Label], image_records: dict[str, dict[str, Any]], split: str | None) -> dict[str, Any]:
    selected = [label for label in labels if split is None or label.split == split]
    images = [
        record
        for record in sorted(image_records.values(), key=lambda item: int(item["id"]))
        if split is None or record["split"] == split
    ]
    annotations: list[dict[str, Any]] = []
    for label in selected:
        x, y, w, h = label.bbox_px
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": image_records[label.source_key]["id"],
                "category_id": CATEGORY_ID,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "question_key": label.item.get("question_key") or "",
                "question_index": int(label.item.get("question_index") or 0),
                "confidence": label.item.get("confidence"),
                "quality_flags": label.quality_flags,
                "split_key": label.split_key,
                "source_key": label.source_key,
                "origin": label.origin,
            }
        )
    return {
        "info": {
            "description": "PXJ weak question-region detector dataset",
            "version": "v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME}],
    }


def write_coco(labels: list[Label], out_dir: Path, image_records: dict[str, dict[str, Any]]) -> None:
    for split in ("train", "val", "test"):
        json_dump(out_dir / "annotations" / f"coco_{split}.json", coco_for_split(labels, image_records, split))
    json_dump(out_dir / "annotations" / "coco_all.json", coco_for_split(labels, image_records, None))


def write_createml(labels: list[Label], out_dir: Path, image_records: dict[str, dict[str, Any]]) -> None:
    grouped: dict[str, list[Label]] = defaultdict(list)
    for label in labels:
        grouped[label.source_key].append(label)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_key, image_record in sorted(image_records.items(), key=lambda item: int(item[1]["id"])):
        image_labels = grouped.get(source_key, [])
        annotations = []
        for label in image_labels:
            x, y, w, h = label.bbox_px
            annotations.append(
                {
                    "label": CATEGORY_NAME,
                    "coordinates": {
                        "x": x + w / 2,
                        "y": y + h / 2,
                        "width": w,
                        "height": h,
                    },
                }
            )
        by_split[image_record["split"]].append({"image": image_record["file_name"], "annotations": annotations})
    for split in ("train", "val", "test"):
        json_dump(out_dir / "annotations" / f"createml_{split}.json", by_split.get(split, []))
    json_dump(out_dir / "annotations" / "createml_all.json", [item for split in ("train", "val", "test") for item in by_split.get(split, [])])


def write_manifest_jsonl(labels: list[Label], out_dir: Path, image_records: dict[str, dict[str, Any]]) -> None:
    target = out_dir / "annotations" / "manifest.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for label in labels:
            image_record = image_records[label.source_key]
            x, y, w, h = label.bbox_px
            fh.write(
                json.dumps(
                    {
                        "image": image_record["file_name"],
                        "split": label.split,
                        "label": CATEGORY_NAME,
                        "bbox_norm": label.rect,
                        "bbox_px": {"x": x, "y": y, "width": w, "height": h},
                        "question_key": label.item.get("question_key") or "",
                        "question_index": int(label.item.get("question_index") or 0),
                        "confidence": label.item.get("confidence"),
                        "quality_flags": label.quality_flags,
                        "split_key": label.split_key,
                        "source_key": label.source_key,
                        "source_filename": label.source_filename,
                        "origin": label.origin,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def write_image_manifest_jsonl(image_records: dict[str, dict[str, Any]], out_dir: Path) -> None:
    target = out_dir / "annotations" / "image_manifest.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for record in sorted(image_records.values(), key=lambda item: int(item["id"])):
            fh.write(
                json.dumps(
                    {
                        "image": record["file_name"],
                        "split": record["split"],
                        "split_key": record.get("split_key") or "",
                        "source_key": record.get("source_key") or "",
                        "source_filename": record.get("source_filename") or "",
                        "origin": record.get("origin") or "",
                        "width": int(record.get("width") or 0),
                        "height": int(record.get("height") or 0),
                        "label_count": int(record.get("label_count") or 0),
                        "is_negative": bool(record.get("is_negative")),
                        "negative_kind": record.get("negative_kind") or "",
                        "quality_flags": record.get("quality_flags") or [],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def write_review_jsonl(review: list[dict[str, Any]], out_dir: Path) -> None:
    target = out_dir / "annotations" / "review.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for item in review:
            fh.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_contact_sheet(labels: list[Label], out_dir: Path, image_records: dict[str, dict[str, Any]], limit: int = 40) -> None:
    grouped: dict[str, list[Label]] = defaultdict(list)
    for label in labels:
        grouped[label.source_key].append(label)
    tiles: list[Image.Image] = []
    for source_key in sorted(grouped, key=lambda key: image_records[key]["id"])[:limit]:
        image_record = image_records[source_key]
        image_path = out_dir / image_record["file_name"]
        with Image.open(image_path) as image:
            img = ImageOps.exif_transpose(image).convert("RGB")
        scale = min(220 / img.width, 170 / img.height)
        thumb = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(thumb)
        for label in grouped[source_key]:
            x, y, w, h = label.bbox_px
            box = [int(x * scale), int(y * scale), int((x + w) * scale), int((y + h) * scale)]
            draw.rectangle(box, outline=(255, 50, 50), width=2)
        tile = Image.new("RGB", (240, 220), "white")
        tile.paste(thumb, ((240 - thumb.width) // 2, 8))
        draw_tile = ImageDraw.Draw(tile)
        draw_tile.text((8, 184), f"{image_record['split']} boxes={len(grouped[source_key])}", fill=(0, 0, 0))
        draw_tile.text((8, 202), Path(image_record["file_name"]).name[:34], fill=(0, 0, 0))
        tiles.append(tile)
    if not tiles:
        return
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 240, rows * 220), (245, 245, 245))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % cols) * 240, (index // cols) * 220))
    sheet.save(out_dir / "preview_contact_sheet.jpg", quality=88)


def build_dataset_audit(
    labels: list[Label],
    skipped: dict[str, int],
    review: list[dict[str, Any]],
    out_dir: Path,
    image_records: dict[str, dict[str, Any]],
    source_row_count: int,
    split_group_aliases: dict[str, str],
    split_grouping: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_images = sum(1 for record in image_records.values() if record["split"] == split)
        split_positive_images = sum(1 for record in image_records.values() if record["split"] == split and int(record.get("label_count") or 0) > 0)
        split_negative_images = sum(1 for record in image_records.values() if record["split"] == split and bool(record.get("is_negative")))
        split_annotations = sum(1 for label in labels if label.split == split)
        raw_split_groups = {str(record.get("split_key") or "") for record in image_records.values() if record["split"] == split}
        split_groups = len({split_group_aliases.get(group, group) for group in raw_split_groups if group})
        split_counts[split] = {
            "images": split_images,
            "positive_images": split_positive_images,
            "negative_images": split_negative_images,
            "annotations": split_annotations,
            "groups": split_groups,
            "raw_groups": len({group for group in raw_split_groups if group}),
        }

    group_splits: dict[str, set[str]] = defaultdict(set)
    for record in image_records.values():
        group_splits[str(record.get("split_key") or "")].add(record["split"])
    group_overlaps = {
        key: sorted(splits)
        for key, splits in sorted(group_splits.items())
        if key and len(splits) > 1
    }

    image_hash_records: list[dict[str, Any]] = []
    for record in image_records.values():
        image_path = out_dir / record["file_name"]
        if not image_path.is_file():
            continue
        image_hash_records.append(
            {
                "image": record["file_name"],
                "split": record["split"],
                "split_key": record.get("split_key") or "",
                "source_filename": record.get("source_filename") or "",
                "sha1": file_sha1(image_path),
                "ahash": image_average_hash(image_path),
            }
        )

    exact_by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in image_hash_records:
        exact_by_hash[record["sha1"]].append(record)
    exact_cross_split: list[dict[str, Any]] = []
    for digest, records in exact_by_hash.items():
        splits = sorted({record["split"] for record in records})
        if len(splits) > 1:
            exact_cross_split.append(
                {
                    "sha1": digest,
                    "splits": splits,
                    "images": [record["image"] for record in records[:10]],
                }
            )

    near_pairs: list[dict[str, Any]] = []
    threshold = max(0, int(args.near_duplicate_threshold))
    max_pairs = max(0, int(args.max_near_duplicate_pairs))
    for left_index, left in enumerate(image_hash_records):
        for right in image_hash_records[left_index + 1:]:
            if left["split"] == right["split"]:
                continue
            distance = hex_hamming(str(left.get("ahash") or ""), str(right.get("ahash") or ""))
            if distance <= threshold:
                near_pairs.append(
                    {
                        "distance": distance,
                        "left": left["image"],
                        "left_split": left["split"],
                        "right": right["image"],
                        "right_split": right["split"],
                    }
                )
                if max_pairs and len(near_pairs) >= max_pairs:
                    break
        if max_pairs and len(near_pairs) >= max_pairs:
            break

    warnings: list[dict[str, Any]] = []

    def warn(code: str, detail: str, severity: str = "warning") -> None:
        warnings.append({"code": code, "severity": severity, "detail": detail})

    source_count = len(image_records)
    annotation_count = len(labels)
    positive_image_count = sum(1 for record in image_records.values() if int(record.get("label_count") or 0) > 0)
    negative_image_count = sum(1 for record in image_records.values() if bool(record.get("is_negative")))
    empty_page_image_count = sum(1 for record in image_records.values() if record.get("negative_kind") == "empty_page")
    raw_split_group_count = len(group_splits)
    split_group_count = len(
        {
            split_group_aliases.get(group, group)
            for group in group_splits
            if group
        }
    )
    if group_overlaps:
        warn("split_group_overlap", f"{len(group_overlaps)} split groups appear in more than one split.", "error")
    if exact_cross_split:
        warn("exact_image_leakage", f"{len(exact_cross_split)} exact image hashes appear across splits.", "error")
    if near_pairs:
        warn(
            "near_duplicate_cross_split",
            f"{len(near_pairs)} cross-split image pairs have aHash distance <= {threshold}.",
            args.near_duplicate_severity,
        )
    if source_count < int(args.min_production_images):
        warn("too_few_images", f"{source_count} source images; target >= {args.min_production_images} for model selection.")
    if annotation_count < int(args.min_production_annotations):
        warn("too_few_annotations", f"{annotation_count} annotations; target >= {args.min_production_annotations} for model selection.")
    if negative_image_count < int(args.min_production_negative_images):
        warn("too_few_negative_images", f"{negative_image_count} negative images; target >= {args.min_production_negative_images} for false-positive control.")
    if split_group_count < int(args.min_production_groups):
        warn("too_few_split_groups", f"{split_group_count} split groups; target >= {args.min_production_groups} for held-out evaluation.")
    for split, counts in split_counts.items():
        if counts["images"] == 0 or counts["annotations"] == 0:
            warn("empty_split", f"{split} has {counts['images']} images and {counts['annotations']} annotations.", "error")
    review_ratio = len(review) / max(1, annotation_count + len(review))
    review_flags = {"low_ocr_confidence", "near_full_page", "missing_question_key", "weak_question_key"}
    risky_label_count = sum(1 for label in labels if set(label.quality_flags) & review_flags)
    risky_label_ratio = risky_label_count / max(1, annotation_count)
    if review_ratio > float(args.max_review_ratio):
        warn("high_review_ratio", f"review ratio is {review_ratio:.3f}; target <= {args.max_review_ratio}.")
    if risky_label_ratio > float(args.max_risky_label_ratio):
        warn("high_risky_label_ratio", f"risky accepted label ratio is {risky_label_ratio:.3f}; target <= {args.max_risky_label_ratio}.", "error")

    has_error = any(item["severity"] == "error" for item in warnings)
    meets_scale = (
        source_count >= int(args.min_production_images)
        and annotation_count >= int(args.min_production_annotations)
        and negative_image_count >= int(args.min_production_negative_images)
        and split_group_count >= int(args.min_production_groups)
    )

    by_origin: dict[str, int] = defaultdict(int)
    by_negative_kind: dict[str, int] = defaultdict(int)
    by_quality_flag: dict[str, int] = defaultdict(int)
    for record in image_records.values():
        by_origin[str(record.get("origin") or "unknown")] += 1
        if record.get("is_negative"):
            by_negative_kind[str(record.get("negative_kind") or "unknown")] += 1
        flags = record.get("quality_flags") if isinstance(record.get("quality_flags"), list) else []
        for flag in flags:
            by_quality_flag[str(flag)] += 1

    return {
        "readiness": {
            "pilot_eval_ready": annotation_count > 0 and not any(item["code"] == "empty_split" for item in warnings),
            "model_training_ready": (not has_error)
            and meets_scale
            and review_ratio <= float(args.max_review_ratio)
            and risky_label_ratio <= float(args.max_risky_label_ratio),
            "has_error": has_error,
        },
        "counts": {
            "source_images": source_count,
            "positive_images": positive_image_count,
            "negative_images": negative_image_count,
            "empty_page_images": empty_page_image_count,
            "annotations": annotation_count,
            "review": len(review),
            "skipped": skipped,
            "split_groups": split_group_count,
            "raw_split_groups": raw_split_group_count,
            "source_rows": source_row_count,
            "review_ratio": round(review_ratio, 4),
            "risky_label_count": risky_label_count,
            "risky_label_ratio": round(risky_label_ratio, 4),
        },
        "split_counts": split_counts,
        "strata_counts": {
            "by_origin": dict(sorted(by_origin.items())),
            "by_negative_kind": dict(sorted(by_negative_kind.items())),
            "by_quality_flag": dict(sorted(by_quality_flag.items())),
        },
        "leakage": {
            "split_group_overlaps": group_overlaps,
            "exact_cross_split_images": exact_cross_split,
            "near_duplicate_threshold": threshold,
            "near_duplicate_severity": args.near_duplicate_severity,
            "near_duplicate_cross_split_pairs": near_pairs,
            "split_grouping": split_grouping,
        },
        "minimums": {
            "production_images": int(args.min_production_images),
            "production_annotations": int(args.min_production_annotations),
            "production_negative_images": int(args.min_production_negative_images),
            "production_split_groups": int(args.min_production_groups),
            "max_review_ratio": float(args.max_review_ratio),
            "max_risky_label_ratio": float(args.max_risky_label_ratio),
        },
        "warnings": warnings,
    }


def write_metadata(
    labels: list[Label],
    skipped: dict[str, int],
    review: list[dict[str, Any]],
    out_dir: Path,
    image_records: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    audit: dict[str, Any],
    split_grouping: dict[str, Any],
) -> None:
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        split_images = sum(1 for record in image_records.values() if record["split"] == split)
        split_positive_images = sum(1 for record in image_records.values() if record["split"] == split and int(record.get("label_count") or 0) > 0)
        split_negative_images = sum(1 for record in image_records.values() if record["split"] == split and bool(record.get("is_negative")))
        split_annotations = sum(1 for label in labels if label.split == split)
        split_counts[split] = {
            "images": split_images,
            "positive_images": split_positive_images,
            "negative_images": split_negative_images,
            "annotations": split_annotations,
        }
    areas = [label.rect["width"] * label.rect["height"] for label in labels]
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_strategy": args.strategy,
        "split_scope": args.split_scope,
        "min_confidence": args.min_confidence,
        "category": CATEGORY_NAME,
        "source_count": len(image_records),
        "positive_source_count": sum(1 for record in image_records.values() if int(record.get("label_count") or 0) > 0),
        "negative_image_count": sum(1 for record in image_records.values() if bool(record.get("is_negative"))),
        "annotation_count": len(labels),
        "review_count": len(review),
        "include_review_labels": bool(args.include_review_labels),
        "split_counts": split_counts,
        "skipped": skipped,
        "area": {
            "min": round(min(areas), 6) if areas else 0,
            "max": round(max(areas), 6) if areas else 0,
            "mean": round(sum(areas) / len(areas), 6) if areas else 0,
        },
        "audit": audit,
        "split_grouping": split_grouping,
        "inputs": {
            "diagnostics": [str(path) for path in args.diagnostics],
            "sqlite": [str(path) for path in args.sqlite],
            "reviewed_prelabels": [str(path) for path in args.reviewed_prelabels],
            "approved_label_statuses": args.approved_label_statuses,
            "empty_page_manifests": [str(path) for path in args.empty_page_manifest],
            "negative_image_dirs": [str(path) for path in args.negative_images_dir],
            "include_sqlite_empty_pages": bool(args.include_sqlite_empty_pages),
            "sqlite_empty_page_mode": args.sqlite_empty_page_mode,
            "data_dir": str(args.data_dir) if args.data_dir else "",
        },
        "outputs": {
            "coco": "annotations/coco_*.json",
            "yolo": "images/{split}, labels/{split}, yolo_dataset.yaml",
            "createml": "annotations/createml_*.json",
            "jsonl": "annotations/manifest.jsonl",
            "image_manifest": "annotations/image_manifest.jsonl",
            "review": "annotations/review.jsonl",
            "preview": "preview_contact_sheet.jpg",
            "audit": "audit.json",
        },
        "notes": [
            "These are weak labels exported from production crop/rect evidence.",
            "Use manual review before final model training, especially for low-confidence and near-full-page labels.",
        ],
    }
    json_dump(out_dir / "metadata.json", metadata)


def export_dataset(args: argparse.Namespace) -> dict[str, Any]:
    if args.all_account_dbs:
        if args.data_dir is None:
            raise ValueError("--data-dir is required with --all-account-dbs")
        discovered = discover_sqlite_dbs(args.data_dir)
        existing = {path.resolve() for path in args.sqlite}
        args.sqlite.extend(path for path in discovered if path.resolve() not in existing)

    for diagnostics_root in args.diagnostics_root:
        discovered_roots = discover_diagnostics_roots(diagnostics_root)
        existing = {path.resolve() for path in args.diagnostics}
        args.diagnostics.extend(path for path in discovered_roots if path.resolve() not in existing)

    rows: list[SourceRow] = []
    for diagnostics_root in args.diagnostics:
        rows.extend(load_diagnostics_rows(diagnostics_root, args.split_scope))
    for sqlite_path in args.sqlite:
        if args.data_dir is None:
            raise ValueError("--data-dir is required when using --sqlite")
        rows.extend(load_sqlite_rows(sqlite_path, args.data_dir, args.split_scope))
    approved_statuses = approved_statuses_from_csv(args.approved_label_statuses)
    for reviewed_path in args.reviewed_prelabels:
        rows.extend(load_reviewed_prelabel_rows(reviewed_path, approved_statuses, args.split_scope))

    negative_rows: list[SourceRow] = []
    for manifest_path in args.empty_page_manifest:
        negative_rows.extend(load_empty_page_manifest_rows(manifest_path, args.split_scope))
    for negative_dir in args.negative_images_dir:
        negative_rows.extend(load_negative_image_dir_rows(negative_dir, args.split_scope))
    if args.include_sqlite_empty_pages:
        if args.data_dir is None:
            raise ValueError("--data-dir is required with --include-sqlite-empty-pages")
        for sqlite_path in args.sqlite:
            negative_rows.extend(load_sqlite_empty_page_rows(sqlite_path, args.data_dir, args.split_scope, args.sqlite_empty_page_mode))

    split_group_aliases, split_grouping = duplicate_split_group_aliases([*rows, *negative_rows], args)
    split_by_effective_group = source_split_map(
        [split_group_aliases.get(row.split_key, row.split_key) for row in rows],
        train_ratio=0.82,
        val_ratio=0.10,
    )
    negative_split_by_group = source_split_map(
        [
            split_group_aliases.get(row.split_key, row.split_key)
            for row in negative_rows
            if split_group_aliases.get(row.split_key, row.split_key) not in split_by_effective_group
        ],
        train_ratio=0.82,
        val_ratio=0.10,
    )
    split_by_effective_group.update(negative_split_by_group)
    split_by_group = {
        row.split_key: split_by_effective_group.get(split_group_aliases.get(row.split_key, row.split_key), "train")
        for row in [*rows, *negative_rows]
    }
    labels, skipped, review = build_labels(
        rows,
        strategy=args.strategy,
        min_confidence=args.min_confidence,
        include_review_labels=args.include_review_labels,
        split_by_group=split_by_group,
    )
    positive_source_keys = {label.source_key for label in labels}
    positive_source_paths: set[Path] = set()
    for label in labels:
        try:
            positive_source_paths.add(label.source_path.resolve())
        except OSError:
            positive_source_paths.add(label.source_path)
    negative_images, negative_skipped = build_negative_images(
        negative_rows,
        split_by_group,
        positive_source_keys=positive_source_keys,
        positive_source_paths=positive_source_paths,
    )
    for key, value in negative_skipped.items():
        skipped[key] = skipped.get(key, 0) + value
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    dataset_images = build_dataset_images(labels, negative_images)
    image_records = copy_images(dataset_images, args.out)
    write_yolo(labels, args.out, image_records)
    write_coco(labels, args.out, image_records)
    write_createml(labels, args.out, image_records)
    write_manifest_jsonl(labels, args.out, image_records)
    write_image_manifest_jsonl(image_records, args.out)
    write_review_jsonl(review, args.out)
    build_contact_sheet(labels, args.out, image_records)
    audit = build_dataset_audit(
        labels,
        skipped,
        review,
        args.out,
        image_records,
        len(rows),
        split_group_aliases,
        split_grouping,
        args,
    )
    json_dump(args.out / "audit.json", audit)
    write_metadata(labels, skipped, review, args.out, image_records, args, audit, split_grouping)
    return {
        "out": str(args.out),
        "source_rows": len(rows),
        "negative_rows": len(negative_rows),
        "images": len(image_records),
        "negative_images": len(negative_images),
        "annotations": len(labels),
        "review": len(review),
        "skipped": skipped,
        "readiness": audit.get("readiness", {}),
        "warnings": audit.get("warnings", []),
        "inputs": {
            "diagnostics": [str(path) for path in args.diagnostics],
            "sqlite": [str(path) for path in args.sqlite],
            "reviewed_prelabels": [str(path) for path in args.reviewed_prelabels],
            "empty_page_manifests": [str(path) for path in args.empty_page_manifest],
            "negative_image_dirs": [str(path) for path in args.negative_images_dir],
            "include_sqlite_empty_pages": bool(args.include_sqlite_empty_pages),
            "sqlite_empty_page_mode": args.sqlite_empty_page_mode,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export question detector training data.")
    parser.add_argument("--diagnostics", type=Path, action="append", default=[], help="Diagnostics data dir containing manifest.json and images/.")
    parser.add_argument("--diagnostics-root", type=Path, action="append", default=[], help="Discover diagnostics datasets below this root.")
    parser.add_argument("--sqlite", type=Path, action="append", default=[], help="Account SQLite DB containing session_question_crops.")
    parser.add_argument("--data-dir", type=Path, default=None, help="PXJ data dir for --sqlite image lookup.")
    parser.add_argument("--all-account-dbs", action="store_true", help="Discover account DBs under --data-dir/accounts/*/*.sqlite3.")
    parser.add_argument("--reviewed-prelabels", type=Path, action="append", default=[], help="Reviewed/approved prelabel export root or file from question_detector_prelabel.py, COCO, or Label Studio.")
    parser.add_argument("--approved-label-statuses", default="approved,accepted,corrected,verified", help="Comma-separated statuses that promote prelabels into detector training labels.")
    parser.add_argument("--empty-page-manifest", type=Path, action="append", default=[], help="JSON/JSONL manifest of explicit empty-page negative images.")
    parser.add_argument("--negative-images-dir", type=Path, action="append", default=[], help="Directory of explicit empty-page negative images; crop-like filenames are ignored.")
    parser.add_argument("--include-sqlite-empty-pages", action="store_true", help="Also export SQLite observation frames with no ready crop as empty-page negatives.")
    parser.add_argument("--sqlite-empty-page-mode", choices=["textless", "all_observations"], default="textless", help="Safety filter for --include-sqlite-empty-pages; textless avoids QA/extract/grade and OCR-bearing frames.")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-dataset-v1"))
    parser.add_argument("--strategy", choices=["v3_safe", "stored_crop", "stored_text"], default="v3_safe")
    parser.add_argument("--split-scope", choices=["batch", "session", "source"], default="session", help="Group boundary for train/val/test split; use session for model selection, batch/source only for small pipeline diagnostics.")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--include-review-labels", action="store_true", help="Include low-confidence/near-full-page labels in train outputs instead of review only.")
    parser.add_argument("--min-production-images", type=int, default=300, help="Minimum source images before audit marks the dataset model-training-ready.")
    parser.add_argument("--min-production-annotations", type=int, default=1000, help="Minimum labels before audit marks the dataset model-training-ready.")
    parser.add_argument("--min-production-negative-images", type=int, default=0, help="Minimum negative images before audit marks the dataset model-training-ready.")
    parser.add_argument("--min-production-groups", type=int, default=50, help="Minimum session/batch/book groups before audit marks held-out evaluation ready.")
    parser.add_argument("--max-review-ratio", type=float, default=0.15, help="Maximum review-risk label ratio for model-training-ready.")
    parser.add_argument("--max-risky-label-ratio", type=float, default=0.15, help="Maximum accepted risky label ratio for model-training-ready, including labels admitted by --include-review-labels.")
    parser.add_argument("--near-duplicate-threshold", type=int, default=4, help="aHash Hamming threshold for cross-split near-duplicate leakage checks.")
    parser.add_argument("--max-near-duplicate-pairs", type=int, default=200, help="Maximum cross-split near-duplicate pairs to record in audit.json; 0 means unlimited.")
    parser.add_argument("--near-duplicate-severity", choices=["warning", "error"], default="error", help="Severity for cross-split aHash near duplicates; default is error for model-selection datasets.")
    parser.add_argument("--no-cluster-near-duplicates", dest="cluster_near_duplicates", action="store_false", help="Disable default exact/aHash duplicate split-group clustering before assigning train/val/test.")
    parser.set_defaults(cluster_near_duplicates=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    if not args.diagnostics and not args.sqlite and not args.diagnostics_root and not args.all_account_dbs and not args.reviewed_prelabels and not args.empty_page_manifest and not args.negative_images_dir:
        parser.error("provide at least one positive or negative data input")
    summary = export_dataset(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
