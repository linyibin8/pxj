"""Prioritize question-detector review workbenches for the next label loop.

This tool does not approve labels or modify review queues. It reads the
consolidated review queue manifest, scores every pending draft box/image by
training value and review risk, and writes a compact plan for the next human
review pass.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HIGH_VALUE_SOURCES = {
    "dense_layout_prelabel": 2.1,
    "layout_prelabel": 1.2,
    "propagated_review_candidate": 1.6,
}

SOURCE_KIND_BONUS = {
    "active": 1.5,
    "fallback_delta": 1.8,
    "propagation": 1.4,
    "unknown": 0.6,
}

HIGH_VALUE_FLAGS = {
    "dense_column_layout": 2.0,
    "page:full_image_fallback": 1.6,
    "few_text_lines": 0.7,
    "short_block": 0.9,
}

RISK_FLAGS = {
    "quarantine": 3.0,
    "manual_page_review_required": 2.8,
    "table_like": 2.2,
    "keyboard_like": 2.2,
    "edge_dominant": 1.8,
    "fragmented_short_blocks": 1.8,
    "low_confidence": 1.2,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def bbox_area_ratio(box: dict[str, Any], width: int, height: int) -> float:
    bbox = box.get("bbox_px") if isinstance(box.get("bbox_px"), dict) else {}
    area = max(0, to_int(bbox.get("width"))) * max(0, to_int(bbox.get("height")))
    page_area = max(1, width * height)
    return area / page_area


def item_source_kind(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
    for key in ("backlog_source_kind", "source_kind", "candidate_kind"):
        value = str(metadata.get(key) or candidate.get(key) or "").strip()
        if value:
            if value.startswith("qa_"):
                return "active"
            return value
    image = str(item.get("image") or item.get("source_candidate") or "")
    if "fallback_delta" in image:
        return "fallback_delta"
    if "active" in image or "_qa_" in image:
        return "active"
    if "propagation" in image:
        return "propagation"
    return "unknown"


def split_group_key(item: dict[str, Any]) -> str:
    candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    for key in ("session_id", "batch_id", "image_id", "qa_event_id"):
        value = str(candidate.get(key) or "").strip()
        if value:
            return f"{key}:{value}"
    for key in ("split_key", "source_key", "source_filename", "backlog_source_review_id"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return f"{key}:{value}"
    return str(item.get("source_candidate") or item.get("image") or "unknown")


def box_score(box: dict[str, Any], item: dict[str, Any], box_count: int) -> dict[str, Any]:
    flags = [str(flag) for flag in (box.get("quality_flags") or [])]
    source = str(box.get("source") or "unknown")
    source_kind = item_source_kind(item)
    score = 1.0
    risk = 0.0
    reasons: list[str] = []

    score += HIGH_VALUE_SOURCES.get(source, 0.4)
    score += SOURCE_KIND_BONUS.get(source_kind, SOURCE_KIND_BONUS["unknown"])
    if source != "layout_prelabel":
        reasons.append(f"source:{source}")
    if source_kind != "unknown":
        reasons.append(f"kind:{source_kind}")

    for flag in flags:
        score += HIGH_VALUE_FLAGS.get(flag, 0.0)
        for risk_flag, value in RISK_FLAGS.items():
            if risk_flag in flag:
                risk += value
                reasons.append(f"risk:{flag}")
                break

    confidence = to_float(box.get("score"))
    if confidence >= 0.55:
        score += 1.1
        reasons.append("higher_score")
    elif confidence >= 0.35:
        score += 0.45
    elif confidence > 0:
        risk += 0.7

    area_ratio = bbox_area_ratio(box, to_int(item.get("width")), to_int(item.get("height")))
    if area_ratio < 0.0018:
        risk += 1.0
        reasons.append("tiny_box")
    elif area_ratio > 0.30:
        risk += 1.4
        reasons.append("large_box")
    else:
        score += 0.3

    if box_count >= 24:
        score += 1.2
        reasons.append("dense_page")
    elif box_count <= 3:
        score += 0.4

    if box.get("training_export_default") is False or box.get("requires_correction_for_training"):
        risk += 4.0
        reasons.append("requires_correction")

    net = score - risk * 0.55
    return {
        "value_score": round(score, 4),
        "risk_score": round(risk, 4),
        "net_score": round(net, 4),
        "reasons": reasons[:8],
    }


def quality_flags_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    for row in rows:
        box = row.get("box") if isinstance(row.get("box"), dict) else {}
        flags.extend(str(flag) for flag in (box.get("quality_flags") or []))
    return flags


def dominant_box_source(rows: list[dict[str, Any]]) -> str:
    counts = Counter(str((row.get("box") or {}).get("source") or "unknown") for row in rows)
    return counts.most_common(1)[0][0] if counts else "unknown"


def image_cohort(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "unknown"
    source_kind = str(rows[0].get("source_kind") or "unknown")
    annotation_status = str(rows[0].get("annotation_status") or "")
    sources = {str((row.get("box") or {}).get("source") or "") for row in rows}
    flags = quality_flags_from_rows(rows)
    box_count = to_int(rows[0].get("box_count_on_image"))

    if "fallback_delta_candidate" in sources or source_kind == "fallback_delta":
        return "fallback_delta"
    if source_kind == "active":
        if "manual_page_review_required" in annotation_status or "quarantine" in annotation_status:
            return "active_quarantine"
        return "active_capture"
    if "dense_layout_prelabel" in sources or any("dense_column_layout" in flag for flag in flags) or box_count >= 16:
        return "dense_layout"
    if any("full_image_fallback" in flag for flag in flags):
        return "full_image_fallback"
    if source_kind == "observation_image_unverified":
        return "observation_unverified"
    return "layout_general"


def load_candidates(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(manifest_path)
    candidates: list[dict[str, Any]] = []
    for workbench in manifest.get("workbenches") or []:
        review_data_path = Path(str(workbench.get("review_data") or ""))
        if not review_data_path.is_file():
            continue
        data = read_json(review_data_path)
        items = data.get("items") if isinstance(data.get("items"), list) else []
        for image_order, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            boxes = item.get("boxes") if isinstance(item.get("boxes"), list) else []
            for box_order, box in enumerate(boxes, start=1):
                if not isinstance(box, dict):
                    continue
                score = box_score(box, item, len(boxes))
                candidates.append(
                    {
                        "workbench_order": workbench.get("order"),
                        "workbench_path": workbench.get("path"),
                        "workbench_html": workbench.get("index_html"),
                        "review_data": str(review_data_path),
                        "image_order": image_order,
                        "box_order": box_order,
                        "review_id": item.get("review_id"),
                        "image": item.get("image"),
                        "source_candidate": item.get("source_candidate"),
                        "source_kind": item_source_kind(item),
                        "split_group_key": split_group_key(item),
                        "annotation_status": item.get("annotation_status"),
                        "image_width": item.get("width"),
                        "image_height": item.get("height"),
                        "box_count_on_image": len(boxes),
                        "box": box,
                        **score,
                    }
                )
    return manifest, candidates


def rank_images(candidates: list[dict[str, Any]], *, max_boxes_per_image: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        grouped[str(row.get("review_id") or row.get("image"))].append(row)

    images: list[dict[str, Any]] = []
    for rows in grouped.values():
        rows.sort(key=lambda row: (row["net_score"], row["value_score"]), reverse=True)
        selected = rows[:max_boxes_per_image]
        total_value = sum(float(row["value_score"]) for row in selected)
        total_risk = sum(float(row["risk_score"]) for row in selected)
        box_source_counts = Counter(str((row.get("box") or {}).get("source") or "unknown") for row in rows)
        images.append(
            {
                "priority_score": round(total_value - total_risk * 0.45 + min(len(rows), 24) * 0.15, 4),
                "review_id": rows[0].get("review_id"),
                "workbench_order": rows[0].get("workbench_order"),
                "workbench_path": rows[0].get("workbench_path"),
                "workbench_html": rows[0].get("workbench_html"),
                "image_order": rows[0].get("image_order"),
                "image": rows[0].get("image"),
                "source_kind": rows[0].get("source_kind"),
                "split_group_key": rows[0].get("split_group_key"),
                "annotation_status": rows[0].get("annotation_status"),
                "box_count_on_image": rows[0].get("box_count_on_image"),
                "selected_box_count": len(selected),
                "dominant_box_source": dominant_box_source(rows),
                "box_source_counts": dict(box_source_counts.most_common()),
                "cohort": image_cohort(rows),
                "top_reasons": list(dict.fromkeys(reason for row in selected for reason in row.get("reasons", [])))[:8],
                "boxes": selected,
                "review_boxes": rows,
            }
        )
    images.sort(key=lambda row: row["priority_score"], reverse=True)
    return images


def balanced_image_slice(
    images: list[dict[str, Any]],
    *,
    limit_images: int,
    max_images_per_cohort: int,
    max_images_per_source_kind: int,
) -> list[dict[str, Any]]:
    if limit_images <= 0:
        return []
    if max_images_per_cohort <= 0:
        max_images_per_cohort = max(4, math.ceil(limit_images * 0.35))
    if max_images_per_source_kind <= 0:
        max_images_per_source_kind = max(6, math.ceil(limit_images * 0.55))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    cohort_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    deferred: list[dict[str, Any]] = []

    for image in images:
        review_id = str(image.get("review_id") or image.get("image"))
        cohort = str(image.get("cohort") or "unknown")
        source_kind = str(image.get("source_kind") or "unknown")
        if cohort_counts[cohort] < max_images_per_cohort and kind_counts[source_kind] < max_images_per_source_kind:
            selected.append(image)
            selected_ids.add(review_id)
            cohort_counts[cohort] += 1
            kind_counts[source_kind] += 1
        else:
            deferred.append(image)
        if len(selected) >= limit_images:
            return selected

    for image in deferred:
        review_id = str(image.get("review_id") or image.get("image"))
        if review_id in selected_ids:
            continue
        selected.append(image)
        selected_ids.add(review_id)
        if len(selected) >= limit_images:
            break
    return selected


def select_ranked_images(
    candidates: list[dict[str, Any]],
    *,
    limit_images: int,
    max_boxes_per_image: int,
    strategy: str,
    max_images_per_cohort: int,
    max_images_per_source_kind: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranked_images = rank_images(candidates, max_boxes_per_image=max_boxes_per_image)
    if strategy == "score":
        return ranked_images[:limit_images], ranked_images
    return (
        balanced_image_slice(
            ranked_images,
            limit_images=limit_images,
            max_images_per_cohort=max_images_per_cohort,
            max_images_per_source_kind=max_images_per_source_kind,
        ),
        ranked_images,
    )


def resolve_review_image_path(item: dict[str, Any]) -> str:
    image_path = Path(str(item.get("image") or ""))
    if image_path.is_absolute():
        return str(image_path)
    root = Path(str(item.get("workbench_path") or "."))
    return str((root / image_path).resolve())


def copy_review_image(item: dict[str, Any], target: Path, index: int) -> tuple[str, int, int]:
    source = Path(resolve_review_image_path(item))
    if not source.is_file():
        raise SystemExit(f"review image not found: {source}")
    suffix = source.suffix.lower() or ".jpg"
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in source.stem)[:72]
    target_rel = f"images/{index:03d}_{safe_name}{suffix}"
    target_path = target / target_rel
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target_path)
    width = to_int(item.get("image_width"))
    height = to_int(item.get("image_height"))
    if width <= 0 or height <= 0:
        try:
            from PIL import Image, ImageOps

            with Image.open(target_path) as image:
                width, height = ImageOps.exif_transpose(image).size
        except Exception:
            width = to_int(item.get("image_width"))
            height = to_int(item.get("image_height"))
    return target_rel, width, height


def review_root_rows(images: list[dict[str, Any]], out: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(images, start=1):
        image_rel, width, height = copy_review_image(item, out, index)
        boxes: list[dict[str, Any]] = []
        review_candidates = item.get("review_boxes") or item.get("boxes") or []
        for box_index, candidate in enumerate(review_candidates, start=1):
            box = dict(candidate.get("box") or {})
            if not box:
                continue
            box.setdefault("source", candidate.get("source") or item.get("dominant_box_source") or "review_priority")
            box.setdefault("quality_flags", [])
            box["review_priority"] = index
            box["priority_box_order"] = box_index
            box["priority_score"] = item.get("priority_score")
            box["priority_cohort"] = item.get("cohort")
            box["priority_reasons"] = candidate.get("reasons") or item.get("top_reasons") or []
            boxes.append(box)
        rows.append(
            {
                "image": image_rel,
                "source_candidate": item.get("source_candidate") or item.get("image"),
                "width": width,
                "height": height,
                "boxes": boxes,
                "metadata": {
                    "review_priority": index,
                    "priority_score": item.get("priority_score"),
                    "cohort": item.get("cohort"),
                    "source_kind": item.get("source_kind"),
                    "dominant_box_source": item.get("dominant_box_source"),
                    "source_workbench": item.get("workbench_path"),
                    "source_workbench_html": item.get("workbench_html"),
                    "source_review_id": item.get("review_id"),
                    "split_group_key": item.get("split_group_key"),
                    "top_reasons": item.get("top_reasons") or [],
                },
                "annotation_status": item.get("annotation_status") or "draft_review_required",
                "review_id": item.get("review_id"),
                "review_priority": index,
            }
        )
    return rows


def write_review_root(out: Path, report: dict[str, Any]) -> dict[str, Any]:
    if out.exists():
        shutil.rmtree(out)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    rows = review_root_rows(report.get("recommended_images") or [], out)
    write_jsonl(out / "annotations" / "draft_boxes.jsonl", rows)
    summary = {
        "generated_at": report.get("generated_at"),
        "source_manifest": report.get("inputs", {}).get("manifest"),
        "source_priority_report": str(report.get("outputs", {}).get("review_priority_json") or ""),
        "images": len(rows),
        "boxes": sum(len(row.get("boxes") or []) for row in rows),
        "recommended_boxes": report.get("summary", {}).get("recommended_boxes"),
        "cohort_counts": dict(Counter(str(row.get("metadata", {}).get("cohort") or "unknown") for row in rows).most_common()),
        "outputs": {
            "draft_boxes": "annotations/draft_boxes.jsonl",
            "images": "images/",
        },
        "notes": [
            "Priority review root; draft boxes require human approval before training.",
            "Build a review UI with question_detector_review_workbench.py --prelabel-root this directory.",
        ],
    }
    write_json(out / "summary.json", summary)
    return summary


def benchmark_rows(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(images, start=1):
        rows.append(
            {
                "order": index,
                "review_id": item.get("review_id"),
                "cohort": item.get("cohort"),
                "source_kind": item.get("source_kind"),
                "dominant_box_source": item.get("dominant_box_source"),
                "box_count_on_image": item.get("box_count_on_image"),
                "selected_box_count": item.get("selected_box_count"),
                "image": item.get("image"),
                "image_path": resolve_review_image_path(item),
                "workbench_html": item.get("workbench_html"),
                "task": "Segment this worksheet image into individual printed-question regions; return question_count and normalized bounding boxes only.",
                "reference_status": "pending_human_review",
            }
        )
    return rows


def make_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Question Detector Review Priority",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Source manifest: `{report['inputs']['manifest']}`",
        f"- Total boxes scored: {summary['total_boxes']}",
        f"- Images ranked: {summary['total_images']}",
        f"- Recommended first pass: {summary['recommended_images']} images / {summary['recommended_boxes']} boxes",
        f"- Selection strategy: `{report['inputs']['strategy']}`",
        "",
        "## Why This Order",
        "",
        "- Prioritizes dense pages, fallback-delta evidence, active-learning candidates, and boxes with useful layout flags.",
        "- In balanced mode, caps each cohort/source kind so the first pass covers different failure shapes instead of only the highest-scoring dense pages.",
        "- De-prioritizes boxes that look tiny, oversized, low-score, quarantined, or correction-only.",
        "- Keeps only a capped number of boxes per image in the first pass so review time covers more split groups.",
        "- Writes `codex_benchmark_manifest.jsonl` from the same images, so Codex-style segmentation can be compared against the human-reviewed boxes without changing the label queue.",
        "",
        "## Recommended First Pass",
        "",
    ]
    for index, image in enumerate(report.get("recommended_images") or [], start=1):
        lines.append(
            f"- {index:02d}. score={image['priority_score']} wb={image['workbench_order']} "
            f"image#{image['image_order']} boxes={image['selected_box_count']}/{image['box_count_on_image']} "
            f"cohort={image['cohort']} kind={image['source_kind']} html=`{image['workbench_html']}`"
        )
        if image.get("top_reasons"):
            lines.append(f"  reasons: {', '.join(image['top_reasons'])}")
    lines.extend(["", "## Recommended Cohorts", ""])
    for key, value in summary["recommended_cohort_counts"].items():
        lines.append(f"- `{key}`: {value}")
    if report.get("outputs", {}).get("review_root"):
        review_summary = report.get("review_root_summary") or {}
        lines.extend(
            [
                "",
                "## Review Root",
                "",
                f"- Prelabel root: `{report['outputs']['review_root']}`",
                f"- Images: {review_summary.get('images', summary['recommended_images'])}",
                f"- Review boxes: {review_summary.get('boxes', summary['recommended_boxes'])}",
                f"- Recommended high-priority boxes: {review_summary.get('recommended_boxes', summary['recommended_boxes'])}",
                "- The review root includes all draft boxes on each selected image so approved training labels do not become partial-image annotations.",
                "",
                "Build the static review UI with:",
                "",
                "```powershell",
                "python scripts\\question_detector_review_workbench.py `",
                f"  --prelabel-root {report['outputs']['review_root']} `",
                f"  --out {report['outputs']['review_root'].removesuffix('-prelabel')}-workbench `",
                "  --clean",
                "```",
            ]
        )
    lines.extend(["", "## Box Source Counts", ""])
    for key, value in summary["source_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Quality Flags", ""])
    for key, value in summary["quality_flag_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def make_html(report: dict[str, Any]) -> str:
    rows = []
    for index, item in enumerate(report.get("recommended_images") or [], start=1):
        link = ""
        if item.get("workbench_html"):
            try:
                link = Path(str(item["workbench_html"])).resolve().as_uri()
            except ValueError:
                link = str(item["workbench_html"])
        reasons = ", ".join(item.get("top_reasons") or [])
        rows.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{html.escape(str(item.get('priority_score')))}</td>"
            f"<td><a href=\"{html.escape(link)}\">open</a></td>"
            f"<td>{html.escape(str(item.get('workbench_order')))}</td>"
            f"<td>{html.escape(str(item.get('image_order')))}</td>"
            f"<td>{html.escape(str(item.get('selected_box_count')))} / {html.escape(str(item.get('box_count_on_image')))}</td>"
            f"<td>{html.escape(str(item.get('cohort')))}</td>"
            f"<td>{html.escape(str(item.get('source_kind')))}</td>"
            f"<td>{html.escape(reasons)}</td>"
            f"<td><code>{html.escape(str(item.get('image')))}</code></td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Question Detector Review Priority</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2937; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #f6f7f8; }}
code {{ font-size: 12px; }}
.metrics {{ display:flex; gap:12px; margin:16px 0; }}
.metric {{ border:1px solid #ddd; border-radius:6px; padding:10px 12px; }}
</style>
</head>
<body>
<h1>Question Detector Review Priority</h1>
<div class="metrics">
<div class="metric"><strong>{report['summary']['total_boxes']}</strong><br>boxes scored</div>
<div class="metric"><strong>{report['summary']['recommended_images']}</strong><br>first-pass images</div>
<div class="metric"><strong>{report['summary']['recommended_boxes']}</strong><br>first-pass boxes</div>
<div class="metric"><strong>{html.escape(str(report['inputs']['strategy']))}</strong><br>strategy</div>
</div>
<table>
<thead><tr><th>#</th><th>Score</th><th>Open</th><th>WB</th><th>Image</th><th>Boxes</th><th>Cohort</th><th>Kind</th><th>Reasons</th><th>Path</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body>
</html>
"""


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest, candidates = load_candidates(args.manifest)
    ranked = sorted(candidates, key=lambda row: (row["net_score"], row["value_score"]), reverse=True)
    recommended_images, ranked_images = select_ranked_images(
        ranked,
        limit_images=args.limit_images,
        max_boxes_per_image=args.max_boxes_per_image,
        strategy=args.strategy,
        max_images_per_cohort=args.max_images_per_cohort,
        max_images_per_source_kind=args.max_images_per_source_kind,
    )
    source_counts = Counter(str(row.get("box", {}).get("source") or "unknown") for row in candidates)
    kind_counts = Counter(str(row.get("source_kind") or "unknown") for row in candidates)
    cohort_counts = Counter(str(row.get("cohort") or "unknown") for row in ranked_images)
    recommended_cohort_counts = Counter(str(row.get("cohort") or "unknown") for row in recommended_images)
    flag_counts: Counter[str] = Counter()
    for row in candidates:
        flag_counts.update(str(flag) for flag in (row.get("box", {}).get("quality_flags") or []))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "manifest": str(args.manifest),
            "limit_images": args.limit_images,
            "max_boxes_per_image": args.max_boxes_per_image,
            "strategy": args.strategy,
            "max_images_per_cohort": args.max_images_per_cohort,
            "max_images_per_source_kind": args.max_images_per_source_kind,
        },
        "summary": {
            "workbench_count": manifest.get("workbench_count"),
            "total_boxes": len(candidates),
            "total_images": len({str(row.get("review_id") or row.get("image")) for row in candidates}),
            "recommended_images": len(recommended_images),
            "recommended_boxes": sum(int(item.get("selected_box_count") or 0) for item in recommended_images),
            "source_counts": dict(source_counts.most_common()),
            "source_kind_counts": dict(kind_counts.most_common()),
            "cohort_counts": dict(cohort_counts.most_common()),
            "recommended_cohort_counts": dict(recommended_cohort_counts.most_common()),
            "quality_flag_counts": dict(flag_counts.most_common(40)),
        },
        "top_boxes": ranked[: args.limit_boxes],
        "ranked_images": ranked_images[: args.limit_ranked_images],
        "recommended_images": recommended_images,
    }
    report["outputs"] = {
        "review_priority_json": str(args.out / "review_priority.json"),
        "review_priority_md": str(args.out / "review_priority.md"),
        "codex_benchmark_manifest": str(args.out / "codex_benchmark_manifest.jsonl"),
    }
    write_json(args.out / "review_priority.json", report)
    write_jsonl(args.out / "codex_benchmark_manifest.jsonl", benchmark_rows(recommended_images))
    review_root_summary: dict[str, Any] | None = None
    if args.review_root:
        review_root_summary = write_review_root(args.review_root, report)
        report["outputs"]["review_root"] = str(args.review_root)
        report["review_root_summary"] = review_root_summary
        write_json(args.out / "review_priority.json", report)
    (args.out / "review_priority.md").write_text(make_markdown(report), encoding="utf-8")
    (args.out / "index.html").write_text(make_html(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prioritize pending question-detector review boxes.")
    parser.add_argument("--manifest", type=Path, required=True, help="review_queue_manifest.json")
    parser.add_argument("--limit-images", type=int, default=40, help="Recommended first-pass image count.")
    parser.add_argument("--limit-boxes", type=int, default=200, help="Number of top boxes to store.")
    parser.add_argument("--limit-ranked-images", type=int, default=200, help="Number of ranked image rows to store.")
    parser.add_argument("--max-boxes-per-image", type=int, default=12)
    parser.add_argument("--strategy", choices=("balanced", "score"), default="balanced")
    parser.add_argument("--max-images-per-cohort", type=int, default=0, help="Balanced-mode cap; 0 chooses a sensible default.")
    parser.add_argument("--max-images-per-source-kind", type=int, default=0, help="Balanced-mode cap; 0 chooses a sensible default.")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-review-priority"))
    parser.add_argument("--review-root", type=Path, help="Optional prelabel-style root for the selected priority slice.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    report = build_report(args)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "total_boxes": report["summary"]["total_boxes"],
                "recommended_images": report["summary"]["recommended_images"],
                "recommended_boxes": report["summary"]["recommended_boxes"],
                "recommended_cohorts": report["summary"]["recommended_cohort_counts"],
                "review_root": report.get("outputs", {}).get("review_root"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
