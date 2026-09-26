"""Check live/simulator observation crop runtime telemetry.

Inputs can be JSON, JSONL, or text logs that contain JSON objects. The checker
looks for iOS client metrics and backend batch response metrics such as:

- question_crop_client_metrics.analysis_duration_ms
- question_crop_client_metrics.cached_frame_count
- question_crop_client_metrics.ocr_frame_count
- question_crop_client_metrics.attached_crop_file_count
- question_crop_server_duration_ms

The output is release-gate evidence; it does not train or approve labels.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JSON_OBJECT_RE = re.compile(r"\{.*\}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def input_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for pattern in ("*.json", "*.jsonl", "*.log", "*.txt"):
                files.extend(sorted(path.rglob(pattern)))
    return files


def infer_evidence_kind(inputs: list[Path], out: Path, requested: str | None) -> str:
    if requested:
        return requested
    names = [str(out), *(str(path) for path in inputs)]
    if any("fixture" in name.lower() for name in names):
        return "fixture"
    return "live"


def parse_json_values(path: Path) -> list[Any]:
    text = read_text(path)
    values: list[Any] = []
    if path.suffix.lower() == ".json":
        try:
            return [json.loads(text)]
        except json.JSONDecodeError:
            pass
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        candidates = [line]
        match = JSON_OBJECT_RE.search(line)
        if match and match.group(0) != line:
            candidates.append(match.group(0))
        for candidate in candidates:
            try:
                values.append(json.loads(candidate))
                break
            except json.JSONDecodeError:
                continue
    return values


def walk_objects(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        found.append(value)
        for key, child in value.items():
            if key == "question_crop_client_metrics" and isinstance(child, dict):
                continue
            found.extend(walk_objects(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(walk_objects(child))
    return found


def number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def maybe_record(obj: dict[str, Any], source: str) -> dict[str, Any] | None:
    client = obj.get("question_crop_client_metrics") if isinstance(obj.get("question_crop_client_metrics"), dict) else {}
    metrics = {**client, **obj}
    keys = {
        "analysis_duration_ms",
        "cached_frame_count",
        "ocr_frame_count",
        "attached_crop_file_count",
        "question_crop_server_duration_ms",
        "totalDurationMs",
        "total_duration_ms",
        "frame_count",
    }
    if not any(key in metrics for key in keys):
        return None
    cached = number(metrics.get("cached_frame_count"))
    ocr = number(metrics.get("ocr_frame_count"))
    frame_count = number(metrics.get("frame_count")) or cached + ocr
    return {
        "source": source,
        "analysis_duration_ms": number(metrics.get("analysis_duration_ms"), math.nan),
        "cached_frame_count": int(cached),
        "ocr_frame_count": int(ocr),
        "frame_count": int(frame_count),
        "attached_crop_file_count": int(number(metrics.get("attached_crop_file_count"))),
        "question_crop_server_duration_ms": number(metrics.get("question_crop_server_duration_ms"), math.nan),
        "total_duration_ms": number(metrics.get("total_duration_ms", metrics.get("totalDurationMs")), math.nan),
        "coreml_model_loaded_frame_count": int(number(metrics.get("coreml_model_loaded_frame_count"))),
        "fallback_to_ocr_frame_count": int(number(metrics.get("fallback_to_ocr_frame_count"))),
    }


def collect_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for file_path in input_files(paths):
        for value in parse_json_values(file_path):
            for obj in walk_objects(value):
                record = maybe_record(obj, str(file_path))
                if record:
                    records.append(record)
    return records


def finite(values: list[float]) -> list[float]:
    return sorted(value for value in values if math.isfinite(value))


def percentile(values: list[float], pct: float) -> float | None:
    clean = finite(values)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    index = (len(clean) - 1) * pct
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return clean[int(index)]
    return clean[low] * (high - index) + clean[high] * (index - low)


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str, values: dict[str, Any]) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail, "values": values})


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    analysis_values = [number(row.get("analysis_duration_ms"), math.nan) for row in records]
    server_values = [number(row.get("question_crop_server_duration_ms"), math.nan) for row in records]
    total_frames = sum(int(row.get("frame_count") or 0) for row in records)
    ocr_frames = sum(int(row.get("ocr_frame_count") or 0) for row in records)
    cached_frames = sum(int(row.get("cached_frame_count") or 0) for row in records)
    attached_crop_files = sum(int(row.get("attached_crop_file_count") or 0) for row in records)
    fresh_ocr_ratio = (ocr_frames / max(1, ocr_frames + cached_frames)) if (ocr_frames or cached_frames) else 0.0
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "frame_count": total_frames,
        "cached_frame_count": cached_frames,
        "ocr_frame_count": ocr_frames,
        "fresh_ocr_ratio": round(fresh_ocr_ratio, 6),
        "attached_crop_file_count": attached_crop_files,
        "analysis_duration_ms": {
            "p50": percentile(analysis_values, 0.50),
            "p95": percentile(analysis_values, 0.95),
            "max": max(finite(analysis_values), default=None),
        },
        "question_crop_server_duration_ms": {
            "p50": percentile(server_values, 0.50),
            "p95": percentile(server_values, 0.95),
            "max": max(finite(server_values), default=None),
        },
        "thresholds": {
            "min_records": args.min_records,
            "max_analysis_p50_ms": args.max_analysis_p50_ms,
            "max_analysis_p95_ms": args.max_analysis_p95_ms,
            "max_server_p95_ms": args.max_server_p95_ms,
            "max_fresh_ocr_ratio": args.max_fresh_ocr_ratio,
            "max_attached_crop_files": args.max_attached_crop_files,
        },
        "records_sample": records[:20],
    }
    checks: list[dict[str, Any]] = []
    analysis_p50 = summary["analysis_duration_ms"]["p50"]
    analysis_p95 = summary["analysis_duration_ms"]["p95"]
    server_p95 = summary["question_crop_server_duration_ms"]["p95"]
    add_check(checks, "minimum runtime records", len(records) >= args.min_records, "Need enough live/simulator runtime records.", {"record_count": len(records)})
    add_check(checks, "rect-only upload", attached_crop_files <= args.max_attached_crop_files, "Observation upload should not attach question crop JPEG files.", {"attached_crop_file_count": attached_crop_files})
    add_check(checks, "analysis p50 latency", analysis_p50 is not None and analysis_p50 <= args.max_analysis_p50_ms, "Local observation analysis p50 must stay within budget.", {"p50": analysis_p50})
    add_check(checks, "analysis p95 latency", analysis_p95 is not None and analysis_p95 <= args.max_analysis_p95_ms, "Local observation analysis p95 must stay within budget.", {"p95": analysis_p95})
    if server_p95 is not None:
        add_check(checks, "server crop p95 latency", server_p95 <= args.max_server_p95_ms, "Backend canonical crop expansion p95 must stay within budget.", {"p95": server_p95})
    add_check(checks, "fresh OCR ratio", fresh_ocr_ratio <= args.max_fresh_ocr_ratio, "Batch upload should mostly reuse cached live-scan segmentation instead of rerunning OCR.", {"fresh_ocr_ratio": fresh_ocr_ratio})
    summary["checks"] = checks
    summary["passed"] = all(item["passed"] for item in checks)
    return summary


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Question Observation Runtime Perf Check",
        "",
        f"- Generated: {summary.get('generated_at')}",
        f"- Passed: `{str(summary.get('passed')).lower()}`",
        f"- Records: {summary.get('record_count')}",
        f"- Frames: {summary.get('frame_count')}",
        f"- Fresh OCR ratio: {summary.get('fresh_ocr_ratio')}",
        f"- Attached crop files: {summary.get('attached_crop_file_count')}",
        f"- Analysis p50/p95: {summary.get('analysis_duration_ms', {}).get('p50')} / {summary.get('analysis_duration_ms', {}).get('p95')} ms",
        f"- Server crop p95: {summary.get('question_crop_server_duration_ms', {}).get('p95')} ms",
        "",
        "## Checks",
        "",
    ]
    for check in summary.get("checks") or []:
        lines.append(f"- `{str(check.get('passed')).lower()}` {check.get('name')}: {check.get('detail')} {check.get('values')}")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    records = collect_records(args.input)
    summary = summarize(records, args)
    summary["evidence_kind"] = infer_evidence_kind(args.input, args.out, args.evidence_kind)
    summary["inputs"] = [str(path) for path in args.input]
    write_json(args.out / "summary.json", summary)
    (args.out / "summary.md").write_text(markdown(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Check iOS/backend observation runtime performance telemetry.")
    parser.add_argument("--input", type=Path, action="append", required=True, help="JSON/JSONL/log file or directory containing runtime telemetry.")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-observation-runtime-perf-check"))
    parser.add_argument("--min-records", type=int, default=5)
    parser.add_argument("--max-analysis-p50-ms", type=float, default=120.0)
    parser.add_argument("--max-analysis-p95-ms", type=float, default=300.0)
    parser.add_argument("--max-server-p95-ms", type=float, default=250.0)
    parser.add_argument("--max-fresh-ocr-ratio", type=float, default=0.75)
    parser.add_argument("--max-attached-crop-files", type=int, default=0)
    parser.add_argument("--evidence-kind", choices=["live", "simulator", "fixture"], help="Mark whether this telemetry is real release evidence or a parser/integration fixture.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "passed": summary["passed"],
                "record_count": summary["record_count"],
                "fresh_ocr_ratio": summary["fresh_ocr_ratio"],
                "attached_crop_file_count": summary["attached_crop_file_count"],
                "analysis_p95_ms": summary["analysis_duration_ms"]["p95"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
