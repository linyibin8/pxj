"""Build a consolidated manifest for detector review workbenches.

This does not approve boxes or create training labels. It inventories review
workbenches, decisions exports, approved JSONL files, and the merge command
needed after review is complete.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import question_detector_iteration_report as iteration_report


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def training_plan_workbenches(paths: list[Path]) -> list[Path]:
    workbenches: list[Path] = []
    for path in paths:
        plan_path = path if path.is_file() else path / "training_readiness_plan.json"
        if not plan_path.is_file():
            continue
        try:
            report = read_json(plan_path)
        except (OSError, json.JSONDecodeError):
            continue
        inputs = report.get("inputs") if isinstance(report.get("inputs"), dict) else {}
        for workbench in inputs.get("review_workbenches") or []:
            if workbench:
                workbenches.append(Path(str(workbench)))
    return workbenches


def dedupe_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def build_merge_command(approved_inputs: list[str], merge_out: Path) -> str:
    if not approved_inputs:
        return ""
    return (
        "python scripts\\question_detector_merge_reviewed.py "
        + " ".join(approved_inputs)
        + f" --out {merge_out} --clean"
    )


def path_uri(path: str) -> str:
    if not path:
        return ""
    try:
        return Path(path).resolve().as_uri()
    except ValueError:
        return path


def html_report(report: dict[str, Any]) -> str:
    rows = []
    for item in report.get("workbenches") or []:
        index_html = str(item.get("index_html") or "")
        link = f'<a href="{html.escape(path_uri(index_html))}">open</a>' if index_html else ""
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('order') or ''))}</td>"
            f"<td>{link}</td>"
            f"<td>{html.escape(str(item.get('boxes') or 0))}</td>"
            f"<td>{html.escape(str(item.get('pending_box_count') or 0))}</td>"
            f"<td>{html.escape(str(item.get('decision_file_count') or 0))}</td>"
            f"<td>{html.escape(str(item.get('approved_box_count') or 0))}</td>"
            f"<td><code>{html.escape(str(item.get('path') or ''))}</code></td>"
            "</tr>"
        )
    merge = html.escape(str(report.get("merge_command") or ""))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Question Detector Review Queue</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
th {{ background: #f6f7f8; }}
code {{ font-size: 12px; }}
.summary {{ display: flex; gap: 18px; margin: 16px 0 24px; }}
.metric {{ border: 1px solid #ddd; padding: 10px 12px; border-radius: 6px; }}
</style>
</head>
<body>
<h1>Question Detector Review Queue</h1>
<div class="summary">
  <div class="metric"><strong>{report.get('workbench_count')}</strong><br>workbenches</div>
  <div class="metric"><strong>{report.get('total_boxes')}</strong><br>boxes</div>
  <div class="metric"><strong>{report.get('pending_boxes')}</strong><br>pending boxes</div>
  <div class="metric"><strong>{report.get('approved_boxes')}</strong><br>approved boxes</div>
</div>
<h2>Workbenches</h2>
<table>
<thead><tr><th>#</th><th>Open</th><th>Boxes</th><th>Pending</th><th>Decision files</th><th>Approved boxes</th><th>Path</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
<h2>Merge Command</h2>
<pre>{merge}</pre>
</body>
</html>
"""


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Question Detector Review Queue",
        "",
        f"- Generated: {report.get('generated_at')}",
        f"- Workbenches: {report.get('workbench_count')}",
        f"- Boxes: {report.get('total_boxes')}",
        f"- Pending boxes: {report.get('pending_boxes')}",
        f"- Approved boxes: {report.get('approved_boxes')}",
        "",
        "## Workbenches",
        "",
    ]
    for item in report.get("workbenches") or []:
        lines.append(
            f"- {item.get('order'):02d}. `{item.get('path')}` boxes={item.get('boxes')} "
            f"pending={item.get('pending_box_count')} decisions={item.get('decision_file_count')} "
            f"approved={item.get('approved_box_count')} html=`{item.get('index_html')}`"
        )
        if item.get("export_command"):
            lines.append(f"  export: `{item.get('export_command')}`")
    lines.extend(["", "## Merge", ""])
    if report.get("merge_command"):
        lines.append(f"`{report['merge_command']}`")
    else:
        lines.append("No approved JSONL inputs are available yet.")
    lines.append("")
    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    approved_statuses = {part.strip().lower() for part in args.approved_statuses.split(",") if part.strip()}
    paths = dedupe_paths(list(args.review_workbench) + training_plan_workbenches(args.training_plan))
    summaries = [iteration_report.review_workbench_summary(path, approved_statuses) for path in paths]
    for index, item in enumerate(summaries, start=1):
        item["order"] = index
    approved_inputs: list[str] = []
    for item in summaries:
        approved_inputs.extend(str(path) for path in (item.get("approved_inputs") or []) if path)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "review_workbenches": [str(path) for path in args.review_workbench],
            "training_plans": [str(path) for path in args.training_plan],
            "merge_out": str(args.merge_out),
        },
        "workbench_count": len(summaries),
        "total_boxes": sum(to_int(item.get("boxes")) for item in summaries),
        "pending_boxes": sum(to_int(item.get("pending_box_count")) for item in summaries),
        "decision_files": sum(to_int(item.get("decision_file_count")) for item in summaries),
        "approved_boxes": sum(to_int(item.get("approved_box_count")) for item in summaries),
        "needs_review_count": sum(1 for item in summaries if item.get("needs_review")),
        "needs_export_count": sum(1 for item in summaries if item.get("needs_export")),
        "approved_inputs": approved_inputs,
        "merge_command": build_merge_command(approved_inputs, args.merge_out),
        "workbenches": summaries,
    }
    write_json(args.out / "review_queue_manifest.json", report)
    (args.out / "review_queue_manifest.md").write_text(markdown_report(report), encoding="utf-8")
    (args.out / "index.html").write_text(html_report(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a consolidated detector review queue manifest.")
    parser.add_argument("--review-workbench", type=Path, action="append", default=[])
    parser.add_argument("--training-plan", type=Path, action="append", default=[])
    parser.add_argument("--merge-out", type=Path, default=Path("diagnostics/question-detector-merged-reviewed-next"))
    parser.add_argument("--approved-statuses", default="approved,accepted,corrected,verified")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-review-queue-manifest"))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    report = build_report(args)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "workbenches": report["workbench_count"],
                "boxes": report["total_boxes"],
                "pending_boxes": report["pending_boxes"],
                "approved_boxes": report["approved_boxes"],
                "needs_review_count": report["needs_review_count"],
                "needs_export_count": report["needs_export_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
