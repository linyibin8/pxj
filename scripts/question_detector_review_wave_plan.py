"""Generate sequential active-learning review waves from one prelabel pool.

This wraps question_detector_active_batch.py for repeated use: each generated
wave is automatically added to the queued-exclusion set before selecting the
next wave. The result shows whether a prelabel pool can sustain multiple review
rounds or is exhausted.
"""

from __future__ import annotations

import argparse
import json
import shutil
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import question_detector_active_batch as active_batch
import question_detector_review_workbench as review_workbench


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_active_wave(args: argparse.Namespace, wave_root: Path, exclude_queued: list[Path]) -> dict[str, Any]:
    wave_args = Namespace(
        prelabel_root=args.prelabel_root,
        out=wave_root,
        limit=args.limit,
        max_per_session=args.max_per_session,
        max_boxes_total=args.max_boxes_per_wave,
        ahash_threshold=args.ahash_threshold,
        include_quarantine=args.include_quarantine,
        allow_exact_duplicates=args.allow_exact_duplicates,
        exclude_approved_jsonl=args.exclude_approved_jsonl,
        exclude_queued_root=exclude_queued,
        clean=True,
    )
    return active_batch.run(wave_args)


def build_wave_workbench(args: argparse.Namespace, wave_root: Path, workbench_root: Path) -> dict[str, Any]:
    workbench_args = Namespace(
        prelabel_root=wave_root,
        out=workbench_root,
        include_quarantine=False,
        limit=0,
        no_copy_images=False,
        decisions=None,
        approved_jsonl=workbench_root / "approved_boxes.jsonl",
        approved_statuses="approved,accepted,corrected,verified",
        clean=True,
    )
    return review_workbench.build_workbench(workbench_args)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    exclude_queued = list(args.exclude_queued_root)
    waves: list[dict[str, Any]] = []
    exhausted = False
    for wave_index in range(1, args.max_waves + 1):
        wave_root = args.out / f"wave-{wave_index:03d}"
        summary = run_active_wave(args, wave_root, exclude_queued)
        selected_images = int(summary.get("selected_images") or 0)
        selected_boxes = int(summary.get("selected_boxes") or 0)
        wave = {
            "wave": wave_index,
            "root": str(wave_root),
            "candidate_count": summary.get("candidate_count"),
            "selected_images": selected_images,
            "selected_boxes": selected_boxes,
            "skipped": summary.get("skipped") or {},
            "selection": summary.get("selection") or {},
            "workbench": "",
            "workbench_summary": {},
        }
        if selected_images <= 0 or selected_boxes <= 0:
            exhausted = True
            waves.append(wave)
            break
        exclude_queued.append(wave_root)
        if args.build_workbenches:
            workbench_root = args.out / f"wave-{wave_index:03d}-workbench"
            workbench_summary = build_wave_workbench(args, wave_root, workbench_root)
            wave["workbench"] = str(workbench_root)
            wave["workbench_summary"] = workbench_summary
        waves.append(wave)
        if selected_images < args.limit and args.stop_when_underfilled and args.max_boxes_per_wave <= 0:
            exhausted = True
            break
    total_images = sum(int(wave.get("selected_images") or 0) for wave in waves)
    total_boxes = sum(int(wave.get("selected_boxes") or 0) for wave in waves)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prelabel_root": str(args.prelabel_root),
        "out": str(args.out),
        "max_waves": args.max_waves,
        "limit_per_wave": args.limit,
        "max_boxes_per_wave": args.max_boxes_per_wave,
        "include_quarantine": bool(args.include_quarantine),
        "build_workbenches": bool(args.build_workbenches),
        "exhausted": exhausted,
        "total_selected_images": total_images,
        "total_selected_boxes": total_boxes,
        "waves": waves,
        "inputs": {
            "exclude_approved_jsonl": [str(path) for path in args.exclude_approved_jsonl],
            "exclude_queued_root": [str(path) for path in args.exclude_queued_root],
        },
        "notes": [
            "Wave outputs are review queues, not approved training data.",
            "If exhausted=true and total_selected_boxes is small, generate more prelabels from a larger historical candidate pool.",
        ],
    }
    write_json(args.out / "review_wave_plan.json", summary)
    (args.out / "review_wave_plan.md").write_text(markdown(summary), encoding="utf-8")
    return summary


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Question Detector Review Wave Plan",
        "",
        f"- Generated: {summary.get('generated_at')}",
        f"- Prelabel root: `{summary.get('prelabel_root')}`",
        f"- Exhausted: `{str(summary.get('exhausted')).lower()}`",
        f"- Total selected: {summary.get('total_selected_images')} images / {summary.get('total_selected_boxes')} boxes",
        f"- Max boxes per wave: {summary.get('max_boxes_per_wave')}",
        "",
        "## Waves",
        "",
    ]
    for wave in summary.get("waves") or []:
        lines.append(
            f"- wave-{int(wave.get('wave') or 0):03d}: candidates={wave.get('candidate_count')} "
            f"selected={wave.get('selected_images')} images / {wave.get('selected_boxes')} boxes "
            f"root=`{wave.get('root')}` workbench=`{wave.get('workbench')}`"
        )
    lines.extend(["", "## Notes", ""])
    for note in summary.get("notes") or []:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sequential active-learning review waves from one prelabel root.")
    parser.add_argument("--prelabel-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-review-wave-plan"))
    parser.add_argument("--max-waves", type=int, default=4)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--max-per-session", type=int, default=4)
    parser.add_argument("--max-boxes-per-wave", type=int, default=0, help="Soft cap on draft boxes per generated wave; 0 disables the cap.")
    parser.add_argument("--ahash-threshold", type=int, default=4)
    parser.add_argument("--include-quarantine", action="store_true")
    parser.add_argument("--allow-exact-duplicates", action="store_true")
    parser.add_argument("--exclude-approved-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--exclude-queued-root", type=Path, action="append", default=[])
    parser.add_argument("--build-workbenches", action="store_true")
    parser.add_argument("--no-stop-when-underfilled", dest="stop_when_underfilled", action="store_false")
    parser.set_defaults(stop_when_underfilled=True)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "exhausted": summary["exhausted"],
                "max_boxes_per_wave": summary["max_boxes_per_wave"],
                "total_selected_images": summary["total_selected_images"],
                "total_selected_boxes": summary["total_selected_boxes"],
                "waves": [
                    {
                        "wave": wave["wave"],
                        "candidate_count": wave["candidate_count"],
                        "selected_images": wave["selected_images"],
                        "selected_boxes": wave["selected_boxes"],
                        "workbench": wave["workbench"],
                    }
                    for wave in summary["waves"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
