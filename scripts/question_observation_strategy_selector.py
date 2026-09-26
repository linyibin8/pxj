"""Compare observation crop strategies across existing evaluation reports.

This script is a read-only aggregator. It does not create ground truth and does
not re-score crops. It normalizes already-generated reports into one Pareto
table so product decisions compare recall, fallback pressure, and VLM pixel
cost with the same vocabulary.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def resolve_summary(path: Path) -> Path:
    if path.is_file():
        return path
    return path / "summary.json"


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def replay_label(summary: dict[str, Any], report_path: Path) -> str:
    replays = summary.get("replays") if isinstance(summary.get("replays"), list) else []
    if replays and isinstance(replays[0], dict):
        replay_path = str(replays[0].get("path") or "")
        if replay_path:
            return Path(replay_path).name
    return report_path.parent.name


def evidence_kind(label: str, path: Path) -> str:
    text = f"{label} {path}".lower()
    if "synthetic" in text:
        return "synthetic_stress"
    if "kpai" in text:
        return "reviewed_kpai"
    if "propagated" in text:
        return "reviewed_propagated"
    return "unknown"


@dataclass
class StrategyRow:
    strategy_id: str
    family: str
    report: str
    replay: str
    evidence_kind: str
    image_count: int = 0
    represented_gt_key_count: int = 0
    crop_gt_recall: float | None = None
    policy_gt_recall: float | None = None
    fallback_image_count: int | None = None
    fallback_image_ratio: float | None = None
    crop_pixels_vs_full: float | None = None
    section_pixels_vs_full: float | None = None
    total_pixels_vs_full: float | None = None
    full_frame_all_pixels: int = 0
    vlm_pixels: int | None = None
    selected_candidate_count: int | None = None
    skipped_candidate_count: int | None = None
    skipped_strong_candidate_count: int | None = None
    skipped_confident_candidate_count: int | None = None
    section_count: int | None = None
    status: str = "diagnostic"
    recommendation_notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "family": self.family,
            "report": self.report,
            "replay": self.replay,
            "evidence_kind": self.evidence_kind,
            "image_count": self.image_count,
            "represented_gt_key_count": self.represented_gt_key_count,
            "crop_gt_recall": self.crop_gt_recall,
            "policy_gt_recall": self.policy_gt_recall,
            "fallback_image_count": self.fallback_image_count,
            "fallback_image_ratio": self.fallback_image_ratio,
            "crop_pixels_vs_full": self.crop_pixels_vs_full,
            "section_pixels_vs_full": self.section_pixels_vs_full,
            "total_pixels_vs_full": self.total_pixels_vs_full,
            "full_frame_all_pixels": self.full_frame_all_pixels,
            "vlm_pixels": self.vlm_pixels,
            "selected_candidate_count": self.selected_candidate_count,
            "skipped_candidate_count": self.skipped_candidate_count,
            "skipped_strong_candidate_count": self.skipped_strong_candidate_count,
            "skipped_confident_candidate_count": self.skipped_confident_candidate_count,
            "section_count": self.section_count,
            "status": self.status,
            "recommendation_notes": self.recommendation_notes,
            "extra": self.extra,
        }


def base_row(summary: dict[str, Any], report_path: Path, family: str, suffix: str) -> StrategyRow:
    replay = replay_label(summary, report_path)
    return StrategyRow(
        strategy_id=f"{replay}:{suffix}",
        family=family,
        report=str(report_path),
        replay=replay,
        evidence_kind=evidence_kind(replay, report_path),
        image_count=as_int(summary.get("image_count") or (summary.get("counts") or {}).get("images")),
        represented_gt_key_count=as_int(summary.get("represented_gt_key_count") or (summary.get("counts") or {}).get("represented_gt_key_count")),
        full_frame_all_pixels=as_int(summary.get("full_frame_all_pixels") or ((summary.get("metrics") or {}).get("full_frame_resized_pixels"))),
    )


def full_frame_row(summary: dict[str, Any], report_path: Path) -> StrategyRow:
    row = base_row(summary, report_path, "full_frame_baseline", "full_frame")
    row.crop_gt_recall = 1.0 if row.represented_gt_key_count else None
    row.policy_gt_recall = row.crop_gt_recall
    row.fallback_image_count = row.image_count
    row.fallback_image_ratio = 1.0 if row.image_count else None
    row.crop_pixels_vs_full = 0.0
    row.total_pixels_vs_full = 1.0
    row.vlm_pixels = row.full_frame_all_pixels or None
    row.status = "baseline"
    row.recommendation_notes.append("Full-frame VLM baseline: accurate fallback reference, highest expected transfer/VLM cost.")
    return row


def parse_candidate_eval(path: Path) -> list[StrategyRow]:
    summary = read_json(resolve_summary(path))
    report_path = resolve_summary(path)
    rows = [full_frame_row(summary, report_path)]
    counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    row = base_row(summary, report_path, "candidate_eval", "all_candidates_crop_only")
    row.image_count = as_int(counts.get("images"), row.image_count)
    row.represented_gt_key_count = as_int(counts.get("represented_gt_key_count"), row.represented_gt_key_count)
    row.crop_gt_recall = as_float(metrics.get("represented_gt_recall"))
    row.policy_gt_recall = row.crop_gt_recall
    row.fallback_image_count = 0
    row.fallback_image_ratio = 0.0
    row.crop_pixels_vs_full = as_float(metrics.get("crop_pixels_vs_full_frame_pixels"))
    row.total_pixels_vs_full = row.crop_pixels_vs_full
    row.full_frame_all_pixels = as_int(metrics.get("full_frame_resized_pixels"), row.full_frame_all_pixels)
    row.vlm_pixels = as_int(metrics.get("crop_resized_pixels")) or None
    row.selected_candidate_count = as_int(counts.get("candidate_count"))
    row.status = "candidate_set"
    row.extra = {
        "overcrop_match_count": counts.get("overcrop_match_count"),
        "unverified_candidate_count": counts.get("unverified_candidate_count"),
        "reviewed_candidate_match_rate": metrics.get("reviewed_candidate_match_rate"),
    }
    rows.append(row)
    return rows


def row_from_policy(summary: dict[str, Any], report_path: Path, family: str, suffix: str, policy: dict[str, Any], status: str) -> StrategyRow:
    row = base_row(summary, report_path, family, suffix)
    row.image_count = as_int(policy.get("image_count"), row.image_count)
    row.represented_gt_key_count = as_int(policy.get("represented_gt_key_count"), row.represented_gt_key_count)
    row.crop_gt_recall = as_float(policy.get("crop_only_gt_recall") or policy.get("selected_gt_recall") or policy.get("combined_crop_gt_recall"))
    row.policy_gt_recall = as_float(policy.get("policy_gt_recall"), row.crop_gt_recall or 0.0)
    row.fallback_image_count = as_int(policy.get("fallback_image_count"))
    row.fallback_image_ratio = as_float(policy.get("fallback_image_ratio"))
    row.crop_pixels_vs_full = as_float(policy.get("crop_pixels_vs_full") or policy.get("selected_crop_pixels_vs_full"))
    row.section_pixels_vs_full = as_float(policy.get("section_pixels_vs_full")) if "section_pixels_vs_full" in policy else None
    row.total_pixels_vs_full = as_float(policy.get("total_pixels_vs_full"))
    row.full_frame_all_pixels = as_int(policy.get("full_frame_all_pixels"), row.full_frame_all_pixels)
    row.vlm_pixels = as_int(policy.get("total_vlm_pixels")) or None
    row.selected_candidate_count = as_int(policy.get("selected_candidate_count") or policy.get("uploaded_selected_candidate_count")) or None
    row.skipped_candidate_count = as_int(policy.get("skipped_candidate_count")) or None
    row.skipped_strong_candidate_count = as_int(policy.get("skipped_strong_candidate_count")) or None
    row.skipped_confident_candidate_count = as_int(policy.get("skipped_confident_candidate_count")) or None
    row.section_count = as_int(policy.get("section_count")) or None
    row.status = status
    row.extra = {
        "fallback_reason_counts": policy.get("fallback_reason_counts") or {},
        "policy_id": policy.get("policy_id"),
        "cap": policy.get("cap"),
        "strategy_id": policy.get("strategy_id"),
        "selected_covered_gt_key_count": policy.get("selected_covered_gt_key_count"),
        "section_covered_gt_key_count": policy.get("section_covered_gt_key_count"),
        "combined_crop_covered_gt_key_count": policy.get("combined_crop_covered_gt_key_count"),
    }
    return row


def parse_fallback_tune(path: Path) -> list[StrategyRow]:
    summary = read_json(resolve_summary(path))
    report_path = resolve_summary(path)
    rows = [full_frame_row(summary, report_path)]
    current = summary.get("baseline_backend_current")
    legacy = summary.get("baseline_backend_legacy_before_tuned")
    recommendation = summary.get("recommendation")
    if isinstance(legacy, dict):
        rows.append(row_from_policy(summary, report_path, "fallback_policy", "backend_legacy", legacy, "legacy"))
    if isinstance(current, dict):
        rows.append(row_from_policy(summary, report_path, "fallback_policy", "backend_current", current, "live"))
    if isinstance(recommendation, dict) and recommendation != current:
        rows.append(row_from_policy(summary, report_path, "fallback_policy", "recommended_policy", recommendation, "recommended"))
    return rows


def parse_cap_sweep(path: Path) -> list[StrategyRow]:
    summary = read_json(resolve_summary(path))
    report_path = resolve_summary(path)
    rows = [full_frame_row(summary, report_path)]
    current = summary.get("current_cap")
    recommendation = summary.get("recommendation")
    if isinstance(current, dict):
        rows.append(row_from_policy(summary, report_path, "candidate_cap", f"cap_{current.get('cap_label') or current.get('cap')}", current, "live"))
    if isinstance(recommendation, dict):
        suffix = f"cap_{recommendation.get('cap_label') or recommendation.get('cap')}_recommended"
        rows.append(row_from_policy(summary, report_path, "candidate_cap", suffix, recommendation, "diagnostic_recommendation"))
    return rows


def parse_section_sender(path: Path) -> list[StrategyRow]:
    summary = read_json(resolve_summary(path))
    report_path = resolve_summary(path)
    rows = [full_frame_row(summary, report_path)]
    cap_only = summary.get("cap_only")
    sender = summary.get("sender")
    if isinstance(cap_only, dict):
        rows.append(row_from_policy(summary, report_path, "section_sender", "cap_only", cap_only, "baseline_cap_only"))
    if isinstance(sender, dict):
        row = row_from_policy(summary, report_path, "section_sender", "ios_sender", sender, "live")
        row.crop_gt_recall = as_float(sender.get("combined_crop_gt_recall"), row.crop_gt_recall or 0.0)
        row.extra["section_only_gt_recall"] = sender.get("section_only_gt_recall")
        row.extra["total_pixels_delta_vs_cap_only"] = summary.get("total_pixels_delta_vs_cap_only")
        rows.append(row)
    return rows


def parse_section_sweep(path: Path) -> list[StrategyRow]:
    summary = read_json(resolve_summary(path))
    report_path = resolve_summary(path)
    rows = [full_frame_row(summary, report_path)]
    recommendation = summary.get("recommendation")
    if isinstance(recommendation, dict):
        strategy = recommendation.get("strategy_id") or "section_recommended"
        row = row_from_policy(summary, report_path, "section_crop_sweep", str(strategy), recommendation, "diagnostic_recommendation")
        row.crop_gt_recall = as_float(recommendation.get("section_gt_recall"))
        row.crop_pixels_vs_full = as_float(recommendation.get("section_pixels_vs_full"))
        rows.append(row)
    return rows


def load_rows(args: argparse.Namespace) -> list[StrategyRow]:
    rows: list[StrategyRow] = []
    for path in args.candidate_eval:
        rows.extend(parse_candidate_eval(path))
    for path in args.fallback_tune:
        rows.extend(parse_fallback_tune(path))
    for path in args.candidate_cap_sweep:
        rows.extend(parse_cap_sweep(path))
    for path in args.section_sender_eval:
        rows.extend(parse_section_sender(path))
    for path in args.section_crop_sweep:
        rows.extend(parse_section_sweep(path))
    return rows


def dedupe_rows(rows: list[StrategyRow]) -> list[StrategyRow]:
    deduped: list[StrategyRow] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = (
            row.strategy_id,
            row.family,
            row.replay,
            row.evidence_kind,
            row.status,
            row.crop_gt_recall,
            row.policy_gt_recall,
            row.fallback_image_ratio,
            row.total_pixels_vs_full,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def add_quality_notes(rows: list[StrategyRow], min_gt: int, min_recall: float, max_pixels: float) -> None:
    for row in rows:
        if row.family == "full_frame_baseline":
            continue
        if row.represented_gt_key_count < min_gt and row.evidence_kind != "synthetic_stress":
            row.recommendation_notes.append(f"Evidence is small: {row.represented_gt_key_count} represented reviewed GT keys.")
        if row.crop_gt_recall is not None and row.crop_gt_recall < min_recall:
            row.recommendation_notes.append(f"Crop recall below target {min_recall:.3f}.")
        if row.policy_gt_recall is not None and row.policy_gt_recall < min_recall:
            row.recommendation_notes.append(f"Fallback-protected recall below target {min_recall:.3f}.")
        if row.total_pixels_vs_full is not None and row.total_pixels_vs_full >= 1.0:
            row.recommendation_notes.append("Costs at least as much VLM pixels as full-frame baseline.")
        if row.total_pixels_vs_full is not None and row.total_pixels_vs_full > max_pixels:
            row.recommendation_notes.append(f"Above target total pixel ratio {max_pixels:.3f}.")
        if row.skipped_strong_candidate_count:
            row.recommendation_notes.append("Skips strong candidates; needs section or fallback protection.")
        if row.skipped_confident_candidate_count:
            row.recommendation_notes.append("Skips high-confidence candidates; needs section or fallback protection.")
        if not row.recommendation_notes:
            row.recommendation_notes.append("Passes configured recall/cost screen for this evidence set.")


def comparable_group(row: StrategyRow) -> str:
    return f"{row.evidence_kind}:{row.replay}"


def dominates(left: StrategyRow, right: StrategyRow) -> bool:
    if comparable_group(left) != comparable_group(right):
        return False
    left_recall = left.policy_gt_recall if left.policy_gt_recall is not None else left.crop_gt_recall
    right_recall = right.policy_gt_recall if right.policy_gt_recall is not None else right.crop_gt_recall
    left_pixels = left.total_pixels_vs_full
    right_pixels = right.total_pixels_vs_full
    if left_recall is None or right_recall is None or left_pixels is None or right_pixels is None:
        return False
    left_fallback = left.fallback_image_ratio if left.fallback_image_ratio is not None else 1.0
    right_fallback = right.fallback_image_ratio if right.fallback_image_ratio is not None else 1.0
    at_least_as_good = left_recall >= right_recall and left_pixels <= right_pixels and left_fallback <= right_fallback
    strictly_better = left_recall > right_recall or left_pixels < right_pixels or left_fallback < right_fallback
    return at_least_as_good and strictly_better


def pareto_flags(rows: list[StrategyRow]) -> dict[int, bool]:
    flags: dict[int, bool] = {}
    for index, row in enumerate(rows):
        flags[index] = not any(dominates(other, row) for other in rows if other is not row)
    return flags


def rank_rows(rows: list[StrategyRow], min_recall: float) -> list[StrategyRow]:
    def score(row: StrategyRow) -> tuple[int, float, float, float, int]:
        recall = row.policy_gt_recall if row.policy_gt_recall is not None else row.crop_gt_recall
        pixels = row.total_pixels_vs_full if row.total_pixels_vs_full is not None else 99.0
        fallback = row.fallback_image_ratio if row.fallback_image_ratio is not None else 1.0
        recall_ok = 1 if recall is not None and recall >= min_recall else 0
        live_bonus = 1 if row.status == "live" else 0
        return (-recall_ok, pixels, fallback, -(recall or 0.0), -live_bonus)

    return sorted(rows, key=score)


def summarize(rows: list[StrategyRow], args: argparse.Namespace) -> dict[str, Any]:
    rows = dedupe_rows(rows)
    add_quality_notes(rows, args.min_represented_gt_keys, args.min_recall, args.max_total_pixels_vs_full)
    row_dicts = [row.to_dict() for row in rows]
    flags = pareto_flags(rows)
    for index, item in enumerate(row_dicts):
        item["pareto_within_replay"] = flags.get(index, False)
    ranked = rank_rows(rows, args.min_recall)
    reviewed_live_rows = [
        row
        for row in ranked
        if row.status == "live" and row.family != "full_frame_baseline" and row.evidence_kind != "synthetic_stress"
    ]
    live_rows = [row for row in ranked if row.status == "live" and row.family != "full_frame_baseline"]
    diagnostic_rows = [row for row in ranked if row.status in {"diagnostic_recommendation", "candidate_set"}]
    synthetic_rows = [row for row in ranked if row.evidence_kind == "synthetic_stress"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "min_represented_gt_keys": args.min_represented_gt_keys,
            "min_recall": args.min_recall,
            "max_total_pixels_vs_full": args.max_total_pixels_vs_full,
        },
        "row_count": len(row_dicts),
        "rows": row_dicts,
        "best_reviewed_live_by_cost": [row.to_dict() for row in reviewed_live_rows[:5]],
        "best_live_by_cost": [row.to_dict() for row in live_rows[:5]],
        "best_diagnostic_by_cost": [row.to_dict() for row in diagnostic_rows[:5]],
        "synthetic_stress_rows": [row.to_dict() for row in synthetic_rows[:8]],
        "notes": [
            "Rows are comparable only within the same replay/evidence_kind group.",
            "Use best_reviewed_live_by_cost for production decisions; synthetic stress rows are negative gates and must not count as release training evidence.",
            "Full-frame rows are baselines: they model VLM cost, not a local detector strategy.",
        ],
    }


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def markdown(report: dict[str, Any]) -> str:
    rows = report.get("rows") if isinstance(report.get("rows"), list) else []
    lines = [
        "# Observation Strategy Selection",
        "",
        f"- Generated: {report.get('generated_at')}",
        f"- Rows: {report.get('row_count')}",
        "",
        "## Best Reviewed Live Rows",
        "",
        "| replay | strategy | family | recall | fallback | pixels/full | notes |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in report.get("best_reviewed_live_by_cost") or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row.get("replay")),
                    fmt(row.get("strategy_id")),
                    fmt(row.get("family")),
                    fmt(row.get("policy_gt_recall") if row.get("policy_gt_recall") is not None else row.get("crop_gt_recall")),
                    fmt(row.get("fallback_image_ratio")),
                    fmt(row.get("total_pixels_vs_full")),
                    "; ".join(row.get("recommendation_notes") or []),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Synthetic Stress Rows",
            "",
            "| replay | strategy | family | status | recall | fallback | pixels/full | notes |",
            "|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in report.get("synthetic_stress_rows") or []:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row.get("replay")),
                    fmt(row.get("strategy_id")),
                    fmt(row.get("family")),
                    fmt(row.get("status")),
                    fmt(row.get("policy_gt_recall") if row.get("policy_gt_recall") is not None else row.get("crop_gt_recall")),
                    fmt(row.get("fallback_image_ratio")),
                    fmt(row.get("total_pixels_vs_full")),
                    "; ".join(row.get("recommendation_notes") or []),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Pareto Rows",
            "",
            "| replay | strategy | family | status | recall | fallback | pixels/full | evidence |",
            "|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        if not row.get("pareto_within_replay"):
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row.get("replay")),
                    fmt(row.get("strategy_id")),
                    fmt(row.get("family")),
                    fmt(row.get("status")),
                    fmt(row.get("policy_gt_recall") if row.get("policy_gt_recall") is not None else row.get("crop_gt_recall")),
                    fmt(row.get("fallback_image_ratio")),
                    fmt(row.get("total_pixels_vs_full")),
                    fmt(row.get("evidence_kind")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Notes", ""])
    for note in report.get("notes") or []:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args)
    report = summarize(rows, args)
    write_json(args.out / "strategy_selection.json", report)
    write_jsonl(args.out / "strategy_rows.jsonl", [row for row in report["rows"]])
    (args.out / "strategy_selection.md").write_text(markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate crop/section/fallback strategy reports into one Pareto table.")
    parser.add_argument("--candidate-eval", type=Path, action="append", default=[])
    parser.add_argument("--fallback-tune", type=Path, action="append", default=[])
    parser.add_argument("--candidate-cap-sweep", type=Path, action="append", default=[])
    parser.add_argument("--section-sender-eval", type=Path, action="append", default=[])
    parser.add_argument("--section-crop-sweep", type=Path, action="append", default=[])
    parser.add_argument("--min-represented-gt-keys", type=int, default=10)
    parser.add_argument("--min-recall", type=float, default=0.999)
    parser.add_argument("--max-total-pixels-vs-full", type=float, default=0.90)
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-observation-strategy-selection"))
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "rows": report["row_count"],
                "out": str(args.out),
                "best_reviewed_live_by_cost": [
                    {
                        "strategy_id": row["strategy_id"],
                        "family": row["family"],
                        "total_pixels_vs_full": row["total_pixels_vs_full"],
                        "policy_gt_recall": row["policy_gt_recall"],
                    }
                    for row in report["best_reviewed_live_by_cost"][:5]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
