"""Smoke-test backend question full-frame fallback policy boundaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.main import (  # noqa: E402
    QUESTION_FULL_FRAME_FALLBACK_POLICY,
    _question_full_frame_fallback_reason,
)
from scripts.question_observation_fallback_tune import current_backend_policy  # noqa: E402


def assert_reason(name: str, stats: dict[str, Any], expected: str) -> dict[str, Any]:
    actual = _question_full_frame_fallback_reason(stats)
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")
    return {"name": name, "reason": actual}


def main() -> None:
    tuning_policy = current_backend_policy()
    for key, value in QUESTION_FULL_FRAME_FALLBACK_POLICY.items():
        if tuning_policy.get(key) != value:
            raise AssertionError(f"policy mismatch for {key}: backend={value!r}, tuning={tuning_policy.get(key)!r}")

    cases = [
        assert_reason("missing stats", {}, "no_crops"),
        assert_reason("zero crops", {"count": 0}, "no_crops"),
        assert_reason("generation errors", {"count": 2, "error_count": 2}, "all_crop_generation_errors"),
        assert_reason(
            "old cap telemetry remains conservative",
            {"count": 12, "area": 0.5, "max_area": 0.05, "limited_count": 1, "ranked_limit_telemetry_count": 0},
            "limited_candidate_frame",
        ),
        assert_reason(
            "new cap skipped strong remains conservative",
            {
                "count": 12,
                "area": 0.5,
                "max_area": 0.05,
                "limited_count": 1,
                "ranked_limit_telemetry_count": 1,
                "limited_strong_count": 1,
            },
            "limited_candidate_frame",
        ),
        assert_reason(
            "section protects covered cap risk",
            {
                "count": 1,
                "area": 0.01,
                "max_area": 0.01,
                "limited_count": 1,
                "limited_unique_count": 1,
                "ranked_limit_telemetry_count": 1,
                "limited_strong_count": 1,
                "section_count": 1,
                "section_covered_question_count": 1,
            },
            "",
        ),
        assert_reason("single low coverage", {"count": 1, "area": 0.29, "max_area": 0.29}, "single_low_coverage_crop"),
        assert_reason("single threshold pass", {"count": 1, "area": 0.30, "max_area": 0.30}, ""),
        assert_reason("sparse low coverage", {"count": 2, "area": 0.24, "max_area": 0.18}, "sparse_low_coverage_crops"),
        assert_reason("tiny coverage", {"count": 3, "area": 0.11, "max_area": 0.05}, "tiny_crop_coverage"),
        assert_reason("no large crop", {"count": 3, "area": 0.35, "max_area": 0.11}, "no_large_question_crop"),
        assert_reason(
            "all weak alone no longer falls back",
            {"count": 3, "area": 0.39, "max_area": 0.32, "weak_count": 3, "low_conf_count": 1},
            "",
        ),
        assert_reason(
            "low confidence alone no longer falls back",
            {"count": 3, "area": 0.45, "max_area": 0.20, "low_conf_count": 3},
            "",
        ),
    ]
    print(json.dumps({"passed": True, "cases": cases}, ensure_ascii=False))


if __name__ == "__main__":
    main()
