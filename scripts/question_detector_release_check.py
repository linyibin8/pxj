"""Release-readiness gate for the on-device question detector.

This checker is intentionally stricter than the smoke/evaluation helpers. It
answers the product question: is this detector safe to bundle into the iOS app
as QuestionRegionDetector, with thresholds aligned and payload/latency evidence
showing that rect-only crop upload is still the right strategy?
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_BASENAME = "QuestionRegionDetector"
MODEL_SUFFIXES = (".mlmodelc", ".mlmodel", ".mlpackage")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_text_flexible(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def resolve_file(path: Path, default_name: str) -> Path:
    if path.is_dir():
        return path / default_name
    return path


def resolve_existing_path(raw: str, anchors: list[Path]) -> Path | None:
    if not raw:
        return None
    value = Path(raw)
    candidates = [value]
    for anchor in anchors:
        candidates.append(anchor / value)
        candidates.append(anchor.parent / value)
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def metric_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def replay_fingerprints(data: dict[str, Any]) -> list[dict[str, Any]]:
    replays = data.get("replays") if isinstance(data.get("replays"), list) else []
    fingerprints: list[dict[str, Any]] = []
    for item in replays:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").replace("\\", "/").strip()
        fingerprints.append(
            {
                "path": path,
                "image_count": int(item.get("image_count") or data.get("image_count") or 0),
                "candidate_count": int(item.get("candidate_count") or data.get("candidate_count") or 0),
                "full_frame_resized_pixels": int(item.get("full_frame_resized_pixels") or data.get("full_frame_all_pixels") or 0),
            }
        )
    if not fingerprints:
        fingerprints.append(
            {
                "path": "",
                "image_count": int(data.get("image_count") or 0),
                "candidate_count": int(data.get("candidate_count") or 0),
                "full_frame_resized_pixels": int(data.get("full_frame_all_pixels") or 0),
            }
        )
    return fingerprints


def replay_fingerprint_key(replay: dict[str, Any]) -> tuple[Any, ...]:
    return (
        replay.get("path") or "",
        replay.get("image_count") or 0,
        replay.get("candidate_count") or 0,
        replay.get("full_frame_resized_pixels") or 0,
    )


def replay_fingerprint_set(summary: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    replays = summary.get("replays") if isinstance(summary.get("replays"), list) else []
    return tuple(sorted(replay_fingerprint_key(replay) for replay in replays if isinstance(replay, dict)))


def ground_truth_manifest_sources(data: dict[str, Any]) -> list[str]:
    manifests = data.get("ground_truth_manifests") if isinstance(data.get("ground_truth_manifests"), list) else []
    return sorted(str(item).replace("\\", "/").strip() for item in manifests if str(item or "").strip())


def add_check(
    checks: list[dict[str, Any]],
    category: str,
    name: str,
    passed: bool,
    detail: str,
    severity: str = "hard",
    values: dict[str, Any] | None = None,
) -> None:
    checks.append(
        {
            "category": category,
            "name": name,
            "passed": bool(passed),
            "severity": severity,
            "detail": detail,
            "values": values or {},
        }
    )


def summarize_dataset(path: Path, data: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    readiness = data.get("readiness") if isinstance(data.get("readiness"), dict) else {}
    counts = data.get("counts") if isinstance(data.get("counts"), dict) else {}
    warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
    summary = {
        "path": str(path),
        "pilot_eval_ready": bool(readiness.get("pilot_eval_ready")),
        "model_training_ready": bool(readiness.get("model_training_ready")),
        "has_error": bool(readiness.get("has_error")),
        "source_images": int(counts.get("source_images") or 0),
        "positive_images": int(counts.get("positive_images") or 0),
        "negative_images": int(counts.get("negative_images") or 0),
        "annotations": int(counts.get("annotations") or 0),
        "split_groups": int(counts.get("split_groups") or 0),
        "warning_count": len(warnings),
    }
    add_check(
        checks,
        "dataset",
        f"{path}: model_training_ready",
        summary["model_training_ready"],
        "Dataset audit must report model_training_ready=true before detector selection is considered releasable.",
        values=summary,
    )
    add_check(
        checks,
        "dataset",
        f"{path}: no audit error",
        not summary["has_error"],
        "Dataset audit must not contain blocking data errors.",
        values={"has_error": summary["has_error"]},
    )
    if warnings:
        add_check(
            checks,
            "dataset",
            f"{path}: audit warnings",
            False,
            f"Dataset audit contains {len(warnings)} warning(s); inspect before relying on the release report.",
            "warning",
            {"warnings": warnings[:5]},
        )
    return summary


def selected_run_manifests(selection: dict[str, Any], selection_path: Path) -> list[Path]:
    best = selection.get("best") if isinstance(selection.get("best"), dict) else {}
    runs = best.get("runs") if isinstance(best.get("runs"), list) else []
    manifests: list[Path] = []
    seen: set[str] = set()
    for run in runs:
        if not isinstance(run, dict):
            continue
        resolved = resolve_existing_path(str(run.get("manifest") or ""), [selection_path.parent, Path.cwd()])
        if resolved is None:
            continue
        key = str(resolved.resolve())
        if key not in seen:
            seen.add(key)
            manifests.append(resolved)
    return manifests


def discover_train_manifests(paths: list[Path]) -> list[Path]:
    manifests: list[Path] = []
    for path in paths:
        if path.is_file():
            manifests.append(path)
            continue
        direct = path / "train_manifest.json"
        if direct.is_file():
            manifests.append(direct)
        if path.is_dir():
            manifests.extend(sorted(path.rglob("train_manifest.json")))
    unique: dict[str, Path] = {}
    for path in manifests:
        unique[str(path.resolve())] = path
    return sorted(unique.values())


def artifact_candidates_from_selection(selection: dict[str, Any], selection_path: Path) -> list[Path]:
    candidates: list[Path] = []
    best = selection.get("best") if isinstance(selection.get("best"), dict) else {}
    runs = best.get("runs") if isinstance(best.get("runs"), list) else []
    anchors = [selection_path.parent, Path.cwd()]
    for run in runs:
        if not isinstance(run, dict):
            continue
        resolved = resolve_existing_path(str(run.get("coreml_artifact") or ""), anchors)
        if resolved is not None:
            candidates.append(resolved)
    return candidates


def artifact_candidates_from_manifests(manifest_paths: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for path in manifest_paths:
        try:
            manifest = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        resolved = resolve_existing_path(str(manifest.get("coreml_artifact") or ""), [path.parent, Path.cwd()])
        if resolved is not None:
            candidates.append(resolved)
    unique: dict[str, Path] = {}
    for path in candidates:
        unique[str(path.resolve())] = path
    return sorted(unique.values())


def ios_model_resources(ios_root: Path) -> list[Path]:
    resources: list[Path] = []
    if not ios_root.exists():
        return resources
    for suffix in MODEL_SUFFIXES:
        resources.extend(sorted(ios_root.rglob(f"{MODEL_BASENAME}{suffix}")))
    unique: dict[str, Path] = {}
    for path in resources:
        unique[str(path.resolve())] = path
    return sorted(unique.values())


def parse_swift_detector_config(ios_root: Path) -> dict[str, Any]:
    swift_path = ios_root / "QuestionSegmenter.swift"
    result: dict[str, Any] = {
        "path": str(swift_path),
        "exists": swift_path.is_file(),
        "bundle_lookup_mlmodelc": False,
        "fast_min_confidence": None,
        "accurate_min_confidence": None,
        "detector_first_requires_matched_text": False,
        "detector_fast_path_requires_completion": False,
        "detector_merges_unmatched_ocr": False,
        "detector_no_early_first_return": False,
        "detector_textless_fallback_telemetry": False,
        "detector_reuses_first_pass_boxes_after_fallback": False,
    }
    if not swift_path.is_file():
        return result
    text = swift_path.read_text(encoding="utf-8")
    result["bundle_lookup_mlmodelc"] = (
        'forResource: "QuestionRegionDetector"' in text and 'withExtension: "mlmodelc"' in text
    )
    result["detector_first_requires_matched_text"] = bool(
        re.search(r"if\s+detector\.matchedRegionCount\s*>\s*0\s*\{[\s\S]{0,900}coreml_v4_detector_fast_complete", text)
    )
    result["detector_fast_path_requires_completion"] = bool(
        re.search(r"if\s+fast,\s*detectorFastPathLooksComplete\(detector\.regions\)", text)
        and "detectorFastPathLooksComplete" in text
    )
    result["detector_merges_unmatched_ocr"] = (
        "mergeDetectorAndOCRRegions" in text
        and "overlapsSameQuestion" in text
        and "coreml_v4_ocr_v3_merged" in text
    )
    result["detector_no_early_first_return"] = "coreml_v4_detector_first" not in text
    result["detector_textless_fallback_telemetry"] = (
        "coreMLTextlessFallback" in text and '"coreml_textless_fallback"' in text
    )
    result["detector_reuses_first_pass_boxes_after_fallback"] = bool(
        re.search(r"detectorRegions\(for:\s*detector\.boxes,\s*lines:\s*lines,\s*allowTextless:\s*false\)", text)
        and "detector_boxes_then_full_page" in text
    )
    for key, pattern in {
        "fast_min_confidence": r"detectorFastMinConfidence\s*:[^=]+=\s*([0-9.]+)",
        "accurate_min_confidence": r"detectorAccurateMinConfidence\s*:[^=]+=\s*([0-9.]+)",
    }.items():
        match = re.search(pattern, text)
        if match:
            try:
                result[key] = float(match.group(1))
            except ValueError:
                result[key] = None
    return result


def parse_swift_observation_dedupe_config(ios_root: Path) -> dict[str, Any]:
    swift_path = ios_root / "ContentView.swift"
    result: dict[str, Any] = {
        "path": str(swift_path),
        "exists": swift_path.is_file(),
        "weak_text_part_function": False,
        "weak_text_compatible_function": False,
        "weak_text_guard_in_signature_similarity": False,
        "notext_is_not_compatible": False,
        "uses_near_text_similarity": False,
        "candidate_rank_function": False,
        "upload_uses_ranked_candidates": False,
        "rank_mode_telemetry": False,
        "limited_risk_telemetry": False,
        "no_raw_prefix_upload": False,
        "pixel_resize_helper": False,
        "vision_resize_uses_pixel_helper": False,
        "jpeg_uses_pixel_helper": False,
        "pixel_renderer_scale_one": False,
        "section_crop_function": False,
        "section_crop_manifest_fields": False,
        "section_crop_risk_gated": False,
        "section_companion_top2": False,
        "risk_section_function": False,
        "risk_section_top1_companion": False,
        "section_crop_metrics": False,
        "empty_section_dedupe_state": False,
        "empty_section_duplicate_guard": False,
        "empty_section_signature_persisted": False,
    }
    if not swift_path.is_file():
        return result
    text = swift_path.read_text(encoding="utf-8")
    result["weak_text_part_function"] = "weakObservationQuestionTextPart" in text
    result["weak_text_compatible_function"] = "weakObservationQuestionTextCompatible" in text
    result["weak_text_guard_in_signature_similarity"] = bool(
        re.search(
            r"guard\s+weakObservationQuestionTextCompatible\(\s*lhs\.key\s*,\s*rhs\.key\s*\)\s+else\s*\{\s*return\s+false\s*\}",
            text,
        )
    )
    result["notext_is_not_compatible"] = '"notext"' in text and 'return text == "notext" ? "" : text' in text
    result["uses_near_text_similarity"] = bool(
        re.search(
            r"weakObservationQuestionTextCompatible[\s\S]+questionKeyNearSimilar\(\s*left\s*,\s*right\s*\)",
            text,
        )
    )
    result["candidate_rank_function"] = "rankedObservationQuestionCandidates" in text
    result["upload_uses_ranked_candidates"] = (
        "let rankedCandidates = Self.rankedObservationQuestionCandidates" in text
        and "for candidate in uploadSelectedCandidates" in text
    )
    result["rank_mode_telemetry"] = '"candidate_rank_mode"' in text and '"candidateRankMode"' in text
    result["limited_risk_telemetry"] = (
        '"frame_limited_unique_candidate_count"' in text
        and '"frame_limited_strong_candidate_count"' in text
        and '"frame_limited_confident_candidate_count"' in text
        and '"frame_limited_max_confidence"' in text
    )
    result["no_raw_prefix_upload"] = not bool(
        re.search(r"for\s+candidate\s+in\s+candidates\.prefix\(\s*candidateLimitPerFrame\s*\)", text)
    )
    result["pixel_resize_helper"] = "resizedToPixelMaxSide" in text and "cgImage.map" in text
    result["vision_resize_uses_pixel_helper"] = bool(
        re.search(r"func\s+resizedForVision\(maxSide:\s*CGFloat\)\s*->\s*UIImage\s*\{\s*resizedToPixelMaxSide\(maxSide:\s*maxSide\)", text)
    )
    result["jpeg_uses_pixel_helper"] = "let resized = image.resizedToPixelMaxSide(maxSide: maxSide)" in text
    result["pixel_renderer_scale_one"] = "format.scale = 1" in text
    result["section_crop_function"] = (
        "observationQuestionSectionCrops" in text
        and "ios_page_dense_v1" in text
        and "observationQuestionRiskFrameSectionCrops" in text
        and "ios_page_risk_v1" in text
        and "observationQuestionEmptyFrameSectionCrops" in text
        and "ios_page_empty_v1" in text
        and "no_candidate_study_frame" in text
    )
    result["section_crop_manifest_fields"] = (
        '"crop_kind": "section"' in text
        and '"question_key_strength": "section_crop"' in text
        and '"covered_question_count"' in text
        and '"covered_subrects"' in text
        and '"source": "ios-observation-section-rect"' in text
    )
    result["section_crop_risk_gated"] = (
        "guard rankedCandidates.count > candidateLimit" in text
        and "guard !skippedUniqueCandidates.isEmpty" in text
        and "guard candidates.isEmpty else { return [] }" in text
        and "guard hasStudyMaterial else { return [] }" in text
        and "uploadSelectedCandidates" in text
    )
    result["section_companion_top2"] = (
        "riskFrameSectionCrops.isEmpty ? 2 : 1" in text
        and "Array(selectedCandidates.prefix(sectionCompanionCandidateLimit))" in text
        and '"section_companion_candidate_limit"' in text
        and '"section_companion_mode"' in text
    )
    result["risk_section_function"] = (
        "observationQuestionRiskFrameSectionCrops" in text
        and "observationQuestionRiskFrameSectionRect" in text
        and "ios_page_risk_v1" in text
        and "weak_low_coverage_frame" in text
        and "guard skippedUniqueCandidates.isEmpty else { return [] }" in text
        and "rankedCandidates.count <= min(4, candidateLimit)" in text
    )
    result["risk_section_top1_companion"] = "riskFrameSectionCrops.isEmpty ? 2 : 1" in text
    result["section_crop_metrics"] = '"section_crop_count"' in text and '"section_frame_count"' in text
    result["empty_section_dedupe_state"] = (
        "observationQuestionUploadedSectionSignatures" in text
        and "uploadedSectionSignatures" in text
        and "knownSectionSignatures" in text
        and "newSectionSignatures" in text
    )
    result["empty_section_duplicate_guard"] = bool(
        re.search(
            r"if\s+section\.candidates\.isEmpty\s*,\s*[\s\S]{0,220}"
            r"knownSectionSignatures\.contains[\s\S]{0,220}continue",
            text,
        )
    )
    result["empty_section_signature_persisted"] = (
        "newSectionSignatures.append(sectionSignature)" in text
        and "observationQuestionUploadedSectionSignatures.append(contentsOf: cropUpload.newSectionSignatures)" in text
        and "observationQuestionUploadedSectionSignatures.removeAll()" in text
    )
    return result


def parse_backend_question_source_binding_config(backend_main: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(backend_main),
        "exists": backend_main.is_file(),
        "parser_reads_input_index": False,
        "parser_accepts_source_index_alias": False,
        "batch_sources_assign_input_index": False,
        "batch_prompt_requests_input_index": False,
        "matcher_prefers_input_index": False,
        "source_meta_persists_input_index": False,
        "apply_source_meta_after_match": False,
        "persists_candidate_cap_risk_telemetry": False,
        "uses_ranked_limit_telemetry_guard": False,
        "uses_limited_strong_confident_guard": False,
        "weak_layout_key_ignored_for_crop_source_dedupe": False,
        "parses_section_crop_kind": False,
        "section_crop_key_is_weak_for_source_dedupe": False,
        "persists_section_crop_telemetry": False,
        "source_meta_persists_section_crop_kind": False,
        "normalizes_section_extraction_sources": False,
        "section_prompt_allows_multiple_questions": False,
        "section_crop_protects_limited_fallback": False,
    }
    if not backend_main.is_file():
        return result
    text = backend_main.read_text(encoding="utf-8")
    result["parser_reads_input_index"] = '"input_index"' in text and '"inputIndex"' in text
    result["parser_accepts_source_index_alias"] = '"source_index"' in text and '"sourceIndex"' in text
    result["batch_sources_assign_input_index"] = bool(
        re.search(r"sources\s*=\s*\[\{\*\*source,\s*\"input_index\":\s*index\s*\+\s*1\}", text)
    )
    result["batch_prompt_requests_input_index"] = "Return input_index for every question" in text
    result["matcher_prefers_input_index"] = bool(
        re.search(
            r"def\s+_best_source_for_extracted_question[\s\S]+if\s+1\s*<=\s*input_index\s*<=\s*len\(sources\):\s*[\r\n]+\s*return\s+sources\[input_index\s*-\s*1\]",
            text,
        )
    )
    result["source_meta_persists_input_index"] = '"source_input_index"' in text
    result["apply_source_meta_after_match"] = bool(
        re.search(
            r"matched_source\s*=\s*_best_source_for_extracted_question[\s\S]+_apply_question_source_meta\(\s*q\s*,\s*matched_source",
            text,
        )
    )
    result["persists_candidate_cap_risk_telemetry"] = (
        '"candidate_rank_mode"' in text
        and '"frame_limited_unique_candidate_count"' in text
        and '"frame_limited_strong_candidate_count"' in text
        and '"frame_limited_confident_candidate_count"' in text
        and '"frame_limited_max_confidence"' in text
    )
    result["uses_ranked_limit_telemetry_guard"] = (
        "ranked_limit_telemetry_count" in text
        and re.search(r"limited_count\s*>\s*0\s+and[\s\S]{0,120}\(", text) is not None
    )
    result["uses_limited_strong_confident_guard"] = "limited_strong_count > 0" in text and "limited_confident_count > 0" in text
    result["weak_layout_key_ignored_for_crop_source_dedupe"] = bool(
        re.search(r"weak_question_key\s*=\s*question_crop_key_is_weak\(question_key,\s*question_key_strength\)", text)
        and re.search(
            r"dedupe_key\s*=\s*\([\s\S]{0,400}\(\"\"\s+if\s+weak_question_key\s+else\s+question_key\)[\s\S]{0,400}\(\"\"\s+if\s+weak_question_key\s+else\s+item\.get\(\"fingerprint\"\)\)",
            text,
        )
    )
    result["parses_section_crop_kind"] = "def question_crop_kind" in text and '"crop_kind"' in text and '"cropKind"' in text
    result["section_crop_key_is_weak_for_source_dedupe"] = '"section_crop"' in text and 'key.startswith("section:")' in text
    result["persists_section_crop_telemetry"] = (
        '"section_key"' in text
        and '"covered_question_count"' in text
        and '"covered_subrects"' in text
        and "question_crop_section_metadata" in text
        and "client_telemetry[meta_key]" in text
    )
    result["source_meta_persists_section_crop_kind"] = '"source_crop_kind"' in text and '"source_section_key"' in text
    result["normalizes_section_extraction_sources"] = "question_crop_section_metadata_from_telemetry" in text and '"crop_kind": crop_kind' in text
    result["section_prompt_allows_multiple_questions"] = "Some inputs are section crops and may contain multiple questions" in text
    result["section_crop_protects_limited_fallback"] = (
        "section_limited_protection" in text
        and "section_fallback_protection" in text
        and '"section_covered_question_count"' in text
        and '"section_union_area"' in text
        and "not section_limited_protection" in text
        and re.search(r"if\s+section_fallback_protection:\s*[\r\n]+\s*return\s+\"\"", text) is not None
    )
    return result


def compare_float(actual: Any, expected: Any, tolerance: float) -> bool:
    if actual is None or expected is None:
        return False
    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def summarize_selection(
    path: Path,
    data: dict[str, Any],
    checks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    best = data.get("best") if isinstance(data.get("best"), dict) else {}
    ios_rec = data.get("ios_recommendation") if isinstance(data.get("ios_recommendation"), dict) else {}
    summary = {
        "path": str(path),
        "manifest_count": int(data.get("manifest_count") or 0),
        "candidate_count": int(data.get("candidate_count") or 0),
        "aggregated_candidate_count": int(data.get("aggregated_candidate_count") or 0),
        "has_best": bool(best),
        "fold_count": int(best.get("fold_count") or 0),
        "all_folds_pass_eval_gate": bool(best.get("all_folds_pass_eval_gate")),
        "min_recall": metric_float(best, "min_recall"),
        "mean_precision": metric_float(best, "mean_precision"),
        "max_fp_per_image": metric_float(best, "max_fp_per_image"),
        "min_score": metric_float(best, "min_score"),
        "ios_recommendation": ios_rec,
    }
    add_check(
        checks,
        "selection",
        "selection has best candidate",
        summary["has_best"],
        "Model selection must produce a best candidate.",
        values=summary,
    )
    add_check(
        checks,
        "selection",
        "cross-validation fold count",
        summary["fold_count"] >= args.min_folds,
        f"Selected candidate must aggregate at least {args.min_folds} folds.",
        values={"fold_count": summary["fold_count"], "min_folds": args.min_folds},
    )
    add_check(
        checks,
        "selection",
        "all selected folds pass eval gate",
        summary["all_folds_pass_eval_gate"],
        "Every selected fold must pass the detector eval gate.",
        values={"all_folds_pass_eval_gate": summary["all_folds_pass_eval_gate"]},
    )
    add_check(
        checks,
        "selection",
        "minimum recall",
        summary["min_recall"] >= args.min_recall,
        f"Worst-fold recall must be >= {args.min_recall:.3f}.",
        values={"min_recall": summary["min_recall"]},
    )
    add_check(
        checks,
        "selection",
        "mean precision",
        summary["mean_precision"] >= args.min_precision,
        f"Mean precision must be >= {args.min_precision:.3f}.",
        values={"mean_precision": summary["mean_precision"]},
    )
    add_check(
        checks,
        "selection",
        "max false positives per image",
        summary["max_fp_per_image"] <= args.max_fp_per_image,
        f"Max false positives per image must be <= {args.max_fp_per_image:.3f}.",
        values={"max_fp_per_image": summary["max_fp_per_image"]},
    )
    add_check(
        checks,
        "selection",
        "iOS threshold recommendation",
        bool(ios_rec),
        "Selection must include iOS detector confidence recommendations.",
        values=ios_rec,
    )
    return summary


def summarize_train_manifests(manifest_paths: list[Path], checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in manifest_paths:
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            add_check(
                checks,
                "training",
                f"{path}: readable manifest",
                False,
                f"Could not read train_manifest.json: {exc}",
            )
            continue
        readiness = data.get("audit_readiness") if isinstance(data.get("audit_readiness"), dict) else {}
        summary = {
            "path": str(path),
            "status": data.get("status") or "",
            "model": data.get("model") or "",
            "name": data.get("name") or "",
            "dataset": data.get("dataset") or "",
            "audit_model_training_ready": bool(readiness.get("model_training_ready")),
            "coreml_artifact": data.get("coreml_artifact") or "",
        }
        summaries.append(summary)
        add_check(
            checks,
            "training",
            f"{path}: trained status",
            str(summary["status"]) in {"trained", "eval_passed", "exported"},
            "Selected train manifest must represent a completed training/evaluation run.",
            values=summary,
        )
        add_check(
            checks,
            "training",
            f"{path}: ready source dataset",
            summary["audit_model_training_ready"],
            "Selected train manifest must come from an audit-ready dataset, not a smoke or unready dataset.",
            values=summary,
        )
    return summaries


def summarize_replay(
    path: Path,
    data: dict[str, Any],
    checks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    counts = data.get("counts") if isinstance(data.get("counts"), dict) else {}
    upload = data.get("upload_bytes") if isinstance(data.get("upload_bytes"), dict) else {}
    vlm = data.get("vlm_proxy") if isinstance(data.get("vlm_proxy"), dict) else {}
    layout = data.get("layout_measurement") if isinstance(data.get("layout_measurement"), dict) else {}
    layout_latency = layout.get("latency_ms") if isinstance(layout.get("latency_ms"), dict) else {}
    summary = {
        "path": str(path),
        "images": int(counts.get("images") or 0),
        "cross_frame_dedup_boxes": int(counts.get("cross_frame_dedup_boxes") or 0),
        "rect_only_cross_frame_dedup_vs_full": metric_float(upload, "rect_only_cross_frame_dedup_vs_full", 999.0),
        "full_plus_cross_frame_dedup_crop_vs_full": metric_float(upload, "full_plus_cross_frame_dedup_crop_vs_full", 999.0),
        "crop_pixels_vs_full_pixels": metric_float(vlm, "crop_pixels_vs_full_pixels", 999.0),
        "layout_latency_median_ms": metric_float(layout_latency, "median", 999.0),
        "layout_latency_p95_ms": metric_float(layout_latency, "p95", 999.0),
        "layout_attempted_images": int(layout.get("attempted_images") or 0),
    }
    add_check(
        checks,
        "replay",
        f"{path}: image coverage",
        summary["images"] >= args.min_replay_images,
        f"Replay evidence must include at least {args.min_replay_images} images.",
        values={"images": summary["images"]},
    )
    add_check(
        checks,
        "replay",
        f"{path}: rect-only network overhead",
        summary["rect_only_cross_frame_dedup_vs_full"] <= args.max_rect_overhead_ratio,
        f"Rect-only manifest strategy must stay <= {args.max_rect_overhead_ratio:.4f}x full-frame bytes.",
        values={"ratio": summary["rect_only_cross_frame_dedup_vs_full"]},
    )
    add_check(
        checks,
        "replay",
        f"{path}: crop VLM pixel ratio",
        summary["crop_pixels_vs_full_pixels"] <= args.max_crop_pixel_ratio,
        f"Deduped crop pixels must stay <= {args.max_crop_pixel_ratio:.3f} of full-frame VLM pixels.",
        values={"ratio": summary["crop_pixels_vs_full_pixels"]},
    )
    if summary["layout_attempted_images"]:
        add_check(
            checks,
            "replay",
            f"{path}: local layout median latency",
            summary["layout_latency_median_ms"] <= args.max_layout_median_ms,
            f"Measured local layout median must be <= {args.max_layout_median_ms:.1f} ms.",
            values={"median_ms": summary["layout_latency_median_ms"], "p95_ms": summary["layout_latency_p95_ms"]},
        )
    else:
        add_check(
            checks,
            "replay",
            f"{path}: measured local layout latency",
            False,
            "Replay summary does not contain measured local layout latency.",
            "warning",
        )
    return summary


def summarize_dedupe_tune(
    path: Path,
    data: dict[str, Any],
    checks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    gt = data.get("gt") if isinstance(data.get("gt"), dict) else {}
    recommendation = data.get("recommendation") if isinstance(data.get("recommendation"), dict) else {}
    baselines = data.get("baselines") if isinstance(data.get("baselines"), list) else []
    swift_text_guard = next(
        (
            item
            for item in baselines
            if isinstance(item, dict)
            and item.get("strategy") == "swift_signature"
            and isinstance(item.get("params"), dict)
            and item["params"].get("weak_text_guard") is True
        ),
        {},
    )
    summary = {
        "path": str(path),
        "candidate_count": int(data.get("candidate_count") or 0),
        "gt_available": bool(gt.get("gt_available")),
        "represented_gt_key_count": int(gt.get("represented_gt_key_count") or 0),
        "matched_candidate_count": int(gt.get("matched_candidate_count") or 0),
        "recommendation": recommendation,
        "swift_text_guard_baseline": swift_text_guard,
    }
    add_check(
        checks,
        "dedupe",
        f"{path}: reviewed GT available",
        summary["gt_available"] and summary["represented_gt_key_count"] >= args.min_dedupe_gt_keys,
        f"Dedupe tuning must cover at least {args.min_dedupe_gt_keys} represented reviewed question keys.",
        values={
            "gt_available": summary["gt_available"],
            "represented_gt_key_count": summary["represented_gt_key_count"],
            "matched_candidate_count": summary["matched_candidate_count"],
        },
    )
    add_check(
        checks,
        "dedupe",
        f"{path}: recommendation exists",
        bool(recommendation),
        "Dedupe tuning must produce a recommendation under the configured recall/false-merge constraints.",
        values={"recommendation": recommendation},
    )
    if recommendation:
        add_check(
            checks,
            "dedupe",
            f"{path}: recommendation GT recall",
            metric_float(recommendation, "gt_recall") >= args.min_dedupe_gt_recall,
            f"Recommended dedupe settings must keep represented-GT recall >= {args.min_dedupe_gt_recall:.3f}.",
            values={"gt_recall": recommendation.get("gt_recall")},
        )
        add_check(
            checks,
            "dedupe",
            f"{path}: recommendation lost GT keys",
            int(recommendation.get("lost_gt_key_count") or 0) <= args.max_dedupe_lost_gt_keys,
            f"Recommended dedupe settings may lose at most {args.max_dedupe_lost_gt_keys} represented GT key(s).",
            values={"lost_gt_key_count": recommendation.get("lost_gt_key_count"), "lost_gt_keys": recommendation.get("lost_gt_keys") or []},
        )
        add_check(
            checks,
            "dedupe",
            f"{path}: recommendation false merges",
            int(recommendation.get("false_merge_count") or 0) <= args.max_dedupe_false_merges,
            f"Recommended dedupe settings may have at most {args.max_dedupe_false_merges} known false merge(s).",
            values={"false_merge_count": recommendation.get("false_merge_count")},
        )
        add_check(
            checks,
            "dedupe",
            f"{path}: recommendation unknown GT suppressions",
            int(recommendation.get("unknown_gt_suppression_count") or 0) <= args.max_dedupe_unknown_gt_suppressions,
            f"Recommended dedupe settings may suppress at most {args.max_dedupe_unknown_gt_suppressions} reviewed candidate(s) against an unverified kept crop.",
            values={"unknown_gt_suppression_count": recommendation.get("unknown_gt_suppression_count")},
        )
        add_check(
            checks,
            "dedupe",
            f"{path}: recommendation rect-only overhead",
            metric_float(recommendation, "rect_only_selected_vs_full", 999.0) <= args.max_rect_overhead_ratio,
            f"Recommended dedupe rect-only overhead must stay <= {args.max_rect_overhead_ratio:.4f}x full-frame bytes.",
            values={"rect_only_selected_vs_full": recommendation.get("rect_only_selected_vs_full")},
        )
        add_check(
            checks,
            "dedupe",
            f"{path}: recommendation crop pixel ratio",
            metric_float(recommendation, "selected_crop_pixels_vs_full", 999.0) <= args.max_crop_pixel_ratio,
            f"Recommended dedupe crop pixels must stay <= {args.max_crop_pixel_ratio:.3f} of full-frame pixels.",
            values={"selected_crop_pixels_vs_full": recommendation.get("selected_crop_pixels_vs_full")},
        )
    add_check(
        checks,
        "dedupe",
        f"{path}: Swift text-guarded weak-key baseline exists",
        bool(swift_text_guard),
        "Dedupe tuning must include the iOS Swift weak-layout text-guard baseline.",
        values={"swift_text_guard_baseline": swift_text_guard},
    )
    if swift_text_guard:
        add_check(
            checks,
            "dedupe",
            f"{path}: Swift text-guarded GT recall",
            metric_float(swift_text_guard, "gt_recall") >= args.min_dedupe_gt_recall,
            f"Current iOS weak-key dedupe baseline must keep represented-GT recall >= {args.min_dedupe_gt_recall:.3f}.",
            values={"gt_recall": swift_text_guard.get("gt_recall")},
        )
        add_check(
            checks,
            "dedupe",
            f"{path}: Swift text-guarded lost GT keys",
            int(swift_text_guard.get("lost_gt_key_count") or 0) <= args.max_dedupe_lost_gt_keys,
            f"Current iOS weak-key dedupe baseline may lose at most {args.max_dedupe_lost_gt_keys} represented GT key(s).",
            values={"lost_gt_key_count": swift_text_guard.get("lost_gt_key_count"), "lost_gt_keys": swift_text_guard.get("lost_gt_keys") or []},
        )
        add_check(
            checks,
            "dedupe",
            f"{path}: Swift text-guarded unknown GT suppressions",
            int(swift_text_guard.get("unknown_gt_suppression_count") or 0) <= args.max_dedupe_unknown_gt_suppressions,
            f"Current iOS weak-key dedupe baseline may suppress at most {args.max_dedupe_unknown_gt_suppressions} reviewed candidate(s) against an unverified kept crop.",
            values={"unknown_gt_suppression_count": swift_text_guard.get("unknown_gt_suppression_count")},
        )
    return summary


def summarize_fallback_tune(
    path: Path,
    data: dict[str, Any],
    checks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    baseline = data.get("baseline_backend_current") if isinstance(data.get("baseline_backend_current"), dict) else {}
    union_baseline = data.get("baseline_backend_current_union_area") if isinstance(data.get("baseline_backend_current_union_area"), dict) else {}
    delta = data.get("fallback_delta_legacy_to_current") if isinstance(data.get("fallback_delta_legacy_to_current"), dict) else {}
    delta_review_queue = data.get("policy_delta_review_queue") if isinstance(data.get("policy_delta_review_queue"), dict) else {}
    represented_gt = int(data.get("represented_gt_key_count") or baseline.get("represented_gt_key_count") or 0)
    backend_recall = baseline.get("policy_gt_recall")
    try:
        backend_recall_float = float(backend_recall)
    except (TypeError, ValueError):
        backend_recall_float = 0.0
    try:
        total_pixels_vs_full = float(baseline.get("total_pixels_vs_full"))
    except (TypeError, ValueError):
        total_pixels_vs_full = 999.0
    try:
        fallback_image_ratio = float(baseline.get("fallback_image_ratio"))
    except (TypeError, ValueError):
        fallback_image_ratio = 999.0

    summary = {
        "path": str(path),
        "replays": replay_fingerprints(data),
        "ground_truth_manifests": ground_truth_manifest_sources(data),
        "image_count": int(data.get("image_count") or baseline.get("image_count") or 0),
        "candidate_count": int(data.get("candidate_count") or 0),
        "represented_gt_key_count": represented_gt,
        "crop_only_gt_recall": data.get("crop_only_gt_recall"),
        "backend_current": {
            "policy_id": baseline.get("policy_id"),
            "policy_gt_recall": backend_recall,
            "fallback_image_count": baseline.get("fallback_image_count"),
            "fallback_image_ratio": baseline.get("fallback_image_ratio"),
            "total_pixels_vs_full": baseline.get("total_pixels_vs_full"),
            "fallback_reason_counts": baseline.get("fallback_reason_counts") or {},
        },
        "backend_current_union_area": {
            "policy_id": union_baseline.get("policy_id"),
            "policy_gt_recall": union_baseline.get("policy_gt_recall"),
            "fallback_image_count": union_baseline.get("fallback_image_count"),
            "fallback_image_ratio": union_baseline.get("fallback_image_ratio"),
            "total_pixels_vs_full": union_baseline.get("total_pixels_vs_full"),
            "fallback_reason_counts": union_baseline.get("fallback_reason_counts") or {},
        },
        "legacy_to_current_delta": {
            "before_policy_id": delta.get("before_policy_id"),
            "after_policy_id": delta.get("after_policy_id"),
            "fallback_removed_image_count": delta.get("fallback_removed_image_count"),
            "fallback_added_image_count": delta.get("fallback_added_image_count"),
            "fallback_removed_with_gt_count": delta.get("fallback_removed_with_gt_count"),
            "fallback_removed_with_crop_missed_gt_count": delta.get("fallback_removed_with_crop_missed_gt_count"),
            "review_queue_images": delta_review_queue.get("images"),
            "review_queue_boxes": delta_review_queue.get("boxes"),
        },
        "recommendation": data.get("recommendation") if isinstance(data.get("recommendation"), dict) else {},
    }
    add_check(
        checks,
        "fallback",
        f"{path}: backend current policy exists",
        bool(baseline) and baseline.get("policy_id") == "backend_current",
        "Fallback tuning must report the current backend policy as baseline_backend_current.",
        values=summary["backend_current"],
    )
    add_check(
        checks,
        "fallback",
        f"{path}: reviewed GT available",
        represented_gt >= args.min_fallback_gt_keys,
        f"Fallback tuning must cover at least {args.min_fallback_gt_keys} represented reviewed GT key(s).",
        values={"represented_gt_key_count": represented_gt},
    )
    add_check(
        checks,
        "fallback",
        f"{path}: backend current GT recall",
        backend_recall_float >= args.min_fallback_gt_recall,
        f"Current backend fallback policy must keep represented-GT recall >= {args.min_fallback_gt_recall}.",
        values={"policy_gt_recall": backend_recall},
    )
    add_check(
        checks,
        "fallback",
        f"{path}: backend current VLM pixel ratio",
        total_pixels_vs_full <= args.max_fallback_total_pixels_vs_full,
        f"Current backend fallback policy total VLM pixels must stay <= {args.max_fallback_total_pixels_vs_full:.3f}x full-frame pixels.",
        values={"total_pixels_vs_full": baseline.get("total_pixels_vs_full")},
    )
    add_check(
        checks,
        "fallback",
        f"{path}: backend current fallback image ratio",
        fallback_image_ratio <= args.max_fallback_image_ratio,
        f"Current backend fallback policy may send full-frame fallback for at most {args.max_fallback_image_ratio:.3f} of crop-bearing images.",
        values={"fallback_image_ratio": baseline.get("fallback_image_ratio")},
    )
    add_check(
        checks,
        "fallback",
        f"{path}: backend current union-area shadow exists",
        bool(union_baseline) and union_baseline.get("policy_id") == "backend_current_union_area_shadow",
        "Fallback tuning must report a union-area shadow of the current backend policy, so overlapping crop coverage cannot silently hide risky frames.",
        values=summary["backend_current_union_area"],
    )
    add_check(
        checks,
        "fallback",
        f"{path}: legacy-to-current delta exists",
        bool(delta) and delta.get("before_policy_id") and delta.get("after_policy_id") == "backend_current",
        "Fallback tuning must report the legacy-to-current delta so stricter fallback policies cannot silently hide known GT misses.",
        values=summary["legacy_to_current_delta"],
    )
    add_check(
        checks,
        "fallback",
        f"{path}: legacy-removed fallback has no known GT miss",
        int(delta.get("fallback_removed_with_crop_missed_gt_count") or 0) <= args.max_fallback_delta_crop_missed_gt,
        f"Legacy-removed fallback frames may contain at most {args.max_fallback_delta_crop_missed_gt} represented GT miss(es).",
        values=summary["legacy_to_current_delta"],
    )
    review_queue_images = int(delta_review_queue.get("images") or 0)
    if review_queue_images > 0:
        add_check(
            checks,
            "fallback",
            f"{path}: legacy-removed no-GT review queue",
            False,
            "Legacy-removed fallback frames with no represented GT should be reviewed before treating fallback tuning evidence as production-complete.",
            "warning",
            values=summary["legacy_to_current_delta"],
        )
    return summary


def summarize_candidate_eval(
    path: Path,
    data: dict[str, Any],
    checks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    counts = data.get("counts") if isinstance(data.get("counts"), dict) else {}
    metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
    represented_gt = int(counts.get("represented_gt_key_count") or 0)
    raw_represented_gt = int(counts.get("raw_represented_gt_key_count") or represented_gt)
    overcrop_count = int(counts.get("overcrop_match_count") or 0)
    try:
        recall = float(metrics.get("represented_gt_recall"))
    except (TypeError, ValueError):
        recall = 0.0
    try:
        crop_pixels_vs_full = float(metrics.get("crop_pixels_vs_full_frame_pixels"))
    except (TypeError, ValueError):
        crop_pixels_vs_full = 999.0
    summary = {
        "path": str(path),
        "replays": replay_fingerprints(data),
        "ground_truth_manifests": ground_truth_manifest_sources(data),
        "candidate_count": int(counts.get("candidate_count") or 0),
        "raw_represented_gt_key_count": raw_represented_gt,
        "represented_gt_key_count": represented_gt,
        "duplicate_represented_gt_key_count": int(counts.get("duplicate_represented_gt_key_count") or 0),
        "covered_gt_key_count": int(counts.get("covered_gt_key_count") or 0),
        "missed_gt_key_count": int(counts.get("missed_gt_key_count") or 0),
        "overcrop_match_count": overcrop_count,
        "represented_gt_recall": metrics.get("represented_gt_recall"),
        "crop_pixels_vs_full_frame_pixels": metrics.get("crop_pixels_vs_full_frame_pixels"),
        "reviewed_candidate_match_rate": metrics.get("reviewed_candidate_match_rate"),
        "overcrop_ratio_p90": metrics.get("overcrop_ratio_p90"),
    }
    add_check(
        checks,
        "candidate_eval",
        f"{path}: reviewed GT available",
        represented_gt >= args.min_candidate_eval_gt_keys,
        f"Candidate eval must cover at least {args.min_candidate_eval_gt_keys} represented reviewed GT key(s) after duplicate clustering.",
        values={"represented_gt_key_count": represented_gt, "raw_represented_gt_key_count": raw_represented_gt},
    )
    add_check(
        checks,
        "candidate_eval",
        f"{path}: represented GT recall",
        recall >= args.min_candidate_eval_gt_recall,
        f"Observation crop candidates must cover represented-GT recall >= {args.min_candidate_eval_gt_recall}.",
        values={"represented_gt_recall": metrics.get("represented_gt_recall")},
    )
    add_check(
        checks,
        "candidate_eval",
        f"{path}: crop VLM pixel ratio",
        crop_pixels_vs_full <= args.max_candidate_eval_crop_pixels_vs_full,
        f"Observation crop candidates must stay <= {args.max_candidate_eval_crop_pixels_vs_full:.3f}x full-frame VLM pixels.",
        values={"crop_pixels_vs_full_frame_pixels": metrics.get("crop_pixels_vs_full_frame_pixels")},
    )
    add_check(
        checks,
        "candidate_eval",
        f"{path}: overcrop candidates",
        overcrop_count <= args.max_candidate_eval_overcrop_count,
        f"Observation candidate eval may have at most {args.max_candidate_eval_overcrop_count} overcrop candidate(s).",
        values={"overcrop_match_count": overcrop_count, "overcrop_ratio_p90": metrics.get("overcrop_ratio_p90")},
    )
    return summary


def summarize_candidate_cap_sweep(
    path: Path,
    data: dict[str, Any],
    checks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    current = data.get("current_cap") if isinstance(data.get("current_cap"), dict) else {}
    recommendation = data.get("recommendation") if isinstance(data.get("recommendation"), dict) else {}
    selected_recall = metric_float(current, "selected_gt_recall", 0.0)
    policy_recall = metric_float(current, "policy_gt_recall", 0.0)
    total_pixels_vs_full = metric_float(current, "total_pixels_vs_full", 999.0)
    try:
        skipped_strong = int(current.get("skipped_strong_candidate_count") or 0)
    except (TypeError, ValueError):
        skipped_strong = 0
    try:
        skipped_confident = int(current.get("skipped_confident_candidate_count") or 0)
    except (TypeError, ValueError):
        skipped_confident = 0
    summary = {
        "path": str(path),
        "replays": replay_fingerprints(data),
        "ground_truth_manifests": ground_truth_manifest_sources(data),
        "image_count": int(data.get("image_count") or 0),
        "candidate_count": int(data.get("candidate_count") or 0),
        "represented_gt_key_count": int(data.get("represented_gt_key_count") or 0),
        "current_cap": current,
        "recommendation": recommendation,
    }
    add_check(
        checks,
        "candidate_cap",
        f"{path}: current cap row exists",
        bool(current) and int(current.get("cap") or -1) > 0,
        "Candidate cap sweep must report the current iOS per-frame cap.",
        values={"current_cap": current},
    )
    add_check(
        checks,
        "candidate_cap",
        f"{path}: represented GT available",
        summary["represented_gt_key_count"] >= args.min_candidate_cap_gt_keys,
        f"Candidate cap sweep must cover at least {args.min_candidate_cap_gt_keys} represented reviewed GT key(s).",
        values={"represented_gt_key_count": summary["represented_gt_key_count"]},
    )
    add_check(
        checks,
        "candidate_cap",
        f"{path}: current cap crop-only GT recall",
        selected_recall >= args.min_candidate_cap_gt_recall,
        f"Current iOS per-frame cap must keep crop-only represented-GT recall >= {args.min_candidate_cap_gt_recall}.",
        values={"selected_gt_recall": current.get("selected_gt_recall")},
    )
    add_check(
        checks,
        "candidate_cap",
        f"{path}: current cap fallback-protected GT recall",
        policy_recall >= args.min_candidate_cap_gt_recall,
        f"Current iOS per-frame cap with backend fallback must keep represented-GT recall >= {args.min_candidate_cap_gt_recall}.",
        values={"policy_gt_recall": current.get("policy_gt_recall")},
    )
    add_check(
        checks,
        "candidate_cap",
        f"{path}: current cap VLM pixel ratio",
        total_pixels_vs_full <= args.max_candidate_cap_total_pixels_vs_full,
        f"Current iOS per-frame cap total crop+fallback VLM pixels must stay <= {args.max_candidate_cap_total_pixels_vs_full:.3f}x full-frame pixels.",
        values={"total_pixels_vs_full": current.get("total_pixels_vs_full")},
    )
    add_check(
        checks,
        "candidate_cap",
        f"{path}: current cap skips no high-risk candidates",
        skipped_strong == 0 and skipped_confident == 0,
        "Current iOS per-frame cap should not skip strong-OCR or high-confidence candidates; fallback may protect recall, but this is a dense-page cost/risk signal.",
        "warning",
        values={
            "skipped_strong_candidate_count": current.get("skipped_strong_candidate_count"),
            "skipped_confident_candidate_count": current.get("skipped_confident_candidate_count"),
        },
    )
    add_check(
        checks,
        "candidate_cap",
        f"{path}: cap recommendation exists",
        bool(recommendation),
        "Candidate cap sweep should produce a lowest-cost cap recommendation under the configured recall constraints.",
        "warning",
        values={"recommendation": recommendation},
    )
    return summary


def add_reviewed_evidence_consistency_checks(
    checks: list[dict[str, Any]],
    fallback_tune_summaries: list[dict[str, Any]],
    candidate_eval_summaries: list[dict[str, Any]],
    candidate_cap_summaries: list[dict[str, Any]],
) -> None:
    groups = {
        "fallback_tune": fallback_tune_summaries,
        "candidate_eval": candidate_eval_summaries,
        "candidate_cap_sweep": candidate_cap_summaries,
    }
    present = {name: rows for name, rows in groups.items() if rows}
    if len(present) < len(groups):
        return

    replay_values: dict[str, list[list[dict[str, Any]]]] = {
        name: [
            [replay for replay in row.get("replays", []) if isinstance(replay, dict)]
            for row in rows
        ]
        for name, rows in present.items()
    }
    replay_keys = {
        replay_fingerprint_set({"replays": replays})
        for replay_lists in replay_values.values()
        for replays in replay_lists
    }
    missing_replay = any(
        not replay_lists
        or any(not replays or any(not replay.get("path") for replay in replays) for replays in replay_lists)
        for replay_lists in replay_values.values()
    )
    add_check(
        checks,
        "evidence",
        "reviewed replay evidence is same source",
        not missing_replay and len(replay_keys) == 1,
        "Fallback tune, candidate eval, and candidate-cap sweep must be generated from the same complete replay list, image counts, candidate counts, and full-frame pixel totals.",
        values={"replays": replay_values},
    )

    gt_sources: dict[str, list[list[str]]] = {
        name: [list(row.get("ground_truth_manifests") or []) for row in rows]
        for name, rows in present.items()
    }
    gt_source_keys = {
        tuple(sources)
        for source_lists in gt_sources.values()
        for sources in source_lists
    }
    missing_gt_source = any(
        not source_lists or any(not sources for sources in source_lists)
        for source_lists in gt_sources.values()
    )
    add_check(
        checks,
        "evidence",
        "reviewed GT source is consistent across replay evidence",
        not missing_gt_source and len(gt_source_keys) == 1,
        "Fallback tune, candidate eval, and candidate-cap sweep must report the same reviewed GT manifest source list.",
        values={"ground_truth_manifests": gt_sources},
    )

    gt_counts: dict[str, list[int]] = {
        name: [int(row.get("represented_gt_key_count") or 0) for row in rows]
        for name, rows in present.items()
    }
    count_values = {count for counts in gt_counts.values() for count in counts}
    add_check(
        checks,
        "evidence",
        "reviewed GT volume is consistent across replay evidence",
        len(count_values) == 1 and next(iter(count_values), 0) > 0,
        "Fallback tune, candidate eval, and candidate-cap sweep must report the same represented reviewed GT count.",
        values={"represented_gt_key_counts": gt_counts},
    )


def summarize_section_sender_eval(
    path: Path,
    data: dict[str, Any],
    checks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    sender = data.get("sender") if isinstance(data.get("sender"), dict) else {}
    cap_only = data.get("cap_only") if isinstance(data.get("cap_only"), dict) else {}
    settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    represented_gt = int(sender.get("represented_gt_key_count") or data.get("represented_gt_key_count") or 0)
    combined_recall = metric_float(sender, "combined_crop_gt_recall")
    policy_recall = metric_float(sender, "policy_gt_recall")
    total_pixels_vs_full = metric_float(sender, "total_pixels_vs_full", 999.0)
    fallback_ratio = metric_float(sender, "fallback_image_ratio", 999.0)
    candidate_count = int(sender.get("candidate_count") or data.get("candidate_count") or 0)
    skipped_count = int(sender.get("skipped_candidate_count") or 0)
    section_count = int(sender.get("section_count") or 0)
    if settings.get("enable_empty_frame_section") is True and candidate_count == 0 and section_count > 0:
        evidence_kind = "empty_frame"
    elif settings.get("enable_risk_frame_section") is True and candidate_count > 0 and section_count > 0:
        evidence_kind = "risk_frame"
    elif candidate_count > 0 and skipped_count > 0 and section_count > 0:
        evidence_kind = "cap_overflow"
    else:
        evidence_kind = "unknown"
    min_gt_keys = args.min_risk_section_gt_keys if evidence_kind == "risk_frame" else args.min_section_sender_gt_keys
    min_recall = args.min_risk_section_gt_recall if evidence_kind == "risk_frame" else args.min_section_sender_gt_recall
    max_pixels = args.max_risk_section_total_pixels_vs_full if evidence_kind == "risk_frame" else args.max_section_sender_total_pixels_vs_full
    max_fallback_ratio = args.max_risk_section_fallback_image_ratio if evidence_kind == "risk_frame" else args.max_section_sender_fallback_image_ratio
    summary = {
        "path": str(path),
        "image_count": int(sender.get("image_count") or data.get("image_count") or 0),
        "candidate_count": candidate_count,
        "represented_gt_key_count": represented_gt,
        "settings": settings,
        "evidence_kind": evidence_kind,
        "cap_only": cap_only,
        "sender": sender,
        "total_pixels_delta_vs_cap_only": data.get("total_pixels_delta_vs_cap_only"),
    }
    add_check(
        checks,
        "section_sender",
        f"{path}: dense represented GT available",
        represented_gt >= min_gt_keys,
        f"Section sender eval must cover at least {min_gt_keys} represented GT key(s) for {evidence_kind} evidence.",
        values={"represented_gt_key_count": represented_gt, "evidence_kind": evidence_kind},
    )
    add_check(
        checks,
        "section_sender",
        f"{path}: combined crop GT recall",
        combined_recall >= min_recall,
        f"iOS section sender combined crop recall must be >= {min_recall}.",
        values={"combined_crop_gt_recall": sender.get("combined_crop_gt_recall"), "evidence_kind": evidence_kind},
    )
    add_check(
        checks,
        "section_sender",
        f"{path}: fallback-protected GT recall",
        policy_recall >= min_recall,
        f"iOS section sender fallback-protected recall must be >= {min_recall}.",
        values={"policy_gt_recall": sender.get("policy_gt_recall"), "evidence_kind": evidence_kind},
    )
    add_check(
        checks,
        "section_sender",
        f"{path}: dense VLM pixel ratio",
        total_pixels_vs_full <= max_pixels,
        f"iOS section sender total VLM pixels must stay <= {max_pixels:.3f}x full-frame pixels for {evidence_kind} evidence.",
        values={"total_pixels_vs_full": sender.get("total_pixels_vs_full"), "evidence_kind": evidence_kind},
    )
    add_check(
        checks,
        "section_sender",
        f"{path}: dense fallback image ratio",
        fallback_ratio <= max_fallback_ratio,
        f"iOS section sender fallback image ratio must stay <= {max_fallback_ratio:.3f} for {evidence_kind} evidence.",
        values={"fallback_image_ratio": sender.get("fallback_image_ratio"), "evidence_kind": evidence_kind},
    )
    return summary


def strategy_row_recall(row: dict[str, Any]) -> float:
    if row.get("policy_gt_recall") is not None:
        return metric_float(row, "policy_gt_recall")
    return metric_float(row, "crop_gt_recall")


def strategy_row_passes(row: dict[str, Any], args: argparse.Namespace) -> bool:
    represented_gt = int(row.get("represented_gt_key_count") or 0)
    recall = strategy_row_recall(row)
    total_pixels = metric_float(row, "total_pixels_vs_full", 999.0)
    fallback_ratio = metric_float(row, "fallback_image_ratio", 999.0)
    return (
        represented_gt >= args.min_strategy_gt_keys
        and recall >= args.min_strategy_gt_recall
        and total_pixels <= args.max_strategy_total_pixels_vs_full
        and fallback_ratio <= args.max_strategy_fallback_image_ratio
    )


def strategy_row_summary(row: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(row, dict):
        return {}
    return {
        "strategy_id": row.get("strategy_id"),
        "family": row.get("family"),
        "replay": row.get("replay"),
        "evidence_kind": row.get("evidence_kind"),
        "represented_gt_key_count": row.get("represented_gt_key_count"),
        "recall": strategy_row_recall(row),
        "fallback_image_ratio": row.get("fallback_image_ratio"),
        "total_pixels_vs_full": row.get("total_pixels_vs_full"),
        "status": row.get("status"),
    }


def best_strategy_row(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any] | None:
    passing = [row for row in rows if strategy_row_passes(row, args)]
    if not passing:
        return None
    return sorted(
        passing,
        key=lambda row: (
            metric_float(row, "total_pixels_vs_full", 999.0),
            metric_float(row, "fallback_image_ratio", 999.0),
            -strategy_row_recall(row),
        ),
    )[0]


def summarize_strategy_selection(
    path: Path,
    data: dict[str, Any],
    checks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = [row for row in (data.get("rows") or []) if isinstance(row, dict)]
    best_reviewed = [row for row in (data.get("best_reviewed_live_by_cost") or []) if isinstance(row, dict)]
    synthetic = [row for row in (data.get("synthetic_stress_rows") or []) if isinstance(row, dict)]
    reviewed_live = [
        row
        for row in rows
        if row.get("status") == "live"
        and row.get("family") != "full_frame_baseline"
        and str(row.get("evidence_kind") or "").startswith("reviewed_")
    ]
    propagated_live = [row for row in reviewed_live if row.get("evidence_kind") == "reviewed_propagated"]
    best_reviewed_passing = best_strategy_row(reviewed_live, args)
    best_propagated_passing = best_strategy_row(propagated_live, args)
    best_reviewed_has_synthetic = any(row.get("evidence_kind") == "synthetic_stress" for row in best_reviewed)
    summary = {
        "path": str(path),
        "row_count": int(data.get("row_count") or len(rows)),
        "reviewed_live_row_count": len(reviewed_live),
        "propagated_live_row_count": len(propagated_live),
        "synthetic_stress_row_count": len(synthetic),
        "best_reviewed_live_by_cost": [strategy_row_summary(row) for row in best_reviewed[:5]],
        "best_reviewed_passing_strategy": strategy_row_summary(best_reviewed_passing),
        "best_propagated_passing_strategy": strategy_row_summary(best_propagated_passing),
    }
    add_check(
        checks,
        "strategy_selection",
        f"{path}: strategy rows available",
        summary["row_count"] > 0,
        "Strategy selection must include normalized crop/fallback/section rows.",
        values={"row_count": summary["row_count"]},
    )
    add_check(
        checks,
        "strategy_selection",
        f"{path}: reviewed live strategy passes recall/cost gate",
        best_reviewed_passing is not None,
        "At least one reviewed live observation strategy must satisfy recall, fallback-ratio, and VLM-pixel gates.",
        values={
            "min_strategy_gt_keys": args.min_strategy_gt_keys,
            "min_strategy_gt_recall": args.min_strategy_gt_recall,
            "max_strategy_total_pixels_vs_full": args.max_strategy_total_pixels_vs_full,
            "max_strategy_fallback_image_ratio": args.max_strategy_fallback_image_ratio,
            "best_reviewed_passing_strategy": summary["best_reviewed_passing_strategy"],
        },
    )
    add_check(
        checks,
        "strategy_selection",
        f"{path}: propagated reviewed live strategy passes recall/cost gate",
        best_propagated_passing is not None,
        "Broader propagated reviewed evidence must include a live observation strategy that satisfies recall, fallback-ratio, and VLM-pixel gates.",
        values={
            "propagated_live_row_count": len(propagated_live),
            "best_propagated_passing_strategy": summary["best_propagated_passing_strategy"],
        },
    )
    add_check(
        checks,
        "strategy_selection",
        f"{path}: synthetic stress rows are not production recommendations",
        bool(best_reviewed) and not best_reviewed_has_synthetic,
        "Strategy selection must keep synthetic stress rows out of best_reviewed_live_by_cost.",
        values={"best_reviewed_live_by_cost": summary["best_reviewed_live_by_cost"]},
    )
    return summary


def summarize_runtime_perf(
    path: Path,
    data: dict[str, Any],
    checks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    analysis = data.get("analysis_duration_ms") if isinstance(data.get("analysis_duration_ms"), dict) else {}
    server = data.get("question_crop_server_duration_ms") if isinstance(data.get("question_crop_server_duration_ms"), dict) else {}
    record_count = int(data.get("record_count") or 0)
    attached_crop_file_count = int(data.get("attached_crop_file_count") or 0)
    fresh_ocr_ratio = metric_float(data, "fresh_ocr_ratio", 999.0)
    analysis_p95 = metric_float(analysis, "p95", 999.0)
    server_p95 = metric_float(server, "p95", -1.0) if server.get("p95") is not None else None
    raw_evidence_kind = str(data.get("evidence_kind") or "").strip().lower()
    evidence_kind = raw_evidence_kind or ("fixture" if "fixture" in str(path).lower() else "live")
    summary = {
        "path": str(path),
        "evidence_kind": evidence_kind,
        "passed": bool(data.get("passed")),
        "record_count": record_count,
        "frame_count": int(data.get("frame_count") or 0),
        "cached_frame_count": int(data.get("cached_frame_count") or 0),
        "ocr_frame_count": int(data.get("ocr_frame_count") or 0),
        "fresh_ocr_ratio": data.get("fresh_ocr_ratio"),
        "attached_crop_file_count": attached_crop_file_count,
        "analysis_duration_ms": analysis,
        "question_crop_server_duration_ms": server,
    }
    add_check(
        checks,
        "runtime",
        f"{path}: runtime evidence is not a fixture",
        evidence_kind != "fixture" or bool(args.allow_runtime_fixture),
        "Runtime perf fixtures are allowed only for explicit release-gate smoke checks; release candidates need live/simulator telemetry.",
        values={"evidence_kind": evidence_kind, "allow_runtime_fixture": bool(args.allow_runtime_fixture)},
    )
    add_check(
        checks,
        "runtime",
        f"{path}: runtime record coverage",
        record_count >= args.min_runtime_records,
        f"Runtime perf evidence must include at least {args.min_runtime_records} observation batch record(s).",
        values={"record_count": record_count},
    )
    add_check(
        checks,
        "runtime",
        f"{path}: rect-only transport remains fileless",
        attached_crop_file_count <= args.max_runtime_attached_crop_files,
        "Observation question crop uploads must not attach crop JPEG files.",
        values={"attached_crop_file_count": attached_crop_file_count},
    )
    add_check(
        checks,
        "runtime",
        f"{path}: local analysis p95 latency",
        analysis_p95 <= args.max_runtime_analysis_p95_ms,
        f"Live/simulator local analysis p95 must stay <= {args.max_runtime_analysis_p95_ms:.1f} ms.",
        values={"analysis_p95_ms": analysis.get("p95")},
    )
    if server_p95 is not None:
        add_check(
            checks,
            "runtime",
            f"{path}: backend crop p95 latency",
            server_p95 <= args.max_runtime_server_p95_ms,
            f"Backend canonical crop expansion p95 must stay <= {args.max_runtime_server_p95_ms:.1f} ms.",
            values={"server_p95_ms": server.get("p95")},
        )
    add_check(
        checks,
        "runtime",
        f"{path}: cached segmentation reuse",
        fresh_ocr_ratio <= args.max_runtime_fresh_ocr_ratio,
        f"Fresh OCR frame ratio must stay <= {args.max_runtime_fresh_ocr_ratio:.3f}; uploads should reuse live-scan cache.",
        values={"fresh_ocr_ratio": data.get("fresh_ocr_ratio")},
    )
    add_check(
        checks,
        "runtime",
        f"{path}: runtime perf script passed",
        bool(data.get("passed")),
        "Runtime perf check summary must pass its own configured checks.",
        values={"passed": data.get("passed")},
    )
    return summary


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Question Detector Release Check",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Release ready: `{str(report['release_ready']).lower()}`",
        f"- Hard failures: {len(report['hard_failures'])}",
        f"- Warnings: {len(report['warnings'])}",
        "",
    ]
    if report["hard_failures"]:
        lines.extend(["## Hard Failures", ""])
        for item in report["hard_failures"]:
            lines.append(f"- [{item['category']}] {item['name']}: {item['detail']}")
        lines.append("")
    if report["warnings"]:
        lines.extend(["## Warnings", ""])
        for item in report["warnings"]:
            lines.append(f"- [{item['category']}] {item['name']}: {item['detail']}")
        lines.append("")
    lines.extend(["## Passed Checks", ""])
    for item in report["checks"]:
        if item["passed"]:
            lines.append(f"- [{item['category']}] {item['name']}")
    lines.append("")
    return "\n".join(lines)


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)

    checks: list[dict[str, Any]] = []
    dataset_summaries: list[dict[str, Any]] = []
    replay_summaries: list[dict[str, Any]] = []
    dedupe_tune_summaries: list[dict[str, Any]] = []
    fallback_tune_summaries: list[dict[str, Any]] = []
    candidate_eval_summaries: list[dict[str, Any]] = []
    candidate_cap_summaries: list[dict[str, Any]] = []
    section_sender_summaries: list[dict[str, Any]] = []
    strategy_selection_summaries: list[dict[str, Any]] = []
    runtime_perf_summaries: list[dict[str, Any]] = []
    selection_summary: dict[str, Any] = {}

    for dataset in args.dataset:
        audit_path = resolve_file(dataset, "audit.json")
        if not audit_path.is_file():
            add_check(checks, "dataset", f"{dataset}: audit exists", False, "Missing audit.json.")
            continue
        dataset_summaries.append(summarize_dataset(audit_path, load_json(audit_path), checks))

    selection_path = resolve_file(args.selection, "selection_summary.json")
    selection: dict[str, Any] = {}
    if not selection_path.is_file():
        add_check(checks, "selection", "selection summary exists", False, "Missing selection_summary.json.")
    else:
        loaded = load_json(selection_path)
        if isinstance(loaded, dict):
            selection = loaded
            selection_summary = summarize_selection(selection_path, selection, checks, args)
        else:
            add_check(checks, "selection", "selection summary is object", False, "selection_summary.json is not a JSON object.")

    manifest_paths = discover_train_manifests(args.train_manifest)
    for selected_manifest in selected_run_manifests(selection, selection_path):
        if str(selected_manifest.resolve()) not in {str(path.resolve()) for path in manifest_paths}:
            manifest_paths.append(selected_manifest)
    train_summaries = summarize_train_manifests(manifest_paths, checks)

    artifact_paths = artifact_candidates_from_selection(selection, selection_path)
    artifact_paths.extend(artifact_candidates_from_manifests(manifest_paths))
    unique_artifacts = {str(path.resolve()): path for path in artifact_paths}
    artifact_paths = sorted(unique_artifacts.values())
    add_check(
        checks,
        "artifact",
        "Core ML artifact exists",
        bool(artifact_paths),
        "Selected detector must have an exported QuestionRegionDetector Core ML artifact.",
        values={"artifacts": [str(path) for path in artifact_paths]},
    )

    ios_resources = ios_model_resources(args.ios_root)
    add_check(
        checks,
        "ios",
        "QuestionRegionDetector bundled in iOS app sources",
        bool(ios_resources),
        "iOS target must include QuestionRegionDetector.mlmodel/.mlpackage or compiled .mlmodelc.",
        values={"resources": [str(path) for path in ios_resources]},
    )

    swift_config = parse_swift_detector_config(args.ios_root)
    add_check(
        checks,
        "ios",
        "Swift detector hook exists",
        bool(swift_config.get("exists")) and bool(swift_config.get("bundle_lookup_mlmodelc")),
        "QuestionSegmenter must look up QuestionRegionDetector.mlmodelc from the app bundle.",
        values=swift_config,
    )
    add_check(
        checks,
        "ios",
        "Swift detector OCR merge fallback gate exists",
        bool(swift_config.get("exists"))
        and bool(swift_config.get("detector_first_requires_matched_text"))
        and bool(swift_config.get("detector_fast_path_requires_completion"))
        and bool(swift_config.get("detector_merges_unmatched_ocr"))
        and bool(swift_config.get("detector_no_early_first_return"))
        and bool(swift_config.get("detector_textless_fallback_telemetry"))
        and bool(swift_config.get("detector_reuses_first_pass_boxes_after_fallback")),
        "Detector-first segmentation must only skip full-page OCR when OCR-matched detector boxes look complete, otherwise reuse first-pass detector boxes and merge unmatched OCR regions.",
        values=swift_config,
    )

    swift_dedupe_config = parse_swift_observation_dedupe_config(args.ios_root)
    add_check(
        checks,
        "dedupe",
        "Swift weak-layout text guard exists",
        bool(swift_dedupe_config.get("exists"))
        and bool(swift_dedupe_config.get("weak_text_part_function"))
        and bool(swift_dedupe_config.get("weak_text_compatible_function"))
        and bool(swift_dedupe_config.get("weak_text_guard_in_signature_similarity"))
        and bool(swift_dedupe_config.get("notext_is_not_compatible"))
        and bool(swift_dedupe_config.get("uses_near_text_similarity")),
        "ContentView observation dedupe must require compatible short text before weak layout keys can merge by geometry.",
        values=swift_dedupe_config,
    )
    add_check(
        checks,
        "ios",
        "Swift ranked candidate cap telemetry exists",
        bool(swift_dedupe_config.get("exists"))
        and bool(swift_dedupe_config.get("candidate_rank_function"))
        and bool(swift_dedupe_config.get("upload_uses_ranked_candidates"))
        and bool(swift_dedupe_config.get("rank_mode_telemetry"))
        and bool(swift_dedupe_config.get("limited_risk_telemetry"))
        and bool(swift_dedupe_config.get("no_raw_prefix_upload")),
        "Observation upload must rank candidates before the per-frame cap and report skipped-candidate risk telemetry.",
        values=swift_dedupe_config,
    )
    add_check(
        checks,
        "ios",
        "Swift dense section crop sender exists",
        bool(swift_dedupe_config.get("exists"))
        and bool(swift_dedupe_config.get("section_crop_function"))
        and bool(swift_dedupe_config.get("section_crop_manifest_fields"))
        and bool(swift_dedupe_config.get("section_crop_risk_gated"))
        and bool(swift_dedupe_config.get("section_companion_top2"))
        and bool(swift_dedupe_config.get("risk_section_function"))
        and bool(swift_dedupe_config.get("risk_section_top1_companion"))
        and bool(swift_dedupe_config.get("section_crop_metrics")),
        "Observation upload should add section/group rect-only crops for dense cap-risk, sparse weak/low-coverage, and empty study frames; dense sections keep top-2 anchors while risk sections keep a top-1 anchor.",
        values=swift_dedupe_config,
    )
    add_check(
        checks,
        "ios",
        "Swift empty-frame section dedupe exists",
        bool(swift_dedupe_config.get("exists"))
        and bool(swift_dedupe_config.get("empty_section_dedupe_state"))
        and bool(swift_dedupe_config.get("empty_section_duplicate_guard"))
        and bool(swift_dedupe_config.get("empty_section_signature_persisted")),
        "No-candidate page section crops must use separate section signatures so repeated empty-frame sections can be skipped without suppressing later per-question boxes.",
        values=swift_dedupe_config,
    )
    add_check(
        checks,
        "ios",
        "Swift Vision/JPEG resizing is pixel-bound",
        bool(swift_dedupe_config.get("exists"))
        and bool(swift_dedupe_config.get("pixel_resize_helper"))
        and bool(swift_dedupe_config.get("vision_resize_uses_pixel_helper"))
        and bool(swift_dedupe_config.get("jpeg_uses_pixel_helper"))
        and bool(swift_dedupe_config.get("pixel_renderer_scale_one")),
        "Vision analysis and upload JPEG compression must limit real CGImage pixels, not UIImage point size inflated by UIScreen scale.",
        values=swift_dedupe_config,
    )

    backend_source_binding = parse_backend_question_source_binding_config(args.backend_main)
    add_check(
        checks,
        "backend",
        "question crop source input_index binding exists",
        bool(backend_source_binding.get("exists"))
        and bool(backend_source_binding.get("parser_reads_input_index"))
        and bool(backend_source_binding.get("parser_accepts_source_index_alias"))
        and bool(backend_source_binding.get("batch_sources_assign_input_index"))
        and bool(backend_source_binding.get("batch_prompt_requests_input_index"))
        and bool(backend_source_binding.get("matcher_prefers_input_index"))
        and bool(backend_source_binding.get("source_meta_persists_input_index"))
        and bool(backend_source_binding.get("apply_source_meta_after_match")),
        "Batched crop extraction must preserve model-returned input_index and bind questions to the matching crop before falling back to text similarity.",
        values=backend_source_binding,
    )
    add_check(
        checks,
        "backend",
        "question crop candidate cap fallback is risk-aware",
        bool(backend_source_binding.get("exists"))
        and bool(backend_source_binding.get("persists_candidate_cap_risk_telemetry"))
        and bool(backend_source_binding.get("uses_ranked_limit_telemetry_guard"))
        and bool(backend_source_binding.get("uses_limited_strong_confident_guard")),
        "Backend fallback must keep old-client candidate-cap safety while using ranked-cap telemetry to avoid unnecessary full-frame VLM fallback.",
        values=backend_source_binding,
    )
    add_check(
        checks,
        "backend",
        "weak layout crop keys do not pre-dedupe extraction sources",
        bool(backend_source_binding.get("exists"))
        and bool(backend_source_binding.get("weak_layout_key_ignored_for_crop_source_dedupe")),
        "Backend extraction source enumeration must not use weak layout question keys or their fingerprints to suppress crops before VLM extraction.",
        values=backend_source_binding,
    )
    add_check(
        checks,
        "backend",
        "section crop sources stay source-safe",
        bool(backend_source_binding.get("exists"))
        and bool(backend_source_binding.get("parses_section_crop_kind"))
        and bool(backend_source_binding.get("section_crop_key_is_weak_for_source_dedupe"))
        and bool(backend_source_binding.get("persists_section_crop_telemetry"))
        and bool(backend_source_binding.get("source_meta_persists_section_crop_kind"))
        and bool(backend_source_binding.get("normalizes_section_extraction_sources"))
        and bool(backend_source_binding.get("section_prompt_allows_multiple_questions"))
        and bool(backend_source_binding.get("section_crop_protects_limited_fallback")),
        "Section crops may contain multiple questions, so backend manifest parsing, source dedupe, fallback policy, prompt text, and persisted extraction metadata must keep them as extraction sources rather than single-question identities.",
        values=backend_source_binding,
    )

    ios_rec = selection_summary.get("ios_recommendation") if selection_summary else {}
    if isinstance(ios_rec, dict) and ios_rec:
        fast_expected = ios_rec.get("question_region_detector_fast_min_confidence")
        accurate_expected = ios_rec.get("question_region_detector_accurate_min_confidence")
        fast_ok = compare_float(swift_config.get("fast_min_confidence"), fast_expected, args.threshold_tolerance)
        accurate_ok = compare_float(swift_config.get("accurate_min_confidence"), accurate_expected, args.threshold_tolerance)
        add_check(
            checks,
            "ios",
            "Swift fast confidence matches selected threshold",
            fast_ok,
            "detectorFastMinConfidence must match selection_summary iOS recommendation.",
            values={"swift": swift_config.get("fast_min_confidence"), "recommended": fast_expected},
        )
        add_check(
            checks,
            "ios",
            "Swift accurate confidence matches selected threshold",
            accurate_ok,
            "detectorAccurateMinConfidence must match selection_summary iOS recommendation.",
            values={"swift": swift_config.get("accurate_min_confidence"), "recommended": accurate_expected},
        )

    for replay in args.replay:
        replay_path = resolve_file(replay, "summary.json")
        if not replay_path.is_file():
            add_check(checks, "replay", f"{replay}: summary exists", False, "Missing replay summary.json.")
            continue
        loaded = load_json(replay_path)
        if isinstance(loaded, dict):
            replay_summaries.append(summarize_replay(replay_path, loaded, checks, args))
        else:
            add_check(checks, "replay", f"{replay}: summary object", False, "Replay summary is not a JSON object.")

    if not args.dedupe_tune:
        add_check(
            checks,
            "dedupe",
            "dedupe tuning supplied",
            False,
            "Release check requires at least one --dedupe-tune report so local duplicate suppression cannot hide missed questions.",
        )
    for dedupe_tune in args.dedupe_tune:
        dedupe_path = resolve_file(dedupe_tune, "summary.json")
        if not dedupe_path.is_file():
            add_check(checks, "dedupe", f"{dedupe_tune}: summary exists", False, "Missing dedupe tune summary.json.")
            continue
        loaded = load_json(dedupe_path)
        if isinstance(loaded, dict):
            dedupe_tune_summaries.append(summarize_dedupe_tune(dedupe_path, loaded, checks, args))
        else:
            add_check(checks, "dedupe", f"{dedupe_tune}: summary object", False, "Dedupe tune summary is not a JSON object.")

    if not args.fallback_tune:
        add_check(
            checks,
            "fallback",
            "fallback policy tuning supplied",
            False,
            "Release check requires at least one --fallback-tune report so full-frame fallback cost cannot silently regress.",
        )
    for fallback_tune in args.fallback_tune:
        fallback_path = resolve_file(fallback_tune, "summary.json")
        if not fallback_path.is_file():
            add_check(checks, "fallback", f"{fallback_tune}: summary exists", False, "Missing fallback tune summary.json.")
            continue
        loaded = load_json(fallback_path)
        if isinstance(loaded, dict):
            fallback_tune_summaries.append(summarize_fallback_tune(fallback_path, loaded, checks, args))
        else:
            add_check(checks, "fallback", f"{fallback_tune}: summary object", False, "Fallback tune summary is not a JSON object.")

    if not args.candidate_eval:
        add_check(
            checks,
            "candidate_eval",
            "observation candidate eval supplied",
            False,
            "Release check requires at least one --candidate-eval report so local crop-candidate recall cannot silently regress.",
        )
    for candidate_eval in args.candidate_eval:
        candidate_eval_path = resolve_file(candidate_eval, "summary.json")
        if not candidate_eval_path.is_file():
            add_check(checks, "candidate_eval", f"{candidate_eval}: summary exists", False, "Missing candidate eval summary.json.")
            continue
        loaded = load_json(candidate_eval_path)
        if isinstance(loaded, dict):
            candidate_eval_summaries.append(summarize_candidate_eval(candidate_eval_path, loaded, checks, args))
        else:
            add_check(checks, "candidate_eval", f"{candidate_eval}: summary object", False, "Candidate eval summary is not a JSON object.")

    if not args.candidate_cap_sweep:
        add_check(
            checks,
            "candidate_cap",
            "candidate cap sweep supplied",
            False,
            "Release check requires at least one --candidate-cap-sweep report so the iOS per-frame crop cap cannot silently drop questions or inflate fallback cost.",
        )
    for candidate_cap_sweep in args.candidate_cap_sweep:
        candidate_cap_path = resolve_file(candidate_cap_sweep, "summary.json")
        if not candidate_cap_path.is_file():
            add_check(checks, "candidate_cap", f"{candidate_cap_sweep}: summary exists", False, "Missing candidate cap sweep summary.json.")
            continue
        loaded = load_json(candidate_cap_path)
        if isinstance(loaded, dict):
            candidate_cap_summaries.append(summarize_candidate_cap_sweep(candidate_cap_path, loaded, checks, args))
        else:
            add_check(checks, "candidate_cap", f"{candidate_cap_sweep}: summary object", False, "Candidate cap sweep summary is not a JSON object.")

    add_reviewed_evidence_consistency_checks(
        checks,
        fallback_tune_summaries,
        candidate_eval_summaries,
        candidate_cap_summaries,
    )

    if not args.section_sender_eval:
        add_check(
            checks,
            "section_sender",
            "section sender eval supplied",
            False,
            "Dense iOS section sender evidence is required so static sender code cannot ship without recall/cost proof.",
        )
    for section_sender_eval in args.section_sender_eval:
        section_sender_path = resolve_file(section_sender_eval, "summary.json")
        if not section_sender_path.is_file():
            add_check(checks, "section_sender", f"{section_sender_eval}: summary exists", False, "Missing section sender eval summary.json.")
            continue
        loaded = load_json(section_sender_path)
        if isinstance(loaded, dict):
            section_sender_summaries.append(summarize_section_sender_eval(section_sender_path, loaded, checks, args))
        else:
            add_check(checks, "section_sender", f"{section_sender_eval}: summary object", False, "Section sender eval summary is not a JSON object.")
    if args.section_sender_eval:
        section_sender_kinds = {summary.get("evidence_kind") for summary in section_sender_summaries}
        add_check(
            checks,
            "section_sender",
            "cap-overflow section sender evidence supplied",
            "cap_overflow" in section_sender_kinds,
            "Release check requires a dense cap-overflow sender report with skipped candidates and section crops, not just an empty-frame report.",
            values={"evidence_kinds": sorted(str(kind) for kind in section_sender_kinds if kind)},
        )
        add_check(
            checks,
            "section_sender",
            "empty-frame section sender evidence supplied",
            "empty_frame" in section_sender_kinds,
            "Release check requires a no-candidate empty-frame sender report with enable_empty_frame_section=true and section crops, not just a cap-overflow report.",
            values={"evidence_kinds": sorted(str(kind) for kind in section_sender_kinds if kind)},
        )
        add_check(
            checks,
            "section_sender",
            "risk-frame section sender evidence supplied",
            "risk_frame" in section_sender_kinds,
            "Release check requires a sparse weak/low-coverage sender report with enable_risk_frame_section=true, so removed full-frame fallback risk is covered by section crops.",
            values={"evidence_kinds": sorted(str(kind) for kind in section_sender_kinds if kind)},
        )

    if not args.strategy_selection:
        add_check(
            checks,
            "strategy_selection",
            "observation strategy selection supplied",
            False,
            "Release check requires a strategy-selection report so full-frame, crop-only, cap, fallback, and section strategies are compared in one reviewed-evidence table.",
        )
    for strategy_selection in args.strategy_selection:
        strategy_path = resolve_file(strategy_selection, "strategy_selection.json")
        if not strategy_path.is_file():
            add_check(checks, "strategy_selection", f"{strategy_selection}: strategy selection exists", False, "Missing strategy_selection.json.")
            continue
        loaded = load_json(strategy_path)
        if isinstance(loaded, dict):
            strategy_selection_summaries.append(summarize_strategy_selection(strategy_path, loaded, checks, args))
        else:
            add_check(checks, "strategy_selection", f"{strategy_selection}: strategy selection object", False, "Strategy selection report is not a JSON object.")

    if not args.runtime_perf:
        add_check(
            checks,
            "runtime",
            "runtime perf evidence supplied",
            False,
            "Release check requires live/simulator runtime telemetry so local latency, cache reuse, and rect-only transport are gated before iOS promotion.",
        )
    for runtime_perf in args.runtime_perf:
        runtime_path = resolve_file(runtime_perf, "summary.json")
        if not runtime_path.is_file():
            add_check(checks, "runtime", f"{runtime_perf}: summary exists", False, "Missing runtime perf summary.json.")
            continue
        loaded = load_json(runtime_path)
        if isinstance(loaded, dict):
            runtime_perf_summaries.append(summarize_runtime_perf(runtime_path, loaded, checks, args))
        else:
            add_check(checks, "runtime", f"{runtime_perf}: summary object", False, "Runtime perf summary is not a JSON object.")

    if args.xcode_build_log:
        log_path = args.xcode_build_log
        if not log_path.is_file():
            add_check(checks, "ios", "Xcode build log exists", False, "Provided Xcode build log does not exist.")
        else:
            text = read_text_flexible(log_path)
            add_check(
                checks,
                "ios",
                "Xcode build succeeded",
                "BUILD SUCCEEDED" in text or "** BUILD SUCCEEDED **" in text,
                "Latest iOS build log must show a successful build after bundling the detector.",
            )
    else:
        add_check(
            checks,
            "ios",
            "Xcode build proof",
            False,
            "No --xcode-build-log supplied; run a Mac/Xcode build after the model is bundled.",
            "warning",
        )

    hard_failures = [check for check in checks if check["severity"] == "hard" and not check["passed"]]
    warnings = [check for check in checks if check["severity"] == "warning" and not check["passed"]]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release_ready": not hard_failures,
        "thresholds": {
            "min_folds": args.min_folds,
            "min_recall": args.min_recall,
            "min_precision": args.min_precision,
            "max_fp_per_image": args.max_fp_per_image,
            "max_rect_overhead_ratio": args.max_rect_overhead_ratio,
            "max_crop_pixel_ratio": args.max_crop_pixel_ratio,
            "max_layout_median_ms": args.max_layout_median_ms,
            "min_replay_images": args.min_replay_images,
            "min_dedupe_gt_keys": args.min_dedupe_gt_keys,
            "min_dedupe_gt_recall": args.min_dedupe_gt_recall,
            "max_dedupe_lost_gt_keys": args.max_dedupe_lost_gt_keys,
            "max_dedupe_false_merges": args.max_dedupe_false_merges,
            "max_dedupe_unknown_gt_suppressions": args.max_dedupe_unknown_gt_suppressions,
            "min_fallback_gt_keys": args.min_fallback_gt_keys,
            "min_fallback_gt_recall": args.min_fallback_gt_recall,
            "max_fallback_total_pixels_vs_full": args.max_fallback_total_pixels_vs_full,
            "max_fallback_image_ratio": args.max_fallback_image_ratio,
            "min_candidate_eval_gt_keys": args.min_candidate_eval_gt_keys,
            "min_candidate_eval_gt_recall": args.min_candidate_eval_gt_recall,
            "max_candidate_eval_crop_pixels_vs_full": args.max_candidate_eval_crop_pixels_vs_full,
            "max_candidate_eval_overcrop_count": args.max_candidate_eval_overcrop_count,
            "min_candidate_cap_gt_keys": args.min_candidate_cap_gt_keys,
            "min_candidate_cap_gt_recall": args.min_candidate_cap_gt_recall,
            "max_candidate_cap_total_pixels_vs_full": args.max_candidate_cap_total_pixels_vs_full,
            "min_section_sender_gt_keys": args.min_section_sender_gt_keys,
            "min_section_sender_gt_recall": args.min_section_sender_gt_recall,
            "max_section_sender_total_pixels_vs_full": args.max_section_sender_total_pixels_vs_full,
            "max_section_sender_fallback_image_ratio": args.max_section_sender_fallback_image_ratio,
            "min_risk_section_gt_keys": args.min_risk_section_gt_keys,
            "min_risk_section_gt_recall": args.min_risk_section_gt_recall,
            "max_risk_section_total_pixels_vs_full": args.max_risk_section_total_pixels_vs_full,
            "max_risk_section_fallback_image_ratio": args.max_risk_section_fallback_image_ratio,
            "min_strategy_gt_keys": args.min_strategy_gt_keys,
            "min_strategy_gt_recall": args.min_strategy_gt_recall,
            "max_strategy_total_pixels_vs_full": args.max_strategy_total_pixels_vs_full,
            "max_strategy_fallback_image_ratio": args.max_strategy_fallback_image_ratio,
            "min_runtime_records": args.min_runtime_records,
            "max_runtime_analysis_p95_ms": args.max_runtime_analysis_p95_ms,
            "max_runtime_server_p95_ms": args.max_runtime_server_p95_ms,
            "max_runtime_fresh_ocr_ratio": args.max_runtime_fresh_ocr_ratio,
            "max_runtime_attached_crop_files": args.max_runtime_attached_crop_files,
            "allow_runtime_fixture": bool(args.allow_runtime_fixture),
            "threshold_tolerance": args.threshold_tolerance,
        },
        "inputs": {
            "datasets": [str(path) for path in args.dataset],
            "selection": str(args.selection),
            "train_manifests": [str(path) for path in args.train_manifest],
            "replays": [str(path) for path in args.replay],
            "dedupe_tunes": [str(path) for path in args.dedupe_tune],
            "fallback_tunes": [str(path) for path in args.fallback_tune],
            "candidate_evals": [str(path) for path in args.candidate_eval],
            "candidate_cap_sweeps": [str(path) for path in args.candidate_cap_sweep],
            "section_sender_evals": [str(path) for path in args.section_sender_eval],
            "strategy_selections": [str(path) for path in args.strategy_selection],
            "runtime_perf": [str(path) for path in args.runtime_perf],
            "ios_root": str(args.ios_root),
            "backend_main": str(args.backend_main),
            "xcode_build_log": str(args.xcode_build_log) if args.xcode_build_log else "",
        },
        "datasets": dataset_summaries,
        "selection": selection_summary,
        "train_manifests": train_summaries,
        "coreml_artifacts": [str(path) for path in artifact_paths],
        "ios_model_resources": [str(path) for path in ios_resources],
        "swift_detector_config": swift_config,
        "swift_observation_dedupe_config": swift_dedupe_config,
        "backend_question_source_binding_config": backend_source_binding,
        "replays": replay_summaries,
        "dedupe_tunes": dedupe_tune_summaries,
        "fallback_tunes": fallback_tune_summaries,
        "candidate_evals": candidate_eval_summaries,
        "candidate_cap_sweeps": candidate_cap_summaries,
        "section_sender_evals": section_sender_summaries,
        "strategy_selections": strategy_selection_summaries,
        "runtime_perf": runtime_perf_summaries,
        "checks": checks,
        "hard_failures": hard_failures,
        "warnings": warnings,
    }
    write_json(args.out / "release_check.json", report)
    (args.out / "release_check.md").write_text(markdown_report(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate a question detector candidate before iOS release.")
    parser.add_argument("--dataset", type=Path, action="append", default=[], help="Detector dataset root or audit.json.")
    parser.add_argument("--selection", type=Path, required=True, help="selection_summary.json or its parent directory.")
    parser.add_argument("--train-manifest", type=Path, action="append", default=[], help="train_manifest.json or run directory.")
    parser.add_argument("--replay", type=Path, action="append", default=[], help="question_observation_replay root or summary.json.")
    parser.add_argument("--dedupe-tune", type=Path, action="append", default=[], help="question_observation_dedupe_tune root or summary.json.")
    parser.add_argument("--fallback-tune", type=Path, action="append", default=[], help="question_observation_fallback_tune root or summary.json.")
    parser.add_argument("--candidate-eval", type=Path, action="append", default=[], help="question_observation_candidate_eval root or summary.json.")
    parser.add_argument("--candidate-cap-sweep", type=Path, action="append", default=[], help="question_observation_candidate_cap_sweep root or summary.json.")
    parser.add_argument("--section-sender-eval", type=Path, action="append", default=[], help="question_observation_section_sender_eval root or summary.json.")
    parser.add_argument("--strategy-selection", type=Path, action="append", default=[], help="question_observation_strategy_selector root or strategy_selection.json.")
    parser.add_argument("--runtime-perf", type=Path, action="append", default=[], help="question_observation_runtime_perf_check root or summary.json.")
    parser.add_argument("--ios-root", type=Path, default=Path("ios/PXJ/App"))
    parser.add_argument("--backend-main", type=Path, default=Path("backend/app/main.py"), help="Backend main.py used for static question-crop source binding checks.")
    parser.add_argument("--xcode-build-log", type=Path)
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-release-check"))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--min-folds", type=int, default=3)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-precision", type=float, default=0.9)
    parser.add_argument("--max-fp-per-image", type=float, default=0.05)
    parser.add_argument("--max-rect-overhead-ratio", type=float, default=1.005)
    parser.add_argument("--max-crop-pixel-ratio", type=float, default=0.75)
    parser.add_argument("--max-layout-median-ms", type=float, default=80.0)
    parser.add_argument("--min-replay-images", type=int, default=30)
    parser.add_argument("--min-dedupe-gt-keys", type=int, default=10)
    parser.add_argument("--min-dedupe-gt-recall", type=float, default=0.999)
    parser.add_argument("--max-dedupe-lost-gt-keys", type=int, default=0)
    parser.add_argument("--max-dedupe-false-merges", type=int, default=0)
    parser.add_argument("--max-dedupe-unknown-gt-suppressions", type=int, default=0)
    parser.add_argument("--min-fallback-gt-keys", type=int, default=10)
    parser.add_argument("--min-fallback-gt-recall", type=float, default=0.999)
    parser.add_argument("--max-fallback-total-pixels-vs-full", type=float, default=1.05)
    parser.add_argument("--max-fallback-image-ratio", type=float, default=0.50)
    parser.add_argument("--max-fallback-delta-crop-missed-gt", type=int, default=0)
    parser.add_argument("--min-candidate-eval-gt-keys", type=int, default=10)
    parser.add_argument("--min-candidate-eval-gt-recall", type=float, default=0.999)
    parser.add_argument("--max-candidate-eval-crop-pixels-vs-full", type=float, default=0.75)
    parser.add_argument("--max-candidate-eval-overcrop-count", type=int, default=0)
    parser.add_argument("--min-candidate-cap-gt-keys", type=int, default=10)
    parser.add_argument("--min-candidate-cap-gt-recall", type=float, default=0.999)
    parser.add_argument("--max-candidate-cap-total-pixels-vs-full", type=float, default=1.05)
    parser.add_argument("--min-section-sender-gt-keys", type=int, default=200)
    parser.add_argument("--min-section-sender-gt-recall", type=float, default=0.999)
    parser.add_argument("--max-section-sender-total-pixels-vs-full", type=float, default=0.75)
    parser.add_argument("--max-section-sender-fallback-image-ratio", type=float, default=0.05)
    parser.add_argument("--min-risk-section-gt-keys", type=int, default=10)
    parser.add_argument("--min-risk-section-gt-recall", type=float, default=0.999)
    parser.add_argument("--max-risk-section-total-pixels-vs-full", type=float, default=0.90)
    parser.add_argument("--max-risk-section-fallback-image-ratio", type=float, default=0.25)
    parser.add_argument("--min-strategy-gt-keys", type=int, default=10)
    parser.add_argument("--min-strategy-gt-recall", type=float, default=0.999)
    parser.add_argument("--max-strategy-total-pixels-vs-full", type=float, default=0.90)
    parser.add_argument("--max-strategy-fallback-image-ratio", type=float, default=0.25)
    parser.add_argument("--min-runtime-records", type=int, default=5)
    parser.add_argument("--max-runtime-analysis-p95-ms", type=float, default=300.0)
    parser.add_argument("--max-runtime-server-p95-ms", type=float, default=250.0)
    parser.add_argument("--max-runtime-fresh-ocr-ratio", type=float, default=0.75)
    parser.add_argument("--max-runtime-attached-crop-files", type=int, default=0)
    parser.add_argument("--allow-runtime-fixture", action="store_true", help="Smoke only: allow runtime_perf evidence_kind=fixture to exercise release-gate runtime checks.")
    parser.add_argument("--threshold-tolerance", type=float, default=0.0001)
    parser.add_argument("--allow-failed-gate", action="store_true", help="Write the report but exit 0 even when release_ready=false.")
    args = parser.parse_args()
    report = run_check(args)
    print(
        json.dumps(
            {
                "release_ready": report["release_ready"],
                "hard_failures": len(report["hard_failures"]),
                "warnings": len(report["warnings"]),
                "report": str(args.out / "release_check.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not report["release_ready"] and not args.allow_failed_gate:
        sys.exit(1)


if __name__ == "__main__":
    main()
