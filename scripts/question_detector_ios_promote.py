"""Promote a selected question detector Core ML artifact into the iOS app.

This is the "last mile" helper after model selection has already passed. It
does not replace the release gate. It copies the selected
QuestionRegionDetector model into ios/PXJ/App and syncs detector confidence
constants in QuestionSegmenter.swift only when --apply is explicit.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODEL_NAME = "QuestionRegionDetector"
MODEL_SUFFIXES = (".mlmodel", ".mlpackage", ".mlmodelc")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_file(path: Path, default_name: str) -> Path:
    return path / default_name if path.is_dir() else path


def resolve_existing(raw: str, anchors: list[Path]) -> Path | None:
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
                return candidate.resolve()
        except OSError:
            continue
    return None


def metric_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def discover_manifests(paths: list[Path]) -> list[Path]:
    manifests: list[Path] = []
    for path in paths:
        if path.is_file():
            manifests.append(path)
        elif path.is_dir():
            direct = path / "train_manifest.json"
            if direct.is_file():
                manifests.append(direct)
            manifests.extend(sorted(path.rglob("train_manifest.json")))
    unique: dict[str, Path] = {}
    for path in manifests:
        unique[str(path.resolve())] = path
    return sorted(unique.values())


def selected_run_manifests(selection: dict[str, Any], selection_path: Path) -> list[Path]:
    best = selection.get("best") if isinstance(selection.get("best"), dict) else {}
    runs = best.get("runs") if isinstance(best.get("runs"), list) else []
    manifests: list[Path] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        resolved = resolve_existing(str(run.get("manifest") or ""), [selection_path.parent, Path.cwd()])
        if resolved is not None:
            manifests.append(resolved)
    unique = {str(path.resolve()): path for path in manifests}
    return sorted(unique.values())


def artifact_from_selection(selection: dict[str, Any], selection_path: Path) -> Path | None:
    best = selection.get("best") if isinstance(selection.get("best"), dict) else {}
    runs = best.get("runs") if isinstance(best.get("runs"), list) else []
    for run in runs:
        if not isinstance(run, dict):
            continue
        artifact = resolve_existing(str(run.get("coreml_artifact") or ""), [selection_path.parent, Path.cwd()])
        if artifact is not None:
            return artifact
    return None


def artifact_from_manifests(manifests: list[Path]) -> Path | None:
    for manifest_path in manifests:
        try:
            manifest = read_json(manifest_path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        artifact = resolve_existing(str(manifest.get("coreml_artifact") or ""), [manifest_path.parent, Path.cwd()])
        if artifact is not None:
            return artifact
    return None


def validate_artifact(path: Path | None) -> tuple[bool, str]:
    if path is None:
        return False, "No Core ML artifact was found in --artifact, selection runs, or train manifests."
    if path.suffix not in MODEL_SUFFIXES:
        return False, f"Unsupported Core ML suffix: {path.suffix}. Expected one of {MODEL_SUFFIXES}."
    if path.suffix in {".mlpackage", ".mlmodelc"} and not path.is_dir():
        return False, f"{path.suffix} artifact must be a directory: {path}"
    if path.suffix == ".mlmodel" and not path.is_file():
        return False, f".mlmodel artifact must be a file: {path}"
    return True, "artifact exists"


def parse_swift_config(swift_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(swift_path),
        "exists": swift_path.is_file(),
        "bundle_lookup_mlmodelc": False,
        "fast_min_confidence": None,
        "accurate_min_confidence": None,
    }
    if not swift_path.is_file():
        return result
    text = swift_path.read_text(encoding="utf-8")
    result["bundle_lookup_mlmodelc"] = (
        'forResource: "QuestionRegionDetector"' in text and 'withExtension: "mlmodelc"' in text
    )
    patterns = {
        "fast_min_confidence": r"(detectorFastMinConfidence\s*:[^=]+=\s*)([0-9.]+)",
        "accurate_min_confidence": r"(detectorAccurateMinConfidence\s*:[^=]+=\s*)([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            result[key] = metric_float(match.group(2))
    return result


def sync_swift_thresholds(swift_path: Path, fast: float, accurate: float) -> bool:
    text = swift_path.read_text(encoding="utf-8")
    replacements = {
        r"(detectorFastMinConfidence\s*:[^=]+=\s*)([0-9.]+)": f"\\g<1>{fast:.4g}",
        r"(detectorAccurateMinConfidence\s*:[^=]+=\s*)([0-9.]+)": f"\\g<1>{accurate:.4g}",
    }
    changed = False
    for pattern, replacement in replacements.items():
        new_text, count = re.subn(pattern, replacement, text, count=1)
        if count:
            changed = changed or new_text != text
            text = new_text
    if changed:
        swift_path.write_text(text, encoding="utf-8")
    return changed


def existing_ios_models(ios_root: Path) -> list[Path]:
    models: list[Path] = []
    if not ios_root.exists():
        return models
    for suffix in MODEL_SUFFIXES:
        models.extend(sorted(ios_root.rglob(f"{MODEL_NAME}{suffix}")))
    return sorted({str(path.resolve()): path for path in models}.values())


def copy_artifact(source: Path, ios_root: Path) -> Path:
    target = ios_root / f"{MODEL_NAME}{source.suffix}"
    for existing in existing_ios_models(ios_root):
        if existing.name.startswith(MODEL_NAME):
            if existing.is_dir():
                shutil.rmtree(existing)
            else:
                existing.unlink()
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return target


def selection_summary(selection: dict[str, Any]) -> dict[str, Any]:
    best = selection.get("best") if isinstance(selection.get("best"), dict) else {}
    ios = selection.get("ios_recommendation") if isinstance(selection.get("ios_recommendation"), dict) else {}
    return {
        "has_best": bool(best),
        "fold_count": int(best.get("fold_count") or 0),
        "all_folds_pass_eval_gate": bool(best.get("all_folds_pass_eval_gate")),
        "min_recall": metric_float(best.get("min_recall")),
        "mean_precision": metric_float(best.get("mean_precision")),
        "max_fp_per_image": metric_float(best.get("max_fp_per_image")),
        "min_score": metric_float(best.get("min_score")),
        "ios_recommendation": ios,
    }


def add_blocker(blockers: list[dict[str, Any]], code: str, detail: str, values: dict[str, Any] | None = None) -> None:
    blockers.append({"code": code, "detail": detail, "values": values or {}})


def promote(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean and args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    selection_path = resolve_file(args.selection, "selection_summary.json")
    selection = read_json(selection_path)
    selected = selection_summary(selection)
    manifests = discover_manifests(args.train_manifest)
    for manifest in selected_run_manifests(selection, selection_path):
        if str(manifest.resolve()) not in {str(item.resolve()) for item in manifests}:
            manifests.append(manifest)

    artifact = args.artifact.resolve() if args.artifact else None
    if artifact is None:
        artifact = artifact_from_selection(selection, selection_path)
    if artifact is None:
        artifact = artifact_from_manifests(manifests)

    artifact_ok, artifact_detail = validate_artifact(artifact)
    ios_root = args.ios_root
    swift_path = args.swift or (ios_root / "QuestionSegmenter.swift")
    swift_config = parse_swift_config(swift_path)
    ios_rec = selected.get("ios_recommendation") if isinstance(selected.get("ios_recommendation"), dict) else {}
    fast = ios_rec.get("question_region_detector_fast_min_confidence") if isinstance(ios_rec, dict) else None
    accurate = ios_rec.get("question_region_detector_accurate_min_confidence") if isinstance(ios_rec, dict) else None

    blockers: list[dict[str, Any]] = []
    if not selected["has_best"]:
        add_blocker(blockers, "no_selected_candidate", "selection_summary.json has no best candidate.")
    if selected["fold_count"] < args.min_folds and not args.allow_failed_selection:
        add_blocker(blockers, "too_few_folds", f"Selected candidate has {selected['fold_count']} folds; required >= {args.min_folds}.", {"selection": selected})
    if not selected["all_folds_pass_eval_gate"] and not args.allow_failed_selection:
        add_blocker(blockers, "selection_failed_eval_gate", "Selected candidate did not pass all fold eval gates.", {"selection": selected})
    if not artifact_ok:
        add_blocker(blockers, "missing_coreml_artifact", artifact_detail, {"artifact": str(artifact) if artifact else ""})
    if not ios_root.is_dir():
        add_blocker(blockers, "missing_ios_root", f"iOS app root does not exist: {ios_root}")
    if not swift_config["exists"]:
        add_blocker(blockers, "missing_question_segmenter", f"QuestionSegmenter.swift does not exist: {swift_path}")
    elif not swift_config["bundle_lookup_mlmodelc"]:
        add_blocker(blockers, "swift_missing_bundle_lookup", "QuestionSegmenter.swift must load QuestionRegionDetector.mlmodelc from Bundle.main.")
    if fast is None or accurate is None:
        add_blocker(blockers, "missing_ios_thresholds", "selection_summary.json does not include iOS fast/accurate confidence recommendations.")

    can_apply = not blockers
    copied_to = ""
    swift_changed = False
    if args.apply and can_apply:
        copied = copy_artifact(artifact, ios_root)  # type: ignore[arg-type]
        copied_to = str(copied)
        swift_changed = sync_swift_thresholds(swift_path, float(fast), float(accurate))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "apply": bool(args.apply),
        "can_apply": bool(can_apply),
        "applied": bool(args.apply and can_apply),
        "selection": selected,
        "selection_path": str(selection_path),
        "train_manifests": [str(path) for path in manifests],
        "artifact": str(artifact) if artifact else "",
        "artifact_ok": artifact_ok,
        "artifact_detail": artifact_detail,
        "ios_root": str(ios_root),
        "existing_ios_models_before": [str(path) for path in existing_ios_models(ios_root)],
        "swift_config_before": swift_config,
        "copied_to": copied_to,
        "swift_changed": swift_changed,
        "blockers": blockers,
        "next_steps": [
            "Run XcodeGen/Xcode build on the Mac after applying a real model.",
            "Run question_detector_release_check.py with the post-bundle Xcode build log.",
            "Validate live telemetry: coreml_model_loaded_frame_count, selected_segmenter_counts, fallback_to_ocr_frame_count, and paired IoU.",
        ],
    }
    write_json(args.out / "ios_promote_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote selected Core ML question detector into the iOS app.")
    parser.add_argument("--selection", type=Path, required=True, help="selection_summary.json or selection directory.")
    parser.add_argument("--train-manifest", type=Path, action="append", default=[], help="train_manifest.json or run directory.")
    parser.add_argument("--artifact", type=Path, help="Explicit Core ML artifact to install.")
    parser.add_argument("--ios-root", type=Path, default=Path("ios/PXJ/App"))
    parser.add_argument("--swift", type=Path, help="QuestionSegmenter.swift path; defaults under --ios-root.")
    parser.add_argument("--out", type=Path, default=Path("diagnostics/question-detector-ios-promote"))
    parser.add_argument("--min-folds", type=int, default=3)
    parser.add_argument("--allow-failed-selection", action="store_true", help="Diagnostics only; do not use for release.")
    parser.add_argument("--apply", action="store_true", help="Actually copy the model and update Swift thresholds.")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    report = promote(args)
    print(
        json.dumps(
            {
                "can_apply": report["can_apply"],
                "applied": report["applied"],
                "artifact": report["artifact"],
                "blockers": [item["code"] for item in report["blockers"]],
                "out": str(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
