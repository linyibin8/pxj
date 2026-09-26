# Question Detector Release Check

- Generated: 2026-07-01T08:26:24.847445+00:00
- Release ready: `false`
- Hard failures: 10
- Warnings: 1

## Hard Failures

- [selection] selection summary exists: Missing selection_summary.json.
- [artifact] Core ML artifact exists: Selected detector must have an exported QuestionRegionDetector Core ML artifact.
- [ios] QuestionRegionDetector bundled in iOS app sources: iOS target must include QuestionRegionDetector.mlmodel/.mlpackage or compiled .mlmodelc.
- [dedupe] dedupe tuning supplied: Release check requires at least one --dedupe-tune report so local duplicate suppression cannot hide missed questions.
- [fallback] fallback policy tuning supplied: Release check requires at least one --fallback-tune report so full-frame fallback cost cannot silently regress.
- [candidate_eval] observation candidate eval supplied: Release check requires at least one --candidate-eval report so local crop-candidate recall cannot silently regress.
- [candidate_cap] candidate cap sweep supplied: Release check requires at least one --candidate-cap-sweep report so the iOS per-frame crop cap cannot silently drop questions or inflate fallback cost.
- [section_sender] section sender eval supplied: Dense iOS section sender evidence is required so static sender code cannot ship without recall/cost proof.
- [strategy_selection] observation strategy selection supplied: Release check requires a strategy-selection report so full-frame, crop-only, cap, fallback, and section strategies are compared in one reviewed-evidence table.
- [runtime] runtime perf evidence supplied: Release check requires live/simulator runtime telemetry so local latency, cache reuse, and rect-only transport are gated before iOS promotion.

## Warnings

- [ios] Xcode build proof: No --xcode-build-log supplied; run a Mac/Xcode build after the model is bundled.

## Passed Checks

- [ios] Swift detector hook exists
- [ios] Swift detector OCR merge fallback gate exists
- [dedupe] Swift weak-layout text guard exists
- [ios] Swift ranked candidate cap telemetry exists
- [ios] Swift dense section crop sender exists
- [ios] Swift empty-frame section dedupe exists
- [ios] Swift Vision/JPEG resizing is pixel-bound
- [backend] question crop source input_index binding exists
- [backend] question crop candidate cap fallback is risk-aware
- [backend] weak layout crop keys do not pre-dedupe extraction sources
- [backend] section crop sources stay source-safe
