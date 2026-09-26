# Question Detector Release Check

- Generated: 2026-06-30T04:12:26.183981+00:00
- Release ready: `false`
- Hard failures: 11
- Warnings: 2

## Hard Failures

- [dataset] diagnostics\question-detector-bootstrap-kpai-smoke\audit.json: model_training_ready: Dataset audit must report model_training_ready=true before detector selection is considered releasable.
- [selection] cross-validation fold count: Selected candidate must aggregate at least 3 folds.
- [selection] all selected folds pass eval gate: Every selected fold must pass the detector eval gate.
- [selection] minimum recall: Worst-fold recall must be >= 0.950.
- [selection] mean precision: Mean precision must be >= 0.900.
- [training] diagnostics\question-detector-loop-bootstrap-smoke\experiment\runs\question-detector-bootstrap-kpai-smoke__yolo11n__img416__seed42__ep1\train_manifest.json: trained status: Selected train manifest must represent a completed training/evaluation run.
- [training] diagnostics\question-detector-loop-bootstrap-smoke\experiment\runs\question-detector-bootstrap-kpai-smoke__yolo11n__img416__seed42__ep1\train_manifest.json: ready source dataset: Selected train manifest must come from an audit-ready dataset, not a smoke or unready dataset.
- [artifact] Core ML artifact exists: Selected detector must have an exported QuestionRegionDetector Core ML artifact.
- [ios] QuestionRegionDetector bundled in iOS app sources: iOS target must include QuestionRegionDetector.mlmodel/.mlpackage or compiled .mlmodelc.
- [ios] Swift fast confidence matches selected threshold: detectorFastMinConfidence must match selection_summary iOS recommendation.
- [ios] Swift accurate confidence matches selected threshold: detectorAccurateMinConfidence must match selection_summary iOS recommendation.

## Warnings

- [dataset] diagnostics\question-detector-bootstrap-kpai-smoke\audit.json: audit warnings: Dataset audit contains 1 warning(s); inspect before relying on the release report.
- [ios] Xcode build proof: No --xcode-build-log supplied; run a Mac/Xcode build after the model is bundled.

## Passed Checks

- [dataset] diagnostics\question-detector-bootstrap-kpai-smoke\audit.json: no audit error
- [selection] selection has best candidate
- [selection] max false positives per image
- [selection] iOS threshold recommendation
- [ios] Swift detector hook exists
- [replay] diagnostics\question-observation-replay-kpai-layout-measured\summary.json: image coverage
- [replay] diagnostics\question-observation-replay-kpai-layout-measured\summary.json: rect-only network overhead
- [replay] diagnostics\question-observation-replay-kpai-layout-measured\summary.json: crop VLM pixel ratio
- [replay] diagnostics\question-observation-replay-kpai-layout-measured\summary.json: local layout median latency
