# Question Detector Release Check

- Generated: 2026-06-30T05:34:37.121764+00:00
- Release ready: `false`
- Hard failures: 7
- Warnings: 2

## Hard Failures

- [dataset] diagnostics\question-detector-active-batch-kpai-001-approved-smoke\audit.json: model_training_ready: Dataset audit must report model_training_ready=true before detector selection is considered releasable.
- [selection] cross-validation fold count: Selected candidate must aggregate at least 3 folds.
- [training] diagnostics\question-detector-train-select-smoke\train_manifest.json: ready source dataset: Selected train manifest must come from an audit-ready dataset, not a smoke or unready dataset.
- [artifact] Core ML artifact exists: Selected detector must have an exported QuestionRegionDetector Core ML artifact.
- [ios] QuestionRegionDetector bundled in iOS app sources: iOS target must include QuestionRegionDetector.mlmodel/.mlpackage or compiled .mlmodelc.
- [ios] Swift fast confidence matches selected threshold: detectorFastMinConfidence must match selection_summary iOS recommendation.
- [ios] Swift accurate confidence matches selected threshold: detectorAccurateMinConfidence must match selection_summary iOS recommendation.

## Warnings

- [dataset] diagnostics\question-detector-active-batch-kpai-001-approved-smoke\audit.json: audit warnings: Dataset audit contains 4 warning(s); inspect before relying on the release report.
- [ios] Xcode build proof: No --xcode-build-log supplied; run a Mac/Xcode build after the model is bundled.

## Passed Checks

- [dataset] diagnostics\question-detector-active-batch-kpai-001-approved-smoke\audit.json: no audit error
- [selection] selection has best candidate
- [selection] all selected folds pass eval gate
- [selection] minimum recall
- [selection] mean precision
- [selection] max false positives per image
- [selection] iOS threshold recommendation
- [training] diagnostics\question-detector-train-select-smoke\train_manifest.json: trained status
- [ios] Swift detector hook exists
- [dedupe] Swift weak-layout text guard exists
- [backend] question crop source input_index binding exists
- [replay] diagnostics\question-observation-replay-kpai-layout-measured\summary.json: image coverage
- [replay] diagnostics\question-observation-replay-kpai-layout-measured\summary.json: rect-only network overhead
- [replay] diagnostics\question-observation-replay-kpai-layout-measured\summary.json: crop VLM pixel ratio
- [replay] diagnostics\question-observation-replay-kpai-layout-measured\summary.json: local layout median latency
- [dedupe] diagnostics\question-observation-dedupe-tune-kpai\summary.json: reviewed GT available
- [dedupe] diagnostics\question-observation-dedupe-tune-kpai\summary.json: recommendation exists
- [dedupe] diagnostics\question-observation-dedupe-tune-kpai\summary.json: recommendation GT recall
- [dedupe] diagnostics\question-observation-dedupe-tune-kpai\summary.json: recommendation lost GT keys
- [dedupe] diagnostics\question-observation-dedupe-tune-kpai\summary.json: recommendation false merges
- [dedupe] diagnostics\question-observation-dedupe-tune-kpai\summary.json: recommendation unknown GT suppressions
- [dedupe] diagnostics\question-observation-dedupe-tune-kpai\summary.json: recommendation rect-only overhead
- [dedupe] diagnostics\question-observation-dedupe-tune-kpai\summary.json: recommendation crop pixel ratio
- [dedupe] diagnostics\question-observation-dedupe-tune-kpai\summary.json: Swift text-guarded weak-key baseline exists
- [dedupe] diagnostics\question-observation-dedupe-tune-kpai\summary.json: Swift text-guarded GT recall
- [dedupe] diagnostics\question-observation-dedupe-tune-kpai\summary.json: Swift text-guarded lost GT keys
- [dedupe] diagnostics\question-observation-dedupe-tune-kpai\summary.json: Swift text-guarded unknown GT suppressions
- [fallback] diagnostics\question-observation-fallback-tune-kpai\summary.json: backend current policy exists
- [fallback] diagnostics\question-observation-fallback-tune-kpai\summary.json: reviewed GT available
- [fallback] diagnostics\question-observation-fallback-tune-kpai\summary.json: backend current GT recall
- [fallback] diagnostics\question-observation-fallback-tune-kpai\summary.json: backend current VLM pixel ratio
- [fallback] diagnostics\question-observation-fallback-tune-kpai\summary.json: backend current fallback image ratio
- [candidate_eval] diagnostics\question-observation-candidate-eval-kpai\summary.json: reviewed GT available
- [candidate_eval] diagnostics\question-observation-candidate-eval-kpai\summary.json: represented GT recall
- [candidate_eval] diagnostics\question-observation-candidate-eval-kpai\summary.json: crop VLM pixel ratio
- [candidate_eval] diagnostics\question-observation-candidate-eval-kpai\summary.json: overcrop candidates
