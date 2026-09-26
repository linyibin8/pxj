# Question Detector Release Check

- Generated: 2026-06-30T09:51:54.733856+00:00
- Release ready: `false`
- Hard failures: 11
- Warnings: 1

## Hard Failures

- [dataset] diagnostics\question-detector-dataset-merged-reviewed-next-propagated-exact\audit.json: model_training_ready: Dataset audit must report model_training_ready=true before detector selection is considered releasable.
- [selection] cross-validation fold count: Selected candidate must aggregate at least 3 folds.
- [selection] all selected folds pass eval gate: Every selected fold must pass the detector eval gate.
- [selection] minimum recall: Worst-fold recall must be >= 0.950.
- [selection] mean precision: Mean precision must be >= 0.900.
- [training] diagnostics\question-detector-experiment-actual-smoke\runs\fold-00__yolo11n__img320__seed42__ep1\train_manifest.json: trained status: Selected train manifest must represent a completed training/evaluation run.
- [training] diagnostics\question-detector-experiment-actual-smoke\runs\fold-00__yolo11n__img320__seed42__ep1\train_manifest.json: ready source dataset: Selected train manifest must come from an audit-ready dataset, not a smoke or unready dataset.
- [artifact] Core ML artifact exists: Selected detector must have an exported QuestionRegionDetector Core ML artifact.
- [ios] QuestionRegionDetector bundled in iOS app sources: iOS target must include QuestionRegionDetector.mlmodel/.mlpackage or compiled .mlmodelc.
- [ios] Swift fast confidence matches selected threshold: detectorFastMinConfidence must match selection_summary iOS recommendation.
- [ios] Swift accurate confidence matches selected threshold: detectorAccurateMinConfidence must match selection_summary iOS recommendation.

## Warnings

- [dataset] diagnostics\question-detector-dataset-merged-reviewed-next-propagated-exact\audit.json: audit warnings: Dataset audit contains 3 warning(s); inspect before relying on the release report.

## Passed Checks

- [dataset] diagnostics\question-detector-dataset-merged-reviewed-next-propagated-exact\audit.json: no audit error
- [selection] selection has best candidate
- [selection] max false positives per image
- [selection] iOS threshold recommendation
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
- [fallback] diagnostics\question-observation-fallback-tune-cap-aware-kpai\summary.json: backend current policy exists
- [fallback] diagnostics\question-observation-fallback-tune-cap-aware-kpai\summary.json: reviewed GT available
- [fallback] diagnostics\question-observation-fallback-tune-cap-aware-kpai\summary.json: backend current GT recall
- [fallback] diagnostics\question-observation-fallback-tune-cap-aware-kpai\summary.json: backend current VLM pixel ratio
- [fallback] diagnostics\question-observation-fallback-tune-cap-aware-kpai\summary.json: backend current fallback image ratio
- [fallback] diagnostics\question-observation-fallback-tune-cap-aware-kpai\summary.json: backend current union-area shadow exists
- [candidate_eval] diagnostics\question-observation-candidate-eval-kpai\summary.json: reviewed GT available
- [candidate_eval] diagnostics\question-observation-candidate-eval-kpai\summary.json: represented GT recall
- [candidate_eval] diagnostics\question-observation-candidate-eval-kpai\summary.json: crop VLM pixel ratio
- [candidate_eval] diagnostics\question-observation-candidate-eval-kpai\summary.json: overcrop candidates
- [candidate_cap] diagnostics\question-observation-candidate-cap-sweep-kpai\summary.json: current cap row exists
- [candidate_cap] diagnostics\question-observation-candidate-cap-sweep-kpai\summary.json: represented GT available
- [candidate_cap] diagnostics\question-observation-candidate-cap-sweep-kpai\summary.json: current cap crop-only GT recall
- [candidate_cap] diagnostics\question-observation-candidate-cap-sweep-kpai\summary.json: current cap fallback-protected GT recall
- [candidate_cap] diagnostics\question-observation-candidate-cap-sweep-kpai\summary.json: current cap VLM pixel ratio
- [candidate_cap] diagnostics\question-observation-candidate-cap-sweep-kpai\summary.json: current cap skips no high-risk candidates
- [candidate_cap] diagnostics\question-observation-candidate-cap-sweep-kpai\summary.json: cap recommendation exists
- [evidence] reviewed replay evidence is same source
- [evidence] reviewed GT source is consistent across replay evidence
- [evidence] reviewed GT volume is consistent across replay evidence
- [section_sender] diagnostics\question-observation-section-sender-eval-synthetic-dense-oracle\summary.json: dense represented GT available
- [section_sender] diagnostics\question-observation-section-sender-eval-synthetic-dense-oracle\summary.json: combined crop GT recall
- [section_sender] diagnostics\question-observation-section-sender-eval-synthetic-dense-oracle\summary.json: fallback-protected GT recall
- [section_sender] diagnostics\question-observation-section-sender-eval-synthetic-dense-oracle\summary.json: dense VLM pixel ratio
- [section_sender] diagnostics\question-observation-section-sender-eval-synthetic-dense-oracle\summary.json: dense fallback image ratio
- [section_sender] diagnostics\question-observation-section-sender-eval-synthetic-dense-layout-empty-section\summary.json: dense represented GT available
- [section_sender] diagnostics\question-observation-section-sender-eval-synthetic-dense-layout-empty-section\summary.json: combined crop GT recall
- [section_sender] diagnostics\question-observation-section-sender-eval-synthetic-dense-layout-empty-section\summary.json: fallback-protected GT recall
- [section_sender] diagnostics\question-observation-section-sender-eval-synthetic-dense-layout-empty-section\summary.json: dense VLM pixel ratio
- [section_sender] diagnostics\question-observation-section-sender-eval-synthetic-dense-layout-empty-section\summary.json: dense fallback image ratio
- [section_sender] cap-overflow section sender evidence supplied
- [section_sender] empty-frame section sender evidence supplied
- [ios] Xcode build succeeded
