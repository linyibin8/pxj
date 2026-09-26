# Question Detector Release Check

- Generated: 2026-06-30T16:07:04.949526+00:00
- Release ready: `false`
- Hard failures: 5
- Warnings: 2

## Hard Failures

- [dataset] diagnostics\question-detector-dataset-merged-reviewed-next-propagated-exact\audit.json: model_training_ready: Dataset audit must report model_training_ready=true before detector selection is considered releasable.
- [selection] selection summary exists: Missing selection_summary.json.
- [artifact] Core ML artifact exists: Selected detector must have an exported QuestionRegionDetector Core ML artifact.
- [ios] QuestionRegionDetector bundled in iOS app sources: iOS target must include QuestionRegionDetector.mlmodel/.mlpackage or compiled .mlmodelc.
- [runtime] diagnostics\question-observation-runtime-perf-fixture-check\summary.json: runtime evidence is not a fixture: Runtime perf fixtures are allowed only for explicit release-gate smoke checks; release candidates need live/simulator telemetry.

## Warnings

- [dataset] diagnostics\question-detector-dataset-merged-reviewed-next-propagated-exact\audit.json: audit warnings: Dataset audit contains 3 warning(s); inspect before relying on the release report.
- [fallback] diagnostics\question-observation-fallback-tune-reviewed-propagated-layout-v5-gated-backend-tuned\summary.json: legacy-removed no-GT review queue: Legacy-removed fallback frames with no represented GT should be reviewed before treating fallback tuning evidence as production-complete.

## Passed Checks

- [dataset] diagnostics\question-detector-dataset-merged-reviewed-next-propagated-exact\audit.json: no audit error
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
- [replay] diagnostics\question-observation-replay-reviewed-propagated-layout-v5-gated\summary.json: image coverage
- [replay] diagnostics\question-observation-replay-reviewed-propagated-layout-v5-gated\summary.json: rect-only network overhead
- [replay] diagnostics\question-observation-replay-reviewed-propagated-layout-v5-gated\summary.json: crop VLM pixel ratio
- [replay] diagnostics\question-observation-replay-reviewed-propagated-layout-v5-gated\summary.json: local layout median latency
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
- [fallback] diagnostics\question-observation-fallback-tune-reviewed-propagated-layout-v5-gated-backend-tuned\summary.json: backend current policy exists
- [fallback] diagnostics\question-observation-fallback-tune-reviewed-propagated-layout-v5-gated-backend-tuned\summary.json: reviewed GT available
- [fallback] diagnostics\question-observation-fallback-tune-reviewed-propagated-layout-v5-gated-backend-tuned\summary.json: backend current GT recall
- [fallback] diagnostics\question-observation-fallback-tune-reviewed-propagated-layout-v5-gated-backend-tuned\summary.json: backend current VLM pixel ratio
- [fallback] diagnostics\question-observation-fallback-tune-reviewed-propagated-layout-v5-gated-backend-tuned\summary.json: backend current fallback image ratio
- [fallback] diagnostics\question-observation-fallback-tune-reviewed-propagated-layout-v5-gated-backend-tuned\summary.json: backend current union-area shadow exists
- [fallback] diagnostics\question-observation-fallback-tune-reviewed-propagated-layout-v5-gated-backend-tuned\summary.json: legacy-to-current delta exists
- [fallback] diagnostics\question-observation-fallback-tune-reviewed-propagated-layout-v5-gated-backend-tuned\summary.json: legacy-removed fallback has no known GT miss
- [candidate_eval] diagnostics\question-observation-candidate-eval-reviewed-propagated-layout-v5-gated\summary.json: reviewed GT available
- [candidate_eval] diagnostics\question-observation-candidate-eval-reviewed-propagated-layout-v5-gated\summary.json: represented GT recall
- [candidate_eval] diagnostics\question-observation-candidate-eval-reviewed-propagated-layout-v5-gated\summary.json: crop VLM pixel ratio
- [candidate_eval] diagnostics\question-observation-candidate-eval-reviewed-propagated-layout-v5-gated\summary.json: overcrop candidates
- [candidate_cap] diagnostics\question-observation-candidate-cap-sweep-reviewed-propagated-layout-v5-gated-backend-tuned\summary.json: current cap row exists
- [candidate_cap] diagnostics\question-observation-candidate-cap-sweep-reviewed-propagated-layout-v5-gated-backend-tuned\summary.json: represented GT available
- [candidate_cap] diagnostics\question-observation-candidate-cap-sweep-reviewed-propagated-layout-v5-gated-backend-tuned\summary.json: current cap crop-only GT recall
- [candidate_cap] diagnostics\question-observation-candidate-cap-sweep-reviewed-propagated-layout-v5-gated-backend-tuned\summary.json: current cap fallback-protected GT recall
- [candidate_cap] diagnostics\question-observation-candidate-cap-sweep-reviewed-propagated-layout-v5-gated-backend-tuned\summary.json: current cap VLM pixel ratio
- [candidate_cap] diagnostics\question-observation-candidate-cap-sweep-reviewed-propagated-layout-v5-gated-backend-tuned\summary.json: current cap skips no high-risk candidates
- [candidate_cap] diagnostics\question-observation-candidate-cap-sweep-reviewed-propagated-layout-v5-gated-backend-tuned\summary.json: cap recommendation exists
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
- [section_sender] diagnostics\question-observation-section-sender-eval-propagated-risk-section-top1-ios-signals\summary.json: dense represented GT available
- [section_sender] diagnostics\question-observation-section-sender-eval-propagated-risk-section-top1-ios-signals\summary.json: combined crop GT recall
- [section_sender] diagnostics\question-observation-section-sender-eval-propagated-risk-section-top1-ios-signals\summary.json: fallback-protected GT recall
- [section_sender] diagnostics\question-observation-section-sender-eval-propagated-risk-section-top1-ios-signals\summary.json: dense VLM pixel ratio
- [section_sender] diagnostics\question-observation-section-sender-eval-propagated-risk-section-top1-ios-signals\summary.json: dense fallback image ratio
- [section_sender] cap-overflow section sender evidence supplied
- [section_sender] empty-frame section sender evidence supplied
- [section_sender] risk-frame section sender evidence supplied
- [strategy_selection] diagnostics\question-observation-strategy-selection-risk-section-top1\strategy_selection.json: strategy rows available
- [strategy_selection] diagnostics\question-observation-strategy-selection-risk-section-top1\strategy_selection.json: reviewed live strategy passes recall/cost gate
- [strategy_selection] diagnostics\question-observation-strategy-selection-risk-section-top1\strategy_selection.json: propagated reviewed live strategy passes recall/cost gate
- [strategy_selection] diagnostics\question-observation-strategy-selection-risk-section-top1\strategy_selection.json: synthetic stress rows are not production recommendations
- [runtime] diagnostics\question-observation-runtime-perf-fixture-check\summary.json: runtime record coverage
- [runtime] diagnostics\question-observation-runtime-perf-fixture-check\summary.json: rect-only transport remains fileless
- [runtime] diagnostics\question-observation-runtime-perf-fixture-check\summary.json: local analysis p95 latency
- [runtime] diagnostics\question-observation-runtime-perf-fixture-check\summary.json: backend crop p95 latency
- [runtime] diagnostics\question-observation-runtime-perf-fixture-check\summary.json: cached segmentation reuse
- [runtime] diagnostics\question-observation-runtime-perf-fixture-check\summary.json: runtime perf script passed
- [ios] Xcode build succeeded
