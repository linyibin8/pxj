# Question Detector Iteration Report

- Generated: 2026-06-30T16:07:37.335890+00:00
- Stage: `model_selection_passed`

## Next Actions

- P5 `respect_release_blockers`: Do not bundle this detector into iOS until hard release failures are cleared.
- P8 `complete_review_workbenches`: Review workbenches still have about 1030 boxes without a decisions export.
- P20 `grow_reviewed_dataset`: Dataset is not model-training-ready; expand reviewed labels and split groups before release training.
- P70 `run_release_gate`: Experiment matrix passed; run the release readiness gate with replay and iOS artifact evidence.
- P80 `keep_rect_only_upload`: Replay still supports rect-only upload: low network overhead and lower VLM pixel load than full-frame VLM analysis.
- P82 `keep_best_observation_strategy`: Strategy selection supports the current reviewed live observation crop strategy.

## Dataset

- `diagnostics\question-detector-dataset-merged-reviewed-next-propagated-exact\audit.json` ready=False annotations=16 human=16 pseudo=0 images=89 groups=20 gaps={'source_images': 211, 'annotations': 984, 'split_groups': 30, 'negative_images': 0}

## Experiments

- `diagnostics\question-detector-loop-backend-tuned-fallback-delta-dryrun\experiment\experiment_report.json` status=passed best=yolo11n.pt img=416 recall=0.0 precision=0.0 fp/img=0.0

## Release Gates

- `diagnostics\question-detector-release-check-risk-section-top1-propagated-only\release_check.json` ready=False hard=5 categories={'dataset': 1, 'selection': 1, 'artifact': 1, 'ios': 1, 'runtime': 1}

## Replay

- `diagnostics\question-observation-replay-reviewed-propagated-layout-v5-gated\summary.json` rect_only=1.004 crop_pixels=0.3751 layout_median_ms=24.959

## Strategy Selection

- `diagnostics\question-observation-strategy-selection-risk-section-top1\strategy_selection.json` rows=24 best=question-observation-replay-kpai-layout-measured-v5-dense-gated:ios_sender recall=1.0 fallback=0.02439 pixels=0.76223
  propagated: `question-observation-replay-reviewed-propagated-layout-v5-gated:ios_sender` recall=1.0 fallback=0.202247 pixels=0.810149

## Training Readiness

- `diagnostics\question-detector-training-readiness-plan-risk-section-top1\training_readiness_plan.json` pending_boxes=1030 annotations_after_pending=1046 remaining_annotation_gap=0

## Review Queues

- `diagnostics\question-detector-review-queue-risk-section-top1\review_queue_manifest.json` workbenches=14 boxes=1030 pending=1030 approved=0 html=`diagnostics\question-detector-review-queue-risk-section-top1\index.html`

## Error Mining


## Review Workbenches

