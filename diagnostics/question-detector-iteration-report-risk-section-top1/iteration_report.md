# Question Detector Iteration Report

- Generated: 2026-06-30T14:01:52.394715+00:00
- Stage: `model_selection_passed`

## Next Actions

- P5 `respect_release_blockers`: Do not bundle this detector into iOS until hard release failures are cleared.
- P8 `complete_review_workbenches`: Review workbenches still have about 81 boxes without a decisions export.
- P20 `grow_reviewed_dataset`: Dataset is not model-training-ready; expand reviewed labels and split groups before release training.
- P30 `annotate_more_boxes`: Need about 984 more reviewed annotations for the default production target.
- P35 `diversify_sessions`: Need about 30 more split groups to make held-out evaluation reliable.
- P70 `run_release_gate`: Experiment matrix passed; run the release readiness gate with replay and iOS artifact evidence.
- P80 `keep_rect_only_upload`: Replay still supports rect-only upload: low network overhead and lower VLM pixel load than full-frame VLM analysis.

## Dataset

- `diagnostics\question-detector-dataset-merged-reviewed-next-propagated-exact\audit.json` ready=False annotations=16 human=16 pseudo=0 images=89 groups=20 gaps={'source_images': 211, 'annotations': 984, 'split_groups': 30, 'negative_images': 0}

## Experiments

- `diagnostics\question-detector-loop-backend-tuned-fallback-delta-dryrun\experiment\experiment_report.json` status=passed best=yolo11n.pt img=416 recall=0.0 precision=0.0 fp/img=0.0

## Release Gates

- `diagnostics\question-detector-release-check-risk-section-top1-propagated-only\release_check.json` ready=False hard=4 categories={'dataset': 1, 'selection': 1, 'artifact': 1, 'ios': 1}

## Replay

- `diagnostics\question-observation-replay-reviewed-propagated-layout-v5-gated\summary.json` rect_only=1.004 crop_pixels=0.3751 layout_median_ms=24.959

## Error Mining


## Review Workbenches

- `diagnostics\question-detector-active-batch-kpai-002-workbench` boxes=17 decisions=0 approved_exports=0 approved_boxes=0 pending_boxes=17
- `diagnostics\question-detector-active-batch-kpai-003-quarantine-workbench` boxes=32 decisions=0 approved_exports=0 approved_boxes=0 pending_boxes=32
- `diagnostics\question-detector-review-workbench-fallback-delta-propagated-backend-tuned` boxes=32 decisions=0 approved_exports=0 approved_boxes=0 pending_boxes=32
