# Question Detector Iteration Report

- Generated: 2026-06-30T05:51:58.013238+00:00
- Stage: `hard_example_review`

## Next Actions

- P5 `respect_release_blockers`: Do not bundle this detector into iOS until hard release failures are cleared.
- P8 `complete_review_workbenches`: Review workbenches still have about 71 boxes without a decisions export.
- P12 `open_error_workbench`: Detector errors have been converted to reviewable hard examples.
- P15 `mine_and_review_model_errors`: Latest detector failed the eval gate; mine missed/false-positive images and review them before the next matrix run.
- P18 `merge_approved_review_boxes`: 14 approved reviewed boxes are available; merge them into the next detector dataset build.
- P20 `grow_reviewed_dataset`: Dataset is not model-training-ready; expand reviewed labels and split groups before release training.
- P30 `annotate_more_boxes`: Need about 988 more reviewed annotations for the default production target.
- P35 `diversify_sessions`: Need about 30 more split groups to make held-out evaluation reliable.
- P80 `keep_rect_only_upload`: Replay still supports rect-only upload: low network overhead and lower VLM pixel load than full-frame VLM analysis.

## Dataset

- `diagnostics\question-detector-dataset-merged-reviewed-next-clustered\audit.json` ready=False annotations=12 human=12 pseudo=0 images=87 groups=20 gaps={'source_images': 213, 'annotations': 988, 'split_groups': 30, 'negative_images': 0}

## Experiments

- `diagnostics\question-detector-experiment-actual-smoke\experiment_report.json` status=failed_eval_gate best=yolo11n.pt img=320 recall=0.0 precision=0.0 fp/img=0.0

## Release Gates

- `diagnostics\question-detector-release-check-cap-aware-clustered-smoke\release_check.json` ready=False hard=11 categories={'dataset': 1, 'selection': 4, 'training': 2, 'artifact': 1, 'ios': 3}

## Replay

- `diagnostics\question-observation-replay-kpai-layout-measured\summary.json` rect_only=1.002 crop_pixels=0.5584 layout_median_ms=24.052

## Error Mining

- `diagnostics\question-detector-error-mining-bootstrap-smoke\summary.json` images=16 missed=22 false_positive=0

## Review Workbenches

- `diagnostics\question-detector-active-batch-kpai-001-workbench` boxes=41 decisions=1 approved_exports=1 approved_boxes=6 pending_boxes=0
  approved inputs: `diagnostics\question-detector-active-batch-kpai-001-workbench\approved_boxes_smoke.jsonl`
- `diagnostics\question-detector-active-batch-kpai-002-workbench` boxes=17 decisions=0 approved_exports=0 approved_boxes=0 pending_boxes=17
- `diagnostics\question-detector-active-batch-kpai-003-quarantine-workbench` boxes=32 decisions=0 approved_exports=0 approved_boxes=0 pending_boxes=32
- `diagnostics\question-detector-error-mining-bootstrap-smoke-workbench` boxes=22 decisions=0 approved_exports=0 approved_boxes=0 pending_boxes=22
- `diagnostics\question-detector-review-workbench-kpai` boxes=76 decisions=1 approved_exports=1 approved_boxes=8 pending_boxes=0
  approved inputs: `diagnostics\question-detector-review-workbench-kpai\approved_boxes_smoke.jsonl`
