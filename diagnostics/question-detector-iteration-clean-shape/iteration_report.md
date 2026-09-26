# Question Detector Iteration Report

- Generated: 2026-06-30T09:25:17.903252+00:00
- Stage: `hard_example_review`

## Next Actions

- P5 `respect_release_blockers`: Do not bundle this detector into iOS until hard release failures are cleared.
- P8 `complete_review_workbenches`: Review workbenches still have about 440 boxes without a decisions export.
- P10 `review_pseudo_labels`: Bootstrap/pseudo labels are present; approve or correct hard examples before treating this data as training evidence.
- P12 `open_error_workbench`: Detector errors have been converted to reviewable hard examples.
- P15 `mine_and_review_model_errors`: Latest detector failed the eval gate; mine missed/false-positive images and review them before the next matrix run.
- P20 `grow_reviewed_dataset`: Dataset is not model-training-ready; expand reviewed labels and split groups before release training.
- P30 `annotate_more_boxes`: Need about 973 more reviewed annotations for the default production target.
- P35 `diversify_sessions`: Need about 9 more split groups to make held-out evaluation reliable.
- P80 `keep_rect_only_upload`: Replay still supports rect-only upload: low network overhead and lower VLM pixel load than full-frame VLM analysis.

## Dataset

- `diagnostics\question-detector-bootstrap-kpai-clean-shape\audit.json` ready=False annotations=27 human=16 pseudo=11 images=135 groups=41 gaps={'source_images': 165, 'annotations': 973, 'split_groups': 9, 'negative_images': 0}
- `diagnostics\question-detector-dataset-merged-reviewed-next-propagated-exact\audit.json` ready=False annotations=16 human=16 pseudo=0 images=89 groups=20 gaps={'source_images': 211, 'annotations': 984, 'split_groups': 30, 'negative_images': 0}

## Experiments

- `diagnostics\question-detector-experiment-bootstrap-clean-shape\runs\clean-shape__yolo11n__img416__seed42__ep8\train_manifest.json` status=failed_eval_gate best=yolo11n.pt img=416 recall=0.9091 precision=0.0012 fp/img=189.8864
- `diagnostics\question-detector-experiment-bootstrap-clean-shape\runs\clean-shape__yolo11n__img416__seed42__ep25\train_manifest.json` status=failed_eval_gate best=yolo11n.pt img=416 recall=0.9091 precision=0.0008 fp/img=281.2727

## Release Gates

- `diagnostics\question-detector-release-check-p0-empty-section-dedupe\release_check.json` ready=False hard=11 categories={'dataset': 1, 'selection': 4, 'training': 2, 'artifact': 1, 'ios': 3}

## Replay

- `diagnostics\question-observation-replay-kpai-layout-measured\summary.json` rect_only=1.002 crop_pixels=0.5584 layout_median_ms=30.896

## Error Mining

- `diagnostics\question-detector-error-mining-bootstrap-clean-shape\summary.json` images=43 missed=1 false_positive=219
- `diagnostics\question-detector-error-mining-bootstrap-clean-shape-ep25\summary.json` images=37 missed=1 false_positive=219

## Review Workbenches

- `diagnostics\question-detector-error-mining-bootstrap-clean-shape-workbench` boxes=220 decisions=0 approved_exports=0 approved_boxes=0 pending_boxes=220
- `diagnostics\question-detector-error-mining-bootstrap-clean-shape-ep25-workbench` boxes=220 decisions=0 approved_exports=0 approved_boxes=0 pending_boxes=220
