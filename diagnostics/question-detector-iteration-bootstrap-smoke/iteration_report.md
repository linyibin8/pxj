# Question Detector Iteration Report

- Generated: 2026-06-30T04:08:15.562669+00:00
- Stage: `hard_example_review`

## Next Actions

- P5 `respect_release_blockers`: Do not bundle this detector into iOS until hard release failures are cleared.
- P10 `review_pseudo_labels`: Bootstrap/pseudo labels are present; approve or correct hard examples before treating this data as training evidence.
- P12 `open_error_workbench`: Detector errors have been converted to reviewable hard examples.
- P15 `mine_and_review_model_errors`: Latest detector failed the eval gate; mine missed/false-positive images and review them before the next matrix run.
- P20 `grow_reviewed_dataset`: Dataset is not model-training-ready; expand reviewed labels and split groups before release training.
- P30 `annotate_more_boxes`: Need about 946 more reviewed annotations for the default production target.
- P35 `diversify_sessions`: Need about 11 more split groups to make held-out evaluation reliable.
- P80 `keep_rect_only_upload`: Replay still supports rect-only upload: low network overhead and lower VLM pixel load than full-frame VLM analysis.

## Dataset

- `diagnostics\question-detector-bootstrap-kpai-smoke\audit.json` ready=False annotations=54 human=6 pseudo=48 images=126 groups=39 gaps={'source_images': 174, 'annotations': 946, 'split_groups': 11, 'negative_images': 0}

## Experiments

- `diagnostics\question-detector-experiment-bootstrap-smoke\experiment_report.json` status=failed_eval_gate best=yolo11n.pt img=416 recall=0.0 precision=0.0 fp/img=0.0

## Release Gates

- `diagnostics\question-detector-release-check-bootstrap-smoke\release_check.json` ready=False hard=11 categories={'dataset': 1, 'selection': 4, 'training': 2, 'artifact': 1, 'ios': 3}

## Replay

- `diagnostics\question-observation-replay-kpai-layout-measured\summary.json` rect_only=1.002 crop_pixels=0.5584 layout_median_ms=24.052

## Error Mining

- `diagnostics\question-detector-error-mining-bootstrap-smoke\summary.json` images=16 missed=22 false_positive=0
