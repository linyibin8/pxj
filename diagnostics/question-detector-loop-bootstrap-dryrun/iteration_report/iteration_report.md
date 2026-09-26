# Question Detector Iteration Report

- Generated: 2026-06-30T04:11:22.251840+00:00
- Stage: `model_selection_passed`

## Next Actions

- P10 `review_pseudo_labels`: Bootstrap/pseudo labels are present; approve or correct hard examples before treating this data as training evidence.
- P20 `grow_reviewed_dataset`: Dataset is not model-training-ready; expand reviewed labels and split groups before release training.
- P30 `annotate_more_boxes`: Need about 946 more reviewed annotations for the default production target.
- P35 `diversify_sessions`: Need about 11 more split groups to make held-out evaluation reliable.
- P70 `run_release_gate`: Experiment matrix passed; run the release readiness gate with replay and iOS artifact evidence.
- P80 `keep_rect_only_upload`: Replay still supports rect-only upload: low network overhead and lower VLM pixel load than full-frame VLM analysis.

## Dataset

- `diagnostics\question-detector-bootstrap-kpai-smoke\audit.json` ready=False annotations=54 human=6 pseudo=48 images=126 groups=39 gaps={'source_images': 174, 'annotations': 946, 'split_groups': 11, 'negative_images': 0}

## Experiments

- `diagnostics\question-detector-loop-bootstrap-dryrun\experiment\experiment_report.json` status=passed best=yolo11n.pt img=416 recall=0.0 precision=0.0 fp/img=0.0

## Release Gates


## Replay

- `diagnostics\question-observation-replay-kpai-layout-measured\summary.json` rect_only=1.002 crop_pixels=0.5584 layout_median_ms=24.052

## Error Mining

