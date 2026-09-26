# Question Segmentation Codex Benchmark Pack

Generated: 2026-07-01T02:31:44.720643+00:00

This pack contains 40 images selected from the current PXJ question-detector review queue. It is meant to compare Codex-style visual segmentation with the same human-reviewed reference boxes used for detector training.

## Files

- `benchmark_manifest.jsonl`: image list, review IDs, cohort labels, and copied image paths.
- `images/`: copied benchmark images.
- `codex_prompt.md`: task prompt and output schema.
- `codex_predictions_template.jsonl`: empty rows to fill with Codex predictions.
- `draft_reference_boxes.jsonl`: draft detector boxes for sanity checks only; not ground truth.
- `eval_commands.md`: exact commands for draft self-check and reviewed-reference evaluation.

## Required Prediction Shape

Each `codex_predictions.jsonl` row should look like:

```json
{"review_id":"...","image":"images/001_example.jpg","boxes":[{"bbox_px":{"x":0,"y":0,"width":100,"height":100},"score":0.95}]}
```

After human review export exists, evaluate Codex predictions against it with `question_segmentation_benchmark_eval.py`. Until then, the draft self-check only verifies wiring and should not be treated as model quality.

Always inspect `summary.json.input_audit`: reference and prediction `missing_images` should be empty for a valid comparison. A high F1 over a partially matched prediction file is not a usable benchmark result.
