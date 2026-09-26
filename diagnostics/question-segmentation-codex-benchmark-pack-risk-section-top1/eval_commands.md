# Evaluation Commands

Draft wiring self-check:

```powershell
python scripts\question_segmentation_benchmark_eval.py `
  --benchmark-manifest diagnostics\question-segmentation-codex-benchmark-pack-risk-section-top1\benchmark_manifest.jsonl `
  --reference-jsonl diagnostics\question-segmentation-codex-benchmark-pack-risk-section-top1\draft_reference_boxes.jsonl `
  --predictions-jsonl diagnostics\question-segmentation-codex-benchmark-pack-risk-section-top1\draft_reference_boxes.jsonl `
  --label draft_pack_self_check `
  --out diagnostics\question-segmentation-benchmark-pack-self-check `
  --clean
```

After Codex predictions are written to `codex_predictions.jsonl`:

```powershell
python scripts\question_segmentation_benchmark_eval.py `
  --benchmark-manifest diagnostics\question-segmentation-codex-benchmark-pack-risk-section-top1\benchmark_manifest.jsonl `
  --reference-jsonl PATH_TO_HUMAN_APPROVED_BOXES.jsonl `
  --predictions-jsonl diagnostics\question-segmentation-codex-benchmark-pack-risk-section-top1\codex_predictions.jsonl `
  --label codex_vs_reviewed `
  --out diagnostics\question-segmentation-benchmark-codex-vs-reviewed `
  --clean
```

Template file to start from:

```text
diagnostics\question-segmentation-codex-benchmark-pack-risk-section-top1\codex_predictions_template.jsonl
```
