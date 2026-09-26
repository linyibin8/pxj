# Codex Question Segmentation Benchmark

For each image listed in `benchmark_manifest.jsonl`, segment the visible worksheet into individual printed-question regions.

Return `codex_predictions.jsonl` with one JSON object per image. Use the exact `review_id` from the manifest. Use either pixel boxes or normalized boxes:

```json
{"review_id":"...","image":"images/001_example.jpg","boxes":[{"bbox_px":{"x":0,"y":0,"width":100,"height":100},"score":0.95}]}
```

Rules:

- Detect question regions, not answer correctness, not solution steps, and not decorative containers.
- A multi-line stem with its answer blanks/options belongs to one question region.
- Split adjacent numbered questions even when they share a dense two-column or three-column layout.
- Prefer tight boxes around the printed question content; include diagrams/options that belong to that question.
- Do not output full-page fallback boxes unless the whole page is genuinely one question.
- Preserve image order, and write an empty `boxes` list only when no printed question is visible.
