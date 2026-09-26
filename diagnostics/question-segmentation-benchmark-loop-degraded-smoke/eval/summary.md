# Question Segmentation Benchmark Eval

- Generated: 2026-07-01T02:41:02.868356+00:00
- Label: `degraded_loop_smoke`
- Benchmark manifest: `diagnostics\question-segmentation-codex-benchmark-pack-risk-section-top1\benchmark_manifest.jsonl`
- IoU threshold: 0.5
- Images: 40
- Reference matched/missing images: 40 / 0
- Prediction matched/missing images: 38 / 2

## Overall

- Precision: 0.9964
- Recall: 0.9197
- F1: 0.9565
- Matched/reference/predicted: 275 / 299 / 276
- Missed: 24
- False positives: 1

## By Cohort

- `active_capture`: P=1.0 R=1.0 F1=1.0 matched/ref/pred=12/12/12
- `active_quarantine`: P=1.0 R=1.0 F1=1.0 matched/ref/pred=28/28/28
- `dense_layout`: P=0.9931 R=0.8571 F1=0.9201 matched/ref/pred=144/168/145
- `fallback_delta`: P=1.0 R=1.0 F1=1.0 matched/ref/pred=20/20/20
- `observation_unverified`: P=1.0 R=1.0 F1=1.0 matched/ref/pred=71/71/71
