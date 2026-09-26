# Question Observation Runtime Perf Check

- Generated: 2026-06-30T16:06:35.372426+00:00
- Passed: `true`
- Records: 6
- Frames: 39
- Fresh OCR ratio: 0.205128
- Attached crop files: 0
- Analysis p50/p95: 92.0 / 114.5 ms
- Server crop p95: 30.0 ms

## Checks

- `true` minimum runtime records: Need enough live/simulator runtime records. {'record_count': 6}
- `true` rect-only upload: Observation upload should not attach question crop JPEG files. {'attached_crop_file_count': 0}
- `true` analysis p50 latency: Local observation analysis p50 must stay within budget. {'p50': 92.0}
- `true` analysis p95 latency: Local observation analysis p95 must stay within budget. {'p95': 114.5}
- `true` server crop p95 latency: Backend canonical crop expansion p95 must stay within budget. {'p95': 30.0}
- `true` fresh OCR ratio: Batch upload should mostly reuse cached live-scan segmentation instead of rerunning OCR. {'fresh_ocr_ratio': 0.20512820512820512}
