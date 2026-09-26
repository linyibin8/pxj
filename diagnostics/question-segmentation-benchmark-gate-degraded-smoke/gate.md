# Question Segmentation Benchmark Gate

- Gate passed: `false`
- Summary: `diagnostics\question-segmentation-benchmark-loop-degraded-smoke\eval\summary.json`

## Checks

- `PASS` reference image coverage: reference missing images <= 0
- `PASS` reference unmatched rows: reference unmatched rows <= 0
- `FAIL` predictions image coverage: predictions missing images <= 0
- `PASS` predictions unmatched rows: predictions unmatched rows <= 0
- `FAIL` overall recall: overall recall >= 0.95
- `PASS` overall precision: overall precision >= 0.95
- `PASS` overall f1: overall f1 >= 0.95
- `FAIL` missed boxes: missed boxes <= 0
- `FAIL` false positive boxes: false positive boxes <= 0
- `FAIL` cohort gates: all sufficiently represented cohorts pass recall/precision gates
