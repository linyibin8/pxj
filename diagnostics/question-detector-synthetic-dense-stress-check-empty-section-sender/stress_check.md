# Synthetic Dense Stress Check

- Generated: 2026-06-30T09:25:40.066177+00:00
- Stress ready: `false`
- Transport ready: `true`
- Detector ready: `false`
- Hard failures: 1
- Warnings: 2

## Hard Failures

- [layout] current layout candidates cover dense GT: Current local layout/detector path must cover dense expected boxes at recall >= 0.999. Empty-frame section sender may protect transport cost, but it does not prove local question-box precision.

## Warnings

- [candidate_cap] current cap preserves dense crop-only recall: Production cap must not drop dense-page crop-only recall below 0.999. Section sender evidence passes, so this cap-only diagnostic is no longer release-blocking.
- [candidate_cap] current cap dense VLM pixel cost: Dense crop+fallback VLM pixels should stay <= 1.05x full-frame. Section sender evidence passes, so this cap-only diagnostic is no longer release-blocking.

## Key Metrics

- oracle cap-12 crop recall: `0.46875`
- oracle cap-12 fallback-protected recall: `1.0`
- oracle cap-12 total pixels vs full: `1.756221`
- section recommendation: `page__side1200`
- section total pixels vs full: `0.576576`
- section sender represented GT keys: `256`
- section sender combined crop recall: `1.0`
- section sender policy recall: `1.0`
- section sender total pixels vs full: `0.575394`
- section sender fallback image ratio: `0.0`
- empty-frame section represented GT keys: `256`
- empty-frame section combined crop recall: `1.0`
- empty-frame section total pixels vs full: `0.563498`
- empty-frame section fallback image ratio: `0.0`
- layout crop recall: `0.0`
- layout fallback images: `10`
- layout fallback total pixels vs full: `1.0`
