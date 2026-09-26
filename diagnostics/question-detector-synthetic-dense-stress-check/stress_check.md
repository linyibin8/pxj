# Synthetic Dense Stress Check

- Generated: 2026-06-30T07:29:09.987824+00:00
- Stress ready: `false`
- Hard failures: 3
- Warnings: 0

## Hard Failures

- [layout] current layout candidates cover dense GT: Current local layout path must cover dense expected boxes at recall >= 0.999; otherwise dense pages fall back to whole-frame VLM.
- [candidate_cap] current cap preserves dense crop-only recall: Production cap must not drop dense-page crop-only recall below 0.999.
- [candidate_cap] current cap dense VLM pixel cost: Dense crop+fallback VLM pixels should stay <= 1.05x full-frame.

## Key Metrics

- oracle cap-12 crop recall: `0.46875`
- oracle cap-12 fallback-protected recall: `1.0`
- oracle cap-12 total pixels vs full: `1.756221`
- section recommendation: `page__side1200`
- section total pixels vs full: `0.576576`
- layout crop recall: `0.0`
- layout fallback images: `10`
- layout fallback total pixels vs full: `1.0`
