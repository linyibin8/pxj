# Observation Strategy Selection

- Generated: 2026-06-30T14:46:16.308251+00:00
- Rows: 24

## Best Reviewed Live Rows

| replay | strategy | family | recall | fallback | pixels/full | notes |
|---|---|---|---:|---:|---:|---|
| question-observation-replay-kpai-layout-measured-v5-dense-gated | question-observation-replay-kpai-layout-measured-v5-dense-gated:ios_sender | section_sender | 1 | 0.02439 | 0.76223 | Passes configured recall/cost screen for this evidence set. |
| question-observation-replay-kpai-layout-measured-v5-dense-gated | question-observation-replay-kpai-layout-measured-v5-dense-gated:backend_current | fallback_policy | 1 | 0.170732 | 0.784856 | Passes configured recall/cost screen for this evidence set. |
| question-observation-replay-kpai-layout-measured-v5-dense-gated | question-observation-replay-kpai-layout-measured-v5-dense-gated:cap_12 | candidate_cap | 1 | 0.170732 | 0.784856 | Passes configured recall/cost screen for this evidence set. |
| question-observation-replay-reviewed-propagated-layout-v5-gated | question-observation-replay-reviewed-propagated-layout-v5-gated:ios_sender | section_sender | 1 | 0.202247 | 0.810149 | Passes configured recall/cost screen for this evidence set. |
| question-observation-replay-reviewed-propagated-layout-v5-gated | question-observation-replay-reviewed-propagated-layout-v5-gated:cap_12 | candidate_cap | 1 | 0.382022 | 0.8122 | Passes configured recall/cost screen for this evidence set. |

## Synthetic Stress Rows

| replay | strategy | family | status | recall | fallback | pixels/full | notes |
|---|---|---|---|---:|---:|---:|---|
| question-observation-replay-synthetic-dense-layout | question-observation-replay-synthetic-dense-layout:ios_sender | section_sender | live | 1 | 0 | 0.563498 | Passes configured recall/cost screen for this evidence set. |
| question-observation-replay-synthetic-dense-oracle | question-observation-replay-synthetic-dense-oracle:ios_sender | section_sender | live | 1 | 0 | 0.701777 | Passes configured recall/cost screen for this evidence set. |
| question-observation-replay-synthetic-dense-oracle | question-observation-replay-synthetic-dense-oracle:full_frame | full_frame_baseline | baseline | 1 | 1 | 1 | Full-frame VLM baseline: accurate fallback reference, highest expected transfer/VLM cost. |
| question-observation-replay-synthetic-dense-layout | question-observation-replay-synthetic-dense-layout:full_frame | full_frame_baseline | baseline | 1 | 1 | 1 | Full-frame VLM baseline: accurate fallback reference, highest expected transfer/VLM cost. |
| question-observation-replay-synthetic-dense-layout | question-observation-replay-synthetic-dense-layout:cap_only | section_sender | baseline_cap_only | 1 | 1 | 1 | Crop recall below target 0.999.; Costs at least as much VLM pixels as full-frame baseline.; Above target total pixel ratio 0.900. |
| question-observation-replay-synthetic-dense-oracle | question-observation-replay-synthetic-dense-oracle:cap_only | section_sender | baseline_cap_only | 1 | 1 | 1.75622 | Crop recall below target 0.999.; Costs at least as much VLM pixels as full-frame baseline.; Above target total pixel ratio 0.900.; Skips strong candidates; needs section or fallback protection.; Skips high-confidence candidates; needs section or fallback protection. |

## Pareto Rows

| replay | strategy | family | status | recall | fallback | pixels/full | evidence |
|---|---|---|---|---:|---:|---:|---|
| question-observation-replay-kpai-layout-measured-v5-dense-gated | question-observation-replay-kpai-layout-measured-v5-dense-gated:all_candidates_crop_only | candidate_eval | candidate_set | 1 | 0 | 0.625661 | reviewed_kpai |
| question-observation-replay-reviewed-propagated-layout-v5-gated | question-observation-replay-reviewed-propagated-layout-v5-gated:all_candidates_crop_only | candidate_eval | candidate_set | 1 | 0 | 0.434122 | reviewed_propagated |
| question-observation-replay-synthetic-dense-oracle | question-observation-replay-synthetic-dense-oracle:ios_sender | section_sender | live | 1 | 0 | 0.701777 | synthetic_stress |
| question-observation-replay-synthetic-dense-layout | question-observation-replay-synthetic-dense-layout:ios_sender | section_sender | live | 1 | 0 | 0.563498 | synthetic_stress |

## Notes

- Rows are comparable only within the same replay/evidence_kind group.
- Use best_reviewed_live_by_cost for production decisions; synthetic stress rows are negative gates and must not count as release training evidence.
- Full-frame rows are baselines: they model VLM cost, not a local detector strategy.
