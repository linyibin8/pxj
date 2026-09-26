# Question Detector Training Readiness Plan

- Generated: 2026-06-30T16:07:26.450934+00:00
- Dataset ready: `false`
- Current annotations: 16 / 1000 (gap 984)
- Pending review boxes: 1030 across 278 images
- Optimistic annotations after pending review: 1046 / 1000
- Optimistic split groups after pending review: 208 / 50 (verify after approved merge)

## Actions

- P1 `finish_current_review_backlog`: Review/export the current 1030 pending boxes before training; if approval and dedupe hold, this backlog can cover the production annotation target.
- P3 `verify_split_group_gain_after_merge`: Pending review may close the split-group gap only if those groups survive approval/dedup; rerun dataset audit after merge.
- P4 `respect_release_blockers`: Do not promote a Core ML detector until release blockers clear.
- P5 `keep_current_observation_strategy`: Keep the current reviewed-live observation strategy while the training dataset grows.
- P6 `preserve_review_provenance`: Backlog outputs are review queues only; merge only approved/corrected/verified exports into training.

## Strategy

- Keep `question-observation-replay-kpai-layout-measured-v5-dense-gated:ios_sender` while labels grow: recall=1.0 fallback=0.02439 pixels/full=0.76223

## Review Sources

- `diagnostics\question-detector-review-backlog-risk-section-top1-workbench` boxes=80 pending=80 groups~=33 sources={'active': 19, 'fallback_delta': 19}
- `diagnostics\question-detector-review-wave-plan-kpai-after-backlog\wave-001-workbench` boxes=1 pending=1 groups~=1 sources={'unknown': 1}
- `diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-001-workbench` boxes=120 pending=120 groups~=4 sources={'unknown': 5}
- `diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-002-workbench` boxes=120 pending=120 groups~=6 sources={'unknown': 8}
- `diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-003-workbench` boxes=120 pending=120 groups~=10 sources={'unknown': 11}
- `diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-004-workbench` boxes=120 pending=120 groups~=13 sources={'unknown': 23}
- `diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-005-workbench` boxes=102 pending=102 groups~=18 sources={'unknown': 24}
- `diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-006-workbench` boxes=75 pending=75 groups~=15 sources={'unknown': 24}
- `diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-007-workbench` boxes=61 pending=61 groups~=17 sources={'unknown': 24}
- `diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-008-workbench` boxes=57 pending=57 groups~=13 sources={'unknown': 24}
- `diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-009-workbench` boxes=49 pending=49 groups~=13 sources={'unknown': 24}
- `diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-010-workbench` boxes=45 pending=45 groups~=16 sources={'unknown': 24}
- `diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-011-workbench` boxes=49 pending=49 groups~=15 sources={'unknown': 24}
- `diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-012-workbench` boxes=31 pending=31 groups~=14 sources={'unknown': 24}

## Review Wave Capacity

- `diagnostics\question-detector-review-wave-plan-kpai-after-backlog\review_wave_plan.json` exhausted=True selected=1 images / 1 boxes unique_boxes=1
- `diagnostics\question-detector-review-wave-plan-kpai-expanded\review_wave_plan.json` exhausted=True selected=1 images / 1 boxes unique_boxes=1
- `diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\review_wave_plan.json` exhausted=False selected=239 images / 949 boxes unique_boxes=894
