# Question Detector Review Priority

- Generated: 2026-07-01T02:16:14.441722+00:00
- Source manifest: `diagnostics\question-detector-review-queue-risk-section-top1\review_queue_manifest.json`
- Total boxes scored: 1030
- Images ranked: 278
- Recommended first pass: 40 images / 366 boxes
- Selection strategy: `score`

## Why This Order

- Prioritizes dense pages, fallback-delta evidence, active-learning candidates, and boxes with useful layout flags.
- In balanced mode, caps each cohort/source kind so the first pass covers different failure shapes instead of only the highest-scoring dense pages.
- De-prioritizes boxes that look tiny, oversized, low-score, quarantined, or correction-only.
- Keeps only a capped number of boxes per image in the first pass so review time covers more split groups.
- Writes `codex_benchmark_manifest.jsonl` from the same images, so Codex-style segmentation can be compared against the human-reviewed boxes without changing the label queue.

## Recommended First Pass

- 01. score=123.55 wb=3 image#2 boxes=12/26 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-001-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score, dense_page
- 02. score=123.0 wb=3 image#1 boxes=12/37 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-001-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, dense_page
- 03. score=123.0 wb=3 image#3 boxes=12/24 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-001-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, dense_page
- 04. score=123.0 wb=4 image#1 boxes=12/27 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-002-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, dense_page
- 05. score=108.15 wb=4 image#2 boxes=12/22 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-002-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 06. score=108.05 wb=3 image#4 boxes=12/23 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-001-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 07. score=107.75 wb=5 image#1 boxes=12/19 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-003-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 08. score=107.1 wb=4 image#4 boxes=12/16 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-002-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 09. score=107.05 wb=4 image#5 boxes=12/15 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-002-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 10. score=106.85 wb=4 image#6 boxes=12/15 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-002-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 11. score=106.85 wb=5 image#2 boxes=12/14 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-003-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 12. score=105.9 wb=5 image#3 boxes=12/13 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-003-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 13. score=105.85 wb=4 image#3 boxes=12/17 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-002-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 14. score=105.05 wb=5 image#4 boxes=12/12 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-003-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 15. score=98.25 wb=5 image#5 boxes=11/11 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-003-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 16. score=96.2 wb=5 image#7 boxes=11/11 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-003-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 17. score=68.05 wb=3 image#5 boxes=10/10 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-001-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 18. score=52.45 wb=6 image#9 boxes=6/6 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-004-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 19. score=48.65 wb=6 image#3 boxes=7/7 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-004-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 20. score=48.1 wb=6 image#4 boxes=7/7 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-004-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 21. score=47.12 wb=6 image#2 boxes=7/7 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-004-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score, large_box
- 22. score=44.735 wb=5 image#6 boxes=11/11 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-003-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score
- 23. score=41.8 wb=6 image#13 boxes=6/6 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-004-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 24. score=41.15 wb=6 image#11 boxes=6/6 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-004-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 25. score=40.55 wb=6 image#10 boxes=6/6 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-004-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 26. score=40.285 wb=8 image#1 boxes=10/10 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-006-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score
- 27. score=38.99 wb=1 image#8 boxes=6/6 cohort=active_quarantine kind=active html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: kind:active, higher_score, large_box
- 28. score=38.67 wb=5 image#10 boxes=9/9 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-003-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score
- 29. score=38.67 wb=6 image#1 boxes=9/9 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-004-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score
- 30. score=38.67 wb=7 image#1 boxes=9/9 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-005-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score
- 31. score=36.635 wb=5 image#9 boxes=9/9 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-003-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score
- 32. score=36.385 wb=1 image#2 boxes=7/7 cohort=active_quarantine kind=active html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: kind:active, higher_score
- 33. score=34.95 wb=6 image#17 boxes=5/5 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-004-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 34. score=34.44 wb=1 image#1 boxes=7/7 cohort=active_quarantine kind=active html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: kind:active
- 35. score=34.35 wb=6 image#15 boxes=5/5 cohort=dense_layout kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-004-workbench\index.html`
  reasons: source:dense_layout_prelabel, kind:observation_image_unverified, higher_score
- 36. score=30.25 wb=1 image#9 boxes=6/6 cohort=active_capture kind=active html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: kind:active, higher_score
- 37. score=29.785 wb=7 image#8 boxes=7/7 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-005-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score
- 38. score=29.2 wb=7 image#23 boxes=7/7 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-005-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score
- 39. score=29.2 wb=8 image#8 boxes=7/7 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-006-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score
- 40. score=28.605 wb=5 image#8 boxes=7/7 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-003-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score

## Recommended Cohorts

- `dense_layout`: 26
- `observation_unverified`: 10
- `active_quarantine`: 3
- `active_capture`: 1

## Box Source Counts

- `layout_prelabel`: 628
- `dense_layout_prelabel`: 370
- `fallback_delta_candidate`: 32

## Quality Flags

- `page:bright_page`: 638
- `page:full_image_fallback`: 392
- `few_text_lines`: 332
- `dense_column_layout`: 308
- `tall_block`: 179
- `near_full_page`: 146
- `short_block`: 64
- `dense_single_column_layout`: 62
- `fallback_delta_removed`: 32
- `all_weak_crop_keys`: 12
- `single_low_coverage_crop`: 12
- `no_large_question_crop`: 4
- `sparse_low_coverage_crops`: 4
