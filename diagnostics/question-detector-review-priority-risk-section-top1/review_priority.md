# Question Detector Review Priority

- Generated: 2026-07-01T02:28:53.010356+00:00
- Source manifest: `diagnostics\question-detector-review-queue-risk-section-top1\review_queue_manifest.json`
- Total boxes scored: 1030
- Images ranked: 278
- Recommended first pass: 40 images / 299 boxes
- Selection strategy: `balanced`

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
- 15. score=44.735 wb=5 image#6 boxes=11/11 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-003-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score
- 16. score=40.285 wb=8 image#1 boxes=10/10 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-006-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score
- 17. score=38.99 wb=1 image#8 boxes=6/6 cohort=active_quarantine kind=active html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: kind:active, higher_score, large_box
- 18. score=38.67 wb=5 image#10 boxes=9/9 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-003-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score
- 19. score=38.67 wb=6 image#1 boxes=9/9 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-004-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score
- 20. score=38.67 wb=7 image#1 boxes=9/9 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-005-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score
- 21. score=36.635 wb=5 image#9 boxes=9/9 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-003-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score
- 22. score=36.385 wb=1 image#2 boxes=7/7 cohort=active_quarantine kind=active html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: kind:active, higher_score
- 23. score=34.44 wb=1 image#1 boxes=7/7 cohort=active_quarantine kind=active html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: kind:active
- 24. score=30.25 wb=1 image#9 boxes=6/6 cohort=active_capture kind=active html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: kind:active, higher_score
- 25. score=29.785 wb=7 image#8 boxes=7/7 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-005-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score
- 26. score=29.2 wb=7 image#23 boxes=7/7 cohort=observation_unverified kind=observation_image_unverified html=`diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-005-workbench\index.html`
  reasons: kind:observation_image_unverified, higher_score
- 27. score=18.35 wb=1 image#6 boxes=4/4 cohort=fallback_delta kind=fallback_delta html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: source:fallback_delta_candidate, kind:fallback_delta, higher_score
- 28. score=16.95 wb=1 image#37 boxes=3/3 cohort=active_quarantine kind=active html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: kind:active, higher_score
- 29. score=16.07 wb=1 image#12 boxes=3/3 cohort=active_quarantine kind=active html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: kind:active, higher_score, large_box
- 30. score=15.635 wb=1 image#3 boxes=3/3 cohort=fallback_delta kind=fallback_delta html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: source:fallback_delta_candidate, kind:fallback_delta, higher_score
- 31. score=15.5 wb=1 image#7 boxes=3/3 cohort=fallback_delta kind=fallback_delta html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: source:fallback_delta_candidate, kind:fallback_delta, higher_score
- 32. score=14.57 wb=1 image#4 boxes=3/3 cohort=fallback_delta kind=fallback_delta html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: source:fallback_delta_candidate, kind:fallback_delta, higher_score, large_box
- 33. score=14.57 wb=1 image#5 boxes=3/3 cohort=fallback_delta kind=fallback_delta html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: source:fallback_delta_candidate, kind:fallback_delta, higher_score, large_box
- 34. score=11.3 wb=1 image#26 boxes=2/2 cohort=active_capture kind=active html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: kind:active, higher_score
- 35. score=10.62 wb=1 image#25 boxes=2/2 cohort=active_capture kind=active html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: kind:active, higher_score, large_box
- 36. score=10.37 wb=1 image#38 boxes=2/2 cohort=active_capture kind=active html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: kind:active, higher_score, large_box
- 37. score=10.3 wb=1 image#10 boxes=2/2 cohort=fallback_delta kind=fallback_delta html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: source:fallback_delta_candidate, kind:fallback_delta, higher_score
- 38. score=10.3 wb=1 image#11 boxes=2/2 cohort=fallback_delta kind=fallback_delta html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: source:fallback_delta_candidate, kind:fallback_delta, higher_score
- 39. score=6.32 wb=1 image#35 boxes=1/1 cohort=active_quarantine kind=active html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: kind:active, higher_score, large_box
- 40. score=6.32 wb=1 image#36 boxes=1/1 cohort=active_quarantine kind=active html=`diagnostics\question-detector-review-backlog-risk-section-top1-workbench\index.html`
  reasons: kind:active, higher_score, large_box

## Recommended Cohorts

- `dense_layout`: 14
- `observation_unverified`: 8
- `active_quarantine`: 7
- `fallback_delta`: 7
- `active_capture`: 4

## Review Root

- Prelabel root: `diagnostics\question-detector-review-priority-risk-section-top1-prelabel`
- Images: 40
- Review boxes: 411
- Recommended high-priority boxes: 299
- The review root includes all draft boxes on each selected image so approved training labels do not become partial-image annotations.

Build the static review UI with:

```powershell
python scripts\question_detector_review_workbench.py `
  --prelabel-root diagnostics\question-detector-review-priority-risk-section-top1-prelabel `
  --out diagnostics\question-detector-review-priority-risk-section-top1-workbench `
  --clean
```

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
