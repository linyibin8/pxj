# Question Detector Teacher/Student Status - 2026-07-02

## Current Level

- Release-grade Core ML detector: not ready.
- Trusted Codex visual-teacher gold: 9 images / 40 approved boxes.
- Teacher self-check on current package: precision 1.0, recall 1.0, F1 1.0 on its own 40-box manifest.
- Direct teacher dataset export: train 3 images / 12 boxes, val 5 images / 25 boxes, test 1 image / 3 boxes.
- Training readiness: false. The dataset now has a non-empty test split, but it is still far below the pilot and release thresholds.

## What Failed

- The app/workbench split boxes shown by the user are not accurate enough: they include whole-page, cross-page, partial-strip, and page/desk regions.
- Weak OCR/layout workbench labels are not gold. Existing previews show many full-sheet boxes and large paper boxes.
- Dell/reference labels are useful as candidates, not as ground truth. Direct filtered Dell labels were intentionally downgraded to candidate status.
- Regionizing Dell anchors failed the seed gate:
  - precision 0.3333
  - recall 0.375
  - F1 0.3529
  - main failure: two-page/cross-page boxes.
- Historical `merged-reviewed`/`propagated-exact` packages are not clean enough to merge. Visual preview showed full-paper boxes, duplicate views, and local diagram boxes mixed with question boxes.
- Homography transfer worked on near-identical dxue pages but did not generalize to the kpai geometry-paper distribution:
  - seed-003 dxue strict propagation exhausted with 0 new safe candidates.
  - seed-004 kpai 00064-00068 small scan produced 0 accepted candidates; non-seed images failed strict homography gates.
  - full kpai scan was too slow and was stopped; it is not the right expansion path.

## What Worked

- Codex visual teacher seed packages:
  - `diagnostics/question-detector-ai-teacher-seed-001`: 2 images / 8 approved boxes.
  - `diagnostics/question-detector-ai-teacher-seed-002`: 4 images / 18 approved boxes.
  - `diagnostics/question-detector-ai-teacher-seed-003`: 7 images / 33 approved boxes.
  - `diagnostics/question-detector-ai-teacher-seed-004`: 9 images / 40 approved boxes.
- Strict homography transfer from seed-001 produced 2 visually accepted dxue images / 10 boxes, then exhausted that same-layout cluster.
- Manual Codex visual labeling added:
  - 3 dxue left-page images / 15 boxes.
  - 2 kpai geometry-paper images / 7 boxes.
- Current teacher package and dataset:
  - `diagnostics/question-detector-ai-teacher-seed-004`
  - `diagnostics/question-detector-ai-teacher-seed-004-self-check`
  - `diagnostics/question-detector-ai-teacher-seed-004-teacher-dataset`
- A no-box pimage selection contact-sheet set was generated for teacher triage:
  - `diagnostics/question-detector-ai-teacher-selection-pimage`

## GPU Need

GPU is not needed yet. The bottleneck is not training speed; it is gold-label quality and coverage.

Use `dell@100.64.0.5` only after the pilot gold gate is met:

- Minimum pilot: 30-50 verified images, 150-250 boxes, with a non-empty held-out test split.
- Release candidate: 300+ verified images, 1000+ boxes.
- Detector promotion gate before Core ML/TestFlight:
  - recall >= 0.95
  - precision >= 0.90
  - false positives per image <= 0.05
  - no systematic full-page, cross-page, desk/keyboard, partial-question, or same-page-merge boxes.

## Recommended Next Loop

1. Continue Codex visual-teacher labeling from the pimage contact sheets, but skip blurred, cropped, duplicated, and same-paper far-view images.
2. Mark only complete visible question boxes as `verified`; keep partial/non-contiguous/cross-page cases out of training.
3. Repackage to seed-005, seed-006, etc.
4. Once the package reaches 30-50 images / 150-250 boxes, run a small YOLO pilot on Dell GPU and evaluate against the held-out teacher manifest.
5. Only after the model passes the detector gate should it be exported to Core ML and promoted into the iOS TestFlight build.
