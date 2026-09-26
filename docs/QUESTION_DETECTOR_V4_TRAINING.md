# Question Detector V4 Training Loop

## Objective

Train a small on-device iOS/iPad detector for `question_block` regions. The detector should improve speed and recall while keeping the existing OCR heuristic as a safe fallback.

The production contract stays stable:

```swift
QuestionSegmenter.segment(_ image: UIImage, fast: Bool) -> [QuestionRegion]
```

## Data Sources

Primary source:

- `session_question_crops`
- joined with `images`
- optionally joined with `session_observations`

Useful fields:

- source image: `images.filename`
- bbox candidates: `session_question_crops.normalized_rect` and `session_question_crops.crop_rect`
- quality: `confidence`, `source`, `status`, and `normalized_rect.crop_safety`
- dedupe: `question_key`, `fingerprint`, `text_hash`, `crop_hash`
- frame filtering: `session_observations.novelty_status`, `duplicate_of_image_id`, `visual_hash`, `text_hash`

Diagnostics source:

- `diagnostics/crop-f472/data/manifest.json`
- source images under `diagnostics/crop-f472/data/images`

Before training on a machine, inventory available data:

```powershell
python scripts\question_detector_data_inventory.py `
  D:\AI\pxj `
  D:\AI\kpai\backend\data `
  --out diagnostics\question-detector-data-inventory.json
```

The inventory distinguishes labeled weak-training data from unlabeled image pools. A directory with many images but zero `session_question_crops` is useful for replay/manual review, but it is not automatically safe as positive detector training data.

Local historical audit on 2026-06-30:

- `D:\AI\kpai\backend\data` contains 631 DB image rows.
- No scanned SQLite DB currently contains `session_question_crops`.
- The largest account DB can provide QA-linked review candidates and textless empty-page negatives, but not detector boxes.

For unlabeled historical pools, first build a review package:

```powershell
python scripts\question_detector_review_candidates.py `
  --sqlite D:\AI\kpai\backend\data\accounts\c92a0e4ea0244cd281d8224a72639c9d.sqlite3 `
  --data-dir D:\AI\kpai\backend\data `
  --out diagnostics\question-detector-review-candidates-kpai `
  --positive-limit 80 `
  --negative-limit 80 `
  --clean
```

This writes positive/negative image candidates plus contact sheets. Positive candidates are image-level review hints from QA events; they are not bounding boxes. Use them for annotation/active learning, not direct detector training.

To speed up human annotation on those image-level positives, generate draft boxes:

```powershell
python scripts\question_detector_prelabel.py `
  --candidate-root diagnostics\question-detector-review-candidates-kpai `
  --out diagnostics\question-detector-prelabel-kpai `
  --clean
```

This writes:

- `annotations/draft_boxes.jsonl`
- `annotations/coco_draft.json`
- `annotations/createml_draft.json`
- `annotations/label_studio_tasks.json`
- `draft_preview_contact_sheet.jpg`
- `annotations/quarantine.jsonl`
- `annotations/label_studio_quarantine_tasks.json`
- `quarantine_preview_contact_sheet.jpg`

The prelabel output is deliberately marked `draft_review_required`. It is not ground truth and must not be merged into `question_detector_dataset.py` until a human reviewer approves or corrects each box. Images with unreliable page detection, keyboard/table dominance, or fragmented layout are routed to `manual_page_review_required` quarantine outputs so they do not pollute the main draft annotation queue.

Current KPAI historical prelabel run:

- input positive candidates: 55
- draft-review images: 47
- draft boxes: 76
- quarantined images: 8
- quarantine reasons: page rect fallback, keyboard/table-like wide top bands, dense edge-dominant layouts, fragmented short blocks

After human review, promote only approved boxes into a detector dataset:

Build the local review workbench:

```powershell
python scripts\question_detector_review_workbench.py `
  --prelabel-root diagnostics\question-detector-prelabel-kpai `
  --out diagnostics\question-detector-review-workbench-kpai `
  --clean
```

Open `diagnostics\question-detector-review-workbench-kpai\index.html`, approve/reject/correct boxes, then download `question_box_decisions.json`. Convert those decisions into a training-safe approved JSONL:

```powershell
python scripts\question_detector_review_workbench.py `
  --prelabel-root diagnostics\question-detector-prelabel-kpai `
  --decisions diagnostics\question-detector-review-workbench-kpai\question_box_decisions.json `
  --approved-jsonl diagnostics\question-detector-review-workbench-kpai\approved_boxes.jsonl
```

For larger historical pools, create a prioritized active-learning batch before opening the workbench:

```powershell
python scripts\question_detector_active_batch.py `
  --prelabel-root diagnostics\question-detector-prelabel-kpai `
  --exclude-approved-jsonl diagnostics\question-detector-review-workbench-kpai\approved_boxes.jsonl `
  --exclude-queued-root diagnostics\question-detector-active-batch-kpai-001 `
  --limit 24 `
  --max-per-session 4 `
  --out diagnostics\question-detector-active-batch-kpai-001 `
  --clean

python scripts\question_detector_review_workbench.py `
  --prelabel-root diagnostics\question-detector-active-batch-kpai-001 `
  --out diagnostics\question-detector-active-batch-kpai-001-workbench `
  --clean
```

The active batch selector ranks draft rows by uncertainty, quality flags, box count, dense text, and shape risk, then greedily limits exact/near-duplicate images and per-session concentration. It writes `annotations/ranked_candidates.jsonl`, `annotations/draft_boxes.jsonl`, and `active_batch_contact_sheet.jpg`. The output is still only a review queue, not ground truth.

Use `--exclude-queued-root` for previous active batch/workbench roots or JSONL files that have been sent to reviewers but are not yet merged as approved labels. This lets batch 002/003 cover fresh images instead of reusing batch 001 while it is still awaiting review.

When several review queues accumulate, merge them into one prioritized review
backlog before asking for manual decisions. The backlog builder ranks pending
items from active-learning, fallback-delta, propagation, and error-mining
workbenches, removes exact/near-duplicate images, and writes another
prelabel-style root. This output is still only a review queue, never approved
ground truth:

```powershell
python scripts\question_detector_review_backlog.py `
  --workbench diagnostics\question-detector-active-batch-kpai-002-workbench `
  --workbench diagnostics\question-detector-active-batch-kpai-003-quarantine-workbench `
  --workbench diagnostics\question-detector-review-workbench-fallback-delta-propagated-backend-tuned `
  --workbench diagnostics\question-detector-propagation-review-candidates-ahash1-2-workbench `
  --out diagnostics\question-detector-review-backlog-risk-section-top1 `
  --limit 64 `
  --clean

python scripts\question_detector_review_workbench.py `
  --prelabel-root diagnostics\question-detector-review-backlog-risk-section-top1 `
  --out diagnostics\question-detector-review-backlog-risk-section-top1-workbench `
  --clean
```

Before assuming the same prelabel pool can supply another useful active batch,
run the wave planner. It repeatedly calls the active-batch selector while adding
each generated wave to the queued-exclusion set, then optionally builds review
workbenches:

```powershell
python scripts\question_detector_review_wave_plan.py `
  --prelabel-root diagnostics\question-detector-prelabel-kpai `
  --exclude-approved-jsonl diagnostics\question-detector-active-batch-kpai-001-workbench\approved_boxes_smoke.jsonl `
  --exclude-approved-jsonl diagnostics\question-detector-review-workbench-kpai\approved_boxes_smoke.jsonl `
  --exclude-queued-root diagnostics\question-detector-active-batch-kpai-001 `
  --exclude-queued-root diagnostics\question-detector-active-batch-kpai-002 `
  --exclude-queued-root diagnostics\question-detector-active-batch-kpai-003-quarantine `
  --exclude-queued-root diagnostics\question-detector-review-backlog-risk-section-top1 `
  --include-quarantine `
  --build-workbenches `
  --out diagnostics\question-detector-review-wave-plan-kpai-after-backlog `
  --clean
```

Current KPAI wave-capacity result:
`diagnostics\question-detector-review-wave-plan-kpai-after-backlog` is
`exhausted=true` after selecting only 1 additional image / 1 box. Treat that as
evidence that the current KPAI prelabel pool is exhausted; the next data step is
to mine a broader historical SQLite/image pool and prelabel it, not to keep
sampling the same pool.

QA-linked expansion across the two largest useful SQLite DBs did not add new
review capacity: it still produced only 55 positive candidates, and its wave
plan selected the same final 1 image / 1 box. The useful expansion path is to
add unverified `session_observations` as review-only positive candidates, then
let prelabeling and active learning rank them:

```powershell
python scripts\question_detector_review_candidates.py `
  --sqlite D:\AI\kpai\backend\data\accounts\c92a0e4ea0244cd281d8224a72639c9d.sqlite3 `
  --sqlite D:\AI\kpai\backend\data\accounts\f222b087a8f84687b4b85690f7eaaf81.sqlite3 `
  --data-dir D:\AI\kpai\backend\data `
  --out diagnostics\question-detector-review-candidates-kpai-observation-expanded `
  --positive-limit 1000 `
  --negative-limit 300 `
  --include-observation-positives `
  --observation-positive-limit 700 `
  --clean

python scripts\question_detector_prelabel.py `
  --candidate-root diagnostics\question-detector-review-candidates-kpai-observation-expanded `
  --out diagnostics\question-detector-prelabel-kpai-observation-expanded `
  --clean

python scripts\question_detector_review_wave_plan.py `
  --prelabel-root diagnostics\question-detector-prelabel-kpai-observation-expanded `
  --exclude-approved-jsonl diagnostics\question-detector-active-batch-kpai-001-workbench\approved_boxes_smoke.jsonl `
  --exclude-approved-jsonl diagnostics\question-detector-review-workbench-kpai\approved_boxes_smoke.jsonl `
  --exclude-queued-root diagnostics\question-detector-active-batch-kpai-001 `
  --exclude-queued-root diagnostics\question-detector-active-batch-kpai-002 `
  --exclude-queued-root diagnostics\question-detector-active-batch-kpai-003-quarantine `
  --exclude-queued-root diagnostics\question-detector-review-backlog-risk-section-top1 `
  --include-quarantine `
  --build-workbenches `
  --max-waves 12 `
  --max-boxes-per-wave 120 `
  --out diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120 `
  --clean
```

Current observation-expanded result:

- review candidates: 609 positive review-only images, 135 negative candidates
- prelabels: 561 images with boxes, 48 quarantined images, 1334 draft boxes
- uncapped wave capacity: 10 waves, 240 selected images, 950 selected boxes, `exhausted=false`
- preferred capped queue: 12 waves, 239 selected images, 949 selected boxes, `max_boxes_per_wave=120`, `exhausted=false`
- with the existing backlog and the old final 1-box wave, total pending review is 1030 boxes

Useful broader source inventory:

- `D:\AI\kpai\backend\data`: 631 image rows, 569 exact-unique image files, 29 account SQLite DBs.
- `c92a0e4ea0244cd281d8224a72639c9d.sqlite3`: 580 observations; this is the current main KPAI source.
- `f222b087a8f84687b4b85690f7eaaf81.sqlite3`: 43 additional observations for expansion probes.
- `diagnostics\question-detector-kpai-neg-all-observations`: 574 negative candidates.
- `diagnostics\question-detector-kpai-neg-textless`: 112 textless negative candidates.

Then export the detector dataset:

```powershell
python scripts\question_detector_dataset.py `
  --reviewed-prelabels diagnostics\question-detector-review-workbench-kpai\approved_boxes.jsonl `
  --negative-images-dir diagnostics\question-detector-review-candidates-kpai\negative `
  --split-scope session `
  --min-production-negative-images 50 `
  --out diagnostics\question-detector-dataset-reviewed-v1 `
  --clean
```

Accepted prelabel inputs:

- edited `draft_boxes.jsonl` / `approved_boxes.jsonl` rows whose boxes have `annotation_status` in `approved,accepted,corrected,verified`;
- edited COCO files whose annotations or images have an approved status;
- Label Studio exports with final `annotations[].result` rectangles labeled `question_block`.

Label Studio `predictions` and raw draft boxes are intentionally ignored. This lets the historical-image pipeline move quickly without silently turning model guesses into ground truth.

Smoke validation on 2026-06-30:

- raw draft prelabel import produced `0` training labels, as expected;
- an approved-smoke fixture with 8 images / 13 approved boxes plus 80 negatives exported 87 images and 11 accepted annotations;
- 2 duplicate boxes were sent to `annotations/review.jsonl`;
- evaluator self-check against exported COCO/manifest returned recall/precision/F1 = 1.0;
- the earlier 3-fold CV fixture was pilot-ready with no exact or near-duplicate cross-split leakage, but still not model-training-ready because the approved label count was tiny.
- dataset export now clusters exact/aHash near-duplicate split groups before train/val/test assignment by default. `diagnostics\question-detector-dataset-merged-reviewed-next-clustered` exports 87 images / 12 reviewed annotations / 80 negatives with 20 effective split groups, 0 cross-split near-duplicate pairs, `has_error=false`, and `pilot_eval_ready=true`; it remains `model_training_ready=false` only because reviewed scale is still smoke-sized.
- the explicit regression fixture `diagnostics\question-detector-dataset-merged-reviewed-next-no-cluster-smoke` disables clustering and reproduces the old 2 cross-split aHash near-duplicate pairs with `has_error=true`.
- `diagnostics\question-detector-cv-merged-reviewed-next-clustered` exports 3 grouped folds from the clustered dataset; all fold audits have `has_error=false` and 0 residual cross-split near-duplicate pairs, while staying `model_training_ready=false` because the label count is still 12.
- merged-reviewed CV smoke now defaults cross-split aHash near-duplicates to `error`; `diagnostics\question-detector-cv-merged-reviewed-smoke-neardupe-gate` correctly blocks fold-01 with one near-duplicate pair and marks that fold `model_training_ready=false`.
- review workbench smoke exported 5 images / 8 approved boxes from decisions JSON; dataset export accepted all 8 labels with 0 review rows, and evaluator self-check returned recall/precision/F1 = 1.0.
- active-learning batch 001 selected 24 diverse review images / 41 boxes after excluding already approved smoke images; its workbench smoke exported 3 images / 6 boxes, and dataset export accepted all 6 labels with 0 review rows.
- active-learning batch 002 excluded batch 001 with `--exclude-queued-root`, skipped 21 already queued images, selected 10 fresh images / 17 boxes, and generated a workbench.
- active-learning batch 003 added `--include-quarantine`, excluded batches 001 and 002, selected 10 harder images / 32 boxes including quarantine/high-risk cases, and generated a workbench.
- batches 001, 002, and 003 have 0 image overlap and together queue 44 images / 90 candidate boxes for review.

When multiple workbenches have been reviewed, merge them before dataset export:

```powershell
python scripts\question_detector_merge_reviewed.py `
  diagnostics\question-detector-review-workbench-kpai\approved_boxes_smoke.jsonl `
  diagnostics\question-detector-active-batch-kpai-001-workbench\approved_boxes_smoke.jsonl `
  --out diagnostics\question-detector-merged-reviewed-smoke `
  --clean
```

The merger copies referenced images, groups identical images by file hash, removes duplicate boxes at high IoU, records possible conflicts, and writes `annotations/approved_boxes.jsonl` for `question_detector_dataset.py`.

Current merge smoke:

- input approved boxes: 14
- merged images: 7
- merged boxes: 12
- deduped boxes: 2
- conflicts: 0

The loop runner can do this merge automatically before dataset export with
`--merge-reviewed`. Use this when running the next recursive training/review
cycle from multiple approved workbench outputs.

## Reviewed Duplicate Propagation

Exact or visually identical historical screenshots can safely reuse reviewed
boxes when the target image is the same size and its draft boxes overlap the
transferred boxes. This reduces manual review without turning model guesses into
ground truth.

Automatic approved propagation is intentionally limited to exact aHash matches.
The script now rejects non-exact aHash approved output unless `--review-candidates`
is supplied, and the default draft-overlap threshold is `--min-draft-iou 0.45`.

```powershell
python scripts\question_detector_propagate_reviewed.py `
  --approved diagnostics\question-detector-merged-reviewed-next `
  --target diagnostics\question-detector-active-batch-kpai-002 `
  --target diagnostics\question-detector-active-batch-kpai-003-quarantine `
  --target diagnostics\question-detector-error-mining-bootstrap-smoke `
  --out diagnostics\question-detector-propagated-reviewed-next-exact-ahash `
  --clean
```

Current exact-aHash propagation result:

- propagated package: `diagnostics\question-detector-propagated-reviewed-next-exact-ahash`
- propagated images / boxes: 2 images / 4 boxes
- match quality: both propagated rows have `ahash_distance=0` and `max_draft_iou=1.0`
- merged propagated output: `diagnostics\question-detector-merged-reviewed-next-propagated-exact`, 9 images / 16 boxes / 0 conflicts
- dataset audit: `diagnostics\question-detector-dataset-merged-reviewed-next-propagated-exact`, 89 images / 16 reviewed annotations / 80 negatives, `pilot_eval_ready=true`, `has_error=false`, `model_training_ready=false`
- CV audit: `diagnostics\question-detector-cv-merged-reviewed-next-propagated-exact`, 3 folds, all fold audits `has_error=false` with 0 residual cross-split near-duplicate pairs

Near-duplicate propagation beyond exact aHash is review-only. Generate those candidates with `--review-candidates` so they stay as `draft_review_required` rows and cannot enter training until manually accepted:

```powershell
python scripts\question_detector_propagate_reviewed.py `
  --approved diagnostics\question-detector-merged-reviewed-next `
  --target diagnostics\question-detector-active-batch-kpai-002 `
  --target diagnostics\question-detector-active-batch-kpai-003-quarantine `
  --target diagnostics\question-detector-error-mining-bootstrap-smoke `
  --out diagnostics\question-detector-propagation-review-candidates-ahash1-2 `
  --clean `
  --min-ahash-distance 1 `
  --max-ahash-distance 2 `
  --review-candidates
```

Current near-duplicate review-candidate result:

- package: `diagnostics\question-detector-propagation-review-candidates-ahash1-2`
- review workbench: `diagnostics\question-detector-propagation-review-candidates-ahash1-2-workbench`
- propagated review candidates: 1 image / 1 box
- match quality: `ahash_distance=2`, `max_draft_iou=0.543977`
- status: `draft_review_required`; do not merge into detector training until the box is manually approved or corrected

The older exploratory package `diagnostics\question-detector-propagated-reviewed-next` used `--max-ahash-distance 2` without the review-candidate guard and found 3 images / 5 boxes; keep it as diagnostics only, not as training input.

## Mobile Capture Augmentation

After reviewed labels are exported, optionally generate iPhone/iPad-style capture variation:

```powershell
python scripts\question_detector_mobile_augment.py `
  --dataset diagnostics\question-detector-dataset-reviewed-v1 `
  --out diagnostics\question-detector-dataset-reviewed-v1-mobile-aug `
  --variants-per-image 3 `
  --clean
```

This simulates small crops, rotations, exposure/contrast shifts, blur, resampling, and JPEG compression while preserving approved boxes. The output keeps the normal training contract: `images/{split}`, `labels/{split}`, `annotations/manifest.jsonl`, `annotations/image_manifest.jsonl`, COCO exports, and `yolo_dataset.yaml`.

Augmentation improves robustness, but it must not fake source diversity. The augmented audit inherits `model_training_ready=false` unless the source dataset was already model-training-ready, so it cannot bypass the release gate.

Current smoke run:

```powershell
python scripts\question_detector_mobile_augment.py `
  --dataset diagnostics\question-detector-active-batch-kpai-001-approved-smoke `
  --out diagnostics\question-detector-mobile-augment-smoke `
  --variants-per-image 2 `
  --clean
```

Result: 83 original images expanded to 249 images, 6 reviewed boxes expanded to 18 boxes, `skipped_variants=0`, and `model_training_ready=false` because the source dataset is still smoke-scale.

## Synthetic / Dense Stress Fixtures

Synthetic dense fixtures are pressure/regression fixtures, not production reviewed labels. They may contain oracle boxes for repeatable tests, but those boxes are fixture expectations, not human-reviewed real-world ground truth.

Allowed uses:

- pipeline smoke tests and exploratory training only with `--allow-unready`;
- hard-example discovery and model failure mining;
- replay, fallback-tune, candidate-cap sweep, candidate-eval, and union-area stress reports as additional negative gates;
- blocking release when dense pages expose missed boxes, unsafe cap behavior, source-binding errors, or fallback cost regressions.

Not allowed:

- do not pass synthetic dense fixture outputs to `question_detector_dataset.py --reviewed-prelabels`;
- do not merge them with `question_detector_merge_reviewed.py`;
- do not count synthetic boxes toward `model_training_ready`, reviewed annotation counts, represented reviewed GT counts, fold counts, or release selection evidence;
- do not use synthetic dense eval as a substitute for real reviewed held-out session/batch evaluation.

Generate the current dense phone/iPad pressure fixture:

```powershell
python scripts\question_detector_synthetic_dense_fixture.py `
  --out diagnostics\question-detector-synthetic-dense `
  --clean
```

This writes 10 diagnostic images with 256 expected boxes across phone dense, phone cap-overflow, iPad two-column, iPad three-column landscape, and weak-textless profiles. The root includes `annotations/manifest.jsonl` for expected boxes, `annotations/draft_boxes.jsonl` for oracle candidates, `annotations/image_manifest.jsonl`, COCO export, YOLO labels, contact sheet, `audit.json`, and `summary.json`. The audit is forced to `model_training_ready=false`.

Run the dense stress chain:

```powershell
python scripts\question_observation_replay.py `
  --prelabel-root diagnostics\question-detector-synthetic-dense `
  --out diagnostics\question-observation-replay-synthetic-dense-oracle `
  --clean

python scripts\question_observation_replay.py `
  --image-dir diagnostics\question-detector-synthetic-dense\images `
  --measure-layout `
  --use-measured-layout-boxes `
  --out diagnostics\question-observation-replay-synthetic-dense-layout `
  --clean

python scripts\question_observation_candidate_eval.py `
  --replay diagnostics\question-observation-replay-synthetic-dense-oracle `
  --ground-truth-manifest diagnostics\question-detector-synthetic-dense\annotations\manifest.jsonl `
  --out diagnostics\question-observation-candidate-eval-synthetic-dense-oracle `
  --clean

python scripts\question_observation_candidate_eval.py `
  --replay diagnostics\question-observation-replay-synthetic-dense-layout `
  --ground-truth-manifest diagnostics\question-detector-synthetic-dense\annotations\manifest.jsonl `
  --out diagnostics\question-observation-candidate-eval-synthetic-dense-layout `
  --clean

python scripts\question_observation_candidate_cap_sweep.py `
  --replay diagnostics\question-observation-replay-synthetic-dense-oracle `
  --ground-truth-manifest diagnostics\question-detector-synthetic-dense\annotations\manifest.jsonl `
  --out diagnostics\question-observation-candidate-cap-sweep-synthetic-dense-oracle-backend-tuned `
  --clean

python scripts\question_observation_fallback_tune.py `
  --replay diagnostics\question-observation-replay-synthetic-dense-oracle `
  --ground-truth-manifest diagnostics\question-detector-synthetic-dense\annotations\manifest.jsonl `
  --out diagnostics\question-observation-fallback-tune-synthetic-dense-oracle-backend-tuned `
  --clean

python scripts\question_observation_section_crop_sweep.py `
  --replay diagnostics\question-observation-replay-synthetic-dense-oracle `
  --ground-truth-manifest diagnostics\question-detector-synthetic-dense\annotations\manifest.jsonl `
  --out diagnostics\question-observation-section-crop-sweep-synthetic-dense-oracle `
  --clean

python scripts\question_observation_section_sender_eval.py `
  --replay diagnostics\question-observation-replay-synthetic-dense-oracle `
  --ground-truth-manifest diagnostics\question-detector-synthetic-dense\annotations\manifest.jsonl `
  --out diagnostics\question-observation-section-sender-eval-synthetic-dense-oracle `
  --clean

python scripts\question_observation_section_sender_eval.py `
  --replay diagnostics\question-observation-replay-synthetic-dense-layout `
  --ground-truth-manifest diagnostics\question-detector-synthetic-dense\annotations\manifest.jsonl `
  --enable-empty-frame-section `
  --out diagnostics\question-observation-section-sender-eval-synthetic-dense-layout-empty-section `
  --clean

python scripts\question_observation_replay.py `
  --image-dir diagnostics\question-detector-synthetic-dense\images `
  --measure-layout `
  --use-measured-layout-boxes `
  --out diagnostics\question-observation-replay-synthetic-dense-layout-dense-prelabel-v5-gated `
  --clean

python scripts\question_observation_candidate_eval.py `
  --replay diagnostics\question-observation-replay-synthetic-dense-layout-dense-prelabel-v5-gated `
  --ground-truth-manifest diagnostics\question-detector-synthetic-dense\annotations\manifest.jsonl `
  --out diagnostics\question-observation-candidate-eval-synthetic-dense-layout-dense-prelabel-v5-gated `
  --clean

python scripts\question_observation_candidate_cap_sweep.py `
  --replay diagnostics\question-observation-replay-synthetic-dense-layout-dense-prelabel-v5-gated `
  --ground-truth-manifest diagnostics\question-detector-synthetic-dense\annotations\manifest.jsonl `
  --out diagnostics\question-observation-candidate-cap-sweep-synthetic-dense-layout-dense-prelabel-v5-gated-backend-tuned `
  --clean

python scripts\question_observation_fallback_tune.py `
  --replay diagnostics\question-observation-replay-synthetic-dense-layout-dense-prelabel-v5-gated `
  --ground-truth-manifest diagnostics\question-detector-synthetic-dense\annotations\manifest.jsonl `
  --out diagnostics\question-observation-fallback-tune-synthetic-dense-layout-dense-prelabel-v5-gated-backend-tuned `
  --clean

python scripts\question_detector_synthetic_dense_stress_check.py `
  --layout-candidate-eval diagnostics\question-observation-candidate-eval-synthetic-dense-layout-dense-prelabel-v5-gated `
  --oracle-cap-sweep diagnostics\question-observation-candidate-cap-sweep-synthetic-dense-oracle-backend-tuned `
  --layout-cap-sweep diagnostics\question-observation-candidate-cap-sweep-synthetic-dense-layout-dense-prelabel-v5-gated-backend-tuned `
  --oracle-fallback diagnostics\question-observation-fallback-tune-synthetic-dense-oracle-backend-tuned `
  --layout-fallback diagnostics\question-observation-fallback-tune-synthetic-dense-layout-dense-prelabel-v5-gated-backend-tuned `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-synthetic-dense-oracle `
  --empty-section-sender-eval diagnostics\question-observation-section-sender-eval-synthetic-dense-layout-empty-section `
  --out diagnostics\question-detector-synthetic-dense-stress-check-dense-prelabel-v5-gated-backend-tuned `
  --clean

python scripts\question_detector_layout_regression_check.py `
  --out diagnostics\question-detector-layout-regression-check-v5-gated-backend-tuned `
  --clean
```

This checker exits nonzero when `stress_ready=false`; add `--allow-failed-gate` only when intentionally regenerating a failed diagnostic report.

Current dense stress result:

- `diagnostics\question-detector-synthetic-dense-stress-check-dense-prelabel-v5-gated-backend-tuned` reports `stress_ready=true`, `transport_ready=true`, and `detector_ready=true`, with 0 hard failures and 2 expected cap-only warnings;
- dense layout prelabel v5-gated fixes the synthetic dense local-candidate gap while preserving real KPAI behavior: synthetic dense covers 256/256 represented expected boxes with 283 candidates, 0 missed boxes, 0 overcrop hard cases, reviewed-candidate match rate 0.932862, median overcrop ratio 1.241940, p90 1.377941, max 2.130331, and crop pixels at 1.960380x full-frame before section batching;
- the replay at `diagnostics\question-observation-replay-synthetic-dense-layout-dense-prelabel-v5-gated` measures 10/10 successful layout pages, 283 raw boxes, 207 cross-frame-dedup boxes, rect-only cross-frame overhead 1.037756x full-frame bytes, crop pixels after dedupe 1.484400x full-frame, and layout latency around 50 ms median on this Windows machine;
- the failed v4 real-data replay is kept as a regression lesson: `diagnostics\question-observation-replay-kpai-layout-measured-v4` inflated 41 KPAI images from 67 to 403 candidates and `diagnostics\question-observation-candidate-eval-kpai-layout-v4-smoke-gt` fell to 7/11 reviewed GT recall. v5 gates dense splitting to `page_source == full_image_fallback`, so `diagnostics\question-observation-candidate-eval-kpai-layout-v5-dense-gated` restores 67 candidates and 11/11 recall;
- `diagnostics\question-detector-layout-regression-check-v5-gated-backend-tuned` now requires real KPAI reviewed-smoke recall/candidate-count stability, synthetic dense recall/stress readiness, and backend fallback cost gates on both KPAI and propagated reviewed replay before a dense layout change is considered safe;
- the stress checker now keeps current layout-candidate recall separate from zero-candidate fallback evidence via `--empty-layout-cap-sweep` and `--empty-layout-fallback`, so a better local layout no longer invalidates the no-crop fallback safety proof;
- oracle candidates still prove the fixture itself is valid: 256/256 represented expected boxes are coverable;
- production cap 12 by itself keeps only 120/256 oracle candidates before fallback, so crop-only recall is 0.46875 and 120 strong/high-confidence candidates are skipped; this is now a warning because the live sender replaces dense cap-risk frames with section crops;
- cap-only backend fallback recovers oracle policy recall to 1.0, but total crop+fallback VLM pixels rise to 1.756221x full-frame; this is also a warning once section-sender evidence passes. On current measured-layout candidates, cap-only fallback-protected recall is only 0.953125 after disabling standalone all-weak/low-confidence fallback, which reinforces that dense production transport must use section sender rather than cap-only fallback;
- live `ios_page_dense_v1` sender covers 256/256 expected boxes with 10 section crops plus 20 top-2 companion single-question crops, combined crop recall 1.0, fallback-protected recall 1.0, zero fallback images, and 0.701777x full-frame VLM pixels; sending all selected top-12 crops with sections costs 1.331615x and is rejected as too expensive;
- live `ios_page_empty_v1` sender covers the same 256/256 expected boxes when the measured layout emits 0 candidates, with 10 section crops, combined crop recall 1.0, fallback-protected recall 1.0, zero fallback images, and 0.563498x full-frame VLM pixels;
- no-cap oracle replay keeps crop-only recall 1.0, but crop pixels are still 1.565434x full-frame on this synthetic dense set, proving dense multi-question pages need a smarter batching/cap policy rather than simply "more crops forever."
- section-crop sweep finds a diagnostic alternative: `page__side1200` covers 256/256 expected boxes with 10 section crops, no fallback, and 0.576576x full-frame VLM pixels. The KPAI replay also has a passing section strategy, `horizontal__n4__side1200`, with represented-GT recall 1.0 and 0.887219x full-frame VLM pixels.

Section crops are extraction sources, not question identities. iOS now sends risk-gated dense page-tight section rects with `crop_kind="section"`, `question_key_strength="section_crop"`, `section_key`, `covered_question_count`, `covered_question_keys`, `covered_question_numbers`, `covered_subrects`, `section_strategy` (`ios_page_dense_v1` for cap overflow, `ios_page_empty_v1` for no-candidate study frames), `coverage_score`, and `section_companion_candidate_limit=2`. The top-2 companion single-question crops act as anchors if a section crop is slightly off while keeping dense VLM pixels below the 0.75x release ceiling. Backend manifest/source handling preserves the same fields. Backend VLM extraction preserves `input_index` for every question extracted from a section crop and carries `source_crop_kind=section` plus section metadata into the extracted-question payload. When section metadata or section area proves the source covers the risky page region, backend fallback treats the section as protected and does not re-trigger full-frame fallback through later low-coverage or weak-key rules. Training/export code must not treat section crops as single-question reviewed boxes; the remaining product work is to validate this sender policy on broader real dense iPhone/iPad captures and replace smoke-scale detector data with release-ready labels.

## Bootstrap Exploration Dataset

When the reviewed set is still too small to train a useful detector, build a non-release bootstrap dataset from approved labels plus high-confidence draft prelabels:

```powershell
python scripts\question_detector_bootstrap_dataset.py `
  --prelabel-root diagnostics\question-detector-prelabel-kpai `
  --approved-jsonl diagnostics\question-detector-active-batch-kpai-001-workbench\approved_boxes_smoke.jsonl `
  --negative-images-dir diagnostics\question-detector-review-candidates-kpai\negative `
  --out diagnostics\question-detector-bootstrap-kpai-smoke `
  --clean
```

The bootstrap exporter keeps the normal training contract: `images/{split}`, `labels/{split}`, COCO, manifest JSONL, audit, and preview contact sheet. It filters draft boxes by score, area, aspect ratio, and risk flags, adds `pseudo_label` / `bootstrap_draft` quality flags, and keeps reviewed boxes separate from pseudo boxes in `audit.json`.

Important: bootstrap data is for early model exploration and hard-example mining only. It is forced to `model_training_ready=false`, because pseudo boxes are not ground truth.

Current bootstrap run:

- source images: 126
- annotations: 54
- human-reviewed annotations: 6
- pseudo annotations: 48
- negative images: 83
- self-check against COCO/manifest: recall/precision/F1 = 1.0
- draft filter stats: 48 accepted labels, 19 rejected low-score boxes, 3 rejected oversized boxes, 3 draft images skipped because reviewed labels already existed

Short exploratory training on this bootstrap dataset:

```powershell
python scripts\question_detector_experiment_matrix.py `
  --dataset diagnostics\question-detector-bootstrap-kpai-smoke `
  --model yolo11n.pt `
  --imgsz 416 `
  --epochs 5 `
  --batch 4 `
  --workers 0 `
  --device cpu `
  --allow-unready `
  --allow-failed-eval `
  --execute `
  --out diagnostics\question-detector-experiment-bootstrap-smoke `
  --clean
```

Current result: `failed_eval_gate`. At the normal prediction confidence this first bootstrap model produced no held-out test predictions, while Ultralytics validation showed early recall/mAP signal during training. For diagnosis, forcing the exported `best.pt` down to `conf=0.001` recovered 20/22 held-out boxes (recall 0.9091) but produced 7,859 predictions, precision 0.0025, and 186.6429 false positives per image. This means the training path has signal, but the current pseudo-label set is far too weak/noisy to calibrate a usable mobile detector. Use low-confidence output only to mine hard examples, not as release evidence.

Turn those failed detector examples into the next review queue:

```powershell
python scripts\question_detector_error_mining.py `
  --eval-summary diagnostics\question-detector-experiment-bootstrap-smoke\runs\question-detector-bootstrap-kpai-smoke__yolo11n__img416__seed42__ep5\eval\summary.json `
  --dataset-root diagnostics\question-detector-bootstrap-kpai-smoke `
  --out diagnostics\question-detector-error-mining-bootstrap-smoke `
  --clean

python scripts\question_detector_review_workbench.py `
  --prelabel-root diagnostics\question-detector-error-mining-bootstrap-smoke `
  --out diagnostics\question-detector-error-mining-bootstrap-smoke-workbench `
  --clean
```

Current result: 16 hard images / 22 missed boxes were converted into a normal review workbench. Green boxes in `hard_examples_contact_sheet.jpg` are missed ground-truth boxes; red boxes would be false positives when a model emits extra boxes. Approved/corrected rows from this workbench can feed the next reviewed dataset export.

Clean-shape bootstrap diagnosis:

```powershell
python scripts\question_detector_bootstrap_dataset.py `
  --prelabel-root diagnostics\question-detector-prelabel-kpai `
  --approved-jsonl diagnostics\question-detector-merged-reviewed-next-propagated-exact\annotations\approved_boxes.jsonl `
  --negative-images-dir diagnostics\question-detector-review-candidates-kpai\negative `
  --reject-quality-flags few_text_lines,short_block,tall_block,near_full_page `
  --out diagnostics\question-detector-bootstrap-kpai-clean-shape `
  --clean

python scripts\question_detector_train.py `
  --dataset diagnostics\question-detector-bootstrap-kpai-clean-shape `
  --out diagnostics\question-detector-experiment-bootstrap-clean-shape\runs\clean-shape__yolo11n__img416__seed42__ep8 `
  --model yolo11n.pt `
  --name question-block `
  --epochs 8 `
  --imgsz 416 `
  --batch 4 `
  --device cpu `
  --workers 0 `
  --allow-unready `
  --predict-test `
  --conf 0.25 `
  --eval-score-thresholds 0.001,0.002,0.005,0.01,0.02,0.05,0.10,0.20,0.35,0.50 `
  --allow-failed-eval `
  --clean
```

Current result: `failed_eval_gate`. The cleaner bootstrap has 135 images, 27 annotations, 16 human-reviewed boxes, 11 pseudo boxes, and 118 negatives. The 8-epoch `yolo11n` run still only works at extremely low confidence: recall 0.9091, precision 0.0012, and 189.8864 false positives per image. Rejecting bad-shape draft labels did not remove the red-grid false positives, so the next bottleneck is reviewed positive/negative boundary data, not architecture.

Longer clean-shape training does not fix calibration by itself:

```powershell
python scripts\question_detector_train.py `
  --dataset diagnostics\question-detector-bootstrap-kpai-clean-shape `
  --out diagnostics\question-detector-experiment-bootstrap-clean-shape\runs\clean-shape__yolo11n__img416__seed42__ep25 `
  --model yolo11n.pt `
  --name question-block `
  --epochs 25 `
  --imgsz 416 `
  --batch 4 `
  --device cpu `
  --workers 0 `
  --allow-unready `
  --predict-test `
  --conf 0.25 `
  --eval-score-thresholds 0.001,0.002,0.005,0.01,0.02,0.05,0.10,0.20,0.35,0.50 `
  --allow-failed-eval `
  --clean
```

Current result: `failed_eval_gate`. Compared with the 8-epoch run, ep25 keeps recall 0.9091 at `score=0.001` but worsens false positives from 189.8864 to 281.2727 per image. At `score=0.01`, recall is only 0.6364 with 35.4545 false positives per image, and at `score=0.05` recall falls to 0.0. This confirms that simply training longer on the same weak bootstrap set does not produce a usable iOS detector; the next useful work is reviewed boundary data and hard false-positive correction.

Lightweight post-processing cannot rescue this detector distribution:

```powershell
python scripts\question_detector_postprocess_sweep.py `
  --ground-truth diagnostics\question-detector-bootstrap-kpai-clean-shape\annotations\coco_test.json `
  --predictions-yolo-dir diagnostics\question-detector-experiment-bootstrap-clean-shape\runs\clean-shape__yolo11n__img416__seed42__ep8\predict\question-block-test\labels `
  --out diagnostics\question-detector-postprocess-sweep-clean-shape-ep8 `
  --clean

python scripts\question_detector_postprocess_sweep.py `
  --ground-truth diagnostics\question-detector-bootstrap-kpai-clean-shape\annotations\coco_test.json `
  --predictions-yolo-dir diagnostics\question-detector-experiment-bootstrap-clean-shape\runs\clean-shape__yolo11n__img416__seed42__ep25\predict\question-block-test\labels `
  --out diagnostics\question-detector-postprocess-sweep-clean-shape-ep25 `
  --clean
```

Current result: both sweeps have `pass_eval_gate_count=0` and no row reaches the target recall. On ep8, the default 1800-row sweep's best row only reaches recall 0.2727 with 35.8182 false positives per image; a recall-preserving 3840-row diagnostic sweep still tops out at recall 0.7273 with 51.2045 false positives per image. On ep25, the default sweep tops out at recall 0.7273 with 80.1136 false positives per image, and a smaller recall-preserving control reaches recall 0.8182 with 98.5455 false positives per image. This rules out score/shape/NMS/top-k tuning as a release path for the current labels; the model must learn better question boundaries from reviewed hard examples.

The latest hard-example queue is capped for review instead of dumping every low-confidence false positive:

```powershell
python scripts\question_detector_error_mining.py `
  --eval-summary diagnostics\question-detector-experiment-bootstrap-clean-shape\runs\clean-shape__yolo11n__img416__seed42__ep8\eval\summary.json `
  --dataset-root diagnostics\question-detector-bootstrap-kpai-clean-shape `
  --out diagnostics\question-detector-error-mining-bootstrap-clean-shape `
  --limit 48 `
  --min-area-ratio 0.002 `
  --max-area-ratio 0.80 `
  --max-false-positives-per-image 6 `
  --max-boxes-total 220 `
  --clean

python scripts\question_detector_review_workbench.py `
  --prelabel-root diagnostics\question-detector-error-mining-bootstrap-clean-shape `
  --out diagnostics\question-detector-error-mining-bootstrap-clean-shape-workbench `
  --limit 43 `
  --clean
```

Current result: `diagnostics\question-detector-error-mining-bootstrap-clean-shape-workbench` contains 43 images and 220 review boxes: 1 missed box plus 219 capped false-positive boxes. In `question_detector_error_mining.py`, `--threshold` and the summary `threshold` field mean the evaluator IoU threshold key, not the model score threshold; the false positives here come from the low-score sweep exported by the failed training run. Red false-positive boxes now carry `default_review_status="rejected"`, `training_export_default=false`, and `requires_correction_for_training=true`; the workbench bulk-approve action skips them, and export keeps them out of training unless they are deliberately marked `corrected` or `verified`. The iteration report at `diagnostics\question-detector-iteration-clean-shape` sets the stage to `hard_example_review`; do not bundle a detector until those hard examples are reviewed, merged, retrained through cross-validation, exported to Core ML, and passed by the release gate.

The ep25 error queue at `diagnostics\question-detector-error-mining-bootstrap-clean-shape-ep25-workbench` contains 37 images and 220 capped boxes, again with 1 missed box and 219 default-rejected false positives. None of the current clean-shape hard-example rows are known negative-only images, so they are not safe to auto-promote as hard-negative images. Treat them as review/correction work: reject false positives, correct boxes that actually cover a question, then merge only corrected/verified labels into the next dataset.

Summarize the iteration after dataset export, training, release check, replay, and error mining:

```powershell
python scripts\question_detector_iteration_report.py `
  --dataset diagnostics\question-detector-dataset-merged-reviewed-next-propagated-exact `
  --experiment diagnostics\question-detector-loop-backend-tuned-fallback-delta-dryrun\experiment `
  --release-check diagnostics\question-detector-release-check-risk-section-top1-propagated-only `
  --replay diagnostics\question-observation-replay-reviewed-propagated-layout-v5-gated `
  --strategy-selection diagnostics\question-observation-strategy-selection-risk-section-top1 `
  --training-plan diagnostics\question-detector-training-readiness-plan-risk-section-top1 `
  --review-workbench diagnostics\question-detector-review-backlog-risk-section-top1-workbench `
  --review-workbench diagnostics\question-detector-review-wave-plan-kpai-after-backlog\wave-001-workbench `
  --out diagnostics\question-detector-iteration-report-risk-section-top1-backlog `
  --clean
```

The report now audits review workbench closure as well as model/release/replay state. It checks whether each workbench has reviewer decisions, whether those decisions have been exported to `approved_boxes.jsonl`, how many boxes are still pending review, and the exact merge command for approved JSONL inputs.
Iteration reports should include the fallback-delta queues because backend-tuned
fallback removes legacy full-frame protection on no-GT frames; those removals
remain review-closure work even when represented-GT misses are 0.

The risk-section backlog at
`diagnostics\question-detector-review-backlog-risk-section-top1` consolidates
the current active/fallback/propagation queues into 38 selected images and 80
pending boxes after exact/aHash de-duplication. Its workbench at
`diagnostics\question-detector-review-backlog-risk-section-top1-workbench`
is the preferred next manual review entry point for this loop.

Build the current training-readiness plan from that backlog:

```powershell
$argsList = @(
  'scripts\question_detector_training_readiness_plan.py',
  '--dataset', 'diagnostics\question-detector-dataset-merged-reviewed-next-propagated-exact',
  '--review-workbench', 'diagnostics\question-detector-review-backlog-risk-section-top1-workbench',
  '--review-workbench', 'diagnostics\question-detector-review-wave-plan-kpai-after-backlog\wave-001-workbench',
  '--review-backlog', 'diagnostics\question-detector-review-backlog-risk-section-top1',
  '--review-wave-plan', 'diagnostics\question-detector-review-wave-plan-kpai-after-backlog',
  '--review-wave-plan', 'diagnostics\question-detector-review-wave-plan-kpai-expanded',
  '--review-wave-plan', 'diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120',
  '--release-check', 'diagnostics\question-detector-release-check-risk-section-top1-propagated-only',
  '--strategy-selection', 'diagnostics\question-observation-strategy-selection-risk-section-top1',
  '--out', 'diagnostics\question-detector-training-readiness-plan-risk-section-top1',
  '--clean'
)
foreach ($i in 1..12) {
  $argsList += '--review-workbench'
  $argsList += ('diagnostics\question-detector-review-wave-plan-kpai-observation-expanded-boxcap120\wave-{0:D3}-workbench' -f $i)
}
python @argsList
```

Current plan result: 1030 pending review boxes would raise the dataset from 16
to at most 1046 reviewed annotations, enough to cross the default production
annotation target if the boxes are approved after correction/deduplication. The
estimated split-group gain must still be verified by rerunning dataset
export/audit after approved boxes are merged, because only approved, deduped
labels count toward training readiness.

Build one consolidated review queue index from the training plan:

```powershell
python scripts\question_detector_review_queue_manifest.py `
  --training-plan diagnostics\question-detector-training-readiness-plan-risk-section-top1 `
  --merge-out diagnostics\question-detector-merged-reviewed-next-production-candidates `
  --out diagnostics\question-detector-review-queue-risk-section-top1 `
  --clean
```

Current queue result:
`diagnostics\question-detector-review-queue-risk-section-top1\index.html`
links all 14 workbenches, covering 1030 pending boxes. It also tracks decisions,
approved exports, and the eventual merge command once approved JSONL files are
available.

Build a balanced first-pass review and Codex comparison slice from the
consolidated queue:

```powershell
python scripts\question_detector_review_priority.py `
  --manifest diagnostics\question-detector-review-queue-risk-section-top1\review_queue_manifest.json `
  --limit-images 40 `
  --max-boxes-per-image 12 `
  --out diagnostics\question-detector-review-priority-risk-section-top1 `
  --review-root diagnostics\question-detector-review-priority-risk-section-top1-prelabel `
  --clean
```

Current priority result:
`diagnostics\question-detector-review-priority-risk-section-top1\review_priority.md`
selects 40 images from 1030 pending boxes. The default balanced strategy keeps
the first pass from being dominated by dense pages: 14 dense-layout images, 8
observation-unverified images, 7 active-quarantine images, 7 fallback-delta
images, and 4 active-capture images. The report highlights 299 high-priority
boxes for triage, but the generated review root includes all 411 draft boxes on
those 40 selected images so training labels do not become partial-image
annotations. It also writes `codex_benchmark_manifest.jsonl`, a stable list of
the same images for Codex-style segmentation comparison. This output is still a
review queue and benchmark seed only; draft boxes are not ground truth until
reviewed and exported.

Build the concentrated 40-image review UI:

```powershell
python scripts\question_detector_review_workbench.py `
  --prelabel-root diagnostics\question-detector-review-priority-risk-section-top1-prelabel `
  --out diagnostics\question-detector-review-priority-risk-section-top1-workbench `
  --clean
```

Current concentrated workbench result:
`diagnostics\question-detector-review-priority-risk-section-top1-workbench\index.html`
contains 40 images / 411 draft boxes, with median 8 boxes per image. After
review, download `approved_boxes.jsonl` from the workbench and use it as both:

- a reviewed-label input for `question_detector_merge_reviewed.py`;
- the human reference for `question_segmentation_benchmark_eval.py`.

The benchmark evaluator can match this approved JSONL through
`metadata.source_review_id`, so the downloaded workbench export does not need a
manual top-level `review_id` edit.

Current approved-export alias smoke:
`diagnostics\question-segmentation-benchmark-approved-alias-smoke-risk-section-top1\summary.md`
uses a synthetic approved JSONL with no top-level `review_id` and matches
411/411 boxes with precision/recall/F1 = 1.0. This proves the post-review JSONL
shape can be evaluated without manual field repair; it still is not a quality
metric.

Package the same image slice for Codex comparison:

```powershell
python scripts\question_segmentation_benchmark_pack.py `
  --benchmark-manifest diagnostics\question-detector-review-priority-risk-section-top1\codex_benchmark_manifest.jsonl `
  --review-priority diagnostics\question-detector-review-priority-risk-section-top1\review_priority.json `
  --out diagnostics\question-segmentation-codex-benchmark-pack-risk-section-top1 `
  --clean
```

The pack writes copied images, `benchmark_manifest.jsonl`, `codex_prompt.md`,
`codex_predictions_template.jsonl`, `draft_reference_boxes.jsonl`,
`eval_commands.md`, and `README.md`. Fill a copy of
`codex_predictions_template.jsonl` with Codex segmentation boxes as
`codex_predictions.jsonl`; preserve each `review_id` so the evaluator can match
rows even if image filenames change. Use the draft reference only for wiring
checks, not quality claims.

Run the current pack wiring self-check:

```powershell
python scripts\question_segmentation_benchmark_eval.py `
  --benchmark-manifest diagnostics\question-segmentation-codex-benchmark-pack-risk-section-top1\benchmark_manifest.jsonl `
  --reference-jsonl diagnostics\question-segmentation-codex-benchmark-pack-risk-section-top1\draft_reference_boxes.jsonl `
  --predictions-jsonl diagnostics\question-segmentation-codex-benchmark-pack-risk-section-top1\draft_reference_boxes.jsonl `
  --label draft_pack_self_check `
  --out diagnostics\question-segmentation-benchmark-pack-self-check-risk-section-top1 `
  --clean
```

Current self-check result:
`diagnostics\question-segmentation-benchmark-pack-self-check-risk-section-top1\summary.md`
matches 299/299 draft boxes with precision/recall/F1 = 1.0. That proves the
benchmark manifest, copied image paths, JSONL schema, and IoU matching are
wired correctly. It does not prove detector or Codex quality.
For a valid Codex comparison, also require
`summary.json.input_audit.reference.missing_images` and
`summary.json.input_audit.predictions.missing_images` to be empty; otherwise the
score is only over the matched subset.

After the 40-image priority slice is reviewed and exported, compare Codex
against human-approved boxes and mine failures in one command:

```powershell
python scripts\question_segmentation_benchmark_loop.py `
  --benchmark-manifest diagnostics\question-segmentation-codex-benchmark-pack-risk-section-top1\benchmark_manifest.jsonl `
  --reference-jsonl diagnostics\question-detector-review-priority-risk-section-top1-workbench\approved_boxes.jsonl `
  --predictions-jsonl diagnostics\question-segmentation-codex-benchmark-pack-risk-section-top1\codex_predictions.jsonl `
  --label codex_vs_reviewed `
  --out diagnostics\question-segmentation-benchmark-loop-codex-vs-reviewed-risk-section-top1 `
  --clean
```

Gate the comparison by cohort, not only overall F1. Codex must be checked
separately on dense-layout, fallback-delta, active-quarantine, active-capture,
and observation-unverified pages; failures in any cohort should become either
new review labels, detector hard examples, or backend/iOS segmentation policy
changes before promotion.

The loop writes:

- `eval/summary.json`: Codex-vs-reviewed precision/recall/F1, cohort metrics,
  and input coverage audit.
- `gate/gate.json` and `gate/gate.md`: hard pass/fail checks for input
  coverage, overall recall/precision/F1, missed/extra boxes, and cohort gates.
- `errors/annotations/draft_boxes.jsonl`: missed/extra boxes as a review queue.
- `error_workbench/index.html`: static UI for reviewing the mined Codex
  failures.
- `loop_report.json`: commands, return codes, and summaries. Status is
  `passed`, `failed_gate`, or `failed`; `failed_gate` still writes the mined
  error queue so misses can be reviewed.

The same steps can still be run manually:

```powershell
python scripts\question_segmentation_benchmark_eval.py `
  --benchmark-manifest diagnostics\question-segmentation-codex-benchmark-pack-risk-section-top1\benchmark_manifest.jsonl `
  --reference-jsonl diagnostics\question-detector-review-priority-risk-section-top1-workbench\approved_boxes.jsonl `
  --predictions-jsonl diagnostics\question-segmentation-codex-benchmark-pack-risk-section-top1\codex_predictions.jsonl `
  --label codex_vs_reviewed `
  --out diagnostics\question-segmentation-benchmark-codex-vs-reviewed-risk-section-top1 `
  --clean

python scripts\question_segmentation_benchmark_gate.py `
  --summary diagnostics\question-segmentation-benchmark-codex-vs-reviewed-risk-section-top1\summary.json `
  --out diagnostics\question-segmentation-benchmark-codex-vs-reviewed-risk-section-top1-gate `
  --allow-failed-gate

python scripts\question_segmentation_benchmark_mine.py `
  --eval diagnostics\question-segmentation-benchmark-codex-vs-reviewed-risk-section-top1 `
  --out diagnostics\question-segmentation-benchmark-codex-errors-risk-section-top1 `
  --clean

python scripts\question_detector_review_workbench.py `
  --prelabel-root diagnostics\question-segmentation-benchmark-codex-errors-risk-section-top1 `
  --out diagnostics\question-segmentation-benchmark-codex-errors-risk-section-top1-workbench `
  --clean
```

The mining step writes a prelabel-style review root. Missed reference boxes
default to `pending` and can be approved/corrected into training. False-positive
Codex boxes default to `rejected` and only enter training if deliberately
corrected or verified. This makes Codex comparison actionable: every miss or
extra box can flow into the same review/merge/training loop as detector hard
examples.

Current degraded-smoke proof:
`diagnostics\question-segmentation-benchmark-loop-degraded-smoke\loop_report.json`
uses a synthetic broken prediction file and reports 24 missed boxes, 1 false
positive, prediction coverage 38/40 images, and dense-layout recall 0.8571. Its
status is `failed_gate`, with failed checks for prediction image coverage,
overall recall, missed boxes, false-positive boxes, and cohort gates. The loop
still mines a 3-image / 25-box error queue and generates an error workbench
whose single false-positive box is default-rejected.

After review, merge this priority slice with the existing approved reviewed
exports and rerun dataset/training readiness:

```powershell
python scripts\question_detector_merge_reviewed.py `
  diagnostics\question-detector-active-batch-kpai-001-workbench\approved_boxes_smoke.jsonl `
  diagnostics\question-detector-review-workbench-kpai\approved_boxes_smoke.jsonl `
  diagnostics\question-detector-review-priority-risk-section-top1-workbench\approved_boxes.jsonl `
  --out diagnostics\question-detector-merged-reviewed-priority-risk-section-top1 `
  --clean

python scripts\question_detector_dataset.py `
  --reviewed-prelabels diagnostics\question-detector-merged-reviewed-priority-risk-section-top1 `
  --negative-images-dir diagnostics\question-detector-review-candidates-kpai\negative `
  --split-scope session `
  --out diagnostics\question-detector-dataset-priority-risk-section-top1 `
  --clean
```

The expected next release-gate blocker after this merge is no longer "no
priority queue"; it is whether the reviewed/merged labels actually pass dataset
audit, cross-validation/model selection, Core ML export, iOS bundle proof, and
runtime telemetry.

Current review-closure smoke report at `diagnostics\question-detector-iteration-review-closure-smoke`:

- stage: `hard_example_review`
- top action: respect release blockers; do not bundle this detector into iOS
- next data action: complete review workbenches with 71 boxes still missing a decisions export
- approved reviewed boxes available now: 14 boxes across `question-detector-review-workbench-kpai` and `question-detector-active-batch-kpai-001-workbench`
- merge command: `python scripts\question_detector_merge_reviewed.py diagnostics\question-detector-active-batch-kpai-001-workbench\approved_boxes_smoke.jsonl diagnostics\question-detector-review-workbench-kpai\approved_boxes_smoke.jsonl --out diagnostics\question-detector-merged-reviewed-next --clean`
- merged output: `diagnostics\question-detector-merged-reviewed-next` now contains 7 images / 12 boxes after 2 duplicate boxes were removed, with 0 conflicts
- duplicate propagation: `diagnostics\question-detector-propagated-reviewed-next-exact-ahash` safely adds 2 exact-aHash duplicate images / 4 boxes; the merged propagated package has 9 images / 16 boxes and 0 conflicts
- next dataset audit: `diagnostics\question-detector-dataset-merged-reviewed-next-propagated-exact` exports 89 images, 16 reviewed annotations, and 80 negatives; it is `pilot_eval_ready=true`, `has_error=false`, and still `model_training_ready=false` because it remains far below production label volume
- dataset gap: about 984 more reviewed annotations and 30 more effective split groups for the default production target
- replay status: rect-only upload still passes the product tradeoff check, with `rect_only_vs_full=1.002`, `crop_pixels_vs_full=0.5584`, and measured layout median `30.896 ms`

Current propagation-aware iteration report at `diagnostics\question-detector-iteration-propagation-review-next`:

- stage: `hard_example_review`
- release status: blocked by the same 11 expected hard failures across dataset scale, fold selection, training artifact, and iOS bundle/sync proof
- pending review: 72 boxes across active batch 002, active batch 003 quarantine, mined model errors, and the new aHash 1-2 propagation review candidate
- approved reviewed boxes available now: 14 boxes, before exact-aHash propagation
- propagated exact-aHash training-safe dataset: 89 images / 16 reviewed annotations / 80 negatives, with 0 residual cross-split near-duplicate pairs
- next data action: finish review/export for the pending workbenches, then merge approved boxes and rerun dataset/CV/experiment/release checks

For a repeatable one-command loop, use the loop runner:

```powershell
python scripts\question_detector_loop_runner.py `
  --dataset diagnostics\question-detector-dataset-merged-reviewed-next-propagated-exact `
  --models yolo11n.pt `
  --imgsz 416 `
  --epochs 1 `
  --batch 4 `
  --workers 0 `
  --device cpu `
  --allow-unready `
  --allow-failed-eval `
  --execute-training `
  --replay diagnostics\question-observation-replay-reviewed-propagated-layout-v5-gated `
  --dedupe-tune diagnostics\question-observation-dedupe-tune-kpai `
  --fallback-tune diagnostics\question-observation-fallback-tune-reviewed-propagated-layout-v5-gated-backend-tuned `
  --candidate-eval diagnostics\question-observation-candidate-eval-reviewed-propagated-layout-v5-gated `
  --candidate-cap-sweep diagnostics\question-observation-candidate-cap-sweep-reviewed-propagated-layout-v5-gated-backend-tuned `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-synthetic-dense-oracle `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-synthetic-dense-layout-empty-section `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-propagated-risk-section-top1-ios-signals `
  --strategy-selection diagnostics\question-observation-strategy-selection-risk-section-top1 `
  --runtime-perf diagnostics\question-observation-runtime-perf-v1 `
  --training-plan diagnostics\question-detector-training-readiness-plan-risk-section-top1 `
  --review-queue diagnostics\question-detector-review-queue-risk-section-top1 `
  --review-workbench diagnostics\question-detector-review-backlog-risk-section-top1-workbench `
  --review-workbench diagnostics\question-detector-review-wave-plan-kpai-after-backlog\wave-001-workbench `
  --backend-main backend\app\main.py `
  --xcode-build-log diagnostics\ios-build\pxj-question-detector-risk-section-top1-build.log `
  --out diagnostics\question-detector-loop-backend-tuned-fallback-delta-dryrun `
  --clean
```

The loop runner now forwards the same backend-tuned release evidence as the
manual release gate: fallback-tune, candidate-eval, candidate-cap sweep,
cap-overflow/empty/risk-frame section-sender evals, strategy-selection Pareto
evidence, runtime perf evidence, training-readiness planning, backend static
checks, and Xcode build proof. Keep fallback/candidate/cap evidence same-source inside one release
check; use the propagated replay for release gating and keep KPAI as a separate
regression fixture. The runner passes
`--allow-failed-gate` to the release check by default, so expected
`release_ready=false` blockers are preserved in `release_check.json` without
turning the orchestration step into a process failure. Use
`--strict-release-gate` only when a release-candidate CI job should fail on any
remaining blocker.

To start a fresh loop from reviewed workbench exports, let the runner merge
them first:

```powershell
python scripts\question_detector_loop_runner.py `
  --merge-reviewed diagnostics\question-detector-review-workbench-kpai\approved_boxes_smoke.jsonl `
  --merge-reviewed diagnostics\question-detector-active-batch-kpai-001-workbench\approved_boxes_smoke.jsonl `
  --negative-images-dir diagnostics\question-detector-review-candidates-kpai\negative `
  --split-scope session `
  --min-production-negative-images 50 `
  --models yolo11n.pt `
  --imgsz 416 `
  --epochs 1 `
  --allow-unready `
  --allow-failed-eval `
  --train-dry-run `
  --execute-training `
  --replay diagnostics\question-observation-replay-reviewed-propagated-layout-v5-gated `
  --dedupe-tune diagnostics\question-observation-dedupe-tune-kpai `
  --fallback-tune diagnostics\question-observation-fallback-tune-reviewed-propagated-layout-v5-gated-backend-tuned `
  --candidate-eval diagnostics\question-observation-candidate-eval-reviewed-propagated-layout-v5-gated `
  --candidate-cap-sweep diagnostics\question-observation-candidate-cap-sweep-reviewed-propagated-layout-v5-gated-backend-tuned `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-synthetic-dense-oracle `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-synthetic-dense-layout-empty-section `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-propagated-risk-section-top1-ios-signals `
  --strategy-selection diagnostics\question-observation-strategy-selection-risk-section-top1 `
  --runtime-perf diagnostics\question-observation-runtime-perf-v1 `
  --training-plan diagnostics\question-detector-training-readiness-plan-risk-section-top1 `
  --review-queue diagnostics\question-detector-review-queue-risk-section-top1 `
  --review-workbench diagnostics\question-detector-review-backlog-risk-section-top1-workbench `
  --review-workbench diagnostics\question-detector-review-wave-plan-kpai-after-backlog\wave-001-workbench `
  --backend-main backend\app\main.py `
  --xcode-build-log diagnostics\ios-build\pxj-question-detector-risk-section-top1-build.log `
  --out diagnostics\question-detector-loop-merged-reviewed-dryrun `
  --clean
```

The loop runner writes:

- `loop_report.json`: step-level command, status, and artifact summary.
- `merged_reviewed/`: optional merged approved-label package when `--merge-reviewed` is supplied.
- `dataset/`: optional freshly exported detector dataset when `--dataset` is omitted.
- `experiment/`: matrix plan, training manifests, selector output, and logs.
- `release_check/`: hard release-gate report when selector output exists, including replay, dedupe, fallback, candidate, section-sender, strategy-selection, runtime perf, backend static-check, and Xcode build evidence.
- `error_mining/`: missed/false-positive review queue from the selected eval summary.
- `error_workbench/`: static review UI for mined hard examples.
- `iteration_report/`: next-action summary for the next recursive loop, including supplied strategy/training plans, review queue manifests, review workbenches referenced by those training plans, any explicitly supplied review workbench, plus the auto-generated error workbench.

Current loop smoke result: `status=failed_eval_gate`, with experiment, release check, error mining, error workbench, and iteration report all generated successfully.

Current backend-tuned loop dry run:
`diagnostics\question-detector-loop-backend-tuned-fallback-delta-dryrun`
writes a passed orchestration report and an iteration report. Because
`--train-dry-run` does not run model prediction/selection, it intentionally does
not produce `release_check/`; use a real executed matrix with selector output
to exercise the release-check branch.

Current training-plan inheritance smoke:
`diagnostics\question-detector-loop-training-plan-inherit-smoke` was run without
explicit `--review-workbench` arguments. The loop runner still injected all 14
review workbenches from `training_readiness_plan.json`, and the resulting
iteration report shows 1030 pending boxes plus the consolidated review queue.

Current risk-section release gate smoke:
`diagnostics\question-detector-release-check-risk-section-top1-propagated-only`
uses propagated same-source fallback/candidate/cap evidence plus three section
sender reports: cap-overflow, empty-frame, and sparse weak/low-coverage
risk-frame. It reports `release_ready=false` with only the expected release
blockers left: dataset scale, missing selector output, missing Core ML artifact,
missing bundled iOS model, and missing real runtime perf evidence.

Current risk-section review backlog:
`diagnostics\question-detector-review-backlog-risk-section-top1-workbench`
contains 38 images / 80 pending boxes, selected from active batch 002, active
batch 003 quarantine, propagated fallback-delta review, and propagation
candidates. The paired iteration report at
`diagnostics\question-detector-iteration-report-risk-section-top1-backlog`
also includes the old final 1-box wave plus 12 capped observation-expanded waves,
so total pending review is 1030 boxes. It keeps the loop in
`model_selection_passed` and promotes `complete_review_workbenches`; the
remaining release blockers are model/data approval, Core ML artifact, iOS
bundling, and real runtime perf evidence.
bundle proof, not candidate-pool capacity.

Current merged-reviewed loop dry run: `status=passed`; it merged 7 images / 12 boxes with 2 deduped boxes, exported 87 dataset images with 80 empty-page negatives, and the dataset self-check returned recall/precision/F1 = 1.0. The audit still reports `model_training_ready=false`; this is a pipeline proof, not release evidence.

## Export Dataset

Generate a weak-label detector dataset from diagnostics:

```powershell
python scripts\question_detector_dataset.py `
  --diagnostics diagnostics\crop-f472\data `
  --out diagnostics\question-detector-dataset-v1 `
  --clean
```

For the tiny checked-in fixture only, use batch-level splitting when you need non-empty train/val/test outputs for pipeline sanity:

```powershell
python scripts\question_detector_dataset.py `
  --diagnostics diagnostics\crop-f472\data `
  --split-scope batch `
  --out diagnostics\question-detector-dataset-v1 `
  --clean
```

For an account SQLite DB:

```powershell
python scripts\question_detector_dataset.py `
  --sqlite data\accounts\<account-id>\pxj.sqlite3 `
  --data-dir data `
  --out diagnostics\question-detector-dataset-v1 `
  --clean
```

For reviewed historical prelabels:

```powershell
python scripts\question_detector_dataset.py `
  --reviewed-prelabels diagnostics\question-detector-review-workbench-kpai\approved_boxes.jsonl `
  --negative-images-dir diagnostics\question-detector-review-candidates-kpai\negative `
  --out diagnostics\question-detector-dataset-reviewed-v1 `
  --clean
```

On a machine with historical production data, discover all account DBs automatically:

```powershell
python scripts\question_detector_dataset.py `
  --all-account-dbs `
  --data-dir data `
  --out diagnostics\question-detector-dataset-full `
  --clean
```

Discover every diagnostics package below a root:

```powershell
python scripts\question_detector_dataset.py `
  --diagnostics-root diagnostics `
  --out diagnostics\question-detector-dataset-auto `
  --clean
```

Build grouped cross-validation folds from an exported dataset:

```powershell
python scripts\question_detector_cv.py `
  --dataset diagnostics\question-detector-dataset-full `
  --out diagnostics\question-detector-cv-v1 `
  --folds 5 `
  --clean
```

Each `fold-XX` directory is a full dataset root compatible with `question_detector_train.py`. Fold export uses `images[].split_key` by default, so frames from the same session/batch stay together. Before fold assignment, the exporter now clusters exact SHA1 duplicates and aHash near-duplicates into the same effective group, so visually similar pages are kept in the same train/val/test side. The held-out fold becomes `test`; validation groups are selected from the remaining groups with positive-label coverage first, then negative groups fill the requested validation ratio. This is the offline route for comparing model size, image size, confidence thresholds, negative-sample mix, and label-review strategy without leaking near-duplicate frames across splits.

CV still treats any residual exact cross-split image hashes as errors and, by default, treats residual cross-split aHash near-duplicates as errors too. Use `--no-cluster-near-duplicates` only to reproduce leakage diagnostics, and use `--near-duplicate-severity warning` only for tiny pipeline smoke tests where you need to inspect the export despite known near-duplicate leakage.

CV duplicate-clustering smoke on 2026-06-30:

- `diagnostics\question-detector-cv-merged-reviewed-smoke-clustered` reduced 22 raw groups to 20 effective groups, merged the 3 near-duplicate groups into one cluster, and produced 0 cross-split near-duplicate pairs in all 3 folds.
- `diagnostics\question-detector-cv-merged-reviewed-smoke-no-cluster-negative` disables clustering and correctly reproduces the old fold-01 `near_duplicate_cross_split` error.

Add explicit empty-page negative samples when you have frames that should not produce a question crop:

```powershell
python scripts\question_detector_dataset.py `
  --diagnostics-root diagnostics `
  --empty-page-manifest diagnostics\empty_pages.jsonl `
  --negative-images-dir diagnostics\empty-page-images `
  --out diagnostics\question-detector-dataset-full `
  --clean
```

`empty_pages.jsonl` rows can use `source_path` or `source_filename` plus optional `session_id`, `batch_id`, and `negative_kind`. Negative samples are intentionally opt-in: skipped/review-risk positive candidates are not treated as negatives, because false negatives would teach the detector to ignore real questions. Directory scanning also ignores crop-like names such as `qcrop*` and `crop_*`.

For SQLite exports, you can also add observation frames with no ready crop:

```powershell
python scripts\question_detector_dataset.py `
  --sqlite data\accounts\<account-id>\pxj.sqlite3 `
  --data-dir data `
  --include-sqlite-empty-pages `
  --out diagnostics\question-detector-dataset-v1 `
  --clean
```

The default `--sqlite-empty-page-mode textless` avoids QA/extract/grade images, QA-event referenced images, OCR-bearing frames, and frames with text hashes. Use `--sqlite-empty-page-mode all_observations` only when you intentionally want a broad diagnostic export and are prepared to manually review false negatives.

Outputs:

- COCO: `annotations/coco_train.json`, `coco_val.json`, `coco_test.json`
- YOLO: `images/{train,val,test}`, `labels/{train,val,test}`, `yolo_dataset.yaml`
- Create ML: `annotations/createml_train.json`, `createml_val.json`, `createml_test.json`
- Image manifest: `annotations/image_manifest.jsonl`, one row per exported source image, including zero-annotation negatives
- Review queue: `annotations/review.jsonl`
- Visual QA: `preview_contact_sheet.jpg`
- Dataset audit: `audit.json`
- Summary: `metadata.json`

Default behavior is conservative: low-confidence, near-full-page, missing-key, duplicate, and bad-shape labels are withheld from train outputs and written to review. Use `--include-review-labels` only for experiments.

The default split scope is `session`, which is the right setting for model selection because frames from the same observation session are often near-duplicates. For a tiny diagnostics fixture where the goal is only to test the export/eval pipeline, pass `--split-scope batch`.

Short/figure-heavy questions can enter production as `layout:` weak keys when OCR text is too short for a strong fingerprint. The exporter flags these as `weak_question_key` and routes them to review by default, because they are valuable recall evidence but should be checked before training a detector.

`audit.json` is a hard gate, not decoration:

- `model_training_ready` must be true before selecting a detector model;
- `split_group_overlaps` and `exact_cross_split_images` must be empty;
- `near_duplicate_cross_split_pairs` default to an error and must be manually reviewed or eliminated with a coarser split scope; use `--near-duplicate-severity warning` only for tiny pipeline smoke tests;
- negative images should be present before final false-positive tuning; set `--min-production-negative-images` to make this a hard audit minimum;
- `risky_label_ratio` must stay below `--max-risky-label-ratio`, including labels admitted by `--include-review-labels`;
- default production minimums are 300 source images, 1000 annotations, 50 split groups, and review ratio <= 0.15.

Current sample export:

- input rows: 20
- clean training annotations: 17
- review annotations: 3
- source images: 8
- default session split audit: not training-ready because the fixture has only one session and no real held-out split
- batch split audit: pipeline-ready only; cross-split near-duplicates now default to errors unless `--near-duplicate-severity warning` is explicitly used

Current mixed historical smoke export:

- command input: `diagnostics/crop-f472/data` plus textless negatives from the KPAI account DB
- source images: 120
- positive images: 8
- clean annotations: 17
- review annotations: 3
- safe textless negative images: 112
- split groups: 65
- status: `pilot_eval_ready=true`, `model_training_ready=false`
- reason not training-ready: too few positive labels for model selection; dataset/CV audit also reports near-duplicate cross-split leakage that must be removed or reviewed before product model selection

## Label Semantics

The default strategy is `v3_safe`.

It exports safe practical question blocks, not historical thin strips:

- old iOS crop rows are expanded through the same V3 geometry used by production;
- server-generated rect crops use their stored final `crop_rect`;
- suspicious labels are routed to review.

This trains the detector to recall complete question blocks. A slightly larger box is acceptable; a missing question is not.

## Model Training

Train a small single-class object detector on the exported YOLO or COCO dataset. Keep the input size modest for iPhone/iPad, for example 640-960 px depending on latency.

Target metrics before app integration:

- recall at IoU 0.5: high priority;
- missed-question rate: primary product metric;
- median on-device inference: below the current OCR segmentation path;
- false split rate: low enough that OCR fallback/dedupe can recover.

Suggested iterative loop:

1. Export weak labels.
2. Review `preview_contact_sheet.jpg` and `annotations/review.jsonl`.
3. Correct the worst labels in an annotation tool.
4. Confirm `audit.json` has `model_training_ready=true`.
5. Train the detector.
6. Evaluate on held-out session/book splits, not random frames.
7. Convert the best small model to Core ML.
8. Add it to the app as `QuestionRegionDetector.mlmodel`.
9. Compare live speed and missed-question rate against OCR-only V3.

The training wrapper is:

```powershell
python scripts\question_detector_train.py `
  --dataset diagnostics\question-detector-dataset-full `
  --out diagnostics\question-detector-train-v1 `
  --model yolo11n.pt `
  --epochs 80 `
  --imgsz 768 `
  --export-coreml `
  --predict-test `
  --eval-score-thresholds 0.05,0.10,0.20,0.25,0.35,0.50 `
  --clean
```

The wrapper enforces `audit.json` by default. It refuses audit-unready datasets so the tiny fixture cannot accidentally become a product model. For a pipeline smoke test only:

```powershell
python scripts\question_detector_train.py `
  --dataset diagnostics\question-detector-dataset-auto `
  --out diagnostics\question-detector-train-smoke `
  --dry-run `
  --allow-unready `
  --clean
```

Outputs:

- `train_manifest.json`: audit readiness, training config, best checkpoint, Core ML artifact, and eval summary.
- `runs/<name>/weights/best.pt`: best detector checkpoint.
- `coreml/QuestionRegionDetector.mlpackage` or `.mlmodel`: artifact to add to the Xcode target and compile into `QuestionRegionDetector.mlmodelc`.
- `eval/summary.json`: held-out detector metrics through `question_detector_eval.py`.

When `--predict-test` is set, the wrapper now treats detector eval as a real release gate. Defaults are primary IoU recall >= 0.95, precision >= 0.90, and false positives per image <= 0.05. If the gate fails, `train_manifest.json` is written with `status="failed_eval_gate"` and the script exits nonzero. Use `--allow-failed-eval` only for experiments where you intentionally want to keep a failed artifact for inspection.

For K-fold selection, run the wrapper once per fold:

```powershell
foreach ($fold in Get-ChildItem diagnostics\question-detector-cv-v1 -Directory -Filter "fold-*") {
  python scripts\question_detector_train.py `
    --dataset $fold.FullName `
    --out ("diagnostics\question-detector-train-" + $fold.Name) `
    --model yolo11n.pt `
    --epochs 80 `
    --imgsz 768 `
    --predict-test `
    --eval-score-thresholds 0.05,0.10,0.20,0.25,0.35,0.50 `
    --clean
}
```

For multi-model comparisons, use the matrix orchestrator instead of manually copying fold commands:

```powershell
python scripts\question_detector_experiment_matrix.py `
  --cv-root diagnostics\question-detector-cv-v1 `
  --model yolo11n.pt,yolo11s.pt `
  --imgsz 640,768 `
  --epochs 80 `
  --batch -1 `
  --eval-score-thresholds 0.05,0.10,0.20,0.25,0.35,0.50 `
  --execute `
  --out diagnostics\question-detector-experiment-v1 `
  --clean
```

The orchestrator writes:

- `experiment_plan.json`: all fold/model/imgsz/seed/epoch jobs, dataset readiness summaries, and exact commands;
- `run_experiment_matrix.ps1`: a reproducible PowerShell runner;
- `experiment_report.json`: per-run process status, manifest status, eval-gate status, log paths, and selector output;
- `runs/<run-id>/train_manifest.json`: each delegated training/eval manifest;
- `selection/selection_summary.json`: produced automatically after completed prediction/eval runs.

Use `--train-dry-run --execute --allow-unready` only to validate the command matrix on smoke data. A dry-run matrix on the reviewed smoke CV fixture successfully executed 12 planned jobs: 3 folds x 2 model sizes x 2 image sizes.

Actual smoke training also works end-to-end, but the tiny data fails the detector gate, as it should:

```powershell
python scripts\question_detector_experiment_matrix.py `
  --dataset diagnostics\question-detector-reviewed-approved-smoke-cv\fold-00 `
  --model yolo11n.pt `
  --imgsz 320 `
  --epochs 1 `
  --batch 4 `
  --workers 0 `
  --device cpu `
  --allow-unready `
  --allow-failed-eval `
  --execute `
  --out diagnostics\question-detector-experiment-actual-smoke `
  --clean
```

Current result: `experiment_report.json.status=failed_eval_gate`. The 1-epoch smoke model produced 0 predictions against 6 held-out boxes, so recall/precision were 0.0. This is useful evidence that the pipeline and gate are wired correctly; it is not useful model quality.

Then select the best model/threshold combination:

```powershell
python scripts\question_detector_select.py `
  diagnostics\question-detector-train-fold-* `
  --out diagnostics\question-detector-selection-v1
```

The selector reads each `train_manifest.json`, aggregates `eval.score_sweep` across compatible folds, and ranks by: all folds passing the eval gate, worst-fold recall, mean recall, false positives per image, predicted box count, precision, F1, duplicate predictions, and finally the highest threshold that preserves those metrics. Compatibility is intentionally strict: the candidate key includes dataset family, model, train name, image size, epochs, seed, batch, optimizer, prediction confidence, primary IoU threshold, and detector score. The key excludes fold index, so real CV folds aggregate, but duplicate copies of the same fold keep `fold_count` at the unique-fold count and set `duplicate_fold_count`. It writes `selection_summary.json` with `ios_recommendation.question_region_detector_fast_min_confidence` and `question_region_detector_accurate_min_confidence`.

Selector smoke on 2026-06-30:

- `diagnostics\question-detector-selection-actual-keyed-smoke` reads CV metadata and reports `dataset_family`, `epochs`, `seed`, and `fold_ids=["fold:0"]`.
- `diagnostics\question-detector-selector-duplicate-fold-smoke` proves duplicate copies of one fold produce `run_count=2`, `fold_count=1`, and `duplicate_fold_count=1`, so a repeated fold cannot satisfy the release fold-count gate.

When `--eval-score-thresholds` includes a value below `--conf`, the training wrapper lowers the test prediction export confidence to the minimum sweep value and records both requested and effective confidence in `train_manifest.json`. This keeps low-threshold score sweeps honest: the evaluator can only score boxes that were actually exported.

## Release Readiness Gate

Before a detector model is bundled into iOS, run the hard release gate:

```powershell
python scripts\question_detector_release_check.py `
  --dataset diagnostics\question-detector-dataset-v1 `
  --selection diagnostics\question-detector-selection-v1 `
  --replay diagnostics\question-observation-replay-v1 `
  --dedupe-tune diagnostics\question-observation-dedupe-tune-v1 `
  --fallback-tune diagnostics\question-observation-fallback-tune-v1 `
  --candidate-eval diagnostics\question-observation-candidate-eval-v1 `
  --candidate-cap-sweep diagnostics\question-observation-candidate-cap-sweep-v1 `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-v1 `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-empty-frame-v1 `
  --runtime-perf diagnostics\question-observation-runtime-perf-v1 `
  --ios-root ios\PXJ\App `
  --backend-main backend\app\main.py `
  --xcode-build-log diagnostics\ios-build-with-detector.log `
  --out diagnostics\question-detector-release-check-v1 `
  --clean
```

The checker writes:

- `release_check.json`: machine-readable gate result, hard failures, warnings, and all raw check values.
- `release_check.md`: reviewer-friendly summary.

This checker exits nonzero when `release_ready=false`; add `--allow-failed-gate` only for exploratory smoke runs where the failed report is the intended artifact.

`release_ready=true` requires:

- every supplied dataset `audit.json` reports `model_training_ready=true`;
- the selected model aggregates at least 3 folds;
- all selected folds pass the detector eval gate;
- worst-fold recall >= 0.95, mean precision >= 0.90, and max false positives per image <= 0.05;
- a Core ML `QuestionRegionDetector` artifact exists;
- the iOS app sources include `QuestionRegionDetector.mlmodel`, `.mlpackage`, or compiled `.mlmodelc`;
- `QuestionSegmenter` still loads `QuestionRegionDetector.mlmodelc` from the app bundle;
- Swift detector confidence constants match `selection_summary.json` iOS recommendations;
- replay evidence confirms rect-only upload overhead stays <= 1.005x full-frame bytes, deduped crop VLM pixels stay <= 0.75x full-frame pixels, and measured local layout median stays <= 80 ms;
- runtime perf evidence from simulator/device logs or batch responses is supplied with non-fixture `evidence_kind`, covers at least 5 observation batch records, keeps local `analysis_duration_ms` p95 <= 300 ms, backend `question_crop_server_duration_ms` p95 <= 250 ms when present, fresh OCR frame ratio <= 0.75, and `attached_crop_file_count=0`;
- dedupe-tune evidence is supplied, covers at least 10 represented reviewed question keys, keeps represented-GT recall >= 0.999, loses 0 GT keys, has 0 known false merges, has 0 reviewed-to-unverified suppressions, and includes the current iOS Swift text-guarded weak-key baseline;
- fallback-tune evidence is supplied, covers at least 10 represented reviewed question keys, proves the current backend fallback policy keeps represented-GT recall >= 0.999, keeps total crop+fallback VLM pixels <= 1.05x full-frame pixels, sends full-frame fallback for <= 50% of crop-bearing images, includes a union-area shadow of the current policy, and reports a legacy-to-current fallback delta with 0 represented GT misses among frames where full-frame fallback was removed;
- candidate-cap sweep evidence is supplied on the same reviewed fallback inputs, proves the production iOS per-frame cap keeps crop-only and fallback-protected represented-GT recall >= 0.999, and keeps total crop+fallback VLM pixels <= 1.05x full-frame pixels. A lower cap can only replace cap 12 after broader reviewed multi-layout evidence matches recall without increasing fallback cost; smoke-only cap 3 is diagnostic, and cap 1 is unsafe;
- candidate-eval evidence is supplied, covers at least 10 duplicate-clustered represented reviewed question keys, proves local crop candidates cover represented-GT recall >= 0.999, keeps crop VLM pixels <= 0.75x full-frame pixels, and has 0 overcrop candidates above the configured overcrop ratio;
- reviewed replay evidence is same-source across fallback tune, candidate eval, and candidate-cap sweep: the complete replay list, image counts, candidate counts, full-frame resized pixel totals, reviewed GT manifest list, and represented reviewed GT count must match, so release checks cannot be assembled from unrelated easy reports;
- dense section-sender evidence is supplied for both cap-overflow and no-candidate dense frames, each covers at least 200 dense expected boxes, proves combined crop recall and fallback-protected recall >= 0.999, keeps total VLM pixels <= 0.75x full-frame pixels, and sends fallback for <= 5% of dense images. Cap-overflow evidence must mirror the live top-2 companion policy, not the cheaper section-only diagnostic and not the too-expensive section+top12 diagnostic;
- `ContentView.swift` still contains the weak-layout text guard, so stale dedupe-tune reports cannot mask a regression in the live iOS duplicate-suppression code;
- `ContentView.swift` still contains the dense section-crop sender, so cap-risk frames can add rect-only section sources with `crop_kind=section`, section telemetry, and top-2 per-question companion crops instead of relying only on full-frame fallback;
- `ContentView.swift` still keeps no-candidate section signatures separate from per-question crop signatures, so repeated `ios_page_empty_v1` sources can be skipped without suppressing later real question boxes;
- `ContentView.swift` still limits Vision resize and upload JPEG generation by real `CGImage` pixels with renderer scale 1, so iPhone/iPad display scale cannot silently inflate analysis cost or upload bytes;
- `QuestionSegmenter.swift` still gates detector fast-path selection on OCR-matched detector regions with enough page coverage, falls back to full-page OCR for textless or incomplete detector boxes, and merges Core ML regions with unmatched OCR regions instead of returning the first detector hit early;
- `backend/app/main.py` still preserves batched crop source binding: the VLM prompt requests `input_index`, parser accepts `input_index`/`inputIndex` plus source-index aliases, crop sources receive 1-based `input_index`, matching prefers explicit `input_index` before text-similarity fallback, and persisted metadata includes `source_input_index`;
- `backend/app/main.py` still keeps section crops source-safe: section crop kind is parsed, `section:`/`section_crop` identities are weak for source dedupe, section telemetry is persisted, extracted-question payloads carry `source_crop_kind`/`source_section_key`, and the crop-batch prompt allows multiple questions per section input;
- a post-bundle Xcode build log is supplied and shows a successful build.

Build runtime perf evidence from exported simulator/device JSON, JSONL, or logs:

```powershell
python scripts\question_observation_runtime_perf_check.py `
  --input diagnostics\ios-runtime-observation-logs `
  --out diagnostics\question-observation-runtime-perf-v1 `
  --clean
```

The fixture smoke at `diagnostics\question-observation-runtime-perf-fixture-check`
proves the parser/release-gate integration only; it is not release evidence for
a real model candidate. The release gate now rejects `evidence_kind=fixture`
unless `--allow-runtime-fixture` is passed for an explicit smoke run.

After a real selected model passes evaluation and has a Core ML artifact, promote it into iOS with:

```powershell
python scripts\question_detector_ios_promote.py `
  --selection diagnostics\question-detector-selection-v1 `
  --train-manifest diagnostics\question-detector-train-fold-* `
  --ios-root ios\PXJ\App `
  --out diagnostics\question-detector-ios-promote-v1 `
  --apply `
  --clean
```

The promotion helper copies `QuestionRegionDetector.mlmodel`, `.mlpackage`, or `.mlmodelc` into the app source directory and syncs:

- `detectorFastMinConfidence`
- `detectorAccurateMinConfidence`

It refuses to apply unless the selected candidate has enough folds, all folds pass the eval gate, a Core ML artifact exists, and `QuestionSegmenter` still loads `QuestionRegionDetector.mlmodelc` from the app bundle. After promotion, run the Mac/Xcode build and rerun `question_detector_release_check.py` with `--xcode-build-log`.

Current bootstrap promotion dry-run:

```powershell
python scripts\question_detector_ios_promote.py `
  --selection diagnostics\question-detector-experiment-bootstrap-smoke\selection `
  --train-manifest diagnostics\question-detector-experiment-bootstrap-smoke\runs\question-detector-bootstrap-kpai-smoke__yolo11n__img416__seed42__ep5 `
  --ios-root ios\PXJ\App `
  --out diagnostics\question-detector-ios-promote-bootstrap-smoke `
  --clean
```

Current result: `can_apply=false`, blocked by too few folds, failed eval gate, and missing Core ML artifact. No iOS files are changed.

Current repository smoke check:

```powershell
python scripts\question_detector_release_check.py `
  --dataset diagnostics\question-detector-active-batch-kpai-001-approved-smoke `
  --selection diagnostics\question-detector-selection-smoke `
  --replay diagnostics\question-observation-replay-kpai-layout-measured `
  --dedupe-tune diagnostics\question-observation-dedupe-tune-kpai `
  --fallback-tune diagnostics\question-observation-fallback-tune-cap-aware-kpai `
  --candidate-eval diagnostics\question-observation-candidate-eval-kpai `
  --candidate-cap-sweep diagnostics\question-observation-candidate-cap-sweep-kpai `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-synthetic-dense-oracle `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-synthetic-dense-layout-empty-section `
  --ios-root ios\PXJ\App `
  --backend-main backend\app\main.py `
  --out diagnostics\question-detector-release-check-current `
  --clean
```

Current result: `release_ready=false`, with 7 hard failures. The useful part is that it passes rect-only replay, dedupe-tune, fallback-tune, Swift weak-key guard, Swift pixel-bound resize, Swift detector textless fallback, and backend source-binding gates, but blocks release because the data is still smoke-scale, the selected candidate has only 1 fold, no Core ML artifact exists, no detector is bundled in iOS, and Swift thresholds do not match the smoke selection. This is the intended behavior: smoke metrics must never promote a detector into the app.

The actual 1-epoch YOLO smoke run is also blocked by the same release gate:

```powershell
python scripts\question_detector_release_check.py `
  --dataset diagnostics\question-detector-reviewed-approved-smoke-cv\fold-00 `
  --selection diagnostics\question-detector-experiment-actual-smoke\selection `
  --train-manifest diagnostics\question-detector-experiment-actual-smoke\runs\fold-00__yolo11n__img320__seed42__ep1 `
  --replay diagnostics\question-observation-replay-kpai-layout-measured `
  --dedupe-tune diagnostics\question-observation-dedupe-tune-kpai `
  --fallback-tune diagnostics\question-observation-fallback-tune-cap-aware-kpai `
  --candidate-eval diagnostics\question-observation-candidate-eval-kpai `
  --candidate-cap-sweep diagnostics\question-observation-candidate-cap-sweep-kpai `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-synthetic-dense-oracle `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-synthetic-dense-layout-empty-section `
  --ios-root ios\PXJ\App `
  --backend-main backend\app\main.py `
  --out diagnostics\question-detector-release-check-actual-smoke `
  --clean
```

Current result: `release_ready=false`, with 11 hard failures, including dataset readiness, single-fold selection, failed recall/precision, no Core ML artifact, no iOS-bundled detector, and unsynced Swift thresholds.

The bootstrap pseudo-label model is also blocked:

```powershell
python scripts\question_detector_release_check.py `
  --dataset diagnostics\question-detector-bootstrap-kpai-smoke `
  --selection diagnostics\question-detector-experiment-bootstrap-smoke\selection `
  --train-manifest diagnostics\question-detector-experiment-bootstrap-smoke\runs\question-detector-bootstrap-kpai-smoke__yolo11n__img416__seed42__ep5 `
  --replay diagnostics\question-observation-replay-kpai-layout-measured `
  --dedupe-tune diagnostics\question-observation-dedupe-tune-kpai `
  --fallback-tune diagnostics\question-observation-fallback-tune-cap-aware-kpai `
  --candidate-eval diagnostics\question-observation-candidate-eval-kpai `
  --candidate-cap-sweep diagnostics\question-observation-candidate-cap-sweep-kpai `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-synthetic-dense-oracle `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-synthetic-dense-layout-empty-section `
  --ios-root ios\PXJ\App `
  --backend-main backend\app\main.py `
  --out diagnostics\question-detector-release-check-bootstrap-dedupe-smoke `
  --clean
```

Current result: `release_ready=false`, with 11 hard failures. All dedupe checks pass: the release gate sees the current Swift weak-layout text guard in `ContentView.swift`, recommendation recall is 1.0, lost GT keys are 0, false merges are 0, reviewed-to-unverified suppressions are 0, rect-only overhead is 1.002001x, and the Swift text-guarded weak-key baseline also has represented-GT recall 1.0. The remaining blockers confirm that pseudo-label exploration cannot bypass reviewed-data readiness, cross-fold success, Core ML export, iOS bundling, or Swift threshold synchronization.

A negative smoke without `--dedupe-tune` writes `diagnostics\question-detector-release-check-missing-dedupe-smoke` and adds one hard failure in the `dedupe` category. This is intentional: release candidates must prove duplicate suppression is not hiding missed questions.

A negative smoke with a fixture `ContentView.swift` missing the weak-key text guard writes `diagnostics\question-detector-release-check-missing-swift-dedupe-smoke` and adds one hard failure: `Swift weak-layout text guard exists`.

A backend source-binding static smoke writes `diagnostics\question-detector-release-check-source-binding-static-smoke`. The current repo still reports `release_ready=false` because reviewed data, model, artifact, iOS bundle, and build blockers remain, but the backend hard checks `question crop source input_index binding exists` and `section crop sources stay source-safe` pass.

A negative backend fixture missing the crop-batch prompt request writes `diagnostics\question-detector-release-check-missing-source-binding-smoke` and adds one hard failure: `question crop source input_index binding exists`. This keeps VLM output reordering from silently assigning extracted questions to the wrong source crop.

The backend static gate also checks that weak `layout:` crop keys and their
fingerprints do not pre-dedupe extraction sources before VLM extraction. Those
weak keys remain useful telemetry, but different short/formula/figure-heavy
questions must not be suppressed until real crop-hash or source evidence exists.
The same gate treats `section:` / `section_crop` source identities as weak and
verifies that section metadata survives into extracted-question payloads, so
multi-question section crops cannot be mistaken for reviewed one-question boxes.

A fallback-policy smoke writes `diagnostics\question-detector-release-check-backend-tuned-fallback-delta`. The current repo still reports `release_ready=false` for the expected data/model/artifact/iOS blockers, but backend-tuned fallback checks pass on the refreshed KPAI replay: 11 represented GT keys, backend-current represented-GT recall 1.0, total crop+fallback VLM pixels 0.784856x full-frame pixels, fallback image ratio 0.170732, a union-area shadow, and 0 represented GT misses among legacy fallback removals. The report keeps a warning while fallback-delta no-GT review queues remain open.

A negative smoke without `--fallback-tune` writes `diagnostics\question-detector-release-check-missing-fallback-tune-smoke` and adds one hard failure: `fallback policy tuning supplied`.

A cap-aware smoke writes `diagnostics\question-detector-release-check-cap-aware-smoke`. It keeps the expected release blockers from smoke-scale data/model/artifact state, while the newer hard checks pass: `Swift ranked candidate cap telemetry exists`, `Swift Vision/JPEG resizing is pixel-bound`, `Swift detector-first textless fallback gate exists`, `question crop candidate cap fallback is risk-aware`, `backend current union-area shadow exists`, and the candidate-cap sweep checks for current-cap recall/cost.

A clustered-dataset cap-aware smoke writes `diagnostics\question-detector-release-check-cap-aware-clustered-smoke`. It drops the dataset `no audit error` hard failure, leaving 11 expected blockers: smoke-scale dataset, failed/one-fold selection, no completed Core ML artifact, no bundled iOS model, and intentionally unmatched smoke thresholds.

A propagated-exact smoke writes `diagnostics\question-detector-release-check-propagated-exact-smoke`. It keeps the same 11 expected blockers while using the 89-image / 16-annotation propagated dataset, proving that duplicate-label growth does not reintroduce dataset leakage, bypass model/artifact/iOS gates, or regress the iOS pixel/textless-fallback guards, backend weak-key extraction-source guard, or candidate-cap sweep gates.

The current P0 iOS guard check at `diagnostics\question-detector-release-check-risk-section-top1-propagated-only` uses propagated same-source smoke inputs, dense cap-overflow top-2 companion section-sender evidence at `diagnostics\question-observation-section-sender-eval-synthetic-dense-oracle`, no-candidate section-sender evidence at `diagnostics\question-observation-section-sender-eval-synthetic-dense-layout-empty-section`, sparse weak/low-coverage risk-frame top-1 evidence at `diagnostics\question-observation-section-sender-eval-propagated-risk-section-top1-ios-signals`, strategy-selection evidence at `diagnostics\question-observation-strategy-selection-risk-section-top1`, and a Mac generic iOS build log at `diagnostics\ios-build\pxj-question-detector-risk-section-top1-build.log`. It reports `release_ready=false` with only the expected release blockers left: dataset scale, missing selector output, missing Core ML artifact, missing bundled iOS model, and missing real runtime perf evidence. These hard checks pass: pixel-bound Vision/JPEG resizing, detector OCR-merge fallback, ranked candidate-cap telemetry, dense/empty/risk section-crop sender with the correct companion anchors, empty-frame section cross-batch dedupe, backend weak-key source-dedupe guard, backend section source-safe handling, same-source replay and reviewed-GT-manifest evidence across fallback/candidate/cap reports, candidate-cap sweep recall/cost, section-sender recall/cost, strategy-selection reviewed-live and propagated-reviewed Pareto gates, synthetic-stress separation, and Xcode build success.

The current fallback tune at `diagnostics\question-observation-fallback-tune-kpai-layout-v5-dense-gated-backend-tuned` now reports both summed-area and union-area current-policy baselines. On the 41-image KPAI fixture, they match exactly: 7 fallback images, represented-GT recall 1.0, and 0.784856x full-frame VLM pixels. The broader propagated replay at `diagnostics\question-observation-fallback-tune-reviewed-propagated-layout-v5-gated-backend-tuned` keeps 16/16 represented-GT recall with 34/89 fallback images and 0.819493x full-frame VLM pixels, so the stricter smoke recommendation has now been promoted into `backend_current`.

The current iteration report is `diagnostics\question-detector-iteration-report-backend-tuned-fallback-delta`. It classifies the loop as `review_closure`: 92 boxes are still pending across four workbenches, including 43 fallback-delta boxes that directly harden the backend-tuned fallback policy. The next recursive loop should close these workbenches, export approved boxes, merge them into the next reviewed dataset, rebuild grouped CV folds, and rerun model selection before any Core ML promotion.

A candidate-eval smoke writes `diagnostics\question-detector-release-check-candidate-eval-smoke`. The current repo still reports `release_ready=false` for the expected data/model/artifact/iOS blockers, but all candidate-eval checks pass: 11 represented GT keys, represented-GT recall 1.0, crop VLM pixels 0.625661x full-frame pixels, and 0 overcrop candidates.

A negative smoke without `--candidate-eval` writes `diagnostics\question-detector-release-check-missing-candidate-eval-smoke` and adds one hard failure: `observation candidate eval supplied`.

## Evaluate Detector Outputs

Every detector candidate must pass the same offline gate before it is considered for iOS.

Sanity check the evaluation pipeline with the exported weak labels as predictions:

```powershell
python scripts\question_detector_eval.py `
  --ground-truth diagnostics\question-detector-dataset-v1\annotations\coco_all.json `
  --predictions-jsonl diagnostics\question-detector-dataset-v1\annotations\manifest.jsonl `
  --image-root diagnostics\question-detector-dataset-v1 `
  --out diagnostics\question-detector-eval-v1
```

Expected sanity result on the current sample:

- `gt_count`: 17
- `pred_count`: 17
- IoU 0.50 recall/precision/F1: 1.0
- IoU 0.75 recall/precision/F1: 1.0

For a real model, export predictions as either:

- COCO detection JSON: `image_id`, `bbox`, `score`
- JSONL rows compatible with `annotations/manifest.jsonl`: `image`, `bbox_px`, optional `score`
- YOLO label files: `class cx cy w h [score]`

Then run:

```powershell
python scripts\question_detector_eval.py `
  --ground-truth diagnostics\question-detector-dataset-v1\annotations\coco_test.json `
  --predictions path\to\model_predictions.json `
  --image-root diagnostics\question-detector-dataset-v1 `
  --out diagnostics\question-detector-eval-model
```

For YOLO output directories:

```powershell
python scripts\question_detector_eval.py `
  --ground-truth diagnostics\question-detector-dataset-v1\annotations\coco_test.json `
  --predictions-yolo-dir path\to\yolo\labels `
  --image-root diagnostics\question-detector-dataset-v1 `
  --score-thresholds 0.05,0.10,0.20,0.25,0.35,0.50 `
  --out diagnostics\question-detector-eval-yolo
```

The evaluator writes:

- `summary.json`
- `errors_contact_sheet.jpg` when there are missed or false-positive detections

`summary.json.score_sweep` records precision, recall, F1, missed-question rate, false positives per image, duplicate predictions, and gate status at each confidence threshold. Use this to choose the iOS detector threshold. On the current weak-label smoke data, raising the threshold to `0.50` drops recall to `0.8235`, so recall-first thresholds must be selected from actual held-out data rather than intuition. Also run `question_detector_postprocess_sweep.py` on held-out predictions before promotion; if score/shape/NMS/top-k filtering cannot meet recall >= 0.95, precision >= 0.90, and FP/image <= 0.05, the right next step is more reviewed data, not an iOS threshold tweak.

After selection, update the iOS constants in `QuestionSegmenter`:

- `detectorFastMinConfidence`
- `detectorAccurateMinConfidence`
- `detectorNMSOverlapThreshold` only if eval explicitly justifies changing duplicate/overlap behavior

Use this as the first hard gate:

- IoU 0.50 recall >= 0.95
- IoU 0.50 precision >= 0.90
- false positives per image <= 0.05, especially on empty-page negatives
- IoU 0.75 recall/precision should stay high enough to prove boxes are tight, not merely touching a question
- review the summary strata by `quality_flags`, area bucket, origin, confidence bucket, and `negative_kind`
- no repeated missed questions in the same workbook/session cluster
- latency on device must beat or materially reduce the OCR-only path

## Evaluate Product Question Sets

Detector IoU is only the first gate. A detector candidate must also improve the final question set that the user sees.

Compare a reference question set against a production or replayed extraction:

```powershell
python scripts\question_set_eval.py `
  --reference diagnostics\6dca70735ad448b19828e729202b1f87\diag_extract.jsonl `
  --reference-type product_qset `
  --reference-row first `
  --candidate diagnostics\6dca70735ad448b19828e729202b1f87\prod_reextract_after_fix.jsonl `
  --candidate-row best `
  --out diagnostics\question-set-eval-6dca-reextract `
  --clean
```

The evaluator writes:

- `summary.json`: precision, recall, F1, coverage recall, count delta, duplicate clusters, merge suspects, number/order risk, content placeholders, and source coverage.
- `matches.jsonl`: one-to-one matched questions.
- `missed.jsonl`: reference questions not matched one-to-one, with best candidate and whether any candidate covers them.
- `extra.jsonl`: candidate questions without a reference match.
- `merge_suspects.jsonl`: candidate questions that appear to cover multiple reference questions.
- `reference_duplicates.jsonl` and `candidate_duplicates.jsonl`: within-set duplicate clusters.

Current 6dca baseline:

- reference count: 18
- replayed production count: 9
- one-to-one precision / recall / F1: 1.0 / 0.5 / 0.6667
- coverage recall: 0.6667
- merge-suspect candidates: 4
- candidate source coverage: 1.0

Use this as the product gate for OCR-only V3 vs Core ML V4:

- question-set recall and coverage recall must improve or stay flat;
- candidate duplicate clusters and merge suspects must not increase;
- source coverage must stay high, with low fallback pressure;
- missing numbers, duplicate numbers, and placeholder stems must not increase;
- the same session replay must also reduce `analysis_duration_ms` or `ocr_frame_count`.

## iOS Integration

`QuestionSegmenter` already has an optional model hook:

- it looks for `QuestionRegionDetector.mlmodelc` in the app bundle;
- if the model is missing, loading fails, confidence is low, or output is empty, it falls back to the current OCR heuristic;
- when detector boxes exist, the app runs OCR only inside those boxes to attach text when available; boxes without OCR text still flow as weak `layout:` candidates for rect-only upload and backend/VLM reading;
- detector results are NMS-filtered, sorted top-to-bottom/left-to-right, and returned as `[QuestionRegion]`;
- `segmentForPreviewOverlay` keeps mapping corrected-image boxes back to source-image normalized coordinates.

This means the first model can be shipped behind the existing interface without changing `ContentView`. Current repository status: the hook exists, but no `QuestionRegionDetector.mlmodelc`/`.mlmodel`/`.mlpackage` artifact is included in the iOS target yet, so telemetry should show OCR fallback until a trained model is bundled.

## Live Speed Comparison

Every observation batch now carries crop telemetry from iOS to the backend:

- `question_crop_client_metrics.analysis_duration_ms`: local segmentation and manifest-building time.
- `question_crop_client_metrics.cached_frame_count`: frames that reused live-overlay candidates.
- `question_crop_client_metrics.ocr_frame_count`: frames that still required a fresh segmentation pass during upload.
- `question_crop_client_metrics.candidate_count`: raw local question candidates before upload dedupe.
- `question_crop_client_metrics.selected_candidate_count`: candidates selected after ranking by non-duplicate status, strong OCR key, stable crop quality, confidence, and area before the per-frame cap is applied.
- `question_crop_client_metrics.manifest_count`: non-duplicate rects sent to the backend.
- `question_crop_client_metrics.duplicate_count`: local duplicate question candidates skipped.
- `question_crop_client_metrics.weak_layout_candidate_count`: candidates that only had a weak layout key instead of a strong OCR key.
- `question_crop_client_metrics.low_confidence_candidate_count`: local candidates below the confidence watch threshold.
- `question_crop_client_metrics.limited_candidate_count`: candidates dropped by the per-frame upload cap.
- `question_crop_client_metrics.limited_unique_candidate_count`: skipped candidates that were not already known duplicates at cap time.
- `question_crop_client_metrics.limited_strong_candidate_count`: skipped unique candidates with strong OCR keys; these still force full-frame fallback.
- `question_crop_client_metrics.limited_confident_candidate_count`: skipped unique candidates at or above the high-confidence watch threshold; these still force full-frame fallback.
- `question_crop_client_metrics.limited_max_confidence`: maximum confidence among skipped unique candidates.
- `question_crop_client_metrics.limited_frame_count`: frames whose candidate count exceeded the upload cap.
- `question_crop_client_metrics.limited_source_sequences`: source sequence indexes affected by per-frame candidate limiting.
- `question_crop_client_metrics.crop_risk_reason_counts`: why local crops were expanded or considered risky.
- `question_crop_client_metrics.dedupe_drop_rate`: processed candidates suppressed by local duplicate detection.
- `question_crop_client_metrics.selected_segmenter_counts`: fresh frames selected by `ocr_v3`, `coreml_v4`, or no detector result.
- `question_crop_client_metrics.text_recognition_scope_counts`: fresh frames that used `detector_boxes`, `full_page`, or no OCR.
- `question_crop_client_metrics.coreml_model_loaded_frame_count`: fresh frames where the bundled model was available.
- `question_crop_client_metrics.fallback_to_ocr_frame_count`: fresh frames where Core ML did not produce usable boxes.
- `question_crop_client_metrics.paired_mean_iou`: greedy paired IoU between OCR-layout boxes and Core ML boxes on the same frame.
- `question_crop_client_metrics.unmatched_ocr_v3_count` / `unmatched_coreml_v4_count`: boxes not covered by the other local method.
- `question_crop_server_duration_ms`: backend time spent generating canonical rect crops from source keyframes.

Compare OCR-only V3 and Core ML V4 by replaying or capturing the same session clusters, then inspect these fields in the batch response/logs. A model candidate is only better if it reduces `analysis_duration_ms` and `ocr_frame_count` without increasing missed questions, bad crops, duplicate uploads, or full-frame fallbacks.

Important naming note: `ocr_frame_count` means "frames that did not use cached live candidates and therefore ran a fresh segmentation pass during upload." Use `selected_segmenter_counts`, `coreml_model_loaded_frame_count`, `ocr_v3_duration_ms`, `coreml_v4_duration_ms`, paired IoU, and unmatched candidate counts to separate OCR layout from Core ML detector behavior.

Extraction fallback note: per-crop client telemetry now preserves frame-level candidate limiting. Older clients still trigger `limited_candidate_frame` whenever a source frame exceeds the per-frame cap. Ranked-cap clients only trigger that fallback when skipped unique candidates include strong OCR or high-confidence evidence, so duplicate/weak overflow no longer forces a full-frame VLM pass.

Before a bundled model exists, use offline replay to quantify the network/VLM shape on historical images:

```powershell
python scripts\question_observation_replay.py `
  --prelabel-root diagnostics\question-detector-prelabel-kpai `
  --measure-layout `
  --use-measured-layout-boxes `
  --out diagnostics\question-observation-replay-kpai-layout-measured `
  --clean
```

For a standard detector dataset:

```powershell
python scripts\question_observation_replay.py `
  --dataset-root diagnostics\question-detector-dataset-reviewed-v1 `
  --out diagnostics\question-observation-replay-reviewed-v1 `
  --clean
```

Replay outputs:

- `summary.json`: image/box counts, dedupe counts, upload byte ratios, VLM pixel proxy, JPEG timing, and optional layout latency;
- `images.jsonl`: every replayed image, including pages with zero crop candidates, so candidate-eval, cap-sweep, and fallback-tune can count missed GT on empty-candidate pages instead of silently dropping them from represented-GT metrics;
- `details.jsonl`: per-box rect, bytes, area, quality flags, and source;
- `raw_boxes_contact_sheet.jpg` and `dedup_boxes_contact_sheet.jpg`.

Tune local observation dedupe on replay details before changing Swift thresholds:

```powershell
python scripts\question_observation_dedupe_tune.py `
  --replay diagnostics\question-observation-replay-kpai-layout-measured `
  --ground-truth-manifest diagnostics\question-detector-loop-merged-reviewed-dryrun\dataset\annotations\manifest.jsonl `
  --out diagnostics\question-observation-dedupe-tune-kpai `
  --clean
```

The tuner writes `summary.json`, `grid.jsonl`, and baseline suppression traces. It checks Swift-style signature dedupe and replay image-hash/IoU dedupe against represented reviewed boxes when available.

Tune backend full-frame fallback policy on the same replay before changing extraction fallback thresholds:

```powershell
python scripts\question_observation_fallback_tune.py `
  --replay diagnostics\question-observation-replay-kpai-layout-measured `
  --ground-truth-manifest diagnostics\question-detector-dataset-merged-reviewed-smoke-neardupe-gate\annotations\manifest.jsonl `
  --out diagnostics\question-observation-fallback-tune-kpai `
  --clean
```

The tuner writes `summary.json`, `grid.jsonl`, `image_risk.jsonl`, and `summary.md`. It evaluates the current backend policy and a grid of stricter/looser policies by represented-GT recall, full-frame fallback count, fallback reasons, and total crop+fallback VLM pixels versus full-frame pixels. It also writes `baseline_backend_current_union_area` and per-image `union_area` / `overlap_area_ratio`, so dense overlapping crops are audited separately from summed crop area.

Sweep the iOS ranked per-frame candidate cap on the same replay before changing `candidateLimitPerFrame`:

```powershell
python scripts\question_observation_candidate_cap_sweep.py `
  --replay diagnostics\question-observation-replay-kpai-layout-measured `
  --ground-truth-manifest diagnostics\question-detector-dataset-merged-reviewed-smoke-neardupe-gate\annotations\manifest.jsonl `
  --out diagnostics\question-observation-candidate-cap-sweep-kpai `
  --clean
```

The sweep writes `summary.json`, `cap_sweep.jsonl`, `cap_risk.jsonl`, and `summary.md`. It simulates ranking plus per-frame cap before backend fallback, then reports crop-only GT recall, fallback-protected GT recall, skipped candidate risk, selected crop pixels, fallback pixels, and total crop+fallback VLM pixels versus full-frame pixels for each cap.

Sweep dense/adaptive section crops as an alternative extraction-source strategy:

```powershell
python scripts\question_observation_section_crop_sweep.py `
  --replay diagnostics\question-observation-replay-kpai-layout-measured `
  --ground-truth-manifest diagnostics\question-detector-dataset-merged-reviewed-smoke-neardupe-gate\annotations\manifest.jsonl `
  --out diagnostics\question-observation-section-crop-sweep-kpai `
  --clean
```

The section sweep writes `summary.json`, `strategies.jsonl`, `per_image.jsonl`, and `summary.md`. It treats section crops as VLM extraction inputs that may cover multiple questions, computes section-only represented-GT recall by GT coverage, keeps fallback pixels additive, and only recommends strategies that preserve recall while staying below full-frame VLM pixels.

Evaluate the shipped iOS dense section sender policy, not just the generic sweep:

```powershell
python scripts\question_observation_section_sender_eval.py `
  --replay diagnostics\question-observation-replay-synthetic-dense-oracle `
  --ground-truth-manifest diagnostics\question-detector-synthetic-dense\annotations\manifest.jsonl `
  --out diagnostics\question-observation-section-sender-eval-synthetic-dense-oracle `
  --clean

python scripts\question_observation_section_sender_eval.py `
  --replay diagnostics\question-observation-replay-synthetic-dense-layout `
  --ground-truth-manifest diagnostics\question-detector-synthetic-dense\annotations\manifest.jsonl `
  --enable-empty-frame-section `
  --out diagnostics\question-observation-section-sender-eval-synthetic-dense-layout-empty-section `
  --clean
```

The sender eval writes `summary.json`, `per_image.jsonl`, `sections.jsonl`, `cap_risk.jsonl`, and `summary.md`. It compares cap-only fallback against the current section sender, where dense cap-risk frames send one page-tight `ios_page_dense_v1` section crop plus the top 2 selected per-question companion crops, and zero-candidate study frames can use one guarded `ios_page_empty_v1` section crop. Current synthetic dense oracle evidence: combined crop recall 1.0, fallback-protected recall 1.0, 10 section crops, 20 companion crops, 0 fallback images, and 0.701777x full-frame VLM pixels versus 1.756221x for cap-only fallback. Current synthetic dense measured-layout empty-section evidence: combined crop recall 1.0, fallback-protected recall 1.0, 10 section crops, 0 fallback images, and 0.563498x full-frame VLM pixels versus 1.0x full-frame fallback.

Aggregate the current crop/fallback/section evidence into one strategy table:

```powershell
python scripts\question_observation_strategy_selector.py `
  --candidate-eval diagnostics\question-observation-candidate-eval-kpai-layout-v5-dense-gated `
  --candidate-eval diagnostics\question-observation-candidate-eval-reviewed-propagated-layout-v5-gated `
  --fallback-tune diagnostics\question-observation-fallback-tune-kpai-layout-v5-dense-gated-backend-tuned `
  --fallback-tune diagnostics\question-observation-fallback-tune-reviewed-propagated-layout-v5-gated-backend-tuned `
  --candidate-cap-sweep diagnostics\question-observation-candidate-cap-sweep-kpai-layout-v5-dense-gated-backend-tuned `
  --candidate-cap-sweep diagnostics\question-observation-candidate-cap-sweep-reviewed-propagated-layout-v5-gated-backend-tuned `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-synthetic-dense-oracle `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-synthetic-dense-layout-empty-section `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-propagated-risk-section-top1-ios-signals `
  --section-sender-eval diagnostics\question-observation-section-sender-eval-kpai-risk-section-top1-ios-signals `
  --section-crop-sweep diagnostics\question-observation-section-crop-sweep-kpai-layout-v5-dense-gated-backend-tuned `
  --section-crop-sweep diagnostics\question-observation-section-crop-sweep-reviewed-propagated-layout-v5-gated-backend-tuned `
  --out diagnostics\question-observation-strategy-selection-risk-section-top1 `
  --clean
```

The selector writes `strategy_selection.json`, `strategy_rows.jsonl`, and
`strategy_selection.md`. It separates reviewed production evidence from
synthetic stress rows; use `best_reviewed_live_by_cost` for product decisions,
and use synthetic rows only as negative gates.

Evaluate local observation crop candidates directly against reviewed boxes:

```powershell
python scripts\question_observation_candidate_eval.py `
  --replay diagnostics\question-observation-replay-kpai-layout-measured `
  --ground-truth-manifest diagnostics\question-detector-dataset-merged-reviewed-smoke-neardupe-gate\annotations\manifest.jsonl `
  --out diagnostics\question-observation-candidate-eval-kpai `
  --clean
```

The evaluator writes `summary.json`, `matches.jsonl`, `missed_gt.jsonl`, `unmatched_reviewed_candidates.jsonl`, `unverified_candidates.jsonl`, `overcrop_candidates.jsonl`, `per_image.jsonl`, and `summary.md`. It clusters duplicate represented GT boxes by exact image hash plus normalized bbox before computing recall, so duplicate reviewed screenshots do not become false missed-question alerts.

Current KPAI replay result on 2026-06-30:

- layout-prelabel median processing time: about 31 ms/image on this Windows machine;
- 41 deduped historical images produced 67 measured layout boxes and 63 cross-frame dedup boxes;
- full frames plus deduped crop JPEGs were 1.6018x keyframe-only upload;
- rect-only dedup upload was 1.0020x keyframe-only upload;
- crop VLM pixel proxy was 0.5584 of full-frame pixels.

Current approved-smoke dataset replay:

- 86 unique images including negatives, 11 raw boxes, 9 dedup boxes;
- full frames plus deduped crop JPEGs were 1.0526x keyframe-only upload;
- rect-only dedup upload was 1.0002x keyframe-only upload;
- crop VLM pixel proxy was 0.0403 of full-frame pixels.

Current dedupe-tune result:

- old Swift weak-layout rule: 6/67 selected candidates, represented-GT recall 0.0, because position-only weak dedupe suppressed reviewed questions that appeared in similar screen positions;
- text-guarded Swift weak-layout rule: 67/67 selected candidates, represented-GT recall 1.0, no unknown reviewed suppressions;
- replay image-hash/IoU recommendation: `ahash_threshold=2`, `iou_threshold=0.45`, 64/67 selected candidates, represented-GT recall 1.0, rect-only upload still about 1.002x full-frame bytes.

Current fallback-tune result after promoting the backend-tuned policy:

- `python scripts\question_fallback_policy_smoke.py` passes and verifies that backend production thresholds match the offline `backend_current` fallback policy used by tuning, including no-crop, generation-error, candidate-cap, section-protection, low-coverage, all-weak, and low-confidence boundary behavior.
- KPAI measured-layout replay: current backend policy covers 11 represented reviewed GT keys with recall 1.0, sends full-frame fallback for 7/41 crop-bearing images, and uses 0.784856x full-frame VLM pixels. Before this progression, the all-weak fallback sent fallback for 41/41 images and used 1.625661x; the first narrowed policy sent 14/41 and used 0.958488x.
- Propagated reviewed replay: current backend policy covers 16 represented reviewed GT keys with recall 1.0, sends full-frame fallback for 34/89 images, and uses 0.819493x full-frame VLM pixels instead of the old 57/89 and 1.071420x.
- Legacy-to-current delta audit now writes `policy_delta.jsonl`, `policy_delta_contact_sheet.jpg`, and `policy_delta_review_queue` inside each fallback-tune output. KPAI removes 7 legacy fallbacks, 1 has represented GT and 0 crop-missed GT, with 6 no-GT images / 11 boxes queued at `diagnostics\question-detector-review-workbench-fallback-delta-kpai-backend-tuned`. Propagated removes 23 legacy fallbacks, 4 have represented GT and 0 crop-missed GT, with 19 no-GT images / 32 boxes queued at `diagnostics\question-detector-review-workbench-fallback-delta-propagated-backend-tuned`.
- The promoted backend policy keeps no-crop, crop-generation-error, and risky candidate-cap fallback protections, but disables standalone all-weak/low-confidence full-frame fallback. Those weak/low-confidence counts remain diagnostics for future replay review.
- Candidate-cap sweep at `diagnostics\question-observation-candidate-cap-sweep-kpai-layout-v5-dense-gated-backend-tuned` confirms the current production cap 12 keeps crop-only recall 1.0, fallback-protected recall 1.0, and total crop+fallback VLM pixels at 0.784856x full-frame. Cap 3 is only a diagnostic KPAI low-cost point at 0.779875x until larger reviewed multi-layout data confirms it without recall loss.
- Propagated candidate-cap sweep at `diagnostics\question-observation-candidate-cap-sweep-reviewed-propagated-layout-v5-gated-backend-tuned` keeps 16/16 recall at cap 12, with 34/89 fallback images and 0.812200x full-frame VLM pixels. Cap 10 is diagnostic at 0.809000x.
- Sparse weak/low-coverage risk sections now target the exact class of frames exposed by fallback-delta review. `diagnostics\question-observation-section-sender-eval-kpai-risk-section-top1-ios-signals` keeps 11/11 represented-GT recall, reduces fallback images from 7 to 1, and lowers total VLM pixels from 0.784856x to 0.762230x. `diagnostics\question-observation-section-sender-eval-propagated-risk-section-top1-ios-signals` keeps 16/16 recall, reduces fallback images from 34 to 18, and keeps total VLM pixels essentially flat at 0.810149x versus 0.812200x. Section-only replacement is not safe yet because it drops propagated recall to 14/16, so live `ios_page_risk_v1` keeps a top-1 per-question companion anchor while dense cap-overflow sections keep top-2.
- Strategy selector at `diagnostics\question-observation-strategy-selection-risk-section-top1` ranks current reviewed live strategies by cost while keeping synthetic stress evidence separate. The best reviewed live row is the iOS section sender on KPAI: recall 1.0, fallback ratio 0.024390, total VLM pixels 0.762230x full-frame. On the broader propagated reviewed replay, the iOS section sender keeps recall 1.0, fallback ratio 0.202247, and total VLM pixels 0.810149x, slightly below cap-only 0.812200x while cutting fallback pressure from 0.382022.
- Section-crop sweep at `diagnostics\question-observation-section-crop-sweep-kpai-layout-v5-dense-gated-backend-tuned` finds `horizontal__n4__side1200` as a diagnostic passing strategy on current reviewed KPAI evidence: represented-GT recall 1.0, fallback-protected recall 1.0, 42 section crops, 6 fallback images, and 0.765096x full-frame VLM pixels.
- Propagated section-crop sweep at `diagnostics\question-observation-section-crop-sweep-reviewed-propagated-layout-v5-gated-backend-tuned` finds `page__side1200`: represented-GT recall 1.0, fallback-protected recall 1.0, 71 section crops, 34 fallback images, and 0.795277x full-frame VLM pixels.
- Section-sender eval at `diagnostics\question-observation-section-sender-eval-synthetic-dense-oracle` verifies the live `ios_page_dense_v1` top-2 companion policy on 256 dense expected boxes: combined crop recall 1.0, fallback-protected recall 1.0, 10 section crops, 20 companion crops, 0 fallback images, and 0.701777x full-frame VLM pixels. The old cap-only path on the same fixture costs 1.756221x because it sends fallback for all 10 dense images; the section+top12 diagnostic costs 1.331615x and is not the live policy.
- Empty-frame section-sender eval at `diagnostics\question-observation-section-sender-eval-synthetic-dense-layout-empty-section` verifies the live `ios_page_empty_v1` policy when measured layout emits 0 candidates: combined crop recall 1.0, fallback-protected recall 1.0, 10 section crops, 0 fallback images, and 0.563498x full-frame VLM pixels instead of 1.0x full-frame fallback.
- Dense layout prelabel v5-gated evidence is written at `diagnostics\question-observation-candidate-eval-synthetic-dense-layout-dense-prelabel-v5-gated`, `diagnostics\question-observation-candidate-cap-sweep-synthetic-dense-layout-dense-prelabel-v5-gated-backend-tuned`, and `diagnostics\question-observation-fallback-tune-synthetic-dense-layout-dense-prelabel-v5-gated-backend-tuned`: current layout crop-only recall reaches 256/256 with 0 overcrop hard cases, but no-cap per-question crops still cost 1.960380x full-frame VLM pixels, so the production path should continue to use section batching instead of sending every dense question crop separately.
- Real-data guard evidence is written at `diagnostics\question-observation-candidate-eval-kpai-layout-v5-dense-gated` and `diagnostics\question-detector-layout-regression-check-v5-gated-backend-tuned`: KPAI reviewed-smoke recall is 11/11 with 67 candidates and crop pixels 0.625661x full-frame. A broader propagated reviewed replay at `diagnostics\question-observation-candidate-eval-reviewed-propagated-layout-v5-gated` covers 16/16 represented GT with crop pixels 0.434122x full-frame. The regression check now also requires backend-current fallback pixels <= 0.85x on KPAI, <= 0.90x on propagated replay, and 0 known crop-missed GT among legacy fallback removals.
- Union-area shadowing in the same fallback tune matches the current summed-area policy on this fixture: 7/41 fallback images, represented-GT recall 1.0, and 0.784856x full-frame VLM pixels. The maximum observed overlap-area ratio is about 0.081, so this fixture does not yet exercise severe crop-overlap risk.

Current candidate-eval result:

- KPAI measured-layout replay: 67 candidates across 41 images, 11 duplicate-clustered represented reviewed GT keys, represented-GT recall 1.0, reviewed candidate match rate 1.0, crop VLM pixels 0.625661x full-frame pixels, and 0 overcrop candidates.
- Reviewed-approved smoke replay: 11 candidates, 9 duplicate-clustered represented reviewed GT keys after exact duplicate clustering, represented-GT recall 1.0, crop VLM pixels 0.671895x full-frame pixels, and 0 overcrop candidates. Before duplicate clustering, the same fixture appeared as 9/11 because two duplicate reviewed screenshots were counted as separate GT misses.

Use this replay after every detector candidate. A model candidate is not enough if IoU is good; it should also keep rect-only overhead near zero, reduce crop/VLM pixels versus full frames, and lower local segmentation latency compared with OCR/layout fallback.

## Rollout Gates

Do not replace OCR-only V3 until a detector run proves:

- no regression in the diagnostics benchmark bad-crop count;
- detector eval passes recall/precision gates on held-out session/batch splits;
- lower live segmentation latency on target iPhone/iPad devices;
- no increase in duplicate uploads after V3 rect-only dedupe;
- robust fallback when the model is absent or returns empty results;
- source-image normalized coordinate semantics remain stable.
