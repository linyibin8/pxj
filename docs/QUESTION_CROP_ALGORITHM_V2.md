# Question Crop Algorithm V3

## Goal

Smart observation should identify question regions on iPhone/iPad quickly enough for live use, crop only useful question evidence, dedupe repeated frames/questions, and send compact high-quality inputs to the backend VLM.

Full-page images are a safe fallback, but they are expensive for model reasoning: whole-page reasoning asks the VLM to re-detect every question. Because observation batches already upload keyframes, sending crop JPEGs in addition to those frames can nearly double network payload. The better production shape is: upload keyframes once, upload deduped question rects as JSON, and let the backend crop/expand canonical question images from the saved source frames.

## Current Findings

Historical fixture:

- Data: `diagnostics/crop-f472/data`
- Output: `diagnostics/crop-f472/benchmark-v3`
- Command: `python scripts/question_crop_benchmark.py --data diagnostics/crop-f472/data --out diagnostics/crop-f472/benchmark-v3`

| Strategy | Bad / 20 | Near-Dedup Crops | Extra Crop JPEG Bytes vs Full Frames | Keyframes + Dedup Crop JPEGs | Keyframes + Dedup Rect Manifest |
|---|---:|---:|---:|---:|---:|
| historical_crop | 4 | 11 | 0.490 | 1.490 | 1.0022 |
| old_client_padding | 4 | 11 | 0.519 | 1.519 | 1.0022 |
| adaptive_client_v2 | 0 | 11 | 0.581 | 1.581 | 1.0022 |
| server_safety_expand | 0 | 11 | 0.564 | 1.564 | 1.0022 |
| rect_only_v3_server_crop | 0 | 11 | 0.581 | 1.581 | 1.0022 |

Interpretation:

- The old crop is fast but not reliable; short/table-like questions become thin strips.
- V2 geometry fixes reliability: bad crops drop from 4/20 to 0/20 on this fixture.
- Cropping alone is not enough. If iOS uploads both keyframes and crop JPEGs, even deduped safe crops add about 58% extra image bytes.
- V3 rect-only keeps the same safe question crop geometry, but sends only about 5 KB of deduped rect JSON for this fixture. Total upload becomes about 100.22% of keyframe-only upload while still giving the backend per-question crop images for the VLM.

## Implemented V3

iOS:

- `QuestionSegmenter.segmentForPreviewOverlay(_:fast:)` accepts a fast mode and no longer detects the document quad twice.
- Smart observation live scan and batch crop upload use fast OCR.
- Vision resize and upload JPEG compression are now pixel-bound: the app limits real `CGImage` pixels with a renderer scale of 1 instead of relying on `UIImage.size` point dimensions that can be inflated by `UIScreen.main.scale`.
- Observation crop rectangles use `observationCropPlan`:
  - stable OCR blocks get small padding only;
  - narrow/short/thin blocks expand to a safer question block;
  - thin strips account for source image aspect ratio so pixel aspect stays under the bad-crop threshold;
  - manifest includes crop quality and risk reasons.
- Live observation scan results are cached by frame sequence; batch upload reuses cached candidates and only falls back to OCR when the cache is missing.
- Client dedupe uses a conservative question signature: exact/contained key matches merge immediately; fuzzy text matches require nearby crop geometry or nearby same question index.
- Short OCR blocks that cannot form a strong text key now use a `layout:` weak key from reading order, crop geometry, and short text. Weak keys improve recall for formula/short/figure-heavy questions, but weak-to-weak dedupe now requires compatible short text unless the weak key is exactly identical. This avoids suppressing different questions that happen to appear in the same screen position.
- Observation upload is now rect-only:
  - manifest version is `2`;
  - each non-duplicate question sends `transfer_mode=rect_only`, `crop_prepared=false`, source frame index/sequence, OCR key, and normalized `crop_rect`;
  - no `question_crops` JPEG files are attached from iOS for observation batches;
  - upload dedupe signatures still advance after successful manifest upload.
- Dense cap-risk frames now add one page-tight section rect-only source plus the top 2 per-question companion crops. When a frame has candidates beyond the per-frame cap, iOS emits `crop_kind=section` with `question_key_strength=section_crop`, covered candidate keys/subrects, `section_strategy=ios_page_dense_v1`, section telemetry, and `section_companion_candidate_limit=2`. If a study-material frame has no local candidates, iOS emits one guarded `ios_page_empty_v1` section instead of forcing whole-frame VLM. These section crops are VLM extraction sources, not single-question identities.

Backend:

- Crop dedupe checks exact `question_key`, `fingerprint`, `text_hash`, and `crop_hash`.
- Weak `layout:` keys are not used for pre-generation text/fingerprint dedupe; they wait for real crop-hash dedupe after the backend generates the canonical crop. This avoids suppressing different short questions that share position or generic text.
- Backend extraction source enumeration now follows the same rule: weak `layout:` keys and their fingerprints are preserved as telemetry, but they cannot suppress candidate crops before VLM extraction.
- Strong-key duplicates use a best-evidence policy: the backend generates the new candidate crop, scores it by confidence, crop size, crop area, file availability, and safety-expansion status, then replaces the older row when the new crop is materially better. This prevents an early blurry/thin crop from permanently occupying a question.
- Client crop quality/reasons are preserved in crop safety trace.
- Full-frame fallback is still included when crop coverage is suspicious, so partial client crops do not suppress extraction of the rest of the page. Fallback triggers include no crops, crop generation errors, risky per-frame candidate limiting, one very low-coverage crop, two sparse crops, tiny total crop coverage, or no sufficiently large question crop. Weak layout-only and low-confidence crop counts are still logged as diagnostics, but they no longer trigger a standalone full-frame pass after the backend-tuned fallback policy proved too conservative on real reviewed replay. New clients rank candidates before the cap and report whether skipped candidates include unique strong-OCR or high-confidence evidence; old clients without this telemetry remain conservative.
- Backend fallback stats now include union-area coverage and overlap ratios alongside summed crop area. The live fallback decision still follows the currently tuned summed-area thresholds, while diagnostics can detect when overlapping crops make summed coverage misleading.
- `save_question_crop_uploads` is manifest-driven. Old clients with `question_crops` files still work; V3 clients without crop files generate canonical crop JPEGs on the backend from the saved source frame and `crop_rect`.
- Rect-only generation records `server_rect_crop` / `server_rect_expanded`, computes the real crop hash and image size from the generated JPEG, and performs a second hash dedupe after generation.
- Manifest source references are restricted to images in the current batch; conflicting sequence/index/source ids are skipped.
- Batched crop extraction now asks the VLM to return `input_index` for each question and binds extracted questions by that explicit 1-based image order before falling back to text similarity. This reduces source/crop drift when the model reorders or partially merges outputs.
- The release readiness gate statically verifies this backend `input_index` source-binding path, including prompt request, parser aliases, 1-based source assignment, index-first matching, and persisted `source_input_index`.
- Backend crop manifests now accept section/group crop metadata as extraction-source telemetry: `crop_kind=section`, `question_key_strength=section_crop`, `section_key`, `covered_question_count`, `covered_question_keys`, `covered_question_numbers`, `covered_subrects`, `section_strategy`, and `coverage_score`. Section keys are treated as weak identity for pre-VLM source dedupe, and extracted questions preserve `source_crop_kind=section` plus section metadata in their payload.
- The section path is deliberately source-safe: a section crop may return multiple questions from the same `input_index`, but it is not a reviewed single-question training box and does not replace per-question identity.

Benchmark:

- `scripts/question_crop_benchmark.py` replays historical manifests and compares old/new strategies for bad count, area, crop bytes, total keyframe+crop bytes, rect-only manifest bytes, and contact sheets.
- `scripts/question_observation_replay.py` replays observation-level payload strategies on diagnostics, reviewed detector datasets, or historical prelabel packages. It reports full-frame bytes, crop-JPEG extra bytes, rect-only manifest bytes, dedupe effect, VLM pixel proxy, JPEG timing, optional layout-prelabel CPU timing, and `images.jsonl` for zero-candidate page accounting.
- `scripts/question_observation_dedupe_tune.py` sweeps observation dedupe thresholds against replay `details.jsonl` and optional reviewed detector labels. It reports payload savings, selected crop pixels, represented-GT recall, lost GT keys, and suppression traces for Swift-style signatures and image-hash/IoU replay dedupe.
- `scripts/question_observation_fallback_tune.py` sweeps backend full-frame fallback policy thresholds against replay `details.jsonl` and optional reviewed detector labels. It reports represented-GT recall, fallback reasons, fallback image ratio, total crop+fallback VLM pixels versus full-frame pixels, simulates the ranked per-frame candidate cap used by iOS, compares summed crop area with union-area coverage, and now writes a legacy-vs-current `policy_delta.jsonl`, contact sheet, and no-GT review queue for frames where backend-tuned policy removes a full-frame fallback.
- `scripts/question_fallback_policy_smoke.py` verifies the live backend fallback policy boundaries and checks that `question_observation_fallback_tune.py` uses the same `backend_current` thresholds as production code.
- `scripts/question_observation_candidate_cap_sweep.py` replays the iOS ranked per-frame candidate cap before backend fallback. It reports crop-only GT recall, fallback-protected GT recall, skipped candidate risk, and total crop+fallback VLM pixels for each cap.
- `scripts/question_observation_section_crop_sweep.py` simulates dense-page section/group crops as VLM extraction sources. It reports section-only recall, fallback-protected recall, section count, fallback count, and additive crop+fallback VLM pixels so grouped crops are compared fairly against per-question crops and full frames.
- `scripts/question_observation_section_sender_eval.py` mirrors the shipped iOS section sender. It verifies `ios_page_dense_v1` cap-overflow sections, `ios_page_empty_v1` no-candidate study-frame sections, and `ios_page_risk_v1` sparse weak/low-coverage sections against replay evidence.
- `scripts/question_observation_candidate_eval.py` evaluates local crop candidates directly against reviewed boxes. It reports duplicate-clustered represented-GT recall, reviewed candidate match rate, crop VLM pixel ratio, missed GT, unmatched reviewed candidates, unverified candidates, and overcrop candidates.
- `scripts/question_detector_synthetic_dense_fixture.py` generates diagnostic-only dense phone/iPad pages with separate expected boxes and candidate boxes, so cap/fallback/replay tools can test pages with 18 to 36 questions without polluting reviewed training data.
- `scripts/question_detector_synthetic_dense_stress_check.py` summarizes dense stress evidence as a negative gate: passing does not prove release readiness, but failing exposes dense-page regressions that should block promotion until fixed.

Current observation replay baselines:

| Input | Images | Raw Boxes | Dedup Boxes | Full + Dedup Crop JPEG vs Full | Rect-Only Dedup vs Full | Crop Pixels vs Full Pixels | Layout Median |
|---|---:|---:|---:|---:|---:|---:|---:|
| `diagnostics/question-detector-prelabel-kpai` using measured layout boxes | 41 | 67 | 63 | 1.6018 | 1.0020 | 0.5584 | 30.9 ms/image |
| `diagnostics/crop-f472/data` V3 rect replay | 9 | 20 | 12 | 1.5637 | 1.0014 | 0.4572 | n/a |
| `diagnostics/question-detector-reviewed-approved-smoke` | 86 | 11 | 9 | 1.0526 | 1.0002 | 0.0403 | n/a |

Interpretation:

- If observation keyframes are already uploaded, crop JPEGs are additional network payload. On current historical prelabel replay, even after dedupe they add about 60% over keyframe-only upload.
- Rect-only manifests add about 0.02% to 0.20% in these runs, while still allowing the backend to crop canonical per-question images.
- The VLM-side win comes from sending question crops to extraction instead of asking the model to rediscover questions from every full frame. On the approved-smoke set with many negative frames, crop pixels are only about 4% of full-frame pixels.
- The current lightweight OpenCV layout prelabeler is around 31 ms/image median on this Windows machine for 1920x1080 historical images. The target Core ML detector should beat this on-device while improving box quality.
- Dedupe tuning on the KPAI replay plus merged reviewed labels found that the old weak-layout Swift rule was too aggressive: position-only weak dedupe selected 6/67 candidates and represented-GT recall fell to 0.0. With weak-key text compatibility enabled, represented-GT recall returned to 1.0. The replay image-hash/IoU recommendation (`ahash_threshold=2`, `iou_threshold=0.45`) selected 64/67 candidates with represented-GT recall 1.0 and rect-only overhead about 1.002x full-frame bytes.
- Fallback tuning on the same KPAI replay found that unconditional all-weak full-frame fallback was too slow: it sent fallback for 41/41 crop-bearing images and used 1.625661x full-frame VLM pixels. The first narrowed policy reduced that to 14/41 and 0.958488x. The backend-tuned policy now uses stricter low-coverage thresholds, keeps cap/error/no-crop protections, disables standalone all-weak/low-confidence fallback, and preserves represented-GT recall 1.0 while reducing KPAI fallback to 7/41 images and 0.784856x full-frame VLM pixels. On the broader propagated reviewed replay it preserves 16/16 represented GT and reduces fallback from 57/89 at 1.071420x to 34/89 at 0.819493x.
- The fallback delta audit makes the remaining risk explicit: KPAI removes legacy fallback on 7 images, 1 has represented GT and crop covers it, 6 no-GT images are queued for review. Propagated replay removes legacy fallback on 23 images, 4 have represented GT and crop covers all 4, 19 no-GT images / 32 boxes are queued at `diagnostics\question-detector-review-workbench-fallback-delta-propagated-backend-tuned`.
- Ranked candidate-cap telemetry is now part of the same fallback policy: iOS ranks candidates before the per-frame cap and reports skipped unique/strong/confident candidates. With backend-tuned fallback, the KPAI cap-12 replay keeps 11/11 represented-GT recall with 7/41 fallback images and 0.784856x full-frame VLM pixels; the propagated replay keeps 16/16 recall with 34/89 fallback images and 0.812200x. Lower caps such as KPAI cap 3 or propagated cap 10 stay diagnostic until broader reviewed multi-layout evidence proves them.
- Candidate-cap sweep on the same KPAI replay shows the production cap of 12 is not the current bottleneck: crop-only recall and fallback-protected recall both remain 1.0. Cap 1 is unsafe and costs more than full-frame because fallback pressure dominates. Cap reductions remain diagnostic because dense pages are handled by section crops, not by relying on cap-only fallback.
- Synthetic dense stress now covers 10 diagnostic-only phone/iPad pages with 256 expected question boxes. The original measured-layout baseline quarantined all 10 dense pages and covered 0/256 expected boxes, while even oracle candidates with production cap 12 cover only 120/256 crop-only boxes. Dense layout prelabel v5-gated at `diagnostics\question-observation-candidate-eval-synthetic-dense-layout-dense-prelabel-v5-gated` covers 256/256 expected boxes with 283 candidates, 1.0 represented-GT recall, 0 missed boxes, 0 overcrop hard cases, and crop pixels at 1.960380x full-frame before section batching.
- Section-crop sweep gives the first promising dense-page direction: on the same synthetic dense oracle fixture, one section crop per page at `section_max_side=1200` covers 256/256 expected boxes with no fallback and 0.576576x full-frame VLM pixels. On KPAI, `horizontal__n4__side1200` also keeps represented-GT recall 1.0 with 0.887219x full-frame VLM pixels. This is a candidate architecture, not a shipped behavior: section crops must be treated as extraction sources that may return multiple questions, not as one-question crop identities.
- Live sender evaluation now matches that architecture: `ios_page_dense_v1` plus top-2 companion crops on the synthetic dense oracle fixture covers 256/256 expected boxes, triggers 10 section crops and 20 companion single-question crops, uses no full-frame fallback, and keeps total VLM pixels at 0.701777x full-frame. The `ios_page_empty_v1` no-candidate sender on the measured-layout fixture also covers 256/256 expected boxes, triggers 10 section crops, uses no full-frame fallback, and brings total VLM pixels down to 0.563498x full-frame. The new `ios_page_risk_v1` sparse weak/low-coverage sender keeps a top-1 companion anchor: KPAI stays 11/11 recall while reducing fallback from 7 to 1 and total VLM pixels from 0.784856x to 0.762230x; propagated replay stays 16/16 recall while reducing fallback from 34 to 18 and keeping total VLM pixels flat at 0.810149x versus 0.812200x. Section-only replacement is not safe yet because it drops propagated recall to 14/16. The backend now treats a section that covers cap-risk candidates or enough page area as fallback-protected, so later low-coverage/weak-key fallback rules do not re-upgrade that source to full-frame. The old cap-only path on the same fixture needed fallback on all 10 images and cost 1.756221x full-frame; sending all top-12 selected crops with the section costs 1.331615x and is not the default.
- The dense stress negative gate now separates current layout-candidate recall from empty-layout fallback evidence. `diagnostics\question-detector-synthetic-dense-stress-check-dense-prelabel-v5-gated` reports `transport_ready=true`, `detector_ready=true`, `stress_ready=true`, 0 hard failures, and 2 expected cap-only warnings. A v4 attempt proved why real-data cross-checks are required: it passed synthetic dense but over-triggered on KPAI historical captures, inflating 41 real images from 67 to 403 boxes and dropping reviewed-smoke recall to 7/11. v5 gates dense splitting to `page_source == full_image_fallback`, restoring KPAI to 67 boxes and 11/11 recall while preserving 256/256 synthetic dense recall.
- Union-area fallback shadowing is now part of the same replay: on the current 41-image KPAI fixture, summed-area and union-area backend-tuned policies are identical (7/41 fallback images, 1.0 represented-GT recall, 0.784856x full-frame VLM pixels), with max observed crop-overlap ratio about 0.081. Future dense/overlapping pages will now expose sum-vs-union drift in `image_risk.jsonl` and release checks.
- Detector training is not release-ready yet. A clean-shape bootstrap run (`diagnostics\question-detector-bootstrap-kpai-clean-shape`, 27 labels) shows recall signal only at `conf=0.001`: recall 0.9091 but precision 0.0012 and 189.8864 false positives per image. The active-learning output is now `diagnostics\question-detector-error-mining-bootstrap-clean-shape-workbench` with 43 images and 220 hard-example boxes to review before the next Core ML candidate. The 219 false-positive boxes are default-rejected and cannot enter training through bulk approval; they must be corrected or verified deliberately.
- A longer 25-epoch clean-shape run did not solve this: recall stays 0.9091 at `score=0.001`, but false positives worsen to 281.2727 per image; at `score=0.01`, recall is only 0.6364 with 35.4545 false positives per image. This points to reviewed boundary data and hard-example correction, not more epochs on the same weak bootstrap labels.
- Candidate evaluation on the same KPAI replay found that local crop candidates cover 11/11 duplicate-clustered represented reviewed GT keys with crop pixels at 0.625661x full-frame pixels and no overcrop candidates. The backend-tuned v5 layout regression check at `diagnostics\question-detector-layout-regression-check-v5-gated-backend-tuned` now gates real KPAI recall/candidate stability, synthetic dense recall/stress readiness, fallback cost on both KPAI and propagated reviewed replay, and verifies that legacy-removed fallback frames have 0 known crop-missed GT. A larger propagated reviewed replay also covers 16/16 represented GT with crop pixels at 0.434122x full-frame.
- Dataset export now prevents train/val/test leakage earlier in the loop: `diagnostics/question-detector-dataset-merged-reviewed-next-clustered` clusters 22 raw split groups into 20 effective groups, removes the previous 2 cross-split near-duplicate errors, and leaves the dataset blocked only by smoke-scale label volume.
- The follow-on 3-fold export at `diagnostics/question-detector-cv-merged-reviewed-next-clustered` keeps all fold audits free of residual cross-split near-duplicates, so the next detector comparison will fail on real model/data limits rather than leakage.
- Exact-aHash reviewed propagation adds a small safe label gain: `diagnostics/question-detector-dataset-merged-reviewed-next-propagated-exact` now has 89 images, 16 reviewed annotations, 80 negatives, and no audit errors; the remaining blocker is still reviewed label volume, not leakage.
- Non-exact near-duplicate label transfer is now review-only: `diagnostics/question-detector-propagation-review-candidates-ahash1-2-workbench` contains 1 aHash-distance-2 candidate box with draft status, so visually similar pages help grow the review queue without silently becoming detector ground truth.

Runtime telemetry:

- iOS observation manifests now include `metrics.analysis_duration_ms`, `cached_frame_count`, `ocr_frame_count`, `candidate_count`, `manifest_count`, `duplicate_count`, and `attached_crop_file_count`.
- The manifest also includes candidate quality distribution for replay comparison: `weak_layout_candidate_count`, `strong_ocr_candidate_count`, `weak_layout_manifest_count`, `strong_ocr_manifest_count`, `selected_candidate_count`, `limited_candidate_count`, `limited_unique_candidate_count`, `limited_strong_candidate_count`, `limited_confident_candidate_count`, `limited_max_confidence`, `limited_frame_count`, `limited_source_sequences`, `low_confidence_candidate_count`, crop area totals/means/max, crop quality counts, crop risk reason counts, and `dedupe_drop_rate`.
- Fresh segmentation frames now include local path diagnostics: `selected_segmenter`, `selected_segmenter_counts`, `text_recognition_scope_counts`, `fresh_segmentation_frame_count`, `cached_segmentation_frame_count`, `text_recognition_duration_ms`, `ocr_v3_duration_ms`, `coreml_v4_duration_ms`, `segmentation_total_duration_ms`, `coreml_model_loaded_frame_count`, `fallback_to_ocr_frame_count`, paired OCR-vs-CoreML IoU, and unmatched OCR/CoreML candidate counts.
- Core ML detector-first now has a textless fallback gate: detector boxes are accepted immediately only when OCR inside those boxes finds at least one matched question region. Textless detector boxes trigger full-page OCR fallback, and the code reuses first-pass detector boxes instead of running Core ML a second time.
- `/api/sessions/{session_id}/batches` returns `question_crop_client_metrics` and `question_crop_server_duration_ms`.
- The batch response also exposes `question_crop_replaced_count`, which tracks stronger duplicate evidence replacing an older crop.
- Per-crop client telemetry is preserved under `crop_safety.client_telemetry` in the stored crop trace, including key strength, crop kind, section metadata, crop area, confidence, and source sequence/index.
- Use these fields to compare OCR-only V3 against Core ML V4 on the same observation batches:
  - local segmentation latency: `analysis_duration_ms`;
  - cache effectiveness: `cached_frame_count / frame_count`;
  - upload overhead: `attached_crop_file_count` should stay `0` for rect-only observation;
  - server crop CPU: `question_crop_server_duration_ms`;
  - end-to-end usefulness: saved crop count, replaced crop count, duplicate count, skipped count, and missed-question review rate.
  - fallback pressure: full-frame fallback reasons in extract logs, especially `single_low_coverage_crop`, `sparse_low_coverage_crops`, `limited_candidate_frame`, and `all_weak_crop_keys`.
  - local detector risk: weak layout key ratio, low-confidence candidate count, limited unique/strong/confident candidate counts, crop risk reason counts, and dedupe drop rate.
  - model parity: `selected_segmenter_counts`, `coreml_model_loaded_frame_count`, `fallback_to_ocr_frame_count`, `paired_mean_iou`, `unmatched_ocr_v3_count`, and `unmatched_coreml_v4_count`.

Additional release-gate candidates:

- Swift/evaluator parity gate: compare shipped candidate cap, dense/risk/empty section thresholds, companion limits, and fallback-trigger conditions against the replay/eval script configuration.
- Per-image replay gate: inspect section-sender `per_image.jsonl`, not only aggregate recall, and block on missed combined keys or unreviewed fallback removals.
- Backend multipart route smoke: POST a real rect-only question/section manifest through `/api/sessions/{session_id}/batches` and verify persisted `source_image_id`, `input_index`, `crop_kind`, `crop_safety`, and source binding.
- Final question-set gate: run `question_set_eval.py` on representative sessions so box recall improvements translate into correct question count, order, source coverage, and non-merged question text.

## Recommended Architecture

### V2: Heuristic Crop Geometry, Landed

Use Apple Vision OCR to find text rows, group rows into question blocks, then expand risky blocks with geometry rules. This is the fastest path to improve production quality because it preserves existing APIs.

Use this for:

- live overlays;
- observation crop upload;
- immediate question-set extraction after observation.

### V3: Rect-Only Upload, Landed

Instead of sending full frames plus crop JPEGs, iOS sends full keyframes plus question rect manifests. The backend then crops/expands from the already saved source image.

Benefits:

- almost no duplicate iOS upload payload;
- one canonical backend crop implementation;
- easier retries with different crop parameters;
- VLM still receives per-question crop images instead of full pages only.

Risks and mitigations:

- backend does more CPU work;
- live UI still needs local boxes;
- source references are restricted to current-batch images;
- backend recomputes crop hash/size and dedupes after generation;
- suspicious low-coverage keyframes still add full-frame fallback.

### V4: Local Core ML Detector, In Progress

Train a small question-region detector that outputs bounding boxes/masks. Keep the public interface:

```swift
QuestionSegmenter.segment(_ image: UIImage, fast: Bool) -> [QuestionRegion]
```

Pipeline:

1. Export weak labels from `session_question_crops` / diagnostics with `scripts/question_detector_dataset.py`.
2. Review risky labels in `annotations/review.jsonl` and `preview_contact_sheet.jpg`.
3. Train a small detector on COCO/YOLO/Create ML outputs.
4. Add the compiled model to iOS as `QuestionRegionDetector.mlmodelc`.
5. Run detector on a resized frame inside `QuestionSegmenter.segment`.
6. Convert model boxes to `QuestionRegion` coordinates, attach OCR text, NMS, and reading-order sort.
7. Fall back to current Vision OCR rules when the model is missing, empty, low-confidence, or has no OCR text.

Training data:

- Start with existing `session_question_crops` and manifests as weak labels.
- Use benchmark contact sheets to mark bad/overlarge crops.
- Add manual correction for 300-1000 pages across textbooks, worksheets, handwritten answer areas, screens, iPad landscape, and low-light scenes.
- Export labels as `source_image + bbox + quality_flags`.
- Current exporter writes COCO, YOLO, Create ML, positive-label JSONL, image-level JSONL, review queue, and contact sheet under `diagnostics/question-detector-dataset-v1`.
- Add explicit empty-page negatives through `--empty-page-manifest`, `--negative-images-dir`, or `--include-sqlite-empty-pages`; skipped/review-risk positive candidates must not be used as negatives.
- For historical image pools that have no crop boxes, first create review candidates with `scripts/question_detector_review_candidates.py`; QA-linked positives are image-level hints, not detector labels.

Target model:

- Detect `question_block`, optionally `answer_area`, `figure`, and `table`.
- Optimize for recall first; a slightly larger crop is better than a missing question.
- Quantize for iOS and test through `VNCoreMLRequest`.

Implementation status:

- Data inventory: landed; `scripts/question_detector_data_inventory.py` reports SQLite/image readiness and separates labeled crop evidence from unlabeled image pools.
- Dataset exporter: landed; `scripts/question_detector_dataset.py` supports diagnostics packages, reviewed prelabel approvals, auto-discovered diagnostics roots, explicit SQLite DBs, `data/accounts/*/*.sqlite3` discovery, opt-in empty-page negatives with zero-annotation COCO/YOLO images, risky-label ratio gates, default exact/aHash near-duplicate split-group clustering before train/val/test assignment, and release-blocking residual cross-split near-duplicate checks.
- Draft prelabel exporter: landed; `scripts/question_detector_prelabel.py` creates review-only question-box drafts from QA-linked historical positives, exports COCO/Create ML/Label Studio draft tasks, and quarantines unreliable page detections for manual page review. These outputs are explicitly not ground truth until approved.
- Active-learning batch selector: landed; `scripts/question_detector_active_batch.py` prioritizes diverse, high-value draft rows for human review while skipping already-approved, already-queued, and visually duplicate images.
- Review backlog builder: landed; `scripts/question_detector_review_backlog.py` consolidates pending active-learning, fallback-delta, propagation, and error-mining workbenches into one ranked, de-duplicated prelabel queue for manual review without treating the copied boxes as ground truth.
- Review wave planner: landed; `scripts/question_detector_review_wave_plan.py` repeatedly creates active-learning waves from one prelabel pool, excludes already queued waves, builds optional workbenches, and reports when the pool is exhausted so the next loop expands historical candidates instead of re-sampling stale data.
- Review workbench: landed; `scripts/question_detector_review_workbench.py` builds a local HTML review queue for draft boxes and converts reviewer decisions into approved JSONL accepted by `question_detector_dataset.py --reviewed-prelabels`.
- Review queue manifest: landed; `scripts/question_detector_review_queue_manifest.py` builds one dashboard across many review workbenches, tracks pending/approved/export status, and preserves the merge command for the next dataset build.
- Reviewed-label merger: landed; `scripts/question_detector_merge_reviewed.py` merges multiple approved workbench exports, copies images into one package, deduplicates overlapping boxes, and records conflicts before dataset export.
- Reviewed duplicate propagation: landed; `scripts/question_detector_propagate_reviewed.py` transfers approved boxes only to exact/aHash duplicate historical images with same-size and draft-overlap evidence, adding traceable labels without treating model guesses as ground truth. Automatic approved propagation is limited to exact-aHash matches with overlap evidence; non-exact aHash transfers are rejected unless emitted with `--review-candidates` and reviewed manually before training.
- Detector evaluator: landed; `scripts/question_detector_eval.py` accepts COCO, JSONL, and YOLO label predictions, then reports IoU recall/precision, missed boxes, false positives, false positives per image, negative-image FP rate, duplicate predictions, score-threshold sweeps, and strata by quality flag, area, origin, confidence, and negative kind.
- Review candidate exporter: landed; `scripts/question_detector_review_candidates.py` packages unlabeled historical positives/negatives for manual annotation and active learning, and can opt in to unverified `session_observations` as review-only positive candidates when QA-linked positives are exhausted.
- Bootstrap exploration dataset: landed; `scripts/question_detector_bootstrap_dataset.py` merges approved boxes, filtered high-confidence draft prelabels, and negative images into a normal detector dataset for early training and hard-example mining, while forcing `model_training_ready=false`.
- Mobile capture augmentation: landed; `scripts/question_detector_mobile_augment.py` creates iPhone/iPad-style crop, rotation, exposure, blur, resampling, and JPEG variants from approved boxes while preserving the dataset contract and inheriting source dataset readiness.
- Cross-validation exporter: landed; `scripts/question_detector_cv.py` turns one exported dataset into grouped K-fold dataset roots, clusters exact/aHash near-duplicate groups before assigning folds, keeps positive-label coverage in each validation split when enough positive groups exist, and treats any residual cross-split near-duplicates as release-blocking errors for model selection.
- Experiment matrix orchestrator: landed; `scripts/question_detector_experiment_matrix.py` runs or dry-runs fold x model x input-size x seed training grids, records manifests/logs, invokes selector after completed runs, and marks top-level status `failed_eval_gate` when a run only succeeded because `--allow-failed-eval` was used.
- Detector error mining: landed; `scripts/question_detector_error_mining.py` converts eval `problem_images` into a review workbench queue, so missed boxes and false positives become the next active-learning batch.
- Iteration report: landed; `scripts/question_detector_iteration_report.py` summarizes dataset readiness, experiment metrics, release blockers, replay payload savings, strategy-selection conclusions, training-readiness plans, hard-example queues, and review-workbench closure state, including fallback-delta queues created when backend-tuned fallback removes legacy full-frame protection.
- Loop runner: landed; `scripts/question_detector_loop_runner.py` orchestrates a repeatable reviewed-label merge/dataset/CV/experiment/release/error-mining/workbench/iteration-report cycle while keeping training execution explicit, preserving release blockers, and passing replay, dedupe-tune, fallback-tune, candidate-eval, candidate-cap-sweep, section-sender-eval, strategy-selection, training-plan, backend-main, and review-workbench inputs through to the right reports.
- Runtime perf checker: landed; `scripts/question_observation_runtime_perf_check.py` extracts live/simulator observation telemetry from JSON, JSONL, or logs and gates local analysis p95, backend crop p95, fresh OCR ratio, and rect-only `attached_crop_file_count=0`. Fixture evidence is marked separately and release-gate accepted only with explicit smoke allowance.
- Synthetic dense stress fixture: landed; `scripts/question_detector_synthetic_dense_fixture.py` and `scripts/question_detector_synthetic_dense_stress_check.py` generate diagnostic-only dense phone/iPad pages, keep oracle candidates separate from expected boxes, and report negative-gate blockers without counting synthetic boxes toward reviewed training readiness.
- Model selector: landed; `scripts/question_detector_select.py` aggregates compatible fold `train_manifest.json` files, isolates incompatible datasets/train configs/seeds/thresholds, detects duplicate fold inputs, picks the best model and confidence threshold, and emits iOS-ready threshold recommendations.
- Training wrapper: landed; `scripts/question_detector_train.py` enforces dataset `audit.json`, delegates training/export to Ultralytics, can export a `QuestionRegionDetector` Core ML artifact, evaluates held-out test predictions with optional score sweep, and fails with `failed_eval_gate` when recall/precision/FP-density gates are not met.
- Release readiness gate: landed; `scripts/question_detector_release_check.py` ties dataset readiness, cross-fold model selection, Core ML artifact export, iOS bundling, Swift threshold sync, replay payload/latency evidence, non-fixture runtime perf evidence, candidate-eval crop recall/cost evidence, candidate-cap sweep recall/VLM-cost evidence, dense/empty/risk section-sender recall/cost evidence, strategy-selection reviewed-live Pareto evidence, dedupe-tune recall/false-merge evidence, fallback-tune recall/VLM-cost evidence including union-area shadow coverage, legacy-to-current fallback-delta evidence with 0 represented-GT misses and no-GT removed-fallback review queues, static iOS weak-key text-guard verification, static iOS ranked candidate-cap verification, static iOS dense section-crop sender plus top-2 companion and risk section plus top-1 companion verification, static iOS pixel-bound Vision/JPEG resize verification, static iOS detector-first textless fallback verification, static backend crop `input_index` source-binding, weak-key source-dedupe guard, section-crop source-safe handling, risk-aware candidate-cap fallback verification, and post-bundle Xcode build proof into one `release_ready` decision.
- Section-crop strategy sweep: landed; `scripts/question_observation_section_crop_sweep.py` tests grouped dense-page extraction-source crops against replay+GT evidence, keeps section and fallback pixels additive, and requires section-only recall plus total pixels below full-frame before recommending a strategy.
- Observation strategy selector: landed; `scripts/question_observation_strategy_selector.py` normalizes candidate-eval, fallback-tune, candidate-cap, section-sender, and section-sweep summaries into one Pareto table, separating reviewed production evidence from synthetic stress gates.
- Training readiness planner: landed; `scripts/question_detector_training_readiness_plan.py` turns dataset gaps, pending review queues, review-wave capacity, release blockers, and the chosen observation strategy into a concrete reviewed-label plan without counting pending or synthetic boxes as training evidence. When a prelabel pool is exhausted, it emits `expand_candidate_pool` and marks same-pool wave estimates as valid only after expansion.
- iOS promotion helper: landed; `scripts/question_detector_ios_promote.py` copies a selected Core ML artifact into the iOS app and syncs detector confidence thresholds, but refuses to apply when selection, fold count, eval gate, or artifact checks fail.
- Detector post-process sweep: landed; `scripts/question_detector_postprocess_sweep.py` checks whether score, box shape, NMS, and top-k filters can turn held-out detector predictions into an iOS-safe candidate set before spending time on Core ML export.
- Question-set evaluator: landed; `scripts/question_set_eval.py` compares final product question sets against reference sets and reports one-to-one recall, coverage recall, merge suspects, duplicates, source coverage, number/order risk, and content placeholders.
- Optional iOS Core ML hook: landed behind `QuestionSegmenter.segment`; no model file is required, OCR fallback remains the default, and the current repository does not yet include a `QuestionRegionDetector` model artifact. When a model is bundled, fast mode skips full-page OCR only when OCR-matched detector boxes look page-complete; otherwise the app runs full-page OCR and merges unmatched OCR regions with Core ML regions, avoiding the old "first detector hit wins" leak.
- Actual detector training and model selection: pending larger reviewed dataset whose audit reports `model_training_ready=true`. The current clean-shape bootstrap detector is useful only for hard-example mining: it finds recall signal at very low confidence but produces about 190-281 false positives per image before filtering, and post-process sweeps still cannot meet the release recall/precision/FP gate, so the next loop is review/merge/retrain rather than iOS promotion.

## Question-Set Evaluation

Box IoU is necessary but not sufficient. The product succeeds only when the final question set has the right questions, avoids over-merging, keeps source evidence, and preserves usable order/numbering.

Run a product-level comparison on a diagnostics bundle:

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

Current 6dca result:

- reference questions: 18
- production questions: 9
- one-to-one precision / recall / F1: 1.0 / 0.5 / 0.6667
- coverage recall: 0.6667
- candidate duplicate clusters: 0
- merge-suspect candidates: 4
- candidate source coverage: 1.0, from 3 unique source images

Interpretation: the current product set is clean and traceable, but it is too compressed. Some reference questions are intentionally or accidentally merged into longer product questions, while several reference items are still uncovered. This is the metric that should move when improving local detection, fallback pressure, crop extraction, or dedupe rules.

## Product Rule

For observation extraction:

1. Prefer deduped question crops.
2. In observation upload, transmit deduped rects instead of duplicate crop JPEGs.
3. Backend generates canonical crop images from the saved source frames.
4. If a keyframe has very few/low-confidence crops, add the full frame as fallback.
5. If many crops are semantically duplicate, keep the sharpest/largest-safe crop.
6. Never let one crop exists suppress full-frame fallback when crop coverage is suspicious.
