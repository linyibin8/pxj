# Question Observation Stream Summary - 2026-07-05

## Goal

Track every distinct homework/exam question that appears in a student-writing video stream, then save and structure the questions with high correctness, low latency, no duplicates, and minimal image transfer.

## Selected Approach

Use an iOS-first observation pipeline:

- iOS samples only useful keyframes from the live video stream.
- iOS runs fast Vision/OCR/layout segmentation and emits question rectangles, frame fingerprints, OCR token signatures, and risk signals.
- iOS uploads canonical keyframes plus small rect-only JSON manifests instead of crop JPEGs.
- Backend crops canonical question images from the saved keyframes, consolidates by fingerprint/simhash/OCR tokens, and falls back to section/full-frame analysis only for risky or empty frames.
- GPU VLM grounding models are kept as offline teachers/review tools, not the live production path.

## Schemes Tried

1. Full-frame / stop-time replay
   - Tried replaying the whole `D:\AI\fenti\pimage_questions` directory with measured layout.
   - The full 3509-image run was too slow for an interactive stop-time path and was stopped.
   - Conclusion: live candidate caching during capture is required; do not rescan every frame at the end.

2. YOLO / Core ML detector
   - Existing bootstrap detector evidence is not release-ready.
   - Current fenti-pimage YOLO eval at IoU 0.50: precision 0.1154, recall 0.0769, F1 0.0923, fp/image 0.6053, gate failed.
   - Conclusion: keep the Core ML hook, but do not ship this detector as the primary route until there is a larger reviewed dataset and a passing model.

3. LocateAnything / VLM visual grounding
   - NVIDIA's official LocateAnything page describes it as a unified VLM grounding/detection model using Parallel Box Decoding for fast, high-quality localization, including document/OCR/layout tasks.
   - It is valuable as an offline teacher or review assistant, but live production use is less attractive because it is GPU-heavy, model/license dependent, and still requires sending image content.
   - Conclusion: use it as a future teacher path, not the live iOS stream path.

4. iOS Vision/OCR layout + rect-only manifests + backend dedupe/fallback
   - Selected for production because it moves cheap work onto iOS, avoids duplicate crop uploads, and keeps backend/VLM work proportional to unique risky regions.

## Pimage Verification

Dataset inventory:

- Source directory: `D:\AI\fenti\pimage_questions`
- Images in manifest: 3509 kept images across 26+ sources.
- Balanced sample used for fast iteration: 308 images.

Replay on `diagnostics\pimage-questions-balanced-sample\images`:

- Images: 308
- Raw boxes: 537
- Cross-frame dedup boxes: 447
- Images with boxes: 254
- Layout measurement: mean 29.584 ms, median 30.232 ms, p95 54.930 ms, max 86.421 ms.

Network / VLM proxy:

- Full-frame keyframes only: 54,186,965 bytes.
- Adding deduped crop JPEGs: +25,120,772 bytes, total 1.4636x full-frame payload.
- Adding rect-only manifests: +111,765 bytes, total 1.0021x full-frame payload.
- Deduped crop VLM pixels vs full-frame VLM pixels: 0.3813x.

Fallback and section sender:

- Backend current fallback policy: 97 / 308 fallback images, total VLM pixels 0.769505x full-frame.
- Section sender on this no-GT sample: 5 / 308 fallback images, total VLM pixels 0.803472x full-frame.
- Existing reviewed strategy evidence remains stronger for release gates: risk-section/top1 strategy keeps reviewed recall at 1.0 with total VLM pixels around 0.762x to 0.810x, depending on cohort.

Dedup tuning note:

- The no-GT pimage sample produced an over-aggressive `swift_signature` recommendation selecting only 4 / 537 boxes.
- Rejected for production because there is no ground truth in this sample; production should stay with the reviewed, text-guarded dedupe strategy.

## Validation

- `python scripts\question_fallback_policy_smoke.py`: passed.
- `python -m compileall -q backend\app backend\scripts scripts`: passed.
- macOS generic iOS build: succeeded.
- TestFlight publish: succeeded.

## TestFlight

- Bundle: `com.linyibin8.pxj`
- App Store Connect app id: `6785257642`
- Uploaded build: `202607052330`
- Build id / delivery UUID: `00d37075-94ad-40f5-9957-52fd171d9bf2`
- Processing state: `VALID`
- Export compliance: `False`
- Internal group: `PXJ Internal`
- Testers currently in group: `269123786@qq.com`, `3972104921@qq.com`, `643014114@qq.com`, `linyibin8@qq.com`
- Status still reports expected missing testers: `2811903134@qq.com`, `3927141796@qq.com`

## Notes

- Dell GPU host `dell@100.64.0.5` was checked: two RTX 4090 GPUs were present but already memory-occupied. No services were stopped because the selected production path does not require GPU training/inference.
- Local `ContentView.swift` and the published Mac copy have identical Swift text content; the remaining byte hash difference is newline format only.
- The Release archive produced non-blocking warnings: one existing Swift always-true guard warning and app icon assignment warnings.

## Official References

- NVIDIA LocateAnything project page: https://research.nvidia.com/labs/lpr/locate-anything/
- Hugging Face model entry: https://huggingface.co/nvidia/LocateAnything-3B
