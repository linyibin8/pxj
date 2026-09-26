# Observation Fallback Tune

- images: 89
- replay candidates: 262
- represented GT keys: 16
- crop-only GT recall: 1.0

## Backend Current

- GT recall: 1.0
- fallback images: 34
- total VLM pixels vs full frames: 0.819493
- reasons: {'no_crops': 18, 'no_large_question_crop': 3, 'single_low_coverage_crop': 10, 'sparse_low_coverage_crops': 2, 'tiny_crop_coverage': 1}

## Legacy Before Backend-Tuned

- GT recall: 1.0
- fallback images: 57
- total VLM pixels vs full frames: 1.07142
- reasons: {'all_weak_crop_keys': 4, 'no_crops': 18, 'no_large_question_crop': 1, 'single_low_coverage_crop': 24, 'sparse_low_coverage_crops': 6, 'tiny_crop_coverage': 4}

## Legacy To Current Delta

- fallback removed images: 23
- fallback removed images with GT: 4
- fallback removed images with crop-missed GT: 0
- fallback removed no-GT review images: 19
- total VLM pixels: 1.07142 -> 0.819493

## Backend Current With Union-Area Coverage

- GT recall: 1.0
- fallback images: 34
- total VLM pixels vs full frames: 0.819493
- reasons: {'no_crops': 18, 'no_large_question_crop': 3, 'single_low_coverage_crop': 10, 'sparse_low_coverage_crops': 2, 'tiny_crop_coverage': 1}

## Recommendation

- policy: backend_current
- GT recall: 1.0
- fallback images: 34
- total VLM pixels vs full frames: 0.819493
- reasons: {'no_crops': 18, 'no_large_question_crop': 3, 'single_low_coverage_crop': 10, 'sparse_low_coverage_crops': 2, 'tiny_crop_coverage': 1}
