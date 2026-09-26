# Observation Fallback Tune

- images: 308
- replay candidates: 537
- represented GT keys: 0
- crop-only GT recall: None

## Backend Current

- GT recall: None
- fallback images: 97
- total VLM pixels vs full frames: 0.769505
- reasons: {'no_crops': 54, 'no_large_question_crop': 12, 'single_low_coverage_crop': 21, 'sparse_low_coverage_crops': 9, 'tiny_crop_coverage': 1}

## Legacy Before Backend-Tuned

- GT recall: None
- fallback images: 139
- total VLM pixels vs full frames: 0.912438
- reasons: {'all_weak_crop_keys': 18, 'no_crops': 54, 'no_large_question_crop': 9, 'single_low_coverage_crop': 38, 'sparse_low_coverage_crops': 13, 'tiny_crop_coverage': 7}

## Legacy To Current Delta

- fallback removed images: 42
- fallback removed images with GT: 0
- fallback removed images with crop-missed GT: 0
- fallback removed no-GT review images: 42
- total VLM pixels: 0.912438 -> 0.769505

## Backend Current With Union-Area Coverage

- GT recall: None
- fallback images: 97
- total VLM pixels vs full frames: 0.769505
- reasons: {'no_crops': 54, 'no_large_question_crop': 11, 'single_low_coverage_crop': 21, 'sparse_low_coverage_crops': 9, 'tiny_crop_coverage': 2}

## Recommendation

- policy: backend_current
- GT recall: None
- fallback images: 97
- total VLM pixels vs full frames: 0.769505
- reasons: {'no_crops': 54, 'no_large_question_crop': 12, 'single_low_coverage_crop': 21, 'sparse_low_coverage_crops': 9, 'tiny_crop_coverage': 1}
