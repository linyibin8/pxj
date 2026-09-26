# Observation Fallback Tune

- images: 41
- replay candidates: 67
- represented GT keys: 11
- crop-only GT recall: 1.0

## Backend Current

- GT recall: 1.0
- fallback images: 7
- total VLM pixels vs full frames: 0.784856
- reasons: {'no_large_question_crop': 1, 'single_low_coverage_crop': 3, 'sparse_low_coverage_crops': 2, 'tiny_crop_coverage': 1}

## Legacy Before Backend-Tuned

- GT recall: 1.0
- fallback images: 14
- total VLM pixels vs full frames: 0.958488
- reasons: {'all_weak_crop_keys': 2, 'no_large_question_crop': 2, 'single_low_coverage_crop': 7, 'sparse_low_coverage_crops': 2, 'tiny_crop_coverage': 1}

## Legacy To Current Delta

- fallback removed images: 7
- fallback removed images with GT: 1
- fallback removed images with crop-missed GT: 0
- fallback removed no-GT review images: 6
- total VLM pixels: 0.958488 -> 0.784856

## Backend Current With Union-Area Coverage

- GT recall: 1.0
- fallback images: 7
- total VLM pixels vs full frames: 0.784856
- reasons: {'no_large_question_crop': 1, 'single_low_coverage_crop': 3, 'sparse_low_coverage_crops': 2, 'tiny_crop_coverage': 1}

## Recommendation

- policy: backend_current
- GT recall: 1.0
- fallback images: 7
- total VLM pixels vs full frames: 0.784856
- reasons: {'no_large_question_crop': 1, 'single_low_coverage_crop': 3, 'sparse_low_coverage_crops': 2, 'tiny_crop_coverage': 1}
