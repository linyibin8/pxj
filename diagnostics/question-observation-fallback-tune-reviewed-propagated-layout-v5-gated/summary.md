# Observation Fallback Tune

- images: 89
- replay candidates: 262
- represented GT keys: 16
- crop-only GT recall: 1.0

## Backend Current

- GT recall: 1.0
- fallback images: 57
- total VLM pixels vs full frames: 1.07142
- reasons: {'all_weak_crop_keys': 4, 'no_crops': 18, 'no_large_question_crop': 1, 'single_low_coverage_crop': 24, 'sparse_low_coverage_crops': 6, 'tiny_crop_coverage': 4}

## Backend Current With Union-Area Coverage

- GT recall: 1.0
- fallback images: 57
- total VLM pixels vs full frames: 1.07142
- reasons: {'all_weak_crop_keys': 4, 'no_crops': 18, 'no_large_question_crop': 1, 'single_low_coverage_crop': 24, 'sparse_low_coverage_crops': 6, 'tiny_crop_coverage': 4}

## Recommendation

- policy: sum__s0p30__p0p25__t0p12__m0p12__n0p40__w0__wa0p40__wm0p20__l0
- GT recall: 1.0
- fallback images: 34
- total VLM pixels vs full frames: 0.819493
- reasons: {'no_crops': 18, 'no_large_question_crop': 3, 'single_low_coverage_crop': 10, 'sparse_low_coverage_crops': 2, 'tiny_crop_coverage': 1}
