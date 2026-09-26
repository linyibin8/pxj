# Observation Fallback Tune

- images: 41
- replay candidates: 67
- represented GT keys: 11
- crop-only GT recall: 1.0

## Backend Current

- GT recall: 1.0
- fallback images: 14
- total VLM pixels vs full frames: 0.958488
- reasons: {'all_weak_crop_keys': 2, 'no_large_question_crop': 2, 'single_low_coverage_crop': 7, 'sparse_low_coverage_crops': 2, 'tiny_crop_coverage': 1}

## Recommendation

- policy: s0p30__p0p25__t0p12__m0p12__n0p40__w0__wa0p40__wm0p20__l0
- GT recall: 1.0
- fallback images: 7
- total VLM pixels vs full frames: 0.784856
- reasons: {'no_large_question_crop': 1, 'single_low_coverage_crop': 3, 'sparse_low_coverage_crops': 2, 'tiny_crop_coverage': 1}
