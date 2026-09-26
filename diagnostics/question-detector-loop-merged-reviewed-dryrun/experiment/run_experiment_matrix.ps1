$ErrorActionPreference = 'Stop'
# Generated 2026-06-30T04:25:23.519035+00:00
# Runs: 1

C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe D:\AI\pxj\scripts\question_detector_train.py --dataset diagnostics\question-detector-loop-merged-reviewed-dryrun\dataset --out diagnostics\question-detector-loop-merged-reviewed-dryrun\experiment\runs\dataset__yolo11n__img416__seed42__ep1 --model yolo11n.pt --name question-block --epochs 1 --imgsz 416 --batch -1 --workers 4 --seed 42 --conf 0.25 --eval-thresholds '0.5,0.75' --min-eval-recall 0.95 --min-eval-precision 0.9 --max-eval-fp-per-image 0.05 --eval-score-thresholds '0.05,0.10,0.20,0.25,0.35,0.50' --clean --predict-test --allow-unready --allow-failed-eval --dry-run
