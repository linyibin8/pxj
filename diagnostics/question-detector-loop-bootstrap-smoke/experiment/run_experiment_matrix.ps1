$ErrorActionPreference = 'Stop'
# Generated 2026-06-30T04:11:30.528612+00:00
# Runs: 1

C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe D:\AI\pxj\scripts\question_detector_train.py --dataset diagnostics\question-detector-bootstrap-kpai-smoke --out diagnostics\question-detector-loop-bootstrap-smoke\experiment\runs\question-detector-bootstrap-kpai-smoke__yolo11n__img416__seed42__ep1 --model yolo11n.pt --name question-block --epochs 1 --imgsz 416 --batch 4 --workers 0 --seed 42 --conf 0.25 --eval-thresholds '0.5,0.75' --min-eval-recall 0.95 --min-eval-precision 0.9 --max-eval-fp-per-image 0.05 --eval-score-thresholds '0.05,0.10,0.20,0.25,0.35,0.50' --clean --device cpu --predict-test --allow-unready --allow-failed-eval

# Select best candidate after all runs finish.
C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe D:\AI\pxj\scripts\question_detector_select.py diagnostics\question-detector-loop-bootstrap-smoke\experiment\runs\question-detector-bootstrap-kpai-smoke__yolo11n__img416__seed42__ep1 --out diagnostics\question-detector-loop-bootstrap-smoke\experiment\selection
