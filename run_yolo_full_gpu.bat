@echo off
REM Full YOLO pipeline: train yolo11m -> conf sweep -> optional yolo11l / yolo11x if accuracy low.
REM Close Cursor first to avoid WinError 32 on large installs; use NVIDIA driver + CUDA PyTorch.

cd /d "%~dp0"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU OK:', torch.cuda.get_device_name(0))"
if errorlevel 1 (
  echo Fix: install CUDA build of PyTorch and NVIDIA driver, then run this script again.
  pause
  exit /b 1
)

python run_yolo_gpu_pipeline.py --device auto --require-gpu --workers 8
echo.
echo Summary: runs\yolo_pipeline_summary.json
pause
