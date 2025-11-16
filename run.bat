@echo off
REM ==========================================================
REM  Run text detection on video (GPU version, Python 3.11)
REM  Supports memory smoothing (--smooth_hold) and tracker (--use_tracker)
REM ==========================================================

echo [INFO] Checking required Python 3.11 packages...
py -3.11 -c "import cv2, easyocr, torch, numpy; print('[OK] All required packages are installed.')" || (
    echo [ERROR] Missing packages for Python 3.11.
    echo Please install them manually once by running:
    echo.
    echo py -3.11 -m pip install numpy opencv-python-headless easyocr torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo.
    pause
    exit /b
)

echo.
echo -----------------------------------------------------------
echo Parameter: every_n
echo This defines how often frames will be analyzed.
echo Example: 1 = every frame, 5 = every 5th frame, 10 = every 10th frame.
echo -----------------------------------------------------------
set /p EVERY_N=Enter frame step (every_n, default=1): 

if "%EVERY_N%"=="" set EVERY_N=1

echo.
echo -----------------------------------------------------------
echo Parameter: smooth_hold
echo Number of frames to keep text visible when OCR temporarily loses it.
echo Recommended: 5-10
echo -----------------------------------------------------------
set /p SMOOTH_HOLD=Enter smooth_hold (default=7): 

if "%SMOOTH_HOLD%"=="" set SMOOTH_HOLD=7

echo.
echo -----------------------------------------------------------
echo Do you want to enable the tracker?
echo y = enable tracker with IOU smoothing
echo n = simple memory only
echo -----------------------------------------------------------
set /p USE_TRACKER=Enable tracker? (y/n, default=n): 

if /I "%USE_TRACKER%"=="Y" (
    set TRACKER_FLAG=--use_tracker
) else (
    set TRACKER_FLAG=
)

echo [INFO] Starting video text detection...
py -3.11 detect_video_text.py ^
  --video_in input\video.mp4 ^
  --video_out output\annotated.mp4 ^
  --csv_out output\detections.csv ^
  --langs en ru ^
  --gpu ^
  --write_csv ^
  --draw_text ^
  --every_n %EVERY_N% ^
  --conf 0.2 ^
  --smooth_hold %SMOOTH_HOLD% ^
  %TRACKER_FLAG% ^
  --iou 0.3 ^
  --ema 0.5

echo.
echo ===========================================
echo DONE!
echo Results are saved in the output folder:
echo  - annotated.mp4
echo  - detections.csv
echo ===========================================
pause
