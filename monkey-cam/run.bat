@echo off
setlocal
cd /d "%~dp0"

rem Install marker: if the first install was interrupted, start over.
if not exist ".venv\installed.ok" (
  if exist ".venv" rmdir /s /q ".venv"
  echo [Monkey Cam] First run: installing dependencies, this takes a few minutes...
  where py >nul 2>nul
  if not errorlevel 1 (py -3 -m venv .venv) else (python -m venv .venv)
  if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Python 3 not found. Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^) and run again.
    pause
    exit /b 1
  )
  ".venv\Scripts\python.exe" --version
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Failed to install dependencies. Send a screenshot of this window.
    pause
    exit /b 1
  )
  ".venv\Scripts\python.exe" -c "import cv2, mediapipe, pyvirtualcam"
  if errorlevel 1 (
    echo.
    echo Dependencies are installed but do not load. Send a screenshot of this window.
    pause
    exit /b 1
  )
  echo ok> ".venv\installed.ok"
)

".venv\Scripts\python.exe" monkey_cam.py %*
rem Keep the window open on error so the message can be read.
if errorlevel 1 pause
