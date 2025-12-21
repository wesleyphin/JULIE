@echo off

color 03

echo ========================================

echo JULIE001 BOT LAUNCHER

echo ========================================

echo.

REM Step 1: Check if venv exists, if not create it

if not exist "venv" (

echo [1] Creating virtual environment...

python -m venv venv

echo.

) else (

echo [1] Virtual environment already exists

echo.
)


REM Step 2: Activate venv

echo [2] Activating venv...

call venv\Scripts\activate.bat

echo.

REM Step 3: Check if packages are installed

python -c "import pandas, numpy, requests, joblib, sklearn, rich, colorama, click, tkinter, PIL" 2>nul

if errorlevel 1 (

echo [3] Upgrading pip...

python -m pip install --upgrade pip

echo.

echo [4] Installing packages...

echo.

echo Installing pandas...

pip install pandas

echo.

echo Installing numpy...

pip install numpy

echo.

echo Installing requests...

pip install requests

echo.

echo Installing joblib...

pip install joblib

echo.

echo Installing scikit-learn...

pip install scikit-learn

echo.

echo Installing rich...

pip install rich

echo.

echo Installing colorama...

pip install colorama

echo.

echo Installing click...

pip install click

echo.

echo Installing tkinter...

pip install tk

echo.

echo Installing pillow...

pip install pillow

echo.

) else (

echo [3] All packages already installed

echo.

)

REM Step 4: Run the UI Monitor

echo ========================================

echo LAUNCHING JULIE UI MONITOR

echo ========================================

echo.

python launch_ui.py

pause
