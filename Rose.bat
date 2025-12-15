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
python -c "import pandas, numpy, requests, pytz, joblib, sklearn" 2>nul
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

    echo Installing pytz...
    pip install pytz
    echo.

    echo Installing joblib...
    pip install joblib
    echo.

    echo Installing scikit-learn...
    pip install scikit-learn
    echo.
) else (
    echo [3] All packages already installed
    echo.
)

REM Note: tkinter comes with Python, no need to install separately

REM Step 4: Launch the trading bot and GUI monitor
echo ========================================
echo LAUNCHING JULIE TRADING SYSTEM
echo ========================================
echo.
echo Starting GUI Monitor in separate window...
start "JULIE GUI Monitor" python gui_monitor.py
echo.
echo Starting Trading Bot...
echo ========================================
echo.

python julie001.py

pause