@echo off
TITLE JULIE TRADING SYSTEM // INSTITUTIONAL ENGINE
COLOR 03
CLS

:: ===========================================================================
::                           JULIE ALGORITHMIC SYSTEM
:: ===========================================================================

echo.
echo      JJJJJJJJJ   UU     UU   LL          IIIIII    EEEEEEE
echo         JJ       UU     UU   LL            II      EE
echo         JJ       UU     UU   LL            II      EEEEE
echo      J  JJ       UU     UU   LL            II      EE
echo       JJJJ        UUUUUUU    LLLLLLL     IIIIII    EEEEEEE
echo.
echo ===========================================================================
echo                 INSTITUTIONAL EXECUTION ENGINE v2.0
echo ===========================================================================
echo.

:: ---------------------------------------------------------------------------
:: STEP 1: ENVIRONMENT SECURITY CHECK
:: ---------------------------------------------------------------------------
echo [SYSTEM] Checking Secure Environment (VENV)...

if not exist "venv" (
    echo [WARN]  Virtual Environment not found.
    echo [INIT]  Building containment field...
    python -m venv venv
)

:: Activate the Venv
call venv\Scripts\activate.bat
echo [OK]    Environment Activated.
echo.

:: ---------------------------------------------------------------------------
:: STEP 2: FORCE DEPENDENCY UPDATE
:: ---------------------------------------------------------------------------
echo [SCAN]  Validating Strategy Drivers...

:: We update PIP first to prevent installation errors
python -m pip install --upgrade pip >nul

:: Group 1: The New v2.0 Async Drivers (Added signalrcore_async here)
echo [1/4]   Installing Stream Engine (Async IO)...
pip install aiohttp signalrcore signalrcore_async websockets websocket-client >nul

:: Group 2: Market Data Feeds
echo [2/4]   Installing Data Feeds (YFinance, Timezones)...
pip install yfinance pytz tzdata requests >nul

:: Group 3: Math & ML
echo [3/4]   Installing Neural Engine (SciPy, Sklearn)...
pip install numpy pandas scipy joblib scikit-learn >nul

:: Group 4: UI Components
echo [4/4]   Installing Dashboard Graphics...
pip install rich colorama click Pillow >nul

echo [DONE]  All drivers operational.
echo.

:: ---------------------------------------------------------------------------
:: STEP 3: EXECUTION
:: ---------------------------------------------------------------------------
echo ===========================================================================
echo                       SYSTEM READY - LAUNCHING UI
echo ===========================================================================
echo.

if exist launch_ui.py (
    python launch_ui.py
) else (
    python julie_tkinter_ui.py
)

:: ---------------------------------------------------------------------------
:: STEP 4: CRASH REPORT
:: ---------------------------------------------------------------------------
if %ERRORLEVEL% NEQ 0 (
    color 0C
    echo.
    echo ===========================================================================
    echo [CRITICAL ERROR] SYSTEM HALTED
    echo ===========================================================================
    echo.
    echo The bot crashed. Please read the error message above.
    echo.
    pause
)

pause
