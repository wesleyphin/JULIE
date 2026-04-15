@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

set "FILTERLESS_SETUP_NEEDED=0"
if not exist ".venv\Scripts\python.exe" set "FILTERLESS_SETUP_NEEDED=1"
if "!FILTERLESS_SETUP_NEEDED!"=="0" call :check_sentiment_runtime "%~dp0.venv\Scripts\python.exe"

if "!FILTERLESS_SETUP_NEEDED!"=="1" if exist "%~dp0setup_topstep2.ps1" (
    echo Bootstrapping workspace dependencies and sentiment runtime...
    powershell -ExecutionPolicy Bypass -File "%~dp0setup_topstep2.ps1"
)

if defined FILTERLESS_PYTHON (
    call :try_set_python "%FILTERLESS_PYTHON%"
    if defined FILTERLESS_LAUNCH_PYTHON goto launch_launcher
)

call :try_set_python "%~dp0.venv\Scripts\python.exe"
if defined FILTERLESS_LAUNCH_PYTHON goto launch_launcher

if defined FILTERLESS_CHILD_PYTHON (
    call :try_set_python "%FILTERLESS_CHILD_PYTHON%"
    if defined FILTERLESS_LAUNCH_PYTHON goto launch_launcher
)

where python >nul 2>nul
if %errorlevel%==0 (
    for /f "delims=" %%I in ('where python') do if not defined FILTERLESS_LAUNCH_PYTHON call :try_set_python "%%I"
    if defined FILTERLESS_LAUNCH_PYTHON goto launch_launcher
)

for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do if not defined FILTERLESS_LAUNCH_PYTHON call :try_set_python "%%I"

if not defined FILTERLESS_LAUNCH_PYTHON call :try_set_python "%~dp0venv\Scripts\python.exe"

:launch_launcher
if defined FILTERLESS_LAUNCH_PYTHON (
    "%FILTERLESS_LAUNCH_PYTHON%" "%~dp0launch_filterless_workspace.py" %*
    set "FILTERLESS_EXIT=!ERRORLEVEL!"
    if not "!FILTERLESS_EXIT!"=="0" (
        echo.
        echo LaunchFilterlessWorkspace failed with exit code !FILTERLESS_EXIT!.
        if exist "%~dp0logs\filterless_workspace_launcher.err.log" (
            echo See logs\filterless_workspace_launcher.err.log for details.
        )
        pause
    )
    exit /b !FILTERLESS_EXIT!
)

echo Could not find a usable Python interpreter for LaunchFilterlessWorkspace.bat
exit /b 1

:try_set_python
if defined FILTERLESS_LAUNCH_PYTHON goto :eof
set "FILTERLESS_CANDIDATE=%~1"
if not defined FILTERLESS_CANDIDATE goto :eof
if not exist "%FILTERLESS_CANDIDATE%" goto :eof
"%FILTERLESS_CANDIDATE%" -V >nul 2>nul
if errorlevel 1 goto :eof
set "FILTERLESS_LAUNCH_PYTHON=%FILTERLESS_CANDIDATE%"
goto :eof

:check_sentiment_runtime
set "FILTERLESS_RUNTIME_PY=%~1"
if not defined FILTERLESS_RUNTIME_PY (
    set "FILTERLESS_SETUP_NEEDED=1"
    goto :eof
)
if not exist "%FILTERLESS_RUNTIME_PY%" (
    set "FILTERLESS_SETUP_NEEDED=1"
    goto :eof
)
if not exist "%~dp0models\finbert\config.json" (
    set "FILTERLESS_SETUP_NEEDED=1"
    goto :eof
)
"%FILTERLESS_RUNTIME_PY%" -c "import importlib.util, sys; mods=['transformers','torch','accelerate']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; sys.exit(1 if missing else 0)" >nul 2>nul
if errorlevel 1 set "FILTERLESS_SETUP_NEEDED=1"
goto :eof
