@echo off
setlocal
pushd "%~dp0"

color 03

echo ========================================

echo JULIE001 BOT LAUNCHER

echo ========================================

echo.

set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

where uv >nul 2>nul
if errorlevel 1 (
echo [1] uv was not found on PATH.
echo Install uv first, then re-run this launcher.
goto end
)

echo [1] Ensuring workspace setup, Truth Social runtime, and local FinBERT...
powershell -ExecutionPolicy Bypass -File "%~dp0setup_topstep2.ps1"
if errorlevel 1 goto end

if not exist "%~dp0.env" (
echo [info] Truth Social polling still needs TRUTHSOCIAL_* credentials in config_secrets.py or environment variables.
)

echo.

REM Step 2: Run the UI Monitor

echo ========================================

echo LAUNCHING JULIE UI MONITOR

echo ========================================

echo.

.\.venv\Scripts\python.exe launch_ui.py

:end
popd
pause
