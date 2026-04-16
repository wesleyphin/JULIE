$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$ArgumentList = @()
    )

    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        $renderedArgs = if ($ArgumentList.Count -gt 0) { " " + ($ArgumentList -join " ") } else { "" }
        throw "Command failed: $FilePath$renderedArgs"
    }
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    throw "uv was not found on PATH. Install uv first, then re-run this script."
}

Write-Host "[1/5] Ensuring Python 3.13 is available..."
Invoke-Checked -FilePath $uv.Source -ArgumentList @("python", "install", "3.13")

Write-Host ""
Write-Host "[2/5] Creating or refreshing .venv..."
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (Test-Path $python) {
    Write-Host ".venv already exists; keeping the current interpreter."
}
elseif (Test-Path (Join-Path $repoRoot ".venv")) {
    Invoke-Checked -FilePath $uv.Source -ArgumentList @("venv", ".venv", "--python", "3.13", "--clear")
}
else {
    Invoke-Checked -FilePath $uv.Source -ArgumentList @("venv", ".venv", "--python", "3.13")
}

if (-not (Test-Path $python)) {
    throw "Expected virtual environment interpreter was not created: $python"
}

Write-Host ""
Write-Host "[3/5] Installing root requirements..."
Invoke-Checked -FilePath $uv.Source -ArgumentList @("pip", "install", "--python", $python, "-r", "requirements.txt")

Write-Host ""
Write-Host "[4/5] Bootstrapping Truth Social runtime + local FinBERT..."
& $python -X utf8 "tools\bootstrap_truth_social_runtime.py"
if ($LASTEXITCODE -ne 0) {
    throw "Truth Social runtime bootstrap failed."
}

$smokeCode = @'
import importlib

modules = [
    "launch_ui",
    "julie_tkinter_ui",
    "async_market_stream",
    "backtest_mes_et",
    "julie001",
    "truth_social_engine",
    "services.sentiment_service",
]

for name in modules:
    importlib.import_module(name)

print("Import smoke checks passed.")
'@

Write-Host ""
Write-Host "[5/5] Running import smoke checks..."
$smokeCode | & $python -X utf8 -
if ($LASTEXITCODE -ne 0) {
    throw "Import smoke checks failed."
}

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate: .\.venv\Scripts\activate"
Write-Host "UI: .\.venv\Scripts\python.exe launch_ui.py"
Write-Host "Live bot: .\.venv\Scripts\python.exe julie001.py"
Write-Host "Backtest: .\.venv\Scripts\python.exe backtest_mes_et.py"
