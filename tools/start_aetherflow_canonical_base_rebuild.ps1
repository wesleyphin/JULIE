param(
    [string]$SourcePath = "es_master_outrights.parquet",
    [string]$CanonicalPath = "artifacts/aetherflow_corrected_full_2011_2026/manifold_base_outrights_2011_2026.parquet",
    [string]$StagePath = "",
    [string]$LogPath = "",
    [string]$PythonPath = ".venv/Scripts/python.exe"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

function Resolve-RepoPath {
    param([string]$PathText)
    if ([string]::IsNullOrWhiteSpace($PathText)) {
        return $null
    }
    if ([System.IO.Path]::IsPathRooted($PathText)) {
        $candidate = $PathText
    } else {
        $candidate = Join-Path $root $PathText
    }
    return [System.IO.Path]::GetFullPath($candidate)
}

$python = Resolve-RepoPath $PythonPath
$source = Resolve-RepoPath $SourcePath
$canonical = Resolve-RepoPath $CanonicalPath
$canonicalDir = Split-Path -Parent $canonical
New-Item -ItemType Directory -Force -Path $canonicalDir | Out-Null

if ([string]::IsNullOrWhiteSpace($StagePath)) {
    $suffix = [System.IO.Path]::GetExtension($canonical)
    if ([string]::IsNullOrWhiteSpace($suffix)) {
        $suffix = ".parquet"
    }
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($canonical)
    $stageName = "$stem.stateful_build$suffix"
    $stage = Join-Path $canonicalDir $stageName
} else {
    $stage = Resolve-RepoPath $StagePath
}

if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $logDir = Join-Path $root "logs"
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $stamp = [DateTimeOffset]::UtcNow.ToString("yyyyMMdd_HHmmss")
    $log = Join-Path $logDir "aetherflow_canonical_base_rebuild_$stamp.log"
} else {
    $log = Resolve-RepoPath $LogPath
    $logDir = Split-Path -Parent $log
    if (-not [string]::IsNullOrWhiteSpace($logDir)) {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    }
}

$stageMeta = "$stage.meta.json"
$canonicalMeta = "$canonical.meta.json"

"[$([DateTimeOffset]::UtcNow.ToString('o'))] start canonical=$canonical stage=$stage source=$source" | Tee-Object -FilePath $log -Append | Out-Null

$args = @(
    "tools/build_manifold_base_cache.py",
    "--source", $source,
    "--output", $stage,
    "--overwrite"
)

& $python @args 2>&1 | Tee-Object -FilePath $log -Append
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    "[$([DateTimeOffset]::UtcNow.ToString('o'))] build failed exit=$exitCode" | Tee-Object -FilePath $log -Append | Out-Null
    exit $exitCode
}

Move-Item -LiteralPath $stage -Destination $canonical -Force
Move-Item -LiteralPath $stageMeta -Destination $canonicalMeta -Force
$meta = Get-Content -LiteralPath $canonicalMeta -Raw | ConvertFrom-Json
$meta.output_path = $canonical
if ($meta.PSObject.Properties.Name -contains "promoted_from") {
    $meta.promoted_from = $stage
} else {
    $meta | Add-Member -NotePropertyName promoted_from -NotePropertyValue $stage
}
if ($meta.PSObject.Properties.Name -contains "promoted_at") {
    $meta.promoted_at = [DateTimeOffset]::UtcNow.ToString("o")
} else {
    $meta | Add-Member -NotePropertyName promoted_at -NotePropertyValue ([DateTimeOffset]::UtcNow.ToString("o"))
}
$meta | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $canonicalMeta -Encoding UTF8
"[$([DateTimeOffset]::UtcNow.ToString('o'))] promote complete canonical=$canonical" | Tee-Object -FilePath $log -Append | Out-Null
