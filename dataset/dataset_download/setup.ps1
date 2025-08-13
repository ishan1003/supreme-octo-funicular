$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

# 1) Load MSVC if available (VS 2022)
$vc = @(
  "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
  "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
  "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
  "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($vc) {
  Write-Host "[INFO] Using MSVC env:" $vc
  cmd /c "`"$vc`" && set" | ForEach-Object {
    $n,$v = $_ -split '=', 2
    if ($n -and $v) { [Environment]::SetEnvironmentVariable($n, $v, "Process") }
  }
} else {
  Write-Host "[WARN] vcvars64.bat not found; C++ build may be skipped."
}

# 2) VCPKG root (session + persist for user)
$env:VCPKG_ROOT = "D:\vcpkg"
[Environment]::SetEnvironmentVariable("VCPKG_ROOT","D:\vcpkg","User") | Out-Null

# 3) Dataset locations
$datasetDir  = "build\datasets"
$zipPath     = Join-Path $datasetDir "MFCAD_dataset.zip"
$extractDir  = Join-Path $datasetDir "MFCAD++_dataset"

New-Item -ItemType Directory -Force -Path $datasetDir | Out-Null

# If ZIP is missing, ask user to download (Cloudflare blocks scripted downloads)
if (-not (Test-Path $zipPath)) {
  Write-Host "---------------------------------------------"
  Write-Host "Dataset ZIP not found."
  Write-Host "Download manually from:"
  Write-Host "  https://pure.qub.ac.uk/files/278385243/MFCAD_dataset.zip"
  Write-Host "Place the file at:"
  Write-Host "  $zipPath"
  Write-Host "---------------------------------------------"
  Read-Host "Press ENTER after placing the file"
}

if (-not (Test-Path $zipPath)) {
  throw "MFCAD_dataset.zip still not found. Aborting."
}

# 4) Extract if not already extracted
if (-not (Test-Path $extractDir)) {
  Write-Host "[INFO] Extracting dataset..."
  Expand-Archive -Path $zipPath -DestinationPath $datasetDir -Force
  Write-Host "✅ Extracted to $extractDir"
} else {
  Write-Host "[INFO] Dataset already extracted."
}

