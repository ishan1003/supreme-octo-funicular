param(
  [Parameter(Mandatory = $false)]
  [string] $StepPath
)

# --- ask if not provided ---
if (-not $StepPath) {
  $StepPath = Read-Host 'Enter full path to STEP file (.stp/.step)'
}
$StepPath = $StepPath.Trim('"')

if (-not (Test-Path -LiteralPath $StepPath)) {
  Write-Error "File not found: $StepPath"
  exit 1
}

# --- repo paths ---
$Root    = $PSScriptRoot              # folder containing this script
$Convert = Join-Path $Root 'dataset\dataset_generation\convert_step_to_graph.py'
$Predict = Join-Path $Root 'predict_from_pkl.py'
$Ckpt    = Join-Path $Root 'checkpoints\gcn_facecls.pt'

foreach ($p in @($Convert,$Predict,$Ckpt)) {
  if (-not (Test-Path -LiteralPath $p)) {
    Write-Error "Missing file: $p"
    exit 1
  }
}

# --- temp PKL path based on input name ---
$Stem = [System.IO.Path]::GetFileNameWithoutExtension($StepPath)
$Pkl  = Join-Path $env:TEMP ("{0}_graph.pkl" -f $Stem)

# --- locate conda (no need to 'activate'; we use `conda run`) ---
$Conda = (Get-Command conda -ErrorAction SilentlyContinue | Select-Object -First 1).Source
if (-not $Conda) {
  $candidates = @(
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:ProgramData\anaconda3\Scripts\conda.exe",
    "$env:ProgramData\miniconda3\Scripts\conda.exe"
  )
  $Conda = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $Conda) {
  Write-Error "Could not find conda. Start an 'Anaconda Prompt' or edit the script to point to conda.exe."
  exit 1
}

# --- STEP -> PKL (dataset_generation) ---
Write-Host "`n=== STEP -> PKL (dataset_generation) ===" -ForegroundColor Cyan
& $Conda run -n dataset_generation python $Convert $StepPath $Pkl
if ($LASTEXITCODE -ne 0) {
  Write-Error "Conversion failed with exit code $LASTEXITCODE"
  exit $LASTEXITCODE
}

# --- PKL -> predictions (gnn_training) ---
Write-Host "`n=== PKL -> Predictions (gnn_training) ===" -ForegroundColor Cyan
& $Conda run -n gnn_training python $Predict --pkl $Pkl --ckpt $Ckpt --device auto
if ($LASTEXITCODE -ne 0) {
  Write-Error "Prediction failed with exit code $LASTEXITCODE"
  exit $LASTEXITCODE
}

Write-Host "`nDone. PKL: $Pkl" -ForegroundColor Green
