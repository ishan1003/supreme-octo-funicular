# run as:
# powershell -ExecutionPolicy Bypass -File .\run_predict.ps1 "D:\path\part.step"
# optional: -Visualize -Alpha 0.95 -Wire

param(
  [Parameter(Mandatory = $false, Position = 0)]
  [string] $StepPath,

  [string] $CkptPath,
  [string] $OutDir,

  [switch] $Visualize,
  [double] $Alpha = 1.0,
  [switch] $Wire
)

if (-not $StepPath) { $StepPath = Read-Host 'Enter full path to STEP file (.stp/.step)' }
$StepPath = $StepPath.Trim('"')
if (-not (Test-Path -LiteralPath $StepPath)) { Write-Error "File not found: $StepPath"; exit 1 }

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $CkptPath -or $CkptPath -eq '') { $CkptPath = Join-Path $ScriptRoot 'checkpoints\gcn_facecls.pt' }
if (-not $OutDir -or $OutDir -eq '') { $OutDir = $env:TEMP }

$Convert = Join-Path $ScriptRoot 'dataset\dataset_generation\convert_step_to_graph.py'
$Predict = Join-Path $ScriptRoot 'predict_from_pkl.py'
$Viz     = Join-Path $ScriptRoot 'visualize_step_with_preds.py'

foreach ($p in @($Convert,$Predict,$CkptPath,$Viz)) {
  if (-not (Test-Path -LiteralPath $p)) { Write-Error "Missing file: $p"; exit 1 }
}

$stem       = [IO.Path]::GetFileNameWithoutExtension($StepPath)
$Pkl        = Join-Path $OutDir ("{0}_graph.pkl" -f $stem)
$Preds      = Join-Path $OutDir ("{0}_preds.json" -f $stem)
$ColoredStep= Join-Path $OutDir ("{0}_colored.step" -f $stem)

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
if (-not $Conda) { Write-Error "Could not find conda."; exit 1 }

Write-Host "`n=== STEP -> PKL (dataset_generation) ===" -ForegroundColor Cyan
& $Conda run -n dataset_generation python -- "$Convert" "$StepPath" "$Pkl"
if ($LASTEXITCODE -ne 0) { Write-Error "Conversion failed ($LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Host "`n=== PKL -> Predictions (gnn_training) ===" -ForegroundColor Cyan
& $Conda run -n gnn_training python -- "$Predict" --pkl "$Pkl" --ckpt "$CkptPath" --device auto --out_json "$Preds"
if ($LASTEXITCODE -ne 0) { Write-Error "Prediction failed ($LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Host "`n=== Write colored STEP (dataset_generation) ===" -ForegroundColor Cyan
$alphaArg = "{0:N2}" -f $Alpha
$wireArg  = if ($Wire.IsPresent) { "--wire" } else { "" }

# headless export always; add --show only if -Visualize is set
$extra = @()
if ($Visualize) { $extra += "--show"; $extra += "--alpha"; $extra += $alphaArg; if ($Wire.IsPresent){$extra += "--wire"} }

& $Conda run -n dataset_generation python -- "$Viz" `
  --step "$StepPath" `
  --preds "$Preds" `
  --write_step "$ColoredStep" `
  @extra
if ($LASTEXITCODE -ne 0) { Write-Warning "Visualizer/export exited with code $LASTEXITCODE" }

Write-Host "`nArtifacts:" -ForegroundColor Green
Write-Host "  PKL         : $Pkl"
Write-Host "  Predictions : $Preds"
Write-Host "  Colored STEP: $ColoredStep"
Write-Host "`nDone." -ForegroundColor Green
