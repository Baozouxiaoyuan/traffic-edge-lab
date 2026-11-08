$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$modelDirectory = Join-Path $projectRoot "models\lane"
$modelPath = Join-Path $modelDirectory "culane_18.onnx"
$modelUrl = "https://storage.googleapis.com/ailia-models/ultra-fast-lane-detection/culane_18.onnx"
$expectedBytes = 178139214

New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null
Write-Host "Downloading the experimental UFLD CULane model..."
Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath

$actualBytes = (Get-Item -LiteralPath $modelPath).Length
if ($actualBytes -ne $expectedBytes) {
    throw "Incomplete model download: expected $expectedBytes bytes, received $actualBytes bytes."
}

Write-Host "Installed: $modelPath"
Write-Host "This vehicle-mounted forward-view model is an optional calibration assistant; fixed-CCTV human review remains mandatory."
