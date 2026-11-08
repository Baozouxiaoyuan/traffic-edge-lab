param(
  [Parameter(Mandatory = $true)]
  [string]$InputVideo,
  [string]$OutputDirectory = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

if (-not (Test-Path -LiteralPath $InputVideo -PathType Leaf)) {
  throw "Input video not found: $InputVideo"
}
if (-not $OutputDirectory) {
  $OutputDirectory = Join-Path $ProjectRoot ("runs\traffic-" + [guid]::NewGuid().ToString('N').Substring(0,12))
}

& $Python -m traffic_edge.cli run `
  --input $InputVideo `
  --output-dir $OutputDirectory `
  --config (Join-Path $ProjectRoot "config\real_video_template.json") `
  --backend yolo
if ($LASTEXITCODE -ne 0) { throw "Traffic analysis failed." }
