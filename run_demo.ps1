$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$RunDirectory = Join-Path $ProjectRoot ("runs\demo-" + [guid]::NewGuid().ToString('N').Substring(0,12))

& $Python -m traffic_edge.cli generate-demo --output (Join-Path $ProjectRoot "data\demo.mp4")
if ($LASTEXITCODE -ne 0) { throw "Demo video generation failed." }
& $Python -m traffic_edge.cli run `
  --input (Join-Path $ProjectRoot "data\demo.mp4") `
  --output-dir $RunDirectory `
  --config (Join-Path $ProjectRoot "config\demo.json") `
  --backend demo
if ($LASTEXITCODE -ne 0) { throw "Demo analysis failed." }
& $Python -m traffic_edge.cli evaluate `
  --events (Join-Path $RunDirectory "events.jsonl") `
  --ground-truth (Join-Path $ProjectRoot "config\demo_ground_truth.json") `
  --tolerance 0.5 `
  --output (Join-Path $RunDirectory "evaluation.json")
if ($LASTEXITCODE -ne 0) { throw "Demo evaluation failed." }
