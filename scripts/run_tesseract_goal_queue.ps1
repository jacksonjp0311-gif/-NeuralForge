param(
  [string]$RepoRoot = "C:\Users\jacks\OneDrive\Desktop\NeuralForge"
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

Set-Location $RepoRoot

Write-Host "TPN v1.9 guarded multi-goal queue demo..."
python -c "import sys; from neuralforge.tesseract.goal_queue import main; sys.argv=['goal_queue','--demo']; main()"

Write-Host "TPN v1.9 guarded queue report..."
Get-Content .\artifacts\tpn\goal_queue_report_v1_9_latest.json -Raw
