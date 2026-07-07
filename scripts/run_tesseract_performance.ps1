param(
  [string]$RepoRoot = "C:\Users\jacks\OneDrive\Desktop\NeuralForge"
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

Set-Location $RepoRoot

Write-Host "TPN v1.11 performance telemetry governor demo..."
python -c "import sys; from neuralforge.tesseract.performance import main; sys.argv=['performance','--demo']; main()"

Write-Host "TPN v1.11 performance report..."
Get-Content .\artifacts\tpn\performance_report_v1_11_latest.json -Raw
