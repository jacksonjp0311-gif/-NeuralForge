param(
  [string]$RepoRoot = "C:\Users\jacks\OneDrive\Desktop\NeuralForge"
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

Set-Location $RepoRoot

Write-Host "TPN v1.12 stairway compression governor demo..."
python -c "import sys; from neuralforge.tesseract.stairway import main; sys.argv=['stairway','--demo']; main()"

Write-Host "TPN v1.12 stairway report..."
Get-Content .\artifacts\tpn\stairway_report_v1_12_latest.json -Raw
