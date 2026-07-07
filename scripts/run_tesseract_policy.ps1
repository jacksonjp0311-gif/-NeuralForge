param(
  [string]$RepoRoot = "C:\Users\jacks\OneDrive\Desktop\NeuralForge"
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

Set-Location $RepoRoot

Write-Host "TPN v1.10 execution policy governor demo..."
python -c "import sys; from neuralforge.tesseract.policy import main; sys.argv=['policy','--demo']; main()"

Write-Host "TPN v1.10 policy report..."
Get-Content .\artifacts\tpn\policy_report_v1_10_latest.json -Raw
