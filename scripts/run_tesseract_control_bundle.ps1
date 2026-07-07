param(
  [string]$RepoRoot = "C:\Users\jacks\OneDrive\Desktop\NeuralForge"
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

Set-Location $RepoRoot

Write-Host "TPN v1.13 compressed control bundle demo..."
python -c "import sys; from neuralforge.tesseract.control_bundle import main; sys.argv=['control_bundle','--demo']; main()"

Write-Host "TPN v1.13 compressed control bundle report..."
Get-Content .\artifacts\tpn\control_bundle_report_v1_13_latest.json -Raw
