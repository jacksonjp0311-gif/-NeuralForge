param(
  [string]$RepoRoot = "C:\Users\jacks\OneDrive\Desktop\NeuralForge"
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

Set-Location $RepoRoot

Write-Host "TPN v1.13.1 compressed control bundle summary..."
python -c "import sys; from neuralforge.tesseract.control_bundle import main; sys.argv=['control_bundle','--demo','--summary']; main()"

Write-Host ""
Write-Host "Full receipt files remain under .\artifacts\tpn\"
