param(
  [string]$RepoRoot = "C:\Users\jacks\OneDrive\Desktop\NeuralForge"
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

Set-Location $RepoRoot

Write-Host "TPN v1.16 external source intake governor..."
python -c "import sys; from neuralforge.tesseract.source_intake import main; sys.argv=['source_intake']; main()"

Write-Host ""
Write-Host "Full source intake receipts remain under .\artifacts\tpn\"
