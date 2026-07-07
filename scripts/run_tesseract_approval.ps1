param(
  [string]$RepoRoot = "C:\Users\jacks\OneDrive\Desktop\NeuralForge",
  [ValidateSet("", "approve", "reject")]
  [string]$Decision = "",
  [string]$HumanId = "",
  [string]$Reason = ""
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

Set-Location $RepoRoot

if ([string]::IsNullOrWhiteSpace($Decision)) {
  Write-Host "TPN v1.14 human approval ledger pending summary..."
  python -c "import sys; from neuralforge.tesseract.approval import main; sys.argv=['approval','--pending']; main()"
  Write-Host ""
  Write-Host "To record an explicit decision:"
  Write-Host ".\scripts\run_tesseract_approval.ps1 -Decision approve -HumanId `"James`""
  Write-Host ".\scripts\run_tesseract_approval.ps1 -Decision reject -HumanId `"James`""
  exit 0
}

if ([string]::IsNullOrWhiteSpace($HumanId)) {
  throw "HumanId is required when Decision is approve or reject."
}

python -c "import sys; from neuralforge.tesseract.approval import main; sys.argv=['approval','--decision','$Decision','--human-id','$HumanId','--reason','$Reason']; main()"
