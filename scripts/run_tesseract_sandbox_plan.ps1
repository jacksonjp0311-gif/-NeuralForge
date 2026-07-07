param(
  [string]$RepoRoot = "C:\Users\jacks\OneDrive\Desktop\NeuralForge",
  [switch]$DemoApproved
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

Set-Location $RepoRoot

Write-Host "TPN v1.15 sandboxed patch plan receipt..."
if ($DemoApproved.IsPresent) {
  python -c "import sys; from neuralforge.tesseract.sandbox_plan import main; sys.argv=['sandbox_plan','--demo-approved']; main()"
} else {
  python -c "import sys; from neuralforge.tesseract.sandbox_plan import main; sys.argv=['sandbox_plan']; main()"
}

Write-Host ""
Write-Host "Full plan receipts remain under .\artifacts\tpn\"
