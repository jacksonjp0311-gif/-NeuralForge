param(
  [string]$RepoRoot = "C:\Users\jacks\OneDrive\Desktop\NeuralForge"
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

Set-Location $RepoRoot

Write-Host "TPN v1.8 goal-aware cycle demo..."
python -m neuralforge.tesseract.goal_cycle --demo

Write-Host "TPN v1.8 goal-aware cycle report..."
Get-Content .\artifacts\tpn\goal_cycle_report_v1_8_latest.json -Raw
