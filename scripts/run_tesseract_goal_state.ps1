param(
  [string]$RepoRoot = "C:\Users\jacks\OneDrive\Desktop\NeuralForge"
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

Set-Location $RepoRoot

Write-Host "TPN v1.7 goal-state manager demo..."
python -m neuralforge.tesseract.goal_state --demo

Write-Host "TPN v1.7 goal-state summary..."
python -m neuralforge.tesseract.goal_state --summary
