
param(
  [string]$RepoRoot = "C:\Users\jacks\OneDrive\Desktop\NeuralForge",
  [int]$Port = 8767
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

Set-Location $RepoRoot
Write-Host "Starting Tesseract Jarvis runtime in foreground."
Write-Host "Health: http://127.0.0.1:$Port/health"
Write-Host "Stop: Ctrl+C"
python -m neuralforge.tesseract.jarvis `
  --serve `
  --checkpoint artifacts\tpn\tpn_mind_core_v0_6.pt `
  --host 127.0.0.1 `
  --port $Port
