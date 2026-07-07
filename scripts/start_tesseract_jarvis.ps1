
param(
  [string]$RepoRoot = "C:\Users\jacks\OneDrive\Desktop\NeuralForge",
  [int]$Port = 8767
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

function Test-PortOpen {
  param([int]$Port)
  $lines = netstat -ano | Select-String ":$Port"
  foreach ($line in $lines) {
    if ($line.ToString() -match "LISTENING") { return $true }
  }
  return $false
}

Set-Location $RepoRoot
$Base = "http://127.0.0.1:$Port"

if (Test-PortOpen -Port $Port) {
  Write-Host "Tesseract Jarvis already appears to be listening on port $Port."
  try {
    Invoke-RestMethod "$Base/health" | Format-List
  } catch {
    Write-Host "Port is occupied, but health check failed."
  }
  exit 0
}

Write-Host "Starting Tesseract Jarvis runtime in foreground."
Write-Host "Health: $Base/health"
Write-Host "Contract: $Base/contract"
Write-Host "Stop: Ctrl+C"
Write-Host ""

python -c "from neuralforge.tesseract.jarvis import run_jarvis_server; run_jarvis_server(checkpoint=r'artifacts\tpn\tpn_mind_core_v0_6.pt', repo_root='.', host='127.0.0.1', port=$Port)"
