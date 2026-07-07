
param(
  [int]$Port = 8767
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$Base = "http://127.0.0.1:$Port"
$lines = netstat -ano | Select-String ":$Port"
$pids = @()

foreach ($line in $lines) {
  if ($line.ToString() -match "LISTENING") {
    $parts = (($line.ToString()).Trim() -split "\s+")
    $pidValue = $parts[-1]
    if ($pidValue -match "^\d+$") { $pids += [int]$pidValue }
  }
}

$pids = $pids | Select-Object -Unique

if (-not $pids) {
  Write-Host "Tesseract Jarvis is not listening on port $Port."
  exit 1
}

Write-Host ("Tesseract Jarvis listening on port {0}; PID(s): {1}" -f $Port, ($pids -join ", "))

try {
  $health = Invoke-RestMethod "$Base/health"
  $contract = Invoke-RestMethod "$Base/contract"
  Write-Host "Health PASS"
  Write-Host ("runtime: " + $health.runtime)
  Write-Host ("version: " + $health.version)
  Write-Host ("api_contract_version: " + $health.api_contract_version)
  Write-Host ("endpoint_count: " + $contract.endpoint_count)
} catch {
  Write-Host "Health check failed even though port is listening."
  throw
}
