
param(
  [int]$Port = 8767
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$lines = netstat -ano | Select-String ":$Port"
if (-not $lines) {
  Write-Host "No process found on port $Port."
  exit 0
}

$pids = @()
foreach ($line in $lines) {
  $parts = (($line.ToString()).Trim() -split "\s+")
  $pidValue = $parts[-1]
  if ($pidValue -match "^\d+$") { $pids += [int]$pidValue }
}

$pids = $pids | Select-Object -Unique
foreach ($pidValue in $pids) {
  Write-Host "Stopping PID $pidValue on port $Port"
  Stop-Process -Id $pidValue -Force
}
