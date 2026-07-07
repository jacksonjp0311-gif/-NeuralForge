
param(
  [int]$Port = 8767
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

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
  Write-Host "No listening process found on port $Port."
  exit 0
}

foreach ($pidValue in $pids) {
  Write-Host "Stopping Jarvis listener PID $pidValue on port $Port"
  Stop-Process -Id $pidValue -Force
}

Start-Sleep -Milliseconds 350
$stillListening = netstat -ano | Select-String ":$Port" | Where-Object { $_.ToString() -match "LISTENING" }
if ($stillListening) {
  throw "Port $Port is still listening after stop attempt."
}

Write-Host "Tesseract Jarvis stopped."
