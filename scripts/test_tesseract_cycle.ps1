param(
  [int]$Port = 8767
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$Base = "http://127.0.0.1:$Port"

Write-Host "Testing cycle plan-only..."
$body = @{
  objective = "check repo status and recent git log"
  execute = $false
  max_steps = 4
} | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri "$Base/cycle" -Method POST -ContentType "application/json" -Body $body | Format-List

Write-Host "Testing cycle execute..."
$body = @{
  objective = "check repo status, recent git log, and contract"
  execute = $true
  max_steps = 6
} | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri "$Base/cycle" -Method POST -ContentType "application/json" -Body $body | Format-List
