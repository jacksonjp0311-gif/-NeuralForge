param(
  [int]$Port = 8767
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$Base = "http://127.0.0.1:$Port"

Write-Host "Testing plan only..."
$body = @{
  command = "check repo status, recent commits, and read README.md"
  execute = $false
} | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri "$Base/plan" -Method POST -ContentType "application/json" -Body $body | Format-List

Write-Host "Testing plan execute..."
$body = @{
  command = "check repo status and recent git log"
  execute = $true
} | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri "$Base/plan" -Method POST -ContentType "application/json" -Body $body | Format-List

Write-Host "Testing run_plan..."
$planResponse = Invoke-RestMethod -Uri "$Base/plan" -Method POST -ContentType "application/json" -Body (@{
  command = "check contract and ledger"
  execute = $false
} | ConvertTo-Json -Depth 8)

Invoke-RestMethod -Uri "$Base/run_plan" -Method POST -ContentType "application/json" -Body (@{
  plan = $planResponse.plan
} | ConvertTo-Json -Depth 20) | Format-List
