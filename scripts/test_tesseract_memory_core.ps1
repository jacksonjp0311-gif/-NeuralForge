param(
  [int]$Port = 8767
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$Base = "http://127.0.0.1:$Port"

Write-Host "Running a cycle to create episodic memory..."
$body = @{
  objective = "check repo status and recent git log"
  execute = $true
  max_steps = 4
} | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri "$Base/cycle" -Method POST -ContentType "application/json" -Body $body | Format-List

Write-Host "Reading recent episodic memory..."
Invoke-RestMethod "$Base/memory/episodes?limit=5" | Format-List

Write-Host "Searching episodic memory..."
$body = @{
  query = "repo status"
  limit = 5
} | ConvertTo-Json -Depth 8
Invoke-RestMethod -Uri "$Base/memory/episodic/search" -Method POST -ContentType "application/json" -Body $body | Format-List

Write-Host "Consolidating episodic memory..."
Invoke-RestMethod -Uri "$Base/memory/consolidate" -Method POST -ContentType "application/json" -Body "{}" | Format-List
