
param(
  [int]$Port = 8767
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$Base = "http://127.0.0.1:$Port"

Write-Host "Testing integration skills..."
Invoke-RestMethod "$Base/integration/skills" | Format-List

Write-Host "Testing repo.status task..."
$body = @{
  skill_id = "repo.status"
  params = @{}
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "$Base/task" -Method POST -ContentType "application/json" -Body $body | Format-List

Write-Host "Testing repo.log task..."
$body = @{
  skill_id = "repo.log"
  params = @{ limit = 5 }
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "$Base/task" -Method POST -ContentType "application/json" -Body $body | Format-List

Write-Host "Testing file.read task..."
$body = @{
  skill_id = "file.read"
  params = @{ path = "README.md"; max_bytes = 1600 }
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri "$Base/task" -Method POST -ContentType "application/json" -Body $body | Format-List
