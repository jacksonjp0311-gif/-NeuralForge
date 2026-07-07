
param(
  [int]$Port = 8767
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$Base = "http://127.0.0.1:$Port"

Write-Host "Testing health..."
Invoke-RestMethod "$Base/health" | Format-List

Write-Host "Testing skills..."
Invoke-RestMethod "$Base/skills" | Format-List

Write-Host "Testing command..."
$body = @{
  command = "plan the next safe local NeuralForge evolution step"
  execute = $true
} | ConvertTo-Json
Invoke-RestMethod -Uri "$Base/command" -Method POST -ContentType "application/json" -Body $body | Format-List

Write-Host "Testing memory search..."
$body = @{
  query = "Tesseract"
  limit = 5
} | ConvertTo-Json
Invoke-RestMethod -Uri "$Base/memory/search" -Method POST -ContentType "application/json" -Body $body | Format-List

Write-Host "Testing ledger recent..."
Invoke-RestMethod "$Base/ledger/recent?limit=5" | Format-List
