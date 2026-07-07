param(
  [int]$Port = 8767,
  [string]$ExpectedVersion = "tpn.v1.4"
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

$Base = "http://127.0.0.1:$Port"
$health = Invoke-RestMethod "$Base/health"
$contract = Invoke-RestMethod "$Base/contract"
$skills = Invoke-RestMethod "$Base/skills"

if (-not $health.ok) { throw "health failed" }
if ($health.version -ne $ExpectedVersion) { throw "unexpected health version: $($health.version)" }
if ($health.api_contract_version -ne "jarvis.api.v1") { throw "unexpected api contract: $($health.api_contract_version)" }
if (-not $contract.ok) { throw "contract failed" }
if ($contract.version -ne $ExpectedVersion) { throw "unexpected contract version: $($contract.version)" }
if (-not $skills.ok) { throw "skills failed" }

Write-Host "Tesseract Jarvis contract PASS"
Write-Host ("version: " + $health.version)
Write-Host ("api_contract_version: " + $health.api_contract_version)
Write-Host ("endpoint_count: " + $contract.endpoint_count)
