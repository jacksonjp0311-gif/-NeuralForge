param(
  [string]$Base = "http://127.0.0.1:8765",
  [string]$ExpectedVersion = "tpn.v1.14",
  [switch]$RequireLive
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

function Fail {
  param([string]$Message)
  Write-Host ("TPN CONTRACT CHECK FAIL | " + $Message)
  exit 1
}

function Get-Prop {
  param($Object, [string]$Name)
  if ($null -eq $Object) { return $null }
  $prop = $Object.PSObject.Properties[$Name]
  if ($null -eq $prop) { return $null }
  return $prop.Value
}

function Validate-Manifest {
  param($Manifest, [string]$Mode)

  $version = Get-Prop $Manifest "version"
  $api = Get-Prop $Manifest "api_contract_version"

  if ($version -ne $ExpectedVersion) {
    Fail "$Mode contract version mismatch: expected=$ExpectedVersion actual=$version"
  }
  if ($api -ne "jarvis.api.v1") {
    Fail "$Mode api contract mismatch: expected=jarvis.api.v1 actual=$api"
  }

  $checker = Get-Prop $Manifest "contract_checker_version"
  if ($null -ne $checker) {
    if ($checker -ne "tpn.contract_checker.v1.13.2") {
      Fail "$Mode contract checker mismatch: expected=tpn.contract_checker.v1.13.2 actual=$checker"
    }
  }

  Write-Host ("TPN CONTRACT CHECK PASS | mode=" + $Mode + " version=" + $version + " api=" + $api)
  if ($null -ne $checker) {
    Write-Host ("TPN CONTRACT CHECKER | " + $checker)
  }
}

function Invoke-OfflineContractCheck {
  Write-Host "TPN CONTRACT CHECK | live server unavailable; using offline contract_manifest fallback"
  $json = python -c "import json; from neuralforge.tesseract.contract import contract_manifest; print(json.dumps(contract_manifest(), sort_keys=True))"
  if ($LASTEXITCODE -ne 0) {
    Fail "offline contract_manifest import failed"
  }
  $manifest = $json | ConvertFrom-Json
  Validate-Manifest -Manifest $manifest -Mode "offline"
}

try {
  $health = Invoke-RestMethod -Uri "$Base/health" -TimeoutSec 3
  $manifest = Invoke-RestMethod -Uri "$Base/contract" -TimeoutSec 3
  Validate-Manifest -Manifest $manifest -Mode "live"
  $status = Get-Prop $health "status"
  if ($null -ne $status) {
    Write-Host ("TPN CONTRACT HEALTH | status=" + $status)
  }
  exit 0
}
catch {
  $message = $_.Exception.Message
  Write-Host ("TPN CONTRACT CHECK | live check unavailable: " + $message)
  if ($RequireLive.IsPresent) {
    Fail "RequireLive was set and live contract check failed"
  }
  Invoke-OfflineContractCheck
  exit 0
}
