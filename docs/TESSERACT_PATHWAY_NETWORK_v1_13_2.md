# Tesseract Pathway Network v1.13.2

v1.13.2 closes the contract checker operational wound.

## Problem

`check_tesseract_contract.ps1` assumed the Jarvis HTTP server was already running. If the server was down, the checker failed before validating the local contract.

## Fix

The checker now tries the live server first. If the live server is unavailable and `-RequireLive` was not provided, it falls back to the local Python `contract_manifest()`.

```text
live /health + /contract
→ if unavailable
→ offline contract_manifest()
→ validate version + API contract
```

## Commands

Offline-capable check:

```powershell
.\scripts\check_tesseract_contract.ps1
```

Strict live-only check:

```powershell
.\scripts\check_tesseract_contract.ps1 -RequireLive
```

## Boundary

This is checker resilience only. It does not add mutation authority, autonomous execution, or patch application.
