
# Tesseract Pathway Network v0.9

v0.9 adds the daily-use Jarvis substrate around the weighted TPN command mind.

## What changed

```text
TesseractJarvisRuntime
TesseractActionLedger
GET  /skills
POST /memory/search
GET  /ledger/recent
POST /ledger/search
scripts/start_tesseract_jarvis.ps1
scripts/test_tesseract_jarvis.ps1
scripts/stop_tesseract_jarvis.ps1
```

## Purpose

v0.8 proved English command routing. v0.9 makes the runtime usable every day:

```text
start service
check health
list skills
send commands
search local memory
inspect action ledger
stop service
```

## Start

```powershell
.\scripts\start_tesseract_jarvis.ps1
```

## Test from a second PowerShell

```powershell
.\scripts\test_tesseract_jarvis.ps1
```

## Boundary

This remains local, governed, and explicit. It does not perform arbitrary shell execution, external model calls, or autonomous mutation.
