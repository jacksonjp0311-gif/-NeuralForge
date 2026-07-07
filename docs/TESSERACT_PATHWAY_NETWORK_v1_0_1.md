
# Tesseract Pathway Network v1.0.1

v1.0.1 hardens the stable Jarvis runtime.

## What changed

```text
safe local checkpoint loading
launcher no longer uses python -m jarvis
port conflict guard
status script
improved stop script
runtime hardening tests
```

## Scripts

```powershell
.\scripts\start_tesseract_jarvis.ps1
.\scripts\status_tesseract_jarvis.ps1
.\scripts\check_tesseract_contract.ps1
.\scripts\test_tesseract_jarvis.ps1
.\scripts\stop_tesseract_jarvis.ps1
```

## Purpose

v1.0 made the API contract stable. v1.0.1 makes local operation cleaner and safer before adding broader integration skills.

## Boundary

This is runtime hardening. It does not add autonomous authority or arbitrary shell execution.
