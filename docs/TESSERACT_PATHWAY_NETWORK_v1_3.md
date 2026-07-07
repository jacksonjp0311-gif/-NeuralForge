# Tesseract Pathway Network v1.3

v1.3 adds the Observation Cycle Engine.

## Purpose

Jarvis can now run one bounded local cycle:

```text
objective
→ plan
→ optional execute
→ observe
→ report next recommendation
```

## New endpoint

```text
POST /cycle
```

## New runtime object

```text
TesseractCycleEngine
TesseractCycleReport
TesseractCycleObservation
```

## Boundary

This is not autonomous general intelligence. It is a single bounded observe-plan-act-report cycle using only whitelisted integration skills.

## PowerShell test

```powershell
.\scripts\test_tesseract_cycle.ps1
```
