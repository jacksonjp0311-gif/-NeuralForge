# Tesseract Pathway Network v1.13.1

v1.13.1 closes the receipt-output flood by compressing console output while preserving full JSON receipts.

## Purpose

The control bundle was working, but the console output became too dense. This close keeps the full receipts on disk and prints only a compact summary in PowerShell.

```text
full JSON receipt -> saved to artifacts/tpn
compact summary -> printed to console
```

## Run

```powershell
.\scripts\run_tesseract_control_bundle.ps1
```

## Boundary

This is receipt presentation hygiene only. It does not add mutation authority, autonomous execution, or patch application.
