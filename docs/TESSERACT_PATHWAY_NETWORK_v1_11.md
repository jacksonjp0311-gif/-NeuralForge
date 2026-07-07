# Tesseract Pathway Network v1.11

v1.11 adds the Performance Telemetry Governor.

## Purpose

The system can now measure the latency of policy-approved bounded execution.

```text
policy decision
→ queue execution
→ goal-cycle durations
→ per-skill latency
→ threshold evaluation
→ performance receipt
```

## Run

```powershell
.\scripts\run_tesseract_performance.ps1
```

## Boundary

This is performance telemetry only. It does not add autonomy, mutation authority, or execution power.
