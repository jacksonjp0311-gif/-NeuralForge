# Tesseract Pathway Network v1.7

v1.7 adds the Goal-State Manager.

## Purpose

The system can now preserve explicit bounded goals with success criteria, stop conditions, evidence receipts, and status evaluation.

```text
goal
→ success criteria
→ stop conditions
→ evidence receipts
→ evaluate status
→ summarize next recommendation
```

## Run

```powershell
.\scripts\run_tesseract_goal_state.ps1
```

## Boundary

This is goal-state storage and evaluation only. It does not run continuous autonomy, mutate code, or apply patches.
