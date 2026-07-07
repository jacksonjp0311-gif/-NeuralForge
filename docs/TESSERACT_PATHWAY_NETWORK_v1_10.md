# Tesseract Pathway Network v1.10

v1.10 adds the Execution Policy Governor.

## Purpose

The system can now evaluate a goal queue against explicit budget, risk, and stop-condition policy before execution.

```text
queue plan
→ policy decision
→ allowed or blocked
→ guarded queue run only if allowed
→ policy receipt
```

## Run

```powershell
.\scripts\run_tesseract_policy.ps1
```

## Boundary

This is a policy gate above the bounded queue. It does not add continuous autonomy, mutation authority, or unrestricted execution.
