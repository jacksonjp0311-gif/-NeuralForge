# Tesseract Pathway Network v1.8

v1.8 adds Goal-Aware Cycle Selection.

## Purpose

The system can now select one active bounded goal, run one cycle against that goal, record evidence, evaluate success/stop conditions, write a report, and stop.

```text
active goal
→ one bounded cycle
→ observed skills
→ goal evidence
→ goal evaluation
→ report
→ stop
```

## Run

```powershell
.\scripts\run_tesseract_goal_cycle.ps1
```

## Boundary

This is one bounded goal-aware cycle. It is not continuous autonomy, self-running execution, or code mutation authority.
