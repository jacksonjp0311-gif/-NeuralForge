# Tesseract Pathway Network v1.9

v1.9 adds the Guarded Multi-Goal Queue.

## Purpose

The system can now plan and execute a bounded queue of active goals. Each selected goal receives at most one goal-aware cycle. The queue stops when it reaches the max-goal cap, when a goal blocks, or when no eligible goals remain.

```text
active goals
→ sorted bounded queue
→ one cycle per selected goal
→ evidence + evaluation
→ stop on block or cap
→ queue report
```

## Run

```powershell
.\scripts\run_tesseract_goal_queue.ps1
```

## Boundary

This is not continuous autonomy. It does not run forever, does not mutate code, and does not override stop conditions.
