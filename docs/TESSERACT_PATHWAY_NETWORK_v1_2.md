# Tesseract Pathway Network v1.2

v1.2 adds bounded task planning over the Integration Skill Bus.

## Purpose

Jarvis can now convert English intent into an explicit local task plan and execute that plan using only whitelisted integration skills.

```text
POST /plan
POST /run_plan
TesseractTaskPlanner
TesseractTaskPlan
TesseractPlanStep
```

## Example

```json
{
  "command": "check repo status, recent commits, and read README.md",
  "execute": true
}
```

## Boundary

This is not autonomous general intelligence. It is bounded local task planning through approved skills only.

## PowerShell test

```powershell
.\scripts\test_tesseract_planner.ps1
```
