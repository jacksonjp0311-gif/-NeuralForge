# Tesseract Pathway Network v1.15

v1.15 adds the Sandboxed Patch Plan Receipt.

## Purpose

The system can now convert an approved human approval receipt into a sandbox planning artifact.

```text
approval receipt
→ sandbox branch name
→ planned patch steps
→ required verifier list
→ rollback plan
→ plan receipt
```

## Boundary

This module does not create a branch, edit files, apply patches, merge changes, or grant autonomous write authority.

```text
planning_allowed: true
mutation_allowed: false
apply_allowed: false
```

## Commands

Use existing approval receipt:

```powershell
.\scripts\run_tesseract_sandbox_plan.ps1
```

Smoke/demo approved receipt:

```powershell
.\scripts\run_tesseract_sandbox_plan.ps1 -DemoApproved
```
