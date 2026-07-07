
# Tesseract Pathway Network v1.1

v1.1 adds the Integration Skill Bus.

## Purpose

The Jarvis runtime can now interact with the local repository through explicit, whitelisted skills.

```text
POST /task
GET  /integration/skills
```

## Built-in integration skills

```text
system.ping
repo.status
repo.log
repo.contract
file.read
memory.search
ledger.recent
```

## Boundary

This is not arbitrary shell execution. The bus only exposes fixed local skills with bounded inputs and explicit task receipts.

## Example task

```json
{
  "skill_id": "repo.status",
  "params": {}
}
```

## PowerShell test

```powershell
.\scripts\test_tesseract_integration_bus.ps1
```
