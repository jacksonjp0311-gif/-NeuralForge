# Tesseract Pathway Network v1.14

v1.14 adds the Human Approval Ledger.

## Purpose

The system can now record explicit human approval or rejection receipts for proposal bundles.

```text
control bundle proposals
→ pending approval summary
→ explicit human approve/reject command
→ approval decision receipt
→ ledger append
```

## Important boundary

Approval does not apply patches. Approval unlocks the next planning layer only:

```text
sandboxed_patch_plan_receipt
```

Mutation remains disabled.

## Commands

View pending approval:

```powershell
.\scripts\run_tesseract_approval.ps1
```

Record approval:

```powershell
.\scripts\run_tesseract_approval.ps1 -Decision approve -HumanId "James"
```

Record rejection:

```powershell
.\scripts\run_tesseract_approval.ps1 -Decision reject -HumanId "James"
```
