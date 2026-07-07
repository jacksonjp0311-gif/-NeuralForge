# Tesseract Pathway Network v1.6

v1.6 adds the Self-Improvement Proposal Core.

## Purpose

The system can now read benchmark and memory evidence and produce improvement proposals.

```text
benchmark evidence
+ episodic memory evidence
→ proposal list
→ risk level
→ expected impact
→ required gates
```

## Run

```powershell
python -m neuralforge.tesseract.improvement --write
```

or:

```powershell
.\scripts\run_tesseract_improvement.ps1
```

## Boundary

This does not edit files, run arbitrary commands, or mutate the repository. It proposes next actions only.
