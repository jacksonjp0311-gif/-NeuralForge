
# Tesseract Pathway Network v0.6

v0.6 adds adaptive local replay learning.

## Purpose

TPN can now improve its local weights from approved experience records.

```text
receipt / operator correction
→ approved JSONL replay event
→ local fine-tune
→ new checkpoint
→ local English output
```

## New components

```text
TesseractFeedbackRecord
TesseractReplayLedger
append_operator_feedback()
seed_replay_from_synthetic()
train_tpn_from_replay()
```

## Why this matters

This creates a self-learning path without letting the model mutate itself blindly.

The learning loop is:

```text
observe
record
approve
replay
train
save checkpoint
verify
promote
```

## Boundary

This is adaptive replay training from approved local records. It is not uncontrolled self-modification, autonomous authority, or AGI.

The system learns only from records admitted into the ledger.
