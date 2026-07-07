# Tesseract Pathway Network v1.4

v1.4 adds the Intelligence Benchmark Core.

## Purpose

The system now measures bounded local intelligence instead of relying on vibes.

```text
plan accuracy
execution success
safety blocking
cycle latency
cycle recommendation behavior
benchmark report
benchmark history ledger
```

## Run

```powershell
python -m neuralforge.tesseract.benchmark --write
```

## Metrics

```text
mean_score
plan_accuracy
safety_score
duration_ms
case_count
safety_case_count
```

## Boundary

This benchmark measures the bounded local Jarvis stack. It is not an AGI proof and does not test broad human-level reasoning.
