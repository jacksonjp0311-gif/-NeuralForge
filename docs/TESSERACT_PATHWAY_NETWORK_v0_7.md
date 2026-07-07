
# Tesseract Pathway Network v0.7

v0.7 adds a warm local runtime daemon.

## Purpose

TPN now supports a low-overhead Jarvis-style substrate:

```text
load checkpoint once
keep model warm
accept local vector requests
return English + receipt
append feedback to replay
```

## Endpoints

```text
GET  /health
POST /think
POST /feedback
```

## CLI one-shot

```powershell
python -m neuralforge.tesseract.daemon --once --checkpoint artifacts\tpn\tpn_mind_core_v0_6.pt
```

## Local server

```powershell
python -m neuralforge.tesseract.daemon --checkpoint artifacts\tpn\tpn_mind_core_v0_6.pt --host 127.0.0.1 --port 8765
```

## Boundary

This is a warm local TPN runtime. It is not a general LLM, not autonomous authority, and not uncontrolled self-modification.

Its job is to keep the geometric mind core resident in memory so local systems can call it quickly.
