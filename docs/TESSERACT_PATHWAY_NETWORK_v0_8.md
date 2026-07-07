
# Tesseract Pathway Network v0.8

v0.8 adds a governed local command mind.

## Purpose

The TPN can now accept plain English commands and route them through the local weighted mind core.

```text
English command
→ deterministic local vector
→ weighted TPN route
→ receipt
→ skill selection
→ governed action packet
→ optional safe local execution
```

## New components

```text
CommandVectorizer
TesseractSkillRegistry
TesseractActionPacket
TesseractCommandMind
POST /command
```

## Built-in local skills

```text
tpn.status
tpn.echo
tpn.plan
tpn.math
tpn.memory_note
```

`tpn.memory_note` mutates local append-only memory and is blocked unless mutation is explicitly allowed and the route does not require shadow/authority blocking.

## CLI

```powershell
python -m neuralforge.tesseract.command --command "plan the next local step" --execute
```

## Server

```powershell
python -m neuralforge.tesseract.command --serve --checkpoint artifacts\tpn\tpn_mind_core_v0_6.pt --host 127.0.0.1 --port 8766
```

## Boundary

This is a governed local command substrate. It is not a general LLM, not arbitrary shell execution, and not autonomous authority.
