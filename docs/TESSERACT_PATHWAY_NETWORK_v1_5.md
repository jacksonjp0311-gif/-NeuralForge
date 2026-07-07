# Tesseract Pathway Network v1.5

v1.5 adds the Episodic Memory Core.

## Purpose

Jarvis now records bounded local experiences as durable memory episodes.

```text
commands
tasks
plans
cycles
observations
recommendations
```

## New endpoints

```text
GET  /memory/episodes
POST /memory/episodic/search
POST /memory/consolidate
```

## Runtime object

```text
TesseractEpisodicMemory
TesseractMemoryEpisode
```

## Boundary

This is local episodic memory. It is not consciousness and not autonomous self-awareness.

## PowerShell test

```powershell
.\scripts\test_tesseract_memory_core.ps1
```
