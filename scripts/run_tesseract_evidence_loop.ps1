param(
  [string]$RepoRoot = "C:\Users\jacks\OneDrive\Desktop\NeuralForge"
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"

Set-Location $RepoRoot

Write-Host "TPN v1.6.1 evidence loop: benchmark -> memory -> improvement"
Write-Host "1/3 Running benchmark and recording benchmark episode..."
python -m neuralforge.tesseract.benchmark --write --record-memory

Write-Host "2/3 Consolidating memory..."
python -c "from neuralforge.tesseract.memory_core import TesseractEpisodicMemory; import json; m=TesseractEpisodicMemory(); print(json.dumps(m.consolidate(), indent=2, sort_keys=True))"

Write-Host "3/3 Running improvement proposals with evidence available..."
python -m neuralforge.tesseract.improvement --write
