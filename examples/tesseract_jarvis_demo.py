
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuralforge.tesseract.jarvis import JarvisServiceConfig, TesseractJarvisRuntime

runtime = TesseractJarvisRuntime(JarvisServiceConfig(
    checkpoint=str(REPO_ROOT / "artifacts" / "tpn" / "tpn_mind_core_v0_6.pt"),
    memory_path=str(REPO_ROOT / "artifacts" / "tpn" / "jarvis_demo_memory.jsonl"),
    ledger_path=str(REPO_ROOT / "artifacts" / "tpn" / "jarvis_demo_ledger.jsonl"),
))

answer = runtime.command("plan the next safe local NeuralForge evolution step", execute=True)
print(json.dumps({
    "text": answer["text"],
    "jarvis_latency_ms": answer["jarvis_latency_ms"],
    "skills": runtime.skills()["skills"],
    "ledger_recent": runtime.ledger_recent(limit=1)["entries"],
    "claim_boundary": answer["claim_boundary"],
}, indent=2, sort_keys=True))
