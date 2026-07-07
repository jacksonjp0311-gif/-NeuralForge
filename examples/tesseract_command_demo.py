
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuralforge.tesseract.command import TesseractCommandMind

mind = TesseractCommandMind(
    REPO_ROOT / "artifacts" / "tpn" / "tpn_mind_core_v0_6.pt",
    memory_path=REPO_ROOT / "artifacts" / "tpn" / "command_memory_demo.jsonl",
)

answer = mind.handle("plan the next safe NeuralForge evolution step", execute=True)
print(json.dumps({
    "text": answer["text"],
    "skill_id": answer["packet"]["skill_id"],
    "allowed": answer["packet"]["allowed"],
    "selected_vertex": answer["packet"]["selected_vertex"],
    "claim_boundary": answer["claim_boundary"],
}, indent=2, sort_keys=True))
