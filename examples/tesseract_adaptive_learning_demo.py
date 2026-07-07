
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuralforge.tesseract.adaptive import seed_replay_from_synthetic, train_tpn_from_replay
from neuralforge.tesseract.communication import TesseractEnglishAdapter

base = REPO_ROOT / "artifacts" / "tpn" / "tpn_mind_core_v0_4.pt"
replay = REPO_ROOT / "artifacts" / "tpn" / "replay" / "tpn_adaptive_demo.jsonl"
out = REPO_ROOT / "artifacts" / "tpn" / "tpn_mind_core_adaptive_demo.pt"

if replay.exists():
    replay.unlink()

seed_replay_from_synthetic(replay, n=48, seed=71)
result = train_tpn_from_replay(
    checkpoint_path=base,
    replay_path=replay,
    output_checkpoint=out,
    output_manifest=out.with_suffix(".json"),
    epochs=1,
    batch_size=16,
    device="cpu",
)

adapter = TesseractEnglishAdapter.from_checkpoint(out)
answer = adapter.think_english([0.55] * 16, style="compact")

print(json.dumps({
    "adaptive_checkpoint": result["checkpoint_path"],
    "approved_records": result["approved_records"],
    "text": answer["text"],
    "claim_boundary": "Local adaptive replay demo; no external calls.",
}, indent=2, sort_keys=True))
