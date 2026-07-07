
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuralforge.tesseract.checkpoint import TesseractCheckpointConfig, train_tpn_checkpoint
from neuralforge.tesseract.mind import TesseractMindCore

artifact = train_tpn_checkpoint(
    output_dir=REPO_ROOT / "artifacts" / "tpn",
    name="tpn_mind_core_demo",
    config=TesseractCheckpointConfig(samples=64, epochs=1, batch_size=16, d_model=24, top_k=4, seed=23),
)

core = TesseractMindCore.from_checkpoint(artifact["checkpoint_path"])
result = core.think(torch.rand(2, 16))

print(json.dumps({
    "checkpoint_path": artifact["checkpoint_path"],
    "selected_vertices": result["selected_vertices"],
    "receipt_preview": result["receipts"][:1],
    "claim_boundary": result["claim_boundary"],
}, indent=2, sort_keys=True))
