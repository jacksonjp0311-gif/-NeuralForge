
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuralforge.tesseract.communication import TesseractEnglishAdapter

checkpoint = REPO_ROOT / "artifacts" / "tpn" / "tpn_mind_core_v0_4.pt"

adapter = TesseractEnglishAdapter.from_checkpoint(checkpoint)
answer = adapter.think_english([
    0.92, 0.88, 0.35, 0.81,
    1.0,
    0.0, 0.0, 1.0, 0.0,
    0.72, 0.18,
    0.0, 0.0, 0.0, 1.0, 0.0,
], style="operator")

print(answer["text"])
print(json.dumps(answer["receipts"][0], indent=2, sort_keys=True))
