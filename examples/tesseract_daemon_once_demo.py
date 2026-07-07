
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuralforge.tesseract.daemon import TesseractWarmRuntime

runtime = TesseractWarmRuntime(REPO_ROOT / "artifacts" / "tpn" / "tpn_mind_core_v0_6.pt")
result = runtime.think([
    0.93, 0.86, 0.42, 0.79,
    1.0,
    0.0, 0.0, 1.0, 0.0,
    0.73, 0.21,
    0.0, 0.0, 0.0, 1.0, 0.0,
], style="operator")

print(json.dumps({
    "latency_ms": result["latency_ms"],
    "text": result["text"],
    "selected_vertices": result["selected_vertices"],
    "claim_boundary": result["claim_boundary"],
}, indent=2, sort_keys=True))
