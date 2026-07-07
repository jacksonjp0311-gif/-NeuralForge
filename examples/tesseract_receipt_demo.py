
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuralforge.tesseract.network import TesseractPathwayNetwork
from neuralforge.tesseract.receipt import build_tesseract_receipts

torch.manual_seed(11)
model = TesseractPathwayNetwork(input_dim=16, d_model=32, top_k=4)
x = torch.rand(2, 16)

with torch.no_grad():
    outputs = model(x)

print(json.dumps(build_tesseract_receipts(outputs), indent=2, sort_keys=True))
