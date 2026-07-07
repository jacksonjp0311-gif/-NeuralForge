
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuralforge.tesseract.communication import outputs_to_english
from neuralforge.tesseract.network import TesseractPathwayNetwork

torch.manual_seed(44)
model = TesseractPathwayNetwork(input_dim=16, d_model=32, top_k=4)
x = torch.rand(1, 16)

with torch.no_grad():
    outputs = model(x)

message = outputs_to_english(outputs, style="operator")[0]
print(message.as_text())
print(json.dumps(message.receipt, indent=2, sort_keys=True))
