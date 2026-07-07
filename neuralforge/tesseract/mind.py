
"""Local Tesseract mind core.

This is the weight-bearing runtime wrapper around TPN. It performs no external
API calls. It loads local weights, runs local inference, and emits local route
receipts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neuralforge.tesseract.checkpoint import load_tpn_checkpoint
from neuralforge.tesseract.network import TesseractPathwayNetwork
from neuralforge.tesseract.receipt import build_tesseract_receipts


class TesseractMindCore:
    """A local weighted TPN runtime for route-governed system cognition."""

    def __init__(self, model: TesseractPathwayNetwork, metadata: dict[str, Any] | None = None) -> None:
        self.model = model.eval()
        self.metadata = dict(metadata or {})

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "TesseractMindCore":
        model, payload = load_tpn_checkpoint(path, map_location=map_location)
        return cls(model, metadata=payload)

    def think(self, x: torch.Tensor, *, receipts: bool = True) -> dict[str, Any]:
        self.model.eval()
        with torch.no_grad():
            out = self.model(x)
        result: dict[str, Any] = {
            "outputs": out,
            "route_ids": out["route_logits"].argmax(dim=-1).detach().cpu().tolist(),
            "selected_vertices": [f"{int(v):04b}" for v in out["selected_vertex"].detach().cpu().tolist()],
            "claim_boundary": "Local TPN inference receipt; not external truth or autonomous authority.",
        }
        if receipts:
            result["receipts"] = build_tesseract_receipts(out)
        return result

    def think_from_vector(self, vector: list[float] | tuple[float, ...], *, receipts: bool = True) -> dict[str, Any]:
        x = torch.tensor([list(vector)], dtype=torch.float32)
        return self.think(x, receipts=receipts)
