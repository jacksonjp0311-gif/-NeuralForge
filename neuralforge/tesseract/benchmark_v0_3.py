
"""v0.3 sparse pathway benchmark for TPN.

This benchmark is intentionally synthetic and small. It checks execution path,
receipt production, and relative expert activation, not production performance.
"""

from __future__ import annotations

import json
import time

import torch

from neuralforge.tesseract.network import TesseractPathwayNetwork
from neuralforge.tesseract.receipt import build_tesseract_receipts


def run_v0_3_benchmark(batch: int = 64, input_dim: int = 16, d_model: int = 48, top_k: int = 4) -> dict[str, object]:
    torch.manual_seed(7)
    x = torch.rand(batch, input_dim)
    model = TesseractPathwayNetwork(input_dim=input_dim, d_model=d_model, top_k=top_k)
    model.eval()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(x)
        receipts = build_tesseract_receipts(outputs, max_items=3)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    selected = outputs["selected_vertex"].detach().cpu()
    unique_vertices = sorted({int(v) for v in selected.tolist()})
    return {
        "benchmark": "tpn_v0_3_sparse_pathway",
        "batch": batch,
        "input_dim": input_dim,
        "d_model": d_model,
        "top_k": top_k,
        "elapsed_ms": elapsed_ms,
        "mean_sparse_ratio": float(outputs["sparse_ratio"].detach().cpu().item()),
        "unique_selected_vertices": unique_vertices,
        "unique_selected_vertex_count": len(unique_vertices),
        "receipt_preview": receipts,
        "claim_boundary": "Toy synthetic benchmark only; not a hardware or production efficiency claim.",
    }


def main() -> None:
    print(json.dumps(run_v0_3_benchmark(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
