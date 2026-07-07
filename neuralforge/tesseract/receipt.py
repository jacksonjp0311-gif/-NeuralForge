
"""Inference receipts for Tesseract Pathway Network."""

from __future__ import annotations

from typing import Any

import torch

from neuralforge.tesseract.data import ID_TO_ROUTE
from neuralforge.tesseract.geometry import missing_axes

AXIS_NAMES = ("intent", "evidence", "authority", "context")


def _float_list(x: torch.Tensor) -> list[float]:
    return [float(v) for v in x.detach().cpu().tolist()]


def _int_list(x: torch.Tensor) -> list[int]:
    return [int(v) for v in x.detach().cpu().tolist()]


def build_tesseract_receipts(outputs: dict[str, torch.Tensor], max_items: int | None = None) -> list[dict[str, Any]]:
    """Convert model outputs into compact route receipts.

    Receipts are intentionally declarative: they explain which vertex experts
    were considered, which vertex won, and what governance route was predicted.
    """

    axis_scores = outputs["axis_scores"].detach()
    route_ids = outputs["route_logits"].argmax(dim=-1).detach()
    selected = outputs["selected_vertex"].detach()
    topk = outputs["topk_indices"].detach()
    weights = outputs["topk_weights"].detach()
    coherence = outputs["coherence"].detach()
    delta_phi = outputs["delta_phi"].detach()

    batch = int(axis_scores.shape[0])
    if max_items is not None:
        batch = min(batch, int(max_items))

    receipts: list[dict[str, Any]] = []
    for i in range(batch):
        vertex_id = int(selected[i].item())
        vertex = f"{vertex_id:04b}"
        axes = {name: float(axis_scores[i, j].item()) for j, name in enumerate(AXIS_NAMES)}
        receipts.append({
            "receipt_version": "tpn.v0.3",
            "selected_vertex": vertex,
            "selected_vertex_id": vertex_id,
            "missing_axes": missing_axes(vertex),
            "topk_vertices": [f"{idx:04b}" for idx in _int_list(topk[i])],
            "topk_weights": _float_list(weights[i]),
            "axis_scores": axes,
            "route_id": int(route_ids[i].item()),
            "route": ID_TO_ROUTE.get(int(route_ids[i].item()), "unknown"),
            "coherence": float(coherence[i].item()),
            "delta_phi": float(delta_phi[i].item()),
            "claim_boundary": "Receipt explains model routing metadata; it is not external proof of correctness.",
        })
    return receipts
