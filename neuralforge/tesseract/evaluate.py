
"""Evaluation helpers for Tesseract Pathway Network."""

from __future__ import annotations

from typing import Any

import torch

from neuralforge.tesseract.loss import tesseract_compound_loss


@torch.no_grad()
def evaluate_tpn_model(model: torch.nn.Module, loader, device: torch.device | str | None = None) -> dict[str, Any]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()

    total = 0
    route_ok = 0
    authority_ok = 0
    evidence_ok = 0
    vertex_ok = 0
    loss_sum = 0.0
    sparse_ratios: list[float] = []

    for x, targets in loader:
        x = x.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        out = model(x)
        loss = tesseract_compound_loss(out, targets)
        batch = x.shape[0]
        total += batch
        loss_sum += float(loss.item()) * batch
        route_ok += int((out["route_logits"].argmax(dim=-1) == targets["route"]).sum().item())
        authority_ok += int((out["authority_logits"].argmax(dim=-1) == targets["authority"]).sum().item())
        evidence_ok += int((out["evidence_logits"].argmax(dim=-1) == targets["evidence"]).sum().item())
        if "vertex" in targets and "vertex_logits" in out:
            vertex_ok += int((out["vertex_logits"].argmax(dim=-1) == targets["vertex"]).sum().item())
        if "sparse_ratio" in out:
            sparse_ratios.append(float(out["sparse_ratio"].detach().cpu().item()))

    total = max(total, 1)
    return {
        "loss": loss_sum / total,
        "route_accuracy": route_ok / total,
        "authority_accuracy": authority_ok / total,
        "evidence_accuracy": evidence_ok / total,
        "vertex_accuracy": vertex_ok / total,
        "samples": total,
        "mean_sparse_ratio": sum(sparse_ratios) / max(len(sparse_ratios), 1),
        "claim_boundary": "Synthetic route-governance evaluation only; not a production proof.",
    }
