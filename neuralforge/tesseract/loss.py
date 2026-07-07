"""Compound loss for Tesseract Pathway Network outputs."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def _maybe_ce(total: torch.Tensor, logits: torch.Tensor, target: torch.Tensor | None, weight: float) -> torch.Tensor:
    if target is None:
        return total
    return total + float(weight) * F.cross_entropy(logits, target.long())


def _maybe_mse(total: torch.Tensor, pred: torch.Tensor, target: torch.Tensor | None, weight: float) -> torch.Tensor:
    if target is None:
        return total
    return total + float(weight) * F.mse_loss(pred.float(), target.float())


def tesseract_compound_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    weights = dict(weights or {})
    device = next(iter(outputs.values())).device
    total = torch.zeros((), device=device)

    total = _maybe_ce(total, outputs["route_logits"], targets.get("route"), weights.get("route", 1.0))
    total = _maybe_ce(total, outputs["authority_logits"], targets.get("authority"), weights.get("authority", 0.5))
    total = _maybe_ce(total, outputs["evidence_logits"], targets.get("evidence"), weights.get("evidence", 0.5))
    total = _maybe_mse(total, outputs["coherence"], targets.get("coherence"), weights.get("coherence", 0.25))
    total = _maybe_mse(total, outputs["delta_phi"], targets.get("delta_phi"), weights.get("delta_phi", 0.25))
    return total
