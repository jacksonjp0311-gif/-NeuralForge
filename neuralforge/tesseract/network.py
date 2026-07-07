"""PyTorch Tesseract Pathway Network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _all_vertex_bits() -> torch.Tensor:
    return torch.tensor([[int(ch) for ch in f"{i:04b}"] for i in range(16)], dtype=torch.float32)


class ExpertMLP(nn.Module):
    def __init__(self, d_model: int, multiplier: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * multiplier),
            nn.GELU(),
            nn.Linear(d_model * multiplier, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TesseractPathwayBlock(nn.Module):
    def __init__(self, d_model: int = 64, top_k: int = 5) -> None:
        super().__init__()
        self.d_model = d_model
        self.top_k = max(1, min(int(top_k), 16))
        self.experts = nn.ModuleList([ExpertMLP(d_model) for _ in range(16)])
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.register_buffer("vertex_bits", _all_vertex_bits(), persistent=False)

    def vertex_probabilities(self, axis_scores: torch.Tensor) -> torch.Tensor:
        axis_scores = axis_scores.clamp(1e-5, 1.0 - 1e-5)
        bits = self.vertex_bits.to(axis_scores.device)
        probs = axis_scores.unsqueeze(1) * bits.unsqueeze(0)
        probs = probs + (1.0 - axis_scores).unsqueeze(1) * (1.0 - bits.unsqueeze(0))
        return probs.prod(dim=-1)

    def forward(self, hidden: torch.Tensor, axis_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vertex_probs = self.vertex_probabilities(axis_scores)
        top_vals, top_idx = torch.topk(vertex_probs, k=self.top_k, dim=-1)
        mask = torch.zeros_like(vertex_probs)
        mask.scatter_(1, top_idx, top_vals)
        weights = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        expert_outputs = torch.stack([expert(hidden) for expert in self.experts], dim=1)
        routed = torch.einsum("bv,bvd->bd", weights, expert_outputs)
        return self.fuse(routed), vertex_probs


class TesseractPathwayNetwork(nn.Module):
    def __init__(self, input_dim: int = 16, d_model: int = 64, num_routes: int = 5, top_k: int = 5) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_routes = num_routes
        self.input = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.axis_head = nn.Linear(d_model, 4)
        self.block = TesseractPathwayBlock(d_model=d_model, top_k=top_k)
        self.route_head = nn.Linear(d_model, num_routes)
        self.authority_head = nn.Linear(d_model, 3)
        self.evidence_head = nn.Linear(d_model, 3)
        self.coherence_head = nn.Linear(d_model, 1)
        self.delta_phi_head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 2:
            raise ValueError(f"TPN expects [batch, input_dim], got {tuple(x.shape)}")
        hidden = self.input(x)
        axis_scores = torch.sigmoid(self.axis_head(hidden))
        routed, vertex_probs = self.block(hidden, axis_scores)
        return {
            "axis_scores": axis_scores,
            "vertex_probs": vertex_probs,
            "route_logits": self.route_head(routed),
            "authority_logits": self.authority_head(routed),
            "evidence_logits": self.evidence_head(routed),
            "coherence": torch.sigmoid(self.coherence_head(routed)).squeeze(-1),
            "delta_phi": F.softplus(self.delta_phi_head(routed)).squeeze(-1),
            "hidden": routed,
        }
