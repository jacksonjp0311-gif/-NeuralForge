
"""PyTorch Tesseract Pathway Network.

v0.3 adds pathway-sparse dispatch, vertex supervision hooks, and receipt-ready
routing metadata while preserving the v0.2 public output contract.
"""

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


class TesseractSparseDispatcher(nn.Module):
    """Dispatch hidden states through selected tesseract vertex experts.

    The router computes a 16-vertex probability distribution from four axis
    scores. Only the top-k vertex experts selected by that distribution execute
    for each batch item. This is dynamic sparse routing at the pathway level.
    """

    def __init__(self, d_model: int = 64, top_k: int = 5) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.top_k = max(1, min(int(top_k), 16))
        self.experts = nn.ModuleList([ExpertMLP(self.d_model) for _ in range(16)])
        self.fuse = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.register_buffer("vertex_bits", _all_vertex_bits(), persistent=False)

    def vertex_probabilities(self, axis_scores: torch.Tensor) -> torch.Tensor:
        axis_scores = axis_scores.clamp(1e-5, 1.0 - 1e-5)
        bits = self.vertex_bits.to(axis_scores.device)
        probs = axis_scores.unsqueeze(1) * bits.unsqueeze(0)
        probs = probs + (1.0 - axis_scores).unsqueeze(1) * (1.0 - bits.unsqueeze(0))
        return probs.prod(dim=-1)

    def route(self, axis_scores: torch.Tensor) -> dict[str, torch.Tensor]:
        vertex_probs = self.vertex_probabilities(axis_scores)
        top_vals, top_idx = torch.topk(vertex_probs, k=self.top_k, dim=-1)
        top_weights = top_vals / top_vals.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        hard_vertex = top_idx[:, 0]
        return {
            "vertex_probs": vertex_probs,
            "vertex_logits": torch.log(vertex_probs.clamp_min(1e-8)),
            "topk_indices": top_idx,
            "topk_weights": top_weights,
            "selected_vertex": hard_vertex,
        }

    def forward(self, hidden: torch.Tensor, axis_scores: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        route = self.route(axis_scores)
        top_idx = route["topk_indices"]
        top_weights = route["topk_weights"]

        routed = hidden.new_zeros(hidden.shape)
        usage = hidden.new_zeros((16,))

        for expert_id, expert in enumerate(self.experts):
            matches = top_idx == expert_id
            rows = matches.any(dim=1)
            if bool(rows.any().item()):
                usage[expert_id] = 1.0
                row_weight = (matches.float() * top_weights).sum(dim=1, keepdim=True)
                expert_out = expert(hidden[rows])
                routed_rows = row_weight[rows] * expert_out
                routed = routed.index_add(0, rows.nonzero(as_tuple=False).squeeze(-1), routed_rows)

        route["expert_usage"] = usage
        route["sparse_ratio"] = usage.sum() / usage.numel()
        return self.fuse(routed), route


class TesseractPathwayBlock(TesseractSparseDispatcher):
    """Backward-compatible alias for v0.1/v0.2 imports."""


class TesseractPathwayNetwork(nn.Module):
    def __init__(self, input_dim: int = 16, d_model: int = 64, num_routes: int = 5, top_k: int = 5) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.num_routes = int(num_routes)
        self.top_k = max(1, min(int(top_k), 16))
        self.input = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.axis_head = nn.Linear(self.d_model, 4)
        self.block = TesseractSparseDispatcher(d_model=self.d_model, top_k=self.top_k)
        self.route_head = nn.Linear(self.d_model, self.num_routes)
        self.authority_head = nn.Linear(self.d_model, 3)
        self.evidence_head = nn.Linear(self.d_model, 3)
        self.coherence_head = nn.Linear(self.d_model, 1)
        self.delta_phi_head = nn.Linear(self.d_model, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 2:
            raise ValueError(f"TPN expects [batch, input_dim], got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"TPN expected input_dim={self.input_dim}, got {x.shape[-1]}")
        hidden = self.input(x)
        axis_scores = torch.sigmoid(self.axis_head(hidden))
        routed, route = self.block(hidden, axis_scores)
        return {
            "axis_scores": axis_scores,
            "vertex_probs": route["vertex_probs"],
            "vertex_logits": route["vertex_logits"],
            "topk_indices": route["topk_indices"],
            "topk_weights": route["topk_weights"],
            "selected_vertex": route["selected_vertex"],
            "expert_usage": route["expert_usage"],
            "sparse_ratio": route["sparse_ratio"],
            "route_logits": self.route_head(routed),
            "authority_logits": self.authority_head(routed),
            "evidence_logits": self.evidence_head(routed),
            "coherence": torch.sigmoid(self.coherence_head(routed)).squeeze(-1),
            "delta_phi": F.softplus(self.delta_phi_head(routed)).squeeze(-1),
            "hidden": routed,
        }

    def make_receipts(self, outputs: dict[str, torch.Tensor], max_items: int | None = None) -> list[dict[str, object]]:
        from neuralforge.tesseract.receipt import build_tesseract_receipts

        return build_tesseract_receipts(outputs, max_items=max_items)
