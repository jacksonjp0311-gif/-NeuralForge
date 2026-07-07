"""Interpretable sparse router over a four-axis tesseract."""

from __future__ import annotations

from typing import Mapping

from neuralforge.tesseract.axes import AxisScores, build_route_state
from neuralforge.tesseract.geometry import hamming_distance, neighbors, vertex_id


class TesseractRouter:
    def __init__(
        self,
        *,
        target: str = "1111",
        horizon: float = 0.70,
        thresholds: Mapping[str, float] | None = None,
        neighbor_budget: int = 4,
    ) -> None:
        self.target = target
        self.horizon = float(horizon)
        self.thresholds = dict(thresholds or {})
        self.neighbor_budget = int(neighbor_budget)

    def route(
        self,
        scores: AxisScores | Mapping[str, float],
        *,
        mutation_requested: bool = False,
    ) -> dict[str, object]:
        state = build_route_state(
            scores,
            target=self.target,
            thresholds=self.thresholds or None,
            mutation_requested=mutation_requested,
            horizon=self.horizon,
        )
        selected = self.select_experts(state.vertex)
        return {
            "vertex": state.vertex,
            "vertex_id": vertex_id(state.vertex),
            "target": state.target,
            "route": state.route,
            "missing_axes": state.missing,
            "path": state.path,
            "delta_phi": round(state.delta_phi, 6),
            "coherence": round(state.coherence, 6),
            "selected_experts": selected,
            "scores": state.scores.as_dict(),
            "claim_boundary": "TPN route packet is evidence for pathway selection, not proof of correctness, safety, or authority.",
        }

    def select_experts(self, vertex: str) -> list[dict[str, float | int | str]]:
        pool = [vertex] + neighbors(vertex)
        pool = pool[: max(1, self.neighbor_budget + 1)]
        raw = []
        for v in pool:
            score = 1.0 / (1.0 + hamming_distance(vertex, v))
            score += 0.10 / (1.0 + hamming_distance(v, self.target))
            raw.append((v, score))
        total = sum(score for _, score in raw) or 1.0
        return [
            {"vertex": v, "vertex_id": vertex_id(v), "weight": round(score / total, 6)}
            for v, score in raw
        ]
