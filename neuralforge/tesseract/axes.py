"""Axis scoring and route-state construction for TPN."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from neuralforge.tesseract.geometry import AXES, bits_to_vertex, missing_axes, shortest_path


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class AxisScores:
    intent: float = 0.0
    evidence: float = 0.0
    authority: float = 0.0
    context: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, float]) -> "AxisScores":
        return cls(
            intent=_clamp01(float(data.get("intent", 0.0))),
            evidence=_clamp01(float(data.get("evidence", 0.0))),
            authority=_clamp01(float(data.get("authority", 0.0))),
            context=_clamp01(float(data.get("context", 0.0))),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "intent": _clamp01(self.intent),
            "evidence": _clamp01(self.evidence),
            "authority": _clamp01(self.authority),
            "context": _clamp01(self.context),
        }

    def as_bits(self, thresholds: Mapping[str, float] | None = None) -> tuple[int, int, int, int]:
        thresholds = thresholds or {axis: 0.5 for axis in AXES}
        d = self.as_dict()
        return tuple(1 if d[axis] >= float(thresholds.get(axis, 0.5)) else 0 for axis in AXES)  # type: ignore[return-value]

    def vertex(self, thresholds: Mapping[str, float] | None = None) -> str:
        return bits_to_vertex(self.as_bits(thresholds))  # type: ignore[return-value]


@dataclass(frozen=True)
class TesseractRouteState:
    scores: AxisScores
    vertex: str
    target: str
    delta_phi: float
    coherence: float
    missing: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    route: str = "engage"


def compute_delta_phi(scores: AxisScores) -> float:
    values = list(scores.as_dict().values())
    return max(values) - min(values)


def compute_coherence(scores: AxisScores, delta_phi: float) -> float:
    d = scores.as_dict()
    return (d["intent"] * d["evidence"]) / (1.0 + abs(delta_phi))


def classify_route(
    vertex: str,
    coherence: float,
    missing: list[str],
    mutation_requested: bool = False,
    horizon: float = 0.70,
) -> str:
    if mutation_requested and "authority" in missing:
        return "shadow"
    if "authority" in missing:
        return "authority_required"
    if coherence < horizon and missing:
        return "repair_axis"
    if coherence < horizon:
        return "rehydrate_or_retrieve"
    if missing:
        return "repair_axis"
    return "engage"


def build_route_state(
    scores: AxisScores | Mapping[str, float],
    *,
    target: str = "1111",
    thresholds: Mapping[str, float] | None = None,
    mutation_requested: bool = False,
    horizon: float = 0.70,
) -> TesseractRouteState:
    if not isinstance(scores, AxisScores):
        scores = AxisScores.from_mapping(scores)

    vertex = scores.vertex(thresholds)
    delta = compute_delta_phi(scores)
    coherence = compute_coherence(scores, delta)
    missing = missing_axes(vertex, target=target)
    path = shortest_path(vertex, target=target)
    route = classify_route(vertex, coherence, missing, mutation_requested=mutation_requested, horizon=horizon)

    return TesseractRouteState(
        scores=scores,
        vertex=vertex,
        target=target,
        delta_phi=delta,
        coherence=coherence,
        missing=missing,
        path=path,
        route=route,
    )
