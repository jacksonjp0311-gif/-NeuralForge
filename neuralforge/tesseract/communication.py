
"""Local English communication for the Tesseract mind core.

This is not an external language model. It converts TPN receipts into clear,
deterministic English so the local mind core can explain its internal route.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from neuralforge.tesseract.mind import TesseractMindCore
from neuralforge.tesseract.receipt import build_tesseract_receipts

ROUTE_EXPLANATIONS = {
    "engage": "The system sees enough intent, evidence, authority, and context to engage.",
    "repair_axis": "The system found one or more weak axes and should repair the state before acting.",
    "authority_required": "The system found authority missing or insufficient and should request permission.",
    "shadow": "The system should stay in shadow mode because action authority is not established.",
    "rehydrate_or_retrieve": "The system should reload context or retrieve memory before proceeding.",
    "unknown": "The system produced an unknown route label and should not treat this as a confident action.",
}

AXIS_MEANINGS = {
    "intent": "intent clarity",
    "evidence": "evidence support",
    "authority": "permission or authority",
    "context": "context alignment",
}


@dataclass(frozen=True)
class TesseractEnglishMessage:
    route: str
    selected_vertex: str
    summary: str
    explanation: str
    recommended_action: str
    receipt: dict[str, Any]

    def as_text(self) -> str:
        return "\n".join([
            self.summary,
            self.explanation,
            self.recommended_action,
        ])


def _fmt_score(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def _axis_sentence(axis_scores: dict[str, Any]) -> str:
    parts = []
    for axis in ("intent", "evidence", "authority", "context"):
        if axis in axis_scores:
            parts.append(f"{AXIS_MEANINGS[axis]}={_fmt_score(axis_scores[axis])}")
    return ", ".join(parts)


def _missing_sentence(missing_axes: Iterable[str]) -> str:
    missing = list(missing_axes)
    if not missing:
        return "No required axis is missing."
    readable = [AXIS_MEANINGS.get(axis, axis) for axis in missing]
    return "Missing or weak axis: " + ", ".join(readable) + "."


def _recommendation(route: str) -> str:
    if route == "engage":
        return "Recommended action: proceed, while preserving the receipt."
    if route == "repair_axis":
        return "Recommended action: repair the missing or weak axis before acting."
    if route == "authority_required":
        return "Recommended action: request explicit authority before mutation or external action."
    if route == "shadow":
        return "Recommended action: remain local and observational; do not mutate durable state."
    if route == "rehydrate_or_retrieve":
        return "Recommended action: retrieve memory or rehydrate context, then rerun the route."
    return "Recommended action: stop and inspect the receipt before proceeding."


def receipt_to_english(receipt: dict[str, Any], *, style: str = "plain") -> TesseractEnglishMessage:
    route = str(receipt.get("route", "unknown"))
    vertex = str(receipt.get("selected_vertex", "????"))
    missing = receipt.get("missing_axes", [])
    axis_scores = dict(receipt.get("axis_scores", {}))
    topk = receipt.get("topk_vertices", [])
    topk_weights = receipt.get("topk_weights", [])

    summary = (
        f"TPN selected vertex {vertex} and routed to {route}. "
        f"Coherence={_fmt_score(receipt.get('coherence'))}; "
        f"delta_phi={_fmt_score(receipt.get('delta_phi'))}."
    )

    explanation = " ".join([
        ROUTE_EXPLANATIONS.get(route, ROUTE_EXPLANATIONS["unknown"]),
        _missing_sentence(missing),
        "Axis scores: " + _axis_sentence(axis_scores) + ".",
        "Top pathway candidates: " + ", ".join(
            f"{v}@{_fmt_score(w)}" for v, w in zip(topk, topk_weights)
        ) + ".",
    ])

    recommended_action = _recommendation(route)

    if style == "compact":
        summary = f"{route.upper()} via {vertex}: C={_fmt_score(receipt.get('coherence'))}, ΔΦ={_fmt_score(receipt.get('delta_phi'))}."
        explanation = _missing_sentence(missing)
    if style == "operator":
        recommended_action = "Operator note: " + recommended_action.replace("Recommended action: ", "")

    return TesseractEnglishMessage(
        route=route,
        selected_vertex=vertex,
        summary=summary,
        explanation=explanation,
        recommended_action=recommended_action,
        receipt=receipt,
    )


def receipts_to_english(receipts: list[dict[str, Any]], *, style: str = "plain") -> list[TesseractEnglishMessage]:
    return [receipt_to_english(receipt, style=style) for receipt in receipts]


def outputs_to_english(outputs: dict[str, torch.Tensor], *, style: str = "plain", max_items: int | None = None) -> list[TesseractEnglishMessage]:
    receipts = build_tesseract_receipts(outputs, max_items=max_items)
    return receipts_to_english(receipts, style=style)


class TesseractEnglishAdapter:
    """Load local TPN weights and communicate route decisions in English."""

    def __init__(self, core: TesseractMindCore) -> None:
        self.core = core

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> "TesseractEnglishAdapter":
        return cls(TesseractMindCore.from_checkpoint(path, map_location=map_location))

    def think_english(self, vector: list[float] | tuple[float, ...], *, style: str = "plain") -> dict[str, Any]:
        result = self.core.think_from_vector(vector, receipts=True)
        messages = receipts_to_english(result["receipts"], style=style)
        return {
            "text": messages[0].as_text() if messages else "",
            "messages": messages,
            "receipts": result["receipts"],
            "selected_vertices": result["selected_vertices"],
            "route_ids": result["route_ids"],
            "claim_boundary": "Local deterministic English from TPN receipts; not external language generation.",
        }
