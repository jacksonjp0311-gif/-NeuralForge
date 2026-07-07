"""Toy benchmark for tesseract sparse retrieval versus dense scan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)


def run_toy_benchmark(
    *,
    items: int = 1024,
    axis_dim: int = 12,
    top_k: int = 16,
    seed: int = 42,
    output_path: str | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    memory = _normalize(rng.normal(size=(items, 4, axis_dim)))
    query = _normalize(rng.normal(size=(4, axis_dim)))

    axis_scores = (memory * query[None, :, :]).sum(axis=-1)
    full_score = axis_scores.sum(axis=-1) + 0.1 * (axis_scores[:, 0] * axis_scores[:, 1])
    truth = set(np.argsort(full_score)[-top_k:].tolist())

    dense_candidates = set(range(items))

    high = axis_scores >= np.quantile(axis_scores, 0.60, axis=0, keepdims=True)
    route_score = high.sum(axis=-1)
    candidate_count = max(top_k * 4, int(items * 0.18))
    tpn_candidates = set(np.argsort(route_score + 0.01 * full_score)[-candidate_count:].tolist())

    random_candidates = set(rng.choice(items, size=candidate_count, replace=False).tolist())

    def recall(cands: set[int]) -> float:
        return len(truth & cands) / max(1, len(truth))

    report = {
        "items": items,
        "axis_dim": axis_dim,
        "top_k_truth": top_k,
        "dense": {
            "candidate_count": len(dense_candidates),
            "candidate_fraction": 1.0,
            "recall": recall(dense_candidates),
        },
        "random_sparse": {
            "candidate_count": len(random_candidates),
            "candidate_fraction": len(random_candidates) / items,
            "recall": recall(random_candidates),
        },
        "tesseract_sparse": {
            "candidate_count": len(tpn_candidates),
            "candidate_fraction": len(tpn_candidates) / items,
            "recall": recall(tpn_candidates),
        },
        "claim_boundary": "Toy synthetic benchmark only; it does not prove production model efficiency.",
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + chr(10), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run_toy_benchmark(output_path="reports/tpn_benchmark_latest.json"), indent=2, sort_keys=True))
