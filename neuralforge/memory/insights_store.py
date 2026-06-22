from __future__ import annotations
import json, logging, os, time
from typing import Any, Dict, List, Optional
logger = logging.getLogger("neuralforge.memory")

class InsightsStore:
    def __init__(self, store_path="./neuralforge_insights", collection_name="neuralforge"):
        self.store_path = store_path; self.collection_name = collection_name; self._memory_store = []
    def add_insight(self, spec_summary, result_summary, metrics, metadata=None):
        insight_id = f"insight-{int(time.time()*1000)}"
        self._memory_store.append({"id": insight_id, "spec": spec_summary, "result": result_summary, "metrics": metrics, "metadata": metadata or {}, "timestamp": time.time()})
        return insight_id
    def retrieve_similar(self, query, n_results=5, min_metric=None):
        query_lower = query.lower(); scored = []
        for r in self._memory_store:
            score = sum(1 for w in query_lower.split() if w in (r.get("spec","") + r.get("result","")).lower())
            if score > 0: scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True); return [r for _, r in scored[:n_results]]
    def get_best_insights(self, metric_name="accuracy", n=5, maximize=True):
        scored = [(r.get("metrics",{}).get(metric_name, 0), r) for r in self._memory_store if metric_name in r.get("metrics",{})]
        scored.sort(key=lambda x: x[0], reverse=maximize); return [r for _, r in scored[:n]]
    def get_statistics(self):
        if not self._memory_store: return {"count": 0}
        all_metrics = {}
        for r in self._memory_store:
            for k, v in r.get("metrics", {}).items():
                if isinstance(v, (int, float)): all_metrics.setdefault(k, []).append(v)
        stats = {"count": len(self._memory_store)}
        for mn, vals in all_metrics.items():
            import numpy as np; arr = np.array(vals)
            stats[mn] = {"mean": float(arr.mean()), "std": float(arr.std()), "min": float(arr.min()), "max": float(arr.max())}
        return stats
