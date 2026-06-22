"""
NeuralForge Evolution Engine — AGNT's Self-Improvement Brain.

Uses ALL NeuralForge tools internally:
  - WorkflowAnalyzer: execution pattern analysis
  - Smart Engine: retry/optimize/predict/fix decisions
  - DataLearner: pattern learning from execution data
  - Pattern Engine: trend/seasonal/chaotic detection

5-phase evolution cycle: observe → learn → predict → optimize → evolve
"""
from __future__ import annotations
import logging, time
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from neuralforge.analyzer import WorkflowAnalyzer
from neuralforge.smart_engine import SmartEngine
from neuralforge.learner import DataLearner
from neuralforge.pattern_engine import PatternEngine

logger = logging.getLogger("neuralforge.evo")


class EvolutionEngine:
    def __init__(self):
        self.analyzer = WorkflowAnalyzer()
        self.smart = SmartEngine()
        self.learner = DataLearner(device=torch.device("cpu"))
        self.pattern_engine = PatternEngine()

    def evolve(self, executions, workflows=None, focus="all"):
        t0 = time.time()
        results = {"status": "success"}
        if len(executions) < 3:
            return {"status": "error", "error": "Need 3+ executions, got %d" % len(executions)}
        results["observation"] = self._observe(executions)
        if focus in ("all", "patterns"):
            results["learning"] = self._learn(executions)
        if focus in ("all", "failures"):
            results["predictions"] = self._predict(executions)
        if focus in ("all", "performance"):
            results["optimization"] = self._optimize(executions)
        results["evolution"] = self._evolve(executions, results)
        results["training_time_seconds"] = round(time.time() - t0, 2)
        return results

    def _observe(self, executions):
        n = len(executions)
        successes = sum(1 for e in executions if e.get("success", True))
        durations = [float(e.get("duration_ms", 0)) for e in executions]
        workflows = {}
        for e in executions:
            wid = e.get("workflow_id", e.get("workflow_name", "unknown"))
            workflows.setdefault(wid, []).append(e)
        workflow_stats = {}
        for wid, exs in workflows.items():
            ws = sum(1 for e in exs if e.get("success", True))
            wd = [float(e.get("duration_ms", 0)) for e in exs]
            workflow_stats[wid] = {"executions": len(exs), "success_rate": round(ws / len(exs), 3), "avg_duration_ms": round(float(np.mean(wd)), 1) if wd else 0, "failure_count": len(exs) - ws}
        return {"total_executions": n, "overall_success_rate": round(successes / n, 4), "total_failures": n - successes, "avg_duration_ms": round(float(np.mean(durations)), 1) if durations else 0, "workflows_analyzed": len(workflows), "workflow_stats": workflow_stats, "health_score": round(successes / n * 100, 1)}

    def _learn(self, executions):
        window = max(2, min(5, len(executions) // 3))
        rolling = []
        for i in range(window, len(executions) + 1):
            rate = sum(1 for e in executions[i-window:i] if e.get("success", True)) / window
            rolling.append(rate)
        if len(rolling) >= 3:
            pat = self.pattern_engine.analyze(rolling, predict_steps=3, epochs=30)
            trend = {"pattern": pat.get("pattern_type", "unknown"), "confidence": pat.get("confidence", 0), "predicted_next": [round(float(p), 3) for p in pat.get("predictions", [])]}
        else:
            trend = {"pattern": "insufficient_data"}
        return {"success_trend": trend, "data_points": len(executions)}

    def _predict(self, executions):
        workflows = {}
        for e in executions:
            wid = e.get("workflow_id", e.get("workflow_name", "unknown"))
            workflows.setdefault(wid, []).append(e)
        predictions = {}
        for wid, exs in workflows.items():
            if len(exs) < 3: continue
            pred = self.smart.decide("predict", history=[{"success": e.get("success", True), "duration_ms": e.get("duration_ms", 0)} for e in exs])
            durations = [float(e.get("duration_ms", 0)) for e in exs]
            try:
                analysis = self.analyzer.analyze([{"duration_ms": d, "success": e.get("success", True), "step_count": e.get("step_count", 5)} for d, e in zip(durations, exs)])
            except Exception as ex:
                analysis = {"prediction": {}, "trends": {}, "anomalies": {"count": 0}, "stats": {}}
            predictions[wid] = {"will_succeed": pred.get("decision", "unknown"), "confidence": pred.get("confidence", 0), "success_probability": pred.get("success_probability", 0), "risk_level": analysis.get("prediction", {}).get("risk_level", "unknown"), "trend": analysis.get("trends", {}).get("duration_trend", "unknown")}
        return {"workflow_predictions": predictions, "high_risk": [w for w, p in predictions.items() if p.get("risk_level") == "high" or p.get("will_succeed") == "will_fail"]}

    def _optimize(self, executions):
        durations = [float(e.get("duration_ms", 0)) for e in executions]
        if len(durations) >= 5:
            pat = self.pattern_engine.analyze(durations, predict_steps=3, epochs=30)
            opt = {"duration_pattern": pat.get("pattern_type", "unknown"), "predicted_next_durations": [round(float(p), 1) for p in pat.get("predictions", [])], "correlation": pat.get("training_correlation", 0)}
        else:
            opt = {"duration_pattern": "insufficient_data"}
        workflows = {}
        for e in executions:
            wid = e.get("workflow_id", e.get("workflow_name", "unknown"))
            workflows.setdefault(wid, []).append(float(e.get("duration_ms", 0)))
        avg_durs = {wid: float(np.mean(durs)) for wid, durs in workflows.items()}
        sorted_wf = sorted(avg_durs.items(), key=lambda x: x[1], reverse=True)
        opt["slowest_workflows"] = [{"workflow": w, "avg_duration_ms": round(d, 1)} for w, d in sorted_wf[:5]]
        return opt

    def _evolve(self, executions, results):
        recs = []
        obs = results.get("observation", {})
        learning = results.get("learning", {})
        predictions = results.get("predictions", {})
        opt = results.get("optimization", {})
        health = obs.get("health_score", 100)
        if health < 70:
            recs.append({"priority": "critical", "category": "reliability", "action": "Health %.1f%% — investigate failures" % health})
        elif health < 90:
            recs.append({"priority": "high", "category": "reliability", "action": "Health %.1f%% — review <80%% workflows" % health})
        if learning.get("success_trend", {}).get("pattern") == "decreasing":
            recs.append({"priority": "high", "category": "trend", "action": "Success rate trending down"})
        high_risk = predictions.get("high_risk", [])
        if high_risk:
            recs.append({"priority": "high", "category": "prediction", "action": "%d high-risk: %s" % (len(high_risk), ", ".join(high_risk[:3]))})
        slowest = opt.get("slowest_workflows", [])
        if slowest:
            recs.append({"priority": "medium", "category": "performance", "action": "Slowest: '%s' (%dms)" % (slowest[0].get("workflow", "?"), int(slowest[0].get("avg_duration_ms", 0)))})
        if opt.get("duration_pattern") == "increasing":
            recs.append({"priority": "medium", "category": "degradation", "action": "Duration trending up — check for leaks"})
        if not recs:
            recs.append({"priority": "info", "category": "health", "action": "Healthy (%.1f%%)" % health})
        return {"recommendation_count": len(recs), "recommendations": recs, "evolution_stage": "observing" if len(executions) < 20 else ("learning" if len(executions) < 50 else "optimizing")}
