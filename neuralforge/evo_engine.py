"""
NeuralForge Evolution Engine — AGNT's Self-Improvement Brain.

This is the meta-tool that uses ALL of NeuralForge's capabilities to help AGNT evolve:

1. OBSERVES: Collects workflow execution data from AGNT
2. LEARNS: Uses DataLearner to find patterns in what makes workflows succeed/fail
3. PREDICTS: Uses Smart Engine to forecast workflow outcomes
4. OPTIMIZES: Uses Pattern Engine to detect degradation trends
5. EVOLVES: Recommends and applies improvements over time

This is the highest-value tool because it makes AGNT itself smarter over time.
Every workflow execution makes the system learn. Every prediction improves.
"""
from __future__ import annotations
import logging, time, json
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from neuralforge.analyzer import WorkflowAnalyzer
from neuralforge.smart_engine import SmartEngine
from neuralforge.learner import DataLearner
from neuralforge.pattern_engine import PatternEngine

logger = logging.getLogger("neuralforge.evo")


class EvolutionEngine:
    """AGNT's self-improvement brain. Uses all NeuralForge tools internally."""

    def __init__(self):
        self.analyzer = WorkflowAnalyzer()
        self.smart = SmartEngine()
        self.learner = DataLearner(device=torch.device("cpu"))
        self.pattern_engine = PatternEngine()
        self._knowledge_base: List[Dict] = []

    def evolve(
        self,
        executions: List[Dict[str, Any]],
        workflows: Optional[List[Dict]] = None,
        focus: str = "all",
    ) -> Dict[str, Any]:
        """
        Full evolution cycle: observe → learn → predict → optimize → recommend.

        Args:
            executions: Workflow execution history. Each record:
                - workflow_id: which workflow
                - duration_ms: execution time
                - success: true/false
                - step_count: number of steps
                - error_type: error type if failed
                - timestamp: when it ran
                - workflow_name: name of the workflow
            workflows: Optional workflow definitions
            focus: What to focus on — "all", "failures", "performance", "patterns"

        Returns:
            Evolution report with insights, predictions, and recommendations
        """
        t0 = time.time()
        results = {"status": "success"}

        if len(executions) < 3:
            return {
                "status": "error",
                "error": "Need at least 3 execution records, got %d" % len(executions),
                "hint": "Collect more workflow execution data before running evolution analysis.",
            }

        # ── PHASE 1: Observe — Overall system health ──
        results["observation"] = self._observe(executions)

        # ── PHASE 2: Learn — Per-workflow analysis ──
        if focus in ("all", "patterns"):
            results["learning"] = self._learn(executions)

        # ── PHASE 3: Predict — What will fail next ──
        if focus in ("all", "failures"):
            results["predictions"] = self._predict(executions)

        # ── PHASE 4: Optimize — Degradation trends ──
        if focus in ("all", "performance"):
            results["optimization"] = self._optimize(executions)

        # ── PHASE 5: Evolve — Actionable recommendations ──
        results["evolution"] = self._evolve(executions, results)

        results["training_time_seconds"] = round(time.time() - t0, 2)
        return results

    def _observe(self, executions):
        """Phase 1: Overall system health observation."""
        n = len(executions)
        successes = sum(1 for e in executions if e.get("success", True))
        failures = n - successes
        durations = [float(e.get("duration_ms", 0)) for e in executions]

        # Group by workflow
        workflows: Dict[str, List[Dict]] = {}
        for e in executions:
            wid = e.get("workflow_id", e.get("workflow_name", "unknown"))
            workflows.setdefault(wid, []).append(e)

        workflow_stats = {}
        for wid, exs in workflows.items():
            ws = sum(1 for e in exs if e.get("success", True))
            wd = [float(e.get("duration_ms", 0)) for e in exs]
            workflow_stats[wid] = {
                "executions": len(exs),
                "success_rate": round(ws / len(exs), 3),
                "avg_duration_ms": round(float(np.mean(wd)), 1) if wd else 0,
                "failure_count": len(exs) - ws,
            }

        return {
            "total_executions": n,
            "overall_success_rate": round(successes / n, 4),
            "total_failures": failures,
            "avg_duration_ms": round(float(np.mean(durations)), 1) if durations else 0,
            "workflows_analyzed": len(workflows),
            "workflow_stats": workflow_stats,
            "health_score": round(successes / n * 100, 1),
        }

    def _learn(self, executions):
        """Phase 2: Learn patterns from execution data."""
        # Use pattern engine on overall success rate over time
        window = min(5, len(executions) // 3)
        if window < 2:
            window = 2

        rolling_success = []
        for i in range(window, len(executions) + 1):
            window_data = executions[i - window:i]
            rate = sum(1 for e in window_data if e.get("success", True)) / window
            rolling_success.append(rate)

        if len(rolling_success) >= 3:
            pattern = self.pattern_engine.analyze(
                rolling_success, predict_steps=3, epochs=30
            )
            success_trend = {
                "pattern": pattern.get("pattern_type", "unknown"),
                "confidence": pattern.get("confidence", 0),
                "predicted_next_success_rates": [round(float(p), 3) for p in pattern.get("predictions", [])],
            }
        else:
            success_trend = {"pattern": "insufficient_data"}

        # Learn what features correlate with failure
        if len(executions) >= 10:
            # Extract features: duration, step_count, hour_of_day
            features = []
            labels = []
            for e in executions:
                dur = float(e.get("duration_ms", 0))
                steps = float(e.get("step_count", 0))
                ts = e.get("timestamp", "")
                hour = 12  # default
                if ts and "T" in ts:
                    try:
                        hour = int(ts.split("T")[1].split(":")[0])
                    except:
                        pass
                features.append([dur, steps, hour])
                labels.append(1.0 if e.get("success", True) else 0.0)

            if len(set(labels)) >= 2:
                X = np.array(features, dtype=np.float32)
                y = np.array(labels, dtype=np.float32)
                # Normalize
                X_mean = np.mean(X, axis=0)
                X_std = np.std(X, axis=0) + 1e-8
                X_norm = (X - X_mean) / X_std

                # Simple correlation analysis
                correlations = []
                for i in range(X_norm.shape[1]):
                    corr = float(np.corrcoef(X_norm[:, i], y)[0, 1])
                    if np.isnan(corr): corr = 0.0
                    correlations.append(round(corr, 4))

                feature_names = ["duration_ms", "step_count", "hour_of_day"]
                feature_importance = dict(zip(feature_names, correlations))
            else:
                feature_importance = {}
        else:
            feature_importance = {}

        return {
            "success_trend": success_trend,
            "feature_importance": feature_importance,
            "data_points": len(executions),
        }

    def _predict(self, executions):
        """Phase 3: Predict which workflows will fail next."""
        # Group by workflow and predict each
        workflows: Dict[str, List[Dict]] = {}
        for e in executions:
            wid = e.get("workflow_id", e.get("workflow_name", "unknown"))
            workflows.setdefault(wid, []).append(e)

        predictions = {}
        for wid, exs in workflows.items():
            if len(exs) < 3:
                continue

            # Use smart engine for prediction
            pred = self.smart.decide("predict", history=[
                {"success": e.get("success", True), "duration_ms": e.get("duration_ms", 0)}
                for e in exs
            ])

            # Use analyzer for anomaly detection
            durations = [float(e.get("duration_ms", 0)) for e in exs]
            analysis = self.analyzer.analyze([
                {"duration_ms": d, "success": e.get("success", True), "step_count": e.get("step_count", 5)}
                for d, e in zip(durations, exs)
            ])

            predictions[wid] = {
                "will_succeed": pred.get("decision", "unknown"),
                "confidence": pred.get("confidence", 0),
                "success_probability": pred.get("success_probability", 0),
                "risk_level": analysis.get("prediction", {}).get("risk_level", "unknown"),
                "trend": analysis.get("trends", {}).get("duration_trend", "unknown"),
                "anomaly_count": analysis.get("anomalies", {}).get("count", 0),
            }

        return {
            "workflow_predictions": predictions,
            "high_risk_workflows": [
                wid for wid, p in predictions.items()
                if p.get("risk_level") == "high" or p.get("will_succeed") == "will_fail"
            ],
        }

    def _optimize(self, executions):
        """Phase 4: Find optimization opportunities."""
        durations = [float(e.get("duration_ms", 0)) for e in executions]

        if len(durations) >= 5:
            pattern = self.pattern_engine.analyze(durations, predict_steps=3, epochs=30)
            optimization = {
                "duration_pattern": pattern.get("pattern_type", "unknown"),
                "predicted_next_durations": [round(float(p), 1) for p in pattern.get("predictions", [])],
                "correlation": pattern.get("training_correlation", 0),
            }
        else:
            optimization = {"duration_pattern": "insufficient_data"}

        # Find slowest workflows
        workflows: Dict[str, List[float]] = {}
        for e in executions:
            wid = e.get("workflow_id", e.get("workflow_name", "unknown"))
            workflows.setdefault(wid, []).append(float(e.get("duration_ms", 0)))            avg_durations = {wid: float(np.mean(dur)) for wid, dur in workflows.items()}
        sorted_workflows = sorted(avg_durations.items(), key=lambda x: x[1], reverse=True)

        optimization["slowest_workflows"] = [
            {"workflow": wid, "avg_duration_ms": round(dur, 1)}
            for wid, dur in sorted_workflows[:5]
        ]

        return optimization

    def _evolve(self, executions, results):
        """Phase 5: Generate actionable evolution recommendations."""
        recommendations = []
        observation = results.get("observation", {})
        learning = results.get("learning", {})
        predictions = results.get("predictions", {})
        optimization = results.get("optimization", {})

        # Health-based recommendations
        health = observation.get("health_score", 100)
        if health < 70:
            recommendations.append({
                "priority": "critical",
                "category": "reliability",
                "action": "System health is %.1f%%. Investigate failing workflows immediately." % health,
                "workflows": [
                    wid for wid, s in observation.get("workflow_stats", {}).items()
                    if s.get("success_rate", 1) < 0.7
                ],
            })
        elif health < 90:
            recommendations.append({
                "priority": "high",
                "category": "reliability",
                "action": "System health is %.1f%%. Review workflows with <80%% success rate." % health,
            })

        # Trend-based recommendations
        trend = learning.get("success_trend", {})
        if trend.get("pattern") == "decreasing":
            recommendations.append({
                "priority": "high",
                "category": "trend",
                "action": "Success rate is trending downward. Check for systemic issues.",
            })

        # Feature importance recommendations
        fi = learning.get("feature_importance", {})
        if fi:
            most_important = max(fi, key=lambda k: abs(fi[k]))
            if abs(fi[most_important]) > 0.3:
                recommendations.append({
                    "priority": "medium",
                    "category": "insight",
                    "action": "'%s' is the strongest predictor of success (corr=%.3f). Focus optimization here." % (most_important, fi[most_important]),
                })

        # Prediction-based recommendations
        high_risk = predictions.get("high_risk_workflows", [])
        if high_risk:
            recommendations.append({
                "priority": "high",
                "category": "prediction",
                "action": "%d workflow(s) predicted to fail: %s" % (len(high_risk), ", ".join(high_risk[:3])),
            })

        # Performance recommendations
        slowest = optimization.get("slowest_workflows", [])
        if slowest and slowest[0].get("avg_duration_ms", 0) > 0:
            recommendations.append({
                "priority": "medium",
                "category": "performance",
                "action": "Slowest workflow: '%s' (avg %dms). Consider optimization." % (
                    slowest[0].get("workflow", "?"),
                    int(slowest[0].get("avg_duration_ms", 0)),
                ),
            })

        # Duration trend
        if optimization.get("duration_pattern") == "increasing":
            recommendations.append({
                "priority": "medium",
                "category": "degradation",
                "action": "Overall duration is trending upward. Check for resource leaks or growing data.",
            })

        if not recommendations:
            recommendations.append({
                "priority": "info",
                "category": "health",
                "action": "System is healthy (%.1f%% success rate). No immediate action needed." % health,
            })

        return {
            "recommendation_count": len(recommendations),
            "critical_count": sum(1 for r in recommendations if r.get("priority") == "critical"),
            "high_count": sum(1 for r in recommendations if r.get("priority") == "high"),
            "recommendations": recommendations,
            "evolution_stage": "observing" if len(executions) < 20 else ("learning" if len(executions) < 50 else "optimizing"),
        }
