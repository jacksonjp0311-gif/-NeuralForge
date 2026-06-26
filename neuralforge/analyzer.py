"""
NeuralForge Workflow Analyzer — Predict failures & optimize AGNT workflows.

Uses NeuralForge's Pattern Engine + Data Learner to:
  1. Analyze workflow execution history (duration, success/failure, step counts)
  2. Detect patterns (degradation trends, periodic failures, anomaly runs)
  3. Predict next execution outcome (success/failure + estimated duration)
  4. Recommend optimizations (slow steps, retry patterns, resource usage)

This is the "always-called" tool — every workflow owner wants to know
"will my workflow fail next time?" and "why is it getting slower?"
"""
from __future__ import annotations
import logging, time, json
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from neuralforge.pattern_engine import PatternEngine, PatternType
from neuralforge.learner import DataLearner

logger = logging.getLogger("neuralforge.analyzer")


class WorkflowAnalyzer:
    """Analyzes AGNT workflow execution data to predict failures and optimize performance."""

    def __init__(self):
        self.pattern_engine = PatternEngine()
        self.learner = DataLearner(device=torch.device("cpu"))

    def analyze(
        self,
        executions: List[Dict[str, Any]],
        predict_next: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze workflow execution history.

        Args:
            executions: List of execution records, each with:
                - duration_ms: execution duration in milliseconds
                - success: true/false
                - step_count: number of steps executed
                - error_type: error type if failed (optional)
                - timestamp: ISO timestamp (optional)
            predict_next: whether to predict the next execution outcome

        Returns:
            Analysis results with patterns, predictions, and recommendations
        """
        if len(executions) < 3:
            return {
                "status": "error",
                "error": "Need at least 3 execution records, got %d" % len(executions),
            }

        results = {}
        n = len(executions)

        # ── Extract metrics ──
        durations = np.array([e.get("duration_ms", 0) for e in executions], dtype=np.float32)
        successes = np.array([1.0 if e.get("success", True) else 0.0 for e in executions], dtype=np.float32)
        step_counts = np.array([float(e.get("step_count", 0)) for e in executions], dtype=np.float32)

        # ── Basic statistics ──
        results["stats"] = {
            "total_executions": n,
            "success_rate": round(float(np.mean(successes)), 4),
            "health_score": round(float(np.mean(successes)) * 100, 1),
            "avg_duration_ms": round(float(np.mean(durations)), 1),
            "min_duration_ms": round(float(np.min(durations)), 1),
            "max_duration_ms": round(float(np.max(durations)), 1),
            "std_duration_ms": round(float(np.std(durations)), 1),
            "avg_step_count": round(float(np.mean(step_counts)), 1),
            "failure_count": int(np.sum(successes == 0)),
        }

        # ── Pattern Analysis on durations ──
        duration_pattern = self.pattern_engine.analyze(
            durations.tolist(), predict_steps=3, epochs=50
        )
        results["duration_pattern"] = {
            "type": duration_pattern.get("pattern_type", "unknown"),
            "confidence": duration_pattern.get("confidence", 0),
            "predicted_next_durations": duration_pattern.get("predictions", []),
        }

        # ── Pattern Analysis on success rate (rolling window) ──
        if n >= 5:
            window = min(5, n // 2)
            rolling_success = []
            for i in range(window, n + 1):
                rolling_success.append(float(np.mean(successes[i - window:i])))
            if len(rolling_success) >= 3:
                success_pattern = self.pattern_engine.analyze(
                    rolling_success, predict_steps=2, epochs=30
                )
                results["success_pattern"] = {
                    "type": success_pattern.get("pattern_type", "unknown"),
                    "confidence": success_pattern.get("confidence", 0),
                    "predicted_next_success_rate": success_pattern.get("predictions", []),
                }

        # ── Anomaly Detection: find outlier executions ──
        if n >= 5:
            duration_zscores = np.abs((durations - np.mean(durations)) / (np.std(durations) + 1e-10))
            anomaly_indices = np.where(duration_zscores > 2.0)[0].tolist()
            results["anomalies"] = {
                "count": len(anomaly_indices),
                "indices": anomaly_indices,
                "details": [
                    {
                        "index": int(idx),
                        "duration_ms": float(durations[idx]),
                        "z_score": round(float(duration_zscores[idx]), 2),
                        "success": bool(successes[idx] > 0.5),
                    }
                    for idx in anomaly_indices
                ],
            }

        # ── Trend Analysis ──
        if n >= 3:
            # Duration trend
            x = np.arange(n, dtype=np.float32)
            if np.std(durations) > 0:
                dur_corr = float(np.corrcoef(x, durations)[0, 1])
            else:
                dur_corr = 0.0
            results["trends"] = {
                "duration_trend": "increasing" if dur_corr > 0.3 else ("decreasing" if dur_corr < -0.3 else "stable"),
                "duration_correlation": round(dur_corr, 4),
                "success_trend": "improving" if np.mean(successes[-min(3, n):]) > np.mean(successes[:min(3, n)]) else "degrading" if np.mean(successes[-min(3, n):]) < np.mean(successes[:min(3, n)]) else "stable",
            }

        # ── Predict next execution ──
        if predict_next and n >= 3:
            # Use duration pattern to predict next duration
            predicted_durations = results["duration_pattern"].get("predicted_next_durations", [])
            if predicted_durations:
                results["prediction"] = {
                    "estimated_duration_ms": round(float(predicted_durations[0]), 1),
                    "estimated_success_probability": round(float(np.mean(successes[-min(5, n):])), 4),
                    "risk_level": "high" if results["stats"]["success_rate"] < 0.7 else ("medium" if results["stats"]["success_rate"] < 0.9 else "low"),
                }

        # ── Recommendations ──
        recommendations = []
        if results["stats"]["success_rate"] < 0.8:
            recommendations.append({
                "type": "reliability",
                "priority": "high",
                "message": "Success rate is %.1f%%. Investigate failure patterns." % (results["stats"]["success_rate"] * 100),
            })
        if results["trends"]["duration_trend"] == "increasing":
            recommendations.append({
                "type": "performance",
                "priority": "medium",
                "message": "Duration is trending upward. Check for resource leaks or growing data.",
            })
        if results.get("anomalies", {}).get("count", 0) > 0:
            recommendations.append({
                "type": "anomaly",
                "priority": "medium",
                "message": "%d anomalous execution(s) detected (z-score > 2.0). Review these runs." % results["anomalies"]["count"],
            })
        if results["stats"]["std_duration_ms"] > results["stats"]["avg_duration_ms"] * 0.5:
            recommendations.append({
                "type": "consistency",
                "priority": "low",
                "message": "High duration variance (CV=%.1f%%). Workflow timing is unpredictable." % (results["stats"]["std_duration_ms"] / (results["stats"]["avg_duration_ms"] + 1e-10) * 100),
            })
        if not recommendations:
            recommendations.append({
                "type": "health",
                "priority": "info",
                "message": "Workflow is healthy. Success rate %.1f%%, stable duration." % (results["stats"]["success_rate"] * 100),
            })
        results["recommendations"] = recommendations

        results["status"] = "success"
        return results
