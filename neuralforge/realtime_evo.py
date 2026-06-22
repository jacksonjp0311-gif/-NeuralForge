"""
NeuralForge Real-Time Evolution Engine — Continuous AGNT Self-Improvement.

Hooks into AGNT's workflow execution pipeline. After each execution:
1. Ingests the new execution data
2. Updates rolling statistics
3. Detects anomalies and degradation trends
4. Predicts failure probability for next run
5. Generates alerts and recommendations
6. Accumulates knowledge over time

This is the engine that makes AGNT self-improving in real-time.
"""
from __future__ import annotations
import logging, time, json
from typing import Dict, List, Optional, Any
from collections import defaultdict

import numpy as np
import torch

from neuralforge.analyzer import WorkflowAnalyzer
from neuralforge.smart_engine import SmartEngine
from neuralforge.pattern_engine import PatternEngine

logger = logging.getLogger("neuralforge.realtime")


class RealtimeEvolutionEngine:
    """Continuously monitors AGNT workflows and improves predictions over time."""

    def __init__(self, window_size: int = 50, alert_threshold: float = 0.3):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.analyzer = WorkflowAnalyzer()
        self.smart = SmartEngine()
        self.pattern_engine = PatternEngine()

        # Rolling state
        self._executions: List[Dict] = []
        self._workflow_history: Dict[str, List[Dict]] = defaultdict(list)
        self._knowledge_base: List[Dict] = []
        self._alert_count = 0
        self._total_processed = 0

    def process_batch(self, executions: List[Dict]) -> Dict[str, Any]:
        """Process a batch of new executions. Called after each workflow run."""
        self._executions.extend(executions)
        self._total_processed += len(executions)

        # Update per-workflow history
        for e in executions:
            wid = e.get("workflow_id", e.get("workflow_name", "unknown"))
            self._workflow_history[wid].append(e)

        # Analyze recent window
        recent = self._executions[-self.window_size:]

        # Compute health score
        successes = sum(1 for e in recent if e.get("success", True))
        health_score = round(successes / len(recent) * 100, 1) if recent else 100

        # Detect anomalies in recent executions
        alerts = self._detect_alerts(recent)

        # Per-workflow predictions
        predictions = self._predict_workflows()

        # Generate recommendations
        recommendations = self._generate_recommendations(recent, health_score, alerts)

        # Store knowledge
        self._knowledge_base.append({
            "timestamp": time.time(),
            "executions_processed": len(executions),
            "health_score": health_score,
            "alert_count": len(alerts),
            "recommendation_count": len(recommendations),
        })

        return {
            "status": "success",
            "total_processed": self._total_processed,
            "health_score": health_score,
            "recent_window_size": len(recent),
            "alert_count": len(alerts),
            "alerts": alerts,
            "workflow_predictions": predictions,
            "recommendations": recommendations,
            "knowledge_entries": len(self._knowledge_base),
        }

    def _detect_alerts(self, recent: List[Dict]) -> List[Dict]:
        """Detect anomalies and degradation in recent executions."""
        alerts = []

        # Check for high failure rate
        failures = [e for e in recent if not e.get("success", True)]
        failure_rate = len(failures) / len(recent) if recent else 0

        if failure_rate > self.alert_threshold:
            alerts.append({
                "type": "high_failure_rate",
                "severity": "critical" if failure_rate > 0.5 else "warning",
                "message": "Failure rate is %.1f%% in recent %d executions" % (failure_rate * 100, len(recent)),
                "value": round(failure_rate, 3),
            })
            self._alert_count += 1

        # Check for duration spikes
        durations = [float(e.get("duration_ms", 0)) for e in recent if e.get("duration_ms", 0) > 0]
        if len(durations) >= 5:
            mean_dur = np.mean(durations)
            std_dur = np.std(durations)
            if std_dur > 0:
                for e in recent[-5:]:
                    dur = float(e.get("duration_ms", 0))
                    if dur > mean_dur + 2 * std_dur:
                        alerts.append({
                            "type": "duration_spike",
                            "severity": "warning",
                            "message": "Duration spike: %.0fms (mean: %.0fms, std: %.0fms)" % (dur, mean_dur, std_dur),
                            "workflow": e.get("workflow_name", "?"),
                        })
                        self._alert_count += 1
                        break

        # Check for degradation trend
        if len(durations) >= 10:
            x = np.arange(len(durations))
            corr = float(np.corrcoef(x, durations)[0, 1])
            if corr > 0.5:
                alerts.append({
                    "type": "degradation_trend",
                    "severity": "warning",
                    "message": "Duration is trending upward (correlation: %.3f)" % corr,
                })
                self._alert_count += 1

        return alerts

    def _predict_workflows(self) -> Dict[str, Dict]:
        """Predict next execution outcome for each workflow."""
        predictions = {}
        for wid, history in self._workflow_history.items():
            if len(history) < 3:
                continue
            recent = history[-10:]
            successes = sum(1 for e in recent if e.get("success", True))
            success_rate = successes / len(recent)

            durations = [float(e.get("duration_ms", 0)) for e in recent]
            avg_dur = float(np.mean(durations)) if durations else 0

            # Risk assessment
            if success_rate < 0.5:
                risk = "high"
            elif success_rate < 0.8:
                risk = "medium"
            else:
                risk = "low"

            predictions[wid] = {
                "success_rate": round(success_rate, 3),
                "avg_duration_ms": round(avg_dur, 1),
                "risk_level": risk,
                "executions": len(history),
            }
        return predictions

    def _generate_recommendations(self, recent, health_score, alerts) -> List[Dict]:
        """Generate actionable recommendations based on analysis."""
        recs = []

        if health_score < 50:
            recs.append({"priority": "critical", "action": "System health is %.1f%% — investigate failing workflows immediately" % health_score})
        elif health_score < 80:
            recs.append({"priority": "high", "action": "System health is %.1f%% — review workflows with <70%% success rate" % health_score})

        for alert in alerts:
            if alert["type"] == "duration_spike":
                recs.append({"priority": "medium", "action": "Duration spike detected — check for resource contention or slow dependencies"})
            elif alert["type"] == "degradation_trend":
                recs.append({"priority": "medium", "action": "Degradation trend — consider scaling resources or optimizing slow steps"})

        if not recs:
            recs.append({"priority": "info", "action": "System is healthy (%.1f%%) — continue monitoring" % health_score})

        return recs

    def get_knowledge_summary(self) -> Dict:
        """Return accumulated knowledge about the system."""
        if not self._knowledge_base:
            return {"status": "no_data"}

        recent_knowledge = self._knowledge_base[-10:]
        health_trend = [k["health_score"] for k in recent_knowledge]

        return {
            "total_executions_processed": self._total_processed,
            "total_alerts": self._alert_count,
            "workflows_tracked": len(self._workflow_history),
            "current_health": health_trend[-1] if health_trend else None,
            "health_trend": "improving" if len(health_trend) > 1 and health_trend[-1] > health_trend[0] else "degrading" if len(health_trend) > 1 and health_trend[-1] < health_trend[0] else "stable",
            "knowledge_entries": len(self._knowledge_base),
        }
