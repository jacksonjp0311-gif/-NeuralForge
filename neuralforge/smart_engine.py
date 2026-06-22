"""
NeuralForge Smart Engine — The "always-called" AGNT tool.

A single universal tool that agents call for ANY data-driven decision:
  - "Should I retry this failed tool call?" → predicts success probability
  - "What's the best parameter for this tool?" → optimizes via pattern learning
  - "Will this workflow step fail?" → failure prediction from history
  - "What's the pattern in this data?" → pattern detection + forecasting
  - "How do I fix this error?" → error pattern matching + recommendation

This is the tool that gets called constantly because every workflow
eventually needs a smart decision based on data patterns.
"""
from __future__ import annotations
import logging, time, json
from typing import Dict, List, Optional, Any

import numpy as np
import torch

from neuralforge.pattern_engine import PatternEngine
from neuralforge.learner import DataLearner
from neuralforge.analyzer import WorkflowAnalyzer

logger = logging.getLogger("neuralforge.smart")


class SmartEngine:
    """Universal decision engine — one tool, infinite use cases."""

    def __init__(self):
        self.pattern_engine = PatternEngine()
        self.learner = DataLearner(device=torch.device("cpu"))
        self.analyzer = WorkflowAnalyzer()

    def decide(
        self,
        context: str,
        data: Optional[List[Dict]] = None,
        options: Optional[List[str]] = None,
        history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Make a smart decision based on data patterns.

        Args:
            context: What decision to make. One of:
                - "retry": Should I retry a failed action?
                - "optimize": What's the best parameter/value?
                - "predict": Will this succeed or fail?
                - "pattern": What's the pattern in this data?
                - "fix": How do I fix this error?
                - "analyze": General data analysis
            data: Current data points (list of numbers or dicts)
            options: Available choices (for optimize/predict)
            history: Past execution history

        Returns:
            Decision with confidence, reasoning, and recommendations
        """
        t0 = time.time()

        if context == "retry":
            return self._decide_retry(data, history, t0)
        elif context == "optimize":
            return self._decide_optimize(data, options, history, t0)
        elif context == "predict":
            return self._decide_predict(data, options, history, t0)
        elif context == "pattern":
            return self._decide_pattern(data, t0)
        elif context == "fix":
            return self._decide_fix(data, history, t0)
        elif context == "analyze":
            return self._decide_analyze(data, history, t0)
        else:
            # Auto-detect context from data
            return self._decide_auto(data, options, history, t0)

    def _decide_retry(self, data, history, t0):
        """Should I retry? Predict success probability from history."""
        if history and len(history) >= 3:
            successes = [1.0 if h.get("success", True) else 0.0 for h in history]
            durations = [float(h.get("duration_ms", h.get("value", 0))) for h in history]

            success_rate = float(np.mean(successes))
            recent_rate = float(np.mean(successes[-min(3, len(successes)):]))

            # Pattern in success/failure
            if len(successes) >= 5:
                pattern_result = self.pattern_engine.analyze(
                    successes, predict_steps=3, epochs=30
                )
                predicted_success = pattern_result.get("predictions", [success_rate])
                trend = pattern_result.get("pattern_type", "unknown")
            else:
                predicted_success = [success_rate]
                trend = "insufficient_data"

            # Duration trend
            if len(durations) >= 3 and np.std(durations) > 0:
                dur_trend = float(np.corrcoef(range(len(durations)), durations)[0, 1])
            else:
                dur_trend = 0.0

            # Decision
            should_retry = recent_rate > 0.5 or (predicted_success[0] > 0.5 if predicted_success else False)
            confidence = max(recent_rate, predicted_success[0] if predicted_success else 0)

            reasoning = []
            if recent_rate > 0.7:
                reasoning.append("Recent success rate is high (%.0f%%)" % (recent_rate * 100))
            elif recent_rate < 0.3:
                reasoning.append("Recent success rate is low (%.0f%%)" % (recent_rate * 100))
            if trend == "increasing":
                reasoning.append("Success rate is trending upward")
            elif trend == "decreasing":
                reasoning.append("Success rate is trending downward")
            if dur_trend > 0.3:
                reasoning.append("Duration is increasing — may indicate degradation")

            return {
                "status": "success",
                "decision": "retry" if should_retry else "don't retry",
                "confidence": round(confidence, 3),
                "success_probability": round(predicted_success[0] if predicted_success else success_rate, 3),
                "success_rate": round(success_rate, 3),
                "recent_success_rate": round(recent_rate, 3),
                "trend": trend,
                "duration_trend": "increasing" if dur_trend > 0.3 else ("decreasing" if dur_trend < -0.3 else "stable"),
                "reasoning": reasoning,
                "recommendations": [
                    "Retry with same parameters" if should_retry else "Try different approach",
                    "Add exponential backoff" if not should_retry and recent_rate < 0.3 else None,
                ],
                "training_time_seconds": round(time.time() - t0, 2),
            }
        else:
            return {
                "status": "success",
                "decision": "retry",
                "confidence": 0.5,
                "reasoning": ["Insufficient history — defaulting to retry"],
                "recommendations": ["Retry and record the outcome"],
                "training_time_seconds": round(time.time() - t0, 2),
            }

    def _decide_optimize(self, data, options, history, t0):
        """What's the best option? Learn from history which option performs best."""
        if history and len(history) >= 3:
            # Extract option performance from history
            option_scores: Dict[str, List[float]] = {}
            for h in history:
                opt = h.get("option", h.get("choice", h.get("value", "unknown")))
                score = float(h.get("score", h.get("success", 1.0) if h.get("success", True) else 0.0))
                option_scores.setdefault(str(opt), []).append(score)

            if option_scores:
                avg_scores = {k: float(np.mean(v)) for k, v in option_scores.items()}
                best_option = max(avg_scores, key=avg_scores.get)
                best_score = avg_scores[best_option]

                # If options provided, rank them
                if options:
                    ranked = sorted(
                        [(opt, avg_scores.get(opt, 0.5)) for opt in options],
                        key=lambda x: x[1], reverse=True,
                    )
                else:
                    ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

                return {
                    "status": "success",
                    "decision": best_option,
                    "confidence": round(best_score, 3),
                    "ranked_options": [{"option": opt, "score": round(score, 3)} for opt, score in ranked],
                    "option_performance": {k: round(v, 3) for k, v in avg_scores.items()},
                    "reasoning": [
                        "Best option '%s' with average score %.3f" % (best_option, best_score),
                        "Based on %d historical executions" % len(history),
                    ],
                    "training_time_seconds": round(time.time() - t0, 2),
                }

        # No history — use pattern engine on data if available
        if data and len(data) >= 5:
            values = [float(d) if isinstance(d, (int, float)) else float(d.get("value", 0)) for d in data]
            pattern = self.pattern_engine.analyze(values, predict_steps=3, epochs=30)
            predictions = pattern.get("predictions", [])
            return {
                "status": "success",
                "decision": "continue_current",
                "confidence": pattern.get("confidence", 0.3),
                "pattern": pattern.get("pattern_type", "unknown"),
                "predicted_next_values": predictions,
                "reasoning": ["No option history — analyzed data pattern: " + pattern.get("pattern_type", "unknown")],
                "training_time_seconds": round(time.time() - t0, 2),
            }

        return {
            "status": "success",
            "decision": options[0] if options else "unknown",
            "confidence": 0.3,
            "reasoning": ["Insufficient data — defaulting to first option"],
            "training_time_seconds": round(time.time() - t0, 2),
        }

    def _decide_predict(self, data, options, history, t0):
        """Will this succeed or fail? Predict outcome from patterns."""
        if history and len(history) >= 3:
            outcomes = [1.0 if h.get("success", True) else 0.0 for h in history]
            success_rate = float(np.mean(outcomes))

            if len(outcomes) >= 5:
                pattern = self.pattern_engine.analyze(outcomes, predict_steps=3, epochs=30)
                predicted = pattern.get("predictions", [success_rate])
                will_succeed = predicted[0] > 0.5 if predicted else success_rate > 0.5
                confidence = abs(predicted[0] - 0.5) * 2 if predicted else 0.5
            else:
                will_succeed = success_rate > 0.5
                confidence = abs(success_rate - 0.5) * 2
                predicted = [success_rate]

            return {
                "status": "success",
                "decision": "will_succeed" if will_succeed else "will_fail",
                "confidence": round(confidence, 3),
                "success_probability": round(float(predicted[0] if predicted else success_rate), 3),
                "historical_success_rate": round(success_rate, 3),
                "reasoning": [
                    "Historical success rate: %.1f%%" % (success_rate * 100),
                    "Predicted next outcome: %.1f%%" % ((predicted[0] if predicted else success_rate) * 100),
                ],
                "training_time_seconds": round(time.time() - t0, 2),
            }

        return {
            "status": "success",
            "decision": "unknown",
            "confidence": 0.1,
            "reasoning": ["Insufficient history for prediction"],
            "training_time_seconds": round(time.time() - t0, 2),
        }

    def _decide_pattern(self, data, t0):
        """What's the pattern in this data?"""
        if not data or len(data) < 5:
            return {"status": "error", "error": "Need 5+ data points"}

        values = [float(d) if isinstance(d, (int, float)) else float(d.get("value", d.get("duration_ms", 0))) for d in data]
        result = self.pattern_engine.analyze(values, predict_steps=5, epochs=50)

        return {
            "status": "success",
            "pattern_type": result.get("pattern_type", "unknown"),
            "confidence": result.get("confidence", 0),
            "correlation": result.get("training_correlation", 0),
            "predictions": result.get("predictions", []),
            "architecture": result.get("architecture", "unknown"),
            "pattern_scores": result.get("pattern_scores", {}),
            "reasoning": [
                "Pattern: %s (confidence: %.2f)" % (result.get("pattern_type", "?"), result.get("confidence", 0)),
                "Correlation: %.4f" % result.get("training_correlation", 0),
            ],
            "training_time_seconds": round(time.time() - t0, 2),
        }

    def _decide_fix(self, data, history, t0):
        """How do I fix this error? Match error patterns."""
        if not history:
            return {"status": "error", "error": "Need error history"}

        errors = [h for h in history if not h.get("success", True)]
        successes = [h for h in history if h.get("success", True)]

        if not errors:
            return {
                "status": "success",
                "decision": "no_errors",
                "confidence": 0.9,
                "reasoning": ["No errors found in history"],
                "recommendations": ["Continue current approach"],
                "training_time_seconds": round(time.time() - t0, 2),
            }

        # Analyze error patterns
        error_types = {}
        for e in errors:
            et = e.get("error_type", "unknown")
            error_types[et] = error_types.get(et, 0) + 1

        most_common_error = max(error_types, key=error_types.get) if error_types else "unknown"

        # Compare error vs success conditions
        recommendations = []
        if most_common_error == "timeout":
            recommendations.append("Increase timeout duration")
            recommendations.append("Check for slow external dependencies")
            recommendations.append("Add retry with exponential backoff")
        elif most_common_error == "oom" or most_common_error == "memory":
            recommendations.append("Reduce batch size or data volume")
            recommendations.append("Add memory monitoring")
            recommendations.append("Consider streaming processing")
        elif most_common_error == "rate_limit":
            recommendations.append("Add rate limiting to your requests")
            recommendations.append("Implement exponential backoff")
            recommendations.append("Cache responses when possible")
        elif most_common_error == "auth" or most_common_error == "authentication":
            recommendations.append("Check API key validity")
            recommendations.append("Refresh authentication tokens")
            recommendations.append("Verify permissions")
        else:
            recommendations.append("Review error logs for: " + most_common_error)
            recommendations.append("Add more detailed error handling")
            recommendations.append("Consider adding retry logic")

        # Success rate analysis
        total = len(history)
        error_rate = len(errors) / total if total > 0 else 0

        return {
            "status": "success",
            "decision": "fix_%s" % most_common_error,
            "confidence": round(1.0 - error_rate, 3),
            "error_count": len(errors),
            "error_rate": round(error_rate, 3),
            "most_common_error": most_common_error,
            "error_breakdown": error_types,
            "success_count": len(successes),
            "recommendations": recommendations,
            "reasoning": [
                "%d errors out of %d executions (%.1f%% error rate)" % (len(errors), total, error_rate * 100),
                "Most common error: %s (%d occurrences)" % (most_common_error, error_types.get(most_common_error, 0)),
            ],
            "training_time_seconds": round(time.time() - t0, 2),
        }

    def _decide_analyze(self, data, history, t0):
        """General data analysis — combines pattern + statistics."""
        if not data and not history:
            return {"status": "error", "error": "Need data or history"}

        values = []
        if data:
            values = [float(d) if isinstance(d, (int, float)) else float(d.get("value", d.get("duration_ms", 0))) for d in data]
        elif history:
            values = [float(h.get("value", h.get("duration_ms", 0))) for h in history]

        if len(values) < 3:
            return {"status": "error", "error": "Need 3+ values"}

        stats = {
            "count": len(values),
            "mean": round(float(np.mean(values)), 2),
            "std": round(float(np.std(values)), 2),
            "min": round(float(np.min(values)), 2),
            "max": round(float(np.max(values)), 2),
            "median": round(float(np.median(values)), 2),
        }

        if len(values) >= 5:
            pattern = self.pattern_engine.analyze(values, predict_steps=3, epochs=30)
            stats["pattern"] = pattern.get("pattern_type", "unknown")
            stats["predictions"] = pattern.get("predictions", [])
            stats["correlation"] = pattern.get("training_correlation", 0)

        return {
            "status": "success",
            "statistics": stats,
            "training_time_seconds": round(time.time() - t0, 2),
        }

    def _decide_auto(self, data, options, history, t0):
        """Auto-detect the best decision type from available data."""
        if history and len(history) >= 3:
            has_errors = any(not h.get("success", True) for h in history)
            if has_errors:
                return self._decide_fix(data, history, t0)
            else:
                return self._decide_predict(data, options, history, t0)
        elif data and len(data) >= 5:
            return self._decide_pattern(data, t0)
        elif options and len(options) > 1:
            return self._decide_optimize(data, options, history, t0)
        else:
            return {
                "status": "error",
                "error": "Cannot auto-detect context. Provide more data or specify context explicitly.",
                "hint": "Use context='retry', 'optimize', 'predict', 'pattern', 'fix', or 'analyze'",
                "training_time_seconds": round(time.time() - t0, 2),
            }
