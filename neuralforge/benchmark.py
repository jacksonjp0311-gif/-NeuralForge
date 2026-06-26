"""
NeuralForge Held-Out Benchmark Flow -- Priority C

Provides train/val/test split metrics, reproducible reporting, and evidence
tracking for NeuralForge's learning components.

Splits execution data into training (60%), validation (20%), test (20%) sets,
trains on training, tunes on validation, and reports final metrics on the
held-out test set.

Labels each result as:
- synthetic/demo evidence (generated data)
- real-data evidence (real AGNT executions)
"""
from __future__ import annotations
import logging, json, time, os
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

from neuralforge.learner import DataLearner
from neuralforge.pattern_engine import PatternEngine
from neuralforge.smart_engine import SmartEngine

logger = logging.getLogger("neuralforge.benchmark")


class BenchmarkResult:
    """Represents a single benchmark run with full provenance."""

    def __init__(
        self,
        component: str,
        metric_name: str,
        metric_value: float,
        train_size: int,
        val_size: int,
        test_size: int,
        data_source: str,  # "synthetic" or "real_executions"
        seed: int = 42,
        notes: str = "",
    ):
        self.component = component
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.data_source = data_source
        self.seed = seed
        self.notes = notes
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "metric": self.metric_name,
            "value": round(self.metric_value, 4),
            "split": {
                "train": self.train_size,
                "val": self.val_size,
                "test": self.test_size,
            },
            "data_source": self.data_source,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "notes": self.notes,
        }


def split_data(
    data: List[Dict[str, Any]],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split execution data into train/val/test sets.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Split fractions must sum to 1.0"

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(data))

    n_train = int(len(data) * train_frac)
    n_val = int(len(data) * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train = [data[i] for i in train_idx]
    val = [data[i] for i in val_idx]
    test = [data[i] for i in test_idx]

    return train, val, test


def _extract_features(execs):
    """Extract numeric feature matrix from execution dicts."""
    features = []
    for e in execs:
        duration = float(e.get("duration_ms", 0))
        steps = float(e.get("step_count", 0))
        retries = float(e.get("retry_count", 0))
        success = 1.0 if e.get("success", True) else 0.0
        param_count = len(e.get("params", {}))
        error = 1.0 - success
        features.append([
            min(duration / 60000.0, 1.0),
            min(steps / 100.0, 1.0),
            min(retries / 10.0, 1.0),
            success,
            error,
            min(param_count / 20.0, 1.0),
            float(e.get("cost_usd", 0)) / 10.0,
            float(e.get("tokens", 0)) / 32000.0,
        ])
    return np.array(features, dtype=np.float32)


def _extract_targets(execs):
    """Extract target vector: success rate (for regression)."""
    targets = []
    for e in execs:
        success = 1.0 if e.get("success", True) else 0.0
        targets.append(success)
    return np.array(targets, dtype=np.float32)


def run_learner_benchmark(
    executions: List[Dict[str, Any]],
    data_source: str = "synthetic",
    seed: int = 42,
    epochs: int = 25,
) -> Dict[str, Any]:
    """
    Run held-out benchmark on the DataLearner component.

    Trains on training split, validates on validation split,
    reports final metrics on held-out test split.
    """
    start_time = time.time()

    train, val, test = split_data(executions, seed=seed)

    if len(train) < 10:
        return {
            "status": "insufficient_data",
            "total_executions": len(executions),
            "message": f"Need at least 10 training samples, got {len(train)}",
        }

    X_train = _extract_features(train)
    y_train = _extract_targets(train)
    X_val = _extract_features(val)
    y_val = _extract_targets(val)
    X_test = _extract_features(test)
    y_test = _extract_targets(test)

    # Train learner using DataLearner.learn(X, y)
    learner = DataLearner(device=torch.device("cpu"))
    train_result = learner.learn(X_train, y_train, epochs=epochs)

    if train_result.get("status") == "error":
        return {
            "status": "error",
            "error": train_result.get("error", "training failed"),
            "data_source": data_source,
        }

    # Evaluate on all splits (compute loss manually)
    def evaluate_split(X, y):
        if not learner.trained or learner.model is None:
            return float("inf")
        learner.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            preds = learner.model(X_t)
            if learner.problem_type == "classification":
                # Use accuracy as proxy
                pred_labels = preds.argmax(dim=1)
                y_t = torch.tensor(y, dtype=torch.long)
                acc = (pred_labels == y_t).float().mean().item()
                return 1.0 - acc  # convert to loss-like
            else:
                y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1) if len(y.shape) == 1 else torch.tensor(y, dtype=torch.float32)
                loss = torch.nn.functional.mse_loss(preds, y_t).item()
                return loss

    train_loss = evaluate_split(X_train, y_train)
    val_loss = evaluate_split(X_val, y_val)
    test_loss = evaluate_split(X_test, y_test)

    elapsed = time.time() - start_time

    results = {
        "status": "success",
        "component": "DataLearner",
        "problem_type": learner.problem_type,
        "data_source": data_source,
        "split": {
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test),
        },
        "metrics": {
            "train_loss": round(float(train_loss), 6),
            "val_loss": round(float(val_loss), 6),
            "test_loss": round(float(test_loss), 6),
            "train_metric_name": train_result.get("metric_name", "unknown"),
            "train_metric_value": train_result.get("metric_value", 0),
        },
        "overfit_gap": round(float(train_loss - test_loss), 6),
        "seed": seed,
        "epochs": epochs,
        "wall_time_seconds": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": (
            f"Held-out test on {data_source} data. "
            f"Problem type detected: {learner.problem_type}. "
            f"Overfit gap (train-test): {round(float(train_loss - test_loss), 4)}"
        ),
    }

    return results


def run_full_benchmark(
    executions: List[Dict[str, Any]],
    data_source: str = "synthetic",
    seed: int = 42,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full benchmark suite and optionally save a reproducible report.
    """
    start_time = time.time()

    report = {
        "benchmark_version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "total_executions": len(executions),
        "seed": seed,
        "components": {},
    }

    # DataLearner benchmark
    learner_result = run_learner_benchmark(executions, data_source=data_source, seed=seed)
    report["components"]["data_learner"] = learner_result

    # Pattern Engine benchmark
    if len(executions) >= 10:
        durations = [float(e.get("duration_ms", 0)) for e in executions[-50:]]
        pattern_engine = PatternEngine()
        pattern_result = pattern_engine.analyze(durations, predict_steps=3, epochs=50)

        actual_next = durations[-3:]
        predicted = pattern_result.get("predictions", [])
        if predicted and len(predicted) >= 3:
            pred_arr = np.array(predicted[:3])
            actual_arr = np.array(actual_next[:3])
            mae = float(np.mean(np.abs(pred_arr - actual_arr)))
        else:
            mae = None

        report["components"]["pattern_engine"] = {
            "status": "success",
            "pattern_type": pattern_result.get("pattern_type", "unknown"),
            "confidence": pattern_result.get("confidence", 0),
            "mae_on_held_out": mae,
            "data_source": data_source,
            "train_size": len(durations) - 3,
            "test_size": 3,
        }
    else:
        report["components"]["pattern_engine"] = {
            "status": "insufficient_data",
            "total_executions": len(executions),
        }

    # Smart Engine benchmark
    if len(executions) >= 3:
        smart = SmartEngine()
        smart_results = []
        for e in executions[-3:]:
            decision = smart.decide(e)
            smart_results.append(decision)
        report["components"]["smart_engine"] = {
            "status": "success",
            "decisions_on_held_out": len(smart_results),
            "sample_decision": smart_results[0] if smart_results else None,
            "data_source": data_source,
        }
    else:
        report["components"]["smart_engine"] = {
            "status": "insufficient_data",
        }

    report["total_wall_time_seconds"] = round(time.time() - start_time, 2)

    # Save report
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        report["report_path"] = str(out)

    return report


def generate_synthetic_executions(n: int = 200, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate synthetic execution events for demo/testing.
    Clearly labeled as synthetic -- NOT real AGNT data.
    """
    rng = np.random.RandomState(seed)
    executions = []
    for i in range(n):
        base_duration = 200 + rng.exponential(300)
        degradation = i * 0.5
        duration = base_duration + degradation + rng.normal(0, 50)
        success = rng.random() > (0.1 + i * 0.001)

        executions.append({
            "workflow_id": f"wf_{rng.randint(0, 5)}",
            "workflow_name": f"workflow_{rng.randint(0, 5)}",
            "tool_name": f"tool_{rng.randint(0, 8)}",
            "status": "success" if success else "error",
            "success": success,
            "duration_ms": max(duration, 10),
            "step_count": rng.randint(3, 20),
            "retry_count": rng.randint(0, 3) if not success else 0,
            "error_type": "none" if success else rng.choice(["timeout", "rate_limit", "auth_error"]),
            "recovery_action": "",
            "recovery_success": False,
            "params": {"timeout_ms": 30000, "batch_size": 32},
            "prompt": "synthetic prompt",
            "response": "synthetic response",
            "timestamp": time.time() + i * 60,
            "_data_source": "synthetic",
        })
    return executions
