"""Regression tests for NeuroAdaptive execution-data learning."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AGNT_NEUROADAPTIVE import ContinuousLearningEngine


def _executions():
    rows = []
    tools = ["web_search", "python", "browser", "python"]
    for i in range(12):
        failed = i in {3, 7, 11}
        rows.append({
            "workflow_id": "wf-a" if i < 6 else "wf-b",
            "workflow_name": "Research" if i < 6 else "Build",
            "tool_name": tools[i % len(tools)],
            "status": "error" if failed else "success",
            "success": not failed,
            "duration_ms": 800 + i * 125 + (2000 if failed else 0),
            "step_count": 3 + (i % 4),
            "retry_count": 1 if failed else 0,
            "error_type": "timeout" if failed else "none",
            "recovery_action": "retry_with_backoff" if failed else None,
            "recovery_success": 1.0 if failed and i != 11 else 0.0,
            "params": {
                "timeout_ms": 30000 + i * 1000,
                "retry_count": 1 if failed else 0,
                "batch_size": 8 + i,
                "parallel_count": 1 + (i % 2),
                "priority": 0.7 if failed else 0.4,
            },
            "prompt": "build and inspect",
            "response": "completed" if not failed else "timeout while waiting",
            "tokens": 100 + i * 10,
            "validation_passed": not failed,
        })
    return rows


def test_feature_extractors_are_deterministic_and_shaped(tmp_path):
    engine = ContinuousLearningEngine(model_dir=str(tmp_path))
    executions = _executions()

    chat_x, chat_y = engine._chat_features_and_targets(executions)
    tool_x, tool_y = engine._tool_features_and_targets(executions)
    tool_types, param_x, param_y = engine._param_features_and_targets(executions)
    err_x, err_y, err_success = engine._recovery_features_and_targets(executions)

    assert chat_x.shape == (12, 20)
    assert chat_y.shape == (12, 1)
    assert tool_x.shape == (12, 32)
    assert tool_y.shape == (12,)
    assert tool_types.shape == (12,)
    assert param_x.shape == (12, 8)
    assert param_y.shape == (12, 8)
    assert err_x.shape[1] == 16
    assert err_y.shape == (3,)
    assert err_success.shape == (3, 1)

    chat_x_again, chat_y_again = engine._chat_features_and_targets(executions)
    np.testing.assert_allclose(chat_x, chat_x_again)
    np.testing.assert_allclose(chat_y, chat_y_again)


def test_train_on_executions_uses_execution_features(tmp_path):
    engine = ContinuousLearningEngine(model_dir=str(tmp_path))
    engine.training_epochs = 2

    result = engine.train_on_executions(_executions())

    assert result["executions_trained"] == 12
    assert result["chat_scorer"]["feature_source"] == "executions"
    assert result["tool_selector"]["feature_source"] == "executions"
    assert result["param_optimizer"]["feature_source"] == "executions"
    assert result["error_recovery"]["feature_source"] == "executions"
    assert result["error_recovery"]["failed_executions_analyzed"] == 3
    assert (tmp_path / "chat_scorer.pt").exists()

