"""
Tests for NeuralForge AGNT Runtime Bridge (Priority A).
"""
import pytest
import time
import json
import tempfile
from pathlib import Path

from neuralforge.agnt_bridge import (
    normalize_agnt_event,
    record_execution,
    get_health_summary,
    rehydrate_from_log,
    _sanitize_text,
)
from neuralforge.realtime_evo import RealtimeEvolutionEngine


class TestSanitizeText:
    def test_none_returns_empty(self):
        assert _sanitize_text(None) == ""

    def test_short_text_unchanged(self):
        assert _sanitize_text("hello") == "hello"

    def test_long_text_truncated(self):
        long = "x" * 5000
        result = _sanitize_text(long, max_len=100)
        assert len(result) < 5000
        assert "truncated" in result

    def test_numeric_converted(self):
        assert _sanitize_text(42) == "42"


class TestNormalizeAgntEvent:
    def test_minimal_event(self):
        event = normalize_agnt_event({})
        assert event["workflow_id"] == "unknown"
        assert event["success"] is True
        assert event["duration_ms"] == 0.0

    def test_full_event(self):
        raw = {
            "workflow_id": "wf_001",
            "workflow_name": "Test Workflow",
            "tool_name": "web_search",
            "status": "success",
            "success": True,
            "duration_ms": 150.5,
            "step_count": 5,
            "retry_count": 0,
            "error_type": "none",
            "recovery_action": "",
            "recovery_success": False,
            "params": {"query": "AI news"},
            "prompt": "search for AI news",
            "response": "found 5 results",
            "timestamp": 1700000000.0,
        }
        event = normalize_agnt_event(raw)
        assert event["workflow_id"] == "wf_001"
        assert event["tool_name"] == "web_search"
        assert event["success"] is True
        assert event["duration_ms"] == 150.5
        assert event["step_count"] == 5

    def test_failed_event(self):
        raw = {
            "workflow_id": "wf_002",
            "status": "error",
            "error_type": "timeout",
            "duration_ms": 30000,
        }
        event = normalize_agnt_event(raw)
        assert event["success"] is False
        assert event["error_type"] == "timeout"

    def test_status_inference(self):
        """Success should be inferred from status when not explicitly set."""
        event = normalize_agnt_event({"status": "failed"})
        assert event["success"] is False
        assert event["error_type"] == "failed"

    def test_alternative_field_names(self):
        """Should handle camelCase and alternative field names."""
        raw = {
            "workflowId": "wf_003",
            "workflowName": "Alt Workflow",
            "toolName": "execute_javascript",
            "duration": 250.0,
            "steps": 8,
            "retries": 2,
        }
        event = normalize_agnt_event(raw)
        assert event["workflow_id"] == "wf_003"
        assert event["workflow_name"] == "Alt Workflow"
        assert event["tool_name"] == "execute_javascript"
        assert event["duration_ms"] == 250.0
        assert event["step_count"] == 8
        assert event["retry_count"] == 2

    def test_text_sanitization(self):
        """Long text fields should be truncated."""
        raw = {
            "prompt": "x" * 5000,
            "response": "y" * 3000,
        }
        event = normalize_agnt_event(raw)
        assert len(event["prompt"]) < 5000
        assert len(event["response"]) < 3000

    def test_recovery_fields(self):
        raw = {
            "recovery_action": "retry_with_backoff",
            "recovery_success": True,
        }
        event = normalize_agnt_event(raw)
        assert event["recovery_action"] == "retry_with_backoff"
        assert event["recovery_success"] is True

    def test_resolved_as_recovery_success(self):
        """'resolved' should map to recovery_success."""
        raw = {"resolved": True}
        event = normalize_agnt_event(raw)
        assert event["recovery_success"] is True


class TestRecordExecution:
    def test_single_execution(self, tmp_path):
        import os
        os.environ["NEURALFORGE_EVENT_LOG"] = str(tmp_path / "events.jsonl")

        # Reset singleton
        import neuralforge.agnt_bridge as bridge
        bridge._engine = None
        bridge._event_log_path = None

        result = record_execution({
            "workflow_id": "wf_test",
            "status": "success",
            "duration_ms": 100,
        })
        assert result["status"] == "success"
        assert result["ingested_event"]["workflow_id"] == "wf_test"
        assert "health_score" in result

    def test_multiple_executions(self, tmp_path):
        import os
        os.environ["NEURALFORGE_EVENT_LOG"] = str(tmp_path / "events.jsonl")

        import neuralforge.agnt_bridge as bridge
        bridge._engine = None
        bridge._event_log_path = None

        for i in range(10):
            record_execution({
                "workflow_id": f"wf_{i % 3}",
                "status": "success" if i < 8 else "error",
                "duration_ms": 100 + i * 10,
            })

        summary = get_health_summary()
        assert summary["total_executions_processed"] == 10

    def test_persist_to_disk(self, tmp_path):
        import os
        log_path = tmp_path / "events.jsonl"
        os.environ["NEURALFORGE_EVENT_LOG"] = str(log_path)

        import neuralforge.agnt_bridge as bridge
        bridge._engine = None
        bridge._event_log_path = None

        record_execution({
            "workflow_id": "wf_persist",
            "status": "success",
        }, persist=True)

        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["workflow_id"] == "wf_persist"


class TestRehydrate:
    def test_rehydrate_from_log(self, tmp_path):
        log_path = tmp_path / "events.jsonl"
        events = [
            {"workflow_id": f"wf_{i}", "success": True, "duration_ms": 100}
            for i in range(20)
        ]
        with log_path.open("w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        count = rehydrate_from_log(str(log_path))
        assert count == 20

    def test_rehydrate_missing_file(self, tmp_path):
        count = rehydrate_from_log(str(tmp_path / "nonexistent.jsonl"))
        assert count == 0
