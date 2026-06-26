"""Tests for realtime execution ingestion and event logging."""
from __future__ import annotations

import json

from neuralforge.analyzer import WorkflowAnalyzer
from neuralforge.realtime_evo import RealtimeEvolutionEngine


def _event(i: int, success: bool = True):
    return {
        "id": f"exec-{i}",
        "workflow_name": "Nightly Build",
        "status": "success" if success else "error",
        "duration": 1000 + i * 100,
        "steps": 4,
        "error": None if success else "timeout",
    }


def test_workflow_analyzer_reports_health_score():
    result = WorkflowAnalyzer().analyze([
        {"duration_ms": 100, "success": True, "step_count": 2},
        {"duration_ms": 110, "success": False, "step_count": 2},
        {"duration_ms": 120, "success": True, "step_count": 2},
    ])

    assert result["status"] == "success"
    assert result["stats"]["success_rate"] == 0.6667
    assert result["stats"]["health_score"] == 66.7


def test_ingest_execution_persists_jsonl_and_updates_state(tmp_path):
    log_path = tmp_path / "executions.jsonl"
    engine = RealtimeEvolutionEngine(window_size=5, event_log_path=str(log_path))

    result = engine.ingest_execution(_event(1, success=False))

    assert result["status"] == "success"
    assert result["total_processed"] == 1
    assert result["ingested_event"]["success"] is False
    assert log_path.exists()

    stored = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(stored) == 1
    assert stored[0]["workflow_id"] == "Nightly Build"
    assert stored[0]["duration_ms"] == 1100.0
    assert stored[0]["error_type"] == "timeout"


def test_load_existing_event_log_rehydrates_engine(tmp_path):
    log_path = tmp_path / "executions.jsonl"
    seed = RealtimeEvolutionEngine(event_log_path=str(log_path))
    for i in range(3):
        seed.ingest_execution(_event(i, success=i != 1))

    rehydrated = RealtimeEvolutionEngine(event_log_path=str(log_path), load_existing=True)
    summary = rehydrated.get_knowledge_summary()

    assert summary["total_executions_processed"] == 3
    assert summary["workflows_tracked"] == 1
    assert summary["current_health"] == 66.7

