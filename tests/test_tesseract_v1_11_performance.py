from pathlib import Path

from neuralforge.tesseract.goal_cycle import TesseractGoalAwareCycleRunner
from neuralforge.tesseract.goal_queue import TesseractGoalQueueRunner
from neuralforge.tesseract.goal_state import TesseractGoalStateManager
from neuralforge.tesseract.jarvis import JarvisServiceConfig, TesseractJarvisRuntime
from neuralforge.tesseract.performance import PERFORMANCE_VERSION, TesseractPerformanceTelemetryGovernor
from neuralforge.tesseract.policy import TesseractExecutionPolicyGovernor


def _runtime(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    return TesseractJarvisRuntime(JarvisServiceConfig(
        repo_root=str(repo),
        memory_path=str(tmp_path / "command_memory.jsonl"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        episodic_memory_path=str(tmp_path / "episodes.jsonl"),
        contract_path=str(tmp_path / "contract.json"),
    ))


def _perf(tmp_path):
    manager = TesseractGoalStateManager(tmp_path / "goals.json", tmp_path / "events.jsonl")
    cycle = TesseractGoalAwareCycleRunner(goal_manager=manager, runtime=_runtime(tmp_path))
    queue = TesseractGoalQueueRunner(goal_manager=manager, cycle_runner=cycle)
    policy = TesseractExecutionPolicyGovernor(policy_path=tmp_path / "policy.json", goal_manager=manager, queue_runner=queue)
    return manager, TesseractPerformanceTelemetryGovernor(policy_governor=policy)


def test_v1_11_performance_probe_collects_skill_latencies(tmp_path):
    manager, perf = _perf(tmp_path)
    manager.create_goal("A", "check repo status and recent git log", ["repo.status", "repo.log"], ["safety_score below 1.0"], risk="medium")
    manager.create_goal("B", "read README.md", ["file.read"], ["path escapes repo root"], risk="low")
    report = perf.run_probe(max_goals=2, max_steps=4)
    assert report["performance_version"] == PERFORMANCE_VERSION
    assert report["ok"] is True
    assert report["queue_result_present"] is True
    assert report["policy_allowed"] is True
    assert report["completed_goal_count"] == 2
    assert len(report["skill_latencies"]) >= 3


def test_v1_11_performance_warning_for_slow_skill(tmp_path):
    _, perf = _perf(tmp_path)
    warnings = perf.evaluate_warnings(
        total_duration_ms=10.0,
        queue_duration_ms=10.0,
        goal_cycle_durations_ms=[10.0],
        skill_latencies=[{"goal_id": "g", "skill_id": "slow", "duration_ms": 9999.0, "ok": True}],
        completed_goal_count=1,
        blocked_goal_count=0,
        policy_allowed=True,
        queue_result_present=True,
    )
    assert any("slow" in warning for warning in warnings)


def test_v1_11_performance_blocks_missing_queue_result(tmp_path):
    _, perf = _perf(tmp_path)
    guarded = {
        "ok": False,
        "policy_decision": {"allowed": False},
        "queue_result": None,
    }
    report = perf.collect_from_guarded_report(guarded, total_duration_ms=1.0)
    assert report["performance_ok"] is False
    assert "Queue result missing." in report["warnings"]


def test_v1_11_performance_write_report(tmp_path):
    _, perf = _perf(tmp_path)
    report = perf.collect_from_guarded_report({"ok": True, "policy_decision": {"allowed": True}, "queue_result": {"report": {"duration_ms": 1.0, "completed_goal_ids": ["g1"], "blocked_goal_ids": [], "results": []}}}, total_duration_ms=2.0)
    paths = perf.write_report(report, latest_path=tmp_path / "latest.json", history_path=tmp_path / "history.jsonl")
    assert Path(paths["latest"]).exists()
    assert Path(paths["history"]).exists()
