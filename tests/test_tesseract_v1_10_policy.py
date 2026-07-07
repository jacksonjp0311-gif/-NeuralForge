from pathlib import Path

from neuralforge.tesseract.goal_cycle import TesseractGoalAwareCycleRunner
from neuralforge.tesseract.goal_queue import TesseractGoalQueueRunner
from neuralforge.tesseract.goal_state import TesseractGoalStateManager
from neuralforge.tesseract.jarvis import JarvisServiceConfig, TesseractJarvisRuntime
from neuralforge.tesseract.policy import POLICY_VERSION, TesseractExecutionPolicyGovernor


def _runtime(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    return TesseractJarvisRuntime(JarvisServiceConfig(
        repo_root=str(repo),
        memory_path=str(tmp_path / "command_memory.jsonl"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        episodic_memory_path=str(tmp_path / "episodes.jsonl"),
        contract_path=str(tmp_path / "contract.json"),
    ))


def _governor(tmp_path):
    manager = TesseractGoalStateManager(tmp_path / "goals.json", tmp_path / "events.jsonl")
    cycle = TesseractGoalAwareCycleRunner(goal_manager=manager, runtime=_runtime(tmp_path))
    queue = TesseractGoalQueueRunner(goal_manager=manager, cycle_runner=cycle)
    governor = TesseractExecutionPolicyGovernor(
        policy_path=tmp_path / "policy.json",
        goal_manager=manager,
        queue_runner=queue,
    )
    return manager, governor


def test_v1_10_policy_allows_valid_queue(tmp_path):
    manager, governor = _governor(tmp_path)
    manager.create_goal("A", "check repo status and recent git log", ["repo.status"], ["stop condition"], risk="medium")
    plan = governor.queue_runner.plan_queue(max_goals=1)
    decision = governor.evaluate_queue_plan(plan)
    assert decision["allowed"] is True
    assert decision["policy_version"] == POLICY_VERSION


def test_v1_10_policy_blocks_high_risk_by_default(tmp_path):
    manager, governor = _governor(tmp_path)
    manager.create_goal("High", "check repo status and recent git log", ["repo.status"], ["stop condition"], risk="high")
    plan = governor.queue_runner.plan_queue(max_goals=1)
    decision = governor.evaluate_queue_plan(plan)
    assert decision["allowed"] is False
    assert decision["rejected_goal_ids"] == ["goal_0001"]


def test_v1_10_policy_blocks_missing_stop_condition(tmp_path):
    manager, governor = _governor(tmp_path)
    manager.create_goal("No stop", "check repo status and recent git log", ["repo.status"], [], risk="medium")
    plan = governor.queue_runner.plan_queue(max_goals=1)
    decision = governor.evaluate_queue_plan(plan)
    assert decision["allowed"] is False
    assert "no stop conditions" in " ".join(decision["reasons"])


def test_v1_10_guarded_run_executes_when_allowed(tmp_path):
    manager, governor = _governor(tmp_path)
    manager.create_goal(
        "A",
        "check repo status and recent git log",
        ["repo.status", "repo.log"],
        ["safety_score below 1.0"],
        risk="medium",
        max_cycles=1,
    )
    result = governor.guarded_run(max_goals=1, max_steps=4)
    assert result["ok"] is True
    assert result["policy_decision"]["allowed"] is True
    assert result["queue_result"]["report"]["run_count"] == 1


def test_v1_10_guarded_run_does_not_execute_when_blocked(tmp_path):
    manager, governor = _governor(tmp_path)
    manager.create_goal("High", "check repo status and recent git log", ["repo.status"], ["stop"], risk="high")
    result = governor.guarded_run(max_goals=1, max_steps=4)
    assert result["ok"] is False
    assert result["queue_result"] is None
    assert result["policy_decision"]["allowed"] is False
