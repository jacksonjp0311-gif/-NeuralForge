from pathlib import Path

from neuralforge.tesseract.goal_cycle import TesseractGoalAwareCycleRunner
from neuralforge.tesseract.goal_queue import GOAL_QUEUE_VERSION, TesseractGoalQueueRunner
from neuralforge.tesseract.goal_state import TesseractGoalStateManager
from neuralforge.tesseract.jarvis import JarvisServiceConfig, TesseractJarvisRuntime


def _runtime(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    return TesseractJarvisRuntime(JarvisServiceConfig(
        repo_root=str(repo),
        memory_path=str(tmp_path / "command_memory.jsonl"),
        ledger_path=str(tmp_path / "ledger.jsonl"),
        episodic_memory_path=str(tmp_path / "episodes.jsonl"),
        contract_path=str(tmp_path / "contract.json"),
    ))


def _queue(tmp_path):
    manager = TesseractGoalStateManager(tmp_path / "goals.json", tmp_path / "events.jsonl")
    cycle = TesseractGoalAwareCycleRunner(goal_manager=manager, runtime=_runtime(tmp_path))
    return manager, TesseractGoalQueueRunner(goal_manager=manager, cycle_runner=cycle)


def test_v1_9_plan_queue_selects_active_goals(tmp_path):
    manager, queue = _queue(tmp_path)
    manager.create_goal("A", "check repo status and recent git log", ["repo.status"], ["stop"], risk="medium")
    manager.create_goal("B", "read README.md", ["file.read"], ["stop"], risk="low")
    plan = queue.plan_queue(max_goals=2)
    assert plan["ok"] is True
    assert plan["goal_queue_version"] == GOAL_QUEUE_VERSION
    assert len(plan["selected_goal_ids"]) == 2


def test_v1_9_run_queue_two_goals(tmp_path):
    manager, queue = _queue(tmp_path)
    manager.create_goal("A", "check repo status and recent git log", ["repo.status", "repo.log"], ["safety_score below 1.0"], risk="medium")
    manager.create_goal("B", "read README.md", ["file.read"], ["path escapes repo root"], risk="low")
    result = queue.run_queue(max_goals=2, execute=True, max_steps=4)
    assert result["ok"] is True
    assert result["report"]["run_count"] == 2
    assert len(result["report"]["completed_goal_ids"]) == 2


def test_v1_9_queue_stops_on_block(tmp_path):
    manager, queue = _queue(tmp_path)
    manager.create_goal("A", "check repo status and recent git log", ["repo.status"], ["repo.status"], risk="medium")
    manager.create_goal("B", "read README.md", ["file.read"], ["never"], risk="low")
    result = queue.run_queue(max_goals=2, execute=True, max_steps=4, stop_on_block=True)
    assert result["report"]["run_count"] == 1
    assert result["report"]["blocked_goal_ids"]
    assert "blocked" in result["report"]["stopped_reason"]


def test_v1_9_queue_cap(tmp_path):
    manager, queue = _queue(tmp_path)
    manager.create_goal("A", "check repo status and recent git log", ["repo.status"], ["stop"])
    manager.create_goal("B", "read README.md", ["file.read"], ["stop"])
    manager.create_goal("C", "recall memory about Tesseract", ["memory.search"], ["stop"])
    result = queue.run_queue(max_goals=1, execute=True, max_steps=4)
    assert result["report"]["run_count"] == 1


def test_v1_9_no_active_goals(tmp_path):
    _, queue = _queue(tmp_path)
    result = queue.run_queue(max_goals=2, execute=True)
    assert result["ok"] is True
    assert result["report"]["run_count"] == 0
    assert result["report"]["stopped_reason"] == "No active eligible goals."
