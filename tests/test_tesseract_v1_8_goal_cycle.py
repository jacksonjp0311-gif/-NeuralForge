from pathlib import Path

from neuralforge.tesseract.goal_cycle import GOAL_CYCLE_VERSION, TesseractGoalAwareCycleRunner
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


def test_v1_8_selects_active_goal(tmp_path):
    manager = TesseractGoalStateManager(tmp_path / "goals.json", tmp_path / "events.jsonl")
    manager.create_goal(
        "Repo status goal",
        "check repo status and recent git log",
        success_criteria=["repo.status"],
        stop_conditions=["safety_score below 1.0"],
        max_cycles=1,
    )
    runner = TesseractGoalAwareCycleRunner(goal_manager=manager, runtime=_runtime(tmp_path))
    selected = runner.select_goal()
    assert selected["ok"] is True
    assert selected["goal_cycle_version"] == GOAL_CYCLE_VERSION


def test_v1_8_run_once_records_goal_evidence(tmp_path):
    manager = TesseractGoalStateManager(tmp_path / "goals.json", tmp_path / "events.jsonl")
    goal = manager.create_goal(
        "Repo status goal",
        "check repo status and recent git log",
        success_criteria=["repo.status", "repo.log"],
        stop_conditions=["safety_score below 1.0"],
        max_cycles=1,
    )
    runner = TesseractGoalAwareCycleRunner(goal_manager=manager, runtime=_runtime(tmp_path))
    result = runner.run_once(goal_id=goal["goal_id"], execute=True, max_steps=4)
    assert result["ok"] is True
    assert result["report"]["goal_id"] == goal["goal_id"]
    assert "repo.status" in result["report"]["observed_skills"]
    updated = manager.get_goal(goal["goal_id"])["goal"]
    assert updated["cycle_count"] == 1
    assert updated["evidence"]


def test_v1_8_no_active_goal_no_execution(tmp_path):
    manager = TesseractGoalStateManager(tmp_path / "goals.json", tmp_path / "events.jsonl")
    runner = TesseractGoalAwareCycleRunner(goal_manager=manager, runtime=_runtime(tmp_path))
    result = runner.run_once(execute=True)
    assert result["ok"] is False
    assert result["selected"] is False


def test_v1_8_demo_writes_report(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    manager = TesseractGoalStateManager(tmp_path / "goals.json", tmp_path / "events.jsonl")
    runner = TesseractGoalAwareCycleRunner(goal_manager=manager, runtime=_runtime(tmp_path))
    result = runner.demo()
    assert result["ok"] is True
    assert "paths" in result
