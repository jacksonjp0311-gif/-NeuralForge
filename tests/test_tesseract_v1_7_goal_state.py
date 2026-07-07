from pathlib import Path

from neuralforge.tesseract.goal_state import GOAL_STATE_VERSION, TesseractGoalStateManager


def test_v1_7_goal_create_evidence_complete(tmp_path):
    manager = TesseractGoalStateManager(tmp_path / "goals.json", tmp_path / "events.jsonl")
    goal = manager.create_goal(
        "Benchmark memory lock",
        "Verify benchmark evidence is recorded into memory.",
        success_criteria=["benchmark", "memory"],
        stop_conditions=["safety_score below 1.0"],
        risk="medium",
        max_cycles=2,
    )
    assert goal["goal_version"] == GOAL_STATE_VERSION
    manager.add_evidence(goal["goal_id"], "benchmark evidence present", tags=["benchmark"])
    manager.add_evidence(goal["goal_id"], "memory evidence present", tags=["memory"])
    evaluation = manager.evaluate_goal(goal["goal_id"])
    assert evaluation["new_status"] == "completed"
    summary = manager.summarize(tmp_path / "summary.json")
    assert summary["statuses"]["completed"] == 1


def test_v1_7_goal_stop_condition_blocks(tmp_path):
    manager = TesseractGoalStateManager(tmp_path / "goals.json", tmp_path / "events.jsonl")
    goal = manager.create_goal(
        "Safety invariant",
        "Keep safety score locked.",
        success_criteria=["safety_score 1.0"],
        stop_conditions=["autonomous mutation requested"],
        risk="high",
    )
    manager.add_evidence(goal["goal_id"], "autonomous mutation requested by proposed change", tags=["safety"])
    evaluation = manager.evaluate_goal(goal["goal_id"])
    assert evaluation["new_status"] == "blocked"
    assert "autonomous mutation requested" in evaluation["stop_hits"]


def test_v1_7_goal_cycle_pause(tmp_path):
    manager = TesseractGoalStateManager(tmp_path / "goals.json", tmp_path / "events.jsonl")
    goal = manager.create_goal(
        "Bounded cycle goal",
        "Stop after one cycle if not complete.",
        success_criteria=["external evaluation"],
        stop_conditions=["safety break"],
        max_cycles=1,
    )
    manager.increment_cycle(goal["goal_id"])
    evaluation = manager.evaluate_goal(goal["goal_id"])
    assert evaluation["new_status"] == "paused"


def test_v1_7_goal_demo(tmp_path):
    manager = TesseractGoalStateManager(tmp_path / "goals.json", tmp_path / "events.jsonl")
    demo = manager.demo()
    assert demo["ok"] is True
    assert demo["goal_version"] == GOAL_STATE_VERSION
    assert Path(tmp_path / "goals.json").exists()
    assert Path(tmp_path / "events.jsonl").exists()
