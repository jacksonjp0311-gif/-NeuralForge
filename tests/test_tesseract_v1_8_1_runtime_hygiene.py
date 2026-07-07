from pathlib import Path

from neuralforge.tesseract.goal_state import TesseractGoalStateManager


ROOT = Path(__file__).resolve().parents[1]


def test_v1_8_1_goal_summary_recommendation_is_current(tmp_path):
    manager = TesseractGoalStateManager(tmp_path / "goals.json", tmp_path / "events.jsonl")
    goal = manager.create_goal(
        "Completed goal",
        "check repo status and recent git log",
        success_criteria=["repo.status"],
        stop_conditions=["autonomous mutation requested"],
    )
    manager.add_evidence(goal["goal_id"], "repo.status evidence present", tags=["repo.status"])
    manager.evaluate_goal(goal["goal_id"])
    summary = manager.summarize(tmp_path / "summary.json")
    assert "goal-aware cycle selection is available" in summary["next_recommendation"]
    assert "may add goal-aware cycle selection" not in summary["next_recommendation"]


def test_v1_8_1_power_shell_runners_avoid_python_m_runpy_warning():
    goal_state = (ROOT / "scripts" / "run_tesseract_goal_state.ps1").read_text(encoding="utf-8")
    goal_cycle = (ROOT / "scripts" / "run_tesseract_goal_cycle.ps1").read_text(encoding="utf-8")
    assert "python -m neuralforge.tesseract.goal_state" not in goal_state
    assert "python -m neuralforge.tesseract.goal_cycle" not in goal_cycle
    assert "from neuralforge.tesseract.goal_state import main" in goal_state
    assert "from neuralforge.tesseract.goal_cycle import main" in goal_cycle
