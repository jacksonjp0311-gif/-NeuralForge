"""Goal-aware cycle runner for the Tesseract Jarvis roadmap.

v1.8 selects one active bounded goal, runs one bounded cycle, records evidence,
evaluates the goal, writes a report, and stops. It is not continuous autonomy.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neuralforge.tesseract.goal_state import (
    DEFAULT_GOAL_EVENTS_PATH,
    DEFAULT_GOAL_STATE_PATH,
    GOAL_STATE_VERSION,
    TesseractGoalStateManager,
)


GOAL_CYCLE_VERSION = "tpn.goal_cycle.v1.8"
DEFAULT_GOAL_CYCLE_REPORT_PATH = Path("artifacts") / "tpn" / "goal_cycle_report_v1_8_latest.json"
DEFAULT_GOAL_CYCLE_HISTORY_PATH = Path("artifacts") / "tpn" / "goal_cycle_history_v1_8.jsonl"


@dataclass
class TesseractGoalCycleReport:
    goal_id: str
    objective: str
    selected: bool
    executed: bool
    cycle_ok: bool
    evaluation: dict[str, Any]
    observed_skills: list[str] = field(default_factory=list)
    next_recommendation: str = ""
    duration_ms: float = 0.0
    goal_cycle_version: str = GOAL_CYCLE_VERSION
    claim_boundary: str = "One bounded goal-aware cycle; not continuous autonomy."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TesseractGoalAwareCycleRunner:
    def __init__(
        self,
        *,
        goal_manager: TesseractGoalStateManager | None = None,
        runtime: Any | None = None,
    ) -> None:
        self.goal_manager = goal_manager or TesseractGoalStateManager()
        self.runtime = runtime

    def select_goal(self) -> dict[str, Any]:
        active = self.goal_manager.list_goals(status="active").get("goals", [])
        eligible = [
            goal for goal in active
            if int(goal.get("cycle_count", 0)) < int(goal.get("max_cycles", 1))
        ]
        if not eligible:
            return {
                "ok": False,
                "reason": "No active goal below max_cycles.",
                "goal_cycle_version": GOAL_CYCLE_VERSION,
                "claim_boundary": "Goal selection only.",
            }

        risk_rank = {"high": 0, "medium": 1, "low": 2}
        eligible.sort(key=lambda g: (
            risk_rank.get(str(g.get("risk", "medium")).lower(), 1),
            int(g.get("cycle_count", 0)),
            float(g.get("created_at_unix", 0.0)),
        ))
        return {
            "ok": True,
            "goal": eligible[0],
            "goal_cycle_version": GOAL_CYCLE_VERSION,
            "claim_boundary": "Selected one active bounded goal.",
        }

    def run_once(
        self,
        *,
        goal_id: str | None = None,
        execute: bool = True,
        max_steps: int = 6,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        selected = self._resolve_goal(goal_id)
        if not selected.get("ok"):
            return {
                "ok": False,
                "selected": False,
                "reason": selected.get("reason", selected.get("error", "No goal selected.")),
                "goal_cycle_version": GOAL_CYCLE_VERSION,
                "claim_boundary": "No cycle executed.",
            }

        goal = selected["goal"]
        goal_id = goal["goal_id"]

        if goal.get("status") != "active":
            return {
                "ok": False,
                "selected": True,
                "goal_id": goal_id,
                "reason": f"Goal status is {goal.get('status')}; only active goals can run.",
                "goal_cycle_version": GOAL_CYCLE_VERSION,
                "claim_boundary": "No cycle executed.",
            }

        if int(goal.get("cycle_count", 0)) >= int(goal.get("max_cycles", 1)):
            evaluation = self.goal_manager.evaluate_goal(goal_id)
            report = TesseractGoalCycleReport(
                goal_id=goal_id,
                objective=goal.get("objective", ""),
                selected=True,
                executed=False,
                cycle_ok=False,
                evaluation=evaluation,
                next_recommendation="Max cycle count reached before execution.",
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )
            return {"ok": False, "report": report.to_dict(), "goal_cycle_version": GOAL_CYCLE_VERSION}

        runtime = self._runtime()
        cycle_answer = runtime.cycle(goal.get("objective", ""), execute=execute, allow_mutation=False, max_steps=max_steps)
        cycle = cycle_answer.get("cycle", {})
        observed_skills = [
            step.get("skill_id", "")
            for step in cycle.get("plan", {}).get("steps", [])
            if step.get("skill_id")
        ]

        self.goal_manager.increment_cycle(goal_id)
        evidence_summary = (
            f"Goal cycle result ok={bool(cycle_answer.get('ok'))}; "
            f"observed_skills={' '.join(observed_skills)}; "
            f"next={cycle.get('next_recommendation', '')}"
        )
        self.goal_manager.add_evidence(
            goal_id,
            evidence_summary,
            {"cycle_answer": cycle_answer, "observed_skills": observed_skills},
            tags=["goal_cycle", "cycle"] + observed_skills,
        )
        evaluation = self.goal_manager.evaluate_goal(goal_id)

        report = TesseractGoalCycleReport(
            goal_id=goal_id,
            objective=goal.get("objective", ""),
            selected=True,
            executed=bool(cycle.get("executed", execute)),
            cycle_ok=bool(cycle_answer.get("ok")),
            evaluation=evaluation,
            observed_skills=observed_skills,
            next_recommendation=evaluation.get("recommendation", cycle.get("next_recommendation", "")),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        ).to_dict()
        return {
            "ok": bool(cycle_answer.get("ok")),
            "report": report,
            "cycle_answer": cycle_answer,
            "goal_cycle_version": GOAL_CYCLE_VERSION,
            "claim_boundary": "One bounded goal-aware cycle executed, recorded, evaluated, and stopped.",
        }

    def write_report(
        self,
        report: dict[str, Any],
        *,
        latest_path: str | Path = DEFAULT_GOAL_CYCLE_REPORT_PATH,
        history_path: str | Path = DEFAULT_GOAL_CYCLE_HISTORY_PATH,
    ) -> dict[str, str]:
        latest_path = Path(latest_path)
        history_path = Path(history_path)
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, sort_keys=True) + "\n")
        return {"latest": str(latest_path), "history": str(history_path)}

    def demo(self) -> dict[str, Any]:
        goal = self.goal_manager.create_goal(
            "Run one goal-aware repo-status cycle",
            "check repo status and recent git log",
            success_criteria=["repo.status", "repo.log"],
            stop_conditions=["safety_score below 1.0", "autonomous mutation requested"],
            risk="medium",
            max_cycles=1,
        )
        result = self.run_once(goal_id=goal["goal_id"], execute=True, max_steps=4)
        result["paths"] = self.write_report(result)
        result["goal_state_summary"] = self.goal_manager.summarize()
        return result

    def _resolve_goal(self, goal_id: str | None) -> dict[str, Any]:
        if not goal_id:
            return self.select_goal()
        resolved = self.goal_manager.get_goal(goal_id)
        if not resolved.get("ok"):
            return resolved
        return {"ok": True, "goal": resolved["goal"], "goal_cycle_version": GOAL_CYCLE_VERSION}

    def _runtime(self) -> Any:
        if self.runtime is None:
            from neuralforge.tesseract.jarvis import JarvisServiceConfig, TesseractJarvisRuntime
            self.runtime = TesseractJarvisRuntime(JarvisServiceConfig(repo_root="."))
        return self.runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", default=str(DEFAULT_GOAL_STATE_PATH))
    parser.add_argument("--events-path", default=str(DEFAULT_GOAL_EVENTS_PATH))
    parser.add_argument("--goal-id", default="")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--run-once", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    manager = TesseractGoalStateManager(args.state_path, args.events_path)
    runner = TesseractGoalAwareCycleRunner(goal_manager=manager)
    if args.demo:
        print(json.dumps(runner.demo(), indent=2, sort_keys=True))
        return

    if args.run_once:
        result = runner.run_once(goal_id=args.goal_id or None, execute=not args.plan_only, max_steps=args.max_steps)
        if args.write:
            result["paths"] = runner.write_report(result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(json.dumps(runner.select_goal(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
