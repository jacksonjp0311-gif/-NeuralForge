"""Guarded multi-goal queue for the Tesseract Jarvis roadmap.

v1.9 executes a bounded queue of active goals. Each selected goal receives at
most one goal-aware cycle, then the queue stops when it reaches a cap, when a
goal blocks, or when no eligible goals remain.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neuralforge.tesseract.goal_cycle import (
    DEFAULT_GOAL_CYCLE_HISTORY_PATH,
    DEFAULT_GOAL_CYCLE_REPORT_PATH,
    GOAL_CYCLE_VERSION,
    TesseractGoalAwareCycleRunner,
)
from neuralforge.tesseract.goal_state import (
    DEFAULT_GOAL_EVENTS_PATH,
    DEFAULT_GOAL_STATE_PATH,
    TesseractGoalStateManager,
)


GOAL_QUEUE_VERSION = "tpn.goal_queue.v1.9"
DEFAULT_GOAL_QUEUE_REPORT_PATH = Path("artifacts") / "tpn" / "goal_queue_report_v1_9_latest.json"
DEFAULT_GOAL_QUEUE_HISTORY_PATH = Path("artifacts") / "tpn" / "goal_queue_history_v1_9.jsonl"


@dataclass
class TesseractGoalQueueReport:
    selected_goal_ids: list[str]
    run_count: int
    completed_goal_ids: list[str] = field(default_factory=list)
    blocked_goal_ids: list[str] = field(default_factory=list)
    paused_goal_ids: list[str] = field(default_factory=list)
    stopped_reason: str = ""
    results: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0.0
    goal_queue_version: str = GOAL_QUEUE_VERSION
    claim_boundary: str = "Guarded bounded queue; not continuous autonomy."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TesseractGoalQueueRunner:
    def __init__(
        self,
        *,
        goal_manager: TesseractGoalStateManager | None = None,
        cycle_runner: TesseractGoalAwareCycleRunner | None = None,
    ) -> None:
        self.goal_manager = goal_manager or TesseractGoalStateManager()
        self.cycle_runner = cycle_runner or TesseractGoalAwareCycleRunner(goal_manager=self.goal_manager)

    def plan_queue(self, *, max_goals: int = 3) -> dict[str, Any]:
        max_goals = max(1, min(int(max_goals), 10))
        active = self.goal_manager.list_goals(status="active").get("goals", [])
        eligible = [
            goal for goal in active
            if int(goal.get("cycle_count", 0)) < int(goal.get("max_cycles", 1))
        ]
        risk_rank = {"high": 0, "medium": 1, "low": 2}
        eligible.sort(key=lambda g: (
            risk_rank.get(str(g.get("risk", "medium")).lower(), 1),
            int(g.get("cycle_count", 0)),
            float(g.get("created_at_unix", 0.0)),
        ))
        selected = eligible[:max_goals]
        return {
            "ok": True,
            "goal_queue_version": GOAL_QUEUE_VERSION,
            "max_goals": max_goals,
            "eligible_count": len(eligible),
            "selected_goal_ids": [goal["goal_id"] for goal in selected],
            "selected_goals": selected,
            "claim_boundary": "Queue planning only; no execution.",
        }

    def run_queue(
        self,
        *,
        max_goals: int = 3,
        execute: bool = True,
        max_steps: int = 6,
        stop_on_block: bool = True,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        plan = self.plan_queue(max_goals=max_goals)
        selected_goal_ids = list(plan.get("selected_goal_ids", []))
        results: list[dict[str, Any]] = []
        completed: list[str] = []
        blocked: list[str] = []
        paused: list[str] = []
        stopped_reason = "Queue cap reached or selected goals exhausted."

        if not selected_goal_ids:
            stopped_reason = "No active eligible goals."
            report = TesseractGoalQueueReport(
                selected_goal_ids=[],
                run_count=0,
                stopped_reason=stopped_reason,
                duration_ms=(time.perf_counter() - started) * 1000.0,
            ).to_dict()
            return {"ok": True, "plan": plan, "report": report, "goal_queue_version": GOAL_QUEUE_VERSION}

        for goal_id in selected_goal_ids:
            result = self.cycle_runner.run_once(goal_id=goal_id, execute=execute, max_steps=max_steps)
            results.append(result)
            evaluation = result.get("report", {}).get("evaluation", {})
            new_status = evaluation.get("new_status")
            if new_status == "completed":
                completed.append(goal_id)
            if new_status == "blocked":
                blocked.append(goal_id)
                stopped_reason = f"Goal {goal_id} blocked; queue stopped."
                if stop_on_block:
                    break
            if new_status == "paused":
                paused.append(goal_id)
            if not result.get("ok"):
                stopped_reason = f"Goal {goal_id} returned ok=false; queue stopped."
                break

        report = TesseractGoalQueueReport(
            selected_goal_ids=selected_goal_ids,
            run_count=len(results),
            completed_goal_ids=completed,
            blocked_goal_ids=blocked,
            paused_goal_ids=paused,
            stopped_reason=stopped_reason,
            results=results,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        ).to_dict()
        return {
            "ok": True,
            "plan": plan,
            "report": report,
            "goal_queue_version": GOAL_QUEUE_VERSION,
            "claim_boundary": "Guarded bounded multi-goal queue executed and stopped.",
        }

    def write_report(
        self,
        report: dict[str, Any],
        *,
        latest_path: str | Path = DEFAULT_GOAL_QUEUE_REPORT_PATH,
        history_path: str | Path = DEFAULT_GOAL_QUEUE_HISTORY_PATH,
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
        self.goal_manager.create_goal(
            "Queue goal: repo status",
            "check repo status and recent git log",
            success_criteria=["repo.status", "repo.log"],
            stop_conditions=["safety_score below 1.0", "autonomous mutation requested"],
            risk="medium",
            max_cycles=1,
        )
        self.goal_manager.create_goal(
            "Queue goal: README read",
            "read README.md",
            success_criteria=["file.read"],
            stop_conditions=["path escapes repo root", "autonomous mutation requested"],
            risk="low",
            max_cycles=1,
        )
        result = self.run_queue(max_goals=2, execute=True, max_steps=4, stop_on_block=True)
        result["paths"] = self.write_report(result)
        result["goal_state_summary"] = self.goal_manager.summarize()
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", default=str(DEFAULT_GOAL_STATE_PATH))
    parser.add_argument("--events-path", default=str(DEFAULT_GOAL_EVENTS_PATH))
    parser.add_argument("--max-goals", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-stop-on-block", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    manager = TesseractGoalStateManager(args.state_path, args.events_path)
    queue = TesseractGoalQueueRunner(goal_manager=manager)
    if args.demo:
        result = queue.demo()
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.plan_only:
        print(json.dumps(queue.plan_queue(max_goals=args.max_goals), indent=2, sort_keys=True))
        return

    result = queue.run_queue(
        max_goals=args.max_goals,
        execute=True,
        max_steps=args.max_steps,
        stop_on_block=not args.no_stop_on_block,
    )
    if args.write:
        result["paths"] = queue.write_report(result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
