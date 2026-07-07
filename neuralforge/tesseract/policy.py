"""Execution policy governor for the Tesseract Jarvis roadmap.

v1.10 sits above the guarded multi-goal queue. It evaluates whether a planned
goal queue is allowed under explicit budget, risk, and stop rules before any
queue execution occurs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neuralforge.tesseract.goal_queue import (
    GOAL_QUEUE_VERSION,
    TesseractGoalQueueRunner,
)
from neuralforge.tesseract.goal_state import (
    DEFAULT_GOAL_EVENTS_PATH,
    DEFAULT_GOAL_STATE_PATH,
    TesseractGoalStateManager,
)


POLICY_VERSION = "tpn.policy.v1.10"
DEFAULT_POLICY_CONFIG_PATH = Path("artifacts") / "tpn" / "policy_config_v1_10.json"
DEFAULT_POLICY_REPORT_PATH = Path("artifacts") / "tpn" / "policy_report_v1_10_latest.json"
DEFAULT_POLICY_HISTORY_PATH = Path("artifacts") / "tpn" / "policy_history_v1_10.jsonl"

DEFAULT_POLICY = {
    "policy_version": POLICY_VERSION,
    "max_goals_per_run": 2,
    "max_steps_per_goal": 4,
    "allow_high_risk_goals": False,
    "allow_mutation": False,
    "stop_on_block": True,
    "require_success_criteria": True,
    "require_stop_conditions": True,
    "claim_boundary": "Local execution policy only; no autonomous authority.",
}


@dataclass
class TesseractPolicyDecision:
    allowed: bool
    reasons: list[str]
    rejected_goal_ids: list[str] = field(default_factory=list)
    selected_goal_ids: list[str] = field(default_factory=list)
    policy: dict[str, Any] = field(default_factory=dict)
    created_at_unix: float = field(default_factory=time.time)
    policy_version: str = POLICY_VERSION
    claim_boundary: str = "Policy decision only; no execution by itself."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TesseractExecutionPolicyGovernor:
    def __init__(
        self,
        *,
        policy_path: str | Path = DEFAULT_POLICY_CONFIG_PATH,
        goal_manager: TesseractGoalStateManager | None = None,
        queue_runner: TesseractGoalQueueRunner | None = None,
    ) -> None:
        self.policy_path = Path(policy_path)
        self.goal_manager = goal_manager or TesseractGoalStateManager()
        self.queue_runner = queue_runner or TesseractGoalQueueRunner(goal_manager=self.goal_manager)

    def load_policy(self) -> dict[str, Any]:
        if not self.policy_path.exists():
            self.write_policy(DEFAULT_POLICY)
            return dict(DEFAULT_POLICY)
        try:
            policy = json.loads(self.policy_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            policy = dict(DEFAULT_POLICY)
        merged = dict(DEFAULT_POLICY)
        merged.update(policy)
        merged["policy_version"] = POLICY_VERSION
        return merged

    def write_policy(self, policy: dict[str, Any]) -> dict[str, Any]:
        merged = dict(DEFAULT_POLICY)
        merged.update(policy)
        merged["policy_version"] = POLICY_VERSION
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)
        self.policy_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return merged

    def evaluate_queue_plan(self, plan: dict[str, Any], policy: dict[str, Any] | None = None) -> dict[str, Any]:
        policy = policy or self.load_policy()
        reasons: list[str] = []
        rejected: list[str] = []
        selected_goals = list(plan.get("selected_goals", []))
        selected_ids = [str(goal.get("goal_id", "")) for goal in selected_goals]

        if not plan.get("ok"):
            reasons.append("Queue plan is not ok.")
        if int(plan.get("max_goals", 0) or 0) > int(policy.get("max_goals_per_run", 1)):
            reasons.append("Queue plan exceeds max_goals_per_run.")
        if len(selected_goals) > int(policy.get("max_goals_per_run", 1)):
            reasons.append("Selected goal count exceeds policy cap.")

        for goal in selected_goals:
            goal_id = str(goal.get("goal_id", ""))
            risk = str(goal.get("risk", "medium")).lower()
            if risk == "high" and not bool(policy.get("allow_high_risk_goals", False)):
                rejected.append(goal_id)
                reasons.append(f"Goal {goal_id} is high risk and high-risk goals are disabled.")
            if bool(policy.get("require_success_criteria", True)) and not goal.get("success_criteria"):
                rejected.append(goal_id)
                reasons.append(f"Goal {goal_id} has no success criteria.")
            if bool(policy.get("require_stop_conditions", True)) and not goal.get("stop_conditions"):
                rejected.append(goal_id)
                reasons.append(f"Goal {goal_id} has no stop conditions.")
            if int(goal.get("cycle_count", 0) or 0) >= int(goal.get("max_cycles", 1) or 1):
                rejected.append(goal_id)
                reasons.append(f"Goal {goal_id} is already at or above max_cycles.")
            action_text = json.dumps({
                "title": goal.get("title", ""),
                "objective": goal.get("objective", ""),
                "success_criteria": goal.get("success_criteria", []),
            }, sort_keys=True).lower()
            if "allow_mutation" in action_text or "mutation requested" in action_text:
                if not bool(policy.get("allow_mutation", False)):
                    rejected.append(goal_id)
                    reasons.append(f"Goal {goal_id} contains mutation language in action-bearing fields but mutation is disabled.")

        # De-duplicate while preserving order.
        rejected = list(dict.fromkeys([goal_id for goal_id in rejected if goal_id]))
        reasons = list(dict.fromkeys(reasons))
        allowed = bool(plan.get("ok")) and not reasons and not rejected

        if allowed:
            reasons.append("Queue plan satisfies policy.")

        return TesseractPolicyDecision(
            allowed=allowed,
            reasons=reasons,
            rejected_goal_ids=rejected,
            selected_goal_ids=selected_ids,
            policy=policy,
        ).to_dict()

    def guarded_run(
        self,
        *,
        max_goals: int | None = None,
        execute: bool = True,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        policy = self.load_policy()
        max_goals = int(max_goals or policy.get("max_goals_per_run", 1))
        max_steps = int(max_steps or policy.get("max_steps_per_goal", 4))
        plan = self.queue_runner.plan_queue(max_goals=max_goals)
        decision = self.evaluate_queue_plan(plan, policy)
        if not decision.get("allowed"):
            report = {
                "ok": False,
                "policy_version": POLICY_VERSION,
                "policy_decision": decision,
                "plan": plan,
                "queue_result": None,
                "claim_boundary": "Policy blocked queue execution.",
            }
            return report

        queue_result = self.queue_runner.run_queue(
            max_goals=max_goals,
            execute=execute,
            max_steps=max_steps,
            stop_on_block=bool(policy.get("stop_on_block", True)),
        )
        report = {
            "ok": bool(queue_result.get("ok")),
            "policy_version": POLICY_VERSION,
            "policy_decision": decision,
            "plan": plan,
            "queue_result": queue_result,
            "claim_boundary": "Policy-approved bounded queue executed and stopped.",
        }
        return report

    def write_report(
        self,
        report: dict[str, Any],
        *,
        latest_path: str | Path = DEFAULT_POLICY_REPORT_PATH,
        history_path: str | Path = DEFAULT_POLICY_HISTORY_PATH,
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
        self.write_policy(DEFAULT_POLICY)
        self.goal_manager.create_goal(
            "Policy goal: repo status",
            "check repo status and recent git log",
            success_criteria=["repo.status", "repo.log"],
            stop_conditions=["safety_score below 1.0", "autonomous mutation requested"],
            risk="medium",
            max_cycles=1,
        )
        self.goal_manager.create_goal(
            "Policy goal: README read",
            "read README.md",
            success_criteria=["file.read"],
            stop_conditions=["path escapes repo root", "autonomous mutation requested"],
            risk="low",
            max_cycles=1,
        )
        report = self.guarded_run(max_goals=2, execute=True, max_steps=4)
        report["paths"] = self.write_report(report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", default=str(DEFAULT_GOAL_STATE_PATH))
    parser.add_argument("--events-path", default=str(DEFAULT_GOAL_EVENTS_PATH))
    parser.add_argument("--policy-path", default=str(DEFAULT_POLICY_CONFIG_PATH))
    parser.add_argument("--max-goals", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    manager = TesseractGoalStateManager(args.state_path, args.events_path)
    governor = TesseractExecutionPolicyGovernor(policy_path=args.policy_path, goal_manager=manager)
    if args.demo:
        report = governor.demo()
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    if args.plan_only:
        policy = governor.load_policy()
        plan = governor.queue_runner.plan_queue(max_goals=args.max_goals or int(policy.get("max_goals_per_run", 1)))
        print(json.dumps(governor.evaluate_queue_plan(plan, policy), indent=2, sort_keys=True))
        return

    report = governor.guarded_run(
        max_goals=args.max_goals or None,
        execute=True,
        max_steps=args.max_steps or None,
    )
    if args.write:
        report["paths"] = governor.write_report(report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
