"""Goal-state manager for the Tesseract Jarvis roadmap.

v1.7 introduces persistent bounded goals with explicit success criteria,
stop conditions, evidence receipts, and no autonomous execution authority.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


GOAL_STATE_VERSION = "tpn.goal.v1.7"
DEFAULT_GOAL_STATE_PATH = Path("artifacts") / "tpn" / "goals_state_v1_7.json"
DEFAULT_GOAL_EVENTS_PATH = Path("artifacts") / "tpn" / "goal_events_v1_7.jsonl"
DEFAULT_GOAL_SUMMARY_PATH = Path("artifacts") / "tpn" / "goal_summary_v1_7.json"

VALID_STATUSES = {"proposed", "active", "blocked", "completed", "paused"}


@dataclass
class TesseractGoalEvidence:
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created_at_unix: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TesseractGoal:
    goal_id: str
    title: str
    objective: str
    success_criteria: list[str]
    stop_conditions: list[str]
    risk: str = "medium"
    status: str = "active"
    max_cycles: int = 1
    cycle_count: int = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    created_at_unix: float = field(default_factory=time.time)
    updated_at_unix: float = field(default_factory=time.time)
    goal_version: str = GOAL_STATE_VERSION
    claim_boundary: str = "Persistent bounded goal state; not autonomous authority."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TesseractGoalStateManager:
    def __init__(
        self,
        state_path: str | Path = DEFAULT_GOAL_STATE_PATH,
        events_path: str | Path = DEFAULT_GOAL_EVENTS_PATH,
    ) -> None:
        self.state_path = Path(state_path)
        self.events_path = Path(events_path)

    def create_goal(
        self,
        title: str,
        objective: str,
        success_criteria: list[str] | None = None,
        stop_conditions: list[str] | None = None,
        *,
        risk: str = "medium",
        max_cycles: int = 1,
        status: str = "active",
    ) -> dict[str, Any]:
        state = self._load_state()
        goal_id = self._next_goal_id(state)
        goal = TesseractGoal(
            goal_id=goal_id,
            title=title,
            objective=objective,
            success_criteria=success_criteria or [],
            stop_conditions=stop_conditions or [],
            risk=risk,
            max_cycles=max(1, int(max_cycles)),
            status=self._valid_status(status),
        )
        state["goals"][goal_id] = goal.to_dict()
        self._save_state(state)
        self._append_event("goal.created", goal.to_dict())
        return goal.to_dict()

    def list_goals(self, status: str | None = None) -> dict[str, Any]:
        state = self._load_state()
        goals = list(state.get("goals", {}).values())
        if status:
            goals = [goal for goal in goals if goal.get("status") == status]
        return {
            "ok": True,
            "goal_version": GOAL_STATE_VERSION,
            "count": len(goals),
            "goals": goals,
            "claim_boundary": "Local goal-state listing only.",
        }

    def get_goal(self, goal_id: str) -> dict[str, Any]:
        state = self._load_state()
        goal = state.get("goals", {}).get(goal_id)
        if not goal:
            return {"ok": False, "error": f"unknown goal_id: {goal_id}", "goal_version": GOAL_STATE_VERSION}
        return {"ok": True, "goal": goal, "goal_version": GOAL_STATE_VERSION}

    def add_evidence(
        self,
        goal_id: str,
        summary: str,
        payload: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        state = self._load_state()
        goal = state.get("goals", {}).get(goal_id)
        if not goal:
            return {"ok": False, "error": f"unknown goal_id: {goal_id}", "goal_version": GOAL_STATE_VERSION}
        evidence = TesseractGoalEvidence(summary=summary, payload=payload or {}, tags=tags or []).to_dict()
        goal.setdefault("evidence", []).append(evidence)
        goal["updated_at_unix"] = time.time()
        state["goals"][goal_id] = goal
        self._save_state(state)
        self._append_event("goal.evidence", {"goal_id": goal_id, "evidence": evidence})
        return {"ok": True, "goal_id": goal_id, "evidence": evidence, "goal_version": GOAL_STATE_VERSION}

    def evaluate_goal(self, goal_id: str) -> dict[str, Any]:
        state = self._load_state()
        goal = state.get("goals", {}).get(goal_id)
        if not goal:
            return {"ok": False, "error": f"unknown goal_id: {goal_id}", "goal_version": GOAL_STATE_VERSION}

        evidence_text = json.dumps(goal.get("evidence", []), sort_keys=True).lower()
        stop_hits = [cond for cond in goal.get("stop_conditions", []) if str(cond).lower() in evidence_text]
        success_hits = [crit for crit in goal.get("success_criteria", []) if str(crit).lower() in evidence_text]
        success_total = len(goal.get("success_criteria", []))

        old_status = goal.get("status", "active")
        if stop_hits:
            new_status = "blocked"
            recommendation = "Stop condition matched. Do not continue this goal until reviewed."
        elif success_total > 0 and len(success_hits) == success_total:
            new_status = "completed"
            recommendation = "All success criteria matched by evidence."
        elif int(goal.get("cycle_count", 0)) >= int(goal.get("max_cycles", 1)):
            new_status = "paused"
            recommendation = "Max cycle count reached. Human review required before continuing."
        else:
            new_status = "active"
            recommendation = "Goal remains active; collect more evidence or run bounded cycle."

        goal["status"] = new_status
        goal["updated_at_unix"] = time.time()
        state["goals"][goal_id] = goal
        self._save_state(state)

        evaluation = {
            "ok": True,
            "goal_id": goal_id,
            "old_status": old_status,
            "new_status": new_status,
            "success_hits": success_hits,
            "stop_hits": stop_hits,
            "recommendation": recommendation,
            "goal_version": GOAL_STATE_VERSION,
            "claim_boundary": "Goal evaluation only; no autonomous execution.",
        }
        self._append_event("goal.evaluated", evaluation)
        return evaluation

    def increment_cycle(self, goal_id: str) -> dict[str, Any]:
        state = self._load_state()
        goal = state.get("goals", {}).get(goal_id)
        if not goal:
            return {"ok": False, "error": f"unknown goal_id: {goal_id}", "goal_version": GOAL_STATE_VERSION}
        goal["cycle_count"] = int(goal.get("cycle_count", 0)) + 1
        goal["updated_at_unix"] = time.time()
        state["goals"][goal_id] = goal
        self._save_state(state)
        self._append_event("goal.cycle_incremented", {"goal_id": goal_id, "cycle_count": goal["cycle_count"]})
        return {"ok": True, "goal_id": goal_id, "cycle_count": goal["cycle_count"], "goal_version": GOAL_STATE_VERSION}

    def summarize(self, out_path: str | Path = DEFAULT_GOAL_SUMMARY_PATH) -> dict[str, Any]:
        state = self._load_state()
        goals = list(state.get("goals", {}).values())
        statuses: dict[str, int] = {}
        risks: dict[str, int] = {}
        for goal in goals:
            statuses[str(goal.get("status", "unknown"))] = statuses.get(str(goal.get("status", "unknown")), 0) + 1
            risks[str(goal.get("risk", "unknown"))] = risks.get(str(goal.get("risk", "unknown")), 0) + 1
        summary = {
            "ok": True,
            "goal_version": GOAL_STATE_VERSION,
            "goal_count": len(goals),
            "statuses": statuses,
            "risks": risks,
            "active_goal_ids": [goal["goal_id"] for goal in goals if goal.get("status") == "active"],
            "blocked_goal_ids": [goal["goal_id"] for goal in goals if goal.get("status") == "blocked"],
            "next_recommendation": self._recommend(goals),
            "claim_boundary": "Goal summary only; no autonomous execution.",
        }
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._append_event("goal.summary", summary)
        return summary

    def demo(self) -> dict[str, Any]:
        goal = self.create_goal(
            "Establish evidence-grounded roadmap control",
            "Keep the AGI-roadmap evolution tied to tests, benchmarks, memory, and explicit stop conditions.",
            success_criteria=["benchmark", "memory", "proposal"],
            stop_conditions=["safety_score below 1.0", "autonomous mutation requested"],
            risk="medium",
            max_cycles=2,
        )
        self.add_evidence(goal["goal_id"], "Benchmark evidence recorded with safety_score 1.0", {"safety_score": 1.0}, ["benchmark"])
        self.add_evidence(goal["goal_id"], "Memory evidence recorded and proposal engine available", {}, ["memory", "proposal"])
        evaluation = self.evaluate_goal(goal["goal_id"])
        summary = self.summarize()
        return {"ok": True, "goal": goal, "evaluation": evaluation, "summary": summary, "goal_version": GOAL_STATE_VERSION}

    def _valid_status(self, status: str) -> str:
        status = str(status).lower().strip()
        if status not in VALID_STATUSES:
            return "active"
        return status

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"goal_version": GOAL_STATE_VERSION, "goals": {}}
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = {"goal_version": GOAL_STATE_VERSION, "goals": {}}
        state.setdefault("goal_version", GOAL_STATE_VERSION)
        state.setdefault("goals", {})
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        state["goal_version"] = GOAL_STATE_VERSION
        self.state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _append_event(self, kind: str, payload: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "kind": kind,
            "payload": payload,
            "created_at_unix": time.time(),
            "goal_version": GOAL_STATE_VERSION,
            "claim_boundary": "Goal event receipt only.",
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    def _next_goal_id(self, state: dict[str, Any]) -> str:
        existing = state.get("goals", {})
        return f"goal_{len(existing) + 1:04d}"

    def _recommend(self, goals: list[dict[str, Any]]) -> str:
        if not goals:
            return "Create one bounded goal with explicit success criteria and stop conditions."
        blocked = [goal for goal in goals if goal.get("status") == "blocked"]
        active = [goal for goal in goals if goal.get("status") == "active"]
        if blocked:
            return "Review blocked goals before adding autonomy."
        if active:
            return "Run bounded evidence cycles against active goals; do not exceed max_cycles without review."
        return "No active blockers; goal-aware cycle selection is available. Next layer may add guarded multi-goal queueing."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-path", default=str(DEFAULT_GOAL_STATE_PATH))
    parser.add_argument("--events-path", default=str(DEFAULT_GOAL_EVENTS_PATH))
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    manager = TesseractGoalStateManager(args.state_path, args.events_path)
    if args.demo:
        print(json.dumps(manager.demo(), indent=2, sort_keys=True))
        return
    if args.summary:
        print(json.dumps(manager.summarize(), indent=2, sort_keys=True))
        return
    print(json.dumps(manager.list_goals(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
