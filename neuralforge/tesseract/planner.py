"""Bounded local task planner for the Tesseract Jarvis runtime.

v1.2 converts English intent into a small, explicit task plan composed only of
whitelisted integration bus skills.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from neuralforge.tesseract.integration import TesseractIntegrationBus


@dataclass(frozen=True)
class TesseractPlanStep:
    step_id: str
    skill_id: str
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    risk: str = "low"


@dataclass
class TesseractTaskPlan:
    command: str
    steps: list[TesseractPlanStep]
    plan_version: str = "tpn.plan.v1.2"
    allowed: bool = True
    reason: str = "Plan contains only whitelisted local integration skills."
    claim_boundary: str = "Bounded local task plan; not autonomous authority."

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "steps": [asdict(step) for step in self.steps],
            "plan_version": self.plan_version,
            "allowed": self.allowed,
            "reason": self.reason,
            "claim_boundary": self.claim_boundary,
        }


class TesseractTaskPlanner:
    """Small deterministic intent planner for local integration skills."""

    def __init__(self, integration: TesseractIntegrationBus | None = None) -> None:
        self.integration = integration or TesseractIntegrationBus()

    def make_plan(self, command: str, *, max_steps: int = 6) -> dict[str, Any]:
        text = command.lower().strip()
        steps: list[TesseractPlanStep] = []

        def add(skill_id: str, params: dict[str, Any] | None = None, reason: str = "", risk: str = "low") -> None:
            if len(steps) >= max_steps:
                return
            steps.append(TesseractPlanStep(
                step_id=f"step_{len(steps) + 1:02d}",
                skill_id=skill_id,
                params=params or {},
                reason=reason,
                risk=risk,
            ))

        if any(word in text for word in ["health", "alive", "ping"]):
            add("system.ping", reason="User asked for runtime health.")

        if any(word in text for word in ["status", "dirty", "clean", "workspace"]):
            add("repo.status", reason="User asked for repository/workspace status.")

        if any(word in text for word in ["log", "history", "commit", "recent commits"]):
            add("repo.log", {"limit": 8}, reason="User asked for recent repository history.")

        if any(word in text for word in ["contract", "api", "endpoint", "manifest"]):
            add("repo.contract", reason="User asked about the runtime contract.")

        if any(word in text for word in ["memory", "remember", "recall"]):
            query = self._extract_query(command) or command
            add("memory.search", {"query": query, "limit": 8}, reason="User asked for local memory search.")

        if any(word in text for word in ["ledger", "receipt", "actions", "recent tasks"]):
            add("ledger.recent", {"limit": 8}, reason="User asked for recent task/action receipts.")

        read_path = self._extract_read_path(command)
        if read_path:
            add("file.read", {"path": read_path, "max_bytes": 6000}, reason="User asked to read a bounded repo file.", risk="medium")

        if not steps:
            add("system.ping", reason="Default safety preflight.")
            add("repo.status", reason="Default repo grounding.")
            add("repo.log", {"limit": 5}, reason="Default recent history grounding.")

        plan = TesseractTaskPlan(command=command, steps=steps)
        return plan.to_dict()

    def execute_plan(self, plan: dict[str, Any], *, allow_mutation: bool = False) -> dict[str, Any]:
        t0 = time.perf_counter()
        results: list[dict[str, Any]] = []
        for raw_step in plan.get("steps", []):
            skill_id = str(raw_step.get("skill_id", ""))
            params = dict(raw_step.get("params", {}))
            results.append(self.integration.execute(skill_id, params, allow_mutation=allow_mutation))
        return {
            "ok": all(bool(item.get("allowed")) for item in results),
            "plan_version": "tpn.plan.v1.2",
            "command": plan.get("command", ""),
            "step_count": len(results),
            "results": results,
            "duration_ms": (time.perf_counter() - t0) * 1000.0,
            "claim_boundary": "Executed only whitelisted local integration skills.",
        }

    def plan_and_optionally_execute(self, command: str, *, execute: bool = False, allow_mutation: bool = False) -> dict[str, Any]:
        plan = self.make_plan(command)
        result = {
            "ok": True,
            "plan": plan,
            "executed": False,
            "execution": None,
            "claim_boundary": "Bounded local task planning only.",
        }
        if execute:
            result["executed"] = True
            result["execution"] = self.execute_plan(plan, allow_mutation=allow_mutation)
        return result

    def _extract_query(self, command: str) -> str:
        m = re.search(r"(?:memory|remember|recall)\s+(?:for|about|this:|this)?\s*(.+)$", command, flags=re.I)
        if not m:
            return ""
        return m.group(1).strip(" :")

    def _extract_read_path(self, command: str) -> str:
        candidates = re.findall(r"([A-Za-z0-9_./\\-]+\.(?:md|txt|json|py|ps1|toml|yaml|yml))", command)
        if candidates:
            return candidates[0].replace("\\", "/")
        if "readme" in command.lower():
            return "README.md"
        return ""
