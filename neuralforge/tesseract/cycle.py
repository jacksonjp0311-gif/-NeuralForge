"""Observation cycle engine for the Tesseract Jarvis runtime.

v1.3 adds one bounded local cycle:

objective -> plan -> optional execute -> observe -> report

The cycle uses only the v1.2 planner and v1.1 integration skills. It does not
perform arbitrary shell execution or autonomous mutation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from neuralforge.tesseract.planner import TesseractTaskPlanner


@dataclass(frozen=True)
class TesseractCycleObservation:
    step_id: str
    skill_id: str
    allowed: bool
    ok: bool
    summary: str
    duration_ms: float = 0.0


@dataclass
class TesseractCycleReport:
    objective: str
    cycle_version: str = "tpn.cycle.v1.3"
    ok: bool = True
    executed: bool = False
    plan: dict[str, Any] = field(default_factory=dict)
    observations: list[dict[str, Any]] = field(default_factory=list)
    next_recommendation: str = ""
    duration_ms: float = 0.0
    claim_boundary: str = "One bounded local cycle; not autonomous authority."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TesseractCycleEngine:
    """Run one bounded observe-plan-act-report cycle."""

    def __init__(self, planner: TesseractTaskPlanner) -> None:
        self.planner = planner

    def run_cycle(
        self,
        objective: str,
        *,
        execute: bool = True,
        allow_mutation: bool = False,
        max_steps: int = 6,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        max_steps = max(1, min(int(max_steps), 12))
        plan = self.planner.make_plan(objective, max_steps=max_steps)

        execution = None
        observations: list[dict[str, Any]] = []

        if execute:
            execution = self.planner.execute_plan(plan, allow_mutation=allow_mutation)
            observations = self._observe_execution(plan, execution)

        recommendation = self._recommend(plan, execution, observations)

        report = TesseractCycleReport(
            objective=objective,
            ok=True if execution is None else bool(execution.get("ok")),
            executed=execute,
            plan=plan,
            observations=observations,
            next_recommendation=recommendation,
            duration_ms=(time.perf_counter() - t0) * 1000.0,
        )
        return report.to_dict()

    def _observe_execution(self, plan: dict[str, Any], execution: dict[str, Any]) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        steps = plan.get("steps", [])
        results = execution.get("results", [])
        for idx, result in enumerate(results):
            step = steps[idx] if idx < len(steps) else {}
            skill_id = str(result.get("skill_id") or step.get("skill_id") or "")
            allowed = bool(result.get("allowed"))
            result_payload = result.get("result", {})
            ok = bool(result_payload.get("ok", allowed)) if isinstance(result_payload, dict) else allowed
            summary = self._summarize_result(skill_id, result_payload, result)
            observations.append(asdict(TesseractCycleObservation(
                step_id=str(step.get("step_id", f"step_{idx + 1:02d}")),
                skill_id=skill_id,
                allowed=allowed,
                ok=ok,
                summary=summary,
                duration_ms=float(result.get("duration_ms", 0.0) or 0.0),
            )))
        return observations

    def _summarize_result(self, skill_id: str, result_payload: Any, packet: dict[str, Any]) -> str:
        if not packet.get("allowed"):
            return str(packet.get("reason", "blocked"))
        if isinstance(result_payload, dict):
            if skill_id.startswith("repo.") and "stdout" in result_payload:
                stdout = str(result_payload.get("stdout", "")).strip()
                if stdout:
                    return f"{skill_id} returned {len(stdout.splitlines())} line(s)."
                return f"{skill_id} returned no output."
            if skill_id == "file.read":
                path = result_payload.get("path", "file")
                bytes_returned = result_payload.get("bytes_returned", 0)
                return f"Read {bytes_returned} byte(s) from {path}."
            if skill_id == "memory.search":
                return f"Memory search found {result_payload.get('count', 0)} matching row(s)."
            if skill_id == "ledger.recent":
                return f"Ledger contains {result_payload.get('count', 0)} known row(s)."
            if skill_id == "system.ping":
                return "Integration bus responded."
        raw = json.dumps(result_payload, sort_keys=True)[:240] if not isinstance(result_payload, str) else result_payload[:240]
        return raw

    def _recommend(self, plan: dict[str, Any], execution: dict[str, Any] | None, observations: list[dict[str, Any]]) -> str:
        if execution is None:
            return "Review the proposed bounded plan, then run it only if the listed skills are acceptable."
        failures = [obs for obs in observations if not obs.get("ok") or not obs.get("allowed")]
        if failures:
            return "Inspect failed observations and generate a close script for the failing bounded skill only."
        if any(obs.get("skill_id") == "repo.status" for obs in observations):
            return "If repo status is clean, proceed to the next governed evolution layer; otherwise close workspace residue first."
        return "Cycle completed. Use observations to choose the next bounded integration or memory step."
