"""Stable contract metadata for the Tesseract Jarvis core."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

JARVIS_VERSION = "tpn.v1.15"
API_CONTRACT_VERSION = "jarvis.api.v1"
ACTION_PACKET_VERSION = "tpn.action.v1.0"
INTEGRATION_TASK_VERSION = "tpn.integration_task.v1.1"
PLAN_PACKET_VERSION = "tpn.plan.v1.2"
CYCLE_PACKET_VERSION = "tpn.cycle.v1.3"
BENCHMARK_VERSION = "tpn.benchmark.v1.4"
MEMORY_CORE_VERSION = "tpn.memory.v1.5"
IMPROVEMENT_VERSION = "tpn.improvement.v1.6"
EVIDENCE_LOOP_VERSION = "tpn.evidence.v1.6.1"
GOAL_STATE_VERSION = "tpn.goal.v1.7"
GOAL_CYCLE_VERSION = "tpn.goal_cycle.v1.8"
RUNTIME_HYGIENE_VERSION = "tpn.runtime_hygiene.v1.8.1"
GOAL_QUEUE_VERSION = "tpn.goal_queue.v1.9"
POLICY_VERSION = "tpn.policy.v1.10"
PERFORMANCE_VERSION = "tpn.performance.v1.11"
STAIRWAY_VERSION = "tpn.stairway.v1.12"
CONTROL_BUNDLE_VERSION = "tpn.control_bundle.v1.13"
RECEIPT_COMPRESSION_VERSION = "tpn.receipt_compression.v1.13.1"
CONTRACT_CHECKER_VERSION = "tpn.contract_checker.v1.13.2"
APPROVAL_VERSION = "tpn.approval.v1.14"
SANDBOX_PLAN_VERSION = "tpn.sandbox_plan.v1.15"
RUNTIME_KIND = "TesseractJarvisRuntime"
DEFAULT_CONTRACT_PATH = Path("artifacts") / "tpn" / "tesseract_jarvis_manifest_v1_15.json"

STABLE_ENDPOINTS = [
    {"method": "GET", "path": "/health", "description": "Runtime health and paths."},
    {"method": "GET", "path": "/contract", "description": "Stable contract manifest."},
    {"method": "GET", "path": "/skills", "description": "Explicit local skill manifest."},
    {"method": "GET", "path": "/integration/skills", "description": "Explicit integration skill manifest."},
    {"method": "POST", "path": "/command", "description": "Route an English command through TPN."},
    {"method": "POST", "path": "/task", "description": "Execute a governed local integration task."},
    {"method": "POST", "path": "/plan", "description": "Convert English intent into a bounded local task plan."},
    {"method": "POST", "path": "/run_plan", "description": "Execute a bounded task plan through the integration bus."},
    {"method": "POST", "path": "/cycle", "description": "Run one bounded observe-plan-act-report cycle."},
    {"method": "GET", "path": "/memory/episodes", "description": "Read recent episodic memory records."},
    {"method": "POST", "path": "/memory/episodic/search", "description": "Search local episodic memory."},
    {"method": "POST", "path": "/memory/consolidate", "description": "Consolidate local episodic memory into a summary."},
    {"method": "POST", "path": "/memory/search", "description": "Search local JSONL command memory."},
    {"method": "GET", "path": "/ledger/recent", "description": "Read recent local action ledger entries."},
    {"method": "POST", "path": "/ledger/search", "description": "Search local action ledger entries."},
]

RUNTIME_ARTIFACT_POLICY = {
    "commit": [
        "source code",
        "tests",
        "docs",
        "stable seed checkpoints intentionally promoted",
        "stable contract manifest",
    ],
    "ignore": [
        "runtime command memory JSONL",
        "runtime action ledger JSONL",
        "demo scratch ledgers",
        "local process/cache residue",
    ],
}


@dataclass(frozen=True)
class TesseractJarvisContract:
    runtime: str = RUNTIME_KIND
    version: str = JARVIS_VERSION
    api_contract_version: str = API_CONTRACT_VERSION
    action_packet_version: str = ACTION_PACKET_VERSION
    integration_task_version: str = INTEGRATION_TASK_VERSION
    plan_packet_version: str = PLAN_PACKET_VERSION
    cycle_packet_version: str = CYCLE_PACKET_VERSION
    benchmark_version: str = BENCHMARK_VERSION
    memory_core_version: str = MEMORY_CORE_VERSION
    improvement_version: str = IMPROVEMENT_VERSION
    evidence_loop_version: str = EVIDENCE_LOOP_VERSION
    goal_state_version: str = GOAL_STATE_VERSION
    goal_cycle_version: str = GOAL_CYCLE_VERSION
    runtime_hygiene_version: str = RUNTIME_HYGIENE_VERSION
    goal_queue_version: str = GOAL_QUEUE_VERSION
    policy_version: str = POLICY_VERSION
    performance_version: str = PERFORMANCE_VERSION
    stairway_version: str = STAIRWAY_VERSION
    control_bundle_version: str = CONTROL_BUNDLE_VERSION
    receipt_compression_version: str = RECEIPT_COMPRESSION_VERSION
    contract_checker_version: str = CONTRACT_CHECKER_VERSION
    approval_version: str = APPROVAL_VERSION
    sandbox_plan_version: str = SANDBOX_PLAN_VERSION
    endpoint_count: int = len(STABLE_ENDPOINTS)
    claim_boundary: str = (
        "Local governed Jarvis substrate over weighted TPN. "
        "No arbitrary shell execution, no external model call, no autonomous authority."
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["endpoints"] = STABLE_ENDPOINTS
        data["runtime_artifact_policy"] = RUNTIME_ARTIFACT_POLICY
        return data


def contract_manifest() -> dict[str, Any]:
    return TesseractJarvisContract().to_dict()


def write_contract_manifest(path: str | Path = DEFAULT_CONTRACT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract_manifest(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_contract_manifest(path: str | Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return contract_manifest()
    return json.loads(path.read_text(encoding="utf-8"))
