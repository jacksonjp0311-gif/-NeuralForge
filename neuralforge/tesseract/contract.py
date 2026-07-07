"""Stable contract metadata for the Tesseract Jarvis core."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

JARVIS_VERSION = "tpn.v1.2"
API_CONTRACT_VERSION = "jarvis.api.v1"
ACTION_PACKET_VERSION = "tpn.action.v1.0"
INTEGRATION_TASK_VERSION = "tpn.integration_task.v1.1"
PLAN_PACKET_VERSION = "tpn.plan.v1.2"
RUNTIME_KIND = "TesseractJarvisRuntime"
DEFAULT_CONTRACT_PATH = Path("artifacts") / "tpn" / "tesseract_jarvis_manifest_v1_2.json"

STABLE_ENDPOINTS = [
    {"method": "GET", "path": "/health", "description": "Runtime health and paths."},
    {"method": "GET", "path": "/contract", "description": "Stable contract manifest."},
    {"method": "GET", "path": "/skills", "description": "Explicit local skill manifest."},
    {"method": "GET", "path": "/integration/skills", "description": "Explicit integration skill manifest."},
    {"method": "POST", "path": "/command", "description": "Route an English command through TPN."},
    {"method": "POST", "path": "/task", "description": "Execute a governed local integration task."},
    {"method": "POST", "path": "/plan", "description": "Convert English intent into a bounded local task plan."},
    {"method": "POST", "path": "/run_plan", "description": "Execute a bounded task plan through the integration bus."},
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
