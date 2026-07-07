
"""Jarvis-style local service layer for the Tesseract command mind.

v1.1 adds the Integration Skill Bus: governed local repo/file/memory skills
behind explicit task packets.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from neuralforge.tesseract.command import TesseractCommandMind
from neuralforge.tesseract.contract import (
    API_CONTRACT_VERSION,
    DEFAULT_CONTRACT_PATH,
    JARVIS_VERSION,
    contract_manifest,
    load_contract_manifest,
    write_contract_manifest,
)
from neuralforge.tesseract.daemon import DEFAULT_CHECKPOINT, DEFAULT_REPLAY
from neuralforge.tesseract.integration import TesseractIntegrationBus

DEFAULT_MEMORY = Path("artifacts") / "tpn" / "command_memory.jsonl"
DEFAULT_LEDGER = Path("artifacts") / "tpn" / "action_ledger_v0_9.jsonl"


def _now() -> float:
    return time.time()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"raw": line, "decode_error": True})
    return rows


def _append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


@dataclass(frozen=True)
class JarvisServiceConfig:
    checkpoint: str = str(DEFAULT_CHECKPOINT)
    replay_path: str = str(DEFAULT_REPLAY)
    memory_path: str = str(DEFAULT_MEMORY)
    ledger_path: str = str(DEFAULT_LEDGER)
    contract_path: str = str(DEFAULT_CONTRACT_PATH)
    map_location: str = "cpu"
    repo_root: str = "."


class TesseractActionLedger:
    def __init__(self, path: str | Path = DEFAULT_LEDGER) -> None:
        self.path = Path(path)

    def append(self, entry: dict[str, Any]) -> None:
        enriched = {"schema_version": "tpn.action_ledger.v1.1", "created_at_unix": _now(), **entry}
        _append_jsonl(self.path, enriched)

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = _read_jsonl(self.path)
        return rows[-max(1, int(limit)):]

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        q = query.lower().strip()
        rows = _read_jsonl(self.path)
        hits = [row for row in rows if q in json.dumps(row, sort_keys=True).lower()]
        return hits[-max(1, int(limit)):]


class TesseractJarvisRuntime:
    def __init__(self, config: JarvisServiceConfig | None = None) -> None:
        self.config = config or JarvisServiceConfig()
        self.started_at = time.perf_counter()
        self.contract_path = write_contract_manifest(self.config.contract_path)
        self.mind = TesseractCommandMind(
            self.config.checkpoint,
            replay_path=self.config.replay_path,
            memory_path=self.config.memory_path,
            map_location=self.config.map_location,
        )
        self.ledger = TesseractActionLedger(self.config.ledger_path)
        self.integration = TesseractIntegrationBus(
            repo_root=self.config.repo_root,
            memory_path=self.config.memory_path,
            ledger_path=self.config.ledger_path,
        )

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "runtime": "TesseractJarvisRuntime",
            "version": JARVIS_VERSION,
            "api_contract_version": API_CONTRACT_VERSION,
            "uptime_seconds": time.perf_counter() - self.started_at,
            "checkpoint": self.config.checkpoint,
            "replay_path": self.config.replay_path,
            "memory_path": self.config.memory_path,
            "ledger_path": self.config.ledger_path,
            "contract_path": str(self.contract_path),
            "skills": sorted(self.mind.registry.skills.keys()),
            "integration_skills": sorted(self.integration.skills.keys()),
            "claim_boundary": "Local Jarvis substrate over weighted TPN; no external model call.",
        }

    def contract(self) -> dict[str, Any]:
        data = load_contract_manifest(self.contract_path)
        data["ok"] = True
        return data

    def skills(self) -> dict[str, Any]:
        return {
            "ok": True,
            "version": JARVIS_VERSION,
            "api_contract_version": API_CONTRACT_VERSION,
            "skills": [
                {
                    "skill_id": skill.skill_id,
                    "description": skill.description,
                    "risk": skill.risk,
                    "mutates": skill.mutates,
                }
                for skill in self.mind.registry.skills.values()
            ],
            "integration_skills": self.integration.list_skills(),
            "claim_boundary": "Manifest of explicit local skills only.",
        }

    def command(self, command: str, *, execute: bool = True, allow_mutation: bool = False, style: str = "operator") -> dict[str, Any]:
        t0 = time.perf_counter()
        result = self.mind.handle(command, execute=execute, allow_mutation=allow_mutation, style=style)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result["version"] = JARVIS_VERSION
        result["api_contract_version"] = API_CONTRACT_VERSION
        result["jarvis_latency_ms"] = elapsed_ms
        self.ledger.append({
            "kind": "command",
            "command": command,
            "execute": execute,
            "allow_mutation": allow_mutation,
            "text": result.get("text"),
            "packet": result.get("packet"),
            "jarvis_latency_ms": elapsed_ms,
        })
        return result

    def task(self, skill_id: str, *, params: dict[str, Any] | None = None, allow_mutation: bool = False) -> dict[str, Any]:
        packet = self.integration.execute(skill_id, params or {}, allow_mutation=allow_mutation)
        result = {
            "ok": bool(packet.get("allowed")),
            "version": JARVIS_VERSION,
            "api_contract_version": API_CONTRACT_VERSION,
            "task": packet,
            "claim_boundary": "Governed local integration task. No arbitrary shell execution.",
        }
        self.ledger.append({
            "kind": "integration_task",
            "skill_id": skill_id,
            "params": params or {},
            "allow_mutation": allow_mutation,
            "task": packet,
        })
        return result

    def memory_search(self, query: str, *, limit: int = 10) -> dict[str, Any]:
        q = query.lower().strip()
        rows = _read_jsonl(self.config.memory_path)
        hits = [row for row in rows if q in json.dumps(row, sort_keys=True).lower()]
        return {
            "ok": True,
            "version": JARVIS_VERSION,
            "api_contract_version": API_CONTRACT_VERSION,
            "query": query,
            "hits": hits[-max(1, int(limit)):],
            "memory_path": self.config.memory_path,
            "claim_boundary": "Local JSONL memory search only.",
        }

    def ledger_recent(self, limit: int = 10) -> dict[str, Any]:
        return {
            "ok": True,
            "version": JARVIS_VERSION,
            "api_contract_version": API_CONTRACT_VERSION,
            "entries": self.ledger.recent(limit=limit),
            "ledger_path": self.config.ledger_path,
        }

    def ledger_search(self, query: str, *, limit: int = 10) -> dict[str, Any]:
        return {
            "ok": True,
            "version": JARVIS_VERSION,
            "api_contract_version": API_CONTRACT_VERSION,
            "query": query,
            "entries": self.ledger.search(query, limit=limit),
            "ledger_path": self.config.ledger_path,
        }


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_jarvis_handler(runtime: TesseractJarvisRuntime):
    class JarvisHandler(BaseHTTPRequestHandler):
        server_version = "TesseractJarvisRuntime/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            params = parse_qs(parsed.query)
            if path == "/health":
                _json_response(self, 200, runtime.health())
                return
            if path == "/contract":
                _json_response(self, 200, runtime.contract())
                return
            if path == "/skills" or path == "/integration/skills":
                _json_response(self, 200, runtime.skills())
                return
            if path == "/ledger/recent":
                limit = int(params.get("limit", ["10"])[0])
                _json_response(self, 200, runtime.ledger_recent(limit=limit))
                return
            _json_response(self, 404, {"ok": False, "error": f"unknown endpoint: {path}"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                payload = json.loads(raw)
                if path == "/command":
                    _json_response(self, 200, runtime.command(
                        str(payload.get("command", "")),
                        execute=bool(payload.get("execute", True)),
                        allow_mutation=bool(payload.get("allow_mutation", False)),
                        style=str(payload.get("style", "operator")),
                    ))
                    return
                if path == "/task":
                    _json_response(self, 200, runtime.task(
                        str(payload.get("skill_id", "")),
                        params=dict(payload.get("params", {})),
                        allow_mutation=bool(payload.get("allow_mutation", False)),
                    ))
                    return
                if path == "/memory/search":
                    _json_response(self, 200, runtime.memory_search(str(payload.get("query", "")), limit=int(payload.get("limit", 10))))
                    return
                if path == "/ledger/search":
                    _json_response(self, 200, runtime.ledger_search(str(payload.get("query", "")), limit=int(payload.get("limit", 10))))
                    return
                _json_response(self, 404, {"ok": False, "error": f"unknown endpoint: {path}"})
            except Exception as exc:
                _json_response(self, 400, {"ok": False, "error": str(exc)})

    return JarvisHandler


def run_jarvis_server(
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    replay_path: str | Path = DEFAULT_REPLAY,
    memory_path: str | Path = DEFAULT_MEMORY,
    ledger_path: str | Path = DEFAULT_LEDGER,
    contract_path: str | Path = DEFAULT_CONTRACT_PATH,
    repo_root: str | Path = ".",
    host: str = "127.0.0.1",
    port: int = 8767,
    map_location: str = "cpu",
) -> None:
    runtime = TesseractJarvisRuntime(JarvisServiceConfig(
        checkpoint=str(checkpoint),
        replay_path=str(replay_path),
        memory_path=str(memory_path),
        ledger_path=str(ledger_path),
        contract_path=str(contract_path),
        map_location=map_location,
        repo_root=str(repo_root),
    ))
    server = ThreadingHTTPServer((host, int(port)), make_jarvis_handler(runtime))
    print(json.dumps({
        "ok": True,
        "runtime": "TesseractJarvisRuntime",
        "version": JARVIS_VERSION,
        "api_contract_version": API_CONTRACT_VERSION,
        "url": f"http://{host}:{port}",
        "endpoints": ["/health", "/contract", "/skills", "/integration/skills", "/command", "/task", "/memory/search", "/ledger/recent", "/ledger/search"],
        "checkpoint": str(checkpoint),
        "contract_path": str(contract_path),
        "claim_boundary": "Local Jarvis service. Stop with Ctrl+C.",
    }, indent=2, sort_keys=True))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTPN v1.1 Jarvis service stopped.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--replay-path", default=str(DEFAULT_REPLAY))
    parser.add_argument("--memory-path", default=str(DEFAULT_MEMORY))
    parser.add_argument("--ledger-path", default=str(DEFAULT_LEDGER))
    parser.add_argument("--contract-path", default=str(DEFAULT_CONTRACT_PATH))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--map-location", default="cpu")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--command", default="status")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-mutation", action="store_true")
    parser.add_argument("--task", default="")
    parser.add_argument("--params-json", default="{}")
    args = parser.parse_args()

    if args.serve:
        run_jarvis_server(
            checkpoint=args.checkpoint,
            replay_path=args.replay_path,
            memory_path=args.memory_path,
            ledger_path=args.ledger_path,
            contract_path=args.contract_path,
            repo_root=args.repo_root,
            host=args.host,
            port=args.port,
            map_location=args.map_location,
        )
        return

    runtime = TesseractJarvisRuntime(JarvisServiceConfig(
        checkpoint=args.checkpoint,
        replay_path=args.replay_path,
        memory_path=args.memory_path,
        ledger_path=args.ledger_path,
        contract_path=args.contract_path,
        map_location=args.map_location,
        repo_root=args.repo_root,
    ))
    if args.task:
        print(json.dumps(runtime.task(args.task, params=json.loads(args.params_json), allow_mutation=args.allow_mutation), indent=2, sort_keys=True))
        return
    print(json.dumps(runtime.command(args.command, execute=args.execute, allow_mutation=args.allow_mutation), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
