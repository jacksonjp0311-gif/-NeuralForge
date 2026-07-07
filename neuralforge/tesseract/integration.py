
"""Governed integration skill bus for the Tesseract Jarvis runtime.

v1.1 adds whitelisted repo/file/system skills. This is not arbitrary shell
execution. Every integration action is explicit, local, bounded, and returns a
task receipt.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class IntegrationSkill:
    skill_id: str
    description: str
    permission: str
    risk: str
    mutates: bool = False


@dataclass
class IntegrationTaskPacket:
    skill_id: str
    params: dict[str, Any]
    allowed: bool
    reason: str
    result: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    packet_version: str = "tpn.integration_task.v1.1"
    claim_boundary: str = "Governed local integration task; no arbitrary shell execution."


def _safe_text(value: Any, limit: int = 12000) -> str:
    text = str(value)
    if len(text) > limit:
        return text[:limit] + "\n...[truncated]"
    return text


class TesseractIntegrationBus:
    """Explicit local integration bus.

    Skills are hard-coded and whitelisted. The bus can read selected local state
    and execute fixed git status/log commands, but it cannot execute arbitrary
    shell commands supplied by the caller.
    """

    def __init__(
        self,
        repo_root: str | Path = ".",
        *,
        memory_path: str | Path = "artifacts/tpn/command_memory.jsonl",
        ledger_path: str | Path = "artifacts/tpn/action_ledger_v0_9.jsonl",
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.memory_path = Path(memory_path)
        self.ledger_path = Path(ledger_path)
        self.skills: dict[str, IntegrationSkill] = {
            "system.ping": IntegrationSkill("system.ping", "Return integration bus health.", "read", "low"),
            "repo.status": IntegrationSkill("repo.status", "Read git status --short for the repo.", "read", "low"),
            "repo.log": IntegrationSkill("repo.log", "Read recent git log entries.", "read", "low"),
            "repo.contract": IntegrationSkill("repo.contract", "Read the Jarvis contract manifest.", "read", "low"),
            "file.read": IntegrationSkill("file.read", "Read a bounded file inside the repository.", "read", "medium"),
            "memory.search": IntegrationSkill("memory.search", "Search local Jarvis command memory.", "read", "low"),
            "ledger.recent": IntegrationSkill("ledger.recent", "Read recent local action ledger entries.", "read", "low"),
        }

    def list_skills(self) -> list[dict[str, Any]]:
        return [asdict(skill) for skill in self.skills.values()]

    def execute(self, skill_id: str, params: dict[str, Any] | None = None, *, allow_mutation: bool = False) -> dict[str, Any]:
        params = params or {}
        t0 = time.perf_counter()
        if skill_id not in self.skills:
            packet = IntegrationTaskPacket(
                skill_id=skill_id,
                params=params,
                allowed=False,
                reason=f"Unknown integration skill: {skill_id}",
            )
            return asdict(packet)

        skill = self.skills[skill_id]
        if skill.mutates and not allow_mutation:
            packet = IntegrationTaskPacket(
                skill_id=skill_id,
                params=params,
                allowed=False,
                reason=f"Blocked: {skill_id} mutates local state and allow_mutation was false.",
            )
            return asdict(packet)

        try:
            result = self._dispatch(skill_id, params)
            allowed = True
            reason = "Allowed: whitelisted local integration skill."
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
            allowed = False
            reason = f"Integration skill failed: {exc}"

        duration_ms = (time.perf_counter() - t0) * 1000.0
        packet = IntegrationTaskPacket(
            skill_id=skill_id,
            params=params,
            allowed=allowed,
            reason=reason,
            result=result,
            duration_ms=duration_ms,
        )
        return asdict(packet)

    def _dispatch(self, skill_id: str, params: dict[str, Any]) -> dict[str, Any]:
        if skill_id == "system.ping":
            return {
                "ok": True,
                "runtime": "TesseractIntegrationBus",
                "version": "tpn.v1.1",
                "repo_root": str(self.repo_root),
            }
        if skill_id == "repo.status":
            return self._git(["status", "--short"])
        if skill_id == "repo.log":
            limit = int(params.get("limit", 8))
            limit = max(1, min(limit, 50))
            return self._git(["log", "--oneline", f"-{limit}"])
        if skill_id == "repo.contract":
            return self._read_json_file(self.repo_root / "artifacts" / "tpn" / "tesseract_jarvis_manifest_v1_1.json")
        if skill_id == "file.read":
            rel = str(params.get("path", "README.md"))
            max_bytes = int(params.get("max_bytes", 12000))
            return self._read_repo_file(rel, max_bytes=max_bytes)
        if skill_id == "memory.search":
            query = str(params.get("query", ""))
            limit = int(params.get("limit", 10))
            return self._jsonl_search(self.memory_path, query=query, limit=limit)
        if skill_id == "ledger.recent":
            limit = int(params.get("limit", 10))
            return self._jsonl_recent(self.ledger_path, limit=limit)
        raise ValueError(f"unhandled skill: {skill_id}")

    def _git(self, args: list[str]) -> dict[str, Any]:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(self.repo_root),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "cmd": ["git", *args],
            "returncode": proc.returncode,
            "stdout": _safe_text(proc.stdout),
            "stderr": _safe_text(proc.stderr),
        }

    def _resolve_inside_repo(self, rel: str | Path) -> Path:
        target = (self.repo_root / rel).resolve()
        try:
            target.relative_to(self.repo_root)
        except ValueError as exc:
            raise ValueError(f"path escapes repo root: {rel}") from exc
        return target

    def _read_repo_file(self, rel: str, *, max_bytes: int = 12000) -> dict[str, Any]:
        target = self._resolve_inside_repo(rel)
        if not target.exists():
            raise FileNotFoundError(str(rel))
        if not target.is_file():
            raise ValueError(f"not a file: {rel}")
        max_bytes = max(1, min(int(max_bytes), 100000))
        data = target.read_bytes()[:max_bytes]
        return {
            "ok": True,
            "path": str(target.relative_to(self.repo_root)),
            "bytes_returned": len(data),
            "max_bytes": max_bytes,
            "text": data.decode("utf-8", errors="replace"),
        }

    def _read_json_file(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            fallback = self.repo_root / "artifacts" / "tpn" / "tesseract_jarvis_manifest_v1_0.json"
            if fallback.exists():
                path = fallback
        if not path.exists():
            raise FileNotFoundError(str(path))
        return {
            "ok": True,
            "path": str(path),
            "json": json.loads(path.read_text(encoding="utf-8")),
        }

    def _read_jsonl(self, path: str | Path) -> list[dict[str, Any]]:
        path = Path(path)
        if not path.is_absolute():
            path = self.repo_root / path
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

    def _jsonl_search(self, path: str | Path, *, query: str, limit: int = 10) -> dict[str, Any]:
        rows = self._read_jsonl(path)
        q = query.lower().strip()
        hits = [row for row in rows if q in json.dumps(row, sort_keys=True).lower()]
        limit = max(1, min(int(limit), 50))
        return {"ok": True, "query": query, "hits": hits[-limit:], "count": len(hits)}

    def _jsonl_recent(self, path: str | Path, *, limit: int = 10) -> dict[str, Any]:
        rows = self._read_jsonl(path)
        limit = max(1, min(int(limit), 50))
        return {"ok": True, "entries": rows[-limit:], "count": len(rows)}
