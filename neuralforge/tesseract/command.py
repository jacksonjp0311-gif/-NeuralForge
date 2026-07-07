
"""Governed local command mind for Tesseract.

v0.8 turns the warm TPN runtime into a Jarvis-style command substrate:
English command -> local vector -> TPN route receipt -> governed skill packet.

No arbitrary shell execution is provided. Skills are explicit, local, and gated.
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import time
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from neuralforge.tesseract.daemon import DEFAULT_CHECKPOINT, DEFAULT_REPLAY, TesseractWarmRuntime


MUTATION_WORDS = {
    "write", "save", "delete", "remove", "update", "create", "commit", "push",
    "send", "email", "archive", "label", "run", "execute", "install", "modify",
}
AUTHORITY_WORDS = {"approved", "authorize", "permission", "allowed", "confirm", "local only", "safe"}
EVIDENCE_WORDS = {"because", "with", "using", "from", "attached", "log", "report", "error", "traceback"}
CONTEXT_WORDS = {"remember", "memory", "project", "repo", "checkpoint", "receipt", "ledger", "state"}


@dataclass(frozen=True)
class CommandVector:
    command: str
    vector: list[float]
    intent: float
    evidence: float
    authority: float
    context: float
    mutation_requested: bool
    delta_phi: float
    coherence: float
    route_prior: list[float]


@dataclass
class SkillResult:
    ok: bool
    text: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class TesseractActionPacket:
    command: str
    skill_id: str
    skill_description: str
    allowed: bool
    mode: str
    risk: str
    vector: list[float]
    selected_vertex: str
    route: str
    route_id: int
    coherence: float
    delta_phi: float
    reason: str
    receipt: dict[str, Any]
    result: dict[str, Any] | None = None
    packet_version: str = "tpn.action.v1.0"
    claim_boundary: str = "Governed local action packet; not autonomous authority."


class CommandVectorizer:
    """Deterministic local command vectorizer.

    This is deliberately small and transparent. It gives the TPN a stable local
    command input without using an external language model.
    """

    def vectorize(self, command: str) -> CommandVector:
        text = " ".join(command.strip().lower().split())
        words = set(text.replace(".", " ").replace(",", " ").split())
        length_score = min(1.0, max(0.15, len(text) / 180.0))
        intent = 0.35 + 0.45 * length_score
        if any(w in words for w in {"do", "make", "build", "plan", "calculate", "remember", "status", "think"}):
            intent += 0.15
        intent = min(intent, 0.98)

        evidence = 0.35 + 0.12 * sum(1 for w in EVIDENCE_WORDS if w in text)
        if any(ch.isdigit() for ch in text):
            evidence += 0.12
        evidence = min(evidence, 0.95)

        mutation = any(w in words for w in MUTATION_WORDS)
        authority = 0.45
        if any(w in text for w in AUTHORITY_WORDS):
            authority += 0.35
        if mutation:
            authority -= 0.18
        authority = min(max(authority, 0.05), 0.95)

        context = 0.35 + 0.12 * sum(1 for w in CONTEXT_WORDS if w in text)
        if len(words) >= 6:
            context += 0.12
        context = min(context, 0.95)

        axes = [intent, evidence, authority, context]
        delta_phi = max(axes) - min(axes)
        coherence = (intent * evidence) / (1.0 + abs(delta_phi))

        missing = [1.0 if score < 0.70 else 0.0 for score in axes]

        # Route prior layout: engage, repair_axis, authority_required, shadow, rehydrate_or_retrieve
        route_prior = [0.0] * 5
        if mutation and authority < 0.70:
            route_prior[3] = 1.0
        elif context < 0.50:
            route_prior[4] = 1.0
        elif any(v > 0.0 for v in missing):
            route_prior[1] = 1.0
        else:
            route_prior[0] = 1.0

        vector = [
            *axes,
            1.0 if mutation else 0.0,
            *missing,
            float(coherence),
            float(delta_phi),
            *route_prior,
        ]
        return CommandVector(
            command=command,
            vector=[float(v) for v in vector],
            intent=float(intent),
            evidence=float(evidence),
            authority=float(authority),
            context=float(context),
            mutation_requested=mutation,
            delta_phi=float(delta_phi),
            coherence=float(coherence),
            route_prior=route_prior,
        )


_ALLOWED_AST_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Load,
}
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval_math(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_AST_NODES:
            raise ValueError(f"unsupported math syntax: {type(node).__name__}")

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("only numeric constants are allowed")
            return float(node.value)
        if isinstance(node, ast.BinOp):
            op = _ALLOWED_BINOPS[type(node.op)]
            return float(op(visit(node.left), visit(node.right)))
        if isinstance(node, ast.UnaryOp):
            op = _ALLOWED_UNARY[type(node.op)]
            return float(op(visit(node.operand)))
        raise ValueError(f"unsupported math node: {type(node).__name__}")

    return visit(tree)


@dataclass(frozen=True)
class TesseractSkill:
    skill_id: str
    description: str
    risk: str
    mutates: bool
    handler: Callable[[str, dict[str, Any]], SkillResult]


class TesseractSkillRegistry:
    def __init__(self, memory_path: str | Path = "artifacts/tpn/command_memory.jsonl") -> None:
        self.memory_path = Path(memory_path)
        self.skills: dict[str, TesseractSkill] = {}
        self.register_defaults()

    def register(self, skill: TesseractSkill) -> None:
        self.skills[skill.skill_id] = skill

    def register_defaults(self) -> None:
        self.register(TesseractSkill("tpn.status", "Return local mind status.", "low", False, self._status))
        self.register(TesseractSkill("tpn.echo", "Reflect the command without mutation.", "low", False, self._echo))
        self.register(TesseractSkill("tpn.plan", "Generate a compact local plan.", "low", False, self._plan))
        self.register(TesseractSkill("tpn.math", "Evaluate a safe arithmetic expression.", "low", False, self._math))
        self.register(TesseractSkill("tpn.memory_note", "Append an approved local memory note.", "medium", True, self._memory_note))

    def choose(self, command: str) -> TesseractSkill:
        text = command.lower()
        if any(w in text for w in ["status", "health", "alive", "runtime"]):
            return self.skills["tpn.status"]
        if any(w in text for w in ["calculate", "math", "sum", "times", "divide", "+", "-", "*", "/"]):
            return self.skills["tpn.math"]
        if any(w in text for w in ["plan", "roadmap", "steps", "strategy"]):
            return self.skills["tpn.plan"]
        if any(w in text for w in ["remember", "note", "save this", "log this"]):
            return self.skills["tpn.memory_note"]
        return self.skills["tpn.echo"]

    def _status(self, command: str, ctx: dict[str, Any]) -> SkillResult:
        return SkillResult(True, "Tesseract command mind is online.", {"ctx": ctx})

    def _echo(self, command: str, ctx: dict[str, Any]) -> SkillResult:
        return SkillResult(True, f"Observed command: {command}", {"ctx": ctx})

    def _plan(self, command: str, ctx: dict[str, Any]) -> SkillResult:
        return SkillResult(True, "Plan: clarify objective, route through TPN, produce receipt, then execute only approved local steps.", {"ctx": ctx})

    def _math(self, command: str, ctx: dict[str, Any]) -> SkillResult:
        expr = command.lower().replace("calculate", "").replace("math", "").replace("what is", "").strip()
        expr = expr.replace("x", "*").replace("times", "*").replace("divided by", "/")
        value = _safe_eval_math(expr)
        return SkillResult(True, f"Math result: {value}", {"expression": expr, "value": value})

    def _memory_note(self, command: str, ctx: dict[str, Any]) -> SkillResult:
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "tpn.command_memory.v0.8",
            "command": command,
            "ctx": ctx,
            "created_by": "TesseractSkillRegistry",
        }
        with self.memory_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
        return SkillResult(True, "Local memory note appended.", {"memory_path": str(self.memory_path)})


class TesseractCommandMind:
    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        *,
        replay_path: str | Path = DEFAULT_REPLAY,
        memory_path: str | Path = "artifacts/tpn/command_memory.jsonl",
        map_location: str = "cpu",
    ) -> None:
        self.runtime = TesseractWarmRuntime(checkpoint, replay_path=replay_path, map_location=map_location)
        self.vectorizer = CommandVectorizer()
        self.registry = TesseractSkillRegistry(memory_path=memory_path)

    def handle(self, command: str, *, execute: bool = True, allow_mutation: bool = False, style: str = "operator") -> dict[str, Any]:
        vector = self.vectorizer.vectorize(command)
        routed = self.runtime.think(vector.vector, style=style)
        receipt = routed["receipts"][0]
        route = str(receipt.get("route", "unknown"))
        route_id = int(routed["route_ids"][0])
        skill = self.registry.choose(command)

        blocked = False
        if skill.mutates:
            reason = "Allowed: explicit local mutation permission was provided."
        else:
            reason = "Allowed: local non-mutating skill."
        if skill.mutates and not allow_mutation:
            blocked = True
            reason = "Blocked: skill mutates local state and allow_mutation was false."
        if route in {"shadow", "authority_required"} and skill.mutates:
            blocked = True
            reason = f"Blocked: TPN route was {route}, so mutation is not permitted."

        result: SkillResult | None = None
        if execute and not blocked:
            result = skill.handler(command, {
                "route": route,
                "selected_vertex": receipt.get("selected_vertex"),
                "coherence": receipt.get("coherence"),
                "delta_phi": receipt.get("delta_phi"),
            })

        packet = TesseractActionPacket(
            command=command,
            skill_id=skill.skill_id,
            skill_description=skill.description,
            allowed=not blocked,
            mode="execute" if execute and not blocked else "plan_only",
            risk=skill.risk,
            vector=vector.vector,
            selected_vertex=str(receipt.get("selected_vertex", "????")),
            route=route,
            route_id=route_id,
            coherence=float(receipt.get("coherence", 0.0)),
            delta_phi=float(receipt.get("delta_phi", 0.0)),
            reason=reason,
            receipt=receipt,
            result=asdict(result) if result else None,
        )

        return {
            "ok": True,
            "version": "tpn.v1.0",
            "text": self._english(packet),
            "packet": asdict(packet),
            "runtime": {
                "latency_ms": routed["latency_ms"],
                "selected_vertices": routed["selected_vertices"],
            },
            "claim_boundary": "Local governed command mind. No arbitrary shell execution or external model call.",
        }

    def _english(self, packet: TesseractActionPacket) -> str:
        status = "allowed" if packet.allowed else "blocked"
        base = (
            f"Command routed to {packet.skill_id} and {status}. "
            f"Vertex={packet.selected_vertex}; route={packet.route}; "
            f"coherence={packet.coherence:.3f}; delta_phi={packet.delta_phi:.3f}. "
            f"{packet.reason}"
        )
        if packet.result and packet.result.get("text"):
            return base + " " + str(packet.result["text"])
        return base


def _json_response(handler: BaseHTTPRequestHandler, code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def make_command_handler(mind: TesseractCommandMind):
    class CommandHandler(BaseHTTPRequestHandler):
        server_version = "TesseractCommandMind/0.8"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                _json_response(self, 200, {
                    "ok": True,
                    "runtime": "TesseractCommandMind",
                    "version": "tpn.v1.0",
                    "skills": sorted(mind.registry.skills.keys()),
                    "claim_boundary": "Local command mind health endpoint.",
                })
                return
            _json_response(self, 404, {"ok": False, "error": f"unknown endpoint: {path}"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length).decode("utf-8") if length else "{}"
                payload = json.loads(raw)
                if path == "/command":
                    result = mind.handle(
                        str(payload.get("command", "")),
                        execute=bool(payload.get("execute", True)),
                        allow_mutation=bool(payload.get("allow_mutation", False)),
                        style=str(payload.get("style", "operator")),
                    )
                    _json_response(self, 200, result)
                    return
                _json_response(self, 404, {"ok": False, "error": f"unknown endpoint: {path}"})
            except Exception as exc:
                _json_response(self, 400, {"ok": False, "error": str(exc)})

    return CommandHandler


def run_command_server(
    *,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    replay_path: str | Path = DEFAULT_REPLAY,
    memory_path: str | Path = "artifacts/tpn/command_memory.jsonl",
    host: str = "127.0.0.1",
    port: int = 8766,
    map_location: str = "cpu",
) -> None:
    mind = TesseractCommandMind(checkpoint, replay_path=replay_path, memory_path=memory_path, map_location=map_location)
    server = ThreadingHTTPServer((host, int(port)), make_command_handler(mind))
    print(json.dumps({
        "ok": True,
        "runtime": "TesseractCommandMind",
        "version": "tpn.v1.0",
        "url": f"http://{host}:{port}",
        "endpoints": ["/health", "/command"],
        "checkpoint": str(checkpoint),
        "claim_boundary": "Local governed command server. Stop with Ctrl+C.",
    }, indent=2, sort_keys=True))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTPN v0.8 command server stopped.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--replay-path", default=str(DEFAULT_REPLAY))
    parser.add_argument("--memory-path", default="artifacts/tpn/command_memory.jsonl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--map-location", default="cpu")
    parser.add_argument("--command", default="status")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-mutation", action="store_true")
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    if args.serve:
        run_command_server(
            checkpoint=args.checkpoint,
            replay_path=args.replay_path,
            memory_path=args.memory_path,
            host=args.host,
            port=args.port,
            map_location=args.map_location,
        )
        return

    mind = TesseractCommandMind(
        args.checkpoint,
        replay_path=args.replay_path,
        memory_path=args.memory_path,
        map_location=args.map_location,
    )
    print(json.dumps(
        mind.handle(args.command, execute=args.execute, allow_mutation=args.allow_mutation),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
