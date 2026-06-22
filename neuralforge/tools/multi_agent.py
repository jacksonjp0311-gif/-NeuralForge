from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from neuralforge.spec import NeuralForgeSpec, TrainingResult, EvaluationReport
from neuralforge.core.forge import create_model, train, evaluate_and_report, evolve, auto_architecture, optimize
from neuralforge.auto.architect import ArchitectAgent
from neuralforge.optimize.meta_optimizer import MetaOptimizer
from neuralforge.memory.insights_store import InsightsStore

logger = logging.getLogger("neuralforge.tools.multi_agent")


class AgentRole(str, Enum):
    ARCHITECT = "architect"
    OPTIMIZER = "optimizer"
    DATA = "data"
    EVALUATOR = "evaluator"
    DEPLOYER = "deployer"


@dataclass
class BlackboardEntry:
    key: str
    value: Any
    author: str
    timestamp: float = field(default_factory=time.time)


class ForgeBlackboard:
    """Shared blackboard for multi-agent forge sessions."""

    def __init__(self):
        self._entries: Dict[str, BlackboardEntry] = {}
        self._history: List[BlackboardEntry] = []

    def write(self, key: str, value: Any, author: str = "system"):
        entry = BlackboardEntry(key=key, value=value, author=author)
        self._entries[key] = entry
        self._history.append(entry)
        logger.info(f"Blackboard[{key}] ← {author}")

    def read(self, key: str) -> Optional[Any]:
        entry = self._entries.get(key)
        return entry.value if entry else None

    def read_all(self) -> Dict[str, Any]:
        return {k: v.value for k, v in self._entries.items()}

    def get_history(self) -> List[Dict]:
        return [
            {"key": e.key, "author": e.author, "timestamp": e.timestamp}
            for e in self._history
        ]


@dataclass
class ForgeSession:
    """A multi-agent forge session."""
    session_id: str
    task_description: str
    blackboard: ForgeBlackboard = field(default_factory=ForgeBlackboard)
    spec: Optional[NeuralForgeSpec] = None
    model: Optional[Any] = None
    training_result: Optional[TrainingResult] = None
    evaluation_report: Optional[EvaluationReport] = None
    iterations: int = 0
    max_iterations: int = 10
    status: str = "initialized"
    created_at: float = field(default_factory=time.time)


class ArchitectAgentRole:
    """Agent role: proposes architectures."""

    def __init__(self):
        self.architect = ArchitectAgent()

    def run(self, session: ForgeSession) -> Dict[str, Any]:
        logger.info(f"Architect: proposing architectures for '{session.task_description}'")
        session.blackboard.write("status", "architecting", "architect")

        # This would use data_profile from the session
        from neuralforge.spec import DataProfile, TaskType, Constraints

        proposals = self.architect.propose(
            task_description=session.task_description,
            data_profile=DataProfile(
                task_type=TaskType.IMAGE_CLASSIFICATION,
                input_shape=(3, 32, 32),
                num_classes=10,
            ),
            constraints=Constraints(),
            num_proposals=3,
        )

        session.blackboard.write("proposals", [p.model_dump() for p in proposals], "architect")
        return {"proposals": proposals, "count": len(proposals)}


class OptimizerAgentRole:
    """Agent role: optimizes hyperparameters and architecture."""

    def __init__(self):
        self.meta = MetaOptimizer()

    def run(self, session: ForgeSession) -> Dict[str, Any]:
        logger.info("Optimizer: running optimization")
        session.blackboard.write("status", "optimizing", "optimizer")

        proposals = session.blackboard.read("proposals") or []
        if not proposals:
            return {"status": "no_proposals", "message": "No architecture proposals to optimize"}

        # In a real system, this would run Optuna/Ray Tune
        return {"status": "optimized", "trials_completed": 0}


class EvaluatorAgentRole:
    """Agent role: evaluates models and provides feedback."""

    def run(self, session: ForgeSession) -> Dict[str, Any]:
        logger.info("Evaluator: evaluating model")
        session.blackboard.write("status", "evaluating", "evaluator")

        report = session.blackboard.read("evaluation_report")
        if report is None:
            return {"status": "no_model", "message": "No model to evaluate"}

        return {"status": "evaluated", "report": report}


class ForgeOrchestrator:
    """Orchestrates multi-agent forge sessions."""

    def __init__(self):
        self.sessions: Dict[str, ForgeSession] = {}
        self.insights = InsightsStore()
        self.architect_role = ArchitectAgentRole()
        self.optimizer_role = OptimizerAgentRole()
        self.evaluator_role = EvaluatorAgentRole()

    def create_session(self, task_description: str, session_id: str = None) -> ForgeSession:
        import uuid
        sid = session_id or f"forge-{uuid.uuid4().hex[:8]}"
        session = ForgeSession(session_id=sid, task_description=task_description)
        self.sessions[sid] = session
        session.blackboard.write("task", task_description, "orchestrator")
        logger.info(f"Created forge session: {sid}")
        return session

    def run_pipeline(self, session: ForgeSession) -> Dict[str, Any]:
        """Run the full multi-agent pipeline."""
        results = {}

        # Phase 1: Architecture
        arch_result = self.architect_role.run(session)
        results["architecture"] = arch_result

        # Phase 2: Optimization
        opt_result = self.optimizer_role.run(session)
        results["optimization"] = opt_result

        # Phase 3: Evaluation
        eval_result = self.evaluator_role.run(session)
        results["evaluation"] = eval_result

        session.status = "completed"
        session.blackboard.write("status", "completed", "orchestrator")

        return results

    def get_session_status(self, session_id: str) -> Optional[Dict]:
        session = self.sessions.get(session_id)
        if session is None:
            return None
        return {
            "session_id": session.session_id,
            "status": session.status,
            "iterations": session.iterations,
            "blackboard": session.blackboard.read_all(),
        }
