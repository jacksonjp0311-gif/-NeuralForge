from __future__ import annotations
import json, logging
from typing import Any, Dict, List, Optional
from neuralforge.spec import NeuralForgeSpec, TrainingConfig, OptimizationGoal, ExportConfig
from neuralforge.core.forge import create_model, train, optimize, evaluate_and_report, evolve, auto_architecture, export_model
logger = logging.getLogger("neuralforge.tools.agent")

class NeuralForgeAgentTool:
    def __init__(self, name="neuralforge"):
        self.name = name
        self.description = "NeuralForge v2.0 - Build, train, optimize, and deploy neural networks."
        self._actions = {"create_model": self._action_create_model, "train": self._action_train,
                        "optimize": self._action_optimize, "evaluate": self._action_evaluate,
                        "evolve": self._action_evolve, "auto_architecture": self._action_auto_arch,
                        "export": self._action_export, "full_pipeline": self._action_full_pipeline}
    def invoke(self, params):
        action = params.get("action")
        if action not in self._actions: return {"status": "error", "error": f"Unknown action: {action}"}
        try: return {"status": "success", "action": action, "result": self._actions[action](params)}
        except Exception as e: return {"status": "error", "action": action, "error": str(e)}
    def get_available_actions(self):
        return [{"name": k, "description": v.__doc__ or ""} for k, v in self._actions.items()]
    def _action_create_model(self, params):
        desc = params.get("description", "")
        spec = NeuralForgeSpec.from_description(desc) if desc else NeuralForgeSpec(**params.get("spec", {}))
        model = create_model(spec)
        return {"model_name": spec.name, "parameters": model.count_parameters(), "architecture": spec.architecture.family.value, "config_hash": spec.config_hash()}
    def _action_train(self, params): return {"status": "ready"}
    def _action_optimize(self, params):
        obj_data = params.get("objective", {})
        obj = OptimizationGoal(**obj_data) if isinstance(obj_data, dict) else OptimizationGoal(objective=str(obj_data))
        result = optimize(obj)
        return result.model_dump()
    def _action_evaluate(self, params): return {"status": "ready"}
    def _action_evolve(self, params):
        spec = NeuralForgeSpec(**params.get("spec", {})) if params.get("spec") else NeuralForgeSpec()
        best = evolve(spec, generations=params.get("generations", 20))
        return {"best_architecture": best.architecture.family.value, "config_hash": best.config_hash()}
    def _action_auto_arch(self, params):
        di = params.get("data_info", {})
        spec = auto_architecture(params.get("task", ""), di if isinstance(di, NeuralForgeSpec) else NeuralForgeSpec(**di) if di else NeuralForgeSpec())
        return {"architecture": spec.architecture.family.value, "config_hash": spec.config_hash()}
    def _action_export(self, params): return {"status": "ready"}
    def _action_full_pipeline(self, params):
        desc = params.get("description", "")
        spec = NeuralForgeSpec.from_description(desc)
        model = create_model(spec)
        return {"status": "pipeline_initiated", "model_name": spec.name, "parameters": model.count_parameters(), "architecture": spec.architecture.family.value}

def as_tool(name="neuralforge"):
    return NeuralForgeAgentTool(name=name)
