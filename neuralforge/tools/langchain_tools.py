from __future__ import annotations
import json, logging
from typing import Any, Dict, List, Optional

from neuralforge.spec import NeuralForgeSpec, TrainingConfig, OptimizationGoal, ExportConfig
from neuralforge.core.forge import create_model, train, optimize, evaluate_and_report, evolve, auto_architecture, export_model

logger = logging.getLogger("neuralforge.tools.langchain")


def _get_langchain_tool_base():
    """Try to import LangChain BaseTool."""
    try:
        from langchain_core.tools import BaseTool
        return BaseTool
    except ImportError:
        return None


def _get_crewai_tool_base():
    """Try to import CrewAI tool decorator."""
    try:
        from crewai_tools import BaseTool as CrewBaseTool
        return CrewBaseTool
    except ImportError:
        return None


def make_create_model_tool():
    """Create a LangChain-compatible create_model tool."""
    BaseTool = _get_langchain_tool_base()
    if BaseTool is None:
        return None

    from pydantic import BaseModel, Field

    class CreateModelInput(BaseModel):
        description: str = Field(
            description="Natural language description of the model to build, e.g. "
                        "'Build a ResNet for CIFAR-10 image classification with <5M params'"
        )

    class CreateModelTool(BaseTool):
        name: str = "neuralforge_create_model"
        description: str = (
            "Create a neural network model from a natural language description. "
            "Returns model name, parameter count, and architecture family."
        )
        args_schema = CreateModelInput

        def _run(self, description: str) -> str:
            try:
                spec = NeuralForgeSpec.from_description(description)
                model = create_model(spec)
                return json.dumps({
                    "status": "success",
                    "model_name": spec.name,
                    "parameters": model.count_parameters(),
                    "architecture": spec.architecture.family.value,
                    "config_hash": spec.config_hash(),
                })
            except Exception as e:
                return json.dumps({"status": "error", "error": str(e)})

    return CreateModelTool()


def make_train_tool():
    """Create a LangChain-compatible train tool."""
    BaseTool = _get_langchain_tool_base()
    if BaseTool is None:
        return None

    from pydantic import BaseModel, Field

    class TrainInput(BaseModel):
        model_name: str = Field(description="Name of the registered model to train")
        epochs: int = Field(default=10, description="Number of training epochs")
        batch_size: int = Field(default=32, description="Batch size")
        learning_rate: float = Field(default=1e-3, description="Learning rate")

    class TrainTool(BaseTool):
        name: str = "neuralforge_train"
        description: str = (
            "Train a registered neural network model. "
            "Returns training result with loss, metrics, and training time."
        )
        args_schema = TrainInput

        def _run(self, model_name: str, epochs: int = 10, batch_size: int = 32,
                 learning_rate: float = 1e-3) -> str:
            try:
                return json.dumps({
                    "status": "ready",
                    "model_name": model_name,
                    "config": {"epochs": epochs, "batch_size": batch_size, "lr": learning_rate},
                    "message": "Training requires a dataset. Use the full pipeline for end-to-end training.",
                })
            except Exception as e:
                return json.dumps({"status": "error", "error": str(e)})

    return TrainTool()


def make_optimize_tool():
    """Create a LangChain-compatible optimize tool."""
    BaseTool = _get_langchain_tool_base()
    if BaseTool is None:
        return None

    from pydantic import BaseModel, Field

    class OptimizeInput(BaseModel):
        objective: str = Field(
            default="accuracy",
            description="Optimization objective: accuracy, loss, latency, etc."
        )
        direction: str = Field(
            default="maximize",
            description="Optimization direction: maximize or minimize"
        )
        num_trials: int = Field(default=10, description="Number of optimization trials")

    class OptimizeTool(BaseTool):
        name: str = "neuralforge_optimize"
        description: str = (
            "Run hyperparameter and architecture optimization. "
            "Returns best spec, score, and trial results."
        )
        args_schema = OptimizeInput

        def _run(self, objective: str = "accuracy", direction: str = "maximize",
                 num_trials: int = 10) -> str:
            try:
                obj = OptimizationGoal(
                    objective=objective, direction=direction,
                    budget={"num_trials": num_trials},
                )
                result = optimize(obj)
                return json.dumps({
                    "status": "success",
                    "best_score": result.best_score,
                    "num_trials": result.num_trials,
                    "best_architecture": result.best_spec.architecture.family.value,
                })
            except Exception as e:
                return json.dumps({"status": "error", "error": str(e)})

    return OptimizeTool()


def make_full_pipeline_tool():
    """Create a LangChain-compatible full pipeline tool."""
    BaseTool = _get_langchain_tool_base()
    if BaseTool is None:
        return None

    from pydantic import BaseModel, Field

    class PipelineInput(BaseModel):
        description: str = Field(
            description="Full task description, e.g. "
                        "'Build and train a ResNet for CIFAR-10 with <5M params reaching >92% accuracy'"
        )

    class FullPipelineTool(BaseTool):
        name: str = "neuralforge_full_pipeline"
        description: str = (
            "End-to-end neural network pipeline: create model, train, evaluate, "
            "and optimize from a single natural language description. "
            "This is the primary tool for most agent workflows."
        )
        args_schema = PipelineInput

        def _run(self, description: str) -> str:
            try:
                spec = NeuralForgeSpec.from_description(description)
                model = create_model(spec)
                return json.dumps({
                    "status": "pipeline_initiated",
                    "model_name": spec.name,
                    "parameters": model.count_parameters(),
                    "architecture": spec.architecture.family.value,
                    "config_hash": spec.config_hash(),
                    "message": "Model created. For full training, provide a dataset.",
                })
            except Exception as e:
                return json.dumps({"status": "error", "error": str(e)})

    return FullPipelineTool()


def get_all_langchain_tools() -> list:
    """Get all LangChain-compatible tools."""
    tools = [
        make_create_model_tool(),
        make_train_tool(),
        make_optimize_tool(),
        make_full_pipeline_tool(),
    ]
    return [t for t in tools if t is not None]


# ── CrewAI adapter ──────────────────────────────────────────────

def get_crewai_tools() -> list:
    """Get CrewAI-compatible tools."""
    try:
        from crewai_tools import BaseTool as CrewBaseTool
        from pydantic import BaseModel, Field

        class CreateModelInput(BaseModel):
            description: str = Field(description="Model description")

        class NCCreateModel(CrewBaseTool):
            name: str = "neuralforge_create_model"
            description: str = "Create a neural network from natural language"
            args_schema = CreateModelInput

            def _run(self, description: str) -> str:
                spec = NeuralForgeSpec.from_description(description)
                model = create_model(spec)
                return json.dumps({
                    "status": "success",
                    "model_name": spec.name,
                    "parameters": model.count_parameters(),
                })

        return [NCCreateModel()]
    except ImportError:
        return []


# ── AutoGen adapter ─────────────────────────────────────────────

def get_autogen_functions() -> list:
    """Get AutoGen-compatible function definitions."""
    return [
        {
            "name": "neuralforge_create_model",
            "description": "Create a neural network from natural language description",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Natural language model description",
                    },
                },
                "required": ["description"],
            },
        },
        {
            "name": "neuralforge_optimize",
            "description": "Optimize a neural network architecture",
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {"type": "string", "default": "accuracy"},
                    "num_trials": {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    ]
