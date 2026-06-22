"""
NeuralForge v2.0 — Agent-Native Neural Network Forge

A production-ready, pluggable toolkit that turns any AI agent into a
first-class neural network creator, trainer, optimizer, and scientist.
"""

__version__ = "2.0.0"

# Core
from neuralforge.spec import (
    NeuralForgeSpec,
    TrainingConfig,
    OptimizationGoal,
    ComputeBudget,
    DataProfile,
    Constraints,
    ArchitectureSpec,
    ExportConfig,
    ExportFormat,
    TrainingResult,
    EvaluationReport,
    OptimizationResult,
    ArchitectureFamily,
    Backend,
    Precision,
    DistributedStrategy,
    TaskType,
    OptimizerName,
    SchedulerName,
    LoRAConfig,
    PruningConfig,
    QuantizationConfig,
    DistillationConfig,
)

from neuralforge.core.forge import (
    NeuralForgeEngine,
    NeuralForgeModule,
    create_model,
    train,
    optimize,
    evaluate_and_report,
    evolve,
    auto_architecture,
    export_model,
)

from neuralforge.core.registry import ModelRegistry

# Training
from neuralforge.training.engine import (
    TrainingEngine,
    EarlyStopping,
    ExponentialMovingAverage,
)
from neuralforge.training.callbacks import (
    ModelCheckpoint,
    WandbCallback,
    TensorBoardCallback,
    ConsoleLogger,
    LearningRateFinder,
)

# Evaluation
from neuralforge.evaluation.evaluator import ModelEvaluator

# Tools
from neuralforge.tools.agent_tool import NeuralForgeAgentTool, as_tool
from neuralforge.tools.langchain_tools import (
    get_all_langchain_tools,
    get_crewai_tools,
    get_autogen_functions,
)
from neuralforge.tools.multi_agent import ForgeOrchestrator, ForgeSession

# Auto
from neuralforge.auto.architect import ArchitectAgent
from neuralforge.auto.scaling import ScalingLawEstimator

# Optimize
from neuralforge.optimize.meta_optimizer import MetaOptimizer

# Memory
from neuralforge.memory.insights_store import InsightsStore

# Utils
from neuralforge.utils.profiling import profile_model
from neuralforge.utils.export import export_model as export_util
from neuralforge.utils.visualization import (
    plot_training_history,
    plot_confusion_matrix,
)


def get_engine():
    """Get the global NeuralForge engine instance."""
    return NeuralForgeEngine.get_instance()


def list_models():
    """List all registered models."""
    return ModelRegistry.get_instance().list_models()


def quick_build(description: str):
    """Quick-build a model from a natural language description.

    Example:
        model = quick_build("ResNet for CIFAR-10 with <5M params")
    """
    spec = NeuralForgeSpec.from_description(description)
    return create_model(spec)
