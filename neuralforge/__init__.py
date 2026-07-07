"""
NeuralForge v2.5.0 — Agent-Native Neural Network Forge

A pluggable toolkit that turns any AI agent into a first-class neural network
creator, trainer, optimizer, and scientist. Includes real-time evolution,
AGNT runtime bridge, and held-out benchmark flows.
"""

__version__ = "2.5.0"

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

# Analysis & Evolution (real-time + benchmarks + AGNT bridge)
from neuralforge.analyzer import WorkflowAnalyzer
from neuralforge.smart_engine import SmartEngine
from neuralforge.pattern_engine import PatternEngine
from neuralforge.learner import DataLearner
from neuralforge.realtime_evo import RealtimeEvolutionEngine
from neuralforge.agnt_bridge import (
    record_execution,
    get_health_summary,
    rehydrate_from_log,
    normalize_agnt_event,
)
from neuralforge.benchmark import (
    run_full_benchmark,
    run_learner_benchmark,
    generate_synthetic_executions,
    split_data,
    BenchmarkResult,
)

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

# Tesseract Pathway Network
from neuralforge.tesseract import (
    TesseractRouter,
    TesseractPathwayNetwork,
    TesseractPathwayBlock,
    AxisScores,
    build_route_state,
    validate_tesseract,
)

from neuralforge.tesseract import (
    SyntheticTesseractRouteDataset,
    make_tesseract_loaders,
    evaluate_tpn_model,
    train_tpn_synthetic,
)

from neuralforge.tesseract import (
    TesseractSparseDispatcher,
    build_tesseract_receipts,
)

from neuralforge.tesseract import (
    TesseractSparseDispatcher,
    build_tesseract_receipts,
    TesseractCheckpointConfig,
    train_tpn_checkpoint,
    save_tpn_checkpoint,
    load_tpn_checkpoint,
    TesseractMindCore,
)

from neuralforge.tesseract import (
    receipt_to_english,
    receipts_to_english,
    outputs_to_english,
    TesseractEnglishAdapter,
)

from neuralforge.tesseract import (
    TesseractFeedbackRecord,
    TesseractReplayLedger,
    append_operator_feedback,
    seed_replay_from_synthetic,
    train_tpn_from_replay,
)

from neuralforge.tesseract import (
    TesseractWarmRuntime,
)

from neuralforge.tesseract import (
    CommandVectorizer,
    TesseractActionPacket,
    TesseractCommandMind,
)

from neuralforge.tesseract import (
    JarvisServiceConfig,
    TesseractActionLedger,
    TesseractJarvisRuntime,
)

from neuralforge.tesseract import (
    JARVIS_VERSION,
    API_CONTRACT_VERSION,
    ACTION_PACKET_VERSION,
    contract_manifest,
)

from neuralforge.tesseract import TesseractIntegrationBus
