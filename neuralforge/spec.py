from __future__ import annotations
import hashlib, re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class ArchitectureFamily(str, Enum):
    TRANSFORMER = "transformer"; CNN = "cnn"; RESNET = "resnet"
    VISION_TRANSFORMER = "vit"; MAMBA = "mamba"; SSM = "ssm"
    RWKV = "rwkv"; LIQUID = "liquid"; DIFFUSION = "diffusion"
    FLOW_MATCHING = "flow_matching"; GRAPH_NN = "graph_nn"
    MIXTURE_OF_EXPERTS = "moe"; KAN = "kan"; RETNET = "retnet"
    MLP_MIXER = "mlp_mixer"; CUSTOM = "custom"; AUTO = "auto"


class Backend(str, Enum):
    PYTORCH = "pytorch"; JAX = "jax"; TENSORFLOW = "tensorflow"; TINYGRAD = "tinygrad"


class Precision(str, Enum):
    FP32 = "fp32"; FP16 = "fp16"; BF16 = "bf16"; INT8 = "int8"; INT4 = "int4"; MIXED = "mixed"


class DistributedStrategy(str, Enum):
    NONE = "none"; DDP = "ddp"; FSDP = "fsdp"; DEEPSPEED = "deepspeed"; JAX_PMAP = "jax_pmap"


class ExportFormat(str, Enum):
    TORCHSCRIPT = "torchscript"; ONNX = "onnx"; SAFETENSORS = "safetensors"
    GGUF = "gguf"; OPENVINO = "openvino"; TENSORRT = "tensorrt"
    COREML = "coreml"; JAX = "jax"; PYTORCH_STATE_DICT = "pytorch_state_dict"


class TaskType(str, Enum):
    IMAGE_CLASSIFICATION = "image_classification"; TEXT_CLASSIFICATION = "text_classification"
    TEXT_GENERATION = "text_generation"; OBJECT_DETECTION = "object_detection"
    IMAGE_SEGMENTATION = "image_segmentation"; REGRESSION = "regression"
    TIME_SERIES = "time_series"; ANOMALY_DETECTION = "anomaly_detection"
    REINFORCEMENT_LEARNING = "rl_policy"; MULTIMODAL = "multimodal"
    AUDIO_CLASSIFICATION = "audio_classification"; SEQUENCE_TO_SEQUENCE = "seq2seq"
    TOKEN_CLASSIFICATION = "token_classification"; QUESTION_ANSWERING = "question_answering"
    SPEECH_RECOGNITION = "speech_recognition"; GRAPH_PREDICTION = "graph_prediction"
    CUSTOM = "custom"


class OptimizerName(str, Enum):
    ADAM = "adam"; ADAMW = "adamw"; SGD = "sgd"; LION = "lion"
    ADAFACTOR = "adafactor"; LAMB = "lamb"; RMSPROP = "rmsprop"; ADAGRAD = "adagrad"


class SchedulerName(str, Enum):
    COSINE = "cosine"; COSINE_WARM_RESTARTS = "cosine_warm_restarts"
    ONE_CYCLE = "one_cycle"; REDUCE_ON_PLATEAU = "reduce_on_plateau"
    LINEAR_WARMUP = "linear_warmup"; POLYNOMIAL = "polynomial"
    STEP = "step"; CONSTANT = "constant"


class PruningMethod(str, Enum):
    UNSTRUCTURED_GLOBAL = "unstructured_global"; STRUCTURED_L1 = "structured_l1"
    STRUCTURED_LN = "structured_ln"; MOVING_BOUNDARY = "moving_boundary"
    LOTTERY_TICKET = "lottery_ticket"


class QuantizationMethod(str, Enum):
    GPTQ = "gptq"; AWQ = "awq"; BITSANDBYTES = "bitsandbytes"
    HQQ = "hqq"; DYNAMIC_INT8 = "dynamic_int8"; STATIC_INT8 = "static_int8"; QAT = "qat"


class DistillationMethod(str, Enum):
    SOFT_TARGETS = "soft_targets"; FITNET = "fitnet"
    ATTENTION_TRANSFER = "attention_transfer"; RELATION_KD = "relation_kd"; DKD = "dkd"


class DataProfile(BaseModel):
    task_type: TaskType
    input_shape: Tuple[int, ...]
    num_classes: Optional[int] = None
    num_samples: Optional[int] = None
    num_val_samples: Optional[int] = None
    num_test_samples: Optional[int] = None
    vocab_size: Optional[int] = None
    max_sequence_length: Optional[int] = None
    num_channels: Optional[int] = None
    class_distribution: Optional[Dict[str, float]] = None
    is_multimodal: bool = False
    modality_types: List[str] = Field(default_factory=list)
    data_format: str = "image"
    normalization: Optional[Dict[str, List[float]]] = None


class Constraints(BaseModel):
    max_parameters: Optional[int] = None
    max_flops: Optional[int] = None
    max_memory_mb: Optional[int] = None
    max_latency_ms: Optional[float] = None
    min_accuracy: Optional[float] = None
    target_hardware: List[str] = Field(default_factory=lambda: ["cuda"])
    required_export_formats: List[ExportFormat] = Field(default_factory=list)
    max_model_size_mb: Optional[int] = None
    must_support_batch_size: int = 1
    deterministic: bool = False


class ComputeBudget(BaseModel):
    max_hours: float = 24.0
    max_gpu_hours: Optional[float] = None
    max_cost_usd: Optional[float] = None
    num_gpus: int = 1
    gpu_type: Optional[str] = None
    num_trials: int = 10
    max_epochs: int = 100
    max_steps: Optional[int] = None
    early_stopping_patience: int = 10
    priority: str = "balanced"


class LayerSpec(BaseModel):
    type: str
    params: Dict[str, Any] = Field(default_factory=dict)
    name: Optional[str] = None
    activation: Optional[str] = "gelu"
    normalization: Optional[str] = None
    dropout: float = 0.0
    residual: bool = False


class ArchitectureSpec(BaseModel):
    family: ArchitectureFamily = ArchitectureFamily.AUTO
    layers: List[LayerSpec] = Field(default_factory=list)
    depth: Optional[int] = None
    width: Optional[int] = None
    num_heads: Optional[int] = None
    embedding_dim: Optional[int] = None
    num_experts: Optional[int] = None
    expert_capacity: Optional[int] = None
    dropout: float = 0.1
    activation: str = "gelu"
    normalization: str = "layer_norm"
    use_flash_attention: bool = False
    use_rotary_embeddings: bool = False
    custom_code: Optional[str] = None
    pretrained_model_name: Optional[str] = None


class TrainingConfig(BaseModel):
    optimizer: OptimizerName = OptimizerName.ADAMW
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9
    optimizer_extra: Dict[str, Any] = Field(default_factory=dict)
    scheduler: SchedulerName = SchedulerName.COSINE
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    min_lr: float = 1e-6
    scheduler_extra: Dict[str, Any] = Field(default_factory=dict)
    epochs: int = 10
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    precision: Precision = Precision.MIXED
    backend: Backend = Backend.PYTORCH
    distributed_strategy: DistributedStrategy = DistributedStrategy.NONE
    num_workers: int = 4
    pin_memory: bool = True
    compile_model: bool = False
    seed: int = 42
    deterministic_algorithms: bool = False
    log_every_n_steps: int = 50
    eval_every_n_epochs: int = 1
    save_every_n_epochs: int = 1
    checkpoint_dir: str = "./checkpoints"
    resume_from: Optional[str] = None
    experiment_name: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    tracking_backend: Optional[str] = None
    tracking_project: str = "neuralforge"
    tracking_extra: Dict[str, Any] = Field(default_factory=dict)
    use_ema: bool = False
    ema_decay: float = 0.9999
    use_curriculum: bool = False
    curriculum_strategy: Optional[str] = None
    early_stopping_patience: int = 10


class OptimizationGoal(BaseModel):
    objective: str = "accuracy"
    direction: str = "maximize"
    target_value: Optional[float] = None
    constraints: Constraints = Field(default_factory=Constraints)
    multi_objective: List[Dict[str, Any]] = Field(default_factory=list)
    budget: ComputeBudget = Field(default_factory=ComputeBudget)


class ExportConfig(BaseModel):
    format: ExportFormat = ExportFormat.PYTORCH_STATE_DICT
    output_path: str = "./exported_model"
    opset_version: int = 17
    dynamic_axes: bool = True
    quantize_on_export: bool = False
    quantization_method: Optional[QuantizationMethod] = None
    optimize_for_inference: bool = True


class PruningConfig(BaseModel):
    method: PruningMethod = PruningMethod.UNSTRUCTURED_GLOBAL
    amount: float = 0.3
    schedule: str = "gradual"
    num_iterations: int = 5
    fine_tune_epochs: int = 3
    structured: bool = False


class QuantizationConfig(BaseModel):
    method: QuantizationMethod = QuantizationMethod.BITSANDBYTES
    bits: int = 8
    group_size: int = 128
    desc_act: bool = False
    damp_percent: float = 0.01


class DistillationConfig(BaseModel):
    method: DistillationMethod = DistillationMethod.SOFT_TARGETS
    teacher_model_name: Optional[str] = None
    teacher_model_path: Optional[str] = None
    temperature: float = 4.0
    alpha: float = 0.7


class LoRAConfig(BaseModel):
    enabled: bool = False
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    use_dora: bool = False
    use_qlora: bool = False


class EvolutionConfig(BaseModel):
    generations: int = 20
    population_size: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.3
    elite_fraction: float = 0.1
    fitness_fn: Optional[str] = None
    mutation_types: List[str] = Field(default_factory=lambda: ["add_layer", "remove_layer", "change_width", "change_activation"])


class NeuralForgeSpec(BaseModel):
    name: str = "neuralforge-model"
    description: Optional[str] = None
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = Field(default_factory=list)
    task_type: TaskType = TaskType.IMAGE_CLASSIFICATION
    data_profile: Optional[DataProfile] = None
    architecture: ArchitectureSpec = Field(default_factory=ArchitectureSpec)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    optimization: Optional[OptimizationGoal] = None
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    pruning: Optional[PruningConfig] = None
    quantization: Optional[QuantizationConfig] = None
    distillation: Optional[DistillationConfig] = None
    evolution: Optional[EvolutionConfig] = None
    export: Optional[ExportConfig] = None
    constraints: Constraints = Field(default_factory=Constraints)
    budget: ComputeBudget = Field(default_factory=ComputeBudget)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def config_hash(self) -> str:
        data = self.model_dump_json(exclude={"created_at", "metadata"})
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_yaml(self) -> str:
        import yaml
        return yaml.dump(self.model_dump(exclude={"created_at"}, mode="json"), default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "NeuralForgeSpec":
        import yaml
        data = yaml.safe_load(yaml_str)
        return cls(**data)

    @classmethod
    def from_description(cls, description: str, **overrides) -> "NeuralForgeSpec":
        desc_lower = description.lower()
        task_type = TaskType.CUSTOM
        task_keywords = {
            TaskType.IMAGE_CLASSIFICATION: ["image classif", "classify images", "image recognition"],
            TaskType.TEXT_CLASSIFICATION: ["text classif", "sentiment", "text categorization"],
            TaskType.TEXT_GENERATION: ["text generat", "language model", "llm", "generate text"],
            TaskType.OBJECT_DETECTION: ["object detect", "detect objects"],
            TaskType.IMAGE_SEGMENTATION: ["segment", "semantic segment"],
            TaskType.QUESTION_ANSWERING: ["question answer", "qa"],
            TaskType.TIME_SERIES: ["time series", "forecast"],
            TaskType.ANOMALY_DETECTION: ["anomaly", "outlier"],
            TaskType.MULTIMODAL: ["multimodal", "vision-language", "image-text"],
            TaskType.AUDIO_CLASSIFICATION: ["audio classif", "sound classif"],
            TaskType.SPEECH_RECOGNITION: ["speech", "asr"],
            TaskType.REINFORCEMENT_LEARNING: ["reinforcement", "rl", "policy"],
            TaskType.GRAPH_PREDICTION: ["graph", "node classif"],
        }
        for task, keywords in task_keywords.items():
            if any(kw in desc_lower for kw in keywords):
                task_type = task
                break
        arch_family = ArchitectureFamily.AUTO
        arch_keywords = [
            (ArchitectureFamily.VISION_TRANSFORMER, ["vision transformer", "vit", "swin"]),
            (ArchitectureFamily.TRANSFORMER, ["transformer", "bert", "gpt"]),
            (ArchitectureFamily.CNN, ["cnn", "convnet", "convolutional"]),
            (ArchitectureFamily.RESNET, ["resnet", "residual"]),
            (ArchitectureFamily.MAMBA, ["mamba", "selective ssm"]),
            (ArchitectureFamily.RWKV, ["rwkv"]),
            (ArchitectureFamily.DIFFUSION, ["diffusion", "ddpm"]),
            (ArchitectureFamily.MIXTURE_OF_EXPERTS, ["mixture of experts", "moe"]),
            (ArchitectureFamily.KAN, ["kan", "kolmogorov"]),
            (ArchitectureFamily.LIQUID, ["liquid", "neural ode"]),
        ]
        for arch, keywords in arch_keywords:
            if any(kw in desc_lower for kw in keywords):
                arch_family = arch
                break
        constraints = Constraints()
        param_match = re.search(r'<\s*(\d+(?:\.\d+)?)\s*m\s*param', desc_lower)
        if param_match:
            constraints.max_parameters = int(float(param_match.group(1)) * 1_000_000)
        acc_match = re.search(r'>?\s*(\d+(?:\.\d+)?)%\s*(?:accuracy|acc)', desc_lower)
        if acc_match:
            constraints.min_accuracy = float(acc_match.group(1)) / 100.0
        mem_match = re.search(r'<?\s*(\d+)\s*gb\s*(?:vram|gpu|memory)', desc_lower)
        if mem_match:
            constraints.max_memory_mb = int(mem_match.group(1)) * 1024
        time_match = re.search(r'within\s*(\d+)\s*min', desc_lower)
        if time_match:
            budget = ComputeBudget(max_hours=float(time_match.group(1)) / 60.0)
        else:
            time_match2 = re.search(r'within\s*(\d+)\s*hour', desc_lower)
            budget = ComputeBudget(max_hours=float(time_match2.group(1))) if time_match2 else ComputeBudget()
        architecture = ArchitectureSpec(family=arch_family)
        data_profile = None
        if "cifar-10" in desc_lower or "cifar10" in desc_lower:
            data_profile = DataProfile(task_type=task_type, input_shape=(3, 32, 32), num_classes=10, data_format="image")
        elif "mnist" in desc_lower:
            data_profile = DataProfile(task_type=task_type, input_shape=(1, 28, 28), num_classes=10, data_format="image")
        elif "imagenet" in desc_lower:
            data_profile = DataProfile(task_type=task_type, input_shape=(3, 224, 224), num_classes=1000, data_format="image")
        return cls(
            name=re.sub(r'[^\w\s-]', '', description.lower().strip())[:50].replace(' ', '-'),
            description=description, task_type=task_type, data_profile=data_profile,
            architecture=architecture, constraints=constraints, budget=budget, **overrides,
        )


class TrainingResult(BaseModel):
    model_name: str = ""
    spec_hash: str = ""
    epochs_completed: int = 0
    total_steps: int = 0
    final_loss: float = 0.0
    final_metrics: Dict[str, float] = Field(default_factory=dict)
    best_metric: float = 0.0
    best_epoch: int = 0
    training_time_seconds: float = 0.0
    checkpoints: List[str] = Field(default_factory=list)
    history: Dict[str, List[float]] = Field(default_factory=dict)
    status: str = "completed"
    error: Optional[str] = None


class EvaluationReport(BaseModel):
    model_name: str = "model"
    metrics: Dict[str, float] = Field(default_factory=dict)
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    robustness_scores: Optional[Dict[str, float]] = None
    calibration_error: Optional[float] = None
    ood_scores: Optional[Dict[str, float]] = None
    visualizations: List[str] = Field(default_factory=list)
    failure_analysis: Optional[Dict[str, Any]] = None
    recommendations: List[str] = Field(default_factory=list)


class OptimizationResult(BaseModel):
    best_spec: NeuralForgeSpec = Field(default_factory=NeuralForgeSpec)
    best_score: float = 0.0
    num_trials: int = 0
    total_time_seconds: float = 0.0
    pareto_front: Optional[List[Dict[str, Any]]] = None
    ablation_results: Optional[Dict[str, float]] = None
    scaling_law_fit: Optional[Dict[str, float]] = None
    all_trials: List[Dict[str, Any]] = Field(default_factory=list)
