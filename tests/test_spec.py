"""Tests for NeuralForgeSpec and related Pydantic models."""
from __future__ import annotations
import pytest
from neuralforge.spec import (
    NeuralForgeSpec, TrainingConfig, OptimizationGoal, DataProfile,
    Constraints, ComputeBudget, ArchitectureSpec, ExportConfig, ExportFormat,
    ArchitectureFamily, TaskType, OptimizerName, SchedulerName, Precision,
)


class TestNeuralForgeSpec:
    def test_basic_creation(self):
        spec = NeuralForgeSpec(name="test-model")
        assert spec.name == "test-model"
        assert spec.task_type == TaskType.IMAGE_CLASSIFICATION
        assert spec.architecture.family == ArchitectureFamily.AUTO

    def test_config_hash_consistency(self):
        spec1 = NeuralForgeSpec(name="test", task_type=TaskType.IMAGE_CLASSIFICATION)
        spec2 = NeuralForgeSpec(name="test", task_type=TaskType.IMAGE_CLASSIFICATION)
        assert spec1.config_hash() == spec2.config_hash()

    def test_config_hash_different(self):
        spec1 = NeuralForgeSpec(name="test1")
        spec2 = NeuralForgeSpec(name="test2")
        assert spec1.config_hash() != spec2.config_hash()

    def test_yaml_roundtrip(self):
        spec = NeuralForgeSpec(
            name="yaml-test",
            task_type=TaskType.TEXT_CLASSIFICATION,
            architecture=ArchitectureSpec(family=ArchitectureFamily.TRANSFORMER, depth=4),
        )
        yaml_str = spec.to_yaml()
        restored = NeuralForgeSpec.from_yaml(yaml_str)
        assert restored.name == "yaml-test"
        assert restored.task_type == TaskType.TEXT_CLASSIFICATION
        assert restored.architecture.family == ArchitectureFamily.TRANSFORMER

    def test_from_description_cifar(self):
        spec = NeuralForgeSpec.from_description(
            "Build a ResNet for CIFAR-10 image classification with <5M params"
        )
        assert "cifar" in spec.name.lower() or "10" in spec.name
        assert spec.architecture.family == ArchitectureFamily.RESNET
        assert spec.constraints.max_parameters == 5_000_000

    def test_from_description_accuracy(self):
        spec = NeuralForgeSpec.from_description(
            "Build a model with >92% accuracy on image classification"
        )
        assert spec.constraints.min_accuracy == 0.92

    def test_from_description_transformer(self):
        spec = NeuralForgeSpec.from_description(
            "Build a transformer for text generation"
        )
        assert spec.architecture.family == ArchitectureFamily.TRANSFORMER
        assert spec.task_type == TaskType.TEXT_GENERATION


class TestTrainingConfig:
    def test_defaults(self):
        config = TrainingConfig()
        assert config.optimizer == OptimizerName.ADAMW
        assert config.learning_rate == 1e-3
        assert config.scheduler == SchedulerName.COSINE
        assert config.precision == Precision.MIXED
        assert config.seed == 42

    def test_custom(self):
        config = TrainingConfig(
            optimizer=OptimizerName.SGD,
            learning_rate=0.01,
            momentum=0.9,
            scheduler=SchedulerName.ONE_CYCLE,
        )
        assert config.optimizer == OptimizerName.SGD
        assert config.learning_rate == 0.01


class TestDataProfile:
    def test_image_profile(self):
        dp = DataProfile(
            task_type=TaskType.IMAGE_CLASSIFICATION,
            input_shape=(3, 32, 32),
            num_classes=10,
            num_channels=3,
        )
        assert dp.num_classes == 10
        assert dp.data_format == "image"

    def test_text_profile(self):
        dp = DataProfile(
            task_type=TaskType.TEXT_CLASSIFICATION,
            input_shape=(128,),
            vocab_size=10000,
            num_classes=4,
            max_sequence_length=128,
            data_format="text",
        )
        assert dp.vocab_size == 10000
        assert dp.max_sequence_length == 128


class TestConstraints:
    def test_defaults(self):
        c = Constraints()
        assert c.max_parameters is None
        assert c.target_hardware == ["cuda"]
        assert c.deterministic is False

    def test_custom(self):
        c = Constraints(max_parameters=5_000_000, max_memory_mb=4096)
        assert c.max_parameters == 5_000_000
        assert c.max_memory_mb == 4096


class TestOptimizationGoal:
    def test_basic(self):
        obj = OptimizationGoal(objective="accuracy", direction="maximize")
        assert obj.objective == "accuracy"
        assert obj.direction == "maximize"

    def test_with_constraints(self):
        obj = OptimizationGoal(
            objective="accuracy",
            constraints=Constraints(max_parameters=1_000_000),
        )
        assert obj.constraints.max_parameters == 1_000_000
