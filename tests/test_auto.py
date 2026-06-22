"""Tests for auto-architecture and NAS."""
from __future__ import annotations
import pytest
import torch
from neuralforge.spec import (
    NeuralForgeSpec, ArchitectureSpec, ArchitectureFamily,
    DataProfile, TaskType, Constraints,
)
from neuralforge.auto.architect import ArchitectAgent, ARCHITECTURE_TEMPLATES
from neuralforge.auto.nas import DifferentiableNAS, EvolutionarySearch
from neuralforge.auto.scaling import ScalingLawEstimator


class TestArchitectAgent:
    def test_propose_image_classification(self):
        agent = ArchitectAgent()
        dp = DataProfile(
            task_type=TaskType.IMAGE_CLASSIFICATION,
            input_shape=(3, 32, 32),
            num_classes=10,
        )
        proposals = agent.propose("Image classification", dp, num_proposals=3)
        assert len(proposals) >= 1
        assert len(proposals) <= 3

    def test_propose_text_classification(self):
        agent = ArchitectAgent()
        dp = DataProfile(
            task_type=TaskType.TEXT_CLASSIFICATION,
            input_shape=(128,),
            vocab_size=10000,
            num_classes=4,
            data_format="text",
        )
        proposals = agent.propose("Text classification", dp, num_proposals=2)
        assert len(proposals) >= 1

    def test_proposals_are_valid_specs(self):
        agent = ArchitectAgent()
        dp = DataProfile(
            task_type=TaskType.IMAGE_CLASSIFICATION,
            input_shape=(3, 32, 32),
            num_classes=10,
        )
        proposals = agent.propose("Test", dp, num_proposals=3)
        for p in proposals:
            assert isinstance(p, NeuralForgeSpec)
            assert p.architecture.family != ArchitectureFamily.AUTO

    def test_describe_architecture(self):
        agent = ArchitectAgent()
        spec = NeuralForgeSpec(
            name="test",
            architecture=ArchitectureSpec(
                family=ArchitectureFamily.RESNET, depth=4, width=64
            ),
        )
        desc = agent.describe_architecture(spec)
        assert "resnet" in desc.lower()
        assert "4" in desc

    def test_templates_populated(self):
        assert len(ARCHITECTURE_TEMPLATES) > 0
        assert ArchitectureFamily.CNN in ARCHITECTURE_TEMPLATES
        assert ArchitectureFamily.RESNET in ARCHITECTURE_TEMPLATES
        assert ArchitectureFamily.TRANSFORMER in ARCHITECTURE_TEMPLATES


class TestEvolutionarySearch:
    def test_search_returns_spec(self):
        search = EvolutionarySearch(
            population_size=5, generations=2, mutation_rate=0.3
        )
        dp = DataProfile(
            task_type=TaskType.IMAGE_CLASSIFICATION,
            input_shape=(3, 32, 32),
            num_classes=10,
        )
        result = search.search(dp, Constraints())
        assert isinstance(result, NeuralForgeSpec)


class TestScalingLawEstimator:
    def test_add_observation(self):
        est = ScalingLawEstimator()
        est.add_observation(1_000_000, 50_000, 0.5)
        assert len(est.observations) == 1

    def test_fit(self):
        est = ScalingLawEstimator()
        est.add_observation(100_000, 10_000, 1.0)
        est.add_observation(1_000_000, 50_000, 0.5)
        est.add_observation(10_000_000, 100_000, 0.3)
        params = est.fit()
        assert "E" in params
        assert "alpha" in params

    def test_predict(self):
        est = ScalingLawEstimator()
        est.add_observation(1_000_000, 50_000, 0.5)
        est.add_observation(2_000_000, 50_000, 0.4)
        est.add_observation(5_000_000, 50_000, 0.3)
        loss = est.predict_loss(3_000_000, 50_000)
        assert isinstance(loss, float)
        assert loss > 0

    def test_insufficient_data(self):
        est = ScalingLawEstimator()
        params = est.fit()
        assert params is not None  # Returns defaults
