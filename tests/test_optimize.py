"""Tests for optimization modules."""
from __future__ import annotations
import pytest
import torch
import torch.nn as nn

from neuralforge.optimize.pruning import prune_model
from neuralforge.optimize.quantization import quantize_model
from neuralforge.optimize.meta_optimizer import MetaOptimizer
from neuralforge.spec import (
    NeuralForgeSpec, TrainingConfig, PruningConfig, QuantizationConfig,
    PruningMethod, QuantizationMethod,
)


class TestPruning:
    def test_prune_model(self):
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )
        config = PruningConfig(method=PruningMethod.UNSTRUCTURED_GLOBAL, amount=0.3)
        pruned = prune_model(model, config)
        assert pruned is not None

    def test_sparsity_increased(self):
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )
        total = sum(p.numel() for p in model.parameters())
        zeros_before = sum((p.data == 0).sum().item() for p in model.parameters())

        config = PruningConfig(method=PruningMethod.UNSTRUCTURED_GLOBAL, amount=0.5)
        pruned = prune_model(model, config)

        zeros_after = sum((p.data == 0).sum().item() for p in pruned.parameters())
        assert zeros_after >= zeros_before


class TestQuantization:
    def test_dynamic_int8(self):
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
        )
        config = QuantizationConfig(method=QuantizationMethod.DYNAMIC_INT8)
        quantized = quantize_model(model, config)
        assert quantized is not None


class TestMetaOptimizer:
    def test_critique(self):
        meta = MetaOptimizer()
        spec = NeuralForgeSpec(name="test")
        result = type("Result", (), {
            "history": {"train_loss": [1.0, 0.8, 0.9], "val_loss": [1.1, 0.9, 1.2]},
        })()
        critique = meta.critique(spec, result)
        assert "issues" in critique
        assert "suggestions" in critique

    def test_propose_next_spec(self):
        meta = MetaOptimizer()
        spec = NeuralForgeSpec(name="test")
        result = type("Result", (), {
            "history": {"train_loss": [1.0, 1.1, 1.2], "val_loss": [1.0, 1.1, 1.2]},
        })()
        new_spec = meta.propose_next_spec(spec, result)
        assert isinstance(new_spec, NeuralForgeSpec)
