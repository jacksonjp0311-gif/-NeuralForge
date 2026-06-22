"""Tests for training engine and related components."""
from __future__ import annotations
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from neuralforge.spec import (
    NeuralForgeSpec, TrainingConfig, ArchitectureSpec, ArchitectureFamily,
    DataProfile, TaskType, OptimizerName, SchedulerName, Precision,
)
from neuralforge.core.forge import create_model
from neuralforge.training.engine import (
    TrainingEngine, EarlyStopping, ExponentialMovingAverage,
    _build_optimizer, _build_scheduler,
)


class TestEarlyStopping:
    def test_no_stop(self):
        es = EarlyStopping(patience=3)
        assert not es(1.0)
        assert not es(0.9)
        assert not es(0.8)

    def test_triggers(self):
        es = EarlyStopping(patience=2)
        assert not es(1.0)
        assert not es(1.1)
        assert not es(1.2)
        assert es(1.3)

    def test_improvement_resets(self):
        es = EarlyStopping(patience=2)
        assert not es(1.0)
        assert not es(1.1)
        assert not es(0.5)  # improvement resets counter
        assert not es(0.6)
        assert not es(0.7)
        assert es(0.8)


class TestEMA:
    def test_ema_creation(self):
        model = nn.Linear(10, 5)
        ema = ExponentialMovingAverage(model, decay=0.999)
        assert len(ema.shadow) > 0

    def test_ema_update(self):
        model = nn.Linear(10, 5)
        ema = ExponentialMovingAverage(model, decay=0.9)
        old_shadow = {k: v.clone() for k, v in ema.shadow.items()}
        ema.update(model)
        for k in ema.shadow:
            assert not torch.equal(ema.shadow[k], old_shadow[k])

    def test_ema_apply_restore(self):
        model = nn.Linear(10, 5)
        ema = ExponentialMovingAverage(model)
        old_params = {k: v.data.clone() for k, v in model.named_parameters()}
        ema.apply_shadow(model)
        ema.restore(model)
        for name, param in model.named_parameters():
            assert torch.equal(param.data, old_params[name])


class TestOptimizerBuilder:
    def test_adamw(self):
        model = nn.Linear(10, 5)
        config = TrainingConfig(optimizer=OptimizerName.ADAMW, learning_rate=1e-3)
        opt = _build_optimizer(model, config)
        assert isinstance(opt, torch.optim.AdamW)

    def test_sgd(self):
        model = nn.Linear(10, 5)
        config = TrainingConfig(optimizer=OptimizerName.SGD, learning_rate=0.01, momentum=0.9)
        opt = _build_optimizer(model, config)
        assert isinstance(opt, torch.optim.SGD)

    def test_adam(self):
        model = nn.Linear(10, 5)
        config = TrainingConfig(optimizer=OptimizerName.ADAM, learning_rate=1e-3)
        opt = _build_optimizer(model, config)
        assert isinstance(opt, torch.optim.Adam)


class TestSchedulerBuilder:
    def test_cosine(self):
        model = nn.Linear(10, 5)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        config = TrainingConfig(scheduler=SchedulerName.COSINE)
        sched = _build_scheduler(opt, config, total_steps=100)
        assert sched is not None

    def test_one_cycle(self):
        model = nn.Linear(10, 5)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        config = TrainingConfig(scheduler=SchedulerName.ONE_CYCLE)
        sched = _build_scheduler(opt, config, total_steps=100)
        assert sched is not None

    def test_warmup(self):
        model = nn.Linear(10, 5)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        config = TrainingConfig(
            scheduler=SchedulerName.COSINE, warmup_steps=10
        )
        sched = _build_scheduler(opt, config, total_steps=100)
        assert sched is not None


class TestTrainingEngine:
    @pytest.fixture
    def simple_model(self):
        return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    @pytest.fixture
    def simple_loader(self):
        X = torch.randn(64, 10)
        y = torch.randint(0, 3, (64,))
        return DataLoader(TensorDataset(X, y), batch_size=16)

    def test_train_basic(self, simple_model, simple_loader):
        spec = NeuralForgeSpec(name="train-test")
        config = TrainingConfig(epochs=2, batch_size=16, seed=42)
        engine = TrainingEngine(simple_model, spec, config)
        result = engine.train(simple_loader)
        assert result.epochs_completed == 2
        assert result.status == "completed"
        assert len(result.history["train_loss"]) == 2

    def test_train_with_validation(self, simple_model, simple_loader):
        spec = NeuralForgeSpec(name="train-val-test")
        config = TrainingConfig(epochs=3, batch_size=16, seed=42)
        engine = TrainingEngine(simple_model, spec, config)

        X_val = torch.randn(32, 10)
        y_val = torch.randint(0, 3, (32,))
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=16)

        result = engine.train(simple_loader, val_loader)
        assert result.epochs_completed == 3
        assert len(result.history["val_loss"]) == 3

    def test_early_stopping(self):
        """Test that early stopping triggers when loss doesn't improve."""
        spec = NeuralForgeSpec(name="es-test")
        config = TrainingConfig(
            epochs=100, batch_size=16, seed=42, early_stopping_patience=2
        )
        # Create early stopping directly and verify logic
        from neuralforge.training.engine import EarlyStopping
        es = EarlyStopping(patience=2)
        # Simulate plateau
        assert not es(1.0)  # initial
        assert not es(1.0)  # no improvement (counter=1)
        assert not es(1.0)  # no improvement (counter=2)
        assert es(1.0)      # triggers (counter=3 > patience=2)

    def test_early_stopping_with_training(self, simple_model, simple_loader):
        """Test that training with very short patience stops before max epochs."""
        spec = NeuralForgeSpec(name="es-test-2")
        config = TrainingConfig(
            epochs=200, batch_size=16, seed=42, early_stopping_patience=1
        )
        engine = TrainingEngine(simple_model, spec, config)
        result = engine.train(simple_loader)
        # Should stop before hitting the max (may take many epochs on easy data)
        assert result.epochs_completed < 200
        assert result.status == "early_stopped"

    def test_ema_training(self, simple_model, simple_loader):
        spec = NeuralForgeSpec(name="ema-test")
        config = TrainingConfig(epochs=2, batch_size=16, seed=42, use_ema=True)
        engine = TrainingEngine(simple_model, spec, config)
        result = engine.train(simple_loader)
        assert engine.ema is not None
        assert result.epochs_completed == 2

    def test_reproducibility(self, simple_loader):
        spec = NeuralForgeSpec(name="repro-test")
        config = TrainingConfig(epochs=2, batch_size=16, seed=42)

        # Create two identical models
        torch.manual_seed(99)
        m1 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))
        torch.manual_seed(99)
        m2 = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))

        engine1 = TrainingEngine(m1, spec, config)
        result1 = engine1.train(simple_loader)

        engine2 = TrainingEngine(m2, spec, config)
        result2 = engine2.train(simple_loader)

        assert abs(result1.final_loss - result2.final_loss) < 1e-5
