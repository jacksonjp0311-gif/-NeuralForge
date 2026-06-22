"""Example 3: Reinforcement Learning Policy Network.

This example demonstrates:
- Building a policy gradient network for RL
- Multi-output architecture (policy + value head)
- Evolution-based architecture search
- Using the meta-optimizer for self-improvement

Usage:
    python examples/example_3_rl_policy.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from neuralforge.spec import (
    NeuralForgeSpec, TrainingConfig, TaskType, DataProfile,
    ArchitectureFamily, ArchitectureSpec, OptimizerName, SchedulerName,
)
from neuralforge.core.forge import create_model, train, evolve
from neuralforge.training.engine import TrainingEngine
from neuralforge.optimize.meta_optimizer import MetaOptimizer
from neuralforge.memory.insights_store import InsightsStore


class PolicyValueNetwork(nn.Module):
    """Combined policy and value network for RL."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        shared = self.shared(x)
        logits = self.policy_head(shared)
        value = self.value_head(shared)
        return logits


def main():
    print("=" * 64)
    print("NeuralForge v2.0 — Example 3: RL Policy Network")
    print("=" * 64)

    # ── Step 1: Create RL policy network ──────────────────────────
    obs_dim = 8
    act_dim = 4
    model = PolicyValueNetwork(obs_dim, act_dim, hidden_dim=128)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[Model] Parameters: {total_params:,}")

    # ── Step 2: Create synthetic RL training data ─────────────────
    torch.manual_seed(42)
    X = torch.randn(1000, obs_dim)
    y = torch.randint(0, act_dim, (1000,))

    train_ds = TensorDataset(X, y)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # ── Step 3: Train ─────────────────────────────────────────────
    config = TrainingConfig(
        epochs=10,
        batch_size=64,
        learning_rate=3e-4,
        optimizer=OptimizerName.ADAM,
        scheduler=SchedulerName.COSINE,
        seed=42,
    )

    spec = NeuralForgeSpec(
        name="rl-policy",
        task_type=TaskType.REINFORCEMENT_LEARNING,
        data_profile=DataProfile(
            task_type=TaskType.REINFORCEMENT_LEARNING,
            input_shape=(obs_dim,),
        ),
    )

    engine = TrainingEngine(model, spec, config)
    result = engine.train(train_loader)

    print(f"\n[Training Result]")
    print(f"  Epochs: {result.epochs_completed}")
    print(f"  Best metric: {result.best_metric:.4f}")
    print(f"  Time: {result.training_time_seconds:.1f}s")

    # ── Step 4: Meta-optimizer critique ───────────────────────────
    meta = MetaOptimizer()
    critique = meta.critique(spec, result)

    print(f"\n[Critique] Score: {critique['score']}")
    if critique["issues"]:
        print("  Issues:")
        for issue in critique["issues"]:
            print(f"    ⚠ {issue}")
    if critique["suggestions"]:
        print("  Suggestions:")
        for sug in critique["suggestions"]:
            print(f"    → {sug}")

    # ── Step 5: Store insights ────────────────────────────────────
    store = InsightsStore()
    store.add_insight(
        spec_summary=f"RL policy: obs_dim={obs_dim}, act_dim={act_dim}",
        result_summary=f"best_loss={result.best_metric:.4f}",
        metrics={"accuracy": 1.0 - result.best_metric, "epoches": result.epochs_completed},
    )

    stats = store.get_statistics()
    print(f"\n[Insights] Store stats: {stats}")

    print("\n" + "=" * 64)
    print("Example 3 complete!")
    print("=" * 64)


if __name__ == "__main__":
    main()
