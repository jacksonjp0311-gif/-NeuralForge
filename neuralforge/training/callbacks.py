from __future__ import annotations
import json, logging, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("neuralforge.training.callbacks")


class ModelCheckpoint:
    """Save model checkpoints based on metric improvement."""

    def __init__(
        self,
        save_dir: str = "./checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 3,
        save_every_n_epochs: int = 1,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_every_n_epochs = save_every_n_epochs
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.saved_paths: List[str] = []

    def __call__(self, epoch: int, model: nn.Module, metrics: Dict[str, float]):
        value = metrics.get(self.monitor)
        if value is None:
            return

        improved = (
            value < self.best_value if self.mode == "min" else value > self.best_value
        )
        should_save = improved or (epoch % self.save_every_n_epochs == 0)

        if should_save:
            path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                },
                str(path),
            )
            self.saved_paths.append(str(path))
            self.best_value = value
            logger.info(f"Checkpoint saved: {path}")


class WandbCallback:
    """Weights & Biases logging callback."""

    def __init__(self, project: str = "neuralforge", run_name: str = None, **kwargs):
        self.project = project
        self.run_name = run_name
        self.kwargs = kwargs
        self._run = None

    def _init(self):
        if self._run is None:
            try:
                import wandb
                self._run = wandb.init(
                    project=self.project, name=self.run_name, **self.kwargs
                )
            except ImportError:
                logger.warning("wandb not installed, skipping W&B logging")

    def __call__(self, epoch: int, model: nn.Module, metrics: Dict[str, float]):
        self._init()
        if self._run is not None:
            import wandb
            log_data = {k: v for k, v in metrics.items() if v is not None}
            log_data["epoch"] = epoch
            wandb.log(log_data)

    def finish(self):
        if self._run is not None:
            import wandb
            wandb.finish()


class TensorBoardCallback:
    """TensorBoard logging callback."""

    def __init__(self, log_dir: str = "./runs"):
        self.log_dir = log_dir
        self._writer = None

    def _init(self):
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(self.log_dir)
            except ImportError:
                logger.warning("tensorboard not installed")

    def __call__(self, epoch: int, model: nn.Module, metrics: Dict[str, float]):
        self._init()
        if self._writer is not None:
            for k, v in metrics.items():
                if v is not None:
                    self._writer.add_scalar(k, v, epoch)

    def finish(self):
        if self._writer is not None:
            self._writer.close()


class ConsoleLogger:
    """Rich console logging callback."""

    def __init__(self, log_every: int = 1):
        self.log_every = log_every

    def __call__(self, epoch: int, model: nn.Module, metrics: Dict[str, float]):
        if epoch % self.log_every == 0:
            parts = [f"{k}={v:.4f}" for k, v in metrics.items() if v is not None]
            logger.info(f"Epoch {epoch}: {' | '.join(parts)}")


class LearningRateFinder:
    """LR range test callback."""

    def __init__(self, start_lr: float = 1e-7, end_lr: float = 10.0, num_steps: int = 100):
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_steps = num_steps
        self.lrs: List[float] = []
        self.losses: List[float] = []

    def run(self, model, train_loader, device):
        """Run LR finder and return suggested LR."""
        import torch.nn.functional as F

        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.start_lr)
        gamma = (self.end_lr / self.start_lr) ** (1 / self.num_steps)

        for step, batch in enumerate(train_loader):
            if step >= self.num_steps:
                break
            inputs, targets = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()

            self.lrs.append(optimizer.param_groups[0]["lr"])
            self.losses.append(loss.item())
            optimizer.param_groups[0]["lr"] *= gamma

        # Find steepest descent
        if len(self.losses) > 2:
            import numpy as np
            losses = np.array(self.losses)
            lrs = np.array(self.lrs)
            grad = np.gradient(losses, lrs)
            best_idx = np.argmin(grad)
            return float(lrs[best_idx])
        return self.start_lr
