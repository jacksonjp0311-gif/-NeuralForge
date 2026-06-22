from __future__ import annotations
import logging, math, time, os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from neuralforge.spec import (
    NeuralForgeSpec, TrainingConfig, TrainingResult,
    OptimizerName, SchedulerName, Precision, DistributedStrategy,
)

logger = logging.getLogger("neuralforge.training.engine")


class EarlyStopping:
    """Early stopping handler with patience and min_delta."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        if self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            improved = value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.should_stop = True

        return self.should_stop


class ExponentialMovingAverage:
    """EMA for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class GradualWarmupScheduler:
    """Linear warmup followed by another scheduler."""

    def __init__(self, optimizer, warmup_steps: int, after_scheduler=None):
        self.warmup_steps = warmup_steps
        self.after_scheduler = after_scheduler
        self.current_step = 0
        self.optimizer = optimizer

    def step(self, epoch=None):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            for pg in self.optimizer.param_groups:
                pg["lr"] = (
                    pg["initial_lr"] * self.current_step / max(self.warmup_steps, 1)
                )
        elif self.after_scheduler is not None:
            self.after_scheduler.step(epoch)

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


def _build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    params = [{"params": model.parameters()}]

    if config.optimizer == OptimizerName.ADAMW:
        return torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            **config.optimizer_extra,
        )
    elif config.optimizer == OptimizerName.ADAM:
        return torch.optim.Adam(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            **config.optimizer_extra,
        )
    elif config.optimizer == OptimizerName.SGD:
        return torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            **config.optimizer_extra,
        )
    elif config.optimizer == OptimizerName.RMSPROP:
        return torch.optim.RMSprop(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            **config.optimizer_extra,
        )
    else:
        return torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )


def _build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig, total_steps: int
):
    """Build LR scheduler from config."""
    base = None
    warmup_steps = config.warmup_steps or int(total_steps * config.warmup_ratio)

    if config.scheduler == SchedulerName.COSINE:
        base = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(total_steps, 1), eta_min=config.min_lr
        )
    elif config.scheduler == SchedulerName.ONE_CYCLE:
        base = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=max(total_steps, 1),
        )
    elif config.scheduler == SchedulerName.REDUCE_ON_PLATEAU:
        base = torch.optim.lr_scheduler.ReduceOnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    elif config.scheduler == SchedulerName.STEP:
        base = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(total_steps // 10, 1), gamma=0.1
        )
    elif config.scheduler == SchedulerName.POLYNOMIAL:
        base = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=max(total_steps, 1), power=2.0
        )

    if warmup_steps > 0 and base is not None:
        for pg in optimizer.param_groups:
            pg["initial_lr"] = config.learning_rate
        return GradualWarmupScheduler(optimizer, warmup_steps, after_scheduler=base)

    return base


class TrainingEngine:
    """Production-grade training engine with AMP, EMA, distributed, and comprehensive callbacks."""

    def __init__(
        self,
        model: nn.Module,
        spec: NeuralForgeSpec,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.spec = spec
        self.config = config or spec.training or TrainingConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        # Seeds
        self._set_seed(self.config.seed)

        # Optimizer
        self.optimizer = _build_optimizer(self.model, self.config)

        # Scaler
        self.use_amp = self.config.precision in (
            Precision.FP16, Precision.BF16, Precision.MIXED
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(self.use_amp and self.device.type == "cuda"
                     and self.config.precision == Precision.FP16),
        )
        self.amp_dtype = (
            torch.bfloat16 if self.config.precision == Precision.BF16
            else torch.float16
        )

        # State
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "lr": [],
        }
        self.best_metric = float("inf")
        self.best_epoch = 0
        self.best_state: Optional[Dict] = None
        self.ema: Optional[ExponentialMovingAverage] = None
        self._callbacks: List[Callable] = []

    def _set_seed(self, seed: int):
        """Set all seeds for reproducibility."""
        import random
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if self.config.deterministic_algorithms:
            torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def add_callback(self, callback: Callable):
        """Add a training callback: callback(epoch, model, metrics) -> None."""
        self._callbacks.append(callback)

    def _run_callbacks(self, epoch: int, metrics: Dict[str, float]):
        for cb in self._callbacks:
            try:
                cb(epoch, self.model, metrics)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: Optional[int] = None,
    ) -> TrainingResult:
        """Full training loop."""
        cfg = self.config
        num_epochs = epochs or cfg.epochs

        # Setup scheduler (needs total_steps)
        estimator_batches = len(train_loader) or 1
        total_steps = num_epochs * estimator_batches // cfg.gradient_accumulation_steps
        scheduler = _build_scheduler(self.optimizer, cfg, total_steps)

        # EMA
        if cfg.use_ema:
            self.ema = ExponentialMovingAverage(self.model, decay=cfg.ema_decay)

        # Early stopping
        early_stop = EarlyStopping(
            patience=cfg.early_stopping_patience, min_delta=1e-4, mode="min"
        )

        # Compile model if requested
        if cfg.compile_model and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        start_time = time.time()
        checkpoints: List[str] = []
        total_steps_completed = 0
        status = "completed"

        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self._train_epoch(train_loader, scheduler)
            self.history["train_loss"].append(train_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)

            # Validate
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.history["val_loss"].append(val_loss)

            metric = val_loss if val_loss is not None else train_loss

            # Track best
            if metric < self.best_metric:
                self.best_metric = metric
                self.best_epoch = epoch
                self.best_state = {
                    k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in self.model.state_dict().items()
                }

            # Logging
            elapsed = time.time() - start_time
            log_msg = (
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}"
            )
            if val_loss is not None:
                log_msg += f" | Val Loss: {val_loss:.4f}"
            log_msg += f" | LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
            logger.info(log_msg)

            # Callbacks
            self._run_callbacks(epoch, {"train_loss": train_loss, "val_loss": val_loss})

            # Early stopping
            if early_stop(metric):
                logger.info(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(best: {self.best_metric:.4f} @ epoch {self.best_epoch})"
                )
                status = "early_stopped"
                break

        # Restore best weights
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        total_time = time.time() - start_time
        total_steps_completed = epoch * len(train_loader)

        return TrainingResult(
            model_name=self.spec.name,
            spec_hash=self.spec.config_hash(),
            epochs_completed=epoch,
            total_steps=total_steps_completed,
            final_loss=self.history["train_loss"][-1] if self.history["train_loss"] else 0.0,
            final_metrics={"val_loss": val_loss} if val_loss else {},
            best_metric=self.best_metric,
            best_epoch=self.best_epoch,
            training_time_seconds=total_time,
            checkpoints=checkpoints,
            history=self.history,
            status=status,
        )

    def _train_epoch(
        self, train_loader: DataLoader, scheduler
    ) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            inputs, targets = self._unpack_batch(batch)

            with torch.amp.autocast(
                str(self.device),
                enabled=self.use_amp,
                dtype=self.amp_dtype,
            ):
                outputs = self.model(inputs)
                loss = F.cross_entropy(
                    outputs, targets, label_smoothing=self.config.label_smoothing
                ) / self.config.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if scheduler is not None and not isinstance(
                    scheduler, GradualWarmupScheduler
                ):
                    scheduler.step()

            if scheduler is not None and isinstance(scheduler, GradualWarmupScheduler):
                scheduler.step()

            if self.ema is not None:
                self.ema.update(self.model)

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in val_loader:
            inputs, targets = self._unpack_batch(batch)
            with torch.amp.autocast(
                str(self.device),
                enabled=self.use_amp,
                dtype=self.amp_dtype,
            ):
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (list, tuple)):
            return batch[0].to(self.device), batch[1].to(self.device)
        b = batch.to(self.device)
        return b, b
