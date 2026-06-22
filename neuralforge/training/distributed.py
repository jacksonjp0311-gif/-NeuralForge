from __future__ import annotations
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from neuralforge.spec import DistributedStrategy

logger = logging.getLogger("neuralforge.training.distributed")


def setup_distributed(
    strategy: DistributedStrategy,
    backend: str = "nccl",
) -> dict:
    """Setup distributed training configuration."""
    config = {
        "strategy": strategy,
        "backend": backend,
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "rank": int(os.environ.get("RANK", 0)),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
    }

    if strategy == DistributedStrategy.NONE:
        return config

    if strategy == DistributedStrategy.DDP:
        logger.info(f"Setting up DDP: world_size={config['world_size']}")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend)
        return config

    if strategy == DistributedStrategy.FSDP:
        logger.info("Setting up FSDP")
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend)
        return config

    if strategy == DistributedStrategy.DEEPSPEED:
        logger.info("Setting up DeepSpeed")
        try:
            import deepspeed
            config["deepspeed_available"] = True
        except ImportError:
            logger.warning("deepspeed not installed, falling back to DDP")
            config["strategy"] = DistributedStrategy.DDP
        return config

    return config


def wrap_distributed_model(
    model: nn.Module,
    config: dict,
) -> nn.Module:
    """Wrap model for distributed training."""
    strategy = config["strategy"]

    if strategy == DistributedStrategy.NONE:
        return model

    device = torch.device(f"cuda:{config['local_rank']}")
    model = model.to(device)

    if strategy == DistributedStrategy.DDP:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[config["local_rank"]])
    elif strategy == DistributedStrategy.FSDP:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        model = FSDP(model)

    return model


def make_distributed_loader(
    dataset,
    batch_size: int,
    config: dict,
    **loader_kwargs,
) -> DataLoader:
    """Create a DataLoader with optional distributed sampler."""
    sampler = None
    if config["strategy"] != DistributedStrategy.NONE and config.get("world_size", 1) > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=config.get("world_size", 1),
            rank=config.get("rank", 0),
            shuffle=True,
        )
        loader_kwargs["shuffle"] = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=loader_kwargs.pop("num_workers", 4),
        pin_memory=loader_kwargs.pop("pin_memory", True),
        **loader_kwargs,
    )
