from __future__ import annotations
import logging
import torch, torch.nn as nn, torch.nn.utils.prune as prune
from neuralforge.spec import PruningConfig
logger = logging.getLogger("neuralforge.optimize.pruning")

def prune_model(model, config, val_fn=None):
    logger.info(f"Pruning: method={config.method.value}, amount={config.amount}")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=config.amount)
            try: prune.remove(module, "weight")
            except ValueError: pass
    total = sum(p.numel() for p in model.parameters())
    zeros = sum((p.data == 0).sum().item() for p in model.parameters())
    logger.info(f"Sparsity: {zeros/total:.2%}"); return model
