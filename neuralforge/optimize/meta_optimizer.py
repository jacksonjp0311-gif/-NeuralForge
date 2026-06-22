from __future__ import annotations
import json, logging, time, copy
from typing import Any, Dict, List, Optional
from neuralforge.spec import NeuralForgeSpec, TrainingResult
logger = logging.getLogger("neuralforge.optimize.meta")

class MetaOptimizer:
    def __init__(self, memory_path=None):
        self.memory_path = memory_path or "./neuralforge_meta.json"
        self.insights = []; self.generation = 0
    def critique(self, spec, result):
        critique = {"generation": self.generation, "spec_hash": spec.config_hash(), "timestamp": time.time(), "issues": [], "suggestions": [], "score": 0.5}
        train_losses = result.history.get("train_loss", [])
        if len(train_losses) >= 3 and train_losses[-1] > train_losses[0]:
            critique["issues"].append("Training loss increased"); critique["suggestions"].append("Reduce learning rate")
        val_losses = result.history.get("val_loss", [])
        if val_losses and train_losses and (val_losses[-1] - train_losses[-1]) > 0.5:
            critique["issues"].append("Potential overfitting"); critique["suggestions"].append("Increase dropout")
        self.insights.append(critique); self.generation += 1; return critique
    def propose_next_spec(self, current_spec, result):
        critique = self.critique(current_spec, result)
        new_spec = copy.deepcopy(current_spec)
        for s in critique["suggestions"]:
            if "learning rate" in s.lower() and "reduce" in s.lower(): new_spec.training.learning_rate *= 0.5
            if "dropout" in s.lower(): new_spec.architecture.dropout = min(0.5, (new_spec.architecture.dropout or 0.1) + 0.1)
        return new_spec
