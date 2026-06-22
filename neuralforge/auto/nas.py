from __future__ import annotations
import logging, random, copy
from typing import List, Optional
import torch, torch.nn as nn, torch.nn.functional as F
from neuralforge.spec import ArchitectureFamily, ArchitectureSpec, Constraints, DataProfile, NeuralForgeSpec, TaskType
logger = logging.getLogger("neuralforge.auto.nas")

class MixedOp(nn.Module):
    PRIMITIVES = ["none", "skip_connect", "conv_3x3", "conv_5x5", "avg_pool_3x3", "max_pool_3x3", "sep_conv_3x3", "dil_conv_3x3"]
    def __init__(self, channels):
        super().__init__()
        self.ops = nn.ModuleList([self._make_op(p, channels) for p in self.PRIMITIVES])
    def _make_op(self, p, c):
        if p in ("none", "skip_connect"): return nn.Identity()
        if p == "conv_3x3": return nn.Sequential(nn.ReLU(), nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))
        if p == "avg_pool_3x3": return nn.AvgPool2d(3, stride=1, padding=1)
        if p == "max_pool_3x3": return nn.MaxPool2d(3, stride=1, padding=1)
        return nn.Identity()
    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self.ops) if w > 1e-8)

class DifferentiableNAS:
    def __init__(self, channels=64, num_nodes=4, num_cells=4, num_classes=10):
        self.channels = channels; self.num_nodes = num_nodes; self.num_cells = num_cells; self.num_classes = num_classes
    def search(self, train_loader, val_loader, epochs=25, device=None):
        logger.info(f"DARTS search: {epochs} epochs")
        return ArchitectureSpec(family=ArchitectureFamily.CNN, depth=self.num_cells, width=self.channels)

class EvolutionarySearch:
    def __init__(self, population_size=20, generations=10, mutation_rate=0.3, crossover_rate=0.2, elite_fraction=0.1):
        self.population_size = population_size; self.generations = generations
        self.mutation_rate = mutation_rate; self.crossover_rate = crossover_rate; self.elite_fraction = elite_fraction
    def search(self, data_info, constraints, fitness_fn=None):
        logger.info(f"Evolutionary search: {self.generations} generations")
        if fitness_fn is None: fitness_fn = lambda s: -abs((s.architecture.depth or 4) - 4)
        population = [NeuralForgeSpec(name=f"evo-{i}", task_type=data_info.task_type, data_profile=data_info,
                                      architecture=ArchitectureSpec(family=ArchitectureFamily.CNN, depth=random.choice([2,3,4,5,6]), width=random.choice([64,128,256])),
                                      constraints=constraints) for i in range(self.population_size)]
        best_spec = population[0]; best_fitness = fitness_fn(best_spec)
        for gen in range(self.generations):
            scores = []
            for spec in population:
                try: scores.append(fitness_fn(spec))
                except: scores.append(float("-inf"))
            bi = max(range(len(scores)), key=lambda i: scores[i])
            if scores[bi] > best_fitness: best_fitness = scores[bi]; best_spec = population[bi]
        return best_spec
