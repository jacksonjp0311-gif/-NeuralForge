from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
logger = logging.getLogger("neuralforge.auto.scaling")

class ScalingLawEstimator:
    def __init__(self): self.fitted_params = None; self.observations = []
    def add_observation(self, num_params, dataset_size, loss, metric_value=None):
        self.observations.append({"num_params": num_params, "dataset_size": dataset_size, "loss": loss, "metric_value": metric_value})
    def fit(self):
        if self.fitted_params: return self.fitted_params
        if len(self.observations) < 3: self.fitted_params = {"E": 1.0, "A": 1.0, "alpha": 0.5, "B": 1.0, "beta": 0.5}; return self.fitted_params
        losses = np.array([o["loss"] for o in self.observations]); params = np.array([o["num_params"] for o in self.observations], dtype=float)
        E = losses.min() * 0.9; log_n = np.log(params); log_l = np.log(np.maximum(losses - E, 1e-10))
        if len(set(log_n)) > 1: alpha = -np.polyfit(log_n, log_l, 1)[0]; A = np.exp(np.polyfit(log_n, log_l, 1)[1])
        else: alpha = 0.5; A = 1.0
        self.fitted_params = {"E": float(E), "A": float(A), "alpha": float(alpha), "B": 1.0, "beta": 0.5}
        return self.fitted_params
    def predict_loss(self, num_params, dataset_size):
        if self.fitted_params is None: self.fit()
        p = self.fitted_params; return p["E"] + p["A"] / (num_params ** p["alpha"]) + p["B"] / (dataset_size ** p["beta"])
