"""
NeuralForge Learner v2.3 — Fixed & Evolved.

Fixes from v2.2:
  - Anomaly detection now uses Autoencoder (AnomalyNet) instead of ClassificationNet
  - ProblemDetector: anomaly check happens BEFORE classification for binary imbalanced data
  - Added min_samples check to prevent tiny dataset crashes
  - Classification n_classes cast to int properly
"""
from __future__ import annotations
import logging, time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("neuralforge.learner")


class ProblemType:
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    FORECASTING = "forecasting"
    ANOMALY = "anomaly"


class ProblemDetector:
    @staticmethod
    def detect(X: np.ndarray, y: np.ndarray) -> Tuple[str, Dict[str, float]]:
        n_samples = len(X)
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        unique_targets = len(np.unique(y))
        scores = {}

        # ANOMALY: check FIRST — binary with high class imbalance
        is_binary = unique_targets == 2
        if is_binary:
            counts = np.bincount(y.astype(int))
            imbalance_ratio = min(counts) / max(counts)
            if imbalance_ratio < 0.15:  # less than 15% minority class
                scores[ProblemType.ANOMALY] = 0.85
                scores[ProblemType.CLASSIFICATION] = 0.1
                scores[ProblemType.REGRESSION] = 0.05
            else:
                scores[ProblemType.ANOMALY] = 0.05
                scores[ProblemType.CLASSIFICATION] = 0.85
                scores[ProblemType.REGRESSION] = 0.1
        else:
            scores[ProblemType.ANOMALY] = 0.05
            # Forecasting: single feature, sequential, continuous values
            if n_features <= 2 and n_samples > 20:
                # Check if values are continuous (not integer-like)
                y_rounded = np.round(y, 0)
                is_integer_like = np.allclose(y, y_rounded, atol=0.05)
                unique_ratio = unique_targets / n_samples
                # Check sequentiality: uniform diffs between sorted unique values
                y_sorted = np.sort(np.unique(y))
                if len(y_sorted) > 3:
                    diffs = np.diff(y_sorted)
                    diff_cv = np.std(diffs) / (np.mean(np.abs(diffs)) + 1e-10)
                    is_sequential = diff_cv < 2.0
                else:
                    is_sequential = False
                # Forecasting = continuous + sequential + single feature
                if not is_integer_like and is_sequential:
                    scores[ProblemType.FORECASTING] = 0.85
                    scores[ProblemType.REGRESSION] = 0.1
                    scores[ProblemType.CLASSIFICATION] = 0.05
                elif unique_ratio > 0.3 and not is_integer_like:
                    scores[ProblemType.REGRESSION] = 0.8
                    scores[ProblemType.CLASSIFICATION] = 0.05
                    scores[ProblemType.FORECASTING] = 0.15
                elif unique_targets <= max(20, n_samples * 0.05):
                    scores[ProblemType.CLASSIFICATION] = 0.85
                    scores[ProblemType.REGRESSION] = 0.1
                    scores[ProblemType.FORECASTING] = 0.05
                else:
                    scores[ProblemType.REGRESSION] = 0.85
                    scores[ProblemType.CLASSIFICATION] = 0.05
                    scores[ProblemType.FORECASTING] = 0.1
            else:
                if unique_targets <= max(20, n_samples * 0.05) and unique_targets >= 2:
                    scores[ProblemType.CLASSIFICATION] = 0.9
                    scores[ProblemType.REGRESSION] = 0.1
                else:
                    scores[ProblemType.CLASSIFICATION] = 0.05
                    scores[ProblemType.REGRESSION] = 0.85
                scores[ProblemType.FORECASTING] = 0.05

        total = sum(scores.values()) or 1.0
        scores = {k: v / total for k, v in scores.items()}
        return max(scores, key=scores.get), scores


class RegressionNet(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.BatchNorm1d(128), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(), nn.BatchNorm1d(64),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, out_dim),
        )
    def forward(self, x): return self.net(x)


class ClassificationNet(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.BatchNorm1d(128), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.GELU(), nn.BatchNorm1d(64),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, n_classes),
        )
    def forward(self, x): return self.net(x)


class ForecastNet(nn.Module):
    def __init__(self, seq_len, pred_len=1):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(32, pred_len)
    def forward(self, x):
        if len(x.shape) == 2: x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class AnomalyNet(nn.Module):
    """Autoencoder: compress → reconstruct. High reconstruction error = anomaly."""
    def __init__(self, in_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 16), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.GELU(),
            nn.Linear(32, 64), nn.GELU(),
            nn.Linear(64, in_dim),
        )
    def forward(self, x): return self.decoder(self.encoder(x))


ARCH_MAP = {
    ProblemType.REGRESSION: RegressionNet,
    ProblemType.CLASSIFICATION: ClassificationNet,
    ProblemType.FORECASTING: ForecastNet,
    ProblemType.ANOMALY: AnomalyNet,
}


class DataLearner:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = ProblemDetector()
        self.model = None
        self.problem_type = None
        self.trained = False
        self._scaler_mean = None
        self._scaler_std = None

    def _normalize(self, X):
        self._scaler_mean = np.mean(X, axis=0)
        self._scaler_std = np.std(X, axis=0) + 1e-8
        return (X - self._scaler_mean) / self._scaler_std

    def learn(self, X, y, epochs=50, batch_size=32, verbose=False):
        t0 = time.time()
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        if len(X) < 5:
            return {"status": "error", "error": "Need 5+ samples, got %d" % len(X)}
        if len(X) != len(y):
            return {"status": "error", "error": "X/y length mismatch: %d vs %d" % (len(X), len(y))}

        problem_type, problem_scores = self.detector.detect(X, y)
        self.problem_type = problem_type

        if problem_type != ProblemType.FORECASTING:
            X_norm = self._normalize(X)
        else:
            X_norm = X

        n_features = X.shape[1] if len(X.shape) > 1 else 1
        arch_class = ARCH_MAP[problem_type]

        if problem_type == ProblemType.CLASSIFICATION:
            n_cls = int(len(np.unique(y)))
            self.model = arch_class(n_features, n_cls).to(self.device)
        elif problem_type == ProblemType.FORECASTING:
            seq_len = min(n_features, 20)
            self.model = arch_class(seq_len, 1).to(self.device)
        elif problem_type == ProblemType.ANOMALY:
            self.model = arch_class(n_features).to(self.device)
        else:
            self.model = arch_class(n_features, 1).to(self.device)

        X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=self.device)
        if problem_type == ProblemType.CLASSIFICATION:
            y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
            if len(y_tensor.shape) == 1:
                y_tensor = y_tensor.unsqueeze(1)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=min(batch_size, len(X)), shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        self.model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                pred = self.model(xb)
                if problem_type == ProblemType.CLASSIFICATION:
                    loss = F.cross_entropy(pred, yb)
                elif problem_type == ProblemType.ANOMALY:
                    loss = F.mse_loss(pred, xb)
                else:
                    loss = F.mse_loss(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()

        self.trained = True

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor)

            if problem_type == ProblemType.CLASSIFICATION:
                pred_labels = preds.argmax(dim=1)
                accuracy = (pred_labels == y_tensor).float().mean().item()
                metric_name = "accuracy"
                metric_value = round(accuracy, 4)
            elif problem_type == ProblemType.ANOMALY:
                recon_error = F.mse_loss(preds, X_tensor, reduction='none').mean(dim=1)
                threshold = recon_error.mean() + 2 * recon_error.std()
                anomalies = int((recon_error > threshold).sum().item())
                metric_name = "anomalies_detected"
                metric_value = anomalies
            else:
                pred_np = preds.cpu().numpy().flatten()
                target_np = y_tensor.cpu().numpy().flatten()
                ss_res = np.sum((target_np - pred_np) ** 2)
                ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
                r2 = 1.0 - ss_res / max(ss_tot, 1e-10)
                metric_name = "r_squared"
                metric_value = round(float(r2), 4)

        elapsed = time.time() - t0

        return {
            "status": "success", "problem_type": problem_type,
            "problem_scores": {k: round(v, 3) for k, v in problem_scores.items()},
            "architecture": arch_class.__name__,
            "metric_name": metric_name, "metric_value": metric_value,
            "n_samples": len(X), "n_features": n_features,
            "epochs": epochs, "training_time_seconds": round(elapsed, 2),
        }
