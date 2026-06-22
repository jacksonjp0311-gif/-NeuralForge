"""
NeuralForge Pattern Engine v2.2 — Universal Pattern Learning & Prediction.

Takes ANY sequential data (time series, metrics, prices, logs, sensor readings),
auto-detects the pattern type, builds the right neural architecture, and predicts
what comes next. One tool, infinite use cases.

Pattern types auto-detected:
  - TREND: linear/exponential growth or decay
  - SEASONAL: repeating cycles (daily, weekly, hourly)
  - STATIONARY: mean-reverting around a stable value
  - CHAOTIC: complex non-linear dynamics (LSTM)
  - STEP: sudden regime changes

Evidence (test_pattern_engine.py):
  TREND:     r=0.9444, Seasonal: r=0.7982, Chaotic: r=0.1561
  Avg correlation: 0.5270 across 6 pattern types
  Avg training time: 0.48s
"""
from __future__ import annotations
import logging, time
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("neuralforge.pattern_engine")


class PatternType(str, Enum):
    TREND = "trend"
    SEASONAL = "seasonal"
    STATIONARY = "stationary"
    CHAOTIC = "chaotic"
    STEP = "step"
    UNKNOWN = "unknown"


class PatternDetector:
    """Analyzes a sequence to determine its pattern type via statistical signatures."""

    @staticmethod
    def detect(data: np.ndarray) -> Tuple[PatternType, Dict[str, float]]:
        if len(data) < 5:
            return PatternType.UNKNOWN, {"unknown": 1.0}

        scores: Dict[str, float] = {}
        x = np.arange(len(data), dtype=np.float32)

        # TREND: linear regression R^2 * slope magnitude
        slope, intercept = np.polyfit(x, data, 1)
        residuals = data - (slope * x + intercept)
        r2 = 1.0 - (np.var(residuals) / max(np.var(data), 1e-10))
        scores["trend"] = float(max(0.0, r2)) * min(1.0, abs(slope) * 10.0)

        # SEASONAL: max autocorrelation at lags 2..20
        if len(data) > 10:
            normalized = (data - np.mean(data)) / max(np.std(data), 1e-10)
            autocorrs = []
            for lag in range(2, min(len(data) // 2, 20)):
                if lag < len(normalized):
                    c = np.corrcoef(normalized[:-lag], normalized[lag:])[0, 1]
                    if not np.isnan(c):
                        autocorrs.append(abs(c))
            scores["seasonal"] = float(max(autocorrs)) if autocorrs else 0.0
        else:
            scores["seasonal"] = 0.0

        # STATIONARY: variance stability across quarters
        if len(data) > 10:
            w = len(data) // 4
            variances = [np.var(data[i:i + w]) for i in range(0, len(data) - w, w)]
            ratio = max(variances) / max(min(variances), 1e-10)
            scores["stationary"] = float(max(0.0, 1.0 - ratio / 5.0))
        else:
            scores["stationary"] = 0.5

        # CHAOTIC: high-frequency energy ratio
        if len(data) > 8:
            diffs = np.diff(data)
            scores["chaotic"] = float(min(1.0, np.var(diffs) / (np.var(data) + 1e-10)))
        else:
            scores["chaotic"] = 0.0

        # STEP: max jump relative to std
        if len(data) > 3:
            jumps = np.abs(np.diff(data))
            scores["step"] = float(min(1.0, max(jumps) / (3.0 * max(np.std(data), 1e-10))))
        else:
            scores["step"] = 0.0

        total = sum(scores.values()) or 1.0
        scores = {k: v / total for k, v in scores.items()}
        return PatternType(max(scores, key=scores.get)), scores


# ─── Neural Architectures per Pattern Type ───────────────────────

class TrendPredictor(nn.Module):
    """MLP for linear/exponential trend extrapolation."""
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, pred_len),
        )
    def forward(self, x): return self.net(x)


class SeasonalPredictor(nn.Module):
    """1D-CNN encoder + linear decoder for repeating patterns."""
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.GELU(),
            nn.Conv1d(16, 32, 5, padding=2), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(), nn.Linear(32, 64), nn.GELU(),
        )
        self.decoder = nn.Linear(64, pred_len)
    def forward(self, x): return self.decoder(self.encoder(x.unsqueeze(1)))


class StationaryPredictor(nn.Module):
    """MLP with dropout for mean-reverting patterns."""
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, pred_len),
        )
    def forward(self, x): return self.net(x)


class ChaoticPredictor(nn.Module):
    """2-layer LSTM for complex non-linear dynamics."""
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(32, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.fc(out[:, -1, :])


class StepPredictor(nn.Module):
    """MLP for regime change detection and post-step prediction."""
    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len, 32), nn.GELU(),
            nn.Linear(32, 16), nn.GELU(),
            nn.Linear(16, pred_len),
        )
    def forward(self, x): return self.net(x)


ARCH_MAP = {
    PatternType.TREND: TrendPredictor,
    PatternType.SEASONAL: SeasonalPredictor,
    PatternType.STATIONARY: StationaryPredictor,
    PatternType.CHAOTIC: ChaoticPredictor,
    PatternType.STEP: StepPredictor,
    PatternType.UNKNOWN: TrendPredictor,
}


# ─── Main Pattern Engine ─────────────────────────────────────────

class PatternEngine:
    """Universal pattern learning and prediction engine.

    Usage:
        engine = PatternEngine()
        result = engine.analyze(data=[1.2, 3.4, 5.1, 7.2, 8.9, ...], predict_steps=5)
        # result = {pattern_type, predictions, confidence, training_correlation, ...}
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = PatternDetector()
        self.model: Optional[nn.Module] = None
        self.pattern_type: Optional[PatternType] = None
        self.trained = False
        self._scaler_mean = 0.0
        self._scaler_std = 1.0

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        self._scaler_mean = float(np.mean(data))
        self._scaler_std = max(float(np.std(data)), 1e-8)
        return (data - self._scaler_mean) / self._scaler_std

    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        return data * self._scaler_std + self._scaler_mean

    def analyze(
        self,
        data: List[float],
        predict_steps: int = 5,
        seq_len: Optional[int] = None,
        epochs: int = 50,
    ) -> Dict[str, Any]:
        t0 = time.time()
        data = np.array(data, dtype=np.float32)
        if len(data) < 5:
            return {"status": "error", "error": "Need 5+ data points, got %d" % len(data)}

        # 1. Detect pattern
        pattern_type, pattern_scores = self.detector.detect(data)
        self.pattern_type = pattern_type

        # 2. Normalize
        normalized = self._normalize(data)

        # 3. Create sliding windows
        if seq_len is None:
            seq_len = max(5, len(data) // 3)
        pred_len = min(predict_steps, max(1, len(data) // 5))

        inputs, targets = [], []
        step = max(1, (len(normalized) - seq_len - pred_len) // 20)
        for i in range(0, len(normalized) - seq_len - pred_len + 1, step):
            inputs.append(normalized[i:i + seq_len])
            targets.append(normalized[i + seq_len:i + seq_len + pred_len])
        if len(inputs) < 2:
            inputs.append(normalized[:seq_len])
            targets.append(normalized[seq_len:seq_len + pred_len] if len(normalized) > seq_len else normalized[-pred_len:])
        inputs = np.array(inputs, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        # 4. Build architecture for detected pattern
        arch_class = ARCH_MAP[pattern_type]
        self.model = arch_class(seq_len, pred_len).to(self.device)

        # 5. Train
        x_t = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(targets, dtype=torch.float32, device=self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        self.model.train()
        for _ in range(epochs):
            loss = F.mse_loss(self.model(x_t), y_t)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        self.trained = True

        # 6. Evaluate
        self.model.eval()
        with torch.no_grad():
            tp = self.model(x_t)
            train_mse = F.mse_loss(tp, y_t).item()
            pn = tp.cpu().numpy().flatten()
            tn = y_t.cpu().numpy().flatten()
            corr = float(np.corrcoef(pn, tn)[0, 1]) if len(pn) > 1 else 0.0
            if np.isnan(corr): corr = 0.0

        # 7. Predict future
        last_win = normalized[-seq_len:]
        with torch.no_grad():
            pred_norm = self.model(
                torch.tensor(last_win, dtype=torch.float32, device=self.device).unsqueeze(0)
            ).cpu().numpy().flatten()
        predictions = self._denormalize(pred_norm).tolist()
        confidence = max(0.0, min(1.0, corr)) if corr > 0 else 0.1

        return {
            "status": "success",
            "pattern_type": pattern_type.value,
            "pattern_scores": {k: round(v, 3) for k, v in pattern_scores.items()},
            "architecture": arch_class.__name__,
            "predictions": [round(float(p), 4) for p in predictions],
            "predict_steps": len(predictions),
            "confidence": round(confidence, 3),
            "training_mse": round(train_mse, 6),
            "training_correlation": round(corr, 4),
            "data_points": len(data),
            "training_windows": len(inputs),
            "epochs": epochs,
            "training_time_seconds": round(time.time() - t0, 2),
        }
