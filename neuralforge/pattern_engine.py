"""
NeuralForge Pattern Engine v2.3 — Fixed & Evolved.

Fixes from v2.2:
  - Trend detection: requires BOTH high R² AND meaningful slope magnitude
  - Seasonal: only checked if NOT strongly trending (prevents false seasonal)
  - Stationary: uses mean stability + variance stability + mean reversion
  - Step: uses jump_ratio (max/median) instead of max/std
  - Chaotic: uses local divergence + high-frequency energy
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
    TREND = "trend"; SEASONAL = "seasonal"; STATIONARY = "stationary"
    CHAOTIC = "chaotic"; STEP = "step"; UNKNOWN = "unknown"


class PatternDetector:
    @staticmethod
    def detect(data: np.ndarray) -> Tuple[PatternType, Dict[str, float]]:
        if len(data) < 5:
            return PatternType.UNKNOWN, {"unknown": 1.0}
        n = len(data)
        scores: Dict[str, float] = {}
        x = np.arange(n, dtype=np.float32)

        # TREND: R² * slope significance
        slope, intercept = np.polyfit(x, data, 1)
        residuals = data - (slope * x + intercept)
        r2 = 1.0 - (np.var(residuals) / max(np.var(data), 1e-10))
        slope_sig = abs(slope) / (np.std(data) / n + 1e-10)
        scores["trend"] = float(max(0.0, r2)) * min(1.0, slope_sig * 2.0)

        # SEASONAL: only if NOT strongly trending
        if n > 10 and scores["trend"] < 0.5:
            norm = (data - np.mean(data)) / max(np.std(data), 1e-10)
            autocorrs = []
            for lag in range(2, min(n // 2, 25)):
                if lag < len(norm):
                    c = np.corrcoef(norm[:-lag], norm[lag:])[0, 1]
                    if not np.isnan(c): autocorrs.append(abs(c))
            strong = [c for c in autocorrs if c > 0.5]
            scores["seasonal"] = float(len(strong)) / max(len(autocorrs), 1)
        else:
            scores["seasonal"] = 0.0

        # STATIONARY: mean stability + var stability + mean reversion
        if n > 10:
            w = n // 4
            means = [np.mean(data[i:i+w]) for i in range(0, n-w, w)]
            variances = [np.var(data[i:i+w]) for i in range(0, n-w, w)]
            ms = 1.0 - min(1.0, np.std(means) / (np.std(data) + 1e-10))
            vs = 1.0 - min(1.0, max(variances) / (min(variances) + 1e-10) / 3.0)
            lr = max(0.0, -np.corrcoef(data[:-1], data[1:])[0, 1]) if n > 2 else 0.0
            scores["stationary"] = 0.4 * ms + 0.3 * vs + 0.3 * lr
        else:
            scores["stationary"] = 0.3

        # CHAOTIC: local divergence + HF energy
        if n > 15:
            diffs = np.abs(np.diff(data))
            ld = np.mean(diffs[1:] / (diffs[:-1] + 1e-10))
            hf = np.var(np.diff(data, 2)) / max(np.var(data), 1e-10)
            scores["chaotic"] = float(min(1.0, ld * 0.3 + hf * 0.7))
        else:
            scores["chaotic"] = 0.0

        # STEP: jump ratio
        if n > 5:
            diffs = np.abs(np.diff(data))
            jr = max(diffs) / (np.median(diffs) + 1e-10)
            scores["step"] = float(min(1.0, (jr - 1.0) / 5.0)) if jr > 2 else 0.0
        else:
            scores["step"] = 0.0

        total = sum(scores.values()) or 1.0
        scores = {k: v / total for k, v in scores.items()}
        return PatternType(max(scores, key=scores.get)), scores


class TrendPredictor(nn.Module):
    def __init__(self, s, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,p))
    def forward(self, x): return self.net(x)

class SeasonalPredictor(nn.Module):
    def __init__(self, s, p):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(1,16,3,padding=1),nn.GELU(),nn.Conv1d(16,32,5,padding=2),nn.GELU(),nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(32,64),nn.GELU())
        self.dec = nn.Linear(64, p)
    def forward(self, x): return self.dec(self.enc(x.unsqueeze(1)))

class StationaryPredictor(nn.Module):
    def __init__(self, s, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s,64),nn.GELU(),nn.Dropout(0.1),nn.Linear(64,32),nn.GELU(),nn.Linear(32,p))
    def forward(self, x): return self.net(x)

class ChaoticPredictor(nn.Module):
    def __init__(self, s, p):
        super().__init__()
        self.lstm = nn.LSTM(1,32,num_layers=2,batch_first=True,dropout=0.1)
        self.fc = nn.Linear(32, p)
    def forward(self, x):
        if len(x.shape)==2: x=x.unsqueeze(-1)
        out,_ = self.lstm(x); return self.fc(out[:,-1,:])

class StepPredictor(nn.Module):
    def __init__(self, s, p):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(s,32),nn.GELU(),nn.Linear(32,16),nn.GELU(),nn.Linear(16,p))
    def forward(self, x): return self.net(x)

ARCH_MAP = {PatternType.TREND:TrendPredictor,PatternType.SEASONAL:SeasonalPredictor,PatternType.STATIONARY:StationaryPredictor,PatternType.CHAOTIC:ChaoticPredictor,PatternType.STEP:StepPredictor,PatternType.UNKNOWN:TrendPredictor}


class PatternEngine:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = PatternDetector()
        self.model = None; self.pattern_type = None; self.trained = False
        self._scaler_mean = 0.0; self._scaler_std = 1.0

    def _normalize(self, d):
        self._scaler_mean = float(np.mean(d)); self._scaler_std = max(float(np.std(d)), 1e-8)
        return (d - self._scaler_mean) / self._scaler_std
    def _denormalize(self, d): return d * self._scaler_std + self._scaler_mean

    def analyze(self, data, predict_steps=5, seq_len=None, epochs=50):
        t0 = time.time()
        data = np.array(data, dtype=np.float32)
        if len(data) < 5: return {"status":"error","error":"Need 5+ points, got %d"%len(data)}
        pt, ps = self.detector.detect(data); self.pattern_type = pt
        norm = self._normalize(data)
        if seq_len is None: seq_len = max(5, len(data)//3)
        pl = min(predict_steps, max(1, len(data)//5))
        inp, tgt = [], []
        step = max(1, (len(norm)-seq_len-pl)//20)
        for i in range(0, len(norm)-seq_len-pl+1, step):
            inp.append(norm[i:i+seq_len]); tgt.append(norm[i+seq_len:i+seq_len+pl])
        if len(inp)<2: inp.append(norm[:seq_len]); tgt.append(norm[seq_len:seq_len+pl] if len(norm)>seq_len else norm[-pl:])
        inp = np.array(inp, dtype=np.float32); tgt = np.array(tgt, dtype=np.float32)
        ac = ARCH_MAP[pt]; self.model = ac(seq_len, pl).to(self.device)
        xt = torch.tensor(inp, dtype=torch.float32, device=self.device)
        yt = torch.tensor(tgt, dtype=torch.float32, device=self.device)
        op = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        sc = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=epochs)
        self.model.train()
        for _ in range(epochs):
            loss = F.mse_loss(self.model(xt), yt)
            op.zero_grad(); loss.backward(); op.step(); sc.step()
        self.trained = True
        self.model.eval()
        with torch.no_grad():
            tp = self.model(xt)
            mse = F.mse_loss(tp, yt).item()
            pn = tp.cpu().numpy().flatten(); tn = yt.cpu().numpy().flatten()
            cr = float(np.corrcoef(pn,tn)[0,1]) if len(pn)>1 else 0.0
            if np.isnan(cr): cr = 0.0
        lw = norm[-seq_len:]
        with torch.no_grad():
            pn2 = self.model(torch.tensor(lw, dtype=torch.float32, device=self.device).unsqueeze(0)).cpu().numpy().flatten()
        preds = self._denormalize(pn2).tolist()
        conf = max(0.0, min(1.0, cr)) if cr > 0 else 0.1
        return {"status":"success","pattern_type":pt.value,"pattern_scores":{k:round(v,3) for k,v in ps.items()},"architecture":ac.__name__,"predictions":[round(float(p),4) for p in preds],"predict_steps":len(preds),"confidence":round(conf,3),"training_mse":round(mse,6),"training_correlation":round(cr,4),"data_points":len(data),"training_windows":len(inp),"epochs":epochs,"training_time_seconds":round(time.time()-t0,2)}
