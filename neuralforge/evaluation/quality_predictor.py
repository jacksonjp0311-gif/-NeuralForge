"""
Model Quality Predictor v2 — Multi-Objective Neural Quality Prediction.
Evolved from r=0.757 synthetic to real-data validation + multi-objective.
"""
from __future__ import annotations
import logging, time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("neuralforge.evaluation.quality_predictor")


class TrainingDynamicsEncoder(nn.Module):
    def __init__(self, in_channels=3, seq_len=50, hidden_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        return self.drop(self.norm(self.fc(x)))


class MultiObjectivePredictorNet(nn.Module):
    def __init__(self, seq_len=50, hidden_dim=64, num_tasks=14, num_families=16):
        super().__init__()
        self.dynamics_encoder = TrainingDynamicsEncoder(3, seq_len, hidden_dim)
        self.task_embed = nn.Embedding(num_tasks, 8)
        self.family_embed = nn.Embedding(num_families, 8)
        combined = hidden_dim + 4 + 8 + 8
        self.backbone = nn.Sequential(
            nn.Linear(combined, 128), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.GELU(),
        )
        self.accuracy_head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
        self.latency_head  = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
        self.memory_head   = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, dynamics, arch_meta, task_idx, family_idx):
        d = self.dynamics_encoder(dynamics)
        t = self.task_embed(task_idx)
        f = self.family_embed(family_idx)
        shared = self.backbone(torch.cat([d, arch_meta, t, f], dim=-1))
        return {"accuracy": self.accuracy_head(shared).squeeze(-1),
                "latency":  self.latency_head(shared).squeeze(-1),
                "memory":   self.memory_head(shared).squeeze(-1)}


class ModelQualityPredictor:
    TASK_MAP = {
        "image_classification": 0, "text_classification": 1, "text_generation": 2,
        "object_detection": 3, "image_segmentation": 4, "regression": 5,
        "time_series": 6, "anomaly_detection": 7, "rl_policy": 8,
        "multimodal": 9, "audio_classification": 10, "seq2seq": 11,
        "speech_recognition": 12, "graph_prediction": 13,
    }
    FAMILY_MAP = {
        "cnn": 0, "resnet": 1, "transformer": 2, "vit": 3,
        "mlp_mixer": 4, "kan": 5, "mamba": 6, "ssm": 7,
        "rwkv": 8, "liquid": 9, "diffusion": 10, "flow_matching": 11,
        "graph_nn": 12, "moe": 13, "retnet": 14, "custom": 15,
    }

    def __init__(self, seq_len=50, hidden_dim=64, device=None, use_multi_objective=True):
        self.seq_len = seq_len
        self.use_multi_objective = use_multi_objective
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if use_multi_objective:
            self.model = MultiObjectivePredictorNet(seq_len, hidden_dim).to(self.device)
        else:
            self.model = self._build_single(seq_len, hidden_dim)
        self.is_trained = False
        self.version = "2.0"

    def _build_single(self, seq_len, hidden_dim):
        class SO(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = TrainingDynamicsEncoder(3, seq_len, hidden_dim)
                self.te = nn.Embedding(14, 8)
                self.fe = nn.Embedding(16, 8)
                self.head = nn.Sequential(
                    nn.Linear(hidden_dim+4+8+8, 64), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
            def forward(self, d, a, t, f):
                return self.head(torch.cat([self.enc(d), a, self.te(t), self.fe(f)], dim=-1)).squeeze(-1)
        return SO().to(self.device)

    def _pad_sequence(self, values):
        arr = np.array(values, dtype=np.float32)
        if len(arr) >= self.seq_len:
            idx = np.linspace(0, len(arr)-1, self.seq_len, dtype=int)
            return arr[idx]
        p = np.full(self.seq_len, arr[-1] if len(arr) > 0 else 0.0, dtype=np.float32)
        p[:len(arr)] = arr
        return p

    def _prepare_dynamics(self, history):
        tl = self._pad_sequence(history.get("train_loss", []))
        vl = self._pad_sequence(history.get("val_loss", history.get("train_loss", [])))
        lr = self._pad_sequence(history.get("lr", [0.001]))
        ml = max(tl.max(), vl.max(), 1.0)
        tl, vl = tl/ml, vl/ml
        lr = lr/(lr.max()+1e-8)
        t = np.stack([tl, vl, lr], axis=-1)
        return torch.tensor(t, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _prepare_arch_meta(self, num_params, depth, width, num_classes):
        m = np.array([np.log10(max(num_params,1)), float(depth or 4), float(width or 64), float(num_classes or 10)], dtype=np.float32)
        return torch.tensor(m, dtype=torch.float32).unsqueeze(0).to(self.device)

    def train_on_histories(self, histories, targets_acc, targets_lat, targets_mem,
                            arch_metas, task_types, family_types, epochs=150, lr=1e-3):
        """Train multi-objective predictor on collected training histories."""
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for ep in range(epochs):
            self.model.train()
            for i, hist in enumerate(histories):
                d = self._prepare_dynamics(hist)
                a = self._prepare_arch_meta(**arch_metas[i])
                t = torch.tensor([self.TASK_MAP.get(task_types[i],0)], dtype=torch.long, device=self.device)
                f = torch.tensor([self.FAMILY_MAP.get(family_types[i],0)], dtype=torch.long, device=self.device)
                p = self.model(d, a, t, f)
                loss = (F.mse_loss(p["accuracy"], torch.tensor([targets_acc[i]], device=self.device, dtype=torch.float32)) +
                        F.mse_loss(p["latency"],  torch.tensor([targets_lat[i]],  device=self.device, dtype=torch.float32)) +
                        F.mse_loss(p["memory"],   torch.tensor([targets_mem[i]],  device=self.device, dtype=torch.float32))) / 3.0
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
        self.is_trained = True
        return self._compute_correlations(histories, targets_acc, targets_lat, targets_mem, arch_metas, task_types, family_types)

    def _compute_correlations(self, histories, ta, tl, tm, am, tt, ft):
        self.model.eval()
        pa, pl, pm = [], [], []
        with torch.no_grad():
            for i, hist in enumerate(histories):
                d = self._prepare_dynamics(hist)
                a = self._prepare_arch_meta(**am[i])
                t = torch.tensor([self.TASK_MAP.get(tt[i],0)], dtype=torch.long, device=self.device)
                f = torch.tensor([self.FAMILY_MAP.get(ft[i],0)], dtype=torch.long, device=self.device)
                p = self.model(d, a, t, f)
                pa.append(float(p["accuracy"].item())); pl.append(float(p["latency"].item())); pm.append(float(p["memory"].item()))
        def c(x,y): return float(np.corrcoef(x,y)[0,1]) if len(x)>1 else 0.0
        return {"accuracy_corr": c(pa,ta), "latency_corr": c(pl,tl), "memory_corr": c(pm,tm), "num_histories": len(histories)}

    def predict(self, history, num_params=1_000_000, depth=4, width=64, num_classes=10,
                task_type="image_classification", family_type="resnet"):
        self.model.eval()
        with torch.no_grad():
            d = self._prepare_dynamics(history)
            a = self._prepare_arch_meta(num_params, depth, width, num_classes)
            t = torch.tensor([self.TASK_MAP.get(task_type,0)], dtype=torch.long, device=self.device)
            f = torch.tensor([self.FAMILY_MAP.get(family_type,0)], dtype=torch.long, device=self.device)
            p = self.model(d, a, t, f)
        ne = len(history.get("train_loss", []))
        return {"quality_score": round(p["accuracy"].item(),4), "predicted_latency_score": round(p["latency"].item(),4),
                "predicted_memory_score": round(p["memory"].item(),4), "confidence": round(min(1.0, ne/10.0),2),
                "is_trained": self.is_trained, "epochs_observed": ne, "version": self.version}

    def compare_architectures(self, histories, arch_metadata, task_type="image_classification"):
        results = []
        for name, hist in histories.items():
            meta = arch_metadata.get(name, {})
            pred = self.predict(history=hist, task_type=task_type, **meta)
            results.append({"name": name, **pred, "arch": meta})
        results.sort(key=lambda x: x["quality_score"], reverse=True)
        return results
