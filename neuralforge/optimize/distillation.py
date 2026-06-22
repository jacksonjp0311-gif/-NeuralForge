from __future__ import annotations
import logging
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from neuralforge.spec import DistillationConfig
logger = logging.getLogger("neuralforge.optimize.distillation")

def distill_model(teacher, student, config, train_loader, val_loader=None, epochs=10, device=None):
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = teacher.to(device).eval(); student = student.to(device).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        student.train(); total_loss = 0.0; nb = 0
        for batch in train_loader:
            inputs, targets = (batch[0].to(device), batch[1].to(device)) if isinstance(batch, (list, tuple)) else (batch.to(device), batch.to(device))
            with torch.no_grad(): t_logits = teacher(inputs)
            s_logits = student(inputs)
            soft_s = F.log_softmax(s_logits / config.temperature, dim=1)
            soft_t = F.softmax(t_logits / config.temperature, dim=1)
            d_loss = F.kl_div(soft_s, soft_t, reduction="batchmean") * (config.temperature ** 2)
            h_loss = F.cross_entropy(s_logits, targets)
            loss = config.alpha * d_loss + (1 - config.alpha) * h_loss
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); nb += 1
    return student
