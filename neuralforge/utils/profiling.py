from __future__ import annotations
import logging, time
from typing import Dict, Optional, Tuple
import torch, torch.nn as nn
logger = logging.getLogger("neuralforge.utils.profiling")

def profile_model(model, input_shape, device=None, num_warmup=10, num_iterations=100):
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval(); dummy = torch.randn(*input_shape).to(device)
    total_params = sum(p.numel() for p in model.parameters()); trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / 1024 / 1024
    gpu_mem = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(): _ = model(dummy)
        gpu_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
    latencies = []
    with torch.no_grad():
        for _ in range(num_warmup): _ = model(dummy)
        if device.type == "cuda": torch.cuda.synchronize()
        for _ in range(num_iterations):
            start = time.perf_counter(); _ = model(dummy)
            if device.type == "cuda": torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
    import numpy as np; lat = np.array(latencies)
    return {"total_parameters": total_params, "trainable_parameters": trainable, "model_size_mb": round(model_size_mb, 2),
            "gpu_memory_mb": round(gpu_mem, 2), "latency_mean_ms": round(float(lat.mean()), 2),
            "latency_p95_ms": round(float(np.percentile(lat, 95)), 2),
            "throughput_samples_per_sec": round(1000.0/float(lat.mean())*input_shape[0], 1), "device": str(device)}
