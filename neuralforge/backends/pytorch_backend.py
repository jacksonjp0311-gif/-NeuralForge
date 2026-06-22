from __future__ import annotations
import logging, time
import torch, torch.nn as nn
logger = logging.getLogger("neuralforge.backends.pytorch")

class PyTorchBackend:
    def __init__(self, device=None, precision="mixed", distributed=False, compile_model=False):
        self.precision = precision; self.distributed = distributed; self.compile_model = compile_model
        if device: self._device = torch.device(device)
        elif torch.cuda.is_available(): self._device = torch.device("cuda:0")
        else: self._device = torch.device("cpu")
    @property
    def device(self): return self._device
    def to_device(self, model): return model.to(self._device)
    def get_memory_info(self):
        if self._device.type != "cuda": return {"device": str(self._device), "gpu": False}
        return {"device": str(self._device), "gpu": True, "name": torch.cuda.get_device_name(0),
                "total_memory_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2)}
