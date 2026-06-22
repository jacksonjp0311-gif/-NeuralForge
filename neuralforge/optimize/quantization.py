from __future__ import annotations
import logging
import torch, torch.nn as nn
from neuralforge.spec import QuantizationConfig, QuantizationMethod
logger = logging.getLogger("neuralforge.optimize.quantization")

def quantize_model(model, config, calibration_data=None):
    logger.info(f"Quantizing: method={config.method.value}, bits={config.bits}")
    if config.method == QuantizationMethod.DYNAMIC_INT8:
        return torch.quantization.quantize_dynamic(model, {nn.Linear, nn.LSTM}, dtype=torch.qint8)
    logger.info("Quantization method requires additional packages. Returning original model.")
    return model
