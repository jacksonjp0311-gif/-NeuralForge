from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from neuralforge.spec import ExportConfig, ExportFormat

logger = logging.getLogger("neuralforge.utils.export")


def export_model(
    model: nn.Module,
    config: ExportConfig,
    sample_input: Optional[torch.Tensor] = None,
) -> str:
    """Export a model to the specified format."""
    export_dir = Path(config.output_path)
    export_dir.mkdir(parents=True, exist_ok=True)

    fmt = config.format

    if fmt == ExportFormat.PYTORCH_STATE_DICT:
        return _export_state_dict(model, export_dir)

    elif fmt == ExportFormat.SAFETENSORS:
        return _export_safetensors(model, export_dir)

    elif fmt == ExportFormat.TORCHSCRIPT:
        return _export_torchscript(model, export_dir, sample_input)

    elif fmt == ExportFormat.ONNX:
        return _export_onnx(model, export_dir, sample_input, config)

    else:
        logger.warning(f"Export format {fmt} not fully implemented, falling back to state_dict")
        return _export_state_dict(model, export_dir)


def _export_state_dict(model: nn.Module, export_dir: Path) -> str:
    path = export_dir / "model.pt"
    torch.save(model.state_dict(), str(path))
    logger.info(f"Exported state_dict to {path}")
    return str(path)


def _export_safetensors(model: nn.Module, export_dir: Path) -> str:
    try:
        from safetensors.torch import save_file
        path = export_dir / "model.safetensors"
        save_file(model.state_dict(), str(path))
        logger.info(f"Exported safetensors to {path}")
        return str(path)
    except ImportError:
        logger.warning("safetensors not installed, falling back to state_dict")
        return _export_state_dict(model, export_dir)


def _export_torchscript(
    model: nn.Module, export_dir: Path, sample_input: Optional[torch.Tensor]
) -> str:
    if sample_input is None:
        logger.warning("TorchScript export requires sample_input, falling back to state_dict")
        return _export_state_dict(model, export_dir)

    model.eval()
    traced = torch.jit.trace(model, sample_input)
    path = export_dir / "model_traced.pt"
    traced.save(str(path))
    logger.info(f"Exported TorchScript to {path}")
    return str(path)


def _export_onnx(
    model: nn.Module,
    export_dir: Path,
    sample_input: Optional[torch.Tensor],
    config: ExportConfig,
) -> str:
    if sample_input is None:
        logger.warning("ONNX export requires sample_input, falling back to state_dict")
        return _export_state_dict(model, export_dir)

    try:
        path = export_dir / "model.onnx"

        dynamic_axes = {}
        if config.dynamic_axes:
            dynamic_axes = {
                "input": {0: "batch"},
                "output": {0: "batch"},
            }

        torch.onnx.export(
            model,
            sample_input,
            str(path),
            opset_version=config.opset_version,
            dynamic_axes=dynamic_axes if config.dynamic_axes else None,
            do_constant_folding=config.optimize_for_inference,
        )
        logger.info(f"Exported ONNX to {path}")
        return str(path)
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        return _export_state_dict(model, export_dir)
