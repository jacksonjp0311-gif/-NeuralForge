
"""Checkpoint training and loading for the local Tesseract mind core."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


def _torch_load_local_checkpoint(path, *, map_location=None):
    """Load trusted local TPN checkpoints without PyTorch's weights_only FutureWarning."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


from neuralforge.tesseract.data import make_tesseract_loaders
from neuralforge.tesseract.evaluate import evaluate_tpn_model
from neuralforge.tesseract.loss import tesseract_compound_loss
from neuralforge.tesseract.network import TesseractPathwayNetwork


@dataclass(frozen=True)
class TesseractCheckpointConfig:
    input_dim: int = 16
    d_model: int = 32
    num_routes: int = 5
    top_k: int = 4
    samples: int = 256
    epochs: int = 2
    batch_size: int = 32
    seed: int = 17
    lr: float = 3e-3


DEFAULT_OUTPUT_DIR = Path("artifacts") / "tpn"
DEFAULT_NAME = "tpn_mind_core_v0_4"


def _device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_tpn_checkpoint(
    model: TesseractPathwayNetwork,
    path: str | Path,
    *,
    config: TesseractCheckpointConfig,
    metrics: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "neuralforge.tesseract.checkpoint.v0.4",
        "config": asdict(config),
        "metrics": metrics,
        "manifest": dict(manifest or {}),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    return path


def write_manifest(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_tpn_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> tuple[TesseractPathwayNetwork, dict[str, Any]]:
    path = Path(path)
    payload = _torch_load_local_checkpoint(path, map_location=map_location)
    config = TesseractCheckpointConfig(**payload["config"])
    model = TesseractPathwayNetwork(
        input_dim=config.input_dim,
        d_model=config.d_model,
        num_routes=config.num_routes,
        top_k=config.top_k,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def train_tpn_checkpoint(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    name: str = DEFAULT_NAME,
    config: TesseractCheckpointConfig | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    config = config or TesseractCheckpointConfig()
    torch.manual_seed(config.seed)
    dev = _device(device)

    train_loader, val_loader = make_tesseract_loaders(
        n=config.samples,
        seed=config.seed,
        batch_size=config.batch_size,
    )
    model = TesseractPathwayNetwork(
        input_dim=config.input_dim,
        d_model=config.d_model,
        num_routes=config.num_routes,
        top_k=config.top_k,
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

    history: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for x, targets in train_loader:
            x = x.to(dev)
            targets = {k: v.to(dev) for k, v in targets.items()}
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = tesseract_compound_loss(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.item()) * x.shape[0]
            seen += x.shape[0]

        metrics = evaluate_tpn_model(model, val_loader, device=dev)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss / max(seen, 1)
        history.append(metrics)

    final_metrics = evaluate_tpn_model(model, val_loader, device=dev)
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / f"{name}.pt"
    manifest_path = output_dir / f"{name}.json"

    manifest = {
        "artifact": "TesseractMindCore checkpoint",
        "version": "tpn.v0.4",
        "checkpoint_path": str(checkpoint_path.as_posix()),
        "manifest_path": str(manifest_path.as_posix()),
        "config": asdict(config),
        "device": str(dev),
        "final_metrics": final_metrics,
        "history": history,
        "claim_boundary": "Local synthetic checkpoint for TPN mind-core routing; not world knowledge, AGI, or production safety proof.",
    }

    save_tpn_checkpoint(model, checkpoint_path, config=config, metrics=final_metrics, manifest=manifest)
    write_manifest(manifest_path, manifest)

    return {
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = TesseractCheckpointConfig(
        d_model=args.d_model,
        top_k=args.top_k,
        samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    result = train_tpn_checkpoint(
        output_dir=args.output_dir,
        name=args.name,
        config=cfg,
        device=args.device,
    )
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
