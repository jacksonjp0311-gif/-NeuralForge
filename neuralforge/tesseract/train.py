"""Train a Tesseract Pathway Network on synthetic route-governance data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from neuralforge.tesseract.data import make_tesseract_loaders
from neuralforge.tesseract.evaluate import evaluate_tpn_model
from neuralforge.tesseract.loss import tesseract_compound_loss
from neuralforge.tesseract.network import TesseractPathwayNetwork


def train_tpn_synthetic(
    *,
    n: int = 512,
    epochs: int = 3,
    batch_size: int = 32,
    d_model: int = 64,
    lr: float = 3e-3,
    seed: int = 42,
    output_path: str | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_tesseract_loaders(n=n, seed=seed, batch_size=batch_size)
    model = TesseractPathwayNetwork(input_dim=16, d_model=d_model, num_routes=5, top_k=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for x, targets in train_loader:
            x = x.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = tesseract_compound_loss(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item()) * x.shape[0]
            seen += x.shape[0]
        metrics = evaluate_tpn_model(model, val_loader, device=device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = loss_sum / max(seen, 1)
        history.append(metrics)

    final = evaluate_tpn_model(model, val_loader, device=device)
    report = {
        "status": "completed",
        "epochs": epochs,
        "samples": n,
        "batch_size": batch_size,
        "d_model": d_model,
        "device": str(device),
        "final": final,
        "history": history,
        "claim_boundary": "Synthetic TPN training demonstrates code path validity, not production intelligence or safety.",
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=48)
    parser.add_argument("--output", default="reports/tpn_training_latest.json")
    args = parser.parse_args()

    report = train_tpn_synthetic(
        n=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        d_model=args.d_model,
        output_path=args.output,
    )
    print(json.dumps(report["final"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
