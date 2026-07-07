
"""Adaptive replay learning for the local Tesseract mind core.

v0.6 gives TPN a self-learning path without allowing uncontrolled mutation.
Learning happens through an append-only JSONL replay ledger. Only approved
records are admitted into fine-tuning.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from neuralforge.tesseract.checkpoint import (
    TesseractCheckpointConfig,
    load_tpn_checkpoint,
    save_tpn_checkpoint,
    write_manifest,
)
from neuralforge.tesseract.data import SyntheticTesseractRouteDataset
from neuralforge.tesseract.evaluate import evaluate_tpn_model
from neuralforge.tesseract.loss import tesseract_compound_loss


@dataclass(frozen=True)
class TesseractFeedbackRecord:
    vector: list[float]
    targets: dict[str, Any]
    approved: bool = True
    source: str = "operator"
    note: str = ""
    schema_version: str = "tpn.feedback.v0.6"


class TesseractReplayLedger:
    """Append-only local replay ledger for approved learning events."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: TesseractFeedbackRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), sort_keys=True) + "\n")

    def records(self, *, approved_only: bool = True) -> list[TesseractFeedbackRecord]:
        if not self.path.exists():
            return []
        loaded: list[TesseractFeedbackRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            record = TesseractFeedbackRecord(**data)
            if approved_only and not record.approved:
                continue
            loaded.append(record)
        return loaded

    def __len__(self) -> int:
        return len(self.records(approved_only=False))


class TesseractReplayDataset(Dataset):
    def __init__(self, records: Iterable[TesseractFeedbackRecord]) -> None:
        self.records_list = list(records)
        if not self.records_list:
            raise ValueError("TesseractReplayDataset requires at least one approved record.")

    def __len__(self) -> int:
        return len(self.records_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        record = self.records_list[idx]
        targets = record.targets
        axis_scores = targets.get("axis_scores", record.vector[:4])
        return (
            torch.tensor(record.vector, dtype=torch.float32),
            {
                "route": torch.tensor(int(targets["route"]), dtype=torch.long),
                "authority": torch.tensor(int(targets["authority"]), dtype=torch.long),
                "evidence": torch.tensor(int(targets["evidence"]), dtype=torch.long),
                "coherence": torch.tensor(float(targets["coherence"]), dtype=torch.float32),
                "delta_phi": torch.tensor(float(targets["delta_phi"]), dtype=torch.float32),
                "vertex": torch.tensor(int(targets["vertex"]), dtype=torch.long),
                "axis_scores": torch.tensor(list(axis_scores), dtype=torch.float32),
            },
        )


def seed_replay_from_synthetic(path: str | Path, *, n: int = 128, seed: int = 61) -> dict[str, Any]:
    """Create an approved seed replay ledger from the deterministic synthetic task."""

    ledger = TesseractReplayLedger(path)
    dataset = SyntheticTesseractRouteDataset(n=n, seed=seed)
    for x, target in dataset:
        record = TesseractFeedbackRecord(
            vector=[float(v) for v in x.tolist()],
            targets={
                "route": int(target["route"].item()),
                "authority": int(target["authority"].item()),
                "evidence": int(target["evidence"].item()),
                "coherence": float(target["coherence"].item()),
                "delta_phi": float(target["delta_phi"].item()),
                "vertex": int(target["vertex"].item()),
                "axis_scores": [float(v) for v in target["axis_scores"].tolist()],
            },
            approved=True,
            source="synthetic_seed",
            note="Seeded approved replay event for local adaptive TPN training.",
        )
        ledger.append(record)
    return {
        "ledger_path": str(Path(path)),
        "records_written": n,
        "schema_version": "tpn.feedback.v0.6",
    }


def append_operator_feedback(
    path: str | Path,
    *,
    vector: list[float],
    route: int,
    authority: int,
    evidence: int,
    coherence: float,
    delta_phi: float,
    vertex: int,
    axis_scores: list[float] | None = None,
    approved: bool = True,
    note: str = "",
) -> dict[str, Any]:
    """Append a human/operator-approved learning record."""

    if len(vector) != 16:
        raise ValueError(f"TPN feedback vector must have 16 floats, got {len(vector)}.")
    axis_scores = axis_scores or list(vector[:4])
    record = TesseractFeedbackRecord(
        vector=[float(v) for v in vector],
        targets={
            "route": int(route),
            "authority": int(authority),
            "evidence": int(evidence),
            "coherence": float(coherence),
            "delta_phi": float(delta_phi),
            "vertex": int(vertex),
            "axis_scores": [float(v) for v in axis_scores],
        },
        approved=bool(approved),
        source="operator",
        note=note,
    )
    ledger = TesseractReplayLedger(path)
    ledger.append(record)
    return {
        "ledger_path": str(Path(path)),
        "approved": bool(approved),
        "schema_version": record.schema_version,
    }


def train_tpn_from_replay(
    *,
    checkpoint_path: str | Path,
    replay_path: str | Path,
    output_checkpoint: str | Path,
    output_manifest: str | Path | None = None,
    epochs: int = 2,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str | None = None,
) -> dict[str, Any]:
    """Fine-tune a local TPN checkpoint from approved replay records."""

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, payload = load_tpn_checkpoint(checkpoint_path, map_location=dev)
    model.to(dev)

    config = TesseractCheckpointConfig(**payload["config"])
    ledger = TesseractReplayLedger(replay_path)
    records = ledger.records(approved_only=True)
    dataset = TesseractReplayDataset(records)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for x, targets in loader:
            x = x.to(dev)
            targets = {k: v.to(dev) for k, v in targets.items()}
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = tesseract_compound_loss(out, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item()) * x.shape[0]
            seen += x.shape[0]
        history.append({
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "approved_records": len(records),
        })

    metrics = evaluate_tpn_model(model, loader, device=dev)
    manifest = {
        "artifact": "TesseractMindCore adaptive checkpoint",
        "version": "tpn.v0.6",
        "base_checkpoint": str(Path(checkpoint_path).as_posix()),
        "checkpoint_path": str(Path(output_checkpoint).as_posix()),
        "replay_path": str(Path(replay_path).as_posix()),
        "approved_records": len(records),
        "config": asdict(config),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "device": str(dev),
        "metrics": metrics,
        "history": history,
        "claim_boundary": "Adaptive local replay training from approved records; not uncontrolled self-modification or autonomous authority.",
    }

    save_tpn_checkpoint(model, output_checkpoint, config=config, metrics=metrics, manifest=manifest)
    if output_manifest is None:
        output_manifest = str(Path(output_checkpoint).with_suffix(".json"))
    write_manifest(output_manifest, manifest)

    return {
        "checkpoint_path": str(Path(output_checkpoint)),
        "manifest_path": str(Path(output_manifest)),
        "approved_records": len(records),
        "metrics": metrics,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/tpn/tpn_mind_core_v0_4.pt")
    parser.add_argument("--replay", default="artifacts/tpn/replay/tpn_replay_v0_6.jsonl")
    parser.add_argument("--output-checkpoint", default="artifacts/tpn/tpn_mind_core_v0_6.pt")
    parser.add_argument("--output-manifest", default="artifacts/tpn/tpn_mind_core_v0_6.json")
    parser.add_argument("--seed-replay", action="store_true")
    parser.add_argument("--seed-records", type=int, default=128)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    replay_path = Path(args.replay)
    if args.seed_replay or not replay_path.exists():
        if replay_path.exists():
            replay_path.unlink()
        seed_replay_from_synthetic(replay_path, n=args.seed_records, seed=args.seed)

    result = train_tpn_from_replay(
        checkpoint_path=args.checkpoint,
        replay_path=replay_path,
        output_checkpoint=args.output_checkpoint,
        output_manifest=args.output_manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
