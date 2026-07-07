
"""Synthetic datasets for Tesseract Pathway Network training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from neuralforge.tesseract.axes import build_route_state

ROUTE_TO_ID = {
    "engage": 0,
    "repair_axis": 1,
    "authority_required": 2,
    "shadow": 3,
    "rehydrate_or_retrieve": 4,
}
ID_TO_ROUTE = {v: k for k, v in ROUTE_TO_ID.items()}


@dataclass(frozen=True)
class TesseractSample:
    x: torch.Tensor
    route: int
    authority: int
    evidence: int
    coherence: float
    delta_phi: float
    vertex: str
    axis_scores: torch.Tensor


class SyntheticTesseractRouteDataset(Dataset):
    """Generate deterministic route-governance samples.

    Feature layout, length 16:
    - 0:4   axis scores: intent, evidence, authority, context
    - 4     mutation requested flag
    - 5:9   missing-axis binary mask
    - 9     coherence proxy
    - 10    delta_phi
    - 11:16 route one-hot prior/noise channel
    """

    def __init__(self, n: int = 512, seed: int = 42, horizon: float = 0.70) -> None:
        super().__init__()
        self.n = int(n)
        self.seed = int(seed)
        self.horizon = float(horizon)
        g = torch.Generator().manual_seed(seed)
        self.axis_scores = torch.rand((self.n, 4), generator=g)
        self.mutation_requested = (torch.rand((self.n,), generator=g) > 0.55).float()
        self.samples = [self._make(i) for i in range(self.n)]

    def _make(self, idx: int) -> TesseractSample:
        axes = self.axis_scores[idx]
        mutation = bool(self.mutation_requested[idx].item() > 0.5)
        state = build_route_state(
            {
                "intent": float(axes[0]),
                "evidence": float(axes[1]),
                "authority": float(axes[2]),
                "context": float(axes[3]),
            },
            mutation_requested=mutation,
            horizon=self.horizon,
        )
        missing_mask = torch.tensor([
            1.0 if axis in state.missing else 0.0
            for axis in ("intent", "evidence", "authority", "context")
        ])
        route_id = ROUTE_TO_ID[state.route]
        route_prior = torch.zeros(5)
        route_prior[route_id] = 1.0
        x = torch.cat([
            axes.float(),
            torch.tensor([1.0 if mutation else 0.0]),
            missing_mask.float(),
            torch.tensor([float(state.coherence), float(state.delta_phi)]),
            route_prior.float(),
        ])
        authority_label = 0 if "authority" in state.missing else 1
        if mutation and "authority" in state.missing:
            authority_label = 2
        evidence_label = 0 if "evidence" in state.missing else 1
        if state.coherence < self.horizon and "evidence" not in state.missing:
            evidence_label = 2
        return TesseractSample(
            x=x,
            route=route_id,
            authority=authority_label,
            evidence=evidence_label,
            coherence=float(state.coherence),
            delta_phi=float(state.delta_phi),
            vertex=state.vertex,
            axis_scores=axes.float(),
        )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        target = {
            "route": torch.tensor(sample.route, dtype=torch.long),
            "authority": torch.tensor(sample.authority, dtype=torch.long),
            "evidence": torch.tensor(sample.evidence, dtype=torch.long),
            "coherence": torch.tensor(sample.coherence, dtype=torch.float32),
            "delta_phi": torch.tensor(sample.delta_phi, dtype=torch.float32),
            "vertex": torch.tensor(int(sample.vertex, 2), dtype=torch.long),
            "axis_scores": sample.axis_scores.float(),
        }
        return sample.x, target


def collate_tesseract_batch(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    xs = torch.stack([item[0] for item in batch])
    keys = batch[0][1].keys()
    targets = {key: torch.stack([item[1][key] for item in batch]) for key in keys}
    return xs, targets


def make_tesseract_loaders(
    *,
    n: int = 512,
    seed: int = 42,
    batch_size: int = 32,
    val_fraction: float = 0.20,
) -> tuple[DataLoader, DataLoader]:
    dataset = SyntheticTesseractRouteDataset(n=n, seed=seed)
    val_size = max(1, int(n * val_fraction))
    train_size = n - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed + 1),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_tesseract_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_tesseract_batch)
    return train_loader, val_loader
