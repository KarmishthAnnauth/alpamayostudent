"""Open-loop evaluation: minADE_k over the horizon."""
from __future__ import annotations
import torch

from ..data.dataset import move_batch


def ade(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Single-mode ADE. pred: (B, H, 2+); gt: (B, H, 2+)."""
    return torch.linalg.norm(pred[..., :2] - gt[..., :2], dim=-1).mean(-1)


def min_ade(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Per-sample minADE. pred: (B, K, H, 2+); gt: (B, H, 2+)."""
    d = torch.linalg.norm(pred[..., :2] - gt[:, None, :, :2], dim=-1)  # (B, K, H)
    return d.mean(-1).min(-1).values


@torch.no_grad()
def evaluate(model_sample_fn, dataloader, k: int, gt_key: str = "gt_future_xyz",
             device: str = "cuda") -> dict:
    """minADE_k over a dataloader. `model_sample_fn(batch, k) -> (B, K, H, 3+)`."""
    scores = []
    for batch in dataloader:
        batch = move_batch(batch, device)
        if gt_key not in batch:
            raise KeyError(
                f"{gt_key} missing from the batch — the collator must carry the "
                "positional GT for a distance metric; `gt_traj` is action space "
                "and is not a substitute")
        pred = model_sample_fn(batch, k=k)          # (B, K, H, 3)
        gt = batch[gt_key]
        if pred.shape[-2] != gt.shape[-2]:
            raise ValueError(
                f"horizon mismatch: predicted {pred.shape[-2]} waypoints against "
                f"{gt.shape[-2]} GT — check tokens_per_future_traj against "
                "eval.horizon_s")
        scores.append(min_ade(pred, gt).cpu())
    s = torch.cat(scores)
    return {"minade": float(s.mean()), "n": int(s.numel()),
            "p90": float(s.quantile(0.9))}
