"""Stage-1 gate: minADE of the AR tower's decoded trajectory tokens (CoC teacher-forced, batch 1)."""
from __future__ import annotations
import functools
import logging

import torch
from torch.utils.data import DataLoader

from ..data.dataset import Stage1Dataset, collate_stage1
from ..data.splits import load_split
from .open_loop import evaluate

log = logging.getLogger(__name__)


@torch.no_grad()
def coarse_minade(cfg, student, split: str = "challenging", k: int = 6,
                  max_windows: int | None = None) -> float:
    ds = Stage1Dataset(cfg, student.context_builder(),
                       clip_ids=_split_clips(cfg, split),
                       for_generation=True)
    if max_windows is not None and len(ds) > max_windows:
        # fixed prefix so the gate is comparable across epochs
        ds = torch.utils.data.Subset(ds, range(max_windows))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                    collate_fn=functools.partial(
                        collate_stage1, pad_id=student.tokenizer.pad_token_id))

    def sample_fn(batch, k):
        modes = []
        for _ in range(k):
            tok = student.generate_traj_tokens(batch)
            modes.append(student.detokenize_traj(
                tok, batch["hist_xyz"], batch["hist_rot"]).to(batch["gt_future_xyz"]))
        return torch.stack(modes, dim=1)

    out = evaluate(sample_fn, dl, k)
    log.info("coarse minADE_%d on %s: %.3f m (p90 %.3f, n=%d)",
             k, split, out["minade"], out["p90"], out["n"])
    return out["minade"]


def _split_clips(cfg, split: str) -> list[str]:
    return load_split(cfg, split)
