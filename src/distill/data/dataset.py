"""Datasets and collators over cached teacher shards."""
from __future__ import annotations
import json
from pathlib import Path

import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from . import frames
from ..student.context import PREFIX_MASK

log = logging.getLogger(__name__)


def _load_shard(path: Path) -> dict:
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    return d


# student inputs share the clip dir with the target shards
INPUT_SUFFIX = "_input.npz"


def discover_shards(cache_root: Path, clip_ids: list[str] | None = None) -> list[Path]:
    if clip_ids is None:
        with open(cache_root / "manifest.json") as f:
            clip_ids = json.load(f)["clips"]
    shards = []
    for cid in clip_ids:
        shards += sorted(p for p in (cache_root / cid).glob("*.npz")
                         if not p.name.endswith(INPUT_SUFFIX))
    return shards


class DistillShardDataset(Dataset):
    """Returns raw shard dicts; stage-specific collators shape the batch."""

    def __init__(self, cfg, clip_ids: list[str] | None = None):
        self.cfg = cfg
        self.shards = discover_shards(Path(cfg.paths.cache_root), clip_ids)
        if not self.shards:
            raise RuntimeError("No cached shards found - run scripts/02_label.py first")

    def __len__(self) -> int:
        return len(self.shards)

    def __getitem__(self, i: int) -> dict:
        d = _load_shard(self.shards[i])
        d["clip_id"] = self.shards[i].parent.name
        return d


class Stage1Dataset(Dataset):
    """Cached teacher targets + the student's context for the same window."""

    def __init__(self, cfg, context_builder, clip_ids: list[str] | None = None,
                 for_generation: bool = False, cot_generation: bool = False,
                 image_dropout: float = 0.0, filter_contradictions: bool = False,
                 prefix_noise_bins: float = 0.0, prefix_mask_prob: float = 0.0):
        from .preprocess import load_window, window_t0_us

        self.cfg = cfg
        self.ctx = context_builder
        st = getattr(cfg, "stage1", None)
        self.traj_prefix = str(st.get("traj_prefix", "teacher") if st is not None else "teacher")
        if self.traj_prefix not in ("teacher", "gt"):
            raise ValueError(f"stage1.traj_prefix must be 'teacher' or 'gt', got {self.traj_prefix!r}")
        self.route_hint = bool(st.get("route_hint", False)) if st is not None else False
        if not 0.0 <= float(image_dropout) <= 1.0:
            raise ValueError(f"image_dropout must be in [0, 1], got {image_dropout}")
        self.image_dropout = float(image_dropout)
        self.filter_contradictions = bool(filter_contradictions)
        if float(prefix_noise_bins) < 0.0:
            raise ValueError(f"prefix_noise_bins must be >= 0, got {prefix_noise_bins}")
        self.prefix_noise_bins = float(prefix_noise_bins)
        if not 0.0 <= float(prefix_mask_prob) <= 1.0:
            raise ValueError(f"prefix_mask_prob must be in [0, 1], got {prefix_mask_prob}")
        self.prefix_mask_prob = float(prefix_mask_prob)
        # eval: trajectory positions are decoded, not teacher-forced
        self.for_generation = bool(for_generation)
        # inspection: CoC is free-run too
        self.cot_generation = bool(cot_generation)
        self._load_window = load_window
        self._t0 = window_t0_us
        self.cache_root = Path(cfg.paths.cache_root)
        self.shards = discover_shards(self.cache_root, clip_ids)
        if not self.shards:
            raise RuntimeError("No cached shards found - run scripts/02_label.py first")
        self.grounding_stats = None
        if self.filter_contradictions:
            from . import grounding
            self.shards, self.grounding_stats = grounding.grounded_shards(self.shards)
            grounding.log_stats(self.grounding_stats)
            if not self.shards:
                raise RuntimeError("grounding filter dropped every window")
        self._warned_uncached = False

    def __len__(self) -> int:
        return len(self.shards)

    def __getitem__(self, i: int) -> dict:
        path = self.shards[i]
        clip_id, w_idx = path.parent.name, int(path.stem)
        d = _load_shard(path)
        d["clip_id"] = clip_id
        window = self._window(clip_id, w_idx)
        key = "gt_traj_token_ids" if self.traj_prefix == "gt" else "traj_token_ids"
        traj_bins = [int(b) for b in d[key]]
        if not self.for_generation:
            if self.prefix_mask_prob > 0.0 and float(torch.rand(())) < self.prefix_mask_prob:
                traj_bins = [PREFIX_MASK] * len(traj_bins)
            elif self.prefix_noise_bins > 0.0:
                traj_bins = jitter_prefix(traj_bins, self.prefix_noise_bins)
        from . import grounding
        nav = grounding.route_hint(d["gt_future_xyz"]) if self.route_hint else None
        ctx = self.ctx.build(window,
                             coc_text=None if self.cot_generation else str(d["coc_text"]),
                             traj_bins=None if self.for_generation else traj_bins,
                             nav_text=nav,
                             for_generation=self.for_generation)
        d["student"] = ctx
        d["route_hint"] = nav or ""
        d["img_drop"] = bool(self.image_dropout > 0.0
                             and float(torch.rand(())) < self.image_dropout)
        d["hist_xyz"] = window.data["ego_history_xyz"][:, -1][0]
        d["hist_rot"] = window.data["ego_history_rot"][:, -1][0]
        return d

    def _window(self, clip_id: str, w_idx: int):
        """Cached student input if present, else re-stream the clip."""
        cached = frames.input_path(self.cache_root, clip_id, w_idx)
        if cached.exists():
            return frames.load_window_input(cached, clip_id)
        if not self._warned_uncached:
            self._warned_uncached = True
            log.warning(
                "no cached student input at %s — falling back to streaming the "
                "clip for every window, every epoch. Re-run scripts/02_label.py "
                "to backfill (it writes inputs even where targets already exist).",
                cached)
        return self._load_window(self.cfg, clip_id, self._t0(self.cfg, w_idx))


N_FUTURE_BINS = 3000


def jitter_prefix(bins: list[int], max_sigma: float, n_bins: int = N_FUTURE_BINS) -> list[int]:
    """Jitter prefix bins by N(0, sigma), sigma ~ U(0, max_sigma)."""
    sigma = float(torch.rand(())) * float(max_sigma)
    if sigma <= 0.0:
        return list(bins)
    off = torch.round(torch.randn(len(bins)) * sigma).to(torch.long)
    out = (torch.as_tensor(bins, dtype=torch.long) + off).clamp_(0, n_bins - 1)
    return [int(b) for b in out]


def collate_student(items: list[dict], pad_id: int) -> dict:
    """Right-pad the student contexts and stack the image tensors."""
    ctxs = [it["student"] for it in items]
    # image dropout: zero frames and drop CoC/struct targets
    drop = [bool(it.get("img_drop", False)) for it in items]
    lens = [c["input_ids"].shape[0] for c in ctxs]
    L = max(lens)
    B = len(ctxs)
    input_ids = torch.full((B, L), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(B, L, dtype=torch.bool)
    coc_mask = torch.zeros(B, L, dtype=torch.bool)
    struct_mask = torch.zeros(B, L, dtype=torch.bool)
    traj_mask = torch.zeros(B, L, dtype=torch.bool)
    for j, c in enumerate(ctxs):
        n = c["input_ids"].shape[0]
        input_ids[j, :n] = c["input_ids"]
        attention_mask[j, :n] = True
        spans = [(c["traj_span"], traj_mask)]
        if not drop[j]:
            spans += [(c["coc_span"], coc_mask), (c.get("struct_span"), struct_mask)]
        for span, mask in spans:
            if span is not None:
                mask[j, span[0]:span[1]] = True
    out = dict(input_ids=input_ids, attention_mask=attention_mask,
               coc_pos=coc_mask, struct_pos=struct_mask, traj_pos=traj_mask,
               n_prompt=torch.tensor([c["n_prompt"] for c in ctxs]),
               img_drop=torch.tensor(drop, dtype=torch.bool))
    if ctxs[0]["pixel_values"] is not None:
        out["pixel_values"] = torch.cat(
            [torch.zeros_like(c["pixel_values"]) if drop[j] else c["pixel_values"]
             for j, c in enumerate(ctxs)], dim=0)
        out["image_grid_thw"] = torch.cat([c["image_grid_thw"] for c in ctxs], dim=0)
    return out


def move_batch(batch, device="cuda", non_blocking: bool = True):
    """Recursively move a collated batch to `device`."""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {k: move_batch(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_batch(v, device, non_blocking) for v in batch)
    return batch


def collate_stage1(batch: list[dict], pad_id: int) -> dict:
    """Pads token streams; stacks feature targets per teacher layer."""
    def pad(key, dtype=torch.long):
        seqs = [torch.as_tensor(b[key], dtype=dtype) for b in batch]
        L = max(s.shape[0] for s in seqs)
        out = torch.full((len(seqs), L, *seqs[0].shape[1:]), pad_id, dtype=dtype)
        mask = torch.zeros(len(seqs), L, dtype=torch.bool)
        for j, s in enumerate(seqs):
            out[j, : s.shape[0]] = s
            mask[j, : s.shape[0]] = True
        return out, mask

    traj, traj_mask = pad("traj_token_ids")
    gt_traj_tok, _ = pad("gt_traj_token_ids")
    coc, coc_mask = pad("coc_token_ids")
    topk_idx, _ = pad("traj_topk_idx")
    topk_logp, _ = pad("traj_topk_logp", dtype=torch.float32)
    layers = [int(x) for x in batch[0]["feat_layers"]]
    feats = {l: torch.stack([torch.as_tensor(b[f"feat_{l}"], dtype=torch.float32) for b in batch])
             for l in layers}
    out = dict(
        traj=traj, traj_mask=traj_mask, coc=coc, coc_mask=coc_mask,
        topk_idx=topk_idx, topk_logp=topk_logp, feats=feats,
        gt_traj_tok=gt_traj_tok,
        gt_traj=torch.stack([torch.as_tensor(b["gt_traj"], dtype=torch.float32) for b in batch]),
        clip_ids=[b["clip_id"] for b in batch],
    )
    out["gt_future_xyz"] = torch.stack(
        [torch.as_tensor(b["gt_future_xyz"], dtype=torch.float32) for b in batch])
    for key in ("hist_xyz", "hist_rot"):
        if key in batch[0]:
            out[key] = torch.stack(
                [torch.as_tensor(b[key], dtype=torch.float32) for b in batch])
    if "student" in batch[0]:
        out.update(collate_student(batch, pad_id))
    return out


def collate_stage2(batch: list[dict], pad_id: int) -> dict:
    """Flattens the K cached flow targets per window into the batch dim."""
    base = collate_stage1(batch, pad_id)
    if "flow_t" not in batch[0]:
        return base
    t, a_t, v = [], [], []
    owner = []
    for j, b in enumerate(batch):
        k = b["flow_t"].shape[0]
        t.append(torch.as_tensor(b["flow_t"], dtype=torch.float32))
        a_t.append(torch.as_tensor(b["flow_a_t"], dtype=torch.float32))
        v.append(torch.as_tensor(b["flow_v"], dtype=torch.float32))
        owner += [j] * k
    base.update(flow_t=torch.cat(t), flow_a_t=torch.cat(a_t),
                flow_v=torch.cat(v), flow_owner=torch.as_tensor(owner))
    return base
