"""Teacher->student layer correspondence."""
from __future__ import annotations
import torch
import torch.nn as nn


def uniform_map(teacher_layers: list[int], n_student_layers: int) -> dict[int, int]:
    """student_layer -> teacher_layer, ONE student layer per teacher layer."""
    ts = sorted(teacher_layers)
    if len(ts) > n_student_layers:
        raise ValueError(
            f"{len(ts)} cached teacher layers but only {n_student_layers} student "
            "layers — the map cannot be injective; drop entries from "
            "teacher.feat_layers")
    out = {}
    for i, t in enumerate(ts):
        s_idx = round(i * (n_student_layers - 1) / max(len(ts) - 1, 1))
        out[int(s_idx)] = int(t)
    return out


def pool_prompt_segments(hidden: torch.Tensor, n_prompt: torch.Tensor,
                         pool_len: int = 8) -> torch.Tensor:
    """(B, L, D) reasoner states -> (B, pool_len, D), pooled like the teacher."""
    B, L, D = hidden.shape
    out = hidden.new_zeros(B, pool_len, D)
    for b in range(B):
        n = int(n_prompt[b]) if n_prompt is not None else L
        n = max(1, min(n, L))
        segs = torch.chunk(hidden[b, :n], pool_len, dim=0)
        if len(segs) != pool_len:
            raise ValueError(
                f"prompt of {n} tokens cannot be split into {pool_len} segments "
                "the way the teacher's prefill was — the cache and the student "
                "would be pooling different things")
        out[b] = torch.stack([s.mean(0) for s in segs])
    return out


@torch.no_grad()
def cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between (N, Dx) and (N, Dy) feature matrices."""
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    xty = (x.T @ y).norm() ** 2
    return float(xty / (((x.T @ x).norm()) * ((y.T @ y).norm()) + 1e-8))


@torch.no_grad()
def cka_map(student_feats: dict[int, torch.Tensor],
            teacher_feats: dict[int, torch.Tensor]) -> dict[int, int]:
    """Greedy monotone assignment maximizing CKA."""
    s_layers, t_layers = sorted(student_feats), sorted(teacher_feats)
    if len(t_layers) > len(s_layers):
        raise ValueError(f"{len(t_layers)} teacher layers vs {len(s_layers)} "
                         "student layers — no injective monotone map exists")
    out: dict[int, int] = {}
    lo = 0
    for n, tj in enumerate(t_layers):
        hi = len(s_layers) - (len(t_layers) - n - 1)
        best, best_s = -1.0, s_layers[lo]
        for k in range(lo, hi):
            score = cka(student_feats[s_layers[k]], teacher_feats[tj])
            if score > best:
                best, best_s = score, s_layers[k]
        out[int(best_s)] = int(tj)
        lo = s_layers.index(best_s) + 1
    return out


class FeatureProjections(nn.Module):
    """One linear projection per mapped pair: student dim -> teacher dim."""

    def __init__(self, layer_map: dict[int, int], d_student: int, d_teacher: int):
        super().__init__()
        if len(set(layer_map.values())) != len(layer_map):
            raise ValueError(
                f"layer map is not injective ({layer_map}) — forward() keys its "
                "output by teacher layer, so duplicate targets silently discard "
                "projections and train them on nothing")
        self.layer_map = layer_map
        self.proj = nn.ModuleDict({
            str(s): nn.Linear(d_student, d_teacher, bias=False) for s in layer_map
        })

    def forward(self, student_hidden: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
        """Returns teacher_layer -> projected student features."""
        return {self.layer_map[s]: self.proj[str(s)](h)
                for s, h in student_hidden.items() if s in self.layer_map}
