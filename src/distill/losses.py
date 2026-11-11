"""Losses for both stages. Shapes follow data/dataset.py collators."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def gather_targets(logits: torch.Tensor, input_ids: torch.Tensor,
                   pos_mask: torch.Tensor, n_targets: int):
    """Select target positions (shifted by one) and return `(logits, target_ids, valid)`."""
    B, L, V = logits.shape
    out = logits.new_zeros(B, n_targets, V)
    tgt = input_ids.new_zeros(B, n_targets)
    valid = torch.zeros(B, n_targets, dtype=torch.bool, device=logits.device)
    for b in range(B):
        pos = pos_mask[b].nonzero(as_tuple=True)[0]
        pos = pos[pos > 0][:n_targets]          # position 0 has no predictor
        k = pos.shape[0]
        if k == 0:
            continue
        out[b, :k] = logits[b, pos - 1]
        tgt[b, :k] = input_ids[b, pos]
        valid[b, :k] = True
    return out, tgt, valid


def traj_topk_kl_per_pos(student_logits: torch.Tensor, topk_idx: torch.Tensor,
                         topk_logp: torch.Tensor) -> torch.Tensor:
    """Per-position KL(teacher || student), (B, L), before any masking."""
    eps = 1e-9
    logp_s = F.log_softmax(student_logits, dim=-1)
    logp_s_k = torch.gather(logp_s, -1, topk_idx.clamp_min(0))       # (B,L,K)
    p_t_k = topk_logp.exp()                                          # (B,L,K)
    p_t_tail = (1 - p_t_k.sum(-1)).clamp_min(0.0)                    # (B,L)
    p_s_tail = (1 - logp_s_k.exp().sum(-1)).clamp_min(eps)           # (B,L)
    kl = (p_t_k * (topk_logp - logp_s_k)).sum(-1) \
        + p_t_tail * (torch.log(p_t_tail + eps) - torch.log(p_s_tail))
    return kl.clamp_min(0.0)  # numerical guard


def masked_mean(per_pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of a (B, L) per-position loss over a (B, L) boolean mask."""
    return (per_pos * mask).sum() / mask.sum().clamp_min(1)


def traj_topk_kl(student_logits: torch.Tensor, topk_idx: torch.Tensor,
                 topk_logp: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """KL(teacher || student) over the teacher's top-k plus a tail bucket."""
    return masked_mean(
        traj_topk_kl_per_pos(student_logits, topk_idx, topk_logp), mask)


def text_kl_or_ce(student_logits: torch.Tensor, target_ids: torch.Tensor,
                  mask: torch.Tensor, vocab_ok: bool) -> torch.Tensor:
    """CE on teacher-generated text tokens."""
    del vocab_ok
    if student_logits.shape[1] == 0 or not bool(mask.any()):
        # Span fully masked: nothing to score.
        return student_logits.new_zeros(())
    ce = F.cross_entropy(student_logits.transpose(1, 2), target_ids, reduction="none")
    return (ce * mask).sum() / mask.sum().clamp_min(1)


def feature_match(proj_student: dict[int, torch.Tensor],
                  teacher_feats: dict[int, torch.Tensor],
                  kind: str = "smooth_l1") -> torch.Tensor:
    """Regression on pooled features at mapped layers."""
    losses = []
    for layer, tf in teacher_feats.items():
        if layer not in proj_student:
            continue
        sf = proj_student[layer]
        tf = tf.to(sf.dtype)
        if kind == "cosine":
            losses.append(1 - F.cosine_similarity(sf, tf, dim=-1).mean())
        else:
            losses.append(F.smooth_l1_loss(sf, tf))
    return torch.stack(losses).mean() if losses else torch.zeros((), device="cuda")


def gt_traj_ce(student_logits: torch.Tensor, gt_token_ids: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """CE on ground-truth trajectory tokens."""
    ce = F.cross_entropy(student_logits.transpose(1, 2), gt_token_ids, reduction="none")
    return (ce * mask).sum() / mask.sum().clamp_min(1)


def gt_traj_soft_ce(student_logits: torch.Tensor, gt_token_ids: torch.Tensor,
                    mask: torch.Tensor, sigma_bins: float,
                    lo: int, hi: int, per_pos: bool = False) -> torch.Tensor:
    """CE against a Gaussian over bins neighbouring the GT bin, restricted to [lo, hi)."""
    half = max(1, int(round(3.0 * sigma_bins)))                  # +/-3 sigma
    off = torch.arange(-half, half + 1, device=student_logits.device)   # (W,)
    bins = gt_token_ids.unsqueeze(-1) + off                      # (B, T, W)
    in_range = (bins >= lo) & (bins < hi)
    q = torch.exp(-0.5 * (off.float() / sigma_bins) ** 2)        # (W,)
    q = q.expand_as(bins) * in_range
    q = q / q.sum(-1, keepdim=True).clamp_min(1e-12)
    logp = F.log_softmax(student_logits, dim=-1)                 # full-vocab denominator
    logp_w = torch.gather(logp, -1, bins.clamp(lo, hi - 1))      # (B, T, W)
    ce = -(q * logp_w).sum(-1)                                   # (B, T)
    if per_pos:
        return ce * mask          # (B, T), unreduced
    return (ce * mask).sum() / mask.sum().clamp_min(1)


def flow_distill(v_student: torch.Tensor, v_teacher: torch.Tensor) -> torch.Tensor:
    """MSE between student and cached teacher velocity, in fp32."""
    return F.mse_loss(v_student.float(), v_teacher.float())


def flow_matching_gt(v_student: torch.Tensor, a0: torch.Tensor, a1: torch.Tensor) -> torch.Tensor:
    """Flow-matching loss: a_t = t a1 + (1-t) a0, v* = a1 - a0, in fp32."""
    return F.mse_loss(v_student.float(), (a1 - a0).float())


def stage_weights(cfg_stage, step: int, total_steps: int) -> dict[str, float]:
    """Time-varying loss weights from the config schedule fractions."""
    w = dict(cfg_stage.loss_weights.raw)
    frac = step / max(total_steps, 1)
    if "feat" in w:
        warm = cfg_stage.get("feat_warmup_frac", 0.0) or 0.0
        if warm > 0:
            w["feat"] *= min(frac / warm, 1.0)
    if "gt_ce" in w:
        # Linear ramp from gt_ce to gt_ce_end over the run.
        gt_end = cfg_stage.get("gt_ce_end", cfg_stage.get("gt_ce_min", None))
        if gt_end is not None:
            w["gt_ce"] += (float(gt_end) - w["gt_ce"]) * min(max(frac, 0.0), 1.0)
    if "flow_distill" in w:
        anneal = cfg_stage.get("anneal_to_gt_frac", 0.0) or 0.0
        if anneal > 0 and frac > 1 - anneal:
            a = (frac - (1 - anneal)) / anneal
            w["flow_distill"] *= (1 - a)
            w["fm_gt"] *= (1 + a)
    return w
