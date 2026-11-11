"""Stage 1: distill the teacher reasoner into the student's AR tower with LoRA."""
from __future__ import annotations
import argparse
import functools
import os
import logging
import math
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import load_config
from .data.dataset import Stage1Dataset, collate_stage1, move_batch
from .data.splits import load_split
from .student.edge_wrapper import EdgeStudent
from .student.layer_map import (FeatureProjections, cka_map, pool_prompt_segments,
                                uniform_map)
from . import checkpoint, losses
from .optim import build_optimizer, cosine_lr, set_lr, trainable_report
from .eval.coarse_minade import coarse_minade

log = logging.getLogger(__name__)

#: per-term log key -> config `loss_weights` key
_WKEY = {"traj": "traj_kl", "traj_accel": "traj_kl_accel", "text": "text_kl",
         "struct": "struct_ce", "feat": "feat", "gt_soft": "gt_ce"}

#: logged but not summed into the loss
_DIAG = ("gt",)


def _run_best_dir(cfg, stage: str = "stage1") -> Path:
    """Per-run `best` checkpoint dir, named by SLURM job id or timestamp."""
    rid = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(cfg.paths.runs_root) / stage / f"run-{rid}" / "best"


def _promote_best(run_best: Path) -> None:
    """Atomically point `<stage>/best` at this run's checkpoint."""
    stage_root = run_best.parent.parent          # runs_root/<stage>
    link = stage_root / "best"
    if link.exists() and not link.is_symlink() and link.is_dir():
        legacy = stage_root / f"legacy-best-{int(link.stat().st_mtime)}"
        link.rename(legacy)
        log.warning("moved pre-isolation checkpoint %s -> %s", link, legacy)
    tmp = stage_root / f".best.{os.getpid()}.tmp"
    tmp.unlink(missing_ok=True)
    os.symlink(run_best.relative_to(stage_root), tmp)
    os.replace(tmp, link)


@torch.no_grad()
def _coc_nll(cfg, student, split: str = "val", max_windows: int = 64) -> tuple[float, float]:
    """Teacher-forced CoC and structural CE on a held-out split."""
    ds = Stage1Dataset(cfg, student.context_builder(), clip_ids=load_split(cfg, split))
    if max_windows and len(ds) > max_windows:
        ds = torch.utils.data.Subset(ds, range(max_windows))
    dl = DataLoader(ds, batch_size=cfg.stage1.micro_batch, shuffle=False, num_workers=2,
                    collate_fn=functools.partial(
                        collate_stage1, pad_id=student.tokenizer.pad_token_id))
    text_tot = struct_tot = 0.0
    n = 0
    for batch in dl:
        batch = move_batch(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = student.ar_forward(batch, capture_layers=())
            n_coc = int(batch["coc_pos"].sum(1).max())
            coc_logits, coc_tgt, coc_ok = losses.gather_targets(
                out["logits"], batch["input_ids"], batch["coc_pos"], n_coc)
            n_struct = int(batch["struct_pos"].sum(1).max())
            s_logits, s_tgt, s_ok = losses.gather_targets(
                out["logits"], batch["input_ids"], batch["struct_pos"], n_struct)
            text_tot += float(losses.text_kl_or_ce(coc_logits, coc_tgt, coc_ok, vocab_ok=True))
            struct_tot += float(losses.text_kl_or_ce(s_logits, s_tgt, s_ok, vocab_ok=True))
        n += 1
    return (text_tot / max(n, 1), struct_tot / max(n, 1))


@torch.no_grad()
def _coc_gt(cfg, student, n_windows: int) -> dict:
    """Free-running CoC on val, graded against the GT future trajectory."""
    from .train_grpo_coc import _val_eval
    ds = Stage1Dataset(cfg, student.context_builder(), clip_ids=load_split(cfg, "val"),
                       cot_generation=True)
    return _val_eval(cfg, student, ds, n=int(n_windows), pad_id=student.tokenizer.pad_token_id,
                     max_new=int(cfg.stage1.get("coc_gt_max_new_tokens", 64)), route=False)


def _cached_feature_dim(shard_path, layer: int) -> int:
    """D_t, read off a cached shard's own feature array."""
    import numpy as np
    with np.load(shard_path) as z:
        return int(z[f"feat_{layer}"].shape[-1])


@torch.no_grad()
def _build_layer_map(cfg, student, dl, feat_layers, n_student_layers, pool_len):
    """Teacher->student layer correspondence (`uniform` or `cka`)."""
    mode = str(cfg.student.layer_map.get("mode", "uniform")).lower()
    if mode != "cka":
        return uniform_map(feat_layers, n_student_layers)

    n_batches = max(1, int(cfg.student.layer_map.get("cka_probe_batches", 4)))
    s_acc: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_student_layers)}
    t_acc: dict[int, list[torch.Tensor]] = {l: [] for l in feat_layers}
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        batch = move_batch(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = student.ar_forward(batch, capture_layers=list(range(n_student_layers)))
        for k, v in out["hidden"].items():
            pooled = pool_prompt_segments(v, batch["n_prompt"], pool_len)
            s_acc[int(k)].append(pooled.reshape(-1, pooled.shape[-1]).float().cpu())
        for l in feat_layers:
            tf = batch["feats"][l]
            t_acc[l].append(tf.reshape(-1, tf.shape[-1]).float().cpu())

    if not t_acc[feat_layers[0]]:
        log.warning("CKA probe saw no batches; falling back to the uniform map")
        return uniform_map(feat_layers, n_student_layers)
    s_feats = {k: torch.cat(v) for k, v in s_acc.items() if v}
    t_feats = {k: torch.cat(v) for k, v in t_acc.items()}
    return cka_map(s_feats, t_feats)


def main(cfg_path: str, micro_batch: int | None = None,
         grad_accum: int | None = None):
    cfg = load_config(cfg_path)
    # Optional overrides; keep micro_batch * grad_accum constant.
    if micro_batch or grad_accum:
        was = cfg.stage1.micro_batch * cfg.stage1.grad_accum
        if micro_batch:
            cfg.raw["stage1"]["micro_batch"] = int(micro_batch)
        if grad_accum:
            cfg.raw["stage1"]["grad_accum"] = int(grad_accum)
        now = cfg.stage1.micro_batch * cfg.stage1.grad_accum
        log.info("micro_batch/grad_accum overridden -> %d x %d (effective %d)",
                 cfg.stage1.micro_batch, cfg.stage1.grad_accum, now)
        if now != was:
            log.warning("EFFECTIVE BATCH CHANGED %d -> %d; this run's gate is "
                        "not comparable to runs at the config's batch", was, now)
    torch.backends.cuda.matmul.allow_tf32 = True

    student = EdgeStudent(cfg).cuda()
    # weights_only=False: the spec pickles non-tensor objects.
    traj_spec = torch.load(Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt",
                           weights_only=False)
    student.extend_trajectory_vocab(traj_spec)
    if cfg.stage1.grad_checkpoint:
        student.enable_gradient_checkpointing()

    expert_layers = torch.load(Path(cfg.paths.cache_root) / "expert_layers.pt",
                               weights_only=False)
    # Layers present in the cache.
    feat_layers = [int(l) for l in cfg.teacher.raw["feat_layers"]]
    pool_len = int(cfg.teacher.get("feat_pool_len", 8))
    n_student_layers = student.n_layers

    # Train split only; dropout and contradiction filtering apply to train alone.
    train_ds = Stage1Dataset(cfg, student.context_builder(),
                             clip_ids=load_split(cfg, "train"),
                             image_dropout=float(cfg.stage1.get("image_dropout", 0.0)),
                             filter_contradictions=bool(
                                 cfg.stage1.get("filter_contradictions", False)),
                             prefix_noise_bins=float(cfg.stage1.get("prefix_noise_bins", 0.0)),
                             prefix_mask_prob=float(cfg.stage1.get("prefix_mask_prob", 0.0)))
    log.info("context: traj_prefix=%s route_hint=%s | train: image_dropout=%.2f "
             "filter_contradictions=%s prefix_noise_bins=%.1f prefix_mask_prob=%.2f windows=%d",
             train_ds.traj_prefix, train_ds.route_hint, train_ds.image_dropout,
             train_ds.filter_contradictions, train_ds.prefix_noise_bins,
             train_ds.prefix_mask_prob, len(train_ds))
    pad_id = student.tokenizer.pad_token_id
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    # Optional maneuver-stratified sampling.
    ms = cfg.stage1.get("maneuver_sampling")
    sampler = None
    if ms is not None and bool(ms.get("enabled", False)):
        from .data.sampling import log_mix, maneuver_weights
        alpha = float(ms.get("alpha", 0.5))
        weights, mix = maneuver_weights(train_ds.shards, alpha)
        log_mix(mix, alpha)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
    dl = DataLoader(train_ds, batch_size=cfg.stage1.micro_batch,
                    shuffle=sampler is None, sampler=sampler,
                    num_workers=n_workers, pin_memory=True,
                    collate_fn=functools.partial(collate_stage1, pad_id=pad_id))

    lmap = _build_layer_map(cfg, student, dl, feat_layers, n_student_layers, pool_len)
    # Teacher feature dim is read from the cache.
    d_teacher = _cached_feature_dim(train_ds.shards[0], feat_layers[0])
    if d_teacher != int(expert_layers.get("teacher_hidden", d_teacher)):
        log.warning("cached feature dim %d != probed teacher_hidden %d — trusting "
                    "the cache", d_teacher, expert_layers["teacher_hidden"])
    projections = FeatureProjections(
        lmap, student.hidden_size, d_teacher).cuda()
    log.info("layer map (student -> teacher): %s", dict(sorted(lmap.items())))

    groups = student.param_groups_stage1()
    groups.append({"params": list(projections.parameters()),
                   "lr_mult": cfg.student.new_token_lr_mult})
    opt = build_optimizer(groups, cfg.stage1.lr, cfg.stage1.weight_decay)
    log.info("LoRA: %s", student.lora_stats)
    log.info("%s", trainable_report(groups))

    steps_per_epoch = math.ceil(len(dl) / cfg.stage1.grad_accum)
    total_steps = steps_per_epoch * cfg.stage1.epochs
    warmup = int(total_steps * cfg.stage1.warmup_frac)

    run_best = _run_best_dir(cfg)
    log.info("checkpoints -> %s  (<stage>/best symlinks here)", run_best)

    best, patience, step = float("inf"), 0, 0
    for epoch in range(cfg.stage1.epochs):
        for i, batch in enumerate(dl):
            batch = move_batch(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = student.ar_forward(batch, capture_layers=lmap.keys())
                # Pool student states like the cached teacher features.
                proj = projections({
                    int(k): pool_prompt_segments(v, batch["n_prompt"], pool_len)
                    for k, v in out["hidden"].items()})
                w = losses.stage_weights(cfg.stage1, step, total_steps)

                n_traj = batch["traj"].shape[1]
                traj_logits, traj_tgt, traj_ok = losses.gather_targets(
                    out["logits"], batch["input_ids"], batch["traj_pos"], n_traj)
                # Size by the student's CoC length, not the teacher's.
                n_coc = int(batch["coc_pos"].sum(1).max())
                coc_logits, coc_tgt, coc_ok = losses.gather_targets(
                    out["logits"], batch["input_ids"], batch["coc_pos"], n_coc)
                # `<|cot_end|><|traj_future_start|>` boundary tokens.
                n_struct = int(batch["struct_pos"].sum(1).max())
                struct_logits, struct_tgt, struct_ok = losses.gather_targets(
                    out["logits"], batch["input_ids"], batch["struct_pos"], n_struct)
                traj_mask = batch["traj_mask"] & traj_ok
                # Interleaved stream: even = curvature, odd = accel; weighted separately.
                dim0 = torch.zeros_like(traj_mask)
                dim0[:, 0::2] = True
                curv_mask, acc_mask = traj_mask & dim0, traj_mask & ~dim0
                # Cached top-k bins are region-relative.
                topk_idx = batch["topk_idx"] + student.future_base

                # GT terms use cached GT bins, not the teacher tokens in `input_ids`.
                gt_bins = batch["gt_traj_tok"] + student.future_base
                # Per-position KL computed once, reduced under two masks; no_grad when unweighted.
                kl_on = bool(w.get(_WKEY["traj"], 0.0) or w.get(_WKEY["traj_accel"], 0.0))
                with torch.enable_grad() if kl_on else torch.no_grad():
                    traj_kl_pos = losses.traj_topk_kl_per_pos(
                        traj_logits, topk_idx, batch["topk_logp"])
                terms = {
                    "traj": losses.masked_mean(traj_kl_pos, curv_mask),
                    "traj_accel": losses.masked_mean(traj_kl_pos, acc_mask),
                    "text": losses.text_kl_or_ce(
                        coc_logits, coc_tgt, coc_ok, vocab_ok=True),
                    "struct": losses.text_kl_or_ce(
                        struct_logits, struct_tgt, struct_ok, vocab_ok=True),
                    "feat": losses.feature_match(proj, batch["feats"]),
                }
                # Distance-aware GT CE; no_grad when its weight is zero.
                gt_soft_w = w.get(_WKEY["gt_soft"], 0.0)
                with torch.enable_grad() if gt_soft_w else torch.no_grad():
                    terms["gt_soft"] = losses.gt_traj_soft_ce(
                        traj_logits, gt_bins, traj_mask,
                        sigma_bins=float(cfg.stage1.get("gt_soft_sigma_bins", 6.0)),
                        lo=student.future_base,
                        hi=student.future_base + student.n_future_bins)
                loss = sum(w.get(_WKEY[k], 0.0) * v
                           for k, v in terms.items() if k not in _DIAG)
                with torch.no_grad():   # diagnostic only
                    terms["gt"] = losses.gt_traj_ce(traj_logits, gt_bins, traj_mask)
            (loss / cfg.stage1.grad_accum).backward()
            if (i + 1) % cfg.stage1.grad_accum == 0:
                set_lr(opt, cosine_lr(step, total_steps, warmup))
                torch.nn.utils.clip_grad_norm_(
                    student.trainable_parameters() + list(projections.parameters()), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    breakdown = " ".join(f"{k} {float(v):.3f}" for k, v in terms.items())
                    # `drop` = frames zeroed by image dropout.
                    log.info("epoch %d step %d/%d loss %.4f [%s] drop %d/%d",
                             epoch, step, total_steps, loss.item(), breakdown,
                             int(batch["img_drop"].sum()), batch["img_drop"].numel())

        score = coarse_minade(cfg, student, split="challenging",
                              k=int(cfg.eval.minade_k),
                              max_windows=cfg.eval.get("gate_max_windows", None))
        # Teacher-forced CoC / struct CE on val.
        text_nll, struct_nll = _coc_nll(cfg, student,
                                        max_windows=cfg.eval.get("gate_max_windows", 64))
        gt_ce_w = losses.stage_weights(cfg.stage1, step, total_steps).get("gt_ce", 0.0)
        # Selection metric (`stage1.select_on`).
        n_gt = int(cfg.stage1.get("coc_gt_windows", 0))
        gt = _coc_gt(cfg, student, n_gt) if n_gt > 0 else None
        if gt is not None:
            log.info("epoch %d val CoC vs DRIVER (n=%d): gt_score %.3f | consistent %.3f "
                     "false_clear %.3f direction_ok %s maneuver_acc %s termination %s | mix %s",
                     epoch, gt["n"], gt["gt_score"], gt["gt_consistent"], gt["gt_false_clear"],
                     None if gt["direction_ok"] is None else f"{gt['direction_ok']:.3f}",
                     None if gt["maneuver_acc"] is None else f"{gt['maneuver_acc']:.3f}",
                     None if gt["termination"] is None else f"{gt['termination']:.3f}",
                     gt["mix"])
        sel_key = str(cfg.stage1.get("select_on", "minade"))
        # Lower is better; gt_score is negated.
        sel_metrics = {"minade": score, "coc_nll": text_nll,
                       "coc_gt": (-gt["gt_score"]) if gt is not None else None}
        if sel_metrics.get(sel_key) is None:
            raise ValueError(f"stage1.select_on={sel_key!r} needs stage1.coc_gt_windows > 0")
        if sel_key not in sel_metrics:
            raise ValueError(f"stage1.select_on must be one of {sorted(sel_metrics)}, "
                             f"got {sel_key!r}")
        sel = sel_metrics[sel_key]
        log.info("epoch %d challenging coarse-minADE %.3f m | val CoC NLL %.3f "
                 "struct NLL %.3f | gt_ce_w %.3f | select_on %s=%.3f%s",
                 epoch, score, text_nll, struct_nll, gt_ce_w, sel_key, sel,
                 "  <- new best" if sel < best else "")
        if bool(cfg.stage1.get("save_every_epoch", False)):
            # Adapters only, every epoch.
            ep_dir = run_best.parent / f"epoch-{epoch:02d}"
            checkpoint.save_adapters(student, ep_dir, stage="stage1", epoch=epoch,
                                     coarse_minade=float(score),
                                     val_coc_nll=float(text_nll),
                                     coc_gt_score=(float(gt["gt_score"]) if gt is not None else None))
            log.info("adapters -> %s", ep_dir)
        if sel < best:
            best, patience = sel, 0
            # Merged checkpoint in a per-run dir; `<stage>/best` symlinks to it.
            checkpoint.save(student, run_best, stage="stage1", epoch=epoch,
                            coarse_minade=float(score), val_coc_nll=float(text_nll),
                            select_on=sel_key,
                            **({"coc_gt_score": float(gt["gt_score"]),
                                "coc_gt_false_clear": float(gt["gt_false_clear"]),
                                "coc_gt_consistent": float(gt["gt_consistent"])}
                               if gt is not None else {}))
            torch.save(projections.state_dict(), run_best / "projections.pt")
            torch.save(lmap, run_best / "layer_map.pt")
            _promote_best(run_best)
        else:
            patience += 1
            if patience >= cfg.stage1.early_stop_patience:
                log.info("early stop: no %s improvement for %d evals", sel_key, patience)
                return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="override stage1.micro_batch (smaller card); keep "
                         "micro_batch * grad_accum at the config's value")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="override stage1.grad_accum, to hold the effective batch")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(a.config, micro_batch=a.micro_batch, grad_accum=a.grad_accum)
