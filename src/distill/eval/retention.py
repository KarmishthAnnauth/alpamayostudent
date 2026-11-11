"""Retention: text drift on non-AV probes and per-tower weight drift vs the base checkpoint."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

BASELINE_NAME = "retention_baseline.pt"
TOPK = 32


def load_probes(path: str | Path) -> list[dict]:
    blob = json.loads(Path(path).read_text())
    return blob["probes"]


def _encode(student, prompt: str) -> torch.Tensor:
    """Prompt -> [1, T] ids, through the chat template when the tokenizer has one."""
    tok = student.tokenizer
    if getattr(tok, "chat_template", None):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt")
    else:
        ids = tok(prompt, return_tensors="pt").input_ids
    return ids.to(student._device)


def _logits(student, ids: torch.Tensor) -> torch.Tensor:
    """Teacher-forced reasoner logits [1, T, V] for a full id sequence."""
    hidden = student.lm.model.reasoner_forward(input_ids=ids, cache=None)
    return student.lm.lm_head(hidden)


@torch.no_grad()
def build_baseline(student, cfg) -> dict:
    """Run the probes on the UNTRAINED student. Do this once, before stage 1."""
    rcfg = cfg.eval.retention
    probes = load_probes(rcfg.probe_file)
    tok = student.tokenizer
    entries = []
    for p in probes:
        prompt_ids = _encode(student, p["prompt"])
        cont = student.lm.generate_reasoner_text(
            prompt_ids,
            max_new_tokens=int(rcfg.max_new_tokens),
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
            return_only_new_tokens=True,
        )
        if cont.shape[1] == 0:
            log.warning("probe %s produced no continuation; skipped", p["id"])
            continue
        full = torch.cat([prompt_ids, cont], dim=1)
        start = prompt_ids.shape[1] - 1
        logp = F.log_softmax(_logits(student, full)[0, start:-1].float(), dim=-1)
        topk_logp, topk_idx = logp.topk(TOPK, dim=-1)
        tgt = cont[0]
        entries.append({
            "id": p["id"], "category": p.get("category", "?"),
            "prompt_ids": prompt_ids.cpu(), "cont_ids": cont.cpu(),
            "topk_idx": topk_idx.cpu(), "topk_logp": topk_logp.cpu(),
            "base_nll": float(-logp.gather(-1, tgt[:, None]).mean()),
        })
    return {"topk": TOPK, "max_new_tokens": int(rcfg.max_new_tokens),
            "probe_file": str(rcfg.probe_file), "entries": entries}


def save_baseline(blob: dict, path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(blob, path / BASELINE_NAME)
    log.info("retention baseline (%d probes) -> %s", len(blob["entries"]), path)
    return path / BASELINE_NAME


def load_baseline(path: str | Path) -> dict:
    f = Path(path) / BASELINE_NAME
    if not f.exists():
        raise FileNotFoundError(
            f"{f} missing — run `scripts/06_retention.py baseline` on the "
            "untrained checkpoint BEFORE training, not after.")
    return torch.load(f)


@torch.no_grad()
def text_drift(student, baseline: dict) -> dict:
    """Score the current student against a cached baseline. Higher = more drift."""
    per_probe, per_cat = [], {}
    for e in baseline["entries"]:
        prompt_ids = e["prompt_ids"].to(student._device)
        cont = e["cont_ids"].to(student._device)
        full = torch.cat([prompt_ids, cont], dim=1)
        start = prompt_ids.shape[1] - 1
        logp = F.log_softmax(_logits(student, full)[0, start:-1].float(), dim=-1)

        tgt = cont[0]
        nll = float(-logp.gather(-1, tgt[:, None]).mean())
        agree = float((logp.argmax(-1) == tgt).float().mean())

        # top-k truncated KL
        bidx = e["topk_idx"].to(student._device)
        blogp = e["topk_logp"].to(student._device).float()
        bp = F.softmax(blogp, dim=-1)
        slogp = logp.gather(-1, bidx)
        slogp = slogp - slogp.logsumexp(-1, keepdim=True)
        kl = float((bp * (F.log_softmax(blogp, dim=-1) - slogp)).sum(-1).mean())

        row = {"id": e["id"], "category": e["category"], "nll": nll,
               "base_nll": e["base_nll"], "d_nll": nll - e["base_nll"],
               "kl": kl, "agree": agree}
        per_probe.append(row)
        per_cat.setdefault(e["category"], []).append(row)

    def _mean(rows, k):
        return sum(r[k] for r in rows) / max(len(rows), 1)

    return {
        "d_nll": _mean(per_probe, "d_nll"),
        "kl": _mean(per_probe, "kl"),
        "agree": _mean(per_probe, "agree"),
        "by_category": {c: {"d_nll": _mean(r, "d_nll"), "kl": _mean(r, "kl"),
                            "agree": _mean(r, "agree")} for c, r in per_cat.items()},
        "per_probe": per_probe,
    }


@torch.no_grad()
def weight_drift(base_sd: dict, trained_sd: dict) -> dict:
    """Relative Frobenius drift per parameter group, base vs a merged checkpoint."""
    groups: dict[str, list[float]] = {}
    for k, w_base in base_sd.items():
        w_new = trained_sd.get(k)
        if w_new is None or not torch.is_floating_point(w_base):
            continue
        w_new = w_new.detach().to(w_base.device)
        if w_new.shape != w_base.shape:
            # extended table: pretrained rows only
            n = min(w_new.shape[0], w_base.shape[0])
            w_base, w_new = w_base[:n], w_new[:n]
        denom = w_base.float().norm().item()
        if denom == 0:
            continue
        rel = (w_new.float() - w_base.float()).norm().item() / denom
        if "_moe_gen" in k:
            g = "gen_tower"
        elif "embed_tokens" in k or "lm_head" in k:
            g = "shared_tables"
        elif "action2llm" in k or "llm2action" in k:
            g = "action_head"
        elif ".layers." in k:
            g = "ar_tower"
        else:
            g = "other"
        groups.setdefault(g, []).append(rel)
    return {g: {"mean_rel_drift": sum(v) / len(v), "max_rel_drift": max(v),
                "tensors": len(v)} for g, v in sorted(groups.items())}


def format_report(text: dict | None, weights: dict | None) -> str:
    lines = ["=== retention ==="]
    if text:
        lines += [
            f"reasoner breadth over {len(text['per_probe'])} non-AV probes:",
            f"  dNLL  {text['d_nll']:+.4f}   (0 = base answers just as likely)",
            f"  KL    {text['kl']:.4f}",
            f"  agree {text['agree']:.3f}",
            "  by category:",
        ]
        for c, v in sorted(text["by_category"].items()):
            lines.append(f"    {c:22s} dNLL {v['d_nll']:+.4f}  KL {v['kl']:.4f}  "
                         f"agree {v['agree']:.3f}")
    if weights:
        lines.append("relative weight drift (Frobenius, vs base checkpoint):")
        for g, v in weights.items():
            lines.append(f"    {g:22s} mean {v['mean_rel_drift']:.5f}  "
                         f"max {v['max_rel_drift']:.5f}  ({v['tensors']} tensors)")
    lines.append("NOTE: gen-tower denoising loss on generic clips is not wired "
                 "yet — see the module docstring.")
    return "\n".join(lines)
