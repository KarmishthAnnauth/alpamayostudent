"""Maneuver-stratified window sampling with weights (1 / freq[class])**alpha."""
from __future__ import annotations
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..eval.coc_score import maneuver, split_clause

log = logging.getLogger(__name__)

UNASSIGNED = "UNASSIGNED"


def teacher_maneuver(shard: Path) -> str:
    """Maneuver class of one cached window, or UNASSIGNED."""
    with np.load(shard, allow_pickle=True) as z:
        text = str(z["coc_text"]).strip() if "coc_text" in z.files else ""
    if not text:
        return UNASSIGNED
    return maneuver(split_clause(text)[0]) or UNASSIGNED


def maneuver_weights(shards: list[Path], alpha: float) -> tuple[torch.Tensor, dict]:
    """Per-window sampling weights (mean 1.0) and the resulting class mix."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"maneuver_sampling.alpha must be in [0, 1], got {alpha}")
    labels = [teacher_maneuver(p) for p in shards]
    n = len(labels)
    counts = Counter(labels)
    assigned = {c: k for c, k in counts.items() if c != UNASSIGNED}
    if not assigned:
        raise RuntimeError("no cached window parsed to a maneuver class - is this "
                           "the right cache_root?")
    n_assigned = sum(assigned.values())
    class_w = {c: (n_assigned / k) ** alpha for c, k in assigned.items()}
    # Unassigned windows get the mean assigned weight.
    class_w[UNASSIGNED] = sum(class_w[c] * assigned[c] for c in assigned) / n_assigned
    w = torch.tensor([class_w[l] for l in labels], dtype=torch.float64)
    w = w / w.mean()
    total = float(sum(class_w[c] * counts[c] for c in counts))
    mix = {c: (counts[c], counts[c] / n, class_w[c] * counts[c] / total)
           for c in sorted(counts, key=lambda c: -counts[c])}
    return w.float(), mix


def log_mix(mix: dict, alpha: float) -> None:
    log.info("maneuver-stratified sampling, alpha=%.2f (D-036):", alpha)
    log.info("   %-13s %6s %8s %10s", "class", "n", "uniform", "effective")
    for c, (k, u, e) in mix.items():
        log.info("   %-13s %6d %7.1f%% %9.1f%%", c, k, 100 * u, 100 * e)
