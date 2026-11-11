"""Filter windows where the teacher CoC contradicts the GT trajectory."""
from __future__ import annotations
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from ..eval import gt_reward
from ..eval.coc_score import parse

log = logging.getLogger(__name__)

CONTRADICTION = -1.0


def contradicted(coc_text: str, gt_future_xyz) -> bool:
    """True when the stated maneuver or direction is contradicted by the GT trajectory."""
    k = gt_reward.kinematics(gt_future_xyz)
    s = parse(str(coc_text))
    man, d = s.get("maneuver"), s.get("direction")
    if gt_reward.direction_term(man, d, k) <= CONTRADICTION:
        return True
    return gt_reward.kinematic_term(man, k, strict=True) <= CONTRADICTION


def route_hint(gt_future_xyz) -> str:
    """The driver's direction as route-hint text (no speed information)."""
    return gt_reward.route_hint(gt_reward.kinematics(gt_future_xyz))


def grounded_shards(shards: list[Path]) -> tuple[list[Path], dict]:
    """Drop contradicted windows; returns `(kept, stats)`."""
    kept: list[Path] = []
    dropped: Counter = Counter()
    for p in shards:
        with np.load(p, allow_pickle=True) as z:
            text = str(z["coc_text"]) if "coc_text" in z.files else ""
            xyz = z["gt_future_xyz"]
        if contradicted(text, xyz):
            dropped[parse(text).get("maneuver") or "UNPARSED"] += 1
        else:
            kept.append(p)
    stats = {"total": len(shards), "kept": len(kept),
             "dropped": sum(dropped.values()), "dropped_by_maneuver": dict(dropped)}
    return kept, stats


def log_stats(stats: dict) -> None:
    log.info("grounding filter (D-043): kept %d / %d windows, dropped %d (%.1f%%)",
             stats["kept"], stats["total"], stats["dropped"],
             100.0 * stats["dropped"] / max(stats["total"], 1))
    for m, n in sorted(stats["dropped_by_maneuver"].items(), key=lambda x: -x[1]):
        log.info("   dropped %-12s %6d", m, n)
