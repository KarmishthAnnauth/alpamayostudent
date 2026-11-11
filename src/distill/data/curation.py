"""Seeded, stratified clip selection with a geographic holdout."""
from __future__ import annotations
import random

from .preprocess import clip_metadata


def stratum_of(meta: dict, teacher_labels: dict | None) -> str:
    """Map a clip to the first matching stratum name from the config weights."""
    del teacher_labels
    tags = {str(t).lower() for t in meta.get("tags", [])}
    weather = str(meta.get("weather", "")).lower()
    tod = str(meta.get("time_of_day", "")).lower()
    if "construction" in tags:
        return "construction"
    if "pedestrian" in tags or "crosswalk" in tags:
        return "pedestrian"
    if "intersection" in tags or "junction" in tags:
        return "intersection"
    if weather in {"rain", "snow", "fog", "heavy_rain"}:
        return "adverse_weather"
    if tod in {"night", "dusk", "dawn"}:
        return "night"
    return "default"


def curate(cfg, all_clip_ids: list[str], n: int,
           teacher_labels: dict[str, dict] | None = None,
           seed: int = 0) -> list[str]:
    holdout = set(cfg.data.split.raw["holdout_countries"])
    weights = cfg.data.curation.strata_weights.raw
    rng = random.Random(seed)

    weighted: list[tuple[str, float]] = []
    for cid in all_clip_ids:
        meta = clip_metadata(cfg, cid)
        if meta.get("country") in holdout:
            continue  # geographic holdout
        s = stratum_of(meta, (teacher_labels or {}).get(cid))
        weighted.append((cid, float(weights.get(s, weights["default"]))))

    # Deterministic weighted sampling without replacement (exponential-sort trick).
    def key(item):
        cid, w = item
        r = random.Random(f"{seed}:{cid}").random()
        return -(r ** (1.0 / max(w, 1e-6)))  # larger w -> more likely selected

    ranked = sorted(weighted, key=key)
    chosen = [cid for cid, _ in ranked[:n]]
    rng.shuffle(chosen)
    return chosen
