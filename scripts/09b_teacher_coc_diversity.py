"""Sample the teacher's CoC K times per window to measure per-scene maneuver diversity."""
from __future__ import annotations
import argparse
import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
import numpy as np                                                  # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill.data.preprocess import iter_windows                    # noqa: E402
from distill.data.splits import load_split                          # noqa: E402
from distill.eval.coc_score import maneuver, split_clause, direction  # noqa: E402

log = logging.getLogger("teacher_diversity")
COMMITTED = {"NUDGE", "ADAPT_SPEED", "LANE_CHANGE", "TURN", "SLOW", "ACCELERATE", "YIELD", "STOP"}


def classes(text: str) -> tuple[str | None, str | None]:
    act, _ = split_clause(text.strip())
    return maneuver(act), direction(act)


def entropy(c: Counter) -> float:
    n = sum(c.values())
    return -sum(v / n * math.log(v / n) for v in c.values() if v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--clips", type=int, default=20, help="clips (x windows_per_clip windows)")
    ap.add_argument("--k", type=int, default=8, help="CoC samples per window")
    ap.add_argument("--out", default="/data/vla/alpamayo2edge/runs/teacher_coc_diversity.jsonl")
    ap.add_argument("--cameras", default=None,
                    help="comma-separated camera list for the TEACHER. The cache was labelled "
                         "with the 4-camera set; run 5 switched data.cameras to 1 camera, and "
                         "the teacher's window loader follows data.cameras - so without this "
                         "override the probe measures 1-cam-vs-4-cam disagreement, not sampling "
                         "diversity.")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)
    if a.cameras:
        cfg.raw["data"]["cameras"] = [c.strip() for c in a.cameras.split(",") if c.strip()]
    log.info("teacher cameras: %s", list(cfg.data.raw["cameras"]))
    cache_root = Path(cfg.paths.cache_root)

    from distill.teacher.wrapper import TeacherWrapper
    tw = TeacherWrapper(cfg)
    clip_ids = load_split(cfg, a.split)[:a.clips]
    log.info("teacher CoC diversity: %d clips from %s, K=%d, T=%.2f top_p=%.2f",
             len(clip_ids), a.split, a.k, float(cfg.teacher.get("gen_temperature", 1.0)),
             float(cfg.teacher.get("gen_top_p", 1.0)))

    rows = []
    with open(a.out, "w") as f:
        for ci, clip_id in enumerate(clip_ids):
            for w_idx, window in iter_windows(cfg, clip_id):
                shard = cache_root / clip_id / f"{w_idx:02d}.npz"
                cached = ""
                if shard.exists():
                    with np.load(shard, allow_pickle=True) as z:
                        cached = str(z["coc_text"]).strip()
                samples = []
                for _ in range(a.k):
                    out = tw.label_window(window, k_flow=1, topk=1,
                                          max_coc=int(cfg.teacher.max_coc_tokens),
                                          n_traj_samples=1)
                    samples.append(out.coc_text.strip())
                man = Counter(classes(s)[0] for s in samples)
                cm, cd = classes(cached) if cached else (None, None)
                row = {"clip": clip_id, "w": w_idx, "cached": cached, "cached_maneuver": cm,
                       "samples": samples, "maneuvers": dict(man),
                       "majority": man.most_common(1)[0][0], "majority_share": man.most_common(1)[0][1] / a.k,
                       "entropy": entropy(man),
                       "agree_with_cached": (man[cm] / a.k) if cm else None}
                rows.append(row)
                f.write(json.dumps(row) + "\n"); f.flush()
                log.info("[%d/%d] %s/%02d cached=%-12s majority=%-12s share=%.2f H=%.2f agree=%s",
                         ci + 1, len(clip_ids), clip_id[:8], w_idx, cm, row["majority"],
                         row["majority_share"], row["entropy"],
                         f"{row['agree_with_cached']:.2f}" if cm else "-")

    n = len(rows)
    split_scenes = sum(1 for r in rows if r["majority_share"] < 0.75)
    log.info("\nwindows: %d   K=%d", n, a.k)
    log.info("  mean majority share      %.3f", sum(r["majority_share"] for r in rows) / n)
    log.info("  mean entropy (nats)      %.3f   (0 = deterministic; ln2=0.69 = 50/50)",
             sum(r["entropy"] for r in rows) / n)
    log.info("  split scenes (<75%% maj)  %d / %d = %.1f%%", split_scenes, n, 100 * split_scenes / n)
    com = [r for r in rows if r["cached_maneuver"] in COMMITTED]
    pas = [r for r in rows if r["cached_maneuver"] in ("FOLLOW", "KEEP")]
    if com:
        log.info("  cached label COMMITTED (n=%d): fresh samples agree %.2f of the time; "
                 "samples say FOLLOW/KEEP %.2f", len(com),
                 sum(r["agree_with_cached"] for r in com) / len(com),
                 sum((r["maneuvers"].get("FOLLOW", 0) + r["maneuvers"].get("KEEP", 0)) / a.k
                     for r in com) / len(com))
    if pas:
        log.info("  cached label FOLLOW/KEEP (n=%d): fresh samples agree %.2f", len(pas),
                 sum(r["agree_with_cached"] for r in pas) / len(pas))
    log.info("-> %s", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
