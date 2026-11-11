"""Student coarse minADE against GT for a checkpoint, optionally with RL adapters.

    python scripts/05i_student_minade.py --adapters <step dir> --split val --n 300 --k 4 6
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, "src")
import torch                                                        # noqa: E402

from distill.config import load_config                              # noqa: E402
from distill import checkpoint                                      # noqa: E402
from distill.eval.coarse_minade import coarse_minade                # noqa: E402
from distill.student.edge_wrapper import EdgeStudent                # noqa: E402

log = logging.getLogger("student_minade")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="/data/vla/alpamayo2edge/runs/stage1/run-336/best")
    ap.add_argument("--adapters", default=None,
                    help="RL adapter dir to load on top of --ckpt (omit for the init)")
    ap.add_argument("--splits", nargs="+", default=["challenging", "val"])
    ap.add_argument("--n", type=int, default=None, help="max windows per split")
    ap.add_argument("--k", type=int, nargs="+", default=[6],
                    help="samples per window; 4 matches the teacher's cached traj_samples")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(a.config)

    student = EdgeStudent(cfg).cuda()
    student.extend_trajectory_vocab(torch.load(
        Path(cfg.paths.cache_root) / "traj_tokenizer_spec.pt", weights_only=False))
    student.lm._ensure_vision_tower()
    meta = checkpoint.load_into(student, Path(a.ckpt))
    log.info("checkpoint: %s (epoch=%s)", a.ckpt, meta.get("epoch"))
    if a.adapters:
        student.param_groups_stage1()
        am = checkpoint.load_adapters(student, a.adapters)
        log.info("adapters:   %s (step=%s)", a.adapters, am.get("step"))
    student.eval()

    res = {"ckpt": a.ckpt, "adapters": a.adapters, "runs": []}
    for split in a.splits:
        for k in a.k:
            m = coarse_minade(cfg, student, split=split, k=k, max_windows=a.n)
            log.info("  -> %s  minADE_%d = %.3f m", split, k, m)
            res["runs"].append({"split": split, "k": k, "n": a.n, "minade": float(m)})
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=2))
        log.info("-> %s", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
