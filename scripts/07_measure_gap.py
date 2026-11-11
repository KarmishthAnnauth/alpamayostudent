"""Measure ADE headroom: Alpamayo 1.5 teacher vs zero-shot Cosmos 3 Edge on the same clips.

    python scripts/07_measure_gap.py calibrate --clips 5
    python scripts/07_measure_gap.py teacher --clips 50
    python scripts/07_measure_gap.py edge    --clips 50
    python scripts/07_measure_gap.py report
"""
import argparse, json, logging, sys
from pathlib import Path
sys.path.insert(0, "src")
import torch
from distill.config import load_config
from distill.eval import gap

ap = argparse.ArgumentParser()
ap.add_argument("phase", choices=["calibrate", "teacher", "edge", "report"])
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--split", default="challenging",
                help="challenging | holdout_geo | val, read from "
                     "<cache_root>/split_<name>.json if it exists")
ap.add_argument("--clips-file", default=None,
                help="explicit JSON list of clip ids; overrides --split. Use "
                     "<cache_root>/curated_500.json until the split files exist")
ap.add_argument("--clips", type=int, default=50,
                help="how many clips of the split to score")
ap.add_argument("--samples", type=int, default=6,
                help="teacher trajectory samples per window (minADE_k)")
ap.add_argument("--num-steps", type=int, default=35, help="Edge diffusion steps")
ap.add_argument("--out", default=None, help="defaults to <runs_root>/gap")
a = ap.parse_args()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
cfg = load_config(a.config)
out = Path(a.out or (Path(cfg.paths.runs_root) / "gap"))
out.mkdir(parents=True, exist_ok=True)
horizon_steps = int(round(cfg.eval.horizon_s * 10))     # 10 Hz

if a.phase == "report":
    t_file, e_file = out / "teacher.npz", out / "edge.npz"
    missing = [str(f) for f in (t_file, e_file) if not f.exists()]
    if missing:
        sys.exit("run the teacher and edge phases first; missing: " + ", ".join(missing))
    print(gap.report(gap.load(t_file), gap.load(e_file)))
    cal = out / "calibration.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        print(f"\n(edge action decode calibrated at {c['mean_ade']:.3f} m mean ADE "
              f"on {len(c['keys'])} windows)")
    else:
        print("\nWARNING: no calibration.json — the Edge number rests on an "
              "unverified 9-D action layout. Run the calibrate phase.")
    sys.exit(0)

def resolve_clips():
    """Explicit list > split file > the raw clip index."""
    if a.clips_file:
        ids = json.loads(Path(a.clips_file).read_text())
        print(f"clips: {len(ids)} from {a.clips_file}")
        return ids
    split_file = Path(cfg.paths.cache_root) / f"split_{a.split}.json"
    if split_file.exists():
        ids = json.loads(split_file.read_text())
        print(f"clips: {len(ids)} from split '{a.split}'")
        return ids
    from distill.data.preprocess import list_clip_ids
    ids = list_clip_ids(cfg)
    print(f"clips: no split_{a.split}.json and no --clips-file; using the raw "
          f"clip index ({len(ids)} clips). Both phases MUST use the same "
          f"selection — pass --clips-file to pin it if that is a concern.")
    return ids


clips = resolve_clips()[: a.clips]
print(f"{a.phase}: {len(clips)} clips x {cfg.data.windows_per_clip} windows, "
      f"horizon {horizon_steps} steps")

if a.phase == "calibrate":
    res = gap.calibrate(cfg, clips[: max(1, a.clips)], num_steps=a.num_steps)
    (out / "calibration.json").write_text(json.dumps(res, indent=2))
    print(f"\ninverse-dynamics recovery of the KNOWN past: "
          f"{res['mean_ade']:.3f} m mean ADE over {len(res['keys'])} windows")
    if res["mean_ade"] > 2.0:
        print("\nThat is too large to trust. The 9-D 'av' action layout assumed by "
              "gap.av_actions_to_xyz is probably wrong (or the actions need "
              "denormalizing). Fix that before reading anything into the edge phase.")
    else:
        print("\nDecode looks sane — proceed to the teacher and edge phases.")
    sys.exit(0)

if a.phase == "teacher":
    blob = gap.teacher_predictions(cfg, clips, n_samples=a.samples)
    gap.save(blob, out / "teacher.npz")
else:
    blob = gap.edge_predictions(cfg, clips, horizon_steps=horizon_steps,
                                num_steps=a.num_steps)
    gap.save(blob, out / "edge.npz")
print(gap.score(blob))
print(f"\nwrote {out}/{a.phase}.npz — run the other phase, then `report`")
