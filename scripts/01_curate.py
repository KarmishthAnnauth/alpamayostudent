"""Select N curated clips and write the train/val/challenging splits."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, "src")
from distill.config import load_config
from distill.data.curation import curate
from distill.data.preprocess import list_clip_ids
from distill.data.splits import write_splits

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, required=True, help="number of clips, e.g. 500/2000/5000")
a = ap.parse_args()

cfg = load_config()
all_ids = list_clip_ids(cfg)
labels_p = Path(cfg.paths.cache_root) / "teacher_labels_index.json"
labels = json.load(open(labels_p)) if labels_p.exists() else None
chosen = curate(cfg, all_ids, a.n, teacher_labels=labels, seed=0)
out = Path(cfg.paths.cache_root) / f"curated_{a.n}.json"
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(chosen, open(out, "w"))
print(f"wrote {len(chosen)} clip ids -> {out}")

for name, ids in write_splits(cfg, chosen).items():
    print(f"  split_{name}.json: {len(ids)} clips")
