"""Write the train / val / challenging splits over a curated increment."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from distill.config import load_config          # noqa: E402
from distill.data.splits import write_splits    # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--n", type=int, required=True, help="increment size, e.g. 500")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

cfg = load_config()
curated = Path(cfg.paths.cache_root) / f"curated_{a.n}.json"
clips = json.loads(curated.read_text())
splits = write_splits(cfg, clips, seed=a.seed)
for name, ids in splits.items():
    print(f"split_{name}.json: {len(ids)} clips")
print(f"-> {cfg.paths.cache_root}")
