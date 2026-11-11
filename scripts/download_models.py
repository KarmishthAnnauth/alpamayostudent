"""Download teacher and student weights into the configured paths."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.config import load_config  # noqa: E402

TEACHER_ID = "nvidia/Alpamayo-1.5-10B"
STUDENT_ID = "nvidia/Cosmos3-Edge"
STUDENT_REVISION = "a9d944e2"  # revision used in the thesis
PROCESSOR_ID = "nvidia/Cosmos-Reason2-8B"


def fetch(repo_id: str, local_dir: Path, revision: str | None) -> None:
    from huggingface_hub import HfApi, snapshot_download

    sha = HfApi().model_info(repo_id, revision=revision).sha
    print(f"{repo_id} @ {revision or 'main'} -> commit {sha}\n  into {local_dir}")
    snapshot_download(repo_id, revision=sha, local_dir=local_dir)
    (local_dir / "A2E_REVISION").write_text(f"{repo_id}@{sha}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["teacher", "student"])
    ap.add_argument("--teacher-revision", default=None)
    ap.add_argument("--student-revision", default=STUDENT_REVISION)
    ap.add_argument("--config", default="configs/default.yaml")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.only in (None, "teacher"):
        fetch(TEACHER_ID, Path(cfg.paths.teacher_repo), a.teacher_revision)
        # Teacher loads its tokenizer from here (into HF_HUB_CACHE).
        from huggingface_hub import snapshot_download
        snapshot_download(PROCESSOR_ID, allow_patterns=["*.json", "*.txt", "*.jinja",
                                                        "*.model", "*.tiktoken"])
    if a.only in (None, "student"):
        fetch(STUDENT_ID, Path(cfg.paths.student_repo), a.student_revision)
    print("done")


if __name__ == "__main__":
    main()
