"""Path resolution in distill.config: defaults, paths.yaml, env overrides and references."""
from __future__ import annotations

from pathlib import Path

import yaml

from distill.config import REPO_ROOT, load_config


def _write(dir_: Path, paths: dict, local: dict | None = None) -> Path:
    cfg = dir_ / "default.yaml"
    cfg.write_text(yaml.safe_dump({
        "paths": paths,
        "stage1_rl": {"init_ckpt": "${paths.runs_root}/stage1/best"},
    }))
    if local is not None:
        (dir_ / "paths.yaml").write_text(yaml.safe_dump({"paths": local}))
    return cfg


def test_relative_defaults_resolve_against_repo_root(tmp_path, monkeypatch):
    monkeypatch.delenv("A2E_RUNS_ROOT", raising=False)
    c = load_config(_write(tmp_path, {"runs_root": "outputs/runs"}))
    assert c.paths.runs_root == str(REPO_ROOT / "outputs/runs")
    assert c.stage1_rl.init_ckpt == str(REPO_ROOT / "outputs/runs/stage1/best")


def test_local_file_then_env_override(tmp_path, monkeypatch):
    cfg = _write(tmp_path, {"runs_root": "outputs/runs", "cache_root": "data/c"},
                 local={"runs_root": "/mnt/runs", "cache_root": "/mnt/cache"})
    monkeypatch.setenv("A2E_CACHE_ROOT", "/env/cache")
    c = load_config(cfg)
    assert c.paths.runs_root == "/mnt/runs"
    assert c.paths.cache_root == "/env/cache"
    assert c.stage1_rl.init_ckpt == "/mnt/runs/stage1/best"


def test_committed_config_has_no_machine_paths(monkeypatch):
    raw = yaml.safe_load((REPO_ROOT / "configs/default.yaml").read_text())
    for key, value in raw["paths"].items():
        assert not str(value).startswith("/"), f"paths.{key} is absolute: {value}"
