"""Typed access to the YAML config, with machine-specific path overrides."""
from __future__ import annotations
import dataclasses
import os
import re
from pathlib import Path
from typing import Any
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_PATHS_FILE = "paths.yaml"
ENV_PREFIX = "A2E_"
_REF = re.compile(r"\$\{paths\.([A-Za-z0-9_]+)\}")


@dataclasses.dataclass
class Cfg:
    raw: dict[str, Any]

    def __getattr__(self, key: str) -> Any:
        try:
            v = self.raw[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Cfg(v) if isinstance(v, dict) else v

    def get(self, key: str, default: Any = None) -> Any:
        v = self.raw.get(key, default)
        return Cfg(v) if isinstance(v, dict) else v


def _resolve_path(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    p = Path(os.path.expandvars(os.path.expanduser(value)))
    return str(p if p.is_absolute() else (REPO_ROOT / p))


def _substitute(node: Any, paths: dict[str, Any]) -> Any:
    if isinstance(node, dict):
        return {k: _substitute(v, paths) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, paths) for v in node]
    if isinstance(node, str) and "${paths." in node:
        def repl(m: re.Match) -> str:
            key = m.group(1)
            if key not in paths:
                raise KeyError(f"config refers to ${{paths.{key}}}, which is not defined")
            return str(paths[key])
        return _resolve_path(_REF.sub(repl, node))
    return node


def resolve_paths(raw: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    paths = dict(raw.get("paths") or {})
    local = config_dir / LOCAL_PATHS_FILE
    if local.exists():
        with open(local) as f:
            override = yaml.safe_load(f) or {}
        override = override.get("paths", override)   # accept either layout
        paths.update(override)
    for key in list(paths):
        env = os.environ.get(ENV_PREFIX + key.upper())
        if env:
            paths[key] = env
    paths = {k: _resolve_path(v) for k, v in paths.items()}
    out = {k: v for k, v in raw.items() if k != "paths"}
    out = _substitute(out, paths)
    out["paths"] = paths
    return out


def load_config(path: str | Path = "configs/default.yaml") -> Cfg:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Cfg(resolve_paths(raw, path.resolve().parent))
