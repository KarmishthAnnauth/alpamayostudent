"""Resuming a labeling run must not re-stream windows it is about to skip."""
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from distill.data import frames  # noqa: E402
from distill.teacher import labeler  # noqa: E402


class _Cfg(dict):
    """Duck-compatible with `config.Cfg` for the keys `run_labeling` reads."""

    def __getattr__(self, k):
        v = self[k]
        return _Cfg(v) if isinstance(v, dict) else v

    def get(self, k, default=None):
        v = dict.get(self, k, default)
        return _Cfg(v) if isinstance(v, dict) else v

    @property
    def raw(self):
        return {}


def _cfg(root, windows=2):
    return _Cfg(paths=_Cfg(cache_root=str(root)), eval=_Cfg(horizon_s=6.4),
                data=_Cfg(windows_per_clip=windows, cache_student_frames=True),
                teacher=_Cfg(flow_targets_per_window=12, topk_logits=32,
                             max_coc_tokens=48, n_traj_samples=4))


CLIPS = ["clipA", "clipB"]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Run `run_labeling` with expensive calls stubbed, counting loads and labels."""
    calls = types.SimpleNamespace(loads=[], inputs=[], labeled=[])

    def touch(p, text="x"):
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def fake_load_window(cfg, clip_id, t0_us):
        calls.loads.append((clip_id, t0_us))
        return types.SimpleNamespace(clip_id=clip_id)

    def fake_label(window, **kw):
        calls.labeled.append(window.clip_id)
        return types.SimpleNamespace()

    monkeypatch.setattr(labeler, "load_window", fake_load_window)
    monkeypatch.setattr(labeler, "save_shard", lambda path, out: touch(path, "shard"))
    monkeypatch.setattr(labeler, "TeacherWrapper",
                        lambda cfg: types.SimpleNamespace(label_window=fake_label))
    monkeypatch.setattr(frames, "save_window_input",
                        lambda p, w, quality=92: (calls.inputs.append(p), touch(p, "in"))[0])

    def run(preexisting):
        for clip_id, w_idx, kind in preexisting:
            touch(labeler.shard_path(tmp_path, clip_id, w_idx) if kind == "shard"
                  else frames.input_path(tmp_path, clip_id, w_idx))
        calls.loads.clear(), calls.inputs.clear(), calls.labeled.clear()
        labeler.run_labeling(_cfg(tmp_path), list(CLIPS))
        return calls

    return run


ALL_CACHED = [(c, w, k) for c in CLIPS for w in (0, 1) for k in ("shard", "input")]


def test_cold_cache_loads_every_window(harness):
    calls = harness([])
    assert len(calls.loads) == 4
    assert len(calls.labeled) == 4
    assert len(calls.inputs) == 4


def test_fully_cached_resume_loads_nothing(harness):
    """Fully cached resume loads nothing."""
    calls = harness(ALL_CACHED)
    assert calls.loads == []
    assert calls.labeled == []
    assert calls.inputs == []


def test_partial_resume_loads_only_the_missing_clip(harness):
    calls = harness([x for x in ALL_CACHED if x[0] == "clipA"])
    assert {c for c, _ in calls.loads} == {"clipB"}
    assert calls.labeled == ["clipB", "clipB"]


def test_missing_student_inputs_still_force_a_load(harness):
    """Shards present but student inputs absent still forces a load."""
    calls = harness([x for x in ALL_CACHED if x[2] == "shard"])
    assert len(calls.loads) == 4
    assert len(calls.inputs) == 4
    assert calls.labeled == []


def test_window_anchors_are_unchanged_by_the_rewrite():
    """`window_t0s_us` matches per-index `window_t0_us`."""
    from distill.data.preprocess import window_t0_us, window_t0s_us

    cfg = _cfg("/tmp")
    assert window_t0s_us(cfg) == [window_t0_us(cfg, i) for i in range(2)]
