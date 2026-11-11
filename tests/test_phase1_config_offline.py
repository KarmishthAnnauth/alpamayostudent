"""The phase-1 recipe as configured."""
from pathlib import Path

import yaml

CFG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_phase1_recipe_is_the_run8_recipe():
    c = yaml.safe_load(CFG.read_text())
    s = c["stage1"]
    assert s["traj_prefix"] == "gt"
    assert s["route_hint"] is True
    assert s["filter_contradictions"] is True
    assert s["prefix_mask_prob"] > 0 and s["prefix_noise_bins"] >= 64
    assert s["image_dropout"] == 0.0
    assert s["select_on"] == "coc_gt" and s["coc_gt_windows"] >= 100
    assert s["save_every_epoch"] is True
    assert s["loss_weights"]["gt_ce"] > 0 and s["loss_weights"]["traj_kl"] == 0
    assert s["loss_weights"]["text_kl"] > 0 and s["loss_weights"]["struct_ce"] > 0
    assert s["maneuver_sampling"]["enabled"] is False
    assert len(c["data"]["cameras"]) == 4


def test_phase15_recipe_has_both_hedges_charged():
    r = yaml.safe_load(CFG.read_text())["stage1_rl"]["reward"]
    assert r["mode"] == "gt" and r["strict"] is True
    assert r["hazard"] == 0 and r["teacher"] == 0
    assert r["unverifiable"] < 0
