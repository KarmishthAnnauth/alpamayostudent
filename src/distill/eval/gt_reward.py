"""Ground-truth-grounded scoring for the chain-of-causation."""
from __future__ import annotations
import numpy as np

from .coc_score import CLEAR, OBJECTS, objects, parse

REASONS = ("VEHICLE", "VRU", "CONSTRUCTION", "SIGNAL", "SIGN", "JUNCTION")   # things a driver reacts to
TURN_DEG = 40.0
LANE_M = 2.0


def kinematics(gt_future_xyz) -> dict:
    """What the driver did over the horizon, in a few discrete facts."""
    g = np.asarray(gt_future_xyz, dtype=float)[:, :2]
    v = np.linalg.norm(np.diff(g, axis=0), axis=1) * 10.0            # m/s at 10 Hz
    v0, v1 = float(v[:5].mean()), float(v[-10:].mean())
    dv, vmin = v1 - v0, float(v.min())
    bearing = float(np.degrees(np.arctan2(g[-1, 1], g[-1, 0])))    # + = left
    ylat = float(g[-1, 1])
    d = np.diff(g, axis=0)
    heads = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    h_end = float(np.median(heads[-5:])) if len(heads) >= 5 else float(heads[-1])
    speed = ("stops" if vmin < 0.5 else "brakes" if dv < -1.5 else
             "speeds" if dv > 1.5 else "holds")
    if abs(h_end) > TURN_DEG:
        lateral = "turn_left" if h_end > 0 else "turn_right"
    else:
        lateral = "straight"
    reacted = vmin < 0.5 or dv < -2.5 or abs(ylat) > 1.5
    return dict(v0=v0, v1=v1, dv=dv, vmin=vmin, bearing=bearing, ylat=ylat, h_end=h_end,
                speed=speed, lateral=lateral, reacted=reacted,
                braked_from_speed=(speed in ("stops", "brakes") and v0 > 3.0))


def route_hint(k: dict) -> str:
    """The direction the driver took, phrased as a route instruction for the student's `<|route_start|>...<|route_end|>` slot."""
    return {"turn_left": "Turn left ahead", "turn_right": "Turn right ahead",
            "straight": "Continue straight"}[k["lateral"]]


UNVERIFIABLE_STRICT = 0.0


def kinematic_term(man: str | None, k: dict, strict: bool = False,
                   unverifiable: float | None = None) -> float:
    """+1 consistent, -1 contradicted, 0 uncheckable."""
    s = k["speed"]
    hold = -0.5 if strict else 0.0
    if man == "STOP":        return 1.0 if s == "stops" else -1.0
    if man == "SLOW":        return 1.0 if s in ("brakes", "stops") else (-1.0 if s == "speeds" else hold)
    if man == "YIELD":       return 1.0 if s in ("brakes", "stops") else (-1.0 if s == "speeds" else hold)
    if man == "ADAPT_SPEED": return (0.5 if strict else 1.0) if s != "holds" else -0.5
    if man == "ACCELERATE":  return 1.0 if s == "speeds" else (-1.0 if s in ("brakes", "stops") else hold)
    if man == "TURN":        return 1.0 if k["lateral"].startswith("turn") else -1.0
    if man == "LANE_CHANGE": return 1.0 if abs(k["ylat"]) > LANE_M else -0.5   # a volunteered claim
    if man in ("FOLLOW", "KEEP"):
        return -1.0 if k["braked_from_speed"] else (0.0 if strict else 0.5)
    unv = (UNVERIFIABLE_STRICT if unverifiable is None else float(unverifiable)) if strict else 0.0
    if man == "NUDGE":       return unv
    return min(-0.5, unv) if strict else -0.5      # unparsed: never better than a nudge


def direction_term(man: str | None, d: str | None, k: dict) -> float:
    """Against the direction the driver took."""
    lat = k["lateral"]
    if man == "LANE_CHANGE" and d in ("left", "right"):
        if abs(k["ylat"]) < 1.5:
            return 0.0
        return 1.0 if d == ("left" if k["ylat"] > 0 else "right") else -1.0
    if lat == "straight":
        return -0.5 if (man == "TURN" and d in ("left", "right")) else 0.0
    taken = "left" if lat.endswith("left") else "right"
    if d is None:
        return -0.5
    return 1.0 if d == taken else -1.0


def hazard_term(objs: set, k: dict) -> float:
    """Naming a reason should predict a reaction."""
    named = bool(objs & set(REASONS))
    if named:
        return 1.0 if k["reacted"] else -1.0
    return -0.5 if k["reacted"] else 0.5


def gt_reward(student_text: str, terminated: bool, n_tokens: int, k: dict,
              teacher_text: str, w) -> tuple[float, dict]:
    """Scalar reward + its components. `w` = cfg.stage1_rl.reward."""
    if not terminated or n_tokens > int(w.max_tokens):
        return float(w.fail), {"fail": True}
    s = parse(student_text)
    kin = kinematic_term(s["maneuver"], k, strict=bool(getattr(w, "strict", False)),
                         unverifiable=getattr(w, "unverifiable", None))
    dr = direction_term(s["maneuver"], s["direction"], k)
    hz = hazard_term(s["objects"], k)
    t = parse(teacher_text) if teacher_text else {"maneuver": None}
    tm = 1.0 if (t["maneuver"] and s["maneuver"] == t["maneuver"]) else 0.0
    r = (float(w.kin) * kin + float(w.dir) * dr + float(w.hazard) * hz
         + float(w.teacher) * tm)
    return r, {"kin": kin, "dir": dr, "hazard": hz, "teacher": tm,
               "student_maneuver": s["maneuver"], "fail": False}


def gt_metrics(text: str, k: dict) -> dict:
    """Teacher-free evaluation fields for one trace."""
    s = parse(text)
    kt = kinematic_term(s["maneuver"], k)
    return {
        "consistent": None if kt == 0.0 else kt > 0,              # checkable claims only
        "gt_false_clear": ((s["maneuver"] in ("KEEP", "ACCELERATE") or bool(CLEAR.search(text)))
                           if k["braked_from_speed"] else None),
        "hazard_ungrounded": (not k["reacted"]) if (s["objects"] & set(REASONS)) else None,
        "direction_ok": (None if k["lateral"] == "straight" else
                         s["direction"] == ("left" if k["lateral"].endswith("left") else "right")),
        "student_maneuver": s["maneuver"],
    }


def ade_xy(pred_xyz, gt_xyz) -> float:
    """Mean L2 over the horizon, x/y only (open_loop.ade for one trajectory)."""
    p = np.asarray(pred_xyz, dtype=float)[:, :2]
    g = np.asarray(gt_xyz, dtype=float)[:, :2]
    n = min(len(p), len(g))
    return float(np.linalg.norm(p[:n] - g[:n], axis=1).mean())


def traj_reward(ade: float | None, student_text: str, terminated: bool, n_coc_tokens: int,
                k: dict, teacher_text: str, w) -> tuple[float, dict]:
    """The recipe's shape, adapted."""
    if not terminated or n_coc_tokens > int(w.max_tokens) or ade is None:
        return float(w.fail), {"fail": True}
    cap = float(w.ade_cap)
    r_ade = -min(float(ade), cap) / cap
    s = parse(student_text)
    kin = kinematic_term(s["maneuver"], k, strict=bool(getattr(w, "strict", False)),
                         unverifiable=getattr(w, "unverifiable", None))
    dr = direction_term(s["maneuver"], s["direction"], k)
    t = parse(teacher_text) if teacher_text else {"maneuver": None}
    tm = 1.0 if (t["maneuver"] and s["maneuver"] == t["maneuver"]) else 0.0
    r = (float(w.ade) * r_ade + float(w.kin) * kin + float(w.dir) * dr + float(w.teacher) * tm)
    return r, {"ade": float(ade), "r_ade": r_ade, "kin": kin, "dir": dr, "teacher": tm,
               "student_maneuver": s["maneuver"], "fail": False}


def self_consistency_term(man: str | None, direction: str | None, student_xyz) -> float:
    """Does the CoC describe the plan it precedes?"""
    ks = kinematics(student_xyz)
    return kinematic_term(man, ks, strict=True) + direction_term(man, direction, ks)


def perspan_reward(ade: float | None, student_xyz, student_text: str, terminated: bool,
                   n_coc_tokens: int, k: dict, teacher_text: str, w) -> tuple[float, float, dict]:
    """Two rewards for one joint rollout: `(r_traj, r_coc, fields)`."""
    if not terminated or n_coc_tokens > int(w.max_tokens) or ade is None or student_xyz is None:
        return float(w.fail), float(w.fail), {"fail": True}
    cap = float(w.ade_cap)
    r_ade = -min(float(ade), cap) / cap
    s = parse(student_text)
    kin = kinematic_term(s["maneuver"], k, strict=True)
    dr = direction_term(s["maneuver"], s["direction"], k)
    hz = hazard_term(s["objects"], k)
    t = parse(teacher_text) if teacher_text else {"maneuver": None}
    tm = 1.0 if (t["maneuver"] and s["maneuver"] == t["maneuver"]) else 0.0
    sc = self_consistency_term(s["maneuver"], s["direction"], student_xyz)
    w_sc = float(getattr(w, "self_consistency", 0.0) or 0.0)
    r_traj = float(w.ade) * r_ade + w_sc * sc
    r_coc = (float(w.kin) * kin + float(w.dir) * dr + float(w.hazard) * hz
             + float(w.teacher) * tm + w_sc * sc)
    return r_traj, r_coc, {"ade": float(ade), "r_ade": r_ade, "kin": kin, "dir": dr,
                           "hazard": hz, "teacher": tm, "self": sc,
                           "student_maneuver": s["maneuver"], "fail": False}
