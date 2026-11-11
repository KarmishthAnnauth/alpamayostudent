"""Structured scoring of chain-of-causation text: maneuver, direction, objects."""
from __future__ import annotations
import re

CONNECTIVE = re.compile(
    r"\b(since|due to|because of|because|as|to avoid|in order to|so that|"
    r"while|after|before|for)\b", re.I)

# First match wins.
MANEUVER_RULES: list[tuple[str, str]] = [
    ("STOP",        r"\b(stop|halt|come to a stop|wait)\b"),
    ("YIELD",       r"\b(yield|give way)\b"),
    ("FOLLOW",      r"\b(keep|maintain|create|increase)\s+(a\s+)?(safe\s+)?(distance|gap|following)"
                    r"|\bfollow\b|\bgap-search\b"),
    ("LANE_CHANGE", r"\b(lane change|change lanes?|merge|move (in)?to the (left|right))\b"),
    ("TURN",        r"\b(turn|make a (left|right))\b"),
    ("NUDGE",       r"\b(nudge|shift|drift|steer|edge)\b"),
    ("ACCELERATE",  r"\b(accelerate|speed up|resume|increase speed)\b"),
    ("SLOW",        r"\b(slow|decelerate|brake|reduce speed|ease off)\b"),
    ("ADAPT_SPEED", r"\b(adapt|adjust|match)\b"),
    ("KEEP",        r"\b(keep|maintain|remain|stay|continue|proceed|split)\b"),
]

DIRECTIONS = ("left", "right", "straight")

OBJECTS: dict[str, str] = {
    "VEHICLE":      r"\b(vehicles?|cars?|trucks?|buses|bus|vans?|traffic)\b",
    "VRU":          r"\b(pedestrians?|cyclists?|bikes?|bicycles?|motorcycles?|scooters?)\b",
    "SIGNAL":       r"\b(traffic light|lights?|signals?|signalized)\b",
    "SIGN":         r"\b(signs?|stop sign)\b",
    "CONSTRUCTION": r"\b(cones?|barriers?|barricades?|construction|roadworks?)\b",
    "JUNCTION":     r"\b(intersections?|junctions?|crosswalks?|roundabouts?)\b",
}
HAZARDS = ("VEHICLE", "VRU", "CONSTRUCTION")
CLEAR = re.compile(r"\b(clear|empty|unobstructed|no (other )?(vehicles?|traffic|obstacles?))\b", re.I)
PROCEEDING = ("KEEP", "ACCELERATE")


def split_clause(text: str) -> tuple[str, str]:
    """`<action>` before the first causal connective, `<cause>` after it."""
    m = CONNECTIVE.search(text)
    if not m:
        return text.strip(), ""
    return text[:m.start()].strip(), text[m.end():].strip()


def maneuver(action: str) -> str | None:
    for name, pat in MANEUVER_RULES:
        if re.search(pat, action, re.I):
            return name
    return None


def direction(action: str) -> str | None:
    """Direction of the ego maneuver, read from the action clause only."""
    for d in DIRECTIONS:
        if re.search(rf"\b{d}\b", action, re.I):
            return d
    return None


def objects(text: str) -> set[str]:
    return {c for c, pat in OBJECTS.items() if re.search(pat, text, re.I)}


def parse(text: str) -> dict:
    action, cause = split_clause(text.strip())
    return {"text": text.strip(), "action": action, "cause": cause,
            "maneuver": maneuver(action), "direction": direction(action),
            "objects": objects(text)}


def token_f1(a: str, b: str) -> float:
    """Bag-of-words F1."""
    ta = re.findall(r"[a-z]+", a.lower())
    tb = re.findall(r"[a-z]+", b.lower())
    if not ta or not tb:
        return 0.0
    from collections import Counter
    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if not overlap:
        return 0.0
    p, r = overlap / len(ta), overlap / len(tb)
    return 2 * p * r / (p + r)


def score_pair(student: str, teacher: str) -> dict | None:
    """Score one (student, teacher) CoC pair; None if the teacher trace is empty."""
    if not teacher.strip():
        return None
    s, t = parse(student), parse(teacher)
    s_obj, t_obj = s["objects"], t["objects"]
    inter = s_obj & t_obj
    t_haz = t_obj & set(HAZARDS)
    return {
        "maneuver_match": (s["maneuver"] == t["maneuver"]) if t["maneuver"] else None,
        "student_maneuver": s["maneuver"], "teacher_maneuver": t["maneuver"],
        "direction_match": (s["direction"] == t["direction"]) if t["direction"] else None,
        "obj_recall": len(inter) / len(t_obj) if t_obj else None,
        "obj_precision": len(inter) / len(s_obj) if s_obj else None,
        # teacher saw a hazard, student named none and carries on
        "false_clear": bool(t_haz) and not (s_obj & set(HAZARDS))
                       and (s["maneuver"] in PROCEEDING or bool(CLEAR.search(student))),
        "hazard_window": bool(t_haz),
        "token_f1": token_f1(student, teacher),
        "parsed": s["maneuver"] is not None and t["maneuver"] is not None,
    }
