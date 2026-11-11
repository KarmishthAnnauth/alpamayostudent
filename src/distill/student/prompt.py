"""Student context format, mirroring the teacher's input (cameras, ego history, route, instruction)."""
from __future__ import annotations

import dataclasses
from typing import Callable, Literal

# verbatim from the teacher
SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."

INSTRUCTION = (
    "output the chain-of-thought reasoning of the driving process, "
    "then output the future trajectory."
)

N_HISTORY_SLOTS = 48
N_FUTURE_TOKENS = 128

CAMERA_INDEX = {
    "camera_cross_left_120fov": 0,
    "camera_front_wide_120fov": 1,
    "camera_cross_right_120fov": 2,
    "camera_rear_left_70fov": 3,
    "camera_rear_tele_30fov": 4,
    "camera_rear_right_70fov": 5,
    "camera_front_tele_30fov": 6,
}
CAMERA_DISPLAY_NAME = {
    "camera_cross_left_120fov": "Front left camera",
    "camera_cross_right_120fov": "Front right camera",
    "camera_front_wide_120fov": "Front camera",
    "camera_front_tele_30fov": "Front telephoto camera",
    "camera_rear_left_70fov": "Rear left camera",
    "camera_rear_tele_30fov": "Rear camera",
    "camera_rear_right_70fov": "Rear right camera",
}

SPECIAL_TOKENS = [
    "<|traj_history_start|>", "<|traj_history|>", "<|traj_history_end|>",
    "<|route_start|>", "<|route_end|>",
    "<|cot_start|>", "<|cot_end|>",
    "<|traj_future_start|>", "<|traj_future_end|>",
]

SegmentKind = Literal["text", "image", "slots", "bins"]


@dataclasses.dataclass(frozen=True)
class Segment:
    """One piece of the context: text, image, slots (ego history) or bins (trajectory)."""
    kind: SegmentKind
    text: str = ""
    count: int = 0
    values: tuple[int, ...] | None = None


def user_segments(camera_names: list[str], n_frames: int,
                  nav_text: str | None = None,
                  hist_bins: list[int] | None = None) -> list[Segment]:
    """Cameras (sorted by index) -> ego-history slots -> [route] -> instruction."""
    unknown = [c for c in camera_names if c not in CAMERA_INDEX]
    if unknown:
        raise ValueError(f"unknown camera(s) {unknown}; expected {sorted(CAMERA_INDEX)}")
    segs: list[Segment] = []
    for name in sorted(camera_names, key=lambda c: CAMERA_INDEX[c]):
        segs.append(Segment("text", text=f"{CAMERA_DISPLAY_NAME[name]}: "))
        for f in range(n_frames):
            segs.append(Segment("text", text=f"frame {f} "))
            segs.append(Segment("image"))
    segs.append(Segment("text", text="<|traj_history_start|>"))
    segs.append(Segment("slots", text="<|traj_history|>", count=N_HISTORY_SLOTS,
                        values=tuple(hist_bins) if hist_bins is not None else None))
    segs.append(Segment("text", text="<|traj_history_end|>"))
    if nav_text:
        segs.append(Segment("text", text=f"<|route_start|>{nav_text}<|route_end|>"))
    segs.append(Segment("text", text=INSTRUCTION))
    return segs


def assistant_segments(coc_text: str | None = None,
                       traj_bins: list[int] | None = None,
                       n_future: int = N_FUTURE_TOKENS,
                       for_generation: bool = False) -> list[Segment]:
    """Assistant turn; stops after `<|cot_start|>` or `<|traj_future_start|>` for generation."""
    segs = [Segment("text", text="<|cot_start|>")]
    if coc_text is None:
        return segs
    segs.append(Segment("text", text=coc_text))
    segs.append(Segment("text", text="<|cot_end|>"))
    segs.append(Segment("text", text="<|traj_future_start|>"))
    if for_generation:
        return segs
    segs.append(Segment("bins", count=n_future,
                        values=tuple(traj_bins) if traj_bins is not None else None))
    segs.append(Segment("text", text="<|traj_future_end|>"))
    return segs


@dataclasses.dataclass
class Assembled:
    input_ids: list[int]
    history_span: tuple[int, int]          # [start, end) — where ego motion is written
    image_spans: list[tuple[int, int]]     # one per frame, in prompt order
    coc_span: tuple[int, int] | None       # text between cot_start/cot_end
    struct_span: tuple[int, int] | None    # `<|cot_end|><|traj_future_start|>` tokens
    traj_span: tuple[int, int] | None      # the 128 bin positions
    n_prompt: int                          # ids before the assistant turn


def assemble(
    segments: list[Segment],
    encode: Callable[[str], list[int]],
    special_id: Callable[[str], int],
    bin_id: Callable[[int], int],
    image_tokens: Callable[[int], int],
    image_token_id: int,
    hist_bin_id: Callable[[int], int] | None = None,
    n_prompt: int | None = None,
) -> Assembled:
    """Segments -> ids + the spans the trainer needs."""
    ids: list[int] = []
    history_span = (0, 0)
    image_spans: list[tuple[int, int]] = []
    coc_span = struct_span = traj_span = None
    pending_coc_start: int | None = None
    pending_struct_start: int | None = None

    for seg in segments:
        if seg.kind == "text":
            start = len(ids)
            ids.extend(encode(seg.text))
            if seg.text == "<|cot_start|>":
                pending_coc_start = len(ids)
            elif seg.text == "<|cot_end|>" and pending_coc_start is not None:
                coc_span = (pending_coc_start, start)
                pending_struct_start = start            # first `<|cot_end|>` subword
            elif seg.text == "<|traj_future_start|>" and pending_struct_start is not None:
                struct_span = (pending_struct_start, len(ids))   # through the last subword
        elif seg.kind == "slots":
            start = len(ids)
            if seg.values is None:
                ids.extend([special_id(seg.text)] * seg.count)
            else:
                if len(seg.values) != seg.count:
                    raise ValueError(
                        f"expected {seg.count} history bins, got {len(seg.values)}")
                if hist_bin_id is None:
                    raise ValueError("hist_bin_id is required when history bins are given")
                ids.extend(hist_bin_id(b) for b in seg.values)
            history_span = (start, len(ids))
        elif seg.kind == "image":
            start = len(ids)
            ids.extend([image_token_id] * image_tokens(len(image_spans)))
            image_spans.append((start, len(ids)))
        elif seg.kind == "bins":
            start = len(ids)
            if seg.values is None:
                continue                      # generation mode: nothing to force
            if len(seg.values) != seg.count:
                raise ValueError(
                    f"expected {seg.count} trajectory bins, got {len(seg.values)}")
            ids.extend(bin_id(b) for b in seg.values)
            traj_span = (start, len(ids))
        else:
            raise ValueError(f"unknown segment kind {seg.kind!r}")

    return Assembled(
        input_ids=ids, history_span=history_span, image_spans=image_spans,
        coc_span=coc_span, struct_span=struct_span, traj_span=traj_span,
        n_prompt=len(ids) if n_prompt is None else n_prompt,
    )


def render(segments: list[Segment], image_token: str = "<image>") -> str:
    """Human-readable rendering of the context."""
    out = []
    for seg in segments:
        if seg.kind == "text":
            out.append(seg.text)
        elif seg.kind == "slots":
            out.append(seg.text * seg.count if seg.values is None
                       else f"<hist>x{seg.count}")
        elif seg.kind == "image":
            out.append(image_token)
        elif seg.kind == "bins":
            out.append(f"<bin>x{seg.count}")
    return "".join(out)
