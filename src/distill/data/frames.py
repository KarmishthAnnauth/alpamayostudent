"""Cache the student's input frames and ego history per window (frames stored RGB, cv2 is BGR)."""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

FRAME_KEY = "img|{camera}|{index}"


@dataclasses.dataclass
class CachedWindow:
    """Window-shaped object read back from the input cache."""
    clip_id: str
    frames_student: dict[str, np.ndarray]
    data: dict[str, Any]


def input_path(cache_root: Path, clip_id: str, window_idx: int) -> Path:
    return Path(cache_root) / clip_id / f"{window_idx:02d}_input.npz"


def save_window_input(path: Path, window, quality: int = 92) -> Path:
    """Write JPEG frames + ego history. Returns the path written."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for camera, frames in window.frames_student.items():
        for i, frame in enumerate(frames):
            ok, buf = cv2.imencode(".jpg", frame[:, :, ::-1],   # RGB -> BGR
                                   [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if not ok:
                raise RuntimeError(f"JPEG encode failed for {camera} frame {i}")
            arrays[FRAME_KEY.format(camera=camera, index=i)] = buf.reshape(-1)

    for key in ("ego_history_xyz", "ego_history_rot"):
        v = window.data[key]
        arrays[key] = (v.detach().cpu().numpy() if hasattr(v, "detach")
                       else np.asarray(v)).astype(np.float32)
    # Atomic: resume keys off file existence.
    from ..teacher.labeler import atomic_savez
    atomic_savez(path, False, **arrays)
    return path


def load_window_input(path: Path, clip_id: str) -> CachedWindow:
    """Read a cached student input back into a Window-shaped object."""
    import cv2
    import torch

    frames: dict[str, list[tuple[int, np.ndarray]]] = {}
    data: dict[str, Any] = {}
    with np.load(path) as z:
        for key in z.files:
            if not key.startswith("img|"):
                data[key] = torch.from_numpy(z[key])
                continue
            _, camera, index = key.split("|")
            img = cv2.imdecode(z[key], cv2.IMREAD_COLOR)        # BGR
            if img is None:
                raise RuntimeError(f"JPEG decode failed for {key} in {path}")
            frames.setdefault(camera, []).append(
                (int(index), np.ascontiguousarray(img[:, :, ::-1])))

    return CachedWindow(
        clip_id=clip_id,
        # Sort by index only; never compare arrays.
        frames_student={c: np.stack([f for _, f in sorted(v, key=lambda t: t[0])])
                        for c, v in frames.items()},
        data=data,
    )
