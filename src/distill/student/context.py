"""Window + cached teacher targets -> student model inputs."""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable

import numpy as np
import torch

from . import prompt as prompt_mod

log = logging.getLogger(__name__)


def load_image_processor(ckpt: str):
    """Edge's own image processor."""
    from transformers import AutoImageProcessor, AutoProcessor
    try:
        from cosmos_framework.data.generator.processors.cosmos3_edge_processing import (
            build_cosmos3_edge_processor, is_cosmos3_edge_native_snapshot)
        if is_cosmos3_edge_native_snapshot(ckpt):
            proc = build_cosmos3_edge_processor(ckpt)
            ip = getattr(proc, "image_processor", None)
            if ip is None:
                raise RuntimeError(
                    "build_cosmos3_edge_processor returned no image_processor")
            log.info("image processor: %s (native Cosmos3-Edge port)",
                     type(ip).__name__)
            return ip
    except ImportError as e:
        log.info("cosmos-framework processor port unavailable (%s); "
                 "falling back to AutoProcessor", e)

    proc = AutoProcessor.from_pretrained(ckpt)
    ip = getattr(proc, "image_processor", None)
    if ip is not None:
        return ip
    log.info("AutoProcessor exposed no image_processor (%s); trying "
             "AutoImageProcessor", type(proc).__name__)
    return AutoImageProcessor.from_pretrained(ckpt)


def _grid_from_encoding(enc):
    """`(pixel_values [N_patches, C*p*p], image_grid_thw [n_images, 3])`."""
    pixel_values = enc["pixel_values"]
    if "image_grid_thw" in enc:
        return pixel_values, enc["image_grid_thw"]
    ss = enc["spatial_shapes"]
    pixel_values = pixel_values.view(-1, pixel_values.shape[-1])
    t_dim = torch.ones((ss.shape[0], 1), dtype=ss.dtype, device=ss.device)
    return pixel_values, torch.cat([t_dim, ss], dim=1)


PREFIX_MASK = -1


def prefix_bin_id(b: int, future_base: int, mask_id: int) -> int:
    """Region-relative future bin -> vocabulary id; `PREFIX_MASK` -> the mask id."""
    b = int(b)
    return mask_id if b < 0 else future_base + b


@dataclasses.dataclass
class ContextBuilder:
    """Assembles one window's context. Cheap to copy into a worker."""

    tokenizer: Any
    image_processor: Any
    cameras: list[str]
    n_frames: int
    future_base: int
    hist_base: int
    special_ids: dict[str, int]
    image_token_id: int
    hist_tokenize: Callable | None = None
    merge_size: int = 1

    def _encode_text(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _history_bins(self, window) -> list[int] | None:
        """Ego motion -> 48 region-relative delta bins, the teacher's way."""
        if self.hist_tokenize is None:
            return None
        bins = self.hist_tokenize(window.data["ego_history_xyz"],
                                  window.data["ego_history_rot"])
        return [int(b) for b in bins[0]]

    def _encode_images(self, window):
        """Camera frames -> (pixel_values, image_grid_thw, tokens per image)."""
        if self.image_processor is None:
            return None, None, []
        frames = []
        for name in sorted(self.cameras, key=lambda c: prompt_mod.CAMERA_INDEX[c]):
            arr = window.frames_student[name]          # (F, h, w, 3) uint8
            frames.extend(arr[i] for i in range(min(self.n_frames, len(arr))))
        enc = self.image_processor(images=frames, return_tensors="pt")
        pixel_values, grid = _grid_from_encoding(enc)
        per_image = [int(t * h * w) // (self.merge_size ** 2)
                     for t, h, w in grid.tolist()]
        return pixel_values, grid, per_image

    def build(self, window, coc_text: str | None = None,
              traj_bins: list[int] | None = None,
              nav_text: str | None = None,
              for_generation: bool = False) -> dict:
        """Teacher-forced sequence when targets are given; a generation prompt otherwise."""
        pixel_values, grid, per_image = self._encode_images(window)

        user = prompt_mod.user_segments(
            self.cameras, self.n_frames, nav_text=nav_text,
            hist_bins=self._history_bins(window))
        assistant = prompt_mod.assistant_segments(
            coc_text=coc_text, traj_bins=traj_bins, for_generation=for_generation)

        common = dict(
            encode=self._encode_text,
            special_id=self.special_ids.__getitem__,
            bin_id=lambda b: prefix_bin_id(b, self.future_base,
                                           self.special_ids["<|traj_history|>"]),
            hist_bin_id=lambda b: self.hist_base + int(b),
            image_tokens=lambda i: per_image[i] if per_image else 0,
            image_token_id=self.image_token_id,
        )
        n_prompt = len(prompt_mod.assemble(user, **common).input_ids)
        a = prompt_mod.assemble(user + assistant, n_prompt=n_prompt, **common)

        if per_image and len(a.image_spans) != len(per_image):
            raise RuntimeError(
                f"{len(a.image_spans)} image slots in the context but "
                f"{len(per_image)} images processed — camera/frame counts disagree")

        return {
            "input_ids": torch.tensor(a.input_ids, dtype=torch.long),
            "pixel_values": pixel_values,
            "image_grid_thw": grid,
            "coc_span": a.coc_span,
            "struct_span": a.struct_span,
            "traj_span": a.traj_span,
            "history_span": a.history_span,
            "n_prompt": a.n_prompt,
        }


def spans_to_mask(span, length: int) -> torch.Tensor:
    m = torch.zeros(length, dtype=torch.bool)
    if span is not None:
        m[span[0]:span[1]] = True
    return m
