"""Teacher vs zero-shot student trajectory gap on the same windows."""
from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .open_loop import ade, min_ade

log = logging.getLogger(__name__)

AV_DOMAIN_NAME = "av"      # EMBODIMENT_TO_DOMAIN_ID["av"] == 1
AV_RAW_ACTION_DIM = 9      # EMBODIMENT_TO_RAW_ACTION_DIM["av"] == 9


@torch.no_grad()
def teacher_predictions(cfg, clip_ids: list[str], n_samples: int = 6) -> dict:
    """Alpamayo 1.5 future trajectories from the expert rollout, metres, ego frame."""
    from ..data.preprocess import iter_windows
    from ..teacher.wrapper import TeacherWrapper

    tw = TeacherWrapper(cfg)
    space = tw.future_traj_tokenizer.action_space
    preds, gts, keys = [], [], []
    for clip_id in clip_ids:
        for w_idx, window in iter_windows(cfg, clip_id):
            out = tw.label_window(
                window,
                k_flow=1,
                topk=1,
                max_coc=cfg.teacher.max_coc_tokens,
                n_traj_samples=n_samples,
            )
            hist_xyz = window.data["ego_history_xyz"][0, 0]     # (T, 3)
            hist_rot = window.data["ego_history_rot"][0, 0]     # (T, 3, 3)
            k = out.traj_samples.shape[0]
            xyz, _ = space.action_to_traj(
                out.traj_samples.to(hist_xyz.device),
                hist_xyz.unsqueeze(0).expand(k, -1, -1),
                hist_rot.unsqueeze(0).expand(k, -1, -1, -1),
            )                                                   # (k, H, 3)
            preds.append(xyz.float().cpu().numpy())
            gts.append(np.asarray(window.gt_future_xyz, dtype=np.float32))
            keys.append(f"{clip_id}/{w_idx:02d}")
            log.info("teacher %s (%d/%d)", keys[-1], len(keys),
                     len(clip_ids) * cfg.data.windows_per_clip)
    return {"pred": np.stack(preds), "gt": np.stack(gts), "keys": keys,
            "model": "alpamayo1.5", "n_samples": n_samples}


def av_actions_to_xyz(actions: torch.Tensor) -> torch.Tensor:
    """Edge's 9-D action (assumed [dxyz, rot6d]) -> cumulative ego-frame xyz."""
    if actions.shape[-1] < 3:
        raise ValueError(f"expected >=3 action dims, got {actions.shape[-1]}")
    return torch.cumsum(actions[..., :3].float(), dim=-2)


@dataclasses.dataclass
class EdgeAVRunner:
    """Zero-shot Cosmos 3 Edge on the AV embodiment."""
    cfg: Any = None
    num_steps: int = 35
    guidance: float = 7.0
    seed: int = 0
    camera: str = "camera_front_wide_120fov"
    fps: int = 30

    def __post_init__(self):
        from cosmos_framework.data.generator.action.utils.domain_utils import (
            get_action_dim, get_domain_id,
        )
        self.domain_id = get_domain_id(AV_DOMAIN_NAME)
        self.raw_action_dim = get_action_dim(AV_DOMAIN_NAME)
        if self.raw_action_dim != AV_RAW_ACTION_DIM:
            raise RuntimeError(
                f"the shipped 'av' raw action dim is {self.raw_action_dim}, not "
                f"{AV_RAW_ACTION_DIM} — av_actions_to_xyz's layout assumption is "
                "keyed to 9-D and must be revisited")
        self.model = None

    def load(self):
        from huggingface_hub import snapshot_download
        from cosmos_framework.inference.model import Cosmos3OmniModel

        ckpt = snapshot_download(self.cfg.paths.student_repo)
        self.model = Cosmos3OmniModel.from_pretrained_dcp(ckpt).cuda().eval()
        return self

    def _video(self, window) -> torch.Tensor:
        """(C, T, H, W) uint8 from the window's front camera."""
        arr = window.frames_student[self.camera]        # (F, h, w, 3) uint8
        return torch.from_numpy(np.ascontiguousarray(arr)).permute(3, 0, 1, 2)

    @torch.no_grad()
    def run(self, window, mode: str, action_length: int, prompt: str) -> torch.Tensor:
        """One window -> (T, 9) predicted actions ("wam" or "inverse_dynamics")."""
        from cosmos_framework.data.generator.action.utils.transforms import (
            build_sequence_plan_from_mode,
        )
        from cosmos_framework.data.generator.action.utils.action_processing import (
            ActionProcessingRecord, make_batched_action_processing_fields,
        )

        video = self._video(window)
        plan = build_sequence_plan_from_mode(
            mode=mode,
            video_length=video.shape[1],
            action_length=action_length,
            has_text=True,
        )
        max_dim = int(self.model.model.net.config.action_dim)
        batch = {
            "video": [[video]],
            **make_batched_action_processing_fields(
                ActionProcessingRecord(raw_action_dim=self.raw_action_dim,
                                       action_normalizer=None),
                batch_size=1,
            ),
            "action": [[torch.zeros(action_length, max_dim, dtype=torch.float32)]],
            "mode": [mode],
            "ai_caption": [prompt],
            "prompt": [prompt],
            "conditioning_fps": [torch.tensor(self.fps, dtype=torch.long)],
            "image_size": torch.tensor(
                [[video.shape[2], video.shape[3]]], device="cuda"),
            "domain_id": [torch.tensor(self.domain_id, dtype=torch.long)],
            "sequence_plan": [plan],
        }
        samples = self.model.generate_samples_from_batch(
            batch, guidance=self.guidance, seed=[self.seed],
            num_steps=self.num_steps, has_negative_prompt=False,
        )
        return samples["action"][0].float().squeeze(0)[:, : self.raw_action_dim]


DRIVE_PROMPT = "Continue the same driving scene with smooth natural motion."


@torch.no_grad()
def edge_predictions(cfg, clip_ids: list[str], horizon_steps: int = 64,
                     **runner_kw) -> dict:
    """Cosmos 3 Edge zero-shot future trajectories, in metres, ego frame."""
    from ..data.preprocess import iter_windows

    runner = EdgeAVRunner(cfg=cfg, **runner_kw).load()
    preds, gts, keys = [], [], []
    for clip_id in clip_ids:
        for w_idx, window in iter_windows(cfg, clip_id):
            act = runner.run(window, mode="wam", action_length=horizon_steps,
                             prompt=DRIVE_PROMPT)
            preds.append(av_actions_to_xyz(act).cpu().numpy()[None])   # (1, H, 3)
            gts.append(np.asarray(window.gt_future_xyz, dtype=np.float32))
            keys.append(f"{clip_id}/{w_idx:02d}")
            log.info("edge %s (%d)", keys[-1], len(keys))
    return {"pred": np.stack(preds), "gt": np.stack(gts), "keys": keys,
            "model": "cosmos3-edge-zeroshot", "n_samples": 1}


@torch.no_grad()
def calibrate(cfg, clip_ids: list[str], **runner_kw) -> dict:
    """Check `av_actions_to_xyz` by recovering the known past via inverse dynamics."""
    from ..data.preprocess import iter_windows

    runner = EdgeAVRunner(cfg=cfg, **runner_kw).load()
    errs, keys = [], []
    for clip_id in clip_ids:
        for w_idx, window in iter_windows(cfg, clip_id):
            hist = np.asarray(window.data["ego_history_xyz"][0, 0], dtype=np.float32)
            steps = hist.shape[0] - 1
            act = runner.run(window, mode="inverse_dynamics",
                             action_length=steps, prompt=DRIVE_PROMPT)
            rec = av_actions_to_xyz(act).cpu().numpy()          # (steps, 3)
            # Known past, re-expressed as offsets from its first pose.
            truth = hist[1:] - hist[:1]
            n = min(rec.shape[0], truth.shape[0])
            errs.append(float(np.linalg.norm(rec[:n, :2] - truth[:n, :2], axis=-1).mean()))
            keys.append(f"{clip_id}/{w_idx:02d}")
    return {"per_window_ade": errs, "keys": keys,
            "mean_ade": float(np.mean(errs)) if errs else float("nan")}


def save(blob: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pred=blob["pred"], gt=blob["gt"],
                        keys=np.array(blob["keys"]),
                        meta=np.str_(json.dumps(
                            {k: v for k, v in blob.items()
                             if k not in ("pred", "gt", "keys")})))
    log.info("wrote %s (%d windows)", path, len(blob["keys"]))
    return path


def load(path: Path) -> dict:
    with np.load(path) as z:
        return {"pred": z["pred"], "gt": z["gt"],
                "keys": [str(k) for k in z["keys"]],
                **json.loads(str(z["meta"]))}


def score(blob: dict, k: int | None = None) -> dict:
    """ADE on the first mode, plus minADE over however many modes there are."""
    pred = torch.from_numpy(blob["pred"]).float()      # (N, K, H, 3)
    gt = torch.from_numpy(blob["gt"]).float()          # (N, H, 3)
    h = min(pred.shape[2], gt.shape[1])
    pred, gt = pred[:, :, :h], gt[:, :h]
    if k is not None:
        pred = pred[:, :k]
    a = ade(pred[:, 0], gt)
    m = min_ade(pred, gt)
    return {"model": blob.get("model", "?"), "n": int(a.numel()),
            "k": int(pred.shape[1]),
            "ade": float(a.mean()), "ade_p90": float(a.quantile(0.9)),
            "minade": float(m.mean()), "minade_p90": float(m.quantile(0.9))}


def report(teacher: dict, edge: dict) -> str:
    """The headroom, stated plainly."""
    t, e = score(teacher), score(edge)
    if teacher["keys"] != edge["keys"]:
        common = sorted(set(teacher["keys"]) & set(edge["keys"]))
        raise RuntimeError(
            f"the two runs cover different windows ({len(teacher['keys'])} vs "
            f"{len(edge['keys'])}, {len(common)} shared) — the gap is only "
            "meaningful on identical clips; re-run both on one split")
    gap = e["ade"] - t["ade"]
    rel = gap / e["ade"] * 100 if e["ade"] else float("nan")
    lines = [
        "=== headroom: teacher vs zero-shot student ===",
        f"windows: {t['n']}",
        f"  Alpamayo 1.5      ADE {t['ade']:6.3f} m  (p90 {t['ade_p90']:6.3f})  "
        f"minADE_{t['k']} {t['minade']:6.3f}",
        f"  Cosmos 3 Edge 0-shot ADE {e['ade']:6.3f} m  (p90 {e['ade_p90']:6.3f})  "
        f"minADE_{e['k']} {e['minade']:6.3f}",
        f"  GAP {gap:+.3f} m  ({rel:+.1f}% of the zero-shot error)",
        "",
        "Everything the distillation can achieve lives inside that gap. Narrow",
        "means the intervention needs to change (rank, targets, data), not the",
        "hyperparameters. NOTE minADE is not comparable across different k.",
    ]
    return "\n".join(lines)
