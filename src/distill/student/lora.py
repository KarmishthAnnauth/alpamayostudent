"""LoRA injection, merging and gradient masking for the Edge student."""
from __future__ import annotations

import torch
import torch.nn as nn

AR_ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
GEN_ATTN_TARGETS = ["q_proj_moe_gen", "k_proj_moe_gen", "v_proj_moe_gen", "o_proj_moe_gen"]


def _is_lora_linear(module: nn.Module) -> bool:
    return hasattr(module, "lora_A") and hasattr(module, "lora_B")


def lora_modules(root: nn.Module):
    for path, mod in root.named_modules():
        if _is_lora_linear(mod):
            yield path, mod


def lora_parameters(root: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in root.named_parameters() if "lora_" in n]


def inject(root: nn.Module, targets: list[str], rank: int, alpha: int,
           device: torch.device | str) -> int:
    """Inject, materialize and initialize LoRA adapters under `root`; returns the count."""
    from cosmos_framework.utils.generator.lora import (
        init_lora_weights_post_materialization,
        inject_lora_pre_fsdp,
    )

    # re-injecting would nest adapters
    already = sum(1 for _ in lora_modules(root))
    if already:
        raise RuntimeError(
            f"{already} LoRA modules already present under this root — injecting "
            "again would nest adapters. Use one EdgeStudent per stage.")
    inject_lora_pre_fsdp(
        root,
        lora_rank=int(rank),
        lora_alpha=int(alpha),
        lora_target_modules=",".join(targets),
    )
    wrapped = sum(1 for _ in lora_modules(root))
    if wrapped == 0:
        raise RuntimeError(
            f"LoRA injection wrapped 0 modules for targets={targets}. The "
            "framework only warns; for us this silently means 'nothing is "
            "trainable', so it is fatal here.")

    # adapters are created on meta; materialize only them
    for _, mod in lora_modules(root):
        for adapter in (mod.lora_A, mod.lora_B):
            if adapter.weight.is_meta:
                adapter.to_empty(device=device)
    init_lora_weights_post_materialization(root)
    return wrapped


def describe(root: nn.Module) -> dict:
    """Summary of injected adapters."""
    paths = [p for p, _ in lora_modules(root)]
    n_params = sum(p.numel() for p in lora_parameters(root))
    leaves: dict[str, int] = {}
    for p in paths:
        leaves[p.split(".")[-1]] = leaves.get(p.split(".")[-1], 0) + 1
    return {"wrapped": len(paths), "by_leaf": leaves, "lora_params": n_params,
            "example": paths[:3]}


@torch.no_grad()
def merged_state_dict(root: nn.Module) -> dict[str, torch.Tensor]:
    """LoRA-folded CPU state dict, without mutating `root`."""
    sd = root.state_dict()
    out = {k: v for k, v in sd.items() if ".lora_A." not in k and ".lora_B." not in k}
    for path, mod in lora_modules(root):
        key = f"{path}.weight" if path else "weight"
        base = sd[key]
        delta = (mod.lora_B.weight.float() @ mod.lora_A.weight.float()) * mod._lora_scale
        out[key] = (base.float() + delta).to(base.dtype)
    return {k: v.detach().to("cpu") for k, v in out.items()}


@torch.no_grad()
def merge_lora_(root: nn.Module) -> int:
    """Fold adapters into the base weights in place and drop the wrappers."""
    merged = 0
    for parent_name, parent in list(root.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not _is_lora_linear(child):
                continue
            delta = (child.lora_B.weight.float() @ child.lora_A.weight.float()) * child._lora_scale
            child.weight.data.add_(delta.to(child.weight.dtype))
            # meta Linear, then reuse the existing parameters
            plain = nn.Linear(child.in_features, child.out_features,
                              bias=child.bias is not None, device="meta")
            plain.weight = child.weight
            if child.bias is not None:
                plain.bias = child.bias
            setattr(parent, child_name, plain)
            merged += 1
    return merged


def _row_mask_hook(param: nn.Parameter, keep: torch.Tensor):
    """Zero the gradient of every row where `keep` is False."""
    if not param.requires_grad:
        raise RuntimeError("enable requires_grad before registering a row mask")
    mask = keep.to(device=param.device, dtype=param.dtype).view(-1, *([1] * (param.dim() - 1)))
    return param.register_hook(lambda grad: grad.mul_(mask))


def mask_rows_below_(param: nn.Parameter, first_trainable_row: int):
    """Train only rows >= `first_trainable_row` (appended vocabulary)."""
    keep = torch.zeros(param.shape[0], dtype=torch.bool)
    keep[first_trainable_row:] = True
    return _row_mask_hook(param, keep)


def mask_rows_except_(param: nn.Parameter, row: int):
    """Train only `row` (our embodiment domain in the action head)."""
    keep = torch.zeros(param.shape[0], dtype=torch.bool)
    keep[row] = True
    return _row_mask_hook(param, keep)
