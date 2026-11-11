"""Cosmos 3 Edge student wrapper: parameter groups, trajectory vocab, action head."""
from __future__ import annotations
import contextlib
import logging
import re
from pathlib import Path
import torch
import torch.nn as nn

from . import lora, prompt as prompt_mod

# Gen tower = `_moe_gen` suffix + action modules; AR tower = the rest of language_model.
DIFF_TOWER_PAT = re.compile(
    r"(_moe_gen|action2llm|llm2action|action_modality_embed|time_embedder)")
AR_TOWER_PAT = re.compile(r"language_model")  # applied after excluding DIFF matches

log = logging.getLogger(__name__)


def _patch_hf_storage_info_for_torch28() -> None:
    """Let cosmos-framework's 3-arg `_HFStorageInfo(...)` work on torch 2.8."""
    import inspect
    from cosmos_framework.inference import model as _cf_model

    real = _cf_model._HFStorageInfo
    params = inspect.signature(real.__init__).parameters
    if "offset" not in params:
        return                                   # torch >= 2.10: nothing to do
    if getattr(real, "_a2e_shimmed", False):
        return

    def _compat(relative_path, shape, dtype, offset=0, length=0, **kw):
        return real(relative_path=relative_path, offset=offset, length=length,
                    shape=shape, dtype=dtype, **kw)

    _compat._a2e_shimmed = True
    _cf_model._HFStorageInfo = _compat
    log.info("patched _HFStorageInfo for torch %s (offset/length unused on load)",
             torch.__version__)


def _point_tokenizer_at_dir(node, ckpt_dir: str) -> bool:
    """Rewrite a `build_processor_lazy` tokenizer node to load from a local dir."""
    if not isinstance(node, dict):
        return False
    target = str(node.get("_target_", ""))
    if not target.endswith("build_processor_lazy"):
        return False
    if not (node.get("repository") or node.get("tokenizer_type")):
        return False
    for k in ("repository", "revision", "subdir"):
        node.pop(k, None)
    node["tokenizer_type"] = ckpt_dir
    return True


def _patch_tokenizer_nodes(obj, ckpt_dir: str) -> int:
    """Walk the model spec and repoint every processor node at `ckpt_dir`."""
    n = 0
    if isinstance(obj, dict):
        n += _point_tokenizer_at_dir(obj, ckpt_dir)
        for v in obj.values():
            n += _patch_tokenizer_nodes(v, ckpt_dir)
    elif isinstance(obj, list):
        for v in obj:
            n += _patch_tokenizer_nodes(v, ckpt_dir)
    return n


def _omni_config(ckpt_dir: str):
    """The framework's Cosmos3-Edge architecture spec as a Cosmos3OmniConfig."""
    import cosmos_framework
    from cosmos_framework.inference.model import Cosmos3OmniConfig
    from cosmos_framework.inference.common.config import deserialize_config_dict
    from cosmos_framework.inference.common.public_model_config import (
        load_model_config_from_hf_config)

    yaml_path = (Path(cosmos_framework.__file__).parent / "inference" / "configs"
                 / "model" / "Cosmos3-Edge.yaml")
    if not yaml_path.is_file():
        raise FileNotFoundError(
            f"{yaml_path} is missing — the Cosmos3-Edge architecture spec ships "
            "with cosmos-framework and is what the HF repo's config.json does not "
            "carry. Check the ../cosmos-framework checkout.")
    model_dict = load_model_config_from_hf_config(deserialize_config_dict(yaml_path))
    if not _patch_tokenizer_nodes(model_dict, ckpt_dir):
        log.warning("no build_processor_lazy tokenizer node found in %s — if model "
                    "construction stalls, it is re-downloading the processor from "
                    "the hub", yaml_path)
    return Cosmos3OmniConfig(model=model_dict)


class EdgeStudent(nn.Module):
    def __init__(self, cfg, device: str = "cuda"):
        super().__init__()
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        from cosmos_framework.inference.model import Cosmos3OmniConfig, Cosmos3OmniModel

        self.cfg = cfg
        repo = str(cfg.paths.student_repo)
        ckpt = repo if Path(repo).is_dir() else snapshot_download(repo)
        self._ckpt_dir = ckpt
        _patch_hf_storage_info_for_torch28()
        self.model = Cosmos3OmniModel.from_pretrained_dcp(
            Path(ckpt), config=_omni_config(ckpt)).to(device)
        self.omni = self.model.model
        self.net = self.omni.net
        self.lm = self.net.language_model
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.action_domain_id = int(cfg.student.get("action_domain_id", 31))
        self.raw_action_dim = 2          # (accel, curvature)
        self.n_action_tokens = 64        # waypoints @ 10 Hz
        self.max_action_dim = self.net.config.action_dim  # 64 on Edge
        self._device = torch.device(device)
        self._mask_handles: list = []   # gradient row-masks, cleared per stage
        self.lora_stats: dict = {}

    def enable_gradient_checkpointing(self) -> None:
        """Try to enable activation checkpointing on the AR tower (warns if unavailable)."""
        try:
            self.lm.gradient_checkpointing_enable()
        except ValueError as e:
            log.warning("gradient checkpointing unavailable on %s (%s); "
                        "continuing without it - see stage1.grad_checkpoint in "
                        "the config for why it is inert here",
                        type(self.lm).__name__, e)
            return
        log.warning("gradient_checkpointing_enable() accepted, but "
                    "reasoner_forward does not route through a checkpoint "
                    "function - expect NO memory saving")

    # ---------------- geometry ----------------

    @property
    def n_layers(self) -> int:
        """Decoder layers in the AR tower."""
        return int(self.lm.config.num_hidden_layers)

    @property
    def hidden_size(self) -> int:
        """AR-tower hidden width."""
        return int(self.lm.config.hidden_size)

    # ---------------- vocab ----------------

    def extend_trajectory_vocab(self, traj_tok_spec: dict) -> None:
        """Append future bins, history bins and structural tokens to embeddings and lm_head."""
        emb = self.lm.get_input_embeddings()
        old_n, dim = emb.weight.shape
        n_future = int(traj_tok_spec["vocab_size"])                    # 3000
        n_hist = int(traj_tok_spec.get("hist_vocab_size", 0))          # 1000
        n_bins = int(traj_tok_spec.get("total_bins", n_future + n_hist))
        if n_bins != n_future + n_hist:
            raise ValueError(
                f"trajectory regions do not tile the vocabulary: {n_future} future "
                f"+ {n_hist} history != {n_bins} total")
        specials = list(prompt_mod.SPECIAL_TOKENS)
        n_new = n_bins + len(specials)
        new_n = old_n + n_new
        ekw = {"device": emb.weight.device, "dtype": emb.weight.dtype}

        new_emb = nn.Embedding(new_n, dim, **ekw)
        with torch.no_grad():
            new_emb.weight[:old_n] = emb.weight
            new_emb.weight[old_n:] = (emb.weight.mean(dim=0, keepdim=True)
                                      + 0.02 * torch.randn(n_new, dim, **ekw))
        self.lm.set_input_embeddings(new_emb)

        head = self.lm.lm_head
        hkw = {"device": head.weight.device, "dtype": head.weight.dtype}
        new_head = nn.Linear(dim, new_n, bias=head.bias is not None, **hkw)
        with torch.no_grad():
            new_head.weight[:old_n] = head.weight
            new_head.weight[old_n:] = (head.weight.mean(dim=0, keepdim=True)
                                       + 0.02 * torch.randn(n_new, dim, **hkw))
            if head.bias is not None:
                new_head.bias[:old_n] = head.bias
                new_head.bias[old_n:] = head.bias.mean()
        self.lm.lm_head = new_head

        self.lm.vocab_size = new_n
        seen = set()
        for c in (getattr(self.lm, "config", None),
                  getattr(getattr(self.lm, "config", None), "text_config", None),
                  getattr(getattr(self.lm, "model", None), "config", None)):
            if c is None or id(c) in seen:
                continue
            seen.add(id(c))
            if getattr(c, "vocab_size", None) is not None:
                c.vocab_size = new_n

        self.new_token_range = (old_n, new_n)
        self.future_base = old_n
        self.n_future_bins = n_future
        self.hist_base = old_n + n_future
        self.n_hist_bins = n_hist
        self.special_ids = {t: old_n + n_bins + i for i, t in enumerate(specials)}
        self._traj_decode = traj_tok_spec["detokenizer_fn"]
        self.hist_tokenize = traj_tok_spec.get("hist_tokenize_fn")

    def detokenize_traj(self, tokens: torch.Tensor, hist_xyz: torch.Tensor,
                        hist_rot: torch.Tensor) -> torch.Tensor:
        """Region-relative bins in emission order -> (B, H, 3) ego-frame waypoints."""
        from ..teacher.wrapper import swap_action_dims

        toks = swap_action_dims(tokens.reshape(tokens.shape[0], -1).long().cpu())
        fut_xyz, _, _ = self._traj_decode(hist_xyz.float().cpu(),
                                          hist_rot.float().cpu(), toks)
        return fut_xyz

    # ---------------- context assembly ----------------

    def future_bin_id(self, b: int) -> int:
        return self.future_base + int(b)

    def hist_bin_id(self, b: int) -> int:
        return self.hist_base + int(b)

    def special_token_id(self, token: str) -> int:
        return self.special_ids[token]

    def context_builder(self, image_processor=None, cameras=None, n_frames=None):
        """A picklable context assembler for this student's vocabulary."""
        from .context import ContextBuilder, load_image_processor

        if not hasattr(self, "new_token_range"):
            raise RuntimeError("call extend_trajectory_vocab() first")
        if image_processor is None:
            image_processor = load_image_processor(self._ckpt_dir)
        self.lm._ensure_vision_tower()   # plumbs config.image_token_id
        merge = getattr(getattr(self.lm, "visual", None), "spatial_merge_size", None)
        if merge is None:
            merge = getattr(getattr(self.lm, "config", None), "spatial_merge_size", None)
        if merge is None:
            raise RuntimeError(
                "cannot resolve spatial_merge_size from the vision tower or the "
                "LM config; ContextBuilder's placeholder count would be guesswork")
        return ContextBuilder(
            tokenizer=self.tokenizer,
            image_processor=image_processor,
            cameras=list(cameras or self.cfg.data.raw["cameras"]),
            n_frames=int(n_frames or self.cfg.data.context_frames),
            future_base=self.future_base,
            hist_base=self.hist_base,
            special_ids=dict(self.special_ids),
            image_token_id=self._image_token_id,
            hist_tokenize=self.hist_tokenize,
            merge_size=int(merge or 1),
        )

    @property
    def _image_token_id(self) -> int:
        tid = getattr(self.lm.config, "image_token_id", None)
        if tid is None:
            raise RuntimeError(
                "config.image_token_id is unset — it is plumbed by "
                "_ensure_vision_tower(), so touch the vision tower first")
        return int(tid)

    # ---------------- parameter groups ----------------

    def _is_diff(self, name: str) -> bool:
        return bool(DIFF_TOWER_PAT.search(name))

    def _is_ar(self, name: str) -> bool:
        return AR_TOWER_PAT.search(name) is not None and not self._is_diff(name)

    def _decoder_layers(self) -> nn.Module:
        """The decoder-layer stack (both towers, no vision encoder)."""
        return self.lm.model.layers

    def _reset_trainable(self) -> None:
        # Materialize the lazy vision tower before freezing it.
        ensure = getattr(self.lm, "_ensure_vision_tower", None)
        if ensure is not None:
            ensure()
        for h in self._mask_handles:
            h.remove()
        self._mask_handles.clear()
        for p in self.model.parameters():
            p.requires_grad_(False)
        visual = getattr(self.lm, "visual", None)
        if visual is not None:
            for p in visual.parameters():
                p.requires_grad_(False)

    def _enable_new_vocab_rows(self) -> list[nn.Parameter]:
        """Enable the appended vocab rows of embed_tokens and lm_head only."""
        if not hasattr(self, "new_token_range"):
            raise RuntimeError(
                "call extend_trajectory_vocab() before param_groups_stage1()")
        old_n, _ = self.new_token_range
        out = []
        for p in (self.lm.get_input_embeddings().weight, self.lm.lm_head.weight):
            p.requires_grad_(True)
            self._mask_handles.append(lora.mask_rows_below_(p, old_n))
            out.append(p)
        return out

    def _enable_action_domain_rows(self) -> list[nn.Parameter]:
        """Enable only our embodiment row of action2llm/llm2action."""
        out = []
        for proj in (self.net.action2llm, self.net.llm2action):
            for p in (proj.fc.weight, proj.bias.weight):
                p.requires_grad_(True)
                self._mask_handles.append(
                    lora.mask_rows_except_(p, self.action_domain_id))
                out.append(p)
        return out

    def _enable_shared_action_embeds(self) -> list[nn.Parameter]:
        """Enable the action embeddings shared by all embodiments."""
        ps = [self.net.action_modality_embed] + list(self.net.time_embedder.parameters())
        for p in ps:
            p.requires_grad_(True)
        return ps

    def param_groups_stage1(self):
        """AR tower via LoRA + the appended trajectory rows; gen tower frozen."""
        lcfg = self.cfg.student.lora.stage1
        self._reset_trainable()
        if lcfg.get("enabled", True):
            self.lora_stats = self._inject(lcfg, lora.AR_ATTN_TARGETS)
            adapted = lora.lora_parameters(self._decoder_layers())
        else:
            adapted = [p for n, p in self.lm.model.named_parameters()
                       if not self._is_diff(n) and "embed" not in n]
            for p in adapted:
                p.requires_grad_(True)
            self.lora_stats = {"wrapped": 0, "full_ft": True}
        boosted = self._enable_new_vocab_rows()
        return [{"params": adapted, "lr_mult": 1.0},
                {"params": boosted, "lr_mult": self.cfg.student.new_token_lr_mult}]

    def param_groups_stage2(self):
        """Gen tower (LoRA or full FT) + our action-head row; optional AR LoRA."""
        lcfg = self.cfg.student.lora.stage2
        self._reset_trainable()
        acfg = self.cfg.student.lora.get("stage2_ar")
        ar_lora = bool(acfg and acfg.get("enabled", False))
        self.train_ar_context = ar_lora
        if ar_lora and lcfg.get("enabled", True):
            raise NotImplementedError("stage2_ar with a gen-tower LoRA: one EdgeStudent injects once")
        ar_params: list[nn.Parameter] = []
        if ar_lora:
            self.lora_stats = self._inject(acfg, lora.AR_ATTN_TARGETS)
            ar_params = lora.lora_parameters(self._decoder_layers())
        if lcfg.get("enabled", True):
            self.lora_stats = self._inject(lcfg, lora.GEN_ATTN_TARGETS)
            adapted = lora.lora_parameters(self._decoder_layers())
        else:
            adapted = [p for n, p in self.lm.model.named_parameters()
                       if self._is_diff(n) and "lora_" not in n]
            fp32 = bool(self.cfg.stage2.get("fp32_master", True))
            for p in adapted + ar_params:
                if fp32 and p.dtype != torch.float32:
                    p.data = p.data.float()
                p.requires_grad_(True)
            self.lora_stats = {**self.lora_stats, "full_ft": True, "fp32_master": fp32}
        new_iface = self._enable_action_domain_rows()
        if not lcfg.get("enabled", True) and bool(self.cfg.stage2.get("fp32_master", True)):
            for p in new_iface:
                if p.dtype != torch.float32:
                    p.data = p.data.float()
        if self.cfg.student.lora.get("train_shared_action_embeds", False):
            new_iface += self._enable_shared_action_embeds()
        groups = [{"params": adapted, "lr_mult": 1.0},
                  {"params": new_iface, "lr_mult": self.cfg.student.new_token_lr_mult}]
        if ar_params:
            groups.append({"params": ar_params, "lr_mult": float(acfg.get("lr_mult", 1.0))})
        return groups

    def _inject(self, lcfg, default_targets: list[str]) -> dict:
        targets = list(lcfg.get("targets", default_targets))
        lora.inject(self._decoder_layers(), targets=targets,
                    rank=lcfg.rank, alpha=lcfg.alpha, device=self._device)
        return lora.describe(self._decoder_layers())

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Everything with a gradient — what the trainer clips."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def merged_state_dict(self) -> dict:
        """LoRA-folded weights for checkpointing; does not disturb training."""
        return lora.merged_state_dict(self.model)

    # ---------------- forward APIs used by the trainers ----------------

    def ar_forward(self, batch, capture_layers=None) -> dict:
        """Teacher-forced reasoner pass returning logits, captured hidden states and final hidden."""
        fwd, _ = self._reasoner_inputs(batch)
        with self._capture_layers(capture_layers) as hidden:
            h = self.lm.model.reasoner_forward(cache=None, **fwd)
        return {"logits": self.lm.lm_head(h), "hidden": hidden, "final_hidden": h}

    def _reasoner_inputs(self, batch) -> tuple[dict, torch.Tensor | None]:
        """Prefill kwargs for `reasoner_forward`, plus the mrope deltas."""
        pixel_values = batch.get("pixel_values")
        if pixel_values is None:
            return dict(input_ids=batch["input_ids"]), None
        from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl import (
            reasoner_multimodal_utils as mm,
        )
        self.lm._ensure_vision_tower()
        embeds, vis_mask, deepstack, pos_ids, deltas = mm.prepare_multimodal_reasoner_inputs(
            self.lm,
            input_ids=batch["input_ids"],
            pixel_values=pixel_values,
            image_grid_thw=batch["image_grid_thw"],
            attention_mask=batch.get("attention_mask"),
        )
        return (dict(input_ids=None, inputs_embeds=embeds, position_ids=pos_ids,
                     visual_pos_masks=vis_mask, deepstack_visual_embeds=deepstack),
                deltas)

    @contextlib.contextmanager
    def _capture_layers(self, layers=None):
        """Collect per-layer reasoner outputs by shadowing `reasoner_forward`."""
        stack = self._decoder_layers()
        wanted = set(range(len(stack))) if layers is None else {int(i) for i in layers}
        out: dict[int, torch.Tensor] = {}
        patched = []
        for i, layer in enumerate(stack):
            if i not in wanted:
                continue
            original = layer.reasoner_forward
            had_own = "reasoner_forward" in layer.__dict__

            def _wrap(idx, fn):
                def inner(*a, **kw):
                    h = fn(*a, **kw)
                    out[idx] = h
                    return h
                return inner

            layer.reasoner_forward = _wrap(i, original)
            patched.append((layer, original, had_own))
        try:
            yield out
        finally:
            for layer, original, had_own in patched:
                if had_own:
                    layer.reasoner_forward = original
                else:
                    del layer.__dict__["reasoner_forward"]

    def build_flow_context(self, batch, keep_final_hidden: bool = False):
        """Reasoner pass -> per-layer K/V the action tokens attend."""
        from . import flow_path
        return flow_path.build_flow_context(self, batch, keep_final_hidden=keep_final_hidden)

    def sample_actions(self, ctx, owner, **kw) -> dict:
        """Flow-head sampler (ODE, or SDE with per-step log-probs); see flow_path."""
        from . import flow_path
        return flow_path.sample_actions(self, ctx, owner, **kw)

    def flow_forward(self, a_t, t, ctx, owner, grad_checkpoint: bool | None = None) -> torch.Tensor:
        """Gen-tower velocity at (a_t, t), returned in the teacher convention."""
        n = a_t.shape[0]
        if grad_checkpoint is None:
            grad_checkpoint = bool(self.cfg.stage2.get("grad_checkpoint", False))
        sigma = (1.0 - t).clamp(0.0, 1.0)                 # student noise weight
        # zero-pad raw actions to the domain interface width
        x = a_t.new_zeros(n, self.n_action_tokens, self.max_action_dim)
        x[..., : self.raw_action_dim] = a_t
        domain = torch.full((n,), self.action_domain_id,
                            dtype=torch.long, device=a_t.device)

        # domain projection + timestep embedding + modality embedding
        emb = self.net.action2llm(x, domain)               # (N, 64, hidden)
        ts = self.net.time_embedder(self._embedder_timesteps(sigma))
        emb = emb + ts.unsqueeze(1) + self.net.action_modality_embed

        h = self._gen_pathway_forward(emb, ctx, owner, grad_checkpoint)   # (N, 64, hidden)

        v_student = self.net.llm2action(self.net.norm_moe_gen(h)
                                        if hasattr(self.net, "norm_moe_gen") else h,
                                        domain)            # (N, 64, 64)
        v_student = v_student[..., : self.raw_action_dim]  # (N, 64, 2)
        return -v_student  # student target = noise - data; teacher = data - noise

    def _embedder_timesteps(self, sigma: torch.Tensor) -> torch.Tensor:
        """Student noise level -> `time_embedder` input."""
        return sigma * self._timestep_range

    @property
    def _timestep_range(self) -> float:
        cfg = getattr(self.omni, "config", None)
        rng = getattr(getattr(cfg, "diffusion_expert_config", None), "timestep_range", None)
        if rng is None:  # derive it back out of the scale the net was built with
            n = getattr(getattr(cfg, "rectified_flow_inference_config", None),
                        "num_train_timesteps", None)
            rng = float(self.net.timestep_scale) * float(n) if n else 1.0
        return float(rng)

    def _gen_pathway_forward(self, gen_embeds, ctx, owner, grad_checkpoint=False):
        """Dense action-only gen pathway attending the reasoner K/V."""
        from . import flow_path
        return flow_path.gen_pathway_forward(self, gen_embeds, ctx, owner,
                                             grad_checkpoint=grad_checkpoint)

    @torch.no_grad()
    def generate_coc_and_traj(self, batch, max_coc_tokens: int = 64,
                              n_future: int | None = None,
                              coc_temperature: float | None = None) -> list[dict]:
        """One rollout per row: free-running CoC, forced boundary, then trajectory bins."""
        from cosmos_framework.model.generator.mot.unified_mot import (
            ReasonerKVCache, _sample_next_token,
        )
        model = self.lm.model
        n_fut = int(n_future if n_future is not None else prompt_mod.N_FUTURE_TOKENS)
        boundary = self.tokenizer.encode("<|traj_future_start|>", add_special_tokens=False)
        cache = ReasonerKVCache.empty(num_layers=len(model.layers))
        fwd, deltas = self._reasoner_inputs(batch)
        h = model.reasoner_forward(cache=cache, **fwd)
        base_mrope = (deltas.to(dtype=torch.long).unsqueeze(0).expand(3, -1, -1)
                      if deltas is not None else None)
        gcfg = self.cfg.teacher
        temperature = float(gcfg.get("gen_temperature", 1.0))
        top_p = float(gcfg.get("gen_top_p", 1.0))
        coc_t = float(coc_temperature) if coc_temperature is not None else temperature
        vocab_lo = self.new_token_range[0]
        stops = ("<|cot_end|>", "</think>")
        pad = int(self.tokenizer.pad_token_id)

        b = h.shape[0]
        phase = [0] * b
        coc: list[list[int]] = [[] for _ in range(b)]
        bins: list[list[int]] = [[] for _ in range(b)]
        forced: list[list[int]] = [[] for _ in range(b)]
        failed = [False] * b
        logits = self.lm.lm_head(h[:, -1, :])
        max_steps = int(max_coc_tokens) + len(boundary) + n_fut
        for step in range(max_steps):
            free = _sample_next_token(logits, do_sample=True, temperature=coc_t,
                                      top_k=None, top_p=top_p)
            binned = _sample_next_token(self._restrict_to_future_bins(logits), do_sample=True,
                                        temperature=temperature, top_k=None, top_p=top_p)
            tok = torch.full_like(free, pad)
            for j in range(b):
                if phase[j] == 0:
                    tok[j] = free[j]
                    coc[j].append(int(free[j]))
                    tail = self.tokenizer.decode([t for t in coc[j][-12:] if t < vocab_lo])
                    if any(s in tail for s in stops):
                        phase[j] = 1
                        forced[j] = list(boundary)
                    elif len(coc[j]) >= int(max_coc_tokens):
                        phase[j] = 3
                        failed[j] = True
                elif phase[j] == 1:
                    tok[j] = forced[j].pop(0)
                    if not forced[j]:
                        phase[j] = 2
                elif phase[j] == 2:
                    tok[j] = binned[j]
                    bins[j].append(int(binned[j]) - self.future_base)
                    if len(bins[j]) >= n_fut:
                        phase[j] = 3
            if all(p == 3 for p in phase) or step == max_steps - 1:
                break
            position_ids = None if base_mrope is None else base_mrope + cache.seq_len
            h = model.reasoner_forward(tok.unsqueeze(1), cache=cache, position_ids=position_ids)
            logits = self.lm.lm_head(h[:, -1, :])
        out = []
        for j in range(b):
            ok = (not failed[j]) and len(bins[j]) == n_fut
            out.append({"coc_ids": coc[j], "terminated": not failed[j],
                        "bins": torch.tensor(bins[j], dtype=torch.long) if ok else None})
        return out

    def generate_traj_tokens(self, batch, n_tokens: int | None = None) -> torch.Tensor:
        """Sample region-relative trajectory bins, restricted to the future-bin rows."""
        from cosmos_framework.model.generator.mot.unified_mot import (
            ReasonerKVCache, _sample_next_token,
        )

        model = self.lm.model
        n = int(n_tokens if n_tokens is not None else prompt_mod.N_FUTURE_TOKENS)
        cache = ReasonerKVCache.empty(num_layers=len(model.layers))

        fwd, deltas = self._reasoner_inputs(batch)
        h = model.reasoner_forward(cache=cache, **fwd)
        base_mrope = (deltas.to(dtype=torch.long).unsqueeze(0).expand(3, -1, -1)
                      if deltas is not None else None)

        gcfg = self.cfg.teacher
        temperature = float(gcfg.get("gen_temperature", 1.0))
        top_p = float(gcfg.get("gen_top_p", 1.0))

        emitted = []
        logits = self.lm.lm_head(h[:, -1, :])
        for step in range(n):
            tok = _sample_next_token(
                self._restrict_to_future_bins(logits),
                do_sample=True, temperature=temperature, top_k=None, top_p=top_p)
            emitted.append(tok)
            if step == n - 1:
                break
            position_ids = None if base_mrope is None else base_mrope + cache.seq_len
            h = model.reasoner_forward(tok.unsqueeze(1), cache=cache,
                                       position_ids=position_ids)
            logits = self.lm.lm_head(h[:, -1, :])
        return torch.stack(emitted, dim=1) - self.future_base

    @torch.no_grad()
    def generate_coc_text(self, batch, max_new_tokens: int = 256) -> list[list[int]]:
        """Free-run the CoC text over the full vocabulary until it terminates."""
        from cosmos_framework.model.generator.mot.unified_mot import (
            ReasonerKVCache, _sample_next_token,
        )

        model = self.lm.model
        cache = ReasonerKVCache.empty(num_layers=len(model.layers))
        fwd, deltas = self._reasoner_inputs(batch)
        h = model.reasoner_forward(cache=cache, **fwd)
        base_mrope = (deltas.to(dtype=torch.long).unsqueeze(0).expand(3, -1, -1)
                      if deltas is not None else None)

        gcfg = self.cfg.teacher
        temperature = float(gcfg.get("gen_temperature", 1.0))
        top_p = float(gcfg.get("gen_top_p", 1.0))
        vocab_lo = self.new_token_range[0]
        stops = ("<|cot_end|>", "</think>")

        b = h.shape[0]
        done = [False] * b
        rows: list[list[int]] = [[] for _ in range(b)]
        logits = self.lm.lm_head(h[:, -1, :])
        for step in range(int(max_new_tokens)):
            tok = _sample_next_token(logits, do_sample=True, temperature=temperature,
                                     top_k=None, top_p=top_p)
            for j in range(b):
                if done[j]:
                    continue
                rows[j].append(int(tok[j]))
                tail = self.tokenizer.decode([t for t in rows[j][-12:] if t < vocab_lo])
                if any(s in tail for s in stops):
                    done[j] = True
            if all(done) or step == int(max_new_tokens) - 1:
                break
            position_ids = None if base_mrope is None else base_mrope + cache.seq_len
            h = model.reasoner_forward(tok.unsqueeze(1), cache=cache,
                                       position_ids=position_ids)
            logits = self.lm.lm_head(h[:, -1, :])
        return rows

    def _restrict_to_future_bins(self, logits: torch.Tensor) -> torch.Tensor:
        """-inf everywhere outside [future_base, future_base + 3000)."""
        lo, hi = self.future_base, self.future_base + self.n_future_bins
        out = torch.full_like(logits, float("-inf"))
        out[..., lo:hi] = logits[..., lo:hi]
        return out
