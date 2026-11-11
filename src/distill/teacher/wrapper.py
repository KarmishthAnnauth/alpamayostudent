"""TeacherWrapper: the ONLY file that touches the Alpamayo teacher codebase."""
from __future__ import annotations
import dataclasses

import torch
import torch.nn.functional as F


def swap_action_dims(tokens: torch.Tensor) -> torch.Tensor:
    """Convert between the teacher's EMITTED future-token order and the order `DiscreteTrajectoryTokenizer.encode/decode` use."""
    return tokens.reshape(*tokens.shape[:-1], -1, 2).flip(-1).reshape(*tokens.shape)


@dataclasses.dataclass
class TeacherWindowOutput:
    """Everything cached per training window during the offline labeling pass."""
    traj_token_ids: torch.Tensor
    traj_topk_idx: torch.Tensor
    traj_topk_logp: torch.Tensor
    coc_token_ids: torch.Tensor
    coc_text: str
    feats: dict[int, torch.Tensor]      # teacher_layer_idx -> (pool_len, D_t)
    flow_t: torch.Tensor
    flow_a_t: torch.Tensor
    flow_v: torch.Tensor
    traj_samples: torch.Tensor
    gt_traj: torch.Tensor
    gt_traj_token_ids: torch.Tensor
    gt_future_xyz: torch.Tensor


class HistoryTokenize:
    """Ego history -> discrete delta bins, the way A1.5 actually calls it."""

    def __init__(self, hist_tokenizer):
        self.hist_tokenizer = hist_tokenizer

    def __call__(self, ego_history_xyz, ego_history_rot):
        assert ego_history_xyz.ndim == 4, "expected (B, n_traj, T, 3)"
        hist_xyz = ego_history_xyz.flatten(0, 1)
        hist_rot = ego_history_rot.flatten(0, 1)
        return self.hist_tokenizer.encode(
            hist_xyz=hist_xyz[:, :1], hist_rot=hist_rot[:, :1],
            fut_xyz=hist_xyz, fut_rot=hist_rot,
        )


class TeacherWrapper:
    def __init__(self, cfg, device: str = "cuda", load_model: bool = True):
        import hydra.utils as hyu
        from alpamayo1_5.config import Alpamayo1_5Config
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

        self.cfg = cfg
        self.device = device
        self._processor = None
        repo = cfg.paths.teacher_repo
        if load_model:
            dtype = torch.bfloat16 if cfg.teacher.dtype == "bf16" else torch.float16
            self.model = Alpamayo1_5.from_pretrained(repo, dtype=dtype).to(device)
            self.model.eval()
            self.tconfig = self.model.config
            self.tokenizer = self.model.tokenizer
            self.future_traj_tokenizer = self.model.traj_tokenizer
            self.hist_traj_tokenizer = self.model.hist_traj_tokenizer
        else:
            self.model = None
            self.tconfig = Alpamayo1_5Config.from_pretrained(repo)
            self.tokenizer = self._build_tokenizer(self.tconfig)
            self.future_traj_tokenizer = hyu.instantiate(
                self.tconfig.traj_tokenizer_cfg, load_weights=False)
            self.hist_traj_tokenizer = hyu.instantiate(
                self.tconfig.hist_traj_tokenizer_cfg)

    @staticmethod
    def _build_tokenizer(config):
        """Standalone copy of ReasoningVLA._build_tokenizer."""
        from transformers import AutoProcessor
        from alpamayo1_5.models.base_model import SPECIAL_TOKENS, TRAJ_TOKEN

        kwargs = {}
        if config.min_pixels is not None:
            kwargs["min_pixels"] = config.min_pixels
        if config.max_pixels is not None:
            kwargs["max_pixels"] = config.max_pixels
        tokenizer = AutoProcessor.from_pretrained(config.vlm_name_or_path, **kwargs).tokenizer
        if config.traj_vocab_size is not None:
            tokenizer.add_tokens([f"<i{v}>" for v in range(config.traj_vocab_size)])
            tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")
        if config.add_special_tokens:
            tokenizer.add_tokens(list(SPECIAL_TOKENS.values()), special_tokens=True)
        else:
            tokenizer.add_tokens(list(TRAJ_TOKEN.values()), special_tokens=True)
        tokenizer.traj_token_ids = {
            k: tokenizer.convert_tokens_to_ids(v) for k, v in TRAJ_TOKEN.items()}
        return tokenizer

    @property
    def future_id0(self) -> int:
        """First discrete FUTURE-trajectory bin id."""
        return int(self.tconfig.traj_token_start_idx)

    @property
    def n_future_bins(self) -> int:
        """Future tokenizer vocab size (3000)."""
        return int(self.future_traj_tokenizer.vocab_size)

    @property
    def history_id0(self) -> int:
        """First discrete HISTORY bin id: the future region is laid out first."""
        return self.future_id0 + self.n_future_bins

    @property
    def n_history_bins(self) -> int:
        """1000 - DeltaTrajectoryTokenizer's own vocab."""
        return int(self.hist_traj_tokenizer.vocab_size)

    def probe_trajectory_tokenizer(self) -> dict:
        """Discrete trajectory tokenizer spec, BOTH regions."""
        c = self.tconfig
        tok = self.future_traj_tokenizer
        hist = self.hist_traj_tokenizer
        return {
            "vocab_size": self.n_future_bins,          # 3000 (future region)
            "seq_len": c.tokens_per_future_traj,       # 128 = 64 waypoints x 2 dims
            "future_id0": self.future_id0,             # 151669
            "history_id0": self.history_id0,           # 154669
            "hist_vocab_size": self.n_history_bins,    # 1000 (history region)
            "hist_seq_len": c.tokens_per_history_traj, # 48
            "total_bins": int(c.traj_vocab_size),      # 4000 = 3000 + 1000
            "traj_ids": dict(c.traj_token_ids),
            "binning": dict(c.traj_tokenizer_cfg),
            "hist_binning": dict(c.hist_traj_tokenizer_cfg),
            "detokenizer_fn": tok.decode,
            "tokenizer_fn": tok.encode,
            "hist_tokenize_fn": HistoryTokenize(hist),
            "hist_detokenizer_fn": hist.decode,
        }

    def probe_expert_conditioning(self) -> dict:
        """Expert<->reasoner coupling contract, from config + code."""
        from transformers import Qwen3VLConfig

        c = self.tconfig
        vlm_text = Qwen3VLConfig.from_pretrained(c.vlm_name_or_path).text_config
        expert_cfg = dict(c.expert_cfg or {})
        head_dim = expert_cfg.get("head_dim", getattr(
            vlm_text, "head_dim", vlm_text.hidden_size // vlm_text.num_attention_heads))
        return {
            "attended_layers": list(range(vlm_text.num_hidden_layers)),   # 36
            "kv_heads": vlm_text.num_key_value_heads,
            "head_dim": head_dim,
            "teacher_hidden": vlm_text.hidden_size,
            "expert_hidden": expert_cfg.get("hidden_size", vlm_text.hidden_size),  # 2048
            "parameterization": "flow_velocity",
            "interpolation": "x_t = t*x + (1-t)*noise; t=0 noise, t=1 data",
            "num_steps": (c.diffusion_cfg or {}).get("num_inference_steps", 10),
            "schedule": {"train_timestep_sampler": "not shipped in the A1.5 release",
                         "t_transform": "labeler stratifies uniformly over [0, 0.999)"},
            "action_space": dict(c.action_space_cfg),
            "expert_non_causal_attention": c.expert_non_causal_attention,
        }

    def probe_tokenizer_vs(self, student_tokenizer) -> dict:
        """Diff teacher/student text vocabs."""
        t_vocab = self.tokenizer.get_vocab()
        s_vocab = student_tokenizer.get_vocab()
        shared = set(t_vocab) & set(s_vocab)
        same_id = sum(1 for tok in shared if t_vocab[tok] == s_vocab[tok])
        report = {
            "teacher_size": len(t_vocab), "student_size": len(s_vocab),
            "shared_tokens": len(shared),
            "shared_frac_of_teacher": len(shared) / max(len(t_vocab), 1),
            "same_id_frac_of_shared": same_id / max(len(shared), 1),
        }
        report["vocab_ok"] = report["shared_frac_of_teacher"] > 0.999 and \
            report["same_id_frac_of_shared"] > 0.999
        return report

    def _prepare_inputs(self, window) -> dict:
        """Build model inputs via helper.create_message and processor.apply_chat_template."""
        from alpamayo1_5 import helper

        if self._processor is None:
            self._processor = helper.get_processor(self.tokenizer)
        data = window.data
        messages = helper.create_message(
            frames=data["image_frames"].flatten(0, 1),
            camera_indices=data["camera_indices"],
            num_frames_per_camera=int(self.cfg.data.context_frames),
        )
        tokenized = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt")
        return dict(tokenized)

    def detokenize_traj(self, tokens: torch.Tensor, window) -> torch.Tensor:
        """Region-relative bin ids in EMISSION order -> (H, 3) ego-frame waypoints."""
        d = window.data
        hx = d["ego_history_xyz"][:, -1].float().cpu()
        hr = d["ego_history_rot"][:, -1].float().cpu()
        toks = swap_action_dims(tokens.reshape(1, -1).long().cpu())
        fut_xyz, _, _ = self.future_traj_tokenizer.decode(hx, hr, toks)
        return fut_xyz[0]

    @torch.no_grad()
    def label_window(self, window, k_flow: int, topk: int,
                     max_coc: int, n_traj_samples: int,
                     greedy_traj: bool = False) -> TeacherWindowOutput:
        """One full teacher pass, under autocast."""
        dtype = torch.bfloat16 if self.cfg.teacher.dtype == "bf16" else torch.float16
        with torch.autocast(self.device, dtype=dtype):
            return self._label_window(window, k_flow, topk, max_coc,
                                      n_traj_samples, greedy_traj)

    def _label_window(self, window, k_flow: int, topk: int,
                      max_coc: int, n_traj_samples: int,
                      greedy_traj: bool = False) -> TeacherWindowOutput:
        """One full teacher pass over a preprocessed window (KV resident throughout)."""
        import copy
        from alpamayo1_5.helper import to_device
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5, ExpertLogitsProcessor
        from alpamayo1_5.models.token_utils import (
            StopAfterEOS, extract_text_tokens, replace_padding_after_eos,
        )
        from transformers import LogitsProcessorList, StoppingCriteriaList
        from transformers.generation.logits_process import LogitsProcessor

        cfg, c, model = self.cfg, self.tconfig, self.model
        data = to_device(window.data, self.device)
        tokenized = to_device(self._prepare_inputs(window), self.device)

        tokenized["input_ids"] = model.fuse_traj_tokens(
            tokenized["input_ids"],
            {"ego_history_xyz": data["ego_history_xyz"],
             "ego_history_rot": data["ego_history_rot"]},
        )

        future_start = model.special_token_ids["traj_future_start"]
        future_id0, n_future = self.future_id0, self.n_future_bins

        gen_cfg = copy.deepcopy(model.vlm.generation_config)
        gen_cfg.do_sample = True
        gen_cfg.top_p = float(cfg.teacher.get("gen_top_p", 0.98))
        gen_cfg.temperature = float(cfg.teacher.get("gen_temperature", 0.6))
        gen_cfg.max_new_tokens = max(max_coc + 8, 64)
        gen_cfg.return_dict_in_generate = True
        gen_cfg.pad_token_id = self.tokenizer.pad_token_id
        proc = LogitsProcessorList([ExpertLogitsProcessor(
            traj_token_offset=c.traj_token_start_idx,
            traj_vocab_size=c.traj_vocab_size)])
        out_a = model.vlm.generate(
            **tokenized, generation_config=gen_cfg,
            logits_processor=proc,
            stopping_criteria=StoppingCriteriaList([StopAfterEOS(eos_token_id=future_start)]),
            output_hidden_states=True, use_cache=True)
        out_a.sequences = replace_padding_after_eos(
            token_ids=out_a.sequences, eos_token_id=future_start,
            pad_token_id=self.tokenizer.pad_token_id)
        rope_deltas = model.vlm.model.rope_deltas
        prompt_len = tokenized["input_ids"].shape[1]

        feat_layers = [int(l) for l in cfg.teacher.get("feat_layers", []) or []]
        pool_len = int(cfg.teacher.get("feat_pool_len", 8))
        feats = {}
        if feat_layers and out_a.hidden_states:
            prefill_hs = out_a.hidden_states[0]  # tuple(layers+1) of (1, L, D)
            for l in feat_layers:
                h = prefill_hs[l + 1][0].float()             # (L, D); +1 skips embeddings
                seg = torch.chunk(h, pool_len, dim=0)
                feats[l] = torch.stack([s.mean(0) for s in seg])  # (pool_len, D)

        class _RestrictToFutureRegion(LogitsProcessor):
            def __call__(self, input_ids, scores):
                mask = torch.full_like(scores, float("-inf"))
                mask[:, future_id0:future_id0 + n_future] = 0.0
                return scores + mask

        gen_cfg_b = copy.deepcopy(gen_cfg)
        gen_cfg_b.min_new_tokens = c.tokens_per_future_traj      # 128
        gen_cfg_b.max_new_tokens = c.tokens_per_future_traj
        gen_cfg_b.output_logits = True
        if greedy_traj:
            gen_cfg_b.do_sample = False
            gen_cfg_b.top_p = None
            gen_cfg_b.temperature = None
        seq_a = out_a.sequences
        out_b = model.vlm.generate(
            input_ids=seq_a,
            attention_mask=torch.ones_like(seq_a),
            past_key_values=out_a.past_key_values,
            generation_config=gen_cfg_b,
            logits_processor=LogitsProcessorList([_RestrictToFutureRegion()]),
            use_cache=True)
        step_logits = torch.stack([l[0] for l in out_b.logits])      # (T_traj, V)
        logp = F.log_softmax(step_logits.float(), dim=-1)
        region_logp = logp[:, future_id0:future_id0 + n_future]      # (T_traj, 3000)
        topk_logp, topk_idx = region_logp.topk(topk, dim=-1)
        traj_token_ids = (out_b.sequences[0, seq_a.shape[1]:] - future_id0).clamp(0, n_future - 1)

        coc_text = extract_text_tokens(self.tokenizer, seq_a)["cot"][0]
        gen_a = seq_a[0, prompt_len:]
        gen_a = gen_a[(gen_a != future_start) & (gen_a != (gen_cfg.pad_token_id or -1))]
        coc_token_ids = gen_a[:max_coc].cpu()

        action_space = model.action_space
        action = action_space.traj_to_action(
            traj_history_xyz=data["ego_history_xyz"],
            traj_history_rot=data["ego_history_rot"],
            traj_future_xyz=data["ego_future_xyz"],
            traj_future_rot=data["ego_future_rot"],
        ).reshape(*action_space.get_action_space_dims())                  # (H, A) = (64, 2)

        wd = window.data
        gt_traj_token_ids = self.future_traj_tokenizer.encode(
            hist_xyz=wd["ego_history_xyz"][:, -1].float().cpu(),
            hist_rot=wd["ego_history_rot"][:, -1].float().cpu(),
            fut_xyz=wd["ego_future_xyz"][:, -1].float().cpu(),
            fut_rot=wd["ego_future_rot"][:, -1].float().cpu(),
        )[0]                                                              # (T_traj,)
        gt_traj_token_ids = swap_action_dims(gt_traj_token_ids)

        kv = out_b.past_key_values
        kv_len = kv.get_seq_length()
        offset = Alpamayo1_5._find_eos_offset(
            sequences=seq_a, eos_token_id=future_start, device=seq_a.device)   # (1,)
        n_diff = action_space.get_action_space_dims()[0]                   # 64 action tokens
        fkw = {"is_causal": False} if c.expert_non_causal_attention else {}

        def expert_v(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            """Batched expert forward against the resident KV."""
            b = x_t.shape[0]
            pos, attn = Alpamayo1_5._build_expert_pos_ids_and_attn_mask(
                offset=offset.expand(b), rope_deltas=rope_deltas.expand(b, -1),
                kv_cache_seq_len=kv.get_seq_length(), n_diffusion_tokens=n_diff,
                b_star=b, device=x_t.device,
                prefix_mask=tokenized["attention_mask"].expand(b, -1))
            emb = model.action_in_proj(x_t.to(model.dtype), t.to(model.dtype))
            if emb.dim() == 2:
                emb = emb.view(b, n_diff, -1)
            eo = model.expert(inputs_embeds=emb, position_ids=pos, past_key_values=kv,
                              attention_mask=attn, use_cache=True, **fkw)
            kv.crop(kv_len)
            pred = model.action_out_proj(eo.last_hidden_state[:, -n_diff:])
            return pred.view(b, *action_space.get_action_space_dims())

        eb = int(cfg.teacher.get("expert_batch", 4))
        kv.batch_repeat_interleave(eb)

        t_all = self.stratified_timesteps(k_flow).to(self.device)          # (K,)
        noise = torch.randn(k_flow, *action.shape, device=self.device)
        x_t_all = t_all.view(-1, 1, 1) * action[None] + (1 - t_all.view(-1, 1, 1)) * noise
        v_chunks = []
        for i in range(0, k_flow, eb):
            xt = x_t_all[i:i + eb]
            tt = t_all[i:i + eb].view(-1, 1, 1)
            if xt.shape[0] < eb:
                pad = eb - xt.shape[0]
                xt = torch.cat([xt, xt[-1:].expand(pad, -1, -1)])
                tt = torch.cat([tt, tt[-1:].expand(pad, -1, -1)])
                v_chunks.append(expert_v(xt, tt)[:eb - pad])
            else:
                v_chunks.append(expert_v(xt, tt))
        flow_v = torch.cat(v_chunks).float()

        n_s = min(n_traj_samples, eb)
        def step_fn(x, t):
            v = expert_v(x[:eb] if x.shape[0] == eb else
                         torch.cat([x, x[-1:].expand(eb - x.shape[0], -1, -1)]),
                         t if t.shape[0] == eb else
                         torch.cat([t, t[-1:].expand(eb - t.shape[0], *t.shape[1:])]))
            return v[:x.shape[0]]
        traj_samples = model.diffusion.sample(
            batch_size=n_s, step_fn=step_fn, device=self.device)           # (n_s, H, A)

        return TeacherWindowOutput(
            traj_token_ids=traj_token_ids.cpu(),
            traj_topk_idx=topk_idx.cpu(),
            traj_topk_logp=topk_logp.cpu(),
            coc_token_ids=coc_token_ids,
            coc_text=coc_text,
            feats=feats,
            flow_t=t_all.cpu().float(),
            flow_a_t=x_t_all.cpu().float(),
            flow_v=flow_v.cpu(),
            traj_samples=traj_samples.cpu().float(),
            gt_traj=action.cpu().float(),
            gt_traj_token_ids=gt_traj_token_ids.cpu(),
            gt_future_xyz=torch.as_tensor(window.gt_future_xyz),
        )

    def stratified_timesteps(self, k: int) -> torch.Tensor:
        """Low-discrepancy coverage of the schedule: one uniform draw per bin of [0, 0.999) split into k bins."""
        u = torch.rand(k)
        return 0.999 * (torch.arange(k) + u) / k
