"""Probe the teacher's trajectory tokenizer and expert layers; write them to the cache."""
import json, sys, torch
from pathlib import Path
sys.path.insert(0, "src")
from distill.config import load_config
from distill.teacher.wrapper import TeacherWrapper

cfg = load_config()
cache = Path(cfg.paths.cache_root); cache.mkdir(parents=True, exist_ok=True)
teacher = TeacherWrapper(cfg, load_model=False)

spec = teacher.probe_trajectory_tokenizer()
torch.save(spec, cache / "traj_tokenizer_spec.pt")
cond = teacher.probe_expert_conditioning()
torch.save(cond, cache / "expert_layers.pt")
print("traj tokenizer:", {k: spec[k] for k in
                          ("vocab_size", "seq_len", "future_id0", "history_id0",
                           "hist_vocab_size", "hist_seq_len", "total_bins")})
assert spec["vocab_size"] == 3000 and spec["seq_len"] == 128, \
    f"future-trajectory geometry changed: {spec['vocab_size']} bins x {spec['seq_len']} tokens"
assert spec["hist_vocab_size"] == 1000 and spec["hist_seq_len"] == 48, \
    f"history-trajectory geometry changed: {spec['hist_vocab_size']} bins x {spec['hist_seq_len']} slots"
assert spec["total_bins"] == spec["vocab_size"] + spec["hist_vocab_size"], \
    f"the two regions do not tile traj_vocab_size ({spec['total_bins']})"
print("expert attends", len(cond["attended_layers"]), "layers (identity KV) | param:",
      cond["parameterization"], "|", cond["interpolation"])

try:
    from transformers import AutoTokenizer
    edge_tok = AutoTokenizer.from_pretrained(cfg.paths.student_repo)
    report = teacher.probe_tokenizer_vs(edge_tok)
    (cache / "vocab_report.json").write_text(json.dumps(report, indent=2))
    print("vocab overlap:", report)
except Exception as e:
    print(f"vocab probe skipped (student tokenizer unavailable): {e}")

print("OK")
