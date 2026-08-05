"""DAgger-integration smoke test (CPU, no GPU needed).

Verifies, against the REAL datasets:
1. min_start: every DAgger episode's sampling floor equals the first is_intervention
   frame (== press_frame in meta/dagger_interventions.json); plain episodes get 0.
2. _build_item actually samples t >= min_start (random.randint is observed) and the
   supervised subtask text carries the correct block color per dataset.
3. compute_norm_stats excludes pre-intervention chunks (unit check on toy episodes).
4. Norm-stats resolution: the 3-dir dataset loads the FROZEN stats from the new FAST
   tokenizer dir and does NOT touch the legacy 0717 stats file; the legacy 1-dir
   dataset still cache-hits the 0717 file byte-for-byte (old serving path intact).

Run:  python tests/smoke_test_dagger_data.py
"""
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

from qwenvl.action_expert.inference import build_dataset
from qwenvl.data.robot_data import compute_norm_stats, make_action_chunk

DATA = "/iris/projects/humanoid/trossen_data"
MERGED = f"{DATA}/0717_green_yellow_block_mem_merged"
GREEN = f"{DATA}/0730_green_block_mem_dagger"
YELLOW = f"{DATA}/0730_yellow_block_mem_dagger"
NEW_TOK = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717merged_0730dagger"
OLD_TOK = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717merged"
LEGACY_STATS = Path(MERGED) / "qwen_action_expert_norm_stats.json"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---- 3. compute_norm_stats gating (toy, exact) ----
rng = np.random.default_rng(0)
toy = [{"states": rng.normal(size=(10, 2)).astype(np.float32),
        "actions": rng.normal(size=(10, 2)).astype(np.float32), "min_start": 4}]
stats = compute_norm_stats(toy, horizon=2, delta_mask=None)
expect = np.concatenate(
    [make_action_chunk(toy[0]["actions"], toy[0]["states"][t], t, 2, None) for t in range(4, 10)]
).reshape(-1, 2)
assert np.allclose(stats["actions"]["mean"], expect.mean(axis=0), atol=1e-6), "gating broke chunk stats"
assert np.allclose(stats["state"]["mean"], toy[0]["states"][4:].mean(axis=0), atol=1e-6), "gating broke state stats"
print("3. compute_norm_stats excludes pre-min_start frames  OK")

legacy_hash_before = sha(LEGACY_STATS)

# ---- build the 3-dir DAgger-mix dataset (new tokenizer artifact) ----
ds = build_dataset(fast_tok=NEW_TOK, data_dirs=f"{MERGED},{GREEN},{YELLOW}")
assert len(ds.episodes) == 224 + 63 + 63, f"episode count {len(ds.episodes)}"

# ---- 1. min_start vs dagger_interventions.json ----
iv_green = json.load(open(f"{GREEN}/meta/dagger_interventions.json"))
iv_yellow = json.load(open(f"{YELLOW}/meta/dagger_interventions.json"))
assert all(ep["min_start"] == 0 for ep in ds.episodes[:224]), "0717 episode got a nonzero min_start"
for base, iv, color in ((224, iv_green, "green"), (224 + 63, iv_yellow, "yellow")):
    for i in range(63):
        ep = ds.episodes[base + i]
        press = iv[f"episode_{i:06d}"]["interventions"][0]["press_frame"]
        assert ep["min_start"] == press, f"{color} ep {i}: min_start {ep['min_start']} != press {press}"
        sub = ds._subtask_at(ep, ep["min_start"])
        sub_end = ds._subtask_at(ep, len(ep["states"]) - 1)
        assert sub == sub_end == f"pick up {color} block", f"{color} ep {i}: subtask {sub!r} / {sub_end!r}"
        assert ds._subtask_at(ep, 0) == "waiting", "pre-press frames should fall back to default"
print("1. all 126 DAgger min_starts == press_frame; subtask labels correct per color  OK")

# ---- 2. _build_item samples within [min_start, n-1] and supervises the right text ----
calls = []
orig_randint = random.randint
random.randint = lambda lo, hi: (calls.append((lo, hi)) or orig_randint(lo, hi))
try:
    for idx, color in ((224 + 5, "green"), (224 + 63 + 5, "yellow")):
        ep = ds.episodes[idx]
        calls.clear()
        item = ds[idx]
        lo, hi = calls[0]
        assert lo == ep["min_start"] and hi == len(ep["states"]) - 1, f"sample bounds {(lo, hi)}"
        text = ds.tokenizer.decode(np.asarray(item["input_ids"]).ravel().tolist())
        assert f"pick up {color} block" in text, f"{color} not in supervised text"
finally:
    random.randint = orig_randint
print("2. _build_item bounds = [press_frame, end] and supervised text has the right color  OK")

# ---- 4. norm-stats resolution ----
frozen = json.load(open(Path(NEW_TOK) / "norm_stats.json"))
assert ds.norm_stats == frozen, "3-dir dataset did not adopt the tokenizer-dir frozen stats"
assert sha(LEGACY_STATS) == legacy_hash_before, "legacy 0717 stats file was modified!"

ds_old = build_dataset(fast_tok=OLD_TOK, data_dirs=MERGED)
legacy = json.load(open(LEGACY_STATS))
assert ds_old.norm_stats == legacy, "legacy 1-dir dataset no longer cache-hits its stats"
assert sha(LEGACY_STATS) == legacy_hash_before, "legacy stats file rewritten by cache-hit path"
print("4. frozen stats adopted for the mix; legacy 0717 stats untouched and still served  OK")

# ---- 5. clobber guard: 3-dir mix + OLD tokenizer (no norm_stats.json) must hard-error
# instead of recomputing over the legacy 0717 file (the review's major finding).
try:
    build_dataset(fast_tok=OLD_TOK, data_dirs=f"{MERGED},{GREEN},{YELLOW}")
    raise AssertionError("mismatched legacy stats were silently accepted or recomputed")
except ValueError as e:
    assert "Refusing to overwrite" in str(e), f"unexpected error: {e}"
assert sha(LEGACY_STATS) == legacy_hash_before, "clobber guard still rewrote the legacy file"
print("5. old-tokenizer + new-mix config hard-errors; legacy stats file untouched  OK")

print("\nALL DAGGER DATA TESTS PASS")
