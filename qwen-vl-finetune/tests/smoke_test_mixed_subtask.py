"""Mixed subtask supervision smoke test (CPU, real data).

The 0717 merged set has subtask labels; 0731_green_yellow_merged has none. Training on
both must give: labeled episodes an assistant turn supervising the subtask text (exactly
as before), unlabeled episodes NO assistant turn (VLM supervised through FAST alone),
never a supervised "waiting" fallback -- and the expert-insulation mask on unlabeled
samples must match what insulated serving masks (the 3-token generation header).

Run:  python tests/smoke_test_mixed_subtask.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

from qwenvl.action_expert.inference import build_dataset, subtask_token_mask_for

DATA = "/iris/projects/humanoid/trossen_data"
MERGED = f"{DATA}/0717_green_yellow_block_mem_merged"
NEW = f"{DATA}/0731_green_yellow_merged"
NEW_TOK = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717merged_0731gy"
IGNORE = -100

ds = build_dataset(fast_tok=NEW_TOK, data_dirs=f"{MERGED},{NEW}")
assert len(ds.episodes) == 224 + 224, f"episode count {len(ds.episodes)}"
assert all(ep["subtasks"] for ep in ds.episodes[:224]), "a 0717 episode lost its labels"
assert all(not ep["subtasks"] for ep in ds.episodes[224:]), "a 0731 episode has labels?"
assert all(ep["min_start"] == 0 for ep in ds.episodes), "min_start crept into non-DAgger data"

def supervised_non_fast(item):
    """Token ids receiving CE loss, excluding the appended FAST region."""
    labels, fast = item["labels"][0], item["fast_token_mask"][0]
    return labels[(labels != IGNORE) & ~fast].tolist()

# ---- labeled episode (0717): supervised subtask answer, unchanged behavior ----
item = ds[5]
sup = ds.tokenizer.decode(supervised_non_fast(item))
assert ("block" in sup) or ("waiting" in sup), f"unexpected supervised text {sup!r}"
assert item["subtask_token_mask"].any(), "labeled episode lost its insulation mask"
print(f"1. labeled 0717 episode supervises {sup.strip()!r}  OK")

# ---- unlabeled episode (0731): NO language supervision, FAST still supervised ----
item = ds[224 + 5]
assert supervised_non_fast(item) == [], "unlabeled episode got language supervision"
fast_labels = item["labels"][0][item["fast_token_mask"][0]]
# start marker is the unsupervised "prompt" (-100); FAST tokens + end marker supervised.
assert len(fast_labels) > 1 and fast_labels[0] == IGNORE and (fast_labels[1:] != IGNORE).all(), \
    "unlabeled episode lost FAST supervision"
ids = item["input_ids"][0]
n_fast = int(item["fast_token_mask"].sum())
pre_fast = ids[: len(ids) - n_fast]
text_tail = ds.tokenizer.decode(pre_fast[-3:].tolist())
assert text_tail == "<|im_start|>assistant\n", f"sequence should end with the generation header, got {text_tail!r}"
print("2. unlabeled 0731 episode: zero language supervision, FAST intact, generation-header form  OK")

# ---- insulation-mask parity with serving on the unlabeled form ----
serve_mask = subtask_token_mask_for(pre_fast.unsqueeze(0))
train_mask_pre_fast = item["subtask_token_mask"][0][: len(pre_fast)]
assert torch.equal(train_mask_pre_fast, serve_mask[0]), "train/serve insulation masks differ"
assert not item["subtask_token_mask"][0][len(pre_fast):].any(), "FAST region leaked into subtask mask"
assert int(serve_mask.sum()) == 3, f"header mask should cover 3 tokens, got {int(serve_mask.sum())}"
print("3. unlabeled-sample insulation mask == serving's header mask (3 tokens)  OK")

# ---- frozen stats adopted from the new tokenizer artifact ----
import json
frozen = json.load(open(f"{NEW_TOK}/norm_stats.json"))
assert ds.norm_stats == frozen, "mix did not adopt the tokenizer-dir frozen stats"
print("4. frozen stats from fast_tokenizer_trossen_0717merged_0731gy adopted  OK")

print("\nALL MIXED-SUBTASK TESTS PASS")
