"""Phase-2 CPU test: subtask_token_mask correctness in the data pipeline + inference helper.

1. dataset emits subtask_token_mask covering EXACTLY '<|im_start|>assistant\n...<|im_end|>\n'
   (decode the masked span and check), False on FAST tokens and everywhere else
2. collator batches/pads it
3. inference-side subtask_token_mask_for agrees with the dataset mask on the non-FAST part
"""
import os
import sys

import torch

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from qwenvl.action_expert.inference import build_dataset, subtask_token_mask_for
from qwenvl.data.robot_data import RobotActionDataCollator

FAST_TOK = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0714merged"
DATA_DIRS = "/iris/projects/humanoid/trossen_data/0714_green_yellow_block_mem_merged"

ds = build_dataset(fast_tok=FAST_TOK, data_dirs=DATA_DIRS, image_history=True, predict_subtask=True)
item = ds._build_item(0)
ids, mask, fmask = item["input_ids"], item["subtask_token_mask"], item["fast_token_mask"]
assert mask.shape == ids.shape and mask.dtype == torch.bool

# 1. the masked span decodes to exactly the assistant turn
span = ids[0][mask[0]].tolist()
text = ds.tokenizer.decode(span)
print(f"masked span ({len(span)} tokens): {text!r}")
assert text.startswith("<|im_start|>assistant\n"), text
assert text.endswith("<|im_end|>\n"), text
# contiguous single block
idxs = mask[0].nonzero().flatten()
assert (idxs[1:] - idxs[:-1] == 1).all(), "mask not contiguous"
# never overlaps FAST; everything after the turn is FAST only
assert not (mask & fmask).any(), "subtask mask overlaps FAST tokens"
last = int(idxs[-1])
assert fmask[0, last + 1:].all(), "expected only FAST tokens after the assistant turn"
# the labels-supervised region (answer + <|im_end|>\n) lies inside the mask
lab = item["labels"]
sup_nonfast = (lab[0] != -100) & ~fmask[0]
assert bool(mask[0][sup_nonfast].all()), "supervised subtask tokens must be inside the mask"
print("1. dataset mask covers exactly the assistant turn  OK")

# 2. collator batches it (two items of different lengths)
item2 = ds._build_item(1)
batch = RobotActionDataCollator(ds.tokenizer)([item, item2])
assert batch["subtask_token_mask"].shape == batch["input_ids"].shape
assert batch["subtask_token_mask"].dtype == torch.bool
print("2. collator batches subtask_token_mask  OK")

# 3. inference helper agrees on the pre-FAST portion of the sequence (an inference sequence
#    is exactly that: user turn + assistant turn, no FAST appended)
n_obs = int((~fmask[0]).sum())
inf_mask = subtask_token_mask_for(ids[:, :n_obs])
assert torch.equal(inf_mask[0], mask[0, :n_obs]), "inference mask != dataset mask"
print("3. inference-side subtask_token_mask_for matches the dataset mask  OK")

print("\nALL SUBTASK-INSULATION CPU TESTS PASS")
