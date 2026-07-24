"""Phase-2 GPU test on the real 4B model (randomized adaRMS gates so attention truly mixes):

1. INSULATED (expert_attends_subtask=False): expert actions are IDENTICAL for two different
   subtask texts (same obs/noise) -- the expert provably cannot see the subtask turn.
2. NON-insulated (True): the same two sequences give DIFFERENT actions (the conditioning
   really was flowing through the subtask tokens).
3. Training forward with subtask_token_mask + insulation runs, loss finite.
4. Insulated expert on [user turn + assistant turn, masked] == insulated expert on the
   generation-prompt sequence [user turn + header, masked] -- the serve-time skip path sees
   the same observation set as training.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

from qwenvl.action_expert.inference import (
    ACTION_DIM, ACTION_HORIZON, build_dataset, load_model, make_prompt, subtask_token_mask_for, templatize,
)

DEVICE = "cuda"
FAST_TOK = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0714merged"
DATA_DIRS = "/iris/projects/humanoid/trossen_data/0714_green_yellow_block_mem_merged"
torch.manual_seed(0)

ds = build_dataset(fast_tok=FAST_TOK, data_dirs=DATA_DIRS, image_history=True, predict_subtask=True)

def build(insulated):
    m = load_model.__wrapped__ if hasattr(load_model, "__wrapped__") else load_model
    # random-weight build via the inference loader minus the checkpoint: replicate load_model
    # but skip load_state_dict (weights don't matter; gates must be nonzero for real mixing)
    from transformers import Qwen3VLForConditionalGeneration
    from qwenvl.action_expert import ActionExpertConfig, Qwen3VLWithActionExpert
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct", cache_dir=os.environ["HF_HOME"],
        dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    tc = vlm.config.text_config
    cfg = ActionExpertConfig(num_hidden_layers=tc.num_hidden_layers,
                             num_key_value_heads=tc.num_key_value_heads, head_dim=tc.head_dim,
                             hidden_size=1024, intermediate_size=4096, num_attention_heads=16,
                             action_dim=ACTION_DIM, action_horizon=ACTION_HORIZON)
    model = Qwen3VLWithActionExpert(vlm, cfg, train_vlm=False, expert_attends_subtask=not insulated)
    with torch.no_grad():
        for mod in model.action_expert.modules():
            if getattr(mod, "dense", None) is not None:
                torch.nn.init.normal_(mod.dense.weight, std=0.02)
                torch.nn.init.normal_(mod.dense.bias, std=0.02)
    model.action_expert.to(torch.bfloat16)
    model.vlm.resize_token_embeddings(ds.vlm_vocab_size)
    model.vlm.get_input_embeddings().to(torch.bfloat16)
    model.vlm.lm_head.to(torch.bfloat16)
    return model.eval().to(DEVICE)

rng = np.random.default_rng(0)
frames = [Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)) for _ in range(10)]
wrist = [Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))]
state = rng.standard_normal(7).astype(np.float32)
prompt = make_prompt(ds, state)
noise = torch.randn(1, ACTION_HORIZON, ACTION_DIM)

mm_a = templatize(ds, frames, wrist, prompt, "pick up green block", DEVICE)
mm_b = templatize(ds, frames, wrist, prompt, "pick up yellow block", DEVICE)
mm_gen = templatize(ds, frames, wrist, prompt, None, DEVICE)  # generation header, no answer


def run(model, mm):
    mask = subtask_token_mask_for(mm["input_ids"]) if not model.expert_attends_subtask else None
    with torch.inference_mode():
        return model.sample_actions(
            input_ids=mm["input_ids"], attention_mask=torch.ones_like(mm["input_ids"]),
            position_ids=mm["position_ids"], pixel_values=mm["pixel_values"],
            image_grid_thw=mm["image_grid_thw"], pixel_values_videos=mm["pixel_values_videos"],
            video_grid_thw=mm["video_grid_thw"], noise=noise.clone().to(DEVICE), num_steps=3,
            subtask_token_mask=mask)


torch.manual_seed(1)
model_ins = build(insulated=True)
a = run(model_ins, mm_a)
b = run(model_ins, mm_b)
d_ins = (a - b).abs().max().item()
assert d_ins == 0.0, f"insulated expert saw the subtask! maxdiff={d_ins}"
print(f"1. INSULATED: green vs yellow subtask -> IDENTICAL actions (maxdiff {d_ins})  OK")

g = run(model_ins, mm_gen)
d_gen = (a - g).abs().max().item()
# The two sequences differ in LENGTH (full turn vs header only), so SDPA kernels tile
# differently -> bf16 noise ~5e-3 even though the masked keys are excluded from softmax.
# Content-invariance is what matters and test 1 proves it EXACTLY (same-length sequences).
assert d_gen < 2e-2, f"skip-path sequence diverged beyond bf16 noise: {d_gen}"
print(f"2. INSULATED: full turn (masked) ~= generation-header-only (masked)  OK "
      f"(maxdiff {d_gen:.2e}, bf16 length-tiling noise)")

# training forward with the batched mask
item = {"input_ids": mm_a["input_ids"], "position_ids": mm_a["position_ids"]}
stm = subtask_token_mask_for(mm_a["input_ids"])
out = model_ins(input_ids=mm_a["input_ids"], actions=noise.clone().to(DEVICE),
                attention_mask=torch.ones_like(mm_a["input_ids"]), position_ids=mm_a["position_ids"],
                pixel_values=mm_a["pixel_values"], image_grid_thw=mm_a["image_grid_thw"],
                pixel_values_videos=mm_a["pixel_values_videos"], video_grid_thw=mm_a["video_grid_thw"],
                subtask_token_mask=stm)
assert torch.isfinite(out.loss)
print(f"3. training forward w/ insulation: loss finite ({out.loss.item():.3f})  OK")

del model_ins
torch.cuda.empty_cache()
torch.manual_seed(1)  # same expert init -> comparable
model_att = build(insulated=False)
a2 = run(model_att, mm_a)
b2 = run(model_att, mm_b)
d_att = (a2 - b2).abs().max().item()
assert d_att > 1e-4, f"non-insulated expert ignored the subtask?! maxdiff={d_att}"
print(f"4. NON-insulated: green vs yellow -> DIFFERENT actions (maxdiff {d_att:.2e})  OK")

print("\nALL SUBTASK-INSULATION GPU TESTS PASS")
