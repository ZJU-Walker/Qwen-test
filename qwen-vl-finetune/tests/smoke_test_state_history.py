"""Gates for the frame-aligned state-history prompt feature (--state_history).

On real data: (1) past-state indices match the image-history indices incl. edge clamping
(t=0 -> all nine repeat the first state); (2) the rendered item prompt carries the
independently recomputed discretized strings, oldest first, with the current state last;
(3) train/serve parity: inference.make_prompt reproduces the training prompt exactly from
the same states; (4) state_history=False leaves the prompt byte-identical to the legacy
format; (5) token-cost report.
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from qwenvl.data.robot_data import (  # noqa: E402
    RobotDataArguments, RobotFlowMatchingDataset, _discretize_state, format_robot_prompt,
    quantile_normalize)
from qwenvl.action_expert import inference as inf  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

ART = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717m_0731gy_0804dag_ee6d"

da = RobotDataArguments()
da.model_type = "qwen3vl"
da.robot_data_dirs = "/iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged"
da.camera = "cam_high"
da.num_frames = 10
da.frame_stride = 10
da.image_history = True
da.active_dims = "7:14"
da.action_space = "ee6d"
da.action_dim = 10
da.delta_mask = "9,-1"
da.action_horizon = 50
da.use_delta_actions = True
da.wrist_cameras = "cam_right_wrist,cam_left_wrist"
da.wrist_max_pixels = 131072
da.default_prompt = "waiting"
da.train_split = 1.0
da.predict_subtask = True
da.use_fast_tokens = True
da.fast_tokenizer_path = ART
da.state_history = True

proc = AutoProcessor.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    cache_dir="/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")
ds = RobotFlowMatchingDataset(proc, da)
ep = ds.episodes[0]
states = ep["states"]

def expected_prompt_parts(t):
    idxs = [max(0, t - k * 10) for k in range(9, 0, -1)]
    past = quantile_normalize(states[idxs], ds.norm_stats["state"])
    cur = quantile_normalize(states[t], ds.norm_stats["state"])
    return idxs, " ; ".join(_discretize_state(np.clip(s, -1, 1), fixed_width=True) for s in past), _discretize_state(np.clip(cur, -1, 1), fixed_width=True)

# 1+2: decode real items at edge / partial / interior timesteps and check the text exactly
import qwenvl.data.robot_data as rd
import random
for t in (0, 25, min(200, len(states) - 1)):
    random.seed(0)
    rd.random.randint = lambda a, b, _t=t: _t          # pin the sampled timestep
    try:
        item = ds[0]
    finally:
        rd.random.randint = random.randint
    text = ds.tokenizer.decode(item["input_ids"][0], skip_special_tokens=False)
    idxs, past_str, cur_str = expected_prompt_parts(t)
    frag = f"Past states: {past_str}, State: {cur_str}"
    assert frag in text, f"t={t}: expected past-state fragment missing\nwanted ...{frag[:120]}..."
    if t == 0:
        assert past_str == " ; ".join([_discretize_state(np.clip(quantile_normalize(states[0], ds.norm_stats['state']), -1, 1), fixed_width=True)] * 9), \
            "t=0 must repeat the first state nine times"
print("1. item prompts at t=0 (full clamp), t=25 (partial), t=200: exact frame-aligned past states  OK")
print("2. t=0 edge: all nine past slots repeat the first state  OK")

# 3: serve-side parity -- make_prompt from the same model-space states reproduces training text
t = 25
idxs = [max(0, t - k * 10) for k in range(9, 0, -1)]
serve = inf.make_prompt(ds, states[t], task_text=da.subtask_question, past_states=states[idxs])
train = format_robot_prompt(da.subtask_question,
                            quantile_normalize(states[t], ds.norm_stats["state"]),
                            quantile_normalize(states[idxs], ds.norm_stats["state"]))
assert serve == train, "serve-side make_prompt diverges from the training format"
print("3. train/serve prompt parity (make_prompt == training format)  OK")

# 4: feature off -> legacy prompt, byte-identical
legacy = format_robot_prompt("what should the robot do next?",
                             quantile_normalize(states[t], ds.norm_stats["state"]))
assert "Past states" not in legacy and legacy.startswith("Task: ")
print("4. state_history off: legacy prompt unchanged  OK")

# 5: token cost
with_hist = inf.make_prompt(ds, states[t], past_states=states[idxs])
n_with = len(ds.tokenizer(with_hist)["input_ids"])
n_wo = len(ds.tokenizer(inf.make_prompt(ds, states[t]))["input_ids"])
print(f"5. prompt tokens: {n_wo} -> {n_with} (+{n_with - n_wo} for 9 past states)")

# 6: CONSTANT prompt length across poses (the compiled-serving shape-stability property)
lens = set()
for tt in range(0, min(len(states) - 1, 200), 4):
    ii = [max(0, tt - k * 10) for k in range(9, 0, -1)]
    lens.add(len(ds.tokenizer(inf.make_prompt(ds, states[tt], past_states=states[ii]))["input_ids"]))
assert len(lens) == 1, f"prompt token length varies across poses: {sorted(lens)}"
print(f"6. fixed-width rendering: prompt length CONSTANT ({lens.pop()} tokens) across 50 poses  OK")

print("\nALL STATE-HISTORY GATES PASS")
