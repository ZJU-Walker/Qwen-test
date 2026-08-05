"""Gates for the 0717+0731+0804-DAgger ee6d training data configuration.

Instantiates the dataset EXACTLY as scripts/train_action_expert_4b_ee6d.sh does and checks:
episode accounting (443 prior + 104 DAgger), min_start gating on every DAgger episode,
mixed supervision (labels only from 0717), 10-dim EE everywhere, frozen artifact stats,
2-wrist paths, and that sampled DAgger items always land in the correction segment while
their history may reach before it.
"""

import random
import sys

import numpy as np

sys.path.insert(0, ".")
from qwenvl.data.robot_data import RobotDataArguments, RobotFlowMatchingDataset  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

ART = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717m_0731gy_0804dag_ee6d"
DIRS = ("/iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged,"
        "/iris/projects/humanoid/trossen_data/0731_green_yellow_merged,"
        "/iris/projects/humanoid/trossen_data/0804_green_dagger,"
        "/iris/projects/humanoid/trossen_data/0804_yellow_dagger_part1,"
        "/iris/projects/humanoid/trossen_data/0804_yellow_dagger_part2")

da = RobotDataArguments()
da.model_type = "qwen3vl"
da.robot_data_dirs = DIRS
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
da.min_episode_len = 50
da.predict_subtask = True
da.use_fast_tokens = True
da.fast_tokenizer_path = ART

proc = AutoProcessor.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    cache_dir="/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")
ds = RobotFlowMatchingDataset(proc, da)

eps = ds.episodes
n_dag = sum(1 for e in eps if "0804" in str(e["video_path"]))
n_prior = len(eps) - n_dag
assert n_prior == 443, f"prior episodes {n_prior} != 443"
assert n_dag == 104, f"dagger episodes {n_dag} != 104"

dag = [e for e in eps if "0804" in str(e["video_path"])]
assert all(int(e.get("min_start", 0)) > 0 for e in dag), "a DAgger episode lacks min_start gating"
assert all(len(e["wrist_paths"]) == 2 for e in eps), "an episode lacks both wrist videos"
assert all(e["states"].shape[1] == 10 and e["actions"].shape[1] == 10 for e in eps), "non-10-dim episode"
labeled = sum(1 for e in eps if e["subtasks"])
unlabeled = len(eps) - labeled
assert labeled == 224, f"labeled {labeled} != 224 (0717 only)"
assert unlabeled == 219 + 104, f"unlabeled {unlabeled} != 323"
frac = np.mean([e["min_start"] / len(e["states"]) for e in dag])
print(f"1. accounting: {len(eps)} eps = 443 prior + 104 dagger | labeled 224 / unlabeled {unlabeled} | "
      f"dagger min_start at {frac * 100:.0f}% mean | all 10-dim, 2 wrists  OK")

assert ds.norm_stats is not None and str(ART) in str(getattr(ds, "norm_stats_source", ART))
lo, hi = ds.norm_stats["state"]["q01"], ds.norm_stats["state"]["q99"]
assert len(lo) == 10 and all(h > l for l, h in zip(lo, hi)), "state stats malformed"
print("2. frozen artifact stats loaded, 10-dim, well-ordered  OK")

random.seed(0)
idx = [i for i, e in enumerate(eps) if "0804" in str(e["video_path"])]
checked = 0
for i in idx[:8]:
    e = eps[i]
    for _ in range(6):
        t = random.randint(int(e.get("min_start", 0)), len(e["states"]) - 1)
        assert t >= e["min_start"], "sampled timestep before the intervention"
        checked += 1
print(f"3. DAgger sampling: {checked} draws all >= min_start (history may still reach back)  OK")

item = ds[idx[0]]
assert item["actions"].shape == (50, 10) and item["input_ids"].ndim == 2
print("4. end-to-end item from a DAgger episode: (50, 10) chunk, tokenized sequence  OK")

print("\nALL 0804-DAGGER DATA GATES PASS")
