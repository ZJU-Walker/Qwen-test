"""Gates for the human-video-prompt training data configuration (0717+0731 gated + demo pools).

Instantiates the dataset EXACTLY as scripts/train_action_expert_4b_humanprompt.sh does
and checks: waiting-segment gating (per-dataset episode counts computed from the labels
files + parquet lengths -- 0717 ep 149 dropped (mislabeled), 0731 degenerate recordings
dropped (waiting-only labels), min_start > 0 with history clamped AT the cut -- not
before it, unlike DAgger), pool accounting with the eval holdout, per-sample pairing
(same-color demo, fresh draw), the two-video sequence structure (2 video_grid_thw rows,
"Human demonstration:"/"Robot view:" markers, no color word in the user turn), gated +
color-consistent subtask supervision, and frozen 10-dim EE artifact stats.

Prerequisite: 0731 labels installed (python tests/gen_0731_subtask_labels.py --install).

Run on a compute node (CPU is fine, ~2-3 min: it decodes a few videos):
    cd /iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune
    python tests/smoke_test_human_prompt_data.py
"""

import random
import sys

import numpy as np

sys.path.insert(0, ".")
from qwenvl.data.robot_data import RobotDataArguments, RobotFlowMatchingDataset  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

ART = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717m0731_ee6d_gated"
DATA = ("/iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged,"
        "/iris/projects/humanoid/trossen_data/0731_green_yellow_merged")
POOLS = ("green=/iris/projects/humanoid/trossen_data/green_human_prompt,"
         "yellow=/iris/projects/humanoid/trossen_data/yellow_human_prompt")

da = RobotDataArguments()
da.model_type = "qwen3vl"
da.robot_data_dirs = DATA
da.camera = "cam_high"
da.num_frames = 10
da.frame_stride = 10
da.image_history = True
da.state_history = True
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
da.subtask_question = "which colored block did the human demonstrate picking up?"
da.use_fast_tokens = True
da.fast_tokenizer_path = ART
da.skip_leading_subtask = "waiting"
da.human_prompt_dirs = POOLS
da.human_prompt_stride = 10
da.human_prompt_max_frames = 12
da.human_prompt_holdout = 4

proc = AutoProcessor.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    cache_dir="/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")
ds = RobotFlowMatchingDataset(proc, da)

# --- gating: expected counts derived from the labels files + parquet lengths, exactly
# mirroring _scan_episodes: keep an episode iff its labels start with 'waiting' AND have
# a pick segment after it AND the episode has >= min_episode_len frames.
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import pyarrow.parquet as papq  # noqa: E402

expected = {}
for d in DATA.split(","):
    root = Path(d)
    labels = json.load(open(root / "videos" / "chunk-000" / "subtask_labels.json"))
    n = 0
    for key, segs in labels.items():
        if not (segs and segs[0]["task"] == "waiting" and len(segs) > 1):
            continue
        pq = root / "data" / "chunk-000" / key.replace(".mp4", ".parquet")
        if papq.ParquetFile(pq).metadata.num_rows < da.min_episode_len:
            continue
        n += 1
    expected[root.name] = n
actual = {name: sum(1 for e in ds.episodes if f"/{name}/" in e["video_path"])
          for name in expected}
assert actual == expected, f"per-dataset episode counts {actual} != expected {expected}"
assert len(ds.episodes) == sum(expected.values())
assert all(int(e["min_start"]) > 0 for e in ds.episodes), "an episode lacks the waiting gate"
assert all(int(e["hist_min"]) == int(e["min_start"]) for e in ds.episodes), \
    "hist_min must equal min_start under skip_leading_subtask (history clamped AT the cut)"
assert not any("0717_green_yellow_block_mem_merged" in e["video_path"]
               and "episode_000149" in e["video_path"] for e in ds.episodes), "0717 ep 149 not dropped"
ms = np.array([e["min_start"] for e in ds.episodes])
# Cut rule = sustained-motion onset + 10 (gen_0731_subtask_labels.py): pointing phase
# ends frame ~40-90, onset+10 lands ~50-120, dataset means ~80.
assert 40 < ms.mean() < 120, f"suspicious mean cut frame {ms.mean():.0f}"
print(f"gating OK: {len(ds.episodes)} episodes {actual}, "
      f"cut frames min/mean/max {ms.min()}/{ms.mean():.0f}/{ms.max()}")

# --- pools: 31 green / 32 yellow minus 4 held out each ---
assert sorted(ds.human_prompt_pools) == ["green", "yellow"]
assert len(ds.human_prompt_pools["green"]) == 27, "green pool != 31 - 4 holdout"
assert len(ds.human_prompt_pools["yellow"]) == 28, "yellow pool != 32 - 4 holdout"

# --- history clamp: frames at t=min_start must not reach before the cut ---
ep = ds.episodes[0]
frames_at_cut = ds._extract_frames(ep, int(ep["min_start"]))
assert len(frames_at_cut) == 10
# All 10 history frames at the cut are the SAME frame (fully clamped): identical pixels.
arrs = [np.asarray(f) for f in frames_at_cut]
assert all(np.array_equal(arrs[0], a) for a in arrs[1:]), \
    "history at t=min_start should be 10 copies of the cut frame (clamp broken?)"
print("history clamp OK: at t=cut all 10 frames are the cut frame")

# --- pairing + sequence structure over a few samples ---
random.seed(0)
tok = ds.tokenizer
for trial in range(3):
    idx = random.randrange(len(ds.episodes))
    item = ds[idx]
    vg = item["video_grid_thw"]
    assert vg.shape[0] == 2, f"expected 2 videos (prompt + history), got {vg.shape[0]}"
    text = tok.decode(item["input_ids"][0])
    assert "Human demonstration:" in text and "Robot view:" in text, "markers missing"
    # The USER turn must not name the color (that would leak the task around the demo
    # video); the color appears only in the supervised assistant subtask answer.
    user_turn = text.split("assistant")[0]
    assert "green" not in user_turn and "yellow" not in user_turn, \
        f"color word leaked into the user turn:\n{user_turn[-400:]}"
    ep = ds.episodes[idx]
    color = ds._human_prompt_key(ep)
    assert f"pick up {color} block" in text, "assistant subtask answer missing/mismatched"
    n_img = int(item["image_grid_thw"].shape[0])
    print(f"sample {trial}: seq {item['input_ids'].shape[1]} tokens | videos {vg[:, 0].tolist()} "
          f"(temporal patches) | {n_img} wrist stills | paired color: {color}")

# --- fresh pairing: same episode, repeated draws -> more than one distinct clip ---
ep = ds.episodes[0]
pool = ds.human_prompt_pools[ds._human_prompt_key(ep)]
paths = {random.choice(pool) for _ in range(8)}
assert len(paths) > 1, "prompt pool sampling looks degenerate"

# --- frozen artifact stats: 10-dim and fit on THIS exact gated two-dataset mix ---
st = ds.norm_stats
assert len(st["actions"]["q01"]) == 10 and len(st["state"]["q01"]) == 10
assert st["meta"].get("action_space") == "ee6d"
assert st["meta"].get("robot_data_dirs") == DATA, \
    f"FAST/stats fit used wrong data mix: {st['meta'].get('robot_data_dirs')!r}"
expected_min_start = sum(int(e["min_start"]) for e in ds.episodes)
assert st["meta"].get("min_start_frames") == expected_min_start, \
    (f"FAST/stats gated fingerprint {st['meta'].get('min_start_frames')} != "
     f"dataset {expected_min_start}; artifact was not fit on this exact mix")
assert st["meta"].get("min_episode_len") == da.min_episode_len
print(f"norm stats OK: frozen 10-dim ee6d artifact fit on both datasets "
      f"(min_start_frames={expected_min_start})")

print("\nALL HUMAN-PROMPT DATA GATES PASS")
