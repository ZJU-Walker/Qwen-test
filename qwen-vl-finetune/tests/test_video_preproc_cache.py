"""Bitwise + timing gates for the serving video-preprocess cache (preproc_cache.py).

Simulates the RTC client's sliding history window over synthetic frames at the SERVING
size config (visual_budget vis16: min 200704 / max 1600000 px) and requires the wrapper's
full processor output to be torch.equal to the vanilla processor's on every request:
cold-start duplicate window, +1 advances (temporal-pair parity flips), +2 advances,
409-style full re-send, and a mid-stream config change (cache must invalidate).
"""

import statistics
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from qwenvl.action_expert.preproc_cache import CachingVideoWrapper

CACHE_DIR = "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface"
NUM_FRAMES = 10

proc = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", cache_dir=CACHE_DIR)
vp_vanilla = proc.video_processor
# serving-aligned budget (what build_dataset stamps onto the processor)
vp_vanilla.size = {"shortest_edge": 200704, "longest_edge": 1600000}
wrapper = CachingVideoWrapper(vp_vanilla)

rng = np.random.default_rng(7)
def frame(i):
    return Image.fromarray(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8) * 0 +
                           rng.integers(0, 255, (540, 960, 3), dtype=np.uint8))

# stream of unique frames; windows reference them by id like the client's dedup protocol
stream = {i: frame(i) for i in range(64)}

def run(vp_like, ids, set_ids):
    if set_ids:
        wrapper.current_ids = list(ids)
    try:
        return vp_like(videos=[[stream[i] for i in ids]], do_sample_frames=False, return_tensors="pt")
    finally:
        wrapper.current_ids = None

def check(ids, label):
    ref = run(vp_vanilla, ids, set_ids=False)
    got = run(wrapper, ids, set_ids=True)
    assert torch.equal(ref["pixel_values_videos"], got["pixel_values_videos"]), f"{label}: pixel values differ"
    assert torch.equal(ref["video_grid_thw"], got["video_grid_thw"]), f"{label}: grid differs"

# 1. cold start: primed window repeats id 0 ten times (client priming semantics)
check([0] * NUM_FRAMES, "cold-start")
# 2. thirty +1 advances (pair parity flips every step)
window = list(range(NUM_FRAMES))
for step in range(30):
    check(window, f"+1 advance step {step}")
    window = window[1:] + [window[-1] + 1]
# 3. +2 advances (skipped stride)
for step in range(5):
    window = window[2:] + [window[-1] + 1, window[-1] + 2]
    check(window, f"+2 advance step {step}")
# 4. 409-style: cache cleared server-side, full window re-sent
wrapper.reset()
check(window, "post-reset full resend")
print(f"1. bitwise equivalence: 37 sliding-window requests, torch.equal everywhere  OK "
      f"(hits {wrapper.stats['hits']}, misses {wrapper.stats['misses']})")

# 5. config change invalidates: shrink the budget -> new target size -> full recompute, still equal
key_before = wrapper._cache_key
vp_vanilla.size = {"shortest_edge": 4096, "longest_edge": 320000}
check(window, "budget change")
assert wrapper._cache_key != key_before, "cache key did not change with the size config"
vp_vanilla.size = {"shortest_edge": 200704, "longest_edge": 1600000}
check(window, "budget restore")
print("2. size-config change invalidates and re-verifies  OK")

# 6. PRODUCTION PATH: full processor via apply_chat_template, wrapper installed exactly as
# the server installs it. This is the calling convention the direct checks above do NOT
# exercise (apply_chat_template nests the video one batch level deeper) -- the cache MUST
# engage (passthrough == 0) and outputs must be bitwise equal to the unwrapped processor.
proc_plain = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", cache_dir=CACHE_DIR)
proc_plain.video_processor.size = dict(vp_vanilla.size)
proc_wrapped = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct", cache_dir=CACHE_DIR)
proc_wrapped.video_processor.size = dict(vp_vanilla.size)
w2 = CachingVideoWrapper(proc_wrapped.video_processor)
proc_wrapped.video_processor = w2

def chat(proc_obj, ids):
    msgs = [{"role": "user", "content": [
        {"type": "video", "video": [stream[i] for i in ids]},
        {"type": "text", "text": "Task: pick. State: 1 2 3"}]}]
    return proc_obj.apply_chat_template(msgs, tokenize=True, return_dict=True,
                                        return_tensors="pt", add_generation_prompt=True,
                                        do_sample_frames=False)

win = list(range(12, 12 + NUM_FRAMES))
for step in range(6):
    ref = chat(proc_plain, win)
    w2.current_ids = list(win)
    try:
        got = chat(proc_wrapped, win)
    finally:
        w2.current_ids = None
    for k in ("input_ids", "pixel_values_videos", "video_grid_thw"):
        assert torch.equal(ref[k], got[k]), f"apply_chat_template step {step}: {k} differs"
    win = win[1:] + [win[-1] + 1]
assert w2.stats["passthrough"] == 0, f"cache never engaged through apply_chat_template: {w2.stats}"
assert w2.stats["misses"] == NUM_FRAMES + 5, f"unexpected miss pattern: {w2.stats}"
print(f"3. PRODUCTION apply_chat_template path: 6 requests bitwise-equal, cache engaged "
      f"(passthrough 0, misses {w2.stats['misses']})  OK")

# 6. timing: steady-state (1 new frame per request) vs vanilla
def best(fn, n=18, w=3):
    """Minimum over reps: the algorithmic cost, immune to shared-node scheduling spikes."""
    xs = []
    for i in range(n + w):
        s = time.perf_counter(); fn(); e = time.perf_counter() - s
        if i >= w: xs.append(e * 1e3)
    return min(xs), statistics.median(xs)

w0 = list(range(30, 30 + NUM_FRAMES))
run(wrapper, w0, set_ids=True)  # warm the cache
state = {"w": w0}
def step_cached():
    state["w"] = state["w"][1:] + [state["w"][-1] + 1 if state["w"][-1] + 1 in stream else state["w"][0]]
    run(wrapper, state["w"], set_ids=True)
bv, mv = best(lambda: run(vp_vanilla, w0, set_ids=False))
bc, mc = best(step_cached)
print(f"3. timing (min|median): vanilla {bv:.1f}|{mv:.1f} ms -> cached, 1 new frame/request "
      f"{bc:.1f}|{mc:.1f} ms ({bv / bc:.1f}x on min)")
assert bc < bv * 0.75, f"cache best-case speedup below 1.33x ({bv:.1f} -> {bc:.1f})"

print("\nALL VIDEO PREPROC CACHE GATES PASS")
