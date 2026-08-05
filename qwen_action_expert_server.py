"""Realtime inference server for the Qwen3-VL flow-matching action expert (SINGLE server).

Replaces the old two-server setup (a Qwen task-planner + a separate openpi policy). This one
server does BOTH the high-level subtask generation and the low-level action prediction:

  POST /infer  (multipart)
    cam_high        : exactly NUM_FRAMES JPEGs, oldest-first (the strided history)
    wrist           : 1 JPEG, the current right-arm wrist still
    state           : form field, JSON list of 7 floats (right-arm joint state)
    run_fast        : form field, "true"/"false" (default false) -- FAST is a DEBUG signal
                      the action expert IGNORES; off by default to save latency
    num_flow_steps  : form field, int (default = --num_flow_steps)
  ->
    { "subtask": <model's own generated subtask>,           # generated on the fly, NOT provided
      "actions": [[7], ... 50 ...],                          # absolute joint targets (action expert)
      "actions_fast": [[7], ...] | null,                     # only if run_fast
      "timing_ms": {...} }

Preprocessing is byte-identical to training (reuses RobotFlowMatchingDataset via
qwenvl/action_expert/inference.py). Run on the GPU node:

  source /iris/projects/humanoid/miniconda3/bin/activate && conda activate qwen3vl
  python qwen_action_expert_server.py --ckpt <.../checkpoint-7000> --port 8003

    # a) no history + subtask input
    python qwen_action_expert_server.py --ckpt <.../qwen3_4b_ae_nohist_subin/checkpoint-XXXX> \
        --no_image_history --subtask_input

    # b) history + subtask input
    python qwen_action_expert_server.py --ckpt <.../qwen3_4b_ae_hist_subin/checkpoint-XXXX> \
        --subtask_input

    # c) history + subtask output  (defaults — no flags)
    python qwen_action_expert_server.py --ckpt <.../qwen3_4b_action_expert_fast/checkpoint-XXXX>


    # specifying custom fast tokenizer path and norm stats path via data_dirs (norm stats saved in dataset dir):

    python qwen_action_expert_server.py \
    --ckpt checkpoints/qwen3_4b_ae_hist_subpred_0714merged_rtc25/checkpoint-15000 \
    --fast_tok checkpoints/fast_tokenizer_trossen_0714merged \
    --data_dirs /iris/projects/humanoid/trossen_data/0714_green_yellow_block_mem_merged \
    --port 8003
"""

import argparse
import io
import json
import os
import sys
import time

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

# Make the qwenvl package importable, then pull the shared inference core.
REPO = "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from qwenvl.action_expert.inference import (  # noqa: E402
    ACTION_DIM,
    ACTION_HORIZON,
    DATA_DIRS,
    DEFAULT_CKPT,
    FAST_TOK,
    build_dataset,
    load_norm_stats_path,
    load_repr_config,
    load_visual_budget,
    model_actions_to_robot,
    state_to_model,
    generate_subtask,
    generate_subtask_cached,
    load_model,
    make_prompt,
    predict_expert,
    predict_fast,
    resize_wrist,
    subtask_token_mask_for,
    templatize,
)


def _parse_action_prefix(raw: str):
    """RTC: parse the optional `action_prefix` form field -- a JSON (d, ACTION_DIM) list of
    ABSOLUTE joint actions the client has already committed to executing during the inference
    delay. Returns None when absent/empty."""
    if not raw or not raw.strip():
        return None
    prefix = np.asarray(json.loads(raw), dtype=np.float32)
    if prefix.ndim != 2 or prefix.shape[1] != ACTION_DIM:
        raise HTTPException(status_code=400, detail=f"action_prefix must be (d, {ACTION_DIM}), got {list(prefix.shape)}")
    if not 0 < prefix.shape[0] < ACTION_HORIZON:
        raise HTTPException(
            status_code=400,
            detail=f"action_prefix length must be in [1, {ACTION_HORIZON - 1}], got {prefix.shape[0]}",
        )
    return prefix


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8003)
    # 10 Euler steps. NOTE: offline GT-chunk MSE said 5 ~= 10, but ON-ROBOT rollouts are
    # significantly better at 10 (user-observed) -- open-loop MSE is a weak proxy for
    # closed-loop behavior. Keep 10; the ~107 ms saving is not worth the quality hit.
    ap.add_argument("--num_flow_steps", type=int, default=10)
    # Subtasks are <= ~6 tokens; greedy decode stops at <|im_end|> anyway, this only bounds
    # the pathological case (each surplus token costs ~22 ms serial decode).
    ap.add_argument("--max_subtask_new_tokens", type=int, default=16)
    ap.add_argument("--max_fast_new_tokens", type=int, default=96)
    # DLS-IK iteration budget per chunk step (ee6d checkpoints). Warm-seeded reachable
    # solves converge in <10 iterations; 60 gives 6x margin while capping a fully
    # unreachable 50-step chunk at ~1.1 s instead of ~3.6 s at the library default 200.
    # Budget exhaustion is not a jump risk: that step just holds the previous command.
    ap.add_argument("--ik_max_iters", type=int, default=60)
    ap.add_argument("--no_preproc_cache", action="store_true",
                    help="disable the per-frame video preprocessing cache (bit-exact reuse "
                         "of resize+normalize keyed by the dedup frame ids; ~2.5x faster "
                         "templatize for sliding-window requests)")
    # These MUST match how the checkpoint was trained.
    ap.add_argument("--no_image_history", action="store_true",
                    help="checkpoint trained with a single current cam_high still (no history video)")
    ap.add_argument("--subtask_input", action="store_true",
                    help="checkpoint trained subtask-as-input (predict_subtask=False); the client PROVIDES the subtask")
    ap.add_argument("--fast_tok", default=FAST_TOK,
                    help="FAST tokenizer dir; MUST be the one the checkpoint was trained with")
    ap.add_argument("--data_dirs", default=DATA_DIRS,
                    help="dataset dir(s) for norm stats; MUST be the data the checkpoint was trained on")
    ap.add_argument("--profile", action="store_true",
                    help="fine-grained per-phase timing in timing_ms.profile (adds cuda syncs; small overhead)")
    ap.add_argument("--no_cache_reuse", action="store_true",
                    help="predict-subtask mode: fall back to the old double-prefill path "
                         "(generate, re-templatize, prefill again) instead of reusing the "
                         "generation KV cache for the expert")
    # Subtask insulation (checkpoints trained with --expert_attends_subtask False): the
    # expert ignores the subtask turn, so generation is NOT needed for control -- it can be
    # regenerated every N requests (telemetry/monitoring) and skipped otherwise.
    ap.add_argument("--insulated_subtask", action="store_true",
                    help="checkpoint trained with --expert_attends_subtask False")
    ap.add_argument("--no_ema", action="store_true",
                    help="serve the RAW weights even if the checkpoint has ema_expert.pt "
                         "(A/B only; the EMA copy is the deployment default, as in openpi)")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the expert Euler step (CUDA-graph replay of the "
                         "~1000 kernels/step; prefix padded to a fixed bucket so request-"
                         "to-request shape jitter never triggers a recompile). Adds a few "
                         "minutes of warmup at startup and verifies against eager output "
                         "before serving.")
    ap.add_argument("--subtask_every", type=int, default=None,
                    help="regenerate the subtask every N requests (cached text returned in "
                         "between); 0 = NEVER decode (implicit-HL fast path). DEFAULT: 0 for "
                         "insulated checkpoints -- the expert ignores the subtask, so decoding "
                         "it every request only adds latency (pass 1 to restore per-request "
                         "telemetry) -- and 1 for non-insulated checkpoints, whose expert "
                         "conditions on the subtask turn (0 is invalid there).")
    args = ap.parse_args()
    if args.subtask_every is not None and args.subtask_every < 0:
        ap.error("--subtask_every must be >= 0 (0 = never decode)")
    # NOTE: subtask_every vs insulation is validated AFTER model load, where insulation may
    # also come from the checkpoint's visual_budget.json stamp (not just the flag).
    if args.insulated_subtask and args.subtask_input:
        ap.error("--insulated_subtask is a predict-subtask (mode c) feature; it cannot be "
                 "combined with --subtask_input")
    return args


ARGS = parse_args()
DEVICE = "cuda"
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

print("Loading dataset (norm stats + FAST tokenizer + processor alignment)...")
# Visual budget the checkpoint was TRAINED with (written by the training script). Feeding a
# different resolution than training raises no error -- it just quietly degrades the policy --
# so we always take it from the checkpoint when available. {} = pre-2026-07-17 checkpoint,
# which used the old hardcoded defaults (33 tok/history-frame, 45 wrist).
BUDGET = load_visual_budget(ARGS.ckpt)
# Subtask insulation is train-time truth (newer checkpoints stamp it here; it is
# attention-mask-only, so a mismatched serve loads cleanly and degrades silently).
# Auto-apply the stamp; refuse an explicit flag that contradicts it. Popped so the
# budget overlay onto data_args stays purely visual.
_ATTENDS_STAMP = BUDGET.pop("expert_attends_subtask", None)
if _ATTENDS_STAMP is not None:
    if not _ATTENDS_STAMP and ARGS.subtask_input:
        raise SystemExit("checkpoint was trained subtask-INSULATED (a predict-subtask/mode-c "
                         "feature); it cannot be served with --subtask_input")
    if ARGS.insulated_subtask and _ATTENDS_STAMP:
        raise SystemExit("--insulated_subtask passed, but the checkpoint was trained with "
                         "expert_attends_subtask=True (non-insulated) -- mismatch refused")
    if not ARGS.insulated_subtask and not _ATTENDS_STAMP:
        print("[insulation] checkpoint trained subtask-insulated -> enabling --insulated_subtask")
        ARGS.insulated_subtask = True
# Resolve --subtask_every now that insulation is final (flag OR stamp). Insulated experts
# ignore the subtask, so decoding it every request is pure added latency -> default 0.
if ARGS.subtask_every is None:
    ARGS.subtask_every = 0 if ARGS.insulated_subtask else 1
    if ARGS.insulated_subtask:
        print("[subtask] --subtask_every -> 0 (insulated default): subtask never decoded; "
              "pass --subtask_every 1 for per-request telemetry text")
elif ARGS.subtask_every != 1 and not ARGS.insulated_subtask:
    raise SystemExit("--subtask_every != 1 requires an insulated checkpoint (a non-insulated "
                     "expert conditions on the subtask turn, so it must be generated every request)")
if ARGS.compile and ARGS.subtask_every != 0:
    # The compile warmup records the full-prefill shape only; the cached single-prefill path
    # appends a request-dependent subtask length, so a decode that crosses a 64-token padding
    # bucket triggers a fresh multi-minute CUDA-graph compile MID-RUN under the robot.
    print("[compile] WARNING: --compile with per-request subtask decode (--subtask_every "
          f"{ARGS.subtask_every}) can hit an unwarmed shape bucket mid-run (minutes-long "
          "stall). Prefer --subtask_every 0 for robot sessions.")
print(f"[visual budget] from checkpoint: {BUDGET or 'not recorded -> legacy defaults (33 tok/frame)'}")
# Norm stats stamped into the checkpoint at training time (None for older checkpoints ->
# the dataset falls back to the FAST tokenizer dir's copy, then the legacy dataset-dir file).
NORM_STATS_PATH = load_norm_stats_path(ARGS.ckpt)
print(f"[norm stats] from checkpoint: {NORM_STATS_PATH or 'not stamped -> tokenizer-dir/dataset-dir fallback'}")
# Action representation stamped in the checkpoint (joint for all legacy checkpoints;
# ee6d checkpoints predict [xyz, 6D, jaw] while the robot interface stays joint-space).
REPR = load_repr_config(ARGS.ckpt)
print(f"[action repr] {REPR['action_space']} (action_dim {REPR['action_dim']}, "
      f"delta_mask {REPR['delta_mask']})")
DS = build_dataset(budget=BUDGET, repr_cfg=REPR, fast_tok=ARGS.fast_tok, data_dirs=ARGS.data_dirs,
                   image_history=not ARGS.no_image_history, predict_subtask=not ARGS.subtask_input,
                   norm_stats_path=NORM_STATS_PATH)
# Per-frame video-preprocess cache: resize+normalize each history frame ONCE (keyed by the
# client's dedup frame ids), replay only patchify per request. Bit-exact (torch.equal-gated
# by tests/test_video_preproc_cache.py); passthrough whenever ids are unavailable.
VIDEO_PREPROC_CACHE = None
if DS.data_args.image_history and not ARGS.no_preproc_cache:
    from qwenvl.action_expert.preproc_cache import CachingVideoWrapper
    DS.processor.video_processor = VIDEO_PREPROC_CACHE = CachingVideoWrapper(DS.processor.video_processor)
    print("[preproc] per-frame video cache ON (--no_preproc_cache to disable)")
IM_END_ID = DS.tokenizer.convert_tokens_to_ids("<|im_end|>")
print(f"Loading model from {ARGS.ckpt} ...")
# The FAST debug head must be allowed to emit the tokenizer's longest observed encoding
# (ee6d chunks reach 149 tokens vs 96 for joint space) -- a short cap silently truncates
# the decode. Auto-raise from the tokenizer artifact's fit meta.
_fit_meta_path = os.path.join(ARGS.fast_tok, "fast_fit_meta.json")
if os.path.exists(_fit_meta_path):
    with open(_fit_meta_path) as _f:
        _max_fast = int(json.load(_f).get("max_fast_len", 0))
    if _max_fast + 2 > ARGS.max_fast_new_tokens:
        print(f"[fast] raising --max_fast_new_tokens {ARGS.max_fast_new_tokens} -> {_max_fast + 8} "
              f"(tokenizer max_fast_len {_max_fast})")
        ARGS.max_fast_new_tokens = _max_fast + 8

MODEL = load_model(ARGS.ckpt, DS, DEVICE, expert_attends_subtask=not ARGS.insulated_subtask,
                   use_ema=not ARGS.no_ema)
if getattr(DS.data_args, "action_space", "joint") == "ee6d":
    # Trigger the numba JIT compile of the IK chunk solver now, not on the robot's
    # first request (first-ever start ~2-4 s; cached afterwards).
    _t0 = time.perf_counter()
    model_actions_to_robot(DS, np.tile(state_to_model(DS, np.zeros(7, dtype=np.float32)).reshape(1, -1), (2, 1)),
                           np.zeros(7, dtype=np.float32), ik_max_iters=ARGS.ik_max_iters)
    print(f"[ik] chunk solver warmed in {time.perf_counter() - _t0:.1f} s")
# Insulated mode: the most recent generated subtask, reused between --subtask_every refreshes.
SUBTASK_STATE = {"count": 0, "text": None}
# How many cam_high frames the client must send: the history length, or 1 in no-history mode.
CAM_HIGH_N = DS.num_frames if DS.data_args.image_history else 1


def _compile_warmup_and_verify():
    """Enable the compiled expert step, warm it up on synthetic observations shaped
    exactly like production requests, and refuse to serve if compiled != eager output.

    Warmup covers BOTH timestep variants (plain (B,) and RTC per-token (B, ah)) so no
    compilation happens under the robot. Synthetic images are random noise -- shapes are
    all that matter to the compiler."""
    rng = np.random.default_rng(0)
    frames = [Image.fromarray(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8))
              for _ in range(CAM_HIGH_N)]
    # Same pre-resize as the real request path -- a raw 540x960 wrist would warm (and
    # CUDA-graph) a much longer prompt than any real request, causing a recompile stall
    # on the robot's first request.
    wrist = [resize_wrist(Image.fromarray(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)),
                          budget=DS.data_args.wrist_max_pixels)
             for _ in DS.wrist_cameras]
    state = state_to_model(DS, np.zeros(7, dtype=np.float32))
    # State-history checkpoints: warm with the exact production prompt shape (9 past states).
    warm_past = None
    if getattr(DS.data_args, "state_history", False):
        warm_past = state_to_model(DS, np.zeros((DS.num_frames - 1, 7), dtype=np.float32))
    prompt = make_prompt(DS, state, past_states=warm_past)
    # Non-insulated mode-c conditions the expert on an assistant turn; match that shape.
    assistant = None if (ARGS.insulated_subtask or not DS.data_args.predict_subtask) \
        else DS.data_args.default_prompt
    mm = templatize(DS, frames, wrist, prompt, assistant, DEVICE)
    stask = subtask_token_mask_for(mm["input_ids"]) if ARGS.insulated_subtask else None
    noise = torch.randn(1, DS.horizon, DS.data_args.action_dim, device=DEVICE, dtype=torch.float32,
                        generator=torch.Generator(device=DEVICE).manual_seed(0))
    rtc_prefix = np.zeros((8, ACTION_DIM), dtype=np.float32)  # exercises the (B, ah) variant

    def run(timings=None):
        with torch.inference_mode():
            return predict_expert(MODEL, mm, DS, state, noise.clone(), ARGS.num_flow_steps,
                                  subtask_token_mask=stask, timings=timings)

    def run_rtc():
        with torch.inference_mode():
            return predict_expert(MODEL, mm, DS, state, noise.clone(), ARGS.num_flow_steps,
                                  action_prefix_abs=rtc_prefix, subtask_token_mask=stask)

    t_eager = {}
    eager = run(t_eager)
    eager_rtc = run_rtc()

    # Eager-PADDED reference: separates padding numerics (bf16 SDPA tiling changes with
    # sequence length; known-benign class) from compile numerics (what we gate on).
    MODEL.enable_expert_compile(mode=None)  # pad only, no compilation
    pad = run()
    pad_rtc = run_rtc()
    pad_d = max(float(np.max(np.abs(pad - eager))), float(np.max(np.abs(pad_rtc - eager_rtc))))
    print(f"[compile] padding-only effect vs eager: {pad_d:.2e} (informational tiling noise)")

    print("[compile] compiling expert step (first calls take a few minutes)...")
    MODEL.disable_expert_compile()
    MODEL.enable_expert_compile()
    t0 = time.perf_counter()
    run(); run()          # plain-timestep variant: compile + CUDA-graph record
    run_rtc(); run_rtc()  # RTC per-token-timestep variant
    print(f"[compile] warmup done in {time.perf_counter() - t0:.0f}s")

    t_comp = {}
    comp = run(t_comp)
    comp_rtc = run_rtc()
    diff = float(np.max(np.abs(comp - pad)))
    diff_rtc = float(np.max(np.abs(comp_rtc - pad_rtc)))
    print(f"[compile] pure-compile max|compiled - eager_padded| = {diff:.2e} (plain), "
          f"{diff_rtc:.2e} (rtc);  flow loop {t_eager.get('flow_loop_ms')} -> "
          f"{t_comp.get('flow_loop_ms')} ms")
    if max(diff, diff_rtc) > 2e-2:
        raise RuntimeError(
            f"compiled expert diverges from eager-padded (max diff {max(diff, diff_rtc):.3f} rad) "
            "-- not serving. Rerun without --compile and report this.")


if ARGS.compile:
    _compile_warmup_and_verify()
print(f"Ready. image_history={DS.data_args.image_history} (cam_high frames={CAM_HIGH_N}) "
      f"predict_subtask={DS.data_args.predict_subtask} horizon={ACTION_HORIZON} action_dim={ACTION_DIM}")

app = FastAPI(title="Qwen3-VL action-expert server")


class _VisionTimer:
    """Accumulates wall time spent inside the vision tower (ViT) via forward hooks.
    Registered only under --profile; the syncs make timings exact but add overhead."""

    def __init__(self, visual_module):
        self.total_ms = 0.0
        self.calls = 0
        self._t0 = 0.0
        visual_module.register_forward_pre_hook(self._pre)
        visual_module.register_forward_hook(self._post)

    def _pre(self, module, args):
        torch.cuda.synchronize()
        self._t0 = time.perf_counter()

    def _post(self, module, args, output):
        torch.cuda.synchronize()
        self.total_ms += (time.perf_counter() - self._t0) * 1e3
        self.calls += 1

    def reset(self):
        self.total_ms = 0.0
        self.calls = 0


VISION_TIMER = _VisionTimer(MODEL.vlm.visual) if ARGS.profile else None
if ARGS.profile:
    print("Profiling ON: timing_ms.profile will carry per-phase breakdowns (adds cuda syncs).")

# --- cam_high frame-dedup cache (latency optimization) ---
# RTC clients replan every few ticks while the strided history window advances slowly, so
# consecutive requests share most of their cam_high frames. A dedup-aware client sends
# `session_id` + `cam_high_ids` (the FULL id window, oldest-first) and uploads JPEGs only for
# ids named in `cam_high_new_ids` (matching the file order); the server reconstructs the
# window from this cache and then prunes it to the current window (memory stays bounded at
# CAM_HIGH_N frames). Any referenced id that is not cached (e.g. the server restarted
# mid-run) returns 409 frame_cache_miss and the client re-sends everything. Legacy clients
# that omit cam_high_ids upload all frames as before. {session_id: {frame_id: PIL image}}
FRAME_CACHE: dict[str, dict[int, Image.Image]] = {}


def _read_rgb(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


async def _assemble_cam_high(cam_high, cam_high_ids: str, cam_high_new_ids: str, session_id: str):
    """Build the oldest-first cam_high frame list, via the dedup cache when the client sent an
    id window, else by decoding every uploaded file (legacy). Returns (frames, n_uploaded)."""
    files = cam_high or []
    if not cam_high_ids.strip():
        # legacy path: every frame uploaded on every request
        if len(files) != CAM_HIGH_N:
            kind = "history, oldest first" if DS.data_args.image_history else "current frame only"
            raise HTTPException(status_code=400,
                                detail=f"expected {CAM_HIGH_N} cam_high frame(s) ({kind}), got {len(files)}")
        return [_read_rgb(await f.read()) for f in files], len(files), None

    ids = [int(i) for i in json.loads(cam_high_ids)]
    if len(ids) != CAM_HIGH_N:
        raise HTTPException(status_code=400, detail=f"cam_high_ids must list {CAM_HIGH_N} ids, got {len(ids)}")
    new_ids = [int(i) for i in json.loads(cam_high_new_ids)] if cam_high_new_ids.strip() else []
    if len(new_ids) != len(files):
        raise HTTPException(status_code=400,
                            detail=f"cam_high_new_ids ({len(new_ids)}) must match uploaded files ({len(files)})")

    sid = session_id or "default"
    if sid not in FRAME_CACHE:
        FRAME_CACHE.clear()  # single-client server: a new session releases the old cache
        FRAME_CACHE[sid] = {}
        if VIDEO_PREPROC_CACHE is not None:
            VIDEO_PREPROC_CACHE.reset()  # frame ids restart from 0 in a new client session
    cache = FRAME_CACHE[sid]
    for fid, f in zip(new_ids, files):
        cache[fid] = _read_rgb(await f.read())

    missing = sorted({i for i in ids if i not in cache})
    if missing:
        # 409 tells the client its view of the cache is stale (server restart) -> full resend.
        raise HTTPException(status_code=409, detail=f"frame_cache_miss: {missing}")
    frames = [cache[i] for i in ids]
    FRAME_CACHE[sid] = {i: cache[i] for i in set(ids)}  # prune: older ids can't be referenced again
    return frames, len(files), ids


def _decode_model_input(input_ids) -> str:
    """Decode the exact token sequence the model sees (debug). Runs of identical vision pad
    tokens are collapsed to `<|image_pad|>x1234` so the string stays readable."""
    import re

    text = DS.tokenizer.decode(input_ids[0].tolist())
    for pad in ("<|image_pad|>", "<|video_pad|>"):
        text = re.sub(
            f"(?:{re.escape(pad)})+",
            lambda m, p=pad: f"{p}x{m.group(0).count(p)}",
            text,
        )
    return text


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ckpt": ARGS.ckpt,
        "fast_tok": ARGS.fast_tok,
        "data_dirs": ARGS.data_dirs,
        "image_history": DS.data_args.image_history,
        "cam_high_frames": CAM_HIGH_N,
        "frame_stride": DS.stride,
        "predict_subtask": DS.data_args.predict_subtask,
        "action_horizon": DS.horizon,
        "action_dim": DS.data_args.action_dim,       # model-space dim (10 for ee6d)
        "action_space": DS.data_args.action_space,   # robot i/o stays 7-dim joints
        "wrist_cameras": DS.wrist_cameras,
        # RTC: /infer accepts an `action_prefix` form field (absolute joints). Only useful
        # with a checkpoint trained with --rtc_prefix_max_length > 0.
        "supports_action_prefix": True,
        # frame-dedup: /infer accepts session_id + cam_high_ids + cam_high_new_ids so clients
        # can upload only newly acquired history frames (409 = cache miss, resend everything).
        "supports_frame_dedup": True,
        "insulated_subtask": ARGS.insulated_subtask,
        "state_history": bool(getattr(DS.data_args, "state_history", False)),
        "subtask_every": ARGS.subtask_every,
        "num_flow_steps": ARGS.num_flow_steps,
        # SmolVLA layer skipping: the expert uses only the first N VLM layers. When N < L the
        # server early-exits the VLM at layer N on the action path (no subtask decode) -- auto
        # (load_model detects N from the checkpoint). vlm_layers = the VLM's full depth L.
        "expert_num_layers": MODEL.expert_config.num_hidden_layers,
        "vlm_layers": MODEL.vlm.config.text_config.num_hidden_layers,
        "vlm_early_exit": MODEL.expert_config.num_hidden_layers
        < MODEL.vlm.config.text_config.num_hidden_layers,
    }


@app.post("/infer")
async def infer(
    cam_high: list[UploadFile] = File(None),   # optional: dedup clients may upload 0 new frames
    wrist: UploadFile = File(...),
    wrist_left: UploadFile = File(None),   # required iff the checkpoint trained with 2 wrists
    state: str = Form(...),
    state_history: str = Form(""),      # JSON (num_frames-1, 7) ABSOLUTE joint states at the
                                        # history-frame capture ticks, oldest first; REQUIRED
                                        # iff the checkpoint trained with --state_history
    subtask: str = Form(""),            # used only in subtask-input mode (predict_subtask=False)
    action_prefix: str = Form(""),      # RTC: JSON (d, 7) ABSOLUTE joint actions committed during
                                        # the inference delay; needs an RTC-trained checkpoint
    session_id: str = Form(""),         # frame-dedup: stable per-client-run id keying the cache
    cam_high_ids: str = Form(""),       # frame-dedup: JSON list of CAM_HIGH_N frame ids, oldest
                                        # first (the FULL window). Empty = legacy full upload.
    cam_high_new_ids: str = Form(""),   # frame-dedup: JSON ids of the uploaded files, in order
    run_fast: bool = Form(False),
    num_flow_steps: int = Form(0),
    debug: bool = Form(False),          # also return model_input_text: the DECODED token sequence
                                        # the expert conditions on (chat template, vision tokens
                                        # collapsed, prompt, assistant turn) -- input sanity check
):
    try:
        t0 = time.perf_counter()
        frames, n_uploaded, frame_id_window = await _assemble_cam_high(cam_high, cam_high_ids, cam_high_new_ids, session_id)
        if VIDEO_PREPROC_CACHE is not None:
            # ids present (dedup client) -> cached video preprocessing; None -> passthrough.
            VIDEO_PREPROC_CACHE.current_ids = frame_id_window if DS.data_args.image_history else None
        # History mode: frames feed the video processor (it resizes). No-history mode: the single
        # cam_high still is pre-resized like the wrist (the processor does NOT resize pre-loaded PIL).
        top = frames if DS.data_args.image_history else [resize_wrist(frames[0], budget=DS.data_args.max_pixels)]
        # Budget from the checkpoint's visual_budget.json (NOT resize_wrist's legacy
        # default): vis16+ checkpoints trained the wrist at 131072 px; serving at the old
        # 50176 default silently fed the expert ~38% of the trained wrist tokens.
        wrists = [resize_wrist(_read_rgb(await wrist.read()), budget=DS.data_args.wrist_max_pixels)]
        if len(DS.wrist_cameras) > 1:
            if wrist_left is None:
                raise HTTPException(status_code=400, detail=(
                    f"checkpoint trained with wrist_cameras={DS.wrist_cameras}; "
                    "send a `wrist_left` image too"))
            wrists.append(resize_wrist(_read_rgb(await wrist_left.read()),
                                       budget=DS.data_args.wrist_max_pixels))
        state_arr = np.asarray(json.loads(state), dtype=np.float32).reshape(-1)
        if state_arr.shape[0] != 7:
            raise HTTPException(status_code=400, detail=f"state must have 7 joint dims, got {state_arr.shape[0]}")
        # Model-space state ([xyz, 6D, jaw] for ee6d checkpoints; passthrough for joint).
        # state_arr keeps the raw joints: IK seed + the robot-facing response space.
        model_state = state_to_model(DS, state_arr)
        # Frame-aligned past states: an input-shape contract stamped by training (like the
        # wrist list); a state-history checkpoint served without them would degrade silently.
        past_model = None
        if getattr(DS.data_args, "state_history", False):
            n_past = DS.num_frames - 1
            if not state_history.strip():
                raise HTTPException(status_code=400, detail=(
                    f"checkpoint trained with --state_history: send `state_history` = JSON "
                    f"({n_past}, 7) joint states at the history-frame ticks, oldest first"))
            past_arr = np.asarray(json.loads(state_history), dtype=np.float32)
            if past_arr.shape != (n_past, 7):
                raise HTTPException(status_code=400, detail=(
                    f"state_history must be ({n_past}, 7), got {list(past_arr.shape)}"))
            past_model = state_to_model(DS, past_arr)
        elif state_history.strip():
            raise HTTPException(status_code=400, detail=(
                "checkpoint was NOT trained with --state_history; do not send state_history"))
        prefix_abs = _parse_action_prefix(action_prefix)
        nsteps = num_flow_steps or ARGS.num_flow_steps

        prof = {} if ARGS.profile else None
        if VISION_TIMER is not None:
            VISION_TIMER.reset()

        with torch.inference_mode():
            t1 = time.perf_counter()
            prefix_cache = None
            refresh_subtask = True
            if DS.data_args.predict_subtask and ARGS.insulated_subtask:
                SUBTASK_STATE["count"] += 1
                if ARGS.subtask_every == 0:
                    # Pure implicit-HL fast path: never decode a subtask -> always skip ->
                    # every request takes the VLM early-exit. Telemetry text stays the default.
                    refresh_subtask = False
                else:
                    refresh_subtask = (SUBTASK_STATE["text"] is None
                                       or (SUBTASK_STATE["count"] - 1) % ARGS.subtask_every == 0)
            if DS.data_args.predict_subtask and ARGS.insulated_subtask and not refresh_subtask:
                # Insulated skip path: the expert ignores the subtask turn, so no decode at
                # all -- prefill the generation-prompt sequence (inside sample_actions) and
                # reuse the last generated subtask text for telemetry.
                prompt = make_prompt(DS, model_state, past_states=past_model)
                mm = templatize(DS, top, wrists, prompt, None, DEVICE)
                used_subtask = SUBTASK_STATE["text"] or DS.data_args.default_prompt
                if prof is not None:
                    prof["templatize_full_ms"] = round((time.perf_counter() - t1) * 1e3, 1)
                    prof["subtask_skipped"] = True
                t2 = time.perf_counter()
            elif DS.data_args.predict_subtask and not ARGS.no_cache_reuse:
                # Single-prefill path (default): ONE ViT+LM prefill whose KV cache is kept,
                # the subtask decoded on top of it, and the same cache handed to the expert.
                prompt = make_prompt(DS, model_state, past_states=past_model)                     # fixed question
                mm_gen = templatize(DS, top, wrists, prompt, None, DEVICE)
                if prof is not None:
                    prof["templatize_gen_ms"] = round((time.perf_counter() - t1) * 1e3, 1)
                used_subtask, cache, ids_full, pos_full = generate_subtask_cached(
                    MODEL, mm_gen, DS, IM_END_ID, ARGS.max_subtask_new_tokens, timings=prof)
                used_subtask = used_subtask or DS.data_args.default_prompt
                SUBTASK_STATE["text"] = used_subtask
                prefix_cache = (cache, ids_full, pos_full)
                mm = dict(input_ids=ids_full, position_ids=pos_full, pixel_values=None,
                          image_grid_thw=None, pixel_values_videos=None, video_grid_thw=None)
                t2 = time.perf_counter()
            elif DS.data_args.predict_subtask:
                # Legacy double-prefill path (--no_cache_reuse): generate discards its cache,
                # then the sequence is re-templatized and prefilled again for the expert.
                prompt = make_prompt(DS, model_state, past_states=past_model)                     # fixed question
                mm_gen = templatize(DS, top, wrists, prompt, None, DEVICE)
                if prof is not None:
                    prof["templatize_gen_ms"] = round((time.perf_counter() - t1) * 1e3, 1)
                used_subtask = generate_subtask(MODEL, mm_gen, DS, IM_END_ID,
                                                ARGS.max_subtask_new_tokens, timings=prof) \
                    or DS.data_args.default_prompt
                SUBTASK_STATE["text"] = used_subtask
                t2 = time.perf_counter()
                mm = templatize(DS, top, wrists, prompt, used_subtask, DEVICE)
                if prof is not None:
                    prof["templatize_full_ms"] = round((time.perf_counter() - t2) * 1e3, 1)
            else:
                # Subtask-input mode: the client PROVIDES the subtask; it goes into the prompt.
                used_subtask = subtask.strip() or DS.data_args.default_prompt
                prompt = make_prompt(DS, model_state, task_text=used_subtask,
                                     past_states=past_model)
                t2 = time.perf_counter()
                mm = templatize(DS, top, wrists, prompt, None, DEVICE)
                if prof is not None:
                    prof["templatize_full_ms"] = round((time.perf_counter() - t2) * 1e3, 1)

            # Insulated checkpoints: mask the assistant turn (or its dangling generation
            # header on the skip path) out of the expert's attention -- training semantics.
            stask_mask = subtask_token_mask_for(mm["input_ids"]) if ARGS.insulated_subtask else None

            # RTC: the absolute prefix is re-expressed against state_arr inside predict_expert
            # and hard-clamped during denoising -> actions[:d] == prefix, rest continues it.
            actions = predict_expert(MODEL, mm, DS, model_state, None, nsteps,
                                     action_prefix_abs=prefix_abs,
                                     prefix_cache=prefix_cache,
                                     subtask_token_mask=stask_mask,
                                     timings=prof)  # (H, action_dim) absolute, model space
            # ee6d: Gram-Schmidt + sequential IK seeded from the measured joints -> the
            # client always receives (H, 7) joint targets regardless of representation.
            actions, ik_failures = model_actions_to_robot(DS, actions, state_arr,
                                                          ik_max_iters=ARGS.ik_max_iters)
            if ik_failures:
                print(f"[warn] {ik_failures} IK-unconverged chunk steps (held previous command)")
            t3 = time.perf_counter()

            # optional FAST debug head (the action expert ignores it; off by default).
            actions_fast = None
            if run_fast:
                mm_fast = mm
                if mm.get("pixel_values_videos") is None and mm.get("pixel_values") is None:
                    # cached path carries no pixels (the cache replaced them); FAST's generate
                    # needs them, so rebuild the full sequence just for this debug head.
                    mm_fast = templatize(DS, top, wrists, prompt, used_subtask, DEVICE)
                fast, _ = predict_fast(MODEL, mm_fast, DS, model_state, ARGS.max_fast_new_tokens)
                if fast is not None:
                    # Same robot-space contract as the expert chunk: ee6d checkpoints
                    # decode FAST in EE space -> convert to joints for the client.
                    fast, _ = model_actions_to_robot(DS, fast, state_arr,
                                                     ik_max_iters=ARGS.ik_max_iters)
                actions_fast = fast.tolist() if fast is not None else None
            t4 = time.perf_counter()

        timing = {
            "subtask_ms": round((t2 - t1) * 1e3, 1),
            "expert_ms": round((t3 - t2) * 1e3, 1),
            "fast_ms": round((t4 - t3) * 1e3, 1) if run_fast else None,
            "total_ms": round((t4 - t0) * 1e3, 1),
        }
        if prof is not None:
            if VISION_TIMER is not None:
                prof["vision_ms_total"] = round(VISION_TIMER.total_ms, 1)
                prof["vision_calls"] = VISION_TIMER.calls
            # generate_ms includes generate()'s own internal prefill; expert_prefill_ms is a
            # full prefill of (nearly) the same sequence, so the serial decode cost is
            # approximately their difference. Exact split lands with the Phase-1 manual decode.
            if "generate_ms" in prof and "expert_prefill_ms" in prof:
                prof["decode_est_ms"] = round(prof["generate_ms"] - prof["expert_prefill_ms"], 1)
            timing["profile"] = prof
            # Also print to the server console -- the robot client only shows total_ms.
            print(f"[profile] total={timing['total_ms']} subtask={timing['subtask_ms']} "
                  f"expert={timing['expert_ms']} | "
                  + " ".join(f"{k}={v}" for k, v in prof.items()))
        return {
            "status": "success",
            "subtask": used_subtask,
            # The exact prompt string templatized into the model (task/question + discretized
            # state). Returned every request so the client can sanity-check the model input.
            "prompt": prompt,
            "actions": actions.tolist(),
            "actions_fast": actions_fast,
            "prefix_length": int(prefix_abs.shape[0]) if prefix_abs is not None else 0,
            # ee6d only: chunk steps where IK did not converge (those hold the previous
            # command). Always 0 for joint-space checkpoints.
            "ik_failures": ik_failures,
            # frame-dedup visibility: how many cam_high JPEGs this request actually carried
            # (legacy clients: always CAM_HIGH_N; dedup clients: usually 0-1).
            "frames_uploaded": n_uploaded,
            "timing_ms": timing,
            # debug only: the decoded input_ids the expert conditioned on (ground truth).
            "model_input_text": _decode_model_input(mm["input_ids"]) if debug else None,
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"inference error: {e}")
    finally:
        if VIDEO_PREPROC_CACHE is not None:
            VIDEO_PREPROC_CACHE.current_ids = None  # never leak ids into warmup/other paths


if __name__ == "__main__":
    import socket

    import uvicorn

    if ARGS.host == "0.0.0.0":
        # bound to all interfaces: resolve the primary routable IP for the client URL
        try:
            _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            _s.connect(("8.8.8.8", 80))
            _ip = _s.getsockname()[0]
            _s.close()
        except OSError:
            _ip = socket.gethostbyname(socket.gethostname())
    else:
        _ip = ARGS.host
    print(f"Clients connect to: http://{_ip}:{ARGS.port}/infer  "
          f"(host {socket.gethostname()}, health: http://{_ip}:{ARGS.port}/health)")
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
