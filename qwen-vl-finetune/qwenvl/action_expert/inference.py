"""Shared inference core for the Qwen3-VL flow-matching action expert.

Turns an observation (cam_high history frames + a current wrist still + the 7-dim right-arm
state) into an absolute-joint action chunk via the action expert (flow ODE) and, optionally,
FAST (a debug signal the expert itself ignores). Preprocessing is kept byte-identical to
training by reusing `RobotFlowMatchingDataset` for norm stats, the FAST tokenizer, processor
alignment, and `get_rope_index_3` positions.

Used by both the offline viz (`scripts/infer_visualize_action_expert.py`) and the realtime
server (`qwen_action_expert_server.py`). Requires `qwen-vl-finetune` on sys.path.
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from scipy.fft import idct
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from qwenvl.action_expert import ActionExpertConfig, Qwen3VLWithActionExpert
from qwenvl.data.ee_repr import EE_DIM, ee_chunk_to_joints, joints_to_ee
from qwenvl.data.robot_data import (
    RobotDataArguments,
    RobotFlowMatchingDataset,
    apply_delta_actions,
    format_robot_prompt,
    quantile_normalize,
    quantile_unnormalize,
    undo_delta_actions,
)

# ----- fixed to match scripts/train_action_expert_4b.sh -----
MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
DATA_DIRS = "/iris/projects/humanoid/trossen_data/0528_green_yellow_block_mem_merged_bk"
FAST_TOK = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen"
DEFAULT_CKPT = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/qwen3_4b_action_expert_fast/checkpoint-7000"
EXPERT_HIDDEN, EXPERT_INTER, EXPERT_HEADS = 1024, 4096, 16
ACTION_DIM, ACTION_HORIZON = 7, 50
SUBTASK_QUESTION = "which colored block did the human hand point to?"
WRIST_MAX_PIXELS = 50176


def load_visual_budget(ckpt):
    """Read visual_budget.json written next to a checkpoint by the training script.

    The pixel budgets decide how many visual tokens the model sees, and they are NOT
    recoverable from the weights: serving a model trained at 72 tok/frame with the old
    33 tok/frame budget raises no error, it just quietly feeds the policy a blurrier world.
    Looks in the checkpoint dir and its parent (checkpoint-XXXX lives inside output_dir).
    Returns {} for pre-2026-07-17 checkpoints, which trained on the old hardcoded defaults.
    """
    if not ckpt:
        return {}
    for d in (ckpt, os.path.dirname(os.path.normpath(ckpt))):
        p = os.path.join(d, "visual_budget.json")
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}


def load_norm_stats_path(ckpt):
    """Path of the norm_stats.json stamped next to a checkpoint by the training script.

    Serving must normalize state/actions with EXACTLY the stats the checkpoint trained
    with; loading them from the checkpoint (rather than a dataset dir) makes that
    automatic. Looks in the checkpoint dir and its parent, like load_visual_budget.
    Returns None for older checkpoints -- the dataset then falls back to the FAST
    tokenizer dir's copy, then to the legacy per-dataset-dir file (see
    robot_data._load_or_compute_norm_stats), preserving their old behavior exactly.
    """
    if not ckpt:
        return None
    for d in (ckpt, os.path.dirname(os.path.normpath(ckpt))):
        p = os.path.join(d, "norm_stats.json")
        if os.path.exists(p):
            return p
    return None


def load_repr_config(ckpt):
    """Action-representation config from the checkpoint's stamped norm_stats.json meta.

    ee6d checkpoints predict [xyz, 6D rotation, jaw] (10 dims, delta_mask 9,-1) while the
    ROBOT interface stays 7-dim joints -- state_to_model / model_actions_to_robot convert
    at the serve boundary. Legacy checkpoints (no stamp or no action_space key) resolve to
    the historic joint-space defaults, so this is safe to call unconditionally."""
    meta = {}
    p = load_norm_stats_path(ckpt)
    if p:
        with open(p) as f:
            meta = json.load(f).get("meta", {})
    space = meta.get("action_space", "joint")
    return {
        "action_space": space,
        "action_dim": EE_DIM if space == "ee6d" else ACTION_DIM,
        "action_horizon": int(meta.get("horizon", ACTION_HORIZON)),
        "delta_mask": meta.get("delta_mask", "6,-1"),
        "active_dims": meta.get("active_dims", "7:14"),
    }


def state_to_model(ds, joints):
    """Robot state (7 joints) -> the model's state representation."""
    if ds.data_args.action_space == "ee6d":
        return joints_to_ee(np.asarray(joints, dtype=np.float32))
    return joints


def model_actions_to_robot(ds, actions_abs, q_current, ik_max_iters=200):
    """Model-space ABSOLUTE action chunk -> robot joint commands.

    ee6d: Gram-Schmidt + sequential damped-least-squares IK seeded from the measured
    joints (ee_repr.ee_chunk_to_joints); returns (joints, num_ik_failures) where failed
    steps hold the previous verified command. joint space: passthrough, 0 failures.
    ik_max_iters bounds each DLS solve (server passes its --ik_max_iters)."""
    if ds.data_args.action_space == "ee6d":
        return ee_chunk_to_joints(actions_abs, np.asarray(q_current, dtype=np.float64)[:6],
                                  max_iters=ik_max_iters)
    return actions_abs, 0


def build_data_args(fast_tok=FAST_TOK, data_dirs=DATA_DIRS, image_history=True, predict_subtask=True,
                    budget=None, norm_stats_path=None, repr_cfg=None):
    """image_history / predict_subtask MUST match how the checkpoint was trained (they change
    the input the model conditions on -- history video vs single still, and question+predict
    vs subtask-as-input).

    `budget`: dict from load_visual_budget(ckpt); overrides the pixel defaults below so the
    served resolution matches training. Omit only for old checkpoints trained at the defaults.
    """
    da = RobotDataArguments()
    da.model_type = "qwen3vl"
    da.robot_data_dirs = data_dirs
    da.camera = "cam_high"
    da.num_frames = 10
    da.frame_stride = 10
    da.image_history = image_history
    da.active_dims = "7:14"
    da.action_dim = ACTION_DIM
    da.delta_mask = "6,-1"
    da.action_horizon = ACTION_HORIZON
    da.use_delta_actions = True
    da.wrist_cameras = "cam_right_wrist"
    da.wrist_max_pixels = WRIST_MAX_PIXELS
    da.default_prompt = "waiting"
    da.train_split = 1.0
    da.predict_subtask = predict_subtask
    da.subtask_question = SUBTASK_QUESTION
    da.use_fast_tokens = True
    da.fast_tokenizer_path = fast_tok
    # Checkpoint-stamped norm stats (load_norm_stats_path); None -> fall back to the
    # FAST tokenizer dir's copy, then the legacy per-dataset-dir file.
    da.norm_stats_path = norm_stats_path
    da.max_pixels = 50176
    da.min_pixels = 784
    # Action representation the checkpoint was trained with (load_repr_config). Absent ->
    # the joint-space defaults above, so old checkpoints behave exactly as before.
    for k, v in (repr_cfg or {}).items():
        setattr(da, k, v)
    # Overlay the budget recorded at training time (see load_visual_budget). Anything absent
    # keeps the legacy default above, so old checkpoints behave exactly as before.
    # (Includes wrist_cameras for 2-wrist checkpoints.)
    for k, v in (budget or {}).items():
        setattr(da, k, v)
    return da


def build_dataset(fast_tok=FAST_TOK, data_dirs=DATA_DIRS, cache_dir=None, image_history=True,
                  predict_subtask=True, budget=None, norm_stats_path=None, repr_cfg=None):
    """Instantiate the training dataset (registers FAST tokens on the processor, aligns
    processor pixels, loads norm stats). We use it only for its config-aligned objects
    (norm_stats, fast tokenizer + ids, delta_mask, processor, get_rope_index, and the
    image_history / predict_subtask MODE flags that inference must match)."""
    processor = AutoProcessor.from_pretrained(MODEL_PATH, cache_dir=cache_dir or os.environ.get("HF_HOME"))
    return RobotFlowMatchingDataset(
        processor,
        build_data_args(fast_tok, data_dirs, image_history, predict_subtask, budget,
                        norm_stats_path, repr_cfg))


def load_model(ckpt, ds, device, expert_attends_subtask=True, use_ema=True):
    """Rebuild Qwen3VLWithActionExpert exactly as training did, then load the checkpoint.
    `expert_attends_subtask` MUST match training (False = subtask-insulated checkpoint).

    If the checkpoint dir contains `ema_expert.pt` (written by ExpertEMACallback), the
    expert weights are OVERLAID with the EMA copy -- the averaged weights every working
    flow/diffusion policy stack deploys (openpi serves its ema_params; Diffusion Policy
    evals its ema_model). `use_ema=False` serves the raw iterate for A/B comparison."""
    try:
        vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, cache_dir=os.environ.get("HF_HOME"),
            dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        )
    except Exception as e:  # noqa: BLE001
        print(f"[warn] flash_attention_2 unavailable ({e}); falling back to sdpa")
        vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, cache_dir=os.environ.get("HF_HOME"),
            dtype=torch.bfloat16, attn_implementation="sdpa",
        )
    tc = vlm.config.text_config

    sd = torch.load(os.path.join(ckpt, "pytorch_model.bin"), map_location="cpu", mmap=True, weights_only=True)
    # Infer the expert depth from the checkpoint so a SmolVLA layer-skipped model (N < L)
    # loads without being told: expert layer i attends VLM layer i, and the count is however
    # many action_expert.layers.<i> the checkpoint has. Falls back to the VLM depth.
    layer_idxs = [int(k.split(".")[2]) for k in sd if k.startswith("action_expert.layers.")]
    n_expert_layers = (max(layer_idxs) + 1) if layer_idxs else tc.num_hidden_layers
    if n_expert_layers != tc.num_hidden_layers:
        print(f"[load_model] SmolVLA layer skipping: expert has {n_expert_layers}/"
              f"{tc.num_hidden_layers} VLM layers (attends VLM layers 0..{n_expert_layers-1})")

    expert_config = ActionExpertConfig(
        num_hidden_layers=n_expert_layers,
        num_key_value_heads=tc.num_key_value_heads,
        head_dim=tc.head_dim,
        hidden_size=EXPERT_HIDDEN,
        intermediate_size=EXPERT_INTER,
        num_attention_heads=EXPERT_HEADS,
        # From the dataset config so ee6d checkpoints (action_dim 10) rebuild correctly;
        # joint-space datasets carry the historic 7/50.
        action_dim=ds.data_args.action_dim,
        action_horizon=ds.data_args.action_horizon,
    )
    model = Qwen3VLWithActionExpert(vlm, expert_config, train_vlm=True,
                                    expert_attends_subtask=expert_attends_subtask)
    model.action_expert.to(torch.bfloat16)
    model.vlm.resize_token_embeddings(ds.vlm_vocab_size)  # FAST-resized embed + tied lm_head
    model.vlm.get_input_embeddings().to(torch.bfloat16)
    model.vlm.lm_head.to(torch.bfloat16)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    del sd

    ema_path = os.path.join(ckpt, "ema_expert.pt")
    if use_ema and os.path.exists(ema_path):
        ema = torch.load(ema_path, map_location="cpu", weights_only=True)
        # Partial overlay: only the expert/head tensors present in the EMA file are replaced
        # (load_state_dict casts fp32 shadow -> module dtype); the VLM keeps the main weights.
        overlay_missing, overlay_unexpected = model.load_state_dict(ema["params"], strict=False)
        if overlay_unexpected:
            raise RuntimeError(f"EMA overlay has unknown tensors: {overlay_unexpected[:5]}")
        print(f"[load_model] EMA weights ACTIVE: {len(ema['params'])} expert tensors "
              f"(decay {ema['decay']}, step {ema['step']}) from ema_expert.pt")
    elif use_ema:
        print("[load_model] no ema_expert.pt in checkpoint -> serving raw weights "
              "(normal for checkpoints trained before 2026-07-28)")
    else:
        print("[load_model] use_ema=False -> serving RAW (non-EMA) weights")
    return model.eval().to(device)


def make_prompt(ds, state, task_text=None, past_states=None):
    """pi0.5-style prompt with the discretized (quantile-normalized) state. task_text defaults
    to the configured subtask_question (predict-subtask mode); pass the subtask string in
    subtask-input mode (predict_subtask=False), where the subtask is the prompt."""
    if task_text is None:
        task_text = ds.data_args.subtask_question
    norm_state = quantile_normalize(np.asarray(state, dtype=np.float32), ds.norm_stats["state"])
    norm_past = None
    if past_states is not None:
        # (K, D) MODEL-space states at the history-frame timesteps, oldest first --
        # normalized with the same stats, formatted before the current state (training parity).
        norm_past = quantile_normalize(np.asarray(past_states, dtype=np.float32),
                                       ds.norm_stats["state"])
    return format_robot_prompt(task_text, norm_state, past_states=norm_past)


def resize_wrist(img, budget=WRIST_MAX_PIXELS):
    """Pre-resize the wrist still to <= budget pixels (aspect preserved, downscale only) --
    exactly RobotFlowMatchingDataset._extract_wrist_images (apply_chat_template does NOT apply
    max_pixels to pre-loaded PIL images, so a raw frame would blow up the token count)."""
    w, h = img.size
    if w * h > budget:
        scale = (budget / (w * h)) ** 0.5
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
    return img


def templatize(ds, top_visual, wrist, prompt, assistant_text, device):
    """Tokenize [user: top-visual + wrist + state-prompt] (+ optional assistant subtask turn).

    `top_visual` is a LIST of PIL frames: fed as a video when ds.data_args.image_history, or as
    a single still (top_visual[0]) when not. assistant_text=None -> add the generation prompt
    (ready to GENERATE the subtask, or simply no assistant turn in subtask-input mode).
    Otherwise the assistant turn holds `assistant_text`. Returns device tensors plus
    get_rope_index_3 position_ids (used by the flow-matching expert; generate() recomputes
    its own)."""
    if ds.data_args.image_history:
        content = [{"type": "video", "video": top_visual}]
    else:
        content = [{"type": "image", "image": top_visual[0]}]
    content += [{"type": "image", "image": im} for im in wrist]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})

    dd = ds.processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt",
        add_generation_prompt=assistant_text is None, do_sample_frames=False,
    )
    input_ids = dd["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    vgt = dd.get("video_grid_thw")
    spg = None
    if vgt is not None:
        vp = ds.processor.video_processor
        spg = [vp.temporal_patch_size / vp.fps] * len(vgt)
    position_ids, _ = ds.get_rope_index(
        ds.merge_size, input_ids,
        image_grid_thw=dd.get("image_grid_thw"), video_grid_thw=vgt, second_per_grid_ts=spg,
    )

    def to_dev(x, dt=None):
        if x is None:
            return None
        x = x.to(device)
        return x.to(dt) if dt is not None else x

    return dict(
        input_ids=input_ids.to(device),
        position_ids=position_ids.to(device),
        pixel_values=to_dev(dd.get("pixel_values"), torch.bfloat16),
        image_grid_thw=to_dev(dd.get("image_grid_thw")),
        pixel_values_videos=to_dev(dd.get("pixel_values_videos"), torch.bfloat16),
        video_grid_thw=to_dev(vgt),
    )


def generate_subtask(model, mm, ds, im_end_id, max_new_tokens=48, timings=None):
    """Greedily decode the assistant subtask answer from the generation-prompt prefix (the
    model's own subtask, as at deployment) -- text only, stops at <|im_end|>.

    `timings` (optional): dict populated with generate_ms (prefill + all decode steps) and
    gen_tokens (how many tokens came out, incl. the <|im_end|>)."""
    import time as _time

    if timings is not None and mm["input_ids"].is_cuda:
        torch.cuda.synchronize(mm["input_ids"].device)
    _t0 = _time.perf_counter()

    out = model.vlm.generate(
        input_ids=mm["input_ids"],
        attention_mask=torch.ones_like(mm["input_ids"]),
        pixel_values=mm["pixel_values"],
        image_grid_thw=mm["image_grid_thw"],
        pixel_values_videos=mm["pixel_values_videos"],
        video_grid_thw=mm["video_grid_thw"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        eos_token_id=im_end_id,
        pad_token_id=ds.tokenizer.pad_token_id or im_end_id,
    )
    new = out[0, mm["input_ids"].shape[1]:].tolist()

    if timings is not None:
        if mm["input_ids"].is_cuda:
            torch.cuda.synchronize(mm["input_ids"].device)
        timings["generate_ms"] = round((_time.perf_counter() - _t0) * 1e3, 1)
        timings["gen_tokens"] = len(new)

    if im_end_id in new:
        new = new[: new.index(im_end_id)]
    return ds.tokenizer.decode(new, skip_special_tokens=True).strip()


def subtask_token_mask_for(input_ids):
    """(1, L) bool mask over an INFERENCE sequence marking the assistant turn (the
    generation-prompt header and everything after -- generated subtask, <|im_end|>, newline).
    All-False when there is no assistant turn. Mirrors the training-side mask from
    RobotFlowMatchingDataset._build_item (there FAST tokens follow and stay unmasked; at
    inference nothing follows the turn). Token ids: 151644 = <|im_start|>, 77091 = 'assistant'."""
    ids = input_ids[0].tolist()
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for i in range(len(ids) - 1):
        if ids[i] == 151644 and ids[i + 1] == 77091:
            mask[0, i:] = True
            break
    return mask


def generate_subtask_cached(model, mm, ds, im_end_id, max_new_tokens=48, timings=None):
    """Single-prefill subtask generation (Phase 1 of the latency plan).

    Replaces the [generate (prefill #1, cache discarded) -> re-templatize -> prefill #2]
    sequence with ONE prefill whose KV cache is kept and extended: greedy-decode the subtask
    token by token on top of the cache, then append the assistant turn's trailing tokens so
    the cache is exactly what a fresh prefill of templatize(..., assistant_text=subtask)
    would produce. `mm` must be the generation-prompt prefix (templatize(..., None, ...)).

    Returns (subtask_text, cache, input_ids_full, position_ids_full) -- feed the last three
    to predict_expert(..., prefix_cache=(cache, input_ids_full, position_ids_full)).

    Decode-position math: generated tokens are pure text, whose M-RoPE position on all three
    axes is (global max position + k) -- the same rule ds.get_rope_index applies, so the
    incremental positions match a full-sequence recompute exactly.
    """
    import time as _time

    ids = mm["input_ids"]                     # (1, L)
    pos = mm["position_ids"]                  # (3, 1, L)
    device = ids.device
    # the chat template closes the assistant turn with "<|im_end|>\n"
    nl_id = ds.tokenizer.encode("\n")[0]

    if timings is not None and ids.is_cuda:
        torch.cuda.synchronize(device)
    _t0 = _time.perf_counter()

    out = model.vlm.model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        position_ids=pos,
        pixel_values=mm["pixel_values"],
        image_grid_thw=mm["image_grid_thw"],
        pixel_values_videos=mm["pixel_values_videos"],
        video_grid_thw=mm["video_grid_thw"],
        use_cache=True,
    )
    cache = out.past_key_values

    if timings is not None:
        if ids.is_cuda:
            torch.cuda.synchronize(device)
        timings["gen_prefill_ms"] = round((_time.perf_counter() - _t0) * 1e3, 1)
        _t0 = _time.perf_counter()

    def step(token_id, position):
        """Feed one token, extending the cache; returns the logits for the next token."""
        tok = torch.tensor([[token_id]], dtype=ids.dtype, device=device)
        step_pos = torch.full((3, 1, 1), position, dtype=pos.dtype, device=device)
        o = model.vlm.model(input_ids=tok, position_ids=step_pos,
                            past_key_values=cache, use_cache=True)
        return model.vlm.lm_head(o.last_hidden_state[:, -1])

    next_pos = int(pos.amax().item()) + 1
    cur = int(model.vlm.lm_head(out.last_hidden_state[:, -1]).argmax(-1).item())
    gen: list[int] = []
    for _ in range(max_new_tokens):
        gen.append(cur)
        logits = step(cur, next_pos)
        next_pos += 1
        if cur == im_end_id:
            break
        cur = int(logits.argmax(-1).item())
    if gen[-1] != im_end_id:
        # ran out of budget without closing the turn: close it ourselves so the cache
        # matches a well-formed assistant turn
        gen.append(im_end_id)
        step(im_end_id, next_pos)
        next_pos += 1
    # trailing newline after <|im_end|> (chat-template formatting); KV only, logits unused
    step(nl_id, next_pos)
    gen.append(nl_id)
    next_pos += 1

    if timings is not None:
        if ids.is_cuda:
            torch.cuda.synchronize(device)
        timings["decode_ms"] = round((_time.perf_counter() - _t0) * 1e3, 1)
        timings["decode_tokens"] = len(gen)

    gen_t = torch.tensor([gen], dtype=ids.dtype, device=device)
    input_ids_full = torch.cat([ids, gen_t], dim=1)
    tail_pos = torch.arange(int(pos.amax().item()) + 1, next_pos, dtype=pos.dtype, device=device)
    position_ids_full = torch.cat([pos, tail_pos.view(1, 1, -1).expand(3, 1, -1)], dim=-1)

    answer = gen[:-1]                                   # drop the trailing newline
    if im_end_id in answer:
        answer = answer[: answer.index(im_end_id)]
    subtask = ds.tokenizer.decode(answer, skip_special_tokens=True).strip()
    return subtask, cache, input_ids_full, position_ids_full


def fast_decode_tolerant(ds, fast_ids, horizon, action_dim):
    """FAST (DCT->BPE) decode that pads/truncates the coefficient array to horizon*action_dim
    instead of returning all-zeros when the BPE decode is a few coefficients short/long
    (common with greedy generation). Zero-padding at the end drops the highest-frequency DCT
    terms (mild low-pass). Returns (normalized_actions[H,D], exact_length_bool)."""
    tok = ds.fast_tok
    coeffs = np.asarray(list(map(ord, tok.bpe_tokenizer.decode(fast_ids))), dtype=np.float64)
    coeffs = coeffs + tok.min_token
    need = horizon * action_dim
    exact = len(coeffs) == need
    if len(coeffs) < need:
        coeffs = np.concatenate([coeffs, np.zeros(need - len(coeffs))])
    else:
        coeffs = coeffs[:need]
    return idct(coeffs.reshape(horizon, action_dim) / tok.scale, axis=0, norm="ortho"), exact


def normalize_action_prefix(ds, prefix_abs, state):
    """ABSOLUTE-joint action prefix (d, ACTION_DIM) -> the model's normalized action space,
    referenced to the CURRENT state (delta transform + quantile norm -- exactly what training
    applies to action chunks). The RTC contract is that clients send prefixes in ABSOLUTE
    space: with delta actions, model space depends on the observation pose, so a previous
    chunk's model-space actions would be referenced to the wrong (older) pose."""
    prefix_abs = np.asarray(prefix_abs, dtype=np.float32)
    # RTC clients always send executed JOINT actions; for ee6d checkpoints convert each
    # row into the model's EE representation before the delta/normalize transform.
    if getattr(ds.data_args, "action_space", "joint") == "ee6d" and prefix_abs.shape[-1] == 7:
        prefix_abs = joints_to_ee(prefix_abs)
    delta = apply_delta_actions(prefix_abs,
                                np.asarray(state, dtype=np.float32), ds.delta_mask)
    return quantile_normalize(delta, ds.norm_stats["actions"])


def predict_expert(model, mm, ds, state, noise=None, num_steps=10, action_prefix_abs=None,
                   prefix_cache=None, subtask_token_mask=None, timings=None):
    """Action expert: integrate the flow ODE, then unnormalize + undo delta -> absolute joints.
    Returns (ACTION_HORIZON, ACTION_DIM) absolute joint targets.

    `action_prefix_abs` (optional, RTC): the (d, ACTION_DIM) ABSOLUTE-joint actions already
    committed for execution during the inference delay. They are re-expressed in model space
    against `state` and hard-clamped during denoising, so the returned chunk's first d actions
    equal the prefix and the rest continues it smoothly (requires an RTC-trained checkpoint).

    `prefix_cache` (optional, single-prefill fast path): (cache, input_ids_full,
    position_ids_full) as returned by generate_subtask_cached -- the VLM prefix pass is
    skipped and the expert conditions on the reused KV. `mm` is then ignored.

    `timings` (optional): dict populated with expert_prefill_ms / flow_loop_ms / flow_steps
    (adds cuda syncs -- profiling only)."""
    if prefix_cache is not None:
        cache, ids_full, pos_full = prefix_cache
        mm = dict(input_ids=ids_full, position_ids=pos_full, pixel_values=None,
                  image_grid_thw=None, pixel_values_videos=None, video_grid_thw=None)
    else:
        cache = None

    prefix_norm = None
    prefix_len = 0
    if action_prefix_abs is not None and len(action_prefix_abs) > 0:
        prefix_len = len(action_prefix_abs)
        if prefix_len >= ds.horizon:
            raise ValueError(f"action prefix ({prefix_len}) must be shorter than the horizon ({ds.horizon})")
        prefix_norm = torch.from_numpy(
            np.ascontiguousarray(normalize_action_prefix(ds, action_prefix_abs, state))
        )[None].to(mm["input_ids"].device)

    norm = model.sample_actions(
        input_ids=mm["input_ids"],
        attention_mask=torch.ones_like(mm["input_ids"]),
        position_ids=mm["position_ids"],
        pixel_values=mm["pixel_values"],
        image_grid_thw=mm["image_grid_thw"],
        pixel_values_videos=mm["pixel_values_videos"],
        video_grid_thw=mm["video_grid_thw"],
        noise=noise,
        num_steps=num_steps,
        action_prefix=prefix_norm,
        prefix_length=prefix_len if prefix_norm is not None else None,
        prefix_key_values=cache,
        subtask_token_mask=subtask_token_mask,
        timings=timings,
    )
    delta = quantile_unnormalize(norm[0].float().cpu().numpy(), ds.norm_stats["actions"])
    return undo_delta_actions(delta, np.asarray(state, dtype=np.float32), ds.delta_mask)


def predict_fast(model, mm, ds, state, max_new_tokens=96):
    """FAST debug head: greedily generate <|action_start|> ... <|action_end|>, decode, then
    unnormalize + undo delta -> absolute joints. Returns (actions[H,D] or None, (n_gen, n_valid, exact))."""
    start = torch.tensor([[ds.fast_start_id]], device=mm["input_ids"].device)
    gen_ids = torch.cat([mm["input_ids"], start], dim=1)
    out = model.vlm.generate(
        input_ids=gen_ids,
        attention_mask=torch.ones_like(gen_ids),
        pixel_values=mm["pixel_values"],
        image_grid_thw=mm["image_grid_thw"],
        pixel_values_videos=mm["pixel_values_videos"],
        video_grid_thw=mm["video_grid_thw"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        eos_token_id=ds.fast_end_id,
        pad_token_id=ds.tokenizer.pad_token_id or ds.fast_end_id,
    )
    new = out[0, gen_ids.shape[1]:].tolist()
    if ds.fast_end_id in new:
        new = new[: new.index(ds.fast_end_id)]
    fast_ids = [tid - ds.fast_base_id for tid in new
                if ds.fast_base_id <= tid < ds.fast_base_id + ds.fast_vocab_size]
    if not fast_ids:
        return None, (len(new), 0, False)
    dec, exact = fast_decode_tolerant(ds, fast_ids, ds.horizon, ds.data_args.action_dim)
    delta = quantile_unnormalize(dec, ds.norm_stats["actions"])
    return undo_delta_actions(delta, np.asarray(state, dtype=np.float32), ds.delta_mask), (len(new), len(fast_ids), exact)
