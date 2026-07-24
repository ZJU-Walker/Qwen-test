# Flow-Matching Action Expert for Qwen3-VL

This adds a pi0.5-style action expert (arXiv:2504.16054) on top of Qwen3-VL, with
knowledge insulation (arXiv:2505.23705): the expert's gradients **never** reach the
Qwen3-VL weights.

## How it works

```
                     ┌────────────────────────────────────────────┐
 images + text  ──►  │ Qwen3-VL (frozen by default)               │
 (subtask prompt     │ 36 layers → per-layer KV cache             │
  + discretized      └──────────────┬─────────────────────────────┘
  robot state)                      │  detach()   ← knowledge-insulation boundary
                                    ▼
 noisy actions x_t ──► action_in_proj ──► ┌───────────────────────┐
                                          │ Action expert          │
 timestep t ──► time MLP ──► adaRMS cond ►│ 36 small layers; layer i│──► action_out_proj ──► v_t
                                          │ attends to [VLM KV_i ‖ │
                                          │  own KV] (joint attn)  │
                                          └───────────────────────┘
 Training:  x_t = t·noise + (1−t)·actions,  loss = MSE(v_t, noise − actions)
 Inference: 10 Euler steps from noise (t=1) to actions (t=0); prefix runs once.
```

Key correspondences with openpi's pi0.5:

| pi0.5 (openpi)                              | This implementation                            |
|---------------------------------------------|------------------------------------------------|
| PaliGemma 2.6B VLM                          | Qwen3-VL-4B (any Qwen3-VL size works)          |
| Gemma-300M expert, shared attention op      | ~700M Qwen3-style expert attending to VLM KV cache |
| adaRMSNorm timestep conditioning            | same (zero-init scale/shift/gate projection)   |
| state discretized into the prompt (256 bins)| same (`Task: ..., State: 12 240 ...`)          |
| beta(1.5,1) time sampling, u_t = noise−a    | same                                           |
| delta joint actions (grippers absolute)     | same (mask `6,-1,6,-1`)                        |
| quantile normalization (q01/q99 → [−1,1])   | same, stats cached to `qwen_action_expert_norm_stats.json` |

The expert must match the VLM in `num_hidden_layers`, `num_key_value_heads`, and
`head_dim` (so its action-token queries can attend over the concatenation of the VLM's
cached keys/values and its own); hidden width, MLP width, and query-head count are free.
Because the prefix KV is detached (and the VLM runs under `no_grad` when frozen), the
flow-matching loss cannot update Qwen3-VL — verified by `tests/smoke_test_action_expert.py`.

Positions: action tokens get M-RoPE positions continuing right after the prefix
(`max position + 1, …`) with t=h=w, which for text-like tokens is identical to 1D RoPE
and consistent with how the VLM rotated its cached keys. Attention within the action
chunk is bidirectional; the chunk attends to all valid prefix tokens (pi0's block mask).

## Files

- `qwenvl/action_expert/modeling_action_expert.py` — expert transformer (adaRMS,
  QK-norm attention with prefix-KV joint attention, SwiGLU MLP).
- `qwenvl/action_expert/modeling_qwen3vl_with_expert.py` — `Qwen3VLWithActionExpert`:
  training forward (flow-matching loss, optional LM co-training loss) and
  `sample_actions` (Euler ODE integration reusing the prefix KV cache).
- `qwenvl/data/robot_data.py` — LeRobot-format dataset (`RobotFlowMatchingDataset`),
  norm-stats computation, delta-action transform, pi0.5-style prompt formatting.
- `qwenvl/train/train_action_expert.py` — HF-Trainer entry point.
- `scripts/train_action_expert_4b.sh` — launch script (mirrors `sft_qwen3_4b_bk.sh`).
- `tests/smoke_test_action_expert.py` — insulation + end-to-end smoke tests.

## Usage

```bash
conda activate qwen3vl
cd qwen-vl-finetune

# smoke tests (tiny random model; add --real for the 4B model + real data)
python tests/smoke_test_action_expert.py --real

# training
./scripts/train_action_expert_4b.sh
```

Data are read directly from LeRobot-format dirs (`--robot_data_dirs`, comma-separated):
parquet episodes for state/action, `cam_high` h264 videos for frames (10-frame history
at stride 10, same as `PrototypeRobotDataset`), and time-aligned prompts from
`videos/chunk-000/subtask_labels.json` (same as the openpi `pi05_trossen_memory` config).

### Training modes

1. **Frozen VLM (default, `--train_vlm False`)** — the VLM runs under `no_grad`; only
   the expert (+ flow heads) trains. Cheapest, and the strictest insulation.
2. **KI co-training (`--train_vlm True --predict_subtask True`)** — the user turn asks
   a fixed question, the assistant turn is the time-aligned subtask label, and the VLM
   trains with next-token prediction on it while the expert trains on the (detached)
   representations — the knowledge-insulation recipe, with subtask prediction playing
   the role of the discrete/web co-training objective. Note: the paper's FAST-token
   action prediction objective is not implemented yet; subtask labels are the LM signal.

### Inference sketch

```python
actions = model.sample_actions(**prefix_inputs, num_steps=10)   # normalized deltas
actions = quantile_unnormalize(actions.cpu().numpy(), norm_stats["actions"])
actions = undo_delta_actions(actions, current_state, make_delta_mask("6,-1,6,-1"))
```

with `prefix_inputs` built exactly like the dataset does (same chat template,
`format_robot_prompt(task, normalized_state)`, `add_generation_prompt=True`).
`checkpoint/action_expert.pt` holds an expert-only state dict (`expert_config` +
weights, everything not prefixed `vlm.`) for lightweight loading at serve time.

### Knobs worth knowing

- `--expert_hidden_size / --expert_intermediate_size / --expert_num_attention_heads`
  control expert capacity (defaults 1024 / 4096 / 16 ≈ 700M because the layer count is
  pinned to the VLM's 36).
- `--active_dims "7:14" --action_dim 7 --delta_mask "6,-1"` train on the right arm only
  (the left arm is stationary in the block-mem data). `active_dims` slices both state
  and actions at load time; `action_dim`/`delta_mask` describe the *selected* dims.
  At serve time, scatter the sampled 7-dim actions back into the full 14-dim command
  and hold the excluded dims at their current state.
- Quantile normalization is exactly openpi's formula (`(x−q01)/(q99−q01+1e-6)·2−1`,
  no range floor). Beware: near-constant dims divide sensor noise by ~0 and produce
  huge normalized values that dominate the flow loss (that's what a ~30 initial loss
  looks like; healthy is ~2 flow + ~6 LM) — exclude them with `--active_dims`.
- `--gradient_checkpointing` checkpoints **only the expert layers**; the VLM never uses
  HF gradient checkpointing here because it would drop the KV cache the expert needs
  (and the frozen VLM stores no activations anyway).
- `--save_safetensors False` is required: the wrapper is a plain `nn.Module` and Qwen's
  tied embeddings break safetensors' shared-tensor check.

## Training-time RTC (real-time chunking, arXiv:2512.05964)

Port of the openpi implementation (see `openpi_trossen_brian/RTC_NOTES.md`). The deployed
robot executes chunks asynchronously: while the current chunk runs, the next one is generated
conditioned on the actions being committed during the inference delay (the **prefix**), so
chunks splice without a jerk. Training simulates this: per example, sample a delay
`d ~ Uniform[min, max]`, feed the first `d` ground-truth actions as a clean prefix (their
flow timestep pinned to CLEAN), and compute the flow loss on the postfix only.

- **Timestep convention gotcha**: here `x_t = t*noise + (1-t)*actions`, so CLEAN = `t=0`
  (the OPPOSITE of the RTC paper's tau=1). See `_CLEAN_TIMESTEP` in
  `modeling_qwen3vl_with_expert.py`.
- **Enable at training**: `--rtc_prefix_max_length <d_max>` (0 = off, bit-identical to
  vanilla). The launch scripts expose it as `RTC_MAX_DELAY=25 ./scripts/train_..._2gpu*.sh`
  (auto-suffixes OUTPUT_DIR/RUN_NAME). Orthogonal to image_history / predict_subtask.
  Pick `d_max` ≥ deploy latency in control ticks (~0.6-0.8 s expert-only ⇒ ~20-25 @ 30 Hz)
  and ≤ H − exec_horizon.
- **Inference**: `sample_actions(action_prefix=..., prefix_length=...)` hard-clamps the
  prefix slots each Euler step (per-token timesteps via adaRMS — zero new params). The
  server accepts an `action_prefix` form field of **ABSOLUTE** joints; with delta actions,
  model space is pose-relative, so the server re-expresses the prefix against the current
  state (`normalize_action_prefix` in `qwenvl/action_expert/inference.py`).
- **Clients** (robot computer): `client_action_expert_rtc_sync.py` first — validates the
  clamp, the seam, and the `[bound]` prefix→postfix coherence (the real "did RTC training
  work" signal; a non-RTC checkpoint passes clamp/seam but jumps at the bound). Then
  `client_action_expert_rtc.py` — the async fixed-rate loop (stop-and-wait on late chunks).
- **Tests**: `tests/smoke_test_rtc_units.py` (CPU math) and `tests/smoke_test_rtc_gpu.py`
  (full 4B model: per-token adaRMS equivalence, RTC loss, exact prefix clamp, no-prefix
  regression).

## Inference latency (Phase 0-2 of the latency plan, 2026-07-16)

Measured on H200 at 10 frames (~490 tokens), `tests/bench_history_scaling.py`:
templatize ~31 ms | ViT+LM prefill ~60-96 ms | decode ~22 ms/token | flow loop ~21 ms/step.
The old mode-c request did templatize+prefill TWICE (~600 ms total). Note: the video
processor caps TOTAL pixels across the clip, so 100 frames => ~7 tokens/frame (heavily
downscaled) — raise the video budget before planning long-history work.

- **Single prefill (exact, default)**: `generate_subtask_cached` keeps the generation KV
  cache, extends it with the decoded subtask + the assistant turn's trailing tokens, and
  `sample_actions(prefix_key_values=...)` skips the second prefill. Validated token-exact vs
  re-templatizing; actions match to bf16 noise (<3e-3). Fallback: server `--no_cache_reuse`.
- **Flow steps stay at 10**: offline GT-chunk MSE said 5 ~= 10 (0.00130 vs 0.00159,
  `tests/bench_flow_steps.py`), but ON-ROBOT rollouts are significantly better at 10 —
  open-loop MSE is a weak proxy for closed-loop quality. Default kept at 10.
- **Subtask insulation (retrain ablation)**: `--expert_attends_subtask False` — the expert's
  attention excludes the assistant subtask turn (`subtask_token_mask`, same mechanism as
  `fast_token_mask`); the subtask stays a VLM co-training signal. Serve with
  `--insulated_subtask` (+ `--subtask_every N` to regenerate only every Nth request —
  skipped requests run zero decode steps). GPU test proves bit-identical actions for
  different subtask texts. Train via `EXPERT_ATTENDS_SUBTASK=False` in
  `scripts/train_action_expert_4b_2gpu.sh` (suffixes `_subinsul`).
- Profiling: server `--profile` puts a per-phase breakdown in `timing_ms.profile`
  (vision/prefill/decode/flow, cuda-synced).
- **SmolVLA layer skipping (retrain, arXiv:2506.01844)**: `--expert_num_layers N` gives the
  expert only N layers, each attending VLM layer i (i=0..N-1). SmolVLA's verdict is **N = L/2**
  (half) — near-parity task quality at ~half cost; for Qwen3-VL-4B (L=36) that is 18. Every
  prefix→expert path is sliced to N at the single `_detach_prefix_kv` chokepoint; the guard
  allows N ≤ L (never >). The expert provably depends only on VLM layers 0..N-1 (GPU test:
  perturbing layers N..L-1 leaves actions bit-identical, maxdiff 0.0), so at serve time — with
  implicit HL, no subtask decode — the VLM can **early-exit at layer N** (the two optimizations
  compound: ~half the VLM prefill AND half the flow loop). `load_model` auto-detects N from the
  checkpoint's `action_expert.layers.*` count, so serving needs no extra flag. Train via
  `EXPERT_VLM_LAYERS=18` in `scripts/train_action_expert_4b_2gpu.sh` (suffixes `_L18`).
  **Serve-time VLM early-exit is implemented and automatic**: `_prefix_forward` only ever
  feeds the expert (subtask decoding uses a separate `generate_subtask*` path), so at inference
  it truncates the VLM language model to the first N layers (`_vlm_truncated_to_expert_depth`,
  an `nn.ModuleList` slice restored after the call). It fires on every action-only request and
  is a no-op when N==L or when training needs the LM head. GPU-verified bit-identical to the
  full-VLM+slice path (maxdiff 0.0) with only N layers executing. Tests:
  `tests/smoke_test_layer_skip_gpu.py` (model level) and `tests/smoke_test_layer_skip_serve_path.py`
  (the exact server insulated-skip branch: 18/36 layers run, RTC prefix still honored).

### Input-dump sanity check (embedded in training)

`QWEN_DUMP_MODEL_INPUTS=<dir>` (exported by `train_action_expert_4b_2gpu.sh` to
`<OUTPUT_DIR>/input_dumps` by default) makes rank 0 dump the first
`QWEN_DUMP_MODEL_INPUTS_N` (default 2) items per dataloader process at startup, then the
hook goes dead (zero steady-state overhead). Unlike `scripts/inspect_model_inputs.py`
(which simulates the resize), this **reconstructs the images from `pixel_values` /
`pixel_values_videos` by inverting the processor's patch flattening** — provably what the
vision tower sees (a bitwise re-patchify self-check runs on every dump; both the video and
still pipelines share the same permute). Outputs per item: `*_hist_XX_model.png` (true
model resolution), `*_hist_XX_compare.png` ([native | model-view upscaled NEAREST]),
`*_wrist_*.png`, `*_text.txt` (token stream, pad runs collapsed), and `report.txt` with the
token/compression breakdown (tokens per frame, px/token model & native, the per-frame
compression ladder). Workflow: start training, wait for the `[input-dump]` console lines,
inspect, kill the run if you only wanted the check. Code: `qwenvl/data/input_inspect.py`,
hook at the end of `RobotFlowMatchingDataset._build_item`.

### Implicit HL (train with subtasks, no subtask decode at inference)

This is exactly the subtask-insulation path (`--expert_attends_subtask False` + serve
`--insulated_subtask`): the subtask stays a VLM co-training LM signal (representation benefit
kept), but the expert conditions on images+state only, so inference skips subtask generation
entirely. It is the **zero-gap** version of pi0.5's Figure-13 "implicit HL" — because the
expert never attends the subtask in training either, dropping it at inference is exact rather
than an approximation. Compounds with layer skipping: `EXPERT_ATTENDS_SUBTASK=False` +
`EXPERT_VLM_LAYERS=18` in one run.
