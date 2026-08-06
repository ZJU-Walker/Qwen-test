# Human-Video-Prompted VLA — Finalized Plan (v2, 2026-08-06)

Supersedes `human_video_prompt_vla_plan_md`: same goal, adjusted against the actual
codebase and data. Sections marked **[DONE]** are implemented and committed.

## 1. Goal (unchanged)

Replace the language instruction with a short human demonstration video. The robot
watches a human pick up a block (green or yellow, both on the table) and picks up the
same block. First version: raw prompt-video tokens, no query tokens, no resampler.

## 2. Data inventory

| Data | Where | Contents |
|---|---|---|
| Robot demos | `trossen_data/0717_green_yellow_block_mem_merged` | 224 eps (114 green / 111 yellow), 30 fps. Each: human points at a block (~2 s), then robot picks it. |
| Subtask labels | `<dataset>/videos/chunk-000/subtask_labels.json` | Per episode: `waiting` segment (the pointing phase, ends frame 39–89) then `pick up green/yellow block`. `episode_000149` mislabeled (no waiting) → auto-dropped. |
| Human demos (green) | `trossen_data/green_human_prompt` | 31 clips, 30 fps, 68–107 frames (~3 s), same rig/cameras. |
| Human demos (yellow) | `trossen_data/yellow_human_prompt` | 32 clips, same format. |

Block positions were randomized during collection (confirmed) → random pairing gives
natural position mismatch between demo and robot scene; no copy-the-coordinates shortcut.

## 3. Key design decisions (deltas vs v1 plan)

1. **Skip the pointing phase via `--skip_leading_subtask waiting`** [DONE]
   min_start = waiting end + 1 → those frames are never training timesteps. Critically,
   and UNLIKE the DAgger gate: image history and past states are clamped AT the cut
   (`hist_min`), never before it. The pointing phase shows the answer in the robot's own
   view; letting history reach into it would let the model bypass the prompt video —
   and at deployment there is no pointing phase at all.
2. **Random per-sample pairing** (not fixed pairs) [DONE]
   Every `__getitem__` draws a fresh same-color demo clip. The model cannot memorize
   demo↔episode associations; it must read the demo's content.
3. **Prompt-clip sampling: stride 10, not "3 FPS"** [DONE]
   The history video is stride-10 @ 30 fps = 3.33 fps effective, and the processor
   stamps ONE seconds-per-frame for all videos in a sequence — sampling the prompt at
   the same stride keeps M-RoPE timing truthful for both. Final frame (grasp outcome)
   always included; uniformly re-spaced down to 12 frames max. Yields 7–12 frames.
4. **Prompt-video holdout** [DONE]
   Last 4 clips per pool are never used in training → the prompt-swap success test
   measures reading an UNSEEN demo.
5. **No color word anywhere the model can shortcut** [DONE]
   The user turn asks a fixed question ("which colored block did the human demonstrate
   picking up?"); the color appears only in the supervised assistant answer
   (`pick up green block`) — the VLM must extract it from the demo clip.
6. **Sequence structure** [DONE]
   `Human demonstration:` <prompt video> `Robot view:` <history video> <right wrist>
   <left wrist> `Task: <question>, Past states: …, State: …` (+ assistant subtask turn,
   + FAST tokens). Two separate videos in one user turn — natively supported by the
   processor; the expert attends the prompt tokens (default, nothing masked).
7. **Reuse, don't build**: augmentation (per-clip crop/rotate/jitter, independent draw
   per video) — existing `_augment`. Input-shape contract — existing
   `visual_budget.json` stamp, extended with the human-prompt fields [DONE]. Norm stats
   travel with the FAST artifact and the checkpoint — existing mechanism.

## 4. Architecture (unchanged from the ee6d recipe)

Qwen3-VL-4B + 18-layer flow-matching expert (~458M), knowledge insulation (detached KV),
subtask insulation (no decode at serve), L18 early exit, EE6D action space (10-dim,
Gram-Schmidt + seeded DLS IK at serve), FAST co-training (CE on VLM), training-time RTC
d~U[0,10], both wrists, state history, pi05 augmentation, constant LR + EMA 0.999.

## 5. Artifacts [DONE]

`checkpoints/fast_tokenizer_trossen_0717m_ee6d_gated/` — FAST tokenizer + frozen
`norm_stats.json`, fit on exactly the gated distribution: 223 episodes, 29,974 chunks
(50, 10), token lengths mean 71.5 / p95 103 / max 247, round-trip err 0.0108.

## 6. Training [READY]

`scripts/train_action_expert_4b_humanprompt.sh` — gated 0717 + demo pools, effective
batch 64 (GPU-count-invariant), max_steps 30000 as a ceiling (stop on plateau),
save every 2000 keep 3. Output:
`checkpoints/qwen3_4b_ae_humanprompt_0717m_ee6d_rtc10_subinsul_L18_vis16_constlr`.

Pre-flight gate: `tests/smoke_test_human_prompt_data.py` (episode/pool accounting,
history clamp at the cut, 2-video structure, no color leak in the user turn,
color-consistent pairing + supervision, frozen-artifact stats).

## 7. Serving & deployment [TODO — next build step]

- `inference.templatize`: accept prompt frames, build the identical 2-video structure;
  `build_data_args` reads the human-prompt fields from `visual_budget.json`.
- Server: `POST /set_prompt` (JPEG frames, stored under session_id + prompt_id;
  re-encoded per request in v1 — prompt KV caching is a later latency optimization);
  `/infer` gains `prompt_id`; `/health` reports the mode. Compile path unaffected
  (padded-bucket static shapes absorb the longer prefix).
- Client: record demo with rig cam_high; new prompt → new session (fresh RTC queue).
- Latency estimate: prompt video ≈ one extra history-video's worth of prefill
  (~+40–50 ms eager); flow loop (the dominant cost) unchanged.

## 8. Success test (offline first, then robot)

Same robot observation + state, swap only the prompt (held-out clips):
- green demo → predicted EE trajectory toward green; yellow demo → toward yellow;
- controls: wrong prompt / blank prompt / temporally shuffled prompt.
Success = changing only the human demo changes which block the robot approaches.

## 9. Risks / open items

- 63 demo clips total is small; if prompt-reading is weak, record more clips (cheap —
  no robot teleop needed) before touching the architecture.
- Right after the cut, robot-side frames may show the hand retreating — roughly matches
  deployment (hand leaves right before the robot starts). Do not cut earlier.
- Two-video M-RoPE path is exercised by the smoke test + input dumps
  (`QWEN_DUMP_MODEL_INPUTS` writes reconstructed PNGs of the first batches — eyeball
  that the demo clip and history video are both present and distinct).
