# Handoff — Human-Video-Prompted VLA

Written 2026-08-06. Repo: `/iris/projects/humanoid/ke/Qwen3-VL`
(git `git@github.com:ZJU-Walker/Qwen-test.git`, branch `main`, HEAD `e560b9b`).
Synced from a colleague's ("brian's") repo on 2026-08-05; all paths rewritten to `ke`.

---

## 1. What we're building

Replace the language instruction with a **short human demonstration video**. A human is
filmed picking up a green or yellow block; the robot, seeing both blocks on the table,
must pick the same one.

Success criterion: **changing only the human demo video changes which block the robot
approaches** (tested with held-out demo clips; controls = wrong prompt / blank prompt /
temporally shuffled prompt).

Design doc: `human_video_prompt_vla_plan_v2.md` (v2 = the original plan reconciled with
the actual codebase; v1 `human_video_prompt_vla_plan.md` is superseded).

## 2. Data

| Data | Path | Contents |
|---|---|---|
| Robot demos A | `trossen_data/0717_green_yellow_block_mem_merged` | 224 recordings, 30 fps; 223 usable. Human points, then robot picks. Episode 149 is non-canonical and auto-dropped. |
| Robot demos B | `trossen_data/0731_green_yellow_merged` | 224 recordings, same format; 217 usable. Seven partial/non-canonical recordings (128/129/130/133/150/152/153) are waiting-only and auto-dropped. |
| Subtask labels | `<dataset>/videos/chunk-000/subtask_labels.json` | Generated from proprioception after a startup guard: 10 quiet frames, then 5 sustained right-arm-motion frames above 0.005 rad/frame, reject onset before frame 40, cut at onset+10. Final usable cuts span frames 51–177 (mean 85). Original 0717 human labels are preserved as `subtask_labels_human_backup.json` and provide its colors; 0731 colors use the verified 0–109 green / 110+ yellow collection boundary. |
| Human demos (green) | `trossen_data/green_human_prompt` | 31 clips, 30 fps, 68–107 frames (~3 s), same rig/cameras. |
| Human demos (yellow) | `trossen_data/yellow_human_prompt` | 32 clips, same format. |

Block positions were randomized during collection (user-confirmed), so random pairing
gives genuine position mismatch between demo and robot scene — no copy-the-coordinates
shortcut.

Note: the legacy `qwen_action_expert_norm_stats.json` in the 0717 dir belongs to another
user and is **mode 600 / unreadable**. Do not try to use or delete it; norm stats now
travel with the FAST artifact and the checkpoint (see §4).

## 3. Architecture (inherited, unchanged)

- Qwen3-VL-4B-Instruct + ~458M-param **18-layer flow-matching action expert**, which
  cross-attends per-layer to the VLM's **detached** KV cache (knowledge insulation: the
  flow loss can never reach the VLM; the VLM is trained by CE on FAST action tokens +
  subtask text).
- Action space **ee6d** (10-dim: xyz + 6D rotation + jaw). Joints→EE by FK at load;
  back via Gram-Schmidt + seeded DLS IK at serve. Deltas `9,-1`. Horizon 50, 10 Euler
  steps at inference.
- SmolVLA early-exit at VLM layer 18/36; subtask insulation (no subtask decode at serve);
  training-time RTC prefix d~U[0,10] (10, not 20 — matches the deployed client's 8–10
  tick delay; brian's 2026-08-02 change).
- Both wrist cameras + state history (9 past states, frame-aligned to image history).

## 4. What was implemented for this project

All under `qwen-vl-finetune/`:

1. **`--skip_leading_subtask waiting`** (`qwenvl/data/robot_data.py`). Robot episodes
   start with a human *pointing* at the target. That segment is gated out of training
   timesteps **and out of image history / past states** (`hist_min` clamp) — unlike the
   pre-existing DAgger `min_start` gate, which deliberately lets history reach back.
   Rationale: pointing in the robot's own view would leak the task and let the model
   bypass the prompt video, and deployment has no pointing phase at all. Mirrored in
   `scripts/train_fast_tokenizer.py` so stats/FAST cover exactly the trainable timesteps.
2. **Human-prompt conditioning** (`robot_data.py`).
   `--human_prompt_dirs "green=…,yellow=…"` builds per-color clip pools (last 4 of each
   held out for eval → 27 green / 28 yellow in training). Every `__getitem__` draws a
   **fresh random same-color clip** and prepends it as a second video:
   `"Human demonstration:" <clip> "Robot view:" <history video> <2 wrist stills> <text>`.
   Clip sampling: stride 10 (matches the history video's 3.3 fps effective rate so the
   shared M-RoPE seconds-per-frame is truthful), final frame always included, ≤12 frames,
   cached per clip per worker. The color word appears **only** in the supervised
   assistant answer — never in the user turn (`--subtask_question` names no color).
3. **Input-shape contract stamping** (`qwenvl/train/train_action_expert.py`): the
   human-prompt fields are written into `visual_budget.json` next to the checkpoint, so
   serving can auto-match (same mechanism as pixel budgets / wrist cameras / state
   history). New launches also stamp an explicit `human_prompt_enabled` bit and the exact
   `subtask_question`; the current run predates those two keys, so serving has a strict
   compatibility fallback for its known question while still reading mode/stride/max from
   the existing stamp.
4. **Artifacts**: FAST tokenizer + frozen `norm_stats.json` fit on exactly the final
   gated 0717+0731 distribution →
   `checkpoints/fast_tokenizer_trossen_0717m0731_ee6d_gated`
   (440 eps, 59,252 chunks of (50,10), token len mean 48.3 / p95 70 / max 119,
   round-trip err 0.0106, `min_start_frames=37294`). The strengthened smoke test checks
   the exact two-dataset provenance and fingerprint so a partial fit cannot pass again.
5. **Scripts**: `scripts/train_action_expert_4b_humanprompt.sh`, plus
   `scripts/train_action_expert_4b_humanprompt_1gpu.sh` (thin wrapper using a new
   `RUN_TAG` hook so concurrent variants get separate output dirs + wandb names).
6. **Test**: `tests/smoke_test_human_prompt_data.py` — **passes** on 440 episodes. Gates episode/pool
   accounting, the history clamp at the cut, the 2-video structure, absence of color
   leakage in the user turn, pairing↔supervision color consistency, frozen-artifact stats.

## 5. Training status

The pipeline **works and learns**. On 2026-08-07 the repaired 2×H200 SDPA run passed
all former failure points (steps 109, 218, and 267), completed the 300-step LR warmup,
passed repeated virtual-epoch boundaries, and was healthy through step 3845 while this
handoff was updated: recent loss ~0.85–0.95, finite grad norms, no NCCL/SIGSEGV/NaN.
`checkpoint-2000` was structurally verified (both ZeRO optimizer shards, model state,
trainer/RNG/scheduler state, EMA expert) and training resumed after its save.
Throughput is ~13.7 s/step with SDPA on 2×H200 (~12 s/step with FA2), and ~15–17
s/step on 1×H200 (effective batch 64 in both; grad-accum is auto-derived from GPU count,
so training math is identical).

Previous 0717-only output dir:
`checkpoints/qwen3_4b_ae_humanprompt_0717m_ee6d_rtc10_subinsul_L18_vis16_constlr_2gpu_sdpa_noperiodic`.
Saves every 2000 steps, keep 3.

The replacement 0717+0731 run uses 4×H200 on Modal, effective batch 64
(per-device 2 × 4 × accumulation 8), and writes to:
`checkpoints/qwen3_4b_ae_humanprompt_0717m0731_ee6d_rtc10_subinsul_L18_vis16_constlr_modal4gpu`.
Keep the GPU count fixed at four across every 24-hour relaunch because its ZeRO-2
optimizer shards cannot resume under a different world size.

## 6. Open problems

### Problem 1 — NCCL watchdog timeout on multi-GPU (resolved and verified)

**Symptom.** Peer ranks time out in the 390,899,200-element embedding-gradient
all-reduce (`WorkNCCL(SeqNum=…, OpType=ALLREDUCE, NumelIn=390899200,
Timeout(ms)=600000)`) while **rank 0 is absent**; the watchdog aborts the process
(`exitcode -6`). Occurred twice at step 1 locally on 2×H200, and **reproduced
identically on Modal** with 4×H200 (ranks 1–3 waiting) — different hardware entirely,
so not a bad GPU.

**Proven diagnosis.** `QWEN_NAN_DEBUG=1` installed `EmbedNanWatchCallback`, whose
`on_pre_optimizer_step` hook called DeepSpeed
`safe_get_full_grad(model.get_input_embeddings().weight)`. Under ZeRO-2 this is a
full-parameter collective: `152695 * 2560 = 390,899,200`, exactly
the watchdog's element count. At this hook, gradient availability/errors could differ by
rank, so one rank entered that debug-only all-reduce while another followed the Trainer's
normal collective order. The apparent rank-0 I/O stall was therefore a collective-order
mismatch, not W&B, NFS, video decode, the worker count, or a bad GPU. With the gradient
probe skipped whenever `world_size > 1`, a clean 2-GPU run passed step 1 immediately and
the repaired production run passed step 301. The 2-hour fuse and node-local W&B remain
useful hardening, but neither fixed this failure.

### Problem 2 — native SIGSEGV (instrumentation bug fixed; old one-off remains uncertain)

The original uninstrumented FA2 run died once at step 267 with `exitcode -11`, healthy
metrics, and no traceback. During this investigation it appeared to reproduce at step
218 under FA2 and step 109 under SDPA, which initially implicated attention or ZeRO.
Those two reproductions were actually caused by the old stall instrument:

- `QWEN_STALL_DEBUG` armed `faulthandler.dump_traceback_later(90, repeat=True)` on both
  ranks and (before another fix) inside spawned dataloader workers too.
- In the SDPA reproduction the rank log was born at `23:40:15.540`; its 18th 90-second
  dump began at `00:07:15.561`, was cut/corrupted mid-frame, and rank 0 received SIGSEGV
  one second later. The FA2 reproduction's two rank logs likewise stopped on the same
  dump tick immediately before rank 0 died.
- The instrument now enables fatal-signal tracebacks but performs all-thread dumps only
  on explicit `SIGUSR1`, and only in the main rank processes. Never restore a periodic
  `dump_traceback_later(..., repeat=True)` in this workload.

With that change, the SDPA control survived the exact 27-minute/step-109 tick and then
passed steps 218 and 267. SDPA remains the conservative launcher default because the
older uninstrumented FA2 step-267 event cannot be retroactively attributed to the timer.

### Problem 3 — resolved, recorded so it is not reintroduced

An earlier crash produced `nan` in `lm_loss`/`fast_loss` at optimizer step 3, every run,
deterministically, while `flow_loss` stayed finite. Root cause:
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (added for OOM headroom) corrupting
`embed_tokens` rows 0–9 under DeepSpeed `overlap_comm` — a memory-layout bug, not a
gradient one. Both CE losses went nan because Qwen3-VL **ties** the LM head to the
embedding table, so 10 nan rows become 10 nan logit columns everywhere. Removing the flag
fixed it (`6152b9b`); the script carries a comment never to re-add it.

Diagnostic instrumentation from that hunt remains in the tree, **inert unless
`QWEN_NAN_DEBUG=1`**: micro-batch and optimizer-step embedding scans
(`EmbedNanWatchCallback`), corrupted-weights vs. data-trigger discrimination, batch dumps,
and `tests/replay_nan_batch.py` (replays a dump on fresh weights, bisects per sample then
per module).

Also fixed along the way: per-device batch 4 → 2 (two-video sequences OOM an H200 at 4;
effective batch unchanged), `dataloader_drop_last`, per-clip prompt-frame caching, virtual
epochs ×50 (`dataset_epoch_multiplier`) and `prefetch_factor 4` to stop dataloader
restarts from starving the GPU (440-sample epochs are ~7 optimizer steps).

## 7. Serving/deployment status

Built on 2026-08-07:

- `inference.templatize(..., human_prompt=...)` builds the exact training structure and
  rejects both missing prompts on a human-prompt checkpoint and unexpected prompts on a
  legacy checkpoint. Training-only prompt directories are converted to a boolean serving
  mode and never scanned/mounted at deployment.
- `POST /set_prompt`: accepts raw 30 fps JPEGs, applies the exact training index rule
  (stride 10, final frame forced, uniformly re-spaced to ≤12), and stores RGB frames in a
  bounded thread-safe LRU keyed by `(session_id, prompt_id)` with a content digest.
- `/infer` requires that identity, returns it/digest for client verification, and passes
  the stored video through every subtask/expert/FAST templating branch. `/health` exposes
  all human-prompt contract fields.
- `client_action_expert_rtc_ee6d_humanprompt.py`: thin adapter over the original ee6d RTC
  client. It verifies `/health`, records `cam_high` at 30 fps (or loads a clip), uploads
  once, pauses for the human to restore both blocks, then performs the original stationary
  cold-start/prefix warmup and unchanged RTC control loop. A new process creates a new
  session, preventing prompt/history leakage.
- `client_action_expert_rtc_ee6d_humanprompt_standalone.py`: generated, one-file robot
  deployment artifact containing that adapter and the complete RTC client. It has no
  source-time/runtime import of the original client and needs no Qwen checkout, weights,
  tokenizer, or data on the robot. Regenerate it after client edits with
  `qwen-vl-finetune/scripts/build_humanprompt_client_standalone.py`.
- CPU gates pass: exact sampling/storage/two-video order, client multipart/health contract,
  and the real FastAPI endpoint handlers with fake model math.

Still not done/verified:

- A real full-weight GPU `/set_prompt` → `/infer` smoke. `iris-ws-18` was offered for this,
  but its A5000 was already at 24.1/24.6 GB, so no existing workload was disturbed.
- The **offline held-out prompt-swap evaluation** — the actual success test of the project.
- Prompt-KV caching. The current one-video history-preprocessing cache is deliberately
  disabled in two-video mode because it cannot cache the second video correctly; v1
  re-encodes the prompt every request. Expect roughly one extra history-video's prefill.

## 8. How to reproduce / run

```bash
source /iris/projects/humanoid/miniconda3/bin/activate
conda activate /iris/projects/humanoid/miniconda3/envs/qwen3vl   # full path matters:
                                                                 # a personal ~/.conda env shadows it
cd /iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune
```

Data gates (CPU, ~2–3 min):
```bash
python tests/smoke_test_human_prompt_data.py
```

Serving/client CPU gates (from the repo root):
```bash
cd /iris/projects/humanoid/ke/Qwen3-VL
PYTHONPATH=.:qwen-vl-finetune python qwen-vl-finetune/tests/test_human_prompt_serving_cpu.py
PYTHONPATH=.:qwen-vl-finetune python qwen-vl-finetune/tests/test_human_prompt_client_cpu.py
PYTHONPATH=.:qwen-vl-finetune python qwen-vl-finetune/tests/test_human_prompt_server_api_cpu.py
```

Server (on a free inference GPU; start with eager, not `--compile`, for the first smoke):
```bash
source /iris/projects/humanoid/miniconda3/bin/activate
conda activate /iris/projects/humanoid/miniconda3/envs/qwen3vl
cd /iris/projects/humanoid/ke/Qwen3-VL
CUDA_VISIBLE_DEVICES=0 python qwen_action_expert_server.py \
  --ckpt checkpoints/qwen3_4b_ae_humanprompt_0717m_ee6d_rtc10_subinsul_L18_vis16_constlr_2gpu_sdpa_noperiodic/checkpoint-2000 \
  --fast_tok checkpoints/fast_tokenizer_trossen_0717m_ee6d_gated \
  --data_dirs /iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged \
  --port 8003
```
The checkpoint stamp automatically enables subtask insulation and `subtask_every=0`.
Do not pass `--subtask_input`. Confirm `/health` reports `human_prompt: true`, ee6d,
two wrists, state history, stride 10, and the human-demo question before connecting a robot.

Robot client (on the robot computer; copy only the standalone file):
```bash
cd /path/containing/the/copied/client
python client_action_expert_rtc_ee6d_humanprompt_standalone.py \
  --server http://<GPU-NODE-IP>:8003
```
Omit `--no-viz` (as above) for the built-in OpenCV live-demo and exact-input viewer;
add it only for headless operation.
For a held-out prerecorded clip instead of a live demonstration, add
`--prompt-video /path/to/episode_XXXXXX.mp4`. The client time-resamples it to the 30 fps
frame-index contract before upload.

Prompt-understanding diagnostic: restart the same server with `--subtask_every 1`, then
run the standalone client with `--diagnostic-subtask`. The viewer prominently displays
the VLM's real decoded green/yellow answer, and the client requires typed `GO` after
cold-start before sending the first robot action. The decoded text is telemetry only for
this subtask-insulated checkpoint; it is not fed into the action expert and adds decode
latency, so return both flags to their default (`subtask_every=0`, no diagnostic client
flag) for normal low-latency rollouts.

2-GPU production launch (verified past all former failure points):
```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_WORKERS=6 RUN_TAG=2gpu_sdpa_noperiodic \
  PYTHONFAULTHANDLER=1 QWEN_STALL_DEBUG=1 QWEN_NAN_DEBUG= \
  QWEN_ATTN_IMPL=sdpa QWEN_DUMP_MODEL_INPUTS= \
  WANDB_DIR=/tmp/qwen_wandb_sdpa_noperiodic DEEPSPEED_TIMEOUT=120 \
  bash scripts/train_action_expert_4b_humanprompt.sh \
  2>&1 | tee /tmp/train2gpu_sdpa_noperiodic.log
```

Run `bash scripts/monitor_stall.sh &` beside it. `QWEN_STALL_DEBUG=1` is now passive
during healthy training. Only after the sidecar confirms a real stall, send `SIGUSR1`
to the two direct torchrun children; their stacks append to
`/tmp/qwen_stall_rank{0,1}.log`. Do not signal dataloader grandchildren.

1-GPU (known-good, separate output dir via `RUN_TAG=1gpu`):
```bash
CUDA_VISIBLE_DEVICES=0 QWEN_NAN_DEBUG=1 PYTHONFAULTHANDLER=1 \
  bash scripts/train_action_expert_4b_humanprompt_1gpu.sh 2>&1 | tee /tmp/train1gpu.log
```

Rules that bite: switching GPU count requires a **fresh output dir** (ZeRO-2 cannot
resume across a GPU-count change); `pkill -f train_action_expert.py` kills *all* your
training processes on the node, so start concurrent runs in the right order; `save_steps`
is restored from a checkpoint's `trainer_state.json` on resume and silently overrides the
CLI.

Modal backup (4×H200), app + README in `/iris/projects/humanoid/ke/modal_training/`:
```bash
export PATH="$HOME/.local/bin:$PATH"
STAGE=$(mktemp -d /tmp/qwen-modal-stage.XXXXXX)
mkdir -p "$STAGE/repo"
git -C /iris/projects/humanoid/ke/Qwen3-VL archive HEAD | tar -x -C "$STAGE/repo"
modal volume put --force humanoid-training "$STAGE/repo" ke/Qwen3-VL
modal volume put --force humanoid-training \
  /iris/projects/humanoid/trossen_data/0731_green_yellow_merged \
  trossen_data/0731_green_yellow_merged
modal volume put --force humanoid-training \
  /iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged/videos/chunk-000/subtask_labels.json \
  trossen_data/0717_green_yellow_block_mem_merged/videos/chunk-000/subtask_labels.json
modal volume put --force humanoid-training \
  /iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717m0731_ee6d_gated \
  ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717m0731_ee6d_gated
cd /iris/projects/humanoid/ke/modal_training && modal run --detach modal_train.py
```
The repo copy must be re-uploaded whenever committed code changes. Re-upload the changed
0717 label file as well as 0731 and FAST whenever the cut rule/data mix changes.

## 9. Commit trail

```
e560b9b RUN_TAG variant hook + 1-GPU wrapper script for concurrent runs
9197a14 Local wandb dir off NFS so per-step appends can never stall rank 0
2c5fb8b Raise NCCL fuse to 2h (DEEPSPEED_TIMEOUT + ddp_timeout)
80ccd44 Stall diagnostics: faulthandler dumps (QWEN_STALL_DEBUG) + sidecar monitor
d3f666f Throughput: virtual epochs (x50), prefetch_factor 4
8c10189 Cache sampled human-prompt frames per clip
6152b9b Remove expandable_segments (root cause of the embed-row nan corruption)
cc738ed / 89f579c / 6979637  nan-debug instrumentation
ea12726 drop_last
6ba7267 per-device batch 4 -> 2 (two-video OOM)
2cc5eb3 Human-video-prompt training pipeline (pools, pairing, two-video samples)
eebf221 skip_leading_subtask gate (timesteps AND history)
```
