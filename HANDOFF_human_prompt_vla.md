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
| Robot demos | `trossen_data/0717_green_yellow_block_mem_merged` | 224 eps (114 green / 111 yellow), 30 fps. Each: human points at a block (~2 s), then robot picks it. |
| Subtask labels | `<dataset>/videos/chunk-000/subtask_labels.json` | Per episode: `waiting` segment (the pointing phase, ends frame 39–89, mean 58) then `pick up green/yellow block`. `episode_000149` is mislabeled (no `waiting`) → auto-dropped. |
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
   history).
4. **Artifacts**: FAST tokenizer + frozen `norm_stats.json` fit on exactly the gated
   distribution →
   `checkpoints/fast_tokenizer_trossen_0717m_ee6d_gated`
   (223 eps, 29,974 chunks of (50,10), token len mean 71.5 / p95 103 / max 247,
   round-trip err 0.0108).
5. **Scripts**: `scripts/train_action_expert_4b_humanprompt.sh`, plus
   `scripts/train_action_expert_4b_humanprompt_1gpu.sh` (thin wrapper using a new
   `RUN_TAG` hook so concurrent variants get separate output dirs + wandb names).
6. **Test**: `tests/smoke_test_human_prompt_data.py` — **passes**. Gates episode/pool
   accounting, the history clamp at the cut, the 2-video structure, absence of color
   leakage in the user turn, pairing↔supervision color consistency, frozen-artifact stats.

## 5. Training status

The pipeline **works and learns**. Best run reached step 267 before dying (see §6):
loss 28.7 → 4.7, `fast_loss` 22.5 → 4.6, `lm_loss` → ~0, `grad_norm` 537 → ~26.
Throughput ~12 s/step on 2×H200, ~15–17 s/step on 1×H200 (effective batch 64 in both;
grad-accum is auto-derived from GPU count, so training math is identical).

Output dir:
`checkpoints/qwen3_4b_ae_humanprompt_0717m_ee6d_rtc10_subinsul_L18_vis16_constlr`
(`_1gpu` suffix for the single-GPU variant). Saves every 2000 steps, keep 3.

## 6. Open problems

### Problem 1 — NCCL watchdog timeout on multi-GPU (fix committed, NOT verified)

**Symptom.** Peer ranks time out in the 390,899,200-element embedding-gradient
all-reduce (`WorkNCCL(SeqNum=…, OpType=ALLREDUCE, NumelIn=390899200,
Timeout(ms)=600000)`) while **rank 0 is absent**; the watchdog aborts the process
(`exitcode -6`). Occurred twice at step 1 locally on 2×H200, and **reproduced
identically on Modal** with 4×H200 (ranks 1–3 waiting) — different hardware entirely,
so not a bad GPU.

**Diagnosis (mine, mechanism-consistent but not directly captured).** Gradient
all-reduces fire only at accumulation boundaries (`deepspeed/runtime/zero/stage_1_and_2.py:1370,1467`),
so each step is 16 micro-batches of communication-free compute, then one burst of
collectives, then rank-0-only bookkeeping. Rank-0-only work — wandb per-step log writes
(NFS), input-dump PNGs, cold first-batch video decode — can block longer than the **bare
600 s NCCL fuse** while peers wait at the boundary. Single-GPU never fails because it has
no collectives: the same stall merely looks like a slow step. On Modal the rank-0 blocker
was the volume-commit thread locking the mount wandb was writing to.

**Fixes committed** (`2c5fb8b`, `9197a14`):
- Fuse raised to 2 h via **both** `DEEPSPEED_TIMEOUT=120` (minutes) and
  `--ddp_timeout 7200` — the PG init path was ignoring the framework defaults and using
  the bare NCCL value, so one knob alone is insufficient.
- `WANDB_DIR` → node-local `/tmp/qwen_wandb` (per-step appends never touch NFS).
- Input dumps made disableable (`QWEN_DUMP_MODEL_INPUTS=`).
- Modal: wandb → container-local disk, dumps off, volume commits 10 → 30 min.

**Caveats for whoever picks this up.**
- **The 2-GPU run has not been verified to survive past step 1 since the fix.**
- The mechanism was never caught red-handed: py-spy is ptrace-blocked on the cluster;
  `QWEN_STALL_DEBUG=1` writes per-rank faulthandler stacks to `/tmp/qwen_stall_rank*.log`
  every 90 s but was never active during a failing run. That is the instrument to use.
- Two earlier hypotheses were **falsified** — don't re-derive them: (a) dataloader worker
  count (6 vs 4), (b) a faulty GPU 1. The cross-site Modal reproduction killed (b).

### Problem 2 — unexplained segfault (one occurrence)

A 2-GPU run died at step 267 with `exitcode -11`, no Python traceback, healthy losses to
the last logged line. Different signature from Problem 1 (mid-run, well past the
fuse-sensitive window, and a native crash rather than a hang). Never reproduced.
`PYTHONFAULTHANDLER=1` is armed in the launch commands to capture C-level thread stacks
if it returns.

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
restarts from starving the GPU (223-sample epochs were ~3.5 optimizer steps).

## 7. Not yet built

The **serve side**, none of which exists yet:
- two-video `templatize` in `qwenvl/action_expert/inference.py`;
- server `/set_prompt` (store sampled prompt frames under session + `prompt_id`) and
  `prompt_id` on `/infer`; `build_data_args` reading the human-prompt fields back from
  `visual_budget.json`;
- the **offline prompt-swap evaluation** on held-out clips — i.e. the actual success test
  of the whole project.

Latency note for later: the prompt video adds roughly one history-video's worth of
prefill (~+40–50 ms eager); the flow loop, which dominates, is unchanged. Prompt-KV
caching is the obvious optimization once correctness is established.

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

2-GPU (the configuration that fails; all tripwires armed):
```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_WORKERS=6 QWEN_NAN_DEBUG=1 PYTHONFAULTHANDLER=1 QWEN_STALL_DEBUG=1 \
  bash scripts/train_action_expert_4b_humanprompt.sh 2>&1 | tee /tmp/train2gpu.log
```

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
rm -rf /tmp/modal_stage && mkdir -p /tmp/modal_stage/repo
git -C /iris/projects/humanoid/ke/Qwen3-VL archive HEAD | tar -x -C /tmp/modal_stage/repo
modal volume put --force humanoid-training /tmp/modal_stage/repo ke/Qwen3-VL
cd /iris/projects/humanoid/ke/modal_training && modal run --detach modal_train.py
```
The volume `humanoid-training` already holds the datasets and the FAST artifact; the repo
copy must be re-uploaded whenever local code changes.

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
