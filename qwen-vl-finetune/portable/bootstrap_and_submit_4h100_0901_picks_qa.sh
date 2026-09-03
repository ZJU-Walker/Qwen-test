#!/usr/bin/env bash
# One-command new-cluster bootstrap for fresh dense20 + 0901 standalone-pick QA.
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ==============================================================================
# NEW-CLUSTER CONFIGURATION
# ==============================================================================
GCS_RELEASE_URI="${GCS_RELEASE_URI:-gs://YOUR_BUCKET/qwen-sort/dense20-0901-picks-qa-fresh/v1}"
GCS_PUBLIC_READ="${GCS_PUBLIC_READ:-True}"
WORK_ROOT="${WORK_ROOT:-/scratch/${USER:-user}/qwen_dense20_0901_picks_qa}"

WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-qwen-dense20-0901-picks-qa}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"

SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_TIME="${SLURM_TIME:-3-00:00:00}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-32}"
SLURM_MEM="${SLURM_MEM:-700G}"
SLURM_GPU_ARGUMENT="${SLURM_GPU_ARGUMENT:---gres=gpu:h100:4}"
SLURM_EXTRA_ARGUMENTS="${SLURM_EXTRA_ARGUMENTS:-}"
GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-}"

MIN_CAPACITY_GIB="${MIN_CAPACITY_GIB:-340}"
MIN_AVAILABLE_GIB="${MIN_AVAILABLE_GIB:-150}"
# ==============================================================================

RELEASE_ID="dense20-0901-picks-qa-fresh-v1"
QWEN_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
FAST_NAME="fast_tokenizer_trossen_0824sort_0827ball_0901picks_ee6d"
RUN_NAME="qwen3_4b_ae_humanprompt_0824sort_0827ball_0901picksqa_dense20s3_fresh_4h100_ee6d_rtc20_subattend_L18_vis46_constlr"

TOOLS_ROOT="$WORK_ROOT/tools"
ENV_PREFIX="$WORK_ROOT/envs/qwen3vl-h100-cu124"
PIP_CACHE_DIR="$WORK_ROOT/cache/pip"
DOWNLOAD_ROOT="$WORK_ROOT/downloads/$RELEASE_ID"
RUNTIME_ROOT="$WORK_ROOT/runtime/$RELEASE_ID"
DATA_ROOT="$RUNTIME_ROOT/data"
ARTIFACT_ROOT="$RUNTIME_ROOT/artifacts"
MODEL_ROOT="$RUNTIME_ROOT/model/qwen3-vl-4b-$QWEN_REVISION"
CHECKPOINT_ROOT="$WORK_ROOT/checkpoints"
OUTPUT_DIR="$CHECKPOINT_ROOT/$RUN_NAME"
LOG_DIR="$WORK_ROOT/logs"
WANDB_DIR="$WORK_ROOT/wandb"

fail() { echo "ERROR: $*" >&2; exit 1; }
note() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
need() { command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"; }

ensure_work_root() {
  [[ "$WORK_ROOT" == /* ]] || fail "WORK_ROOT must be absolute"
  [[ "$WORK_ROOT" != / && "$WORK_ROOT" != /tmp && "$WORK_ROOT" != "$HOME" ]] || \
    fail "unsafe or undersized WORK_ROOT: $WORK_ROOT"
  mkdir -p "$WORK_ROOT" "$TOOLS_ROOT" "$PIP_CACHE_DIR" "$LOG_DIR" "$WANDB_DIR"
  local available existing capacity
  available="$(df -Pk "$WORK_ROOT" | awk 'NR==2 {printf "%d", $4/1024/1024}')"
  existing="$(du -sk "$WORK_ROOT" | awk '{printf "%d", $1/1024/1024}')"
  capacity=$((available + existing))
  (( capacity >= MIN_CAPACITY_GIB )) || \
    fail "WORK_ROOT effective capacity is ${capacity} GiB; need ${MIN_CAPACITY_GIB} GiB"
  (( available >= MIN_AVAILABLE_GIB )) || \
    fail "filesystem has ${available} GiB free; need ${MIN_AVAILABLE_GIB} GiB"
}

ensure_gcloud() {
  if command -v gcloud >/dev/null 2>&1; then return; fi
  if [[ -x "$TOOLS_ROOT/google-cloud-sdk/bin/gcloud" ]]; then
    export PATH="$TOOLS_ROOT/google-cloud-sdk/bin:$PATH"
    return
  fi
  need curl
  local installer="$TOOLS_ROOT/install-google-cloud-sdk.sh"
  curl -fsSL https://sdk.cloud.google.com -o "$installer"
  bash "$installer" --disable-prompts --install-dir="$TOOLS_ROOT"
  export PATH="$TOOLS_ROOT/google-cloud-sdk/bin:$PATH"
  need gcloud
}

authenticate_gcs() {
  case "${GCS_PUBLIC_READ,,}" in
    true|1|yes)
      export CLOUDSDK_CONFIG="$WORK_ROOT/gcloud-public-anonymous"
      mkdir -p "$CLOUDSDK_CONFIG"
      gcloud config set auth/disable_credentials True >/dev/null
      gcloud storage ls "$GCS_RELEASE_URI/release-manifest.json" >/dev/null || \
        fail "cannot anonymously read $GCS_RELEASE_URI"
      ;;
    false|0|no)
      if [[ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]]; then
        [[ -r "$GOOGLE_APPLICATION_CREDENTIALS" ]] || fail "credential file is unreadable"
        gcloud auth activate-service-account --quiet \
          --key-file="$GOOGLE_APPLICATION_CREDENTIALS"
      elif ! gcloud auth print-access-token >/dev/null 2>&1; then
        gcloud auth login --no-launch-browser
      fi
      gcloud storage ls "$GCS_RELEASE_URI/release-manifest.json" >/dev/null || \
        fail "cannot read $GCS_RELEASE_URI"
      ;;
    *) fail "GCS_PUBLIC_READ must be True or False" ;;
  esac
}

ensure_micromamba() {
  local mamba="$TOOLS_ROOT/micromamba"
  if [[ ! -x "$mamba" ]]; then
    need curl; need tar
    local unpack
    unpack="$(mktemp -d "$WORK_ROOT/.micromamba.XXXXXX")"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
      tar -xj -C "$unpack" bin/micromamba
    mv "$unpack/bin/micromamba" "$mamba"
  fi
  printf '%s\n' "$mamba"
}

ensure_python_env() {
  local mamba requirements_hash ready_hash=""
  mamba="$(ensure_micromamba)"
  requirements_hash="$(sha256sum "$SCRIPT_DIR/requirements-h100-cu124.txt" | awk '{print $1}')"
  [[ -f "$ENV_PREFIX/.portable-ready" ]] && ready_hash="$(cat "$ENV_PREFIX/.portable-ready")"
  if [[ "$ready_hash" == "$requirements_hash" && -x "$ENV_PREFIX/bin/python" ]]; then
    export PATH="$ENV_PREFIX/bin:$PATH"
    return
  fi
  if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    "$mamba" create -y -p "$ENV_PREFIX" -c conda-forge python=3.11 pip zstd
  fi
  "$ENV_PREFIX/bin/python" -m pip install --upgrade pip
  "$ENV_PREFIX/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0
  DS_BUILD_OPS=0 "$ENV_PREFIX/bin/python" -m pip install \
    -r "$SCRIPT_DIR/requirements-h100-cu124.txt"
  PYTHONPATH="$REPO_DIR" "$ENV_PREFIX/bin/python" - <<'PY'
import deepspeed, torch, transformers
assert torch.__version__.startswith("2.6.0") and torch.version.cuda == "12.4"
assert transformers.__version__ == "4.57.6"
assert deepspeed.__version__ == "0.17.1"
print("Pinned Python/CUDA environment: PASS")
PY
  printf '%s\n' "$requirements_hash" > "$ENV_PREFIX/.portable-ready"
  export PATH="$ENV_PREFIX/bin:$PATH"
}

login_wandb() {
  if [[ -z "$WANDB_API_KEY" ]]; then
    [[ -t 0 ]] || fail "WANDB_API_KEY is unset and no terminal is available"
    read -r -s -p "W&B API key: " WANDB_API_KEY; echo
  fi
  WANDB_API_KEY="$WANDB_API_KEY" "$ENV_PREFIX/bin/python" - <<'PY'
import os, wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True, verify=True)
print("W&B authentication: PASS")
PY
  unset WANDB_API_KEY
}

prepare_wandb_run_id() {
  local id_file="$WANDB_DIR/0901-picks-qa-run-id"
  if [[ -z "$WANDB_RUN_ID" ]]; then
    if [[ -s "$id_file" ]]; then
      WANDB_RUN_ID="$(cat "$id_file")"
    else
      WANDB_RUN_ID="d20-0901-picks-qa-$(date -u '+%Y%m%dT%H%M%SZ')"
      (umask 077; printf '%s\n' "$WANDB_RUN_ID" > "$id_file")
    fi
  fi
  export WANDB_RUN_ID
}

download_release() {
  mkdir -p "$DOWNLOAD_ROOT/manifests" "$DOWNLOAD_ROOT/assets"
  gcloud storage cp "$GCS_RELEASE_URI/release-manifest.json" \
    "$DOWNLOAD_ROOT/release-manifest.json"
  gcloud storage cp "$GCS_RELEASE_URI/manifests/SHA256SUMS.assets" \
    "$DOWNLOAD_ROOT/manifests/SHA256SUMS.assets"

  "$ENV_PREFIX/bin/python" - <<PY
import json, subprocess
from pathlib import Path
p = json.loads(Path(${DOWNLOAD_ROOT@Q}, "release-manifest.json").read_text())
if p.get("release_id") != ${RELEASE_ID@Q}:
    raise SystemExit(f"release ID mismatch: {p.get('release_id')}")
if p.get("records", {}).get("total") != 641:
    raise SystemExit(f"record manifest mismatch: {p.get('records')}")
release_commit = p["git_commit"]
local_commit = subprocess.check_output(
    ["git", "-C", ${REPO_DIR@Q}, "rev-parse", "HEAD"], text=True
).strip()
if release_commit != local_commit:
    subprocess.run(
        ["git", "-C", ${REPO_DIR@Q}, "merge-base", "--is-ancestor",
         release_commit, local_commit], check=True
    )
    relevant = [
        "qwenvl/action_expert/human_prompt.py",
        "qwenvl/action_expert/modeling_qwen3vl_with_expert.py",
        "qwenvl/data/robot_data.py",
        "qwenvl/data/subtask_formats_sort.py",
        "qwenvl/train/expert_ema.py",
        "qwenvl/train/train_action_expert.py",
        "scripts/sort_0827_ball_data.sh",
        "scripts/sort_0901_picks_data.sh",
        "scripts/train_action_expert_4b_humanprompt.sh",
        "scripts/train_action_expert_4b_sort_0901_picks.sh",
        "scripts/train_action_expert_4b_sort_0901_picks_dense20_fresh_qa_4h100.sh",
    ]
    subprocess.run(
        ["git", "diff", "--quiet", release_commit, local_commit, "--", *relevant],
        cwd=${REPO_DIR@Q}, check=True,
    )
print("Release manifest: PASS")
PY

  local assets=(
    sort0824-robot-clean.tar.zst
    sort0824-human-clean.tar.zst
    sort0827-green-grey-tape-ball.tar.zst
    sort0901-picks.tar.zst
    "$FAST_NAME.tar.zst"
    "qwen3-vl-4b-$QWEN_REVISION.tar.zst"
  )
  local name expected
  for name in "${assets[@]}"; do
    expected="$(awk -v p="assets/$name" '$2 == p {print $1}' \
      "$DOWNLOAD_ROOT/manifests/SHA256SUMS.assets")"
    [[ -n "$expected" ]] || fail "asset absent from checksum manifest: $name"
    if [[ ! -f "$DOWNLOAD_ROOT/assets/$name" ]] || ! \
       printf '%s  %s\n' "$expected" "$DOWNLOAD_ROOT/assets/$name" | \
         sha256sum -c - >/dev/null 2>&1; then
      gcloud storage cp "$GCS_RELEASE_URI/assets/$name" \
        "$DOWNLOAD_ROOT/assets/$name"
    fi
  done
  (cd "$DOWNLOAD_ROOT" && sha256sum -c manifests/SHA256SUMS.assets)
}

install_runtime() {
  if [[ -f "$RUNTIME_ROOT/release-manifest.json" ]]; then
    cmp -s "$RUNTIME_ROOT/release-manifest.json" "$DOWNLOAD_ROOT/release-manifest.json" || \
      fail "existing runtime belongs to another release"
    return
  fi
  [[ ! -e "$RUNTIME_ROOT" ]] || fail "incomplete runtime exists; move it aside: $RUNTIME_ROOT"
  local stage
  mkdir -p "$WORK_ROOT/runtime"
  stage="$(mktemp -d "$WORK_ROOT/runtime/.${RELEASE_ID}.XXXXXX")"
  mkdir -p "$stage/data" "$stage/artifacts" \
    "$stage/model/qwen3-vl-4b-$QWEN_REVISION"
  tar --zstd -xf "$DOWNLOAD_ROOT/assets/sort0824-robot-clean.tar.zst" -C "$stage/data"
  tar --zstd -xf "$DOWNLOAD_ROOT/assets/sort0824-human-clean.tar.zst" -C "$stage/data"
  tar --zstd -xf "$DOWNLOAD_ROOT/assets/sort0827-green-grey-tape-ball.tar.zst" -C "$stage/data"
  tar --zstd -xf "$DOWNLOAD_ROOT/assets/sort0901-picks.tar.zst" -C "$stage/data"
  tar --zstd -xf "$DOWNLOAD_ROOT/assets/$FAST_NAME.tar.zst" -C "$stage/artifacts"
  tar --zstd -xf "$DOWNLOAD_ROOT/assets/qwen3-vl-4b-$QWEN_REVISION.tar.zst" \
    -C "$stage/model/qwen3-vl-4b-$QWEN_REVISION"
  cp "$DOWNLOAD_ROOT/release-manifest.json" "$stage/release-manifest.json"
  mv "$stage" "$RUNTIME_ROOT"
}

run_data_preflight() {
  [[ -f "$ARTIFACT_ROOT/$FAST_NAME/norm_stats.json" ]] || fail "FAST artifact incomplete"
  [[ -f "$MODEL_ROOT/config.json" ]] || fail "Qwen snapshot incomplete"
  export SORT_0824_ROOT="$DATA_ROOT/0824_prompting"
  export SORT_0827_ROOT="$DATA_ROOT/0827_prompting_playdata/data_0827_prompting_playdata"
  export SORT_0901_ROOT="$DATA_ROOT"
  export SORT_0901_PICKS_FAST_DEFAULT="$ARTIFACT_ROOT/$FAST_NAME"
  PYTHONPATH="$REPO_DIR" "$ENV_PREFIX/bin/python" \
    "$REPO_DIR/tests/smoke_test_sort_0901_picks_recipe.py"
  PYTHONPATH="$REPO_DIR" "$ENV_PREFIX/bin/python" \
    "$REPO_DIR/tests/smoke_test_standalone_pick_training.py"
  PYTHONPATH="$REPO_DIR" "$ENV_PREFIX/bin/python" \
    "$REPO_DIR/tests/smoke_test_sort_0901_picks_dense20_loader.py"
  PYTHONPATH="$REPO_DIR" "$ENV_PREFIX/bin/python" \
    "$REPO_DIR/tests/test_sort_0901_picks_dense20_4h100_recipe.py"
  DENSE20_PROCESSOR_PATH="$MODEL_ROOT" PYTHONPATH="$REPO_DIR" \
    "$ENV_PREFIX/bin/python" \
    "$REPO_DIR/tests/preflight_dense20_prompt_contract_cpu.py"
}

inside_slurm() {
  export PYTHONPATH="$REPO_DIR"
  export PATH="$ENV_PREFIX/bin:$PATH"
  export SORT_0824_ROOT="$DATA_ROOT/0824_prompting"
  export SORT_0827_ROOT="$DATA_ROOT/0827_prompting_playdata/data_0827_prompting_playdata"
  export SORT_0901_ROOT="$DATA_ROOT"
  export SORT_0901_PICKS_FAST_DEFAULT="$ARTIFACT_ROOT/$FAST_NAME"
  export MODEL_PATH="$MODEL_ROOT"
  export CUSTOM_CACHE_DIR="$WORK_ROOT/cache/qwen"
  export CHECKPOINT_ROOT TORCHRUN="$ENV_PREFIX/bin/torchrun"
  export WANDB_DIR WANDB_PROJECT WANDB_MODE=online
  [[ -n "$WANDB_ENTITY" ]] && export WANDB_ENTITY
  export WANDB_RUN_ID WANDB_RESUME=allow
  export NPROC_PER_NODE=4 PER_DEVICE_BATCH=1 NUM_WORKERS=4
  export SAVE_STEPS=500 SAVE_TOTAL_LIMIT=2
  export QWEN_ATTN_IMPL=sdpa TORCH_COMPILE=False PYTHONFAULTHANDLER=1
  export DEEPSPEED_TIMEOUT=120 NCCL_NVLS_ENABLE=0 INIT_FROM=

  cd "$REPO_DIR"
  "$ENV_PREFIX/bin/python" - <<'PY'
import torch
if torch.cuda.device_count() != 4:
    raise SystemExit(f"expected 4 GPUs, got {torch.cuda.device_count()}")
for i in range(4):
    p = torch.cuda.get_device_properties(i)
    gib = p.total_memory / 1024**3
    if "H100" not in p.name or gib < 75:
        raise SystemExit(f"GPU {i} is not an 80-GB H100: {p.name}, {gib:.1f} GiB")
    print(f"GPU {i}: {p.name}, {gib:.1f} GiB")
PY
  if find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' \
      -print -quit 2>/dev/null | grep -q .; then
    "$ENV_PREFIX/bin/python" - <<PY
from qwenvl.train.expert_ema import select_latest_complete_checkpoint
p = select_latest_complete_checkpoint(
    ${OUTPUT_DIR@Q}, expected_ema_decay=0.999,
    expected_world_size=4, require_deepspeed=True,
)
if p is None:
    raise SystemExit("checkpoint directories exist but none is complete")
print(f"Exact-resuming this 0901 run from {p}")
PY
  else
    note "No prior checkpoint: starting a fresh action policy from the pretrained Qwen base"
  fi
  exec bash "$REPO_DIR/scripts/train_action_expert_4b_sort_0901_picks_dense20_fresh_qa_4h100.sh"
}

submit_slurm() {
  need sbatch
  local args job_id gpu_args=() extra_args=()
  args=(--parsable --job-name=qwen-d20-0901-pickqa --nodes=1 --ntasks=1
    --cpus-per-task="$SLURM_CPUS_PER_TASK" --mem="$SLURM_MEM" --time="$SLURM_TIME"
    --output="$LOG_DIR/%x-%j.out" --error="$LOG_DIR/%x-%j.out" --export=ALL)
  [[ -n "$SLURM_PARTITION" ]] && args+=(--partition="$SLURM_PARTITION")
  [[ -n "$SLURM_ACCOUNT" ]] && args+=(--account="$SLURM_ACCOUNT")
  [[ -n "$SLURM_QOS" ]] && args+=(--qos="$SLURM_QOS")
  read -r -a gpu_args <<< "$SLURM_GPU_ARGUMENT"; args+=("${gpu_args[@]}")
  if [[ -n "$SLURM_EXTRA_ARGUMENTS" ]]; then
    read -r -a extra_args <<< "$SLURM_EXTRA_ARGUMENTS"; args+=("${extra_args[@]}")
  fi
  job_id="$(sbatch "${args[@]}" "$SCRIPT_PATH" --inside-slurm)"
  echo "Submitted Slurm job: $job_id"
  echo "Log: $LOG_DIR/qwen-d20-0901-pickqa-${job_id%%;*}.out"
  echo "Output: $OUTPUT_DIR"
}

if [[ "${1:-}" == "--inside-slurm" ]]; then inside_slurm; exit 0; fi
[[ "$GCS_RELEASE_URI" == gs://* && "$GCS_RELEASE_URI" != *YOUR_BUCKET* ]] || \
  fail "export GCS_RELEASE_URI with the published release URI"
ensure_work_root
ensure_gcloud
authenticate_gcs
ensure_python_env
login_wandb
prepare_wandb_run_id
download_release
install_runtime
run_data_preflight
submit_slurm
