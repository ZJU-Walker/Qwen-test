#!/usr/bin/env bash
# One-command target-cluster bootstrap for the dense20 four-H100 no-standalone-QA run.
#
# First invocation (login node): install tools/env, authenticate W&B, fetch and verify
# the immutable GCS release, restore checkpoint-3000, and submit this same script via
# Slurm. The --inside-slurm mode is private and starts torchrun on the allocated node.
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ==============================================================================
# USER CONFIGURATION: EDIT ONLY THIS BLOCK ON THE NEW CLUSTER.
# ==============================================================================

# Immutable release written by publish_gcs_release.sh.
GCS_RELEASE_URI="${GCS_RELEASE_URI:-gs://YOUR_BUCKET/qwen-sort/dense20-noqa-resume3000/v1}"

# IMPORTANT: use shared high-capacity storage visible from login and compute nodes.
# Do not use /tmp or a small home quota. The reviewed retention policy needs >=340 GiB.
WORK_ROOT="${WORK_ROOT:-/scratch/${USER:-user}/qwen_dense20_noqa_resume3000}"

# Paste your key in place of WANDB_API_KEY_HERE, or preferably export WANDB_API_KEY
# before running this script. Never commit/push the edited key.
WANDB_API_KEY="${WANDB_API_KEY:-WANDB_API_KEY_HERE}"
WANDB_ENTITY="${WANDB_ENTITY:-}"  # optional: username/team; empty uses account default
WANDB_PROJECT="${WANDB_PROJECT:-qwen-dense20-noqa}"
# Leave empty to create a new W&B run whose first logged trainer step is 3001.
# Set only if you intentionally want to continue a known existing W&B run ID.
WANDB_RUN_ID="${WANDB_RUN_ID:-}"

# ---- Slurm settings: these are the cluster-specific lines you may need to change. ----
SLURM_PARTITION="${SLURM_PARTITION:-}"       # example: gpu; empty = site default
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"           # example: my_lab; empty = site default
SLURM_QOS="${SLURM_QOS:-}"                   # optional
SLURM_TIME="${SLURM_TIME:-3-00:00:00}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-32}"
SLURM_MEM="${SLURM_MEM:-700G}"
# Change this if the site uses --gpus=4 or a different GRES spelling.
SLURM_GPU_ARGUMENT="${SLURM_GPU_ARGUMENT:---gres=gpu:h100:4}"
# Optional additional sbatch flags, space-separated, e.g. "--constraint=h100 --exclusive".
SLURM_EXTRA_ARGUMENTS="${SLURM_EXTRA_ARGUMENTS:-}"

# GCS authentication: leave empty when `gcloud auth login` is already configured.
# For a service account, point this at a read-only JSON file OUTSIDE the git checkout.
GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-}"

# Capacity includes files this bootstrap has already installed, so reruns remain legal.
# The filesystem must offer ~340 GiB to this release and retain >=150 GiB free.
MIN_CAPACITY_GIB="${MIN_CAPACITY_GIB:-340}"
MIN_AVAILABLE_GIB="${MIN_AVAILABLE_GIB:-150}"

# ==============================================================================

RELEASE_ID="dense20-noqa-resume3000-v1"
QWEN_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
RUN_NAME="qwen3_4b_ae_humanprompt_0824sort_0827ballaction_dense20s3_fresh_norobotqa_4h100_ee6d_rtc20_subattend_L18_vis46_constlr"
CHECKPOINT_NAME="checkpoint-3000"
FAST_NAME="fast_tokenizer_trossen_0824sort_0827ball_ee6d"

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
  [[ "$WORK_ROOT" == /* ]] || fail "WORK_ROOT must be an absolute path"
  [[ "$WORK_ROOT" != / && "$WORK_ROOT" != /tmp && "$WORK_ROOT" != "$HOME" ]] || \
    fail "unsafe/undersized WORK_ROOT: $WORK_ROOT"
  mkdir -p "$WORK_ROOT" "$TOOLS_ROOT" "$PIP_CACHE_DIR" "$LOG_DIR" "$WANDB_DIR"
  local available_gib existing_gib effective_capacity_gib
  available_gib="$(df -Pk "$WORK_ROOT" | awk 'NR==2 {printf "%d", $4 / 1024 / 1024}')"
  existing_gib="$(du -sk "$WORK_ROOT" | awk '{printf "%d", $1 / 1024 / 1024}')"
  effective_capacity_gib=$((available_gib + existing_gib))
  if (( effective_capacity_gib < MIN_CAPACITY_GIB )); then
    fail "WORK_ROOT has only ${effective_capacity_gib} GiB effective capacity; ${MIN_CAPACITY_GIB} GiB is required"
  fi
  if (( available_gib < MIN_AVAILABLE_GIB )); then
    fail "filesystem has only ${available_gib} GiB free; at least ${MIN_AVAILABLE_GIB} GiB is required"
  fi
}

ensure_gcloud() {
  if command -v gcloud >/dev/null 2>&1; then
    return
  fi
  if [[ -x "$TOOLS_ROOT/google-cloud-sdk/bin/gcloud" ]]; then
    export PATH="$TOOLS_ROOT/google-cloud-sdk/bin:$PATH"
    return
  fi
  need curl
  note "Installing Google Cloud CLI under $TOOLS_ROOT ..."
  local installer="$TOOLS_ROOT/install-google-cloud-sdk.sh"
  curl -fsSL https://sdk.cloud.google.com -o "$installer"
  bash "$installer" --disable-prompts --install-dir="$TOOLS_ROOT"
  export PATH="$TOOLS_ROOT/google-cloud-sdk/bin:$PATH"
  need gcloud
}

authenticate_gcs() {
  if [[ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]]; then
    [[ -r "$GOOGLE_APPLICATION_CREDENTIALS" ]] || \
      fail "cannot read GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS"
    gcloud auth activate-service-account --quiet --key-file="$GOOGLE_APPLICATION_CREDENTIALS"
  elif ! gcloud auth print-access-token >/dev/null 2>&1; then
    note "No active Google Cloud login. Follow the one-time browser/device instructions."
    gcloud auth login --no-launch-browser
  fi
  gcloud storage ls "$GCS_RELEASE_URI/release-manifest.json" >/dev/null || \
    fail "cannot read release marker: $GCS_RELEASE_URI/release-manifest.json"
}

ensure_micromamba() {
  local mamba="$TOOLS_ROOT/micromamba"
  if [[ ! -x "$mamba" ]]; then
    need curl
    need tar
    note "Installing micromamba under $TOOLS_ROOT ..." >&2
    local unpack
    unpack="$(mktemp -d "$WORK_ROOT/.micromamba.XXXXXX")"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
      tar -xj -C "$unpack" bin/micromamba
    mv "$unpack/bin/micromamba" "$mamba"
  fi
  printf '%s\n' "$mamba"
}

ensure_python_env() {
  local mamba requirements_hash ready_hash
  mamba="$(ensure_micromamba)"
  requirements_hash="$(sha256sum "$SCRIPT_DIR/requirements-h100-cu124.txt" | awk '{print $1}')"
  ready_hash=""
  [[ -f "$ENV_PREFIX/.portable-ready" ]] && ready_hash="$(cat "$ENV_PREFIX/.portable-ready")"
  if [[ "$ready_hash" == "$requirements_hash" && -x "$ENV_PREFIX/bin/python" ]]; then
    note "Python environment already matches the pinned requirements."
    export PATH="$ENV_PREFIX/bin:$PATH"
    return
  fi

  if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    note "Creating Python 3.11 environment under $ENV_PREFIX ..."
    "$mamba" create -y -p "$ENV_PREFIX" -c conda-forge python=3.11 pip zstd
  fi

  note "Installing pinned CUDA 12.4 PyTorch and training dependencies ..."
  "$ENV_PREFIX/bin/python" -m pip install --upgrade pip
  "$ENV_PREFIX/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchvision==0.21.0
  DS_BUILD_OPS=0 "$ENV_PREFIX/bin/python" -m pip install \
    -r "$SCRIPT_DIR/requirements-h100-cu124.txt"
  PYTHONPATH="$REPO_DIR" "$ENV_PREFIX/bin/python" - <<'PY'
import accelerate, deepspeed, pyarrow, torch, transformers, wandb
assert torch.__version__.startswith("2.6.0"), torch.__version__
assert torch.version.cuda == "12.4", torch.version.cuda
assert transformers.__version__ == "4.57.6", transformers.__version__
assert deepspeed.__version__ == "0.17.1", deepspeed.__version__
print("Pinned Python/CUDA training environment: PASS")
PY
  printf '%s\n' "$requirements_hash" > "$ENV_PREFIX/.portable-ready"
  export PATH="$ENV_PREFIX/bin:$PATH"
}

login_wandb() {
  if [[ "$WANDB_API_KEY" == "WANDB_API_KEY_HERE" || -z "$WANDB_API_KEY" ]]; then
    fail "replace WANDB_API_KEY_HERE in the USER CONFIGURATION block or export WANDB_API_KEY"
  fi
  note "Logging in to Weights & Biases (the key will not be printed)..."
  WANDB_API_KEY="$WANDB_API_KEY" "$ENV_PREFIX/bin/python" - <<'PY'
import os
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True, verify=True)
print("W&B authentication: PASS")
PY
  # Authentication is now stored by W&B in the user's home credentials. Do not export
  # the plaintext key into the Slurm environment.
  unset WANDB_API_KEY
}

prepare_wandb_run_id() {
  local id_file="$WANDB_DIR/portable-run-id"
  if [[ -z "$WANDB_RUN_ID" ]]; then
    if [[ -s "$id_file" ]]; then
      WANDB_RUN_ID="$(cat "$id_file")"
    else
      WANDB_RUN_ID="dense20-noqa-r3k-$(date -u '+%Y%m%dT%H%M%SZ')"
      (umask 077; printf '%s\n' "$WANDB_RUN_ID" > "$id_file")
    fi
  fi
  export WANDB_RUN_ID
  note "W&B run continuity ID: $WANDB_RUN_ID"
}

download_release_metadata() {
  mkdir -p "$DOWNLOAD_ROOT/manifests"
  gcloud storage cp "$GCS_RELEASE_URI/release-manifest.json" \
    "$DOWNLOAD_ROOT/release-manifest.json"
  gcloud storage cp "$GCS_RELEASE_URI/manifests/SHA256SUMS.assets" \
    "$DOWNLOAD_ROOT/manifests/SHA256SUMS.assets"
  gcloud storage cp "$GCS_RELEASE_URI/manifests/SHA256SUMS.checkpoint-3000" \
    "$DOWNLOAD_ROOT/manifests/SHA256SUMS.checkpoint-3000"
  gcloud storage cp "$GCS_RELEASE_URI/manifests/SHA256SUMS.run-root" \
    "$DOWNLOAD_ROOT/manifests/SHA256SUMS.run-root"

  local release_commit local_commit manifest_release
  read -r release_commit manifest_release < <("$ENV_PREFIX/bin/python" - <<PY
import json
p = json.load(open(${DOWNLOAD_ROOT@Q} + "/release-manifest.json"))
print(p["git_commit"], p["release_id"])
PY
  )
  local_commit="$(git -C "$REPO_DIR" rev-parse HEAD)"
  [[ "$release_commit" == "$local_commit" ]] || \
    fail "code/release mismatch: git HEAD=$local_commit, GCS requires $release_commit"
  [[ "$manifest_release" == "$RELEASE_ID" ]] || \
    fail "unexpected release_id=$manifest_release (expected $RELEASE_ID)"
}

asset_is_valid() {
  local relative="$1" expected
  expected="$(awk -v p="$relative" '$2 == p {print $1}' "$DOWNLOAD_ROOT/manifests/SHA256SUMS.assets")"
  [[ -n "$expected" && -f "$DOWNLOAD_ROOT/$relative" ]] || return 1
  printf '%s  %s\n' "$expected" "$DOWNLOAD_ROOT/$relative" | sha256sum -c - >/dev/null 2>&1
}

download_assets() {
  mkdir -p "$DOWNLOAD_ROOT/assets"
  local assets=(
    sort0824-robot-clean.tar.zst
    sort0824-human-clean.tar.zst
    sort0827-green-grey-tape-ball.tar.zst
    "$FAST_NAME.tar.zst"
    "qwen3-vl-4b-$QWEN_REVISION.tar.zst"
  )
  local name relative
  for name in "${assets[@]}"; do
    relative="assets/$name"
    if asset_is_valid "$relative"; then
      note "Verified cached asset: $name"
    else
      note "Downloading asset: $name"
      gcloud storage cp "$GCS_RELEASE_URI/assets/$name" "$DOWNLOAD_ROOT/assets/$name"
    fi
  done
  (cd "$DOWNLOAD_ROOT" && sha256sum -c manifests/SHA256SUMS.assets)
}

install_runtime_assets() {
  if [[ -f "$RUNTIME_ROOT/release-manifest.json" ]]; then
    cmp -s "$RUNTIME_ROOT/release-manifest.json" "$DOWNLOAD_ROOT/release-manifest.json" || \
      fail "existing runtime has a different release manifest: $RUNTIME_ROOT"
    note "Runtime data/model/artifacts are already installed."
    return
  fi
  [[ ! -e "$RUNTIME_ROOT" ]] || \
    fail "incomplete runtime directory exists: $RUNTIME_ROOT (move it aside and rerun)"

  local stage
  mkdir -p "$WORK_ROOT/runtime"
  stage="$(mktemp -d "$WORK_ROOT/runtime/.${RELEASE_ID}.XXXXXX")"
  mkdir -p "$stage/data" "$stage/artifacts" "$stage/model/qwen3-vl-4b-$QWEN_REVISION"
  note "Extracting verified datasets..."
  tar --zstd -xf "$DOWNLOAD_ROOT/assets/sort0824-robot-clean.tar.zst" -C "$stage/data"
  tar --zstd -xf "$DOWNLOAD_ROOT/assets/sort0824-human-clean.tar.zst" -C "$stage/data"
  tar --zstd -xf "$DOWNLOAD_ROOT/assets/sort0827-green-grey-tape-ball.tar.zst" -C "$stage/data"
  note "Extracting FAST artifact and pinned Qwen snapshot..."
  tar --zstd -xf "$DOWNLOAD_ROOT/assets/$FAST_NAME.tar.zst" -C "$stage/artifacts"
  tar --zstd -xf "$DOWNLOAD_ROOT/assets/qwen3-vl-4b-$QWEN_REVISION.tar.zst" \
    -C "$stage/model/qwen3-vl-4b-$QWEN_REVISION"
  cp "$DOWNLOAD_ROOT/release-manifest.json" "$stage/release-manifest.json"
  mv "$stage" "$RUNTIME_ROOT"
}

restore_seed_checkpoint() {
  mkdir -p "$CHECKPOINT_ROOT" "$OUTPUT_DIR"
  if find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit | grep -q .; then
    note "A local checkpoint already exists; preserving it and validating the newest one."
  else
    local stage
    stage="$(mktemp -d "$WORK_ROOT/.checkpoint-seed.XXXXXX")"
    mkdir -p "$stage/$CHECKPOINT_NAME" "$stage/run-root"
    note "Downloading exact four-GPU $CHECKPOINT_NAME (about 72 GiB)..."
    gcloud storage rsync --recursive \
      "$GCS_RELEASE_URI/resume/$CHECKPOINT_NAME" "$stage/$CHECKPOINT_NAME"
    (
      cd "$stage/$CHECKPOINT_NAME"
      sha256sum -c "$DOWNLOAD_ROOT/manifests/SHA256SUMS.checkpoint-3000"
    )
    gcloud storage cp "$GCS_RELEASE_URI/resume/run-root/visual_budget.json" \
      "$stage/run-root/visual_budget.json"
    gcloud storage cp "$GCS_RELEASE_URI/resume/run-root/norm_stats.json" \
      "$stage/run-root/norm_stats.json"
    (
      cd "$stage/run-root"
      sha256sum -c "$DOWNLOAD_ROOT/manifests/SHA256SUMS.run-root"
    )
    mv "$stage/$CHECKPOINT_NAME" "$OUTPUT_DIR/$CHECKPOINT_NAME"
    cp "$stage/run-root/visual_budget.json" "$stage/run-root/norm_stats.json" "$OUTPUT_DIR/"
  fi

  PYTHONPATH="$REPO_DIR" "$ENV_PREFIX/bin/python" - <<PY
from qwenvl.train.expert_ema import select_latest_complete_checkpoint
p = select_latest_complete_checkpoint(
    ${OUTPUT_DIR@Q},
    expected_ema_decay=0.999,
    expected_world_size=4,
    require_deepspeed=True,
)
if p is None:
    raise SystemExit("no exact-resume checkpoint found")
print(f"Exact-resume checkpoint validation: PASS ({p})")
PY
}

run_data_preflight() {
  note "Scanning the installed data and proving robot-QA records are disabled..."
  [[ -f "$ARTIFACT_ROOT/$FAST_NAME/norm_stats.json" ]] || fail "FAST artifact is incomplete"
  [[ -f "$MODEL_ROOT/config.json" ]] || fail "pinned Qwen snapshot is incomplete"
  SORT_0824_ROOT="$DATA_ROOT/0824_prompting" \
  SORT_0827_ROOT="$DATA_ROOT/0827_prompting_playdata/data_0827_prompting_playdata" \
  PYTHONPATH="$REPO_DIR" "$ENV_PREFIX/bin/python" \
    "$REPO_DIR/tests/smoke_test_standalone_pick_action_only.py"
}

inside_slurm() {
  export PYTHONPATH="$REPO_DIR"
  export PATH="$ENV_PREFIX/bin:$PATH"
  export SORT_0824_ROOT="$DATA_ROOT/0824_prompting"
  export SORT_0827_ROOT="$DATA_ROOT/0827_prompting_playdata/data_0827_prompting_playdata"
  export SORT_0827_BALL_FAST_DEFAULT="$ARTIFACT_ROOT/$FAST_NAME"
  export MODEL_PATH="$MODEL_ROOT"
  export CUSTOM_CACHE_DIR="$WORK_ROOT/cache/qwen"
  export CHECKPOINT_ROOT
  export TORCHRUN="$ENV_PREFIX/bin/torchrun"
  export WANDB_DIR WANDB_PROJECT
  export WANDB_MODE=online
  [[ -n "$WANDB_ENTITY" ]] && export WANDB_ENTITY
  if [[ -n "$WANDB_RUN_ID" ]]; then
    export WANDB_RUN_ID WANDB_RESUME=allow
  fi
  export NPROC_PER_NODE=4
  export PER_DEVICE_BATCH=1
  export NUM_WORKERS=4
  export SAVE_STEPS=500
  export SAVE_TOTAL_LIMIT=2
  export QWEN_ATTN_IMPL=sdpa
  export TORCH_COMPILE=False
  export PYTHONFAULTHANDLER=1
  export DEEPSPEED_TIMEOUT=120
  export NCCL_NVLS_ENABLE=0
  export INIT_FROM=

  cd "$REPO_DIR"
  "$ENV_PREFIX/bin/python" - <<'PY'
import torch
n = torch.cuda.device_count()
if n != 4:
    raise SystemExit(f"expected exactly 4 allocated GPUs, got {n}")
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    gib = p.total_memory / 1024**3
    if "H100" not in p.name or gib < 75:
        raise SystemExit(f"GPU {i} is not a reviewed 80-GB H100: {p.name}, {gib:.1f} GiB")
    print(f"GPU {i}: {p.name}, {gib:.1f} GiB")
PY
  PYTHONPATH="$REPO_DIR" "$ENV_PREFIX/bin/python" - <<PY
from qwenvl.train.expert_ema import select_latest_complete_checkpoint
p = select_latest_complete_checkpoint(
    ${OUTPUT_DIR@Q}, expected_ema_decay=0.999,
    expected_world_size=4, require_deepspeed=True,
)
if p is None:
    raise SystemExit("no exact-resume checkpoint available")
print(f"Launching exact resume from {p}")
PY
  exec bash "$REPO_DIR/scripts/train_action_expert_4b_sort_0827_ball_dense20_fresh_no_robot_qa_4h100.sh"
}

submit_slurm() {
  need sbatch
  local args job_id
  args=(
    --parsable
    --job-name=qwen-d20-noqa-r3k
    --nodes=1
    --ntasks=1
    --cpus-per-task="$SLURM_CPUS_PER_TASK"
    --mem="$SLURM_MEM"
    --time="$SLURM_TIME"
    --output="$LOG_DIR/%x-%j.out"
    --error="$LOG_DIR/%x-%j.out"
    --export=ALL
  )
  [[ -n "$SLURM_PARTITION" ]] && args+=(--partition="$SLURM_PARTITION")
  [[ -n "$SLURM_ACCOUNT" ]] && args+=(--account="$SLURM_ACCOUNT")
  [[ -n "$SLURM_QOS" ]] && args+=(--qos="$SLURM_QOS")
  local gpu_args=() extra_args=()
  read -r -a gpu_args <<< "$SLURM_GPU_ARGUMENT"
  args+=("${gpu_args[@]}")
  if [[ -n "$SLURM_EXTRA_ARGUMENTS" ]]; then
    read -r -a extra_args <<< "$SLURM_EXTRA_ARGUMENTS"
    args+=("${extra_args[@]}")
  fi
  job_id="$(sbatch "${args[@]}" "$SCRIPT_PATH" --inside-slurm)"
  echo
  echo "Submitted Slurm job: $job_id"
  echo "W&B: entity=${WANDB_ENTITY:-account-default}, project=$WANDB_PROJECT"
  echo "Log: $LOG_DIR/qwen-d20-noqa-r3k-${job_id%%;*}.out"
  echo "The job will exact-resume from trainer step 3000 (or a newer complete local checkpoint)."
}

if [[ "${1:-}" == "--inside-slurm" ]]; then
  inside_slurm
  exit 0
fi

[[ "$GCS_RELEASE_URI" == gs://* && "$GCS_RELEASE_URI" != *YOUR_BUCKET* ]] || \
  fail "replace YOUR_BUCKET in GCS_RELEASE_URI"
ensure_work_root
ensure_gcloud
authenticate_gcs
ensure_python_env
login_wandb
prepare_wandb_run_id
download_release_metadata
download_assets
install_runtime_assets
restore_seed_checkpoint
run_data_preflight
submit_slurm
