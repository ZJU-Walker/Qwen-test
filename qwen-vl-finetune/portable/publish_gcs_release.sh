#!/usr/bin/env bash
# Build and publish the immutable dense20/no-standalone-robot-QA release.
# Run this ON THE CURRENT /iris CLUSTER after authenticating gcloud.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ==============================================================================
# EDIT THIS BLOCK ONCE. Do not put credentials in this file.
# ==============================================================================
GCS_RELEASE_URI="${GCS_RELEASE_URI:-gs://YOUR_BUCKET/qwen-sort/dense20-noqa-resume3000/v1}"
RELEASE_ID="${RELEASE_ID:-dense20-noqa-resume3000-v1}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/iris/projects/humanoid/trossen_data}"
SOURCE_ARTIFACT_ROOT="${SOURCE_ARTIFACT_ROOT:-/iris/projects/humanoid/ke/Qwen3-VL/checkpoints}"
SOURCE_QWEN_CACHE="${SOURCE_QWEN_CACHE:-/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct}"
SOURCE_CHECKPOINT_ROOT="${SOURCE_CHECKPOINT_ROOT:-/iris/u/kewalk/qwen_checkpoints}"
PYTHON_BIN="${PYTHON_BIN:-/iris/projects/humanoid/miniconda3/envs/qwen3vl/bin/python}"
PUBLISH_STAGE="${PUBLISH_STAGE:-/tmp/qwen_dense20_noqa_resume3000_release_${USER:-publisher}}"
# ==============================================================================

QWEN_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
RUN_NAME="qwen3_4b_ae_humanprompt_0824sort_0827ballaction_dense20s3_fresh_norobotqa_4h100_ee6d_rtc20_subattend_L18_vis46_constlr"
CHECKPOINT_NAME="checkpoint-3000"
FAST_NAME="fast_tokenizer_trossen_0824sort_0827ball_ee6d"
RUN_ROOT="$SOURCE_CHECKPOINT_ROOT/$RUN_NAME"
CHECKPOINT_DIR="$RUN_ROOT/$CHECKPOINT_NAME"
QWEN_SNAPSHOT="$SOURCE_QWEN_CACHE/snapshots/$QWEN_REVISION"

fail() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"; }

[[ "$GCS_RELEASE_URI" == gs://* ]] || fail "set GCS_RELEASE_URI to a gs:// bucket prefix"
[[ "$GCS_RELEASE_URI" != *YOUR_BUCKET* ]] || fail "replace YOUR_BUCKET in GCS_RELEASE_URI"
need gcloud
need tar
need zstd
need sha256sum
need "$PYTHON_BIN"

cd "$REPO_DIR"
RELEVANT_PATHS=(
  qwenvl/action_expert/human_prompt.py
  qwenvl/action_expert/modeling_qwen3vl_with_expert.py
  qwenvl/data/robot_data.py
  qwenvl/data/subtask_formats_sort.py
  qwenvl/train/expert_ema.py
  qwenvl/train/train_action_expert.py
  scripts/sort_0827_ball_data.sh
  scripts/train_action_expert_4b_humanprompt.sh
  scripts/train_action_expert_4b_sort_0827_ball.sh
  scripts/train_action_expert_4b_sort_0827_ball_dense20_fresh_no_robot_qa_4h100.sh
  portable
)
git diff --quiet HEAD -- "${RELEVANT_PATHS[@]}" || \
  fail "portable training files differ from HEAD; commit them before publishing"
GIT_COMMIT="$(git rev-parse HEAD)"

[[ -d "$CHECKPOINT_DIR" ]] || fail "missing checkpoint: $CHECKPOINT_DIR"
[[ -d "$QWEN_SNAPSHOT" ]] || fail "missing pinned Qwen snapshot: $QWEN_SNAPSHOT"
[[ -d "$SOURCE_ARTIFACT_ROOT/$FAST_NAME" ]] || fail "missing FAST artifact"
[[ -f "$RUN_ROOT/visual_budget.json" && -f "$RUN_ROOT/norm_stats.json" ]] || \
  fail "missing run-root visual_budget.json or norm_stats.json"

PYTHONPATH="$REPO_DIR" "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
from qwenvl.train.expert_ema import validate_action_expert_checkpoint

checkpoint = Path(${CHECKPOINT_DIR@Q})
validate_action_expert_checkpoint(
    checkpoint,
    expected_ema_decay=0.999,
    expected_world_size=4,
    require_deepspeed=True,
)
state = json.loads((checkpoint / "trainer_state.json").read_text())
if state.get("global_step") != 3000:
    raise SystemExit(f"expected trainer step 3000, got {state.get('global_step')}")
budget = json.loads(Path(${RUN_ROOT@Q}, "visual_budget.json").read_text())
expected = {
    "human_prompt_stride": 3,
    "human_prompt_max_frames": 20,
    "video_max_pixels": 4_600_000,
    "standalone_robot_qa_enabled": False,
    "image_history": False,
    "state_history": False,
    "expert_attends_subtask": True,
    "lm_loss_per_sample": True,
}
bad = {k: (budget.get(k), v) for k, v in expected.items() if budget.get(k) != v}
if bad:
    raise SystemExit(f"visual contract mismatch: {bad}")
print("Checkpoint and visual contract validation: PASS")
PY

if gcloud storage ls "$GCS_RELEASE_URI/release-manifest.json" >/dev/null 2>&1; then
  fail "$GCS_RELEASE_URI is already complete; choose a new immutable version prefix"
fi

if [[ -e "$PUBLISH_STAGE" ]] && find "$PUBLISH_STAGE" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  fail "publisher staging directory is not empty: $PUBLISH_STAGE (choose a new PUBLISH_STAGE)"
fi
mkdir -p "$PUBLISH_STAGE/assets" "$PUBLISH_STAGE/manifests"

robot_0824=(111 112 113 121 122 123 131 132 133 311 312 313 321 322 323 331 332 333 cup green_block grey_pepper_box tape)
robot_0824_paths=()
for name in "${robot_0824[@]}"; do
  path="0824_prompting/$name"
  [[ -d "$SOURCE_DATA_ROOT/$path" ]] || fail "missing dataset root: $SOURCE_DATA_ROOT/$path"
  robot_0824_paths+=("$path")
done

for path in \
  0824_prompting/human_demo \
  0827_prompting_playdata/data_0827_prompting_playdata/green \
  0827_prompting_playdata/data_0827_prompting_playdata/grey \
  0827_prompting_playdata/data_0827_prompting_playdata/tape \
  0827_prompting_playdata/data_0827_prompting_playdata/ball; do
  [[ -d "$SOURCE_DATA_ROOT/$path" ]] || fail "missing dataset root: $SOURCE_DATA_ROOT/$path"
done

echo "Packaging cleaned 0824 robot roots..."
ZSTD_CLEVEL=3 tar --zstd -cf "$PUBLISH_STAGE/assets/sort0824-robot-clean.tar.zst" \
  -C "$SOURCE_DATA_ROOT" "${robot_0824_paths[@]}"

echo "Packaging cleaned 0824 human prompt pool..."
ZSTD_CLEVEL=3 tar --zstd -cf "$PUBLISH_STAGE/assets/sort0824-human-clean.tar.zst" \
  -C "$SOURCE_DATA_ROOT" 0824_prompting/human_demo

echo "Packaging selected 0827 roots..."
ZSTD_CLEVEL=3 tar --zstd -cf "$PUBLISH_STAGE/assets/sort0827-green-grey-tape-ball.tar.zst" \
  -C "$SOURCE_DATA_ROOT" \
  0827_prompting_playdata/data_0827_prompting_playdata/green \
  0827_prompting_playdata/data_0827_prompting_playdata/grey \
  0827_prompting_playdata/data_0827_prompting_playdata/tape \
  0827_prompting_playdata/data_0827_prompting_playdata/ball

echo "Packaging FAST tokenizer and normalization artifact..."
ZSTD_CLEVEL=3 tar --zstd -cf "$PUBLISH_STAGE/assets/$FAST_NAME.tar.zst" \
  -C "$SOURCE_ARTIFACT_ROOT" "$FAST_NAME"

echo "Packaging pinned Qwen base snapshot (dereferencing HF-cache symlinks)..."
ZSTD_CLEVEL=3 tar --zstd --dereference -cf "$PUBLISH_STAGE/assets/qwen3-vl-4b-$QWEN_REVISION.tar.zst" \
  -C "$QWEN_SNAPSHOT" .

echo "Hashing release assets..."
(
  cd "$PUBLISH_STAGE"
  find assets -type f -print0 | sort -z | xargs -0 sha256sum > manifests/SHA256SUMS.assets
)

echo "Hashing the complete checkpoint (about 72 GiB; this can take several minutes)..."
(
  cd "$CHECKPOINT_DIR"
  find . -type f -print0 | sort -z | xargs -0 sha256sum > "$PUBLISH_STAGE/manifests/SHA256SUMS.checkpoint-3000"
)
(
  cd "$RUN_ROOT"
  sha256sum visual_budget.json norm_stats.json > "$PUBLISH_STAGE/manifests/SHA256SUMS.run-root"
)

cat > "$PUBLISH_STAGE/release-manifest.json" <<JSON
{
  "schema_version": 1,
  "release_id": "$RELEASE_ID",
  "git_commit": "$GIT_COMMIT",
  "qwen_revision": "$QWEN_REVISION",
  "run_name": "$RUN_NAME",
  "checkpoint_name": "$CHECKPOINT_NAME",
  "trainer_step": 3000,
  "deepspeed_world_size": 4,
  "ema_decay": 0.999,
  "records": {"normal_prompted": 371, "standalone_ball_action": 46, "standalone_robot_qa": 0, "total": 417}
}
JSON

echo "Uploading large assets with resumable GCS transfers..."
gcloud storage cp --no-clobber "$PUBLISH_STAGE"/assets/*.tar.zst "$GCS_RELEASE_URI/assets/"
gcloud storage cp --recursive --no-clobber "$CHECKPOINT_DIR" "$GCS_RELEASE_URI/resume/"
gcloud storage cp --no-clobber "$RUN_ROOT/visual_budget.json" "$RUN_ROOT/norm_stats.json" \
  "$GCS_RELEASE_URI/resume/run-root/"
gcloud storage cp --no-clobber "$PUBLISH_STAGE"/manifests/* "$GCS_RELEASE_URI/manifests/"

# Upload the release marker LAST. Its presence means every preceding object was sent.
gcloud storage cp --no-clobber "$PUBLISH_STAGE/release-manifest.json" \
  "$GCS_RELEASE_URI/release-manifest.json"

echo
echo "Published immutable release: $GCS_RELEASE_URI"
echo "Git commit: $GIT_COMMIT"
echo "Checkpoint: $CHECKPOINT_NAME (exact four-GPU resume at trainer step 3000)"
