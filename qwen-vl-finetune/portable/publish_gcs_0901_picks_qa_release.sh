#!/usr/bin/env bash
# Package and publish the immutable fresh dense20/0901-pick-QA input release.
# Run on the /iris cluster after fitting the exact FAST artifact and authenticating
# gcloud. No checkpoint is included: the target run starts a fresh action policy.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

GCS_RELEASE_URI="${GCS_RELEASE_URI:-gs://YOUR_BUCKET/qwen-sort/dense20-0901-picks-qa-fresh/v1}"
RELEASE_ID="${RELEASE_ID:-dense20-0901-picks-qa-fresh-v1}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/iris/projects/humanoid/trossen_data}"
SOURCE_ARTIFACT_ROOT="${SOURCE_ARTIFACT_ROOT:-/iris/projects/humanoid/ke/Qwen3-VL/checkpoints}"
SOURCE_QWEN_CACHE="${SOURCE_QWEN_CACHE:-/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-/iris/projects/humanoid/miniconda3/envs/qwen3vl/bin/python}"
PUBLISH_STAGE="${PUBLISH_STAGE:-$SOURCE_ARTIFACT_ROOT/release_staging/dense20_0901_picks_qa_${USER:-publisher}}"

QWEN_REVISION="ebb281ec70b05090aa6165b016eac8ec08e71b17"
FAST_NAME="fast_tokenizer_trossen_0824sort_0827ball_0901picks_ee6d"
QWEN_SNAPSHOT="$SOURCE_QWEN_CACHE/snapshots/$QWEN_REVISION"

fail() { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"; }

[[ "$GCS_RELEASE_URI" == gs://* && "$GCS_RELEASE_URI" != *YOUR_BUCKET* ]] || \
  fail "set GCS_RELEASE_URI to a real gs:// bucket prefix"
need gcloud
need tar
need zstd
need sha256sum
[[ -x "$PYTHON_BIN" ]] || fail "missing Python: $PYTHON_BIN"
[[ -d "$QWEN_SNAPSHOT" ]] || fail "missing pinned Qwen snapshot: $QWEN_SNAPSHOT"
[[ -f "$SOURCE_ARTIFACT_ROOT/$FAST_NAME/norm_stats.json" ]] || \
  fail "missing exact FAST artifact; run scripts/fit_sort_0901_picks_fast_tokenizer.sh"

cd "$REPO_DIR"
RELEVANT_PATHS=(
  qwenvl/action_expert/human_prompt.py
  qwenvl/action_expert/modeling_qwen3vl_with_expert.py
  qwenvl/data/robot_data.py
  qwenvl/data/subtask_formats_sort.py
  qwenvl/train/expert_ema.py
  qwenvl/train/train_action_expert.py
  scripts/train_fast_tokenizer.py
  scripts/sort_0827_ball_data.sh
  scripts/sort_0901_picks_data.sh
  scripts/train_action_expert_4b_humanprompt.sh
  scripts/train_action_expert_4b_sort_0901_picks.sh
  scripts/train_action_expert_4b_sort_0901_picks_dense20_fresh_qa_4h100.sh
  portable/bootstrap_and_submit_4h100_0901_picks_qa.sh
  portable/requirements-h100-cu124.txt
  tests/smoke_test_sort_0901_picks_recipe.py
  tests/smoke_test_sort_0901_picks_dense20_loader.py
  tests/smoke_test_standalone_pick_training.py
  tests/test_sort_ball_formats.py
  tests/test_sort_0901_picks_dense20_4h100_recipe.py
  tests/preflight_dense20_prompt_contract_cpu.py
)
git diff --quiet HEAD -- "${RELEVANT_PATHS[@]}" || \
  fail "training/release files differ from HEAD; commit them before publishing"
GIT_COMMIT="$(git rev-parse HEAD)"

SORT_0901_ROOT="$SOURCE_DATA_ROOT" \
  "$PYTHON_BIN" tests/smoke_test_sort_0901_picks_recipe.py
"$PYTHON_BIN" tests/gen_0901_subtask_labels.py
cmp portable/0901ball_train_exclude_episodes.json \
  "$SOURCE_DATA_ROOT/0901ball/meta/train_exclude_episodes.json" || \
  fail "0901ball training exclusion differs from the reviewed portable copy"

"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

p = Path(${SOURCE_ARTIFACT_ROOT@Q}, ${FAST_NAME@Q}, "norm_stats.json")
meta = json.loads(p.read_text()).get("meta", {})
dirs = [x for x in meta.get("robot_data_dirs", "").split(",") if x]
if len(dirs) != 29:
    raise SystemExit(f"FAST provenance has {len(dirs)} roots, expected 29")
expected = {
    str(Path(${SOURCE_DATA_ROOT@Q}, "0901ball")): [
        "episode_000008.mp4", "episode_000011.mp4"
    ]
}
if meta.get("train_exclude_episodes") != expected:
    raise SystemExit(
        f"FAST exclusion provenance mismatch: {meta.get('train_exclude_episodes')!r}"
    )
if meta.get("action_space") != "ee6d" or meta.get("horizon") != 50:
    raise SystemExit(f"FAST representation mismatch: {meta}")
print("FAST provenance: PASS (506 clean physical episodes, two exclusions)")
PY

if gcloud storage ls "$GCS_RELEASE_URI/release-manifest.json" >/dev/null 2>&1; then
  fail "$GCS_RELEASE_URI is already complete; choose a new immutable version"
fi
if [[ -e "$PUBLISH_STAGE" ]] && \
   find "$PUBLISH_STAGE" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  fail "publisher staging directory is not empty: $PUBLISH_STAGE"
fi
mkdir -p "$PUBLISH_STAGE/assets" "$PUBLISH_STAGE/manifests"

robot_0824=(111 112 113 121 122 123 131 132 133 311 312 313 321 322 323 331 332 333 cup green_block grey_pepper_box tape)
robot_0824_paths=()
for name in "${robot_0824[@]}"; do
  path="0824_prompting/$name"
  [[ -d "$SOURCE_DATA_ROOT/$path" ]] || fail "missing dataset: $path"
  robot_0824_paths+=("$path")
done
for path in \
  0824_prompting/human_demo \
  0827_prompting_playdata/data_0827_prompting_playdata/green \
  0827_prompting_playdata/data_0827_prompting_playdata/grey \
  0827_prompting_playdata/data_0827_prompting_playdata/tape \
  0827_prompting_playdata/data_0827_prompting_playdata/ball \
  0901ball 0901green 0901grey; do
  [[ -d "$SOURCE_DATA_ROOT/$path" ]] || fail "missing dataset: $path"
done

ZSTD_CLEVEL=3 tar --zstd -cf "$PUBLISH_STAGE/assets/sort0824-robot-clean.tar.zst" \
  -C "$SOURCE_DATA_ROOT" "${robot_0824_paths[@]}"
ZSTD_CLEVEL=3 tar --zstd -cf "$PUBLISH_STAGE/assets/sort0824-human-clean.tar.zst" \
  -C "$SOURCE_DATA_ROOT" 0824_prompting/human_demo
ZSTD_CLEVEL=3 tar --zstd -cf "$PUBLISH_STAGE/assets/sort0827-green-grey-tape-ball.tar.zst" \
  -C "$SOURCE_DATA_ROOT" \
  0827_prompting_playdata/data_0827_prompting_playdata/green \
  0827_prompting_playdata/data_0827_prompting_playdata/grey \
  0827_prompting_playdata/data_0827_prompting_playdata/tape \
  0827_prompting_playdata/data_0827_prompting_playdata/ball
ZSTD_CLEVEL=3 tar --zstd -cf "$PUBLISH_STAGE/assets/sort0901-picks.tar.zst" \
  -C "$SOURCE_DATA_ROOT" 0901ball 0901green 0901grey
ZSTD_CLEVEL=3 tar --zstd -cf "$PUBLISH_STAGE/assets/$FAST_NAME.tar.zst" \
  -C "$SOURCE_ARTIFACT_ROOT" "$FAST_NAME"
ZSTD_CLEVEL=3 tar --zstd --dereference -cf \
  "$PUBLISH_STAGE/assets/qwen3-vl-4b-$QWEN_REVISION.tar.zst" \
  -C "$QWEN_SNAPSHOT" .

(
  cd "$PUBLISH_STAGE"
  find assets -type f -print0 | sort -z | xargs -0 sha256sum \
    > manifests/SHA256SUMS.assets
)

cat > "$PUBLISH_STAGE/release-manifest.json" <<JSON
{
  "schema_version": 1,
  "release_id": "$RELEASE_ID",
  "git_commit": "$GIT_COMMIT",
  "qwen_revision": "$QWEN_REVISION",
  "initialization": "fresh action policy from Qwen/Qwen3-VL-4B-Instruct",
  "records": {
    "normal_prompted": 371,
    "standalone_action": 135,
    "standalone_robot_qa": 135,
    "total": 641
  },
  "clean_action_episodes": 506,
  "excluded_0901_ball_episodes": [8, 11]
}
JSON

gcloud storage cp --no-clobber "$PUBLISH_STAGE"/assets/*.tar.zst \
  "$GCS_RELEASE_URI/assets/"
gcloud storage cp --no-clobber "$PUBLISH_STAGE"/manifests/* \
  "$GCS_RELEASE_URI/manifests/"
# Completeness marker is uploaded last.
gcloud storage cp --no-clobber "$PUBLISH_STAGE/release-manifest.json" \
  "$GCS_RELEASE_URI/release-manifest.json"

echo "Published immutable input release: $GCS_RELEASE_URI"
echo "Git commit: $GIT_COMMIT"
