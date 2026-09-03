# Run dense20 + 0901 pickup QA on a new four-H100 cluster

This is the complete workflow for the new training mix. It keeps the previous 0824 +
0827 experiment unchanged and adds the standalone pickup datasets `0901ball`,
`0901green`, and `0901grey`.

## Exact training contract

- Four 80-GB H100 GPUs.
- Fresh action policy from `Qwen/Qwen3-VL-4B-Instruct`; no old policy checkpoint is
  imported. A later relaunch of this same run exact-resumes its own newest checkpoint.
- Dense human prompts for the ordinary sort data: 30-fps source, stride 3, at most 20
  frames, endpoint preserving, explicit timestamps, 4.6M whole-video pixel budget.
- No robot image history and no past-state history.
- 371 complete prompted sort episodes with the existing six QA formats, FAST loss, and
  flow-matching loss.
- 135 standalone pickup action records: 46 old 0827 ball + 28 clean 0901 ball + 30
  green block + 31 grey box. They use oracle `pick <object>` context and train FAST +
  flow, with no fabricated destination.
- 135 paired robot visual-QA records. They ask only destination-free questions:
  which object is available, what pickup action should happen next, which object was
  picked, and which pickup skill was demonstrated. These records train text CE only.
- The two failed 0901 ball recordings, episodes 8 and 11, remain on disk for audit but
  are excluded by `meta/train_exclude_episodes.json` from both FAST fitting and model
  training.
- Total: 506 clean action-bearing episodes and 641 sampling records.

## A. Prepare and publish the input release on the `/iris` cluster

From the repository used to curate the data:

```bash
cd /iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune

/iris/projects/humanoid/miniconda3/envs/qwen3vl/bin/python \
  tests/gen_0901_subtask_labels.py

/iris/projects/humanoid/miniconda3/envs/qwen3vl/bin/python \
  tests/smoke_test_sort_0901_picks_recipe.py
```

The label command above is a read-only dry run. It must report the two expected ball
QC flags and pass all parquet/video/label checks.

Fit the new FAST tokenizer and normalization artifact once:

```bash
cd /iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune

PYTHON_BIN=/iris/projects/humanoid/miniconda3/envs/qwen3vl/bin/python \
  bash scripts/fit_sort_0901_picks_fast_tokenizer.sh
```

Confirm the artifact and recipe:

```bash
test -f \
  /iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0824sort_0827ball_0901picks_ee6d/norm_stats.json

python3 tests/test_sort_0901_picks_dense20_4h100_recipe.py
```

Commit and push the reviewed training files before publishing. The publisher records
the exact Git commit and refuses dirty training files:

```bash
git status --short
git add \
  qwenvl/data/robot_data.py \
  qwenvl/data/subtask_formats_sort.py \
  scripts/sort_0901_picks_data.sh \
  scripts/train_fast_tokenizer.py \
  scripts/fit_sort_0901_picks_fast_tokenizer.sh \
  scripts/train_action_expert_4b_sort_0901_picks.sh \
  scripts/train_action_expert_4b_sort_0901_picks_dense20_fresh_qa_4h100.sh \
  tests/gen_0901_subtask_labels.py \
  tests/smoke_test_sort_0901_picks_recipe.py \
  tests/smoke_test_sort_0901_picks_dense20_loader.py \
  tests/test_sort_ball_formats.py \
  tests/smoke_test_standalone_pick_training.py \
  tests/test_sort_0901_picks_dense20_4h100_recipe.py \
  tests/preflight_dense20_prompt_contract_cpu.py \
  portable/0901ball_train_exclude_episodes.json \
  portable/bootstrap_and_submit_4h100_0901_picks_qa.sh \
  portable/publish_gcs_0901_picks_qa_release.sh \
  portable/RUN_0901_PICKS_QA_ON_NEW_4H100_CLUSTER.md

git commit -m 'Add dense20 QA training for 0901 standalone picks'
git push
```

Authenticate Google Cloud, select a new immutable release URI, then package and upload
the data, FAST artifact, and pinned Qwen base model:

```bash
gcloud auth login

export GCS_RELEASE_URI='gs://YOUR_BUCKET/qwen-sort/dense20-0901-picks-qa-fresh/v1'

# Optional: choose another high-capacity staging directory. Do not use a small /tmp.
export PUBLISH_STAGE='/iris/u/kewalk/qwen_release_staging/dense20_0901_picks_qa_v1'

bash portable/publish_gcs_0901_picks_qa_release.sh
```

Do not reuse a URI whose `release-manifest.json` already exists. The publisher uploads
that file last, so its presence means the release is complete.

## B. Launch on the new four-H100 cluster

Choose shared storage visible from both login and compute nodes. Keep at least 340 GiB
capacity and 150 GiB free:

```bash
cd /PATH/TO/SHARED/STORAGE
df -h .
```

Clone the branch/commit printed by the publisher:

```bash
git clone --branch YOUR_BRANCH_WITH_0901_RECIPE \
  git@github.com:ZJU-Walker/Qwen-test.git

cd Qwen-test/qwen-vl-finetune
test -x portable/bootstrap_and_submit_4h100_0901_picks_qa.sh
```

Set runtime storage and the exact release URI:

```bash
export WORK_ROOT="$PWD/portable_work_0901_picks_qa"
mkdir -p "$WORK_ROOT"

export GCS_RELEASE_URI='gs://YOUR_BUCKET/qwen-sort/dense20-0901-picks-qa-fresh/v1'
```

Set the Slurm values for the cluster:

```bash
export SLURM_PARTITION='YOUR_GPU_PARTITION'
export SLURM_ACCOUNT='YOUR_SLURM_ACCOUNT'
export SLURM_GPU_ARGUMENT='--gres=gpu:h100:4'

export SLURM_CPUS_PER_TASK='32'
export SLURM_MEM='700G'
export SLURM_TIME='3-00:00:00'
```

If the cluster uses `--gpus=4`, use this instead:

```bash
export SLURM_GPU_ARGUMENT='--gpus=4'
```

Configure W&B without placing the key in shell history:

```bash
read -rsp 'W&B API key: ' WANDB_API_KEY
echo
export WANDB_API_KEY
export WANDB_PROJECT='qwen-dense20-0901-picks-qa'
# export WANDB_ENTITY='YOUR_USER_OR_TEAM'
```

Download, validate, and submit with one command:

```bash
bash portable/bootstrap_and_submit_4h100_0901_picks_qa.sh
```

The bootstrap installs its environment under `WORK_ROOT`, verifies all SHA-256 hashes,
checks the 641-record data contract, authenticates W&B, and submits the four-H100 job.
It does not download or warm-start an old action-policy checkpoint.

## C. Monitor and restart

```bash
squeue --me
tail -f "$WORK_ROOT/logs/qwen-d20-0901-pickqa-JOB_ID.out"
```

Healthy startup reports four 80-GB H100s, 371 normal prompted records, 135 standalone
action records, 135 standalone robot-QA records, and then finite training steps.

After preemption, export the same settings and run the same bootstrap again. It keeps
the W&B run ID and exact-resumes only a complete checkpoint from this same 0901 run:

```bash
cd /PATH/TO/Qwen-test/qwen-vl-finetune
export WORK_ROOT="$PWD/portable_work_0901_picks_qa"
export GCS_RELEASE_URI='gs://YOUR_BUCKET/qwen-sort/dense20-0901-picks-qa-fresh/v1'
export SLURM_PARTITION='YOUR_GPU_PARTITION'
export SLURM_ACCOUNT='YOUR_SLURM_ACCOUNT'
export SLURM_GPU_ARGUMENT='--gres=gpu:h100:4'
export WANDB_PROJECT='qwen-dense20-0901-picks-qa'

read -rsp 'W&B API key: ' WANDB_API_KEY
echo
export WANDB_API_KEY

bash portable/bootstrap_and_submit_4h100_0901_picks_qa.sh
```

## Direct `/iris` launch, if the four H100s are on the current cluster

After the new FAST artifact exists and the allocation exposes exactly four H100s:

```bash
cd /iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune

export NPROC_PER_NODE=4
export PER_DEVICE_BATCH=1
export NUM_WORKERS=4
export QWEN_ATTN_IMPL=sdpa
export NCCL_NVLS_ENABLE=0
export DEEPSPEED_TIMEOUT=120
export WANDB_DIR=/tmp/qwen_wandb_0901_picks_qa

bash scripts/train_action_expert_4b_sort_0901_picks_dense20_fresh_qa_4h100.sh
```
