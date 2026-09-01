# Run dense20/no-robot-QA training on a new four-H100 cluster

This guide downloads the public release, restores the exact four-GPU
`checkpoint-3000`, verifies the data and checkpoint, logs in to W&B, and submits the
training job through Slurm.

Training contract:

- four 80-GB H100 GPUs;
- dense human prompt video, stride 3, at most 20 frames;
- 371 normal prompted records and 46 standalone ball-action records;
- no standalone robot-video-QA records;
- normal subtask language supervision remains enabled;
- action FAST-token and flow-matching losses remain enabled;
- no robot image history and no past-state history;
- exact DeepSpeed/optimizer/scheduler/EMA resume from Trainer step 3000.

## 1. Choose a location and check storage

Move to a persistent filesystem that is visible from both the login node and Slurm
compute nodes. It needs at least 340 GiB available; 500 GiB or more is preferred.

```bash
cd /PATH/TO/YOUR/SHARED/STORAGE
df -h .
```

Do not use `/tmp`. Avoid a small home directory quota.

## 2. Clone the reviewed branch

Using GitHub SSH:

```bash
git clone \
  --branch codex/portable-4h100-noqa-resume3000 \
  --single-branch \
  git@github.com:ZJU-Walker/Qwen-test.git

cd Qwen-test/qwen-vl-finetune
```

If GitHub SSH is not configured, use HTTPS instead:

```bash
git clone \
  --branch codex/portable-4h100-noqa-resume3000 \
  --single-branch \
  https://github.com/ZJU-Walker/Qwen-test.git

cd Qwen-test/qwen-vl-finetune
```

Confirm the portable bootstrap is present:

```bash
test -x portable/bootstrap_and_submit_4h100_noqa_resume3000.sh
git rev-parse --short HEAD
```

The branch should be at commit `de0063e` or a later compatible bootstrap commit.

## 3. Select the working directory

The following keeps all downloaded data, model files, environments, checkpoints, logs,
and W&B state under the cloned repository filesystem:

```bash
export WORK_ROOT="$PWD/portable_work"
mkdir -p "$WORK_ROOT"
df -h "$WORK_ROOT"
```

`WORK_ROOT` is not another repository. It is only the runtime-storage directory. Do not
commit it to GitHub. Keep it after training starts because it contains the checkpoints
needed for restart.

Optionally hide it from local `git status` without changing committed files:

```bash
printf '/qwen-vl-finetune/portable_work/\n' >> ../.git/info/exclude
```

If the repository filesystem does not have enough space, use a separate shared path:

```bash
export WORK_ROOT="/YOUR/LARGE/SHARED/PATH/$USER/qwen_dense20_noqa_resume3000"
mkdir -p "$WORK_ROOT"
df -h "$WORK_ROOT"
```

## 4. Find the cluster's Slurm values

These read-only commands often show the available GPU partitions and your account:

```bash
sinfo -o '%P %G %l %c %m'
sacctmgr show assoc user="$USER" format=Account,Partition,QOS
```

If `sacctmgr` is unavailable, use the account and partition provided by the cluster
documentation or administrator.

## 5. Export the Slurm configuration

Replace the two placeholder values:

```bash
export SLURM_PARTITION='YOUR_GPU_PARTITION'
export SLURM_ACCOUNT='YOUR_SLURM_ACCOUNT'
export SLURM_GPU_ARGUMENT='--gres=gpu:h100:4'
```

If the cluster supplies a default partition or account, omit the corresponding export.

The reviewed defaults are 32 CPUs, 700 GiB host RAM, and three days:

```bash
export SLURM_CPUS_PER_TASK='32'
export SLURM_MEM='700G'
export SLURM_TIME='3-00:00:00'
```

Optional examples—only set values required by the new cluster:

```bash
# export SLURM_QOS='YOUR_QOS'
# export SLURM_EXTRA_ARGUMENTS='--constraint=h100 --exclusive'
```

If the site uses `--gpus=4` instead of GRES, use:

```bash
export SLURM_GPU_ARGUMENT='--gpus=4'
```

## 6. Configure W&B securely

Use a current W&B API key. The command below does not display the key or save it in
shell history:

```bash
read -rsp 'W&B API key: ' WANDB_API_KEY
echo
export WANDB_API_KEY
```

Choose the W&B project and optionally the account/team entity:

```bash
export WANDB_PROJECT='qwen-dense20-noqa'
# export WANDB_ENTITY='YOUR_WANDB_USER_OR_TEAM'
```

Do not put the W&B key in this README, the bootstrap script, or Git.

## 7. Launch the one-command bootstrap

No Google Cloud login, project configuration, or service-account key is required. The
release is public and anonymous download mode is enabled by default.

```bash
bash portable/bootstrap_and_submit_4h100_noqa_resume3000.sh
```

The first run will:

1. Install Google Cloud CLI and the pinned Python/CUDA environment under `WORK_ROOT`.
2. Download the 89.5-GB public release anonymously.
3. Verify every archive and checkpoint file using SHA-256.
4. Extract the selected datasets, FAST artifact, and pinned Qwen model.
5. Restore and validate the complete four-GPU `checkpoint-3000`.
6. Prove the installed loader has 417 records and no standalone robot-video-QA records.
7. Authenticate W&B and create a persistent W&B run ID.
8. Submit the four-H100 Slurm job.

Downloads, installation, and checksum verification can take significant time. It is
safe to rerun the same command after an interrupted download.

## 8. Monitor the submitted job

List your jobs:

```bash
squeue --me
```

The bootstrap prints the exact log path after submission. It has this form:

```bash
tail -f "$WORK_ROOT/logs/qwen-d20-noqa-r3k-JOB_ID.out"
```

Replace `JOB_ID` with the number printed by `sbatch`.

Useful GPU check from an allocated node:

```bash
nvidia-smi
```

Healthy startup should report four H100 GPUs, exact checkpoint resume, dataset
preflight success, and then Trainer steps greater than 3000.

## 9. Restart after preemption

Return to the same clone, keep the same `WORK_ROOT`, export the same Slurm and W&B
values, and run the same bootstrap again:

```bash
cd /PATH/TO/Qwen-test/qwen-vl-finetune
export WORK_ROOT="$PWD/portable_work"

export SLURM_PARTITION='YOUR_GPU_PARTITION'
export SLURM_ACCOUNT='YOUR_SLURM_ACCOUNT'
export SLURM_GPU_ARGUMENT='--gres=gpu:h100:4'

read -rsp 'W&B API key: ' WANDB_API_KEY
echo
export WANDB_API_KEY
export WANDB_PROJECT='qwen-dense20-noqa'

bash portable/bootstrap_and_submit_4h100_noqa_resume3000.sh
```

The script preserves the W&B run ID and selects the newest complete local checkpoint.
It fails closed instead of resuming a partial checkpoint.

## Minimal command checklist

After replacing the Slurm placeholders, this is the complete minimal sequence:

```bash
cd /PATH/TO/YOUR/SHARED/STORAGE

git clone \
  --branch codex/portable-4h100-noqa-resume3000 \
  --single-branch \
  git@github.com:ZJU-Walker/Qwen-test.git

cd Qwen-test/qwen-vl-finetune

export WORK_ROOT="$PWD/portable_work"
export SLURM_PARTITION='YOUR_GPU_PARTITION'
export SLURM_ACCOUNT='YOUR_SLURM_ACCOUNT'
export SLURM_GPU_ARGUMENT='--gres=gpu:h100:4'
export WANDB_PROJECT='qwen-dense20-noqa'

read -rsp 'W&B API key: ' WANDB_API_KEY
echo
export WANDB_API_KEY

bash portable/bootstrap_and_submit_4h100_noqa_resume3000.sh
```
