# Portable four-H100 dense20/no-standalone-QA resume

This release moves the reviewed four-H100 run to another Slurm cluster. It resumes
the complete `checkpoint-3000` with model, four ZeRO-2 optimizer shards, scheduler,
four RNG states, Trainer state, and EMA. It does **not** use `INIT_FROM`.

The experiment contract is:

- dense human prompts: 30-fps source, stride 3, at most 20 frames;
- four 80-GB H100s, microbatch 1/GPU, accumulation 16, global batch 64;
- 371 normal prompted records + 46 standalone ball-action records;
- no standalone robot-video-QA twins;
- normal subtask language supervision and action/FAST/flow losses remain enabled;
- no robot image history and no past-state history; current robot joint state is present;
- checkpoint every 500 Trainer steps, keep two complete checkpoints.

## 1. Publish once from the current `/iris` cluster

Commit and push the portable code first. Authenticate `gcloud`, edit the GCS URI in
`publish_gcs_release.sh` (or export it), then run:

```bash
cd qwen-vl-finetune
GCS_RELEASE_URI=gs://YOUR_BUCKET/qwen-sort/dense20-noqa-resume3000/v1 \
  bash portable/publish_gcs_release.sh
```

The publisher validates the exact checkpoint and visual contract, packages only the
selected cleaned data, hashes every object, uploads the checkpoint without wrapping it
in a second 72-GiB archive, and writes `release-manifest.json` last.

## 2. Clone and run one script on the target cluster

```bash
git clone YOUR_GITHUB_REPOSITORY_URL Qwen3-VL
cd Qwen3-VL
# The public GCS release needs no Google login or project configuration.
# Edit only WORK_ROOT, W&B key/project, and the clearly marked Slurm values.
bash qwen-vl-finetune/portable/bootstrap_and_submit_4h100_noqa_resume3000.sh
```

The bootstrap performs user-space installs, GCS/W&B authentication, resumable
downloads, SHA-256 verification, atomic extraction, checkpoint validation, a real-data
417-record/no-robot-QA preflight, and `sbatch` submission. Re-running it preserves and
resumes a newer complete local checkpoint if training was preempted. It also persists a
W&B run ID under `WORK_ROOT/wandb/`, so a resubmission appends to the same dashboard run.

## Credentials

Never commit credentials. If `WANDB_API_KEY` is unset, the bootstrap asks for it with a
private terminal prompt and removes it from the environment before Slurm submission.
You can also export it before running:

```bash
export WANDB_API_KEY='...'
export WANDB_ENTITY='your-user-or-team'   # optional
export WANDB_PROJECT='qwen-dense20-noqa'
```

The default `GCS_PUBLIC_READ=True` uses an isolated anonymous Cloud SDK configuration,
so the public release requires neither `gcloud init` nor Google login. For a future
private mirror, set `GCS_PUBLIC_READ=False` and either authenticate interactively or
point `GOOGLE_APPLICATION_CREDENTIALS` at a read-only service-account JSON outside the
git checkout. The bootstrap unsets the W&B key before Slurm submission after W&B stores
the login in the user's home credentials.
