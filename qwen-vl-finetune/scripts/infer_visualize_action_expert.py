"""Offline inference + visualization for the Qwen3-VL flow-matching action expert.

For each episode: pick a random timestep t, build the SAME observation prefix the model saw
in training (cam_high history + current wrist still + discretized-state prompt), GENERATE the
subtask on the fly (default; `--teacher_force_subtask` uses the GT label instead), then
produce two action-chunk predictions at t:

  1. action expert  -- integrate the flow ODE (`predict_expert`)
  2. FAST tokens     -- greedily generate `<|action_start|> ... <|action_end|>` and decode

Both predictions are unnormalized + un-delta'd to absolute joint targets and plotted against
the ground-truth chunk `episode.actions[t:t+H]` -- 7 subplots (one per action dim), GT vs
expert vs FAST -- alongside the tiled input images and the predicted vs GT subtask. One PNG
per episode.

The inference core is shared with the realtime server in qwenvl/action_expert/inference.py.

Run (single GPU):
  source /iris/projects/humanoid/miniconda3/bin/activate && conda activate qwen3vl
  python scripts/infer_visualize_action_expert.py            # all 61 episodes -> <ckpt>/offline_viz
  python scripts/infer_visualize_action_expert.py --limit 3  # smoke test
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from qwenvl.action_expert.inference import (
    ACTION_DIM,
    ACTION_HORIZON,
    DEFAULT_CKPT,
    DATA_DIRS,
    FAST_TOK,
    build_dataset,
    generate_subtask,
    load_model,
    make_prompt,
    predict_expert,
    predict_fast,
    templatize,
)


def extract_obs(ds, episode, t):
    """Decode the observation at t once (the expensive video decode). Returns the top visual as
    a LIST of frames (history video, or a single current still if image_history=False)."""
    state = episode["states"][t]
    if ds.data_args.image_history:
        top = ds._extract_frames(episode, t)
    else:
        top = [ds._extract_single_frame(episode["video_path"], t, ds.data_args.max_pixels)]
    wrist = ds._extract_wrist_images(episode, t)
    gt_subtask = ds._subtask_at(episode, t)
    # subtask-input mode: the subtask goes INTO the prompt; predict mode: the question does.
    task_text = None if ds.data_args.predict_subtask else gt_subtask
    prompt = make_prompt(ds, state, task_text)
    return state, top, wrist, gt_subtask, prompt


def gt_chunk(episode, t, horizon):
    gt = episode["actions"][t: t + horizon].astype(np.float32)
    if len(gt) < horizon:
        gt = np.concatenate([gt, np.repeat(gt[-1:], horizon - len(gt), axis=0)], axis=0)
    return gt


def mse(a, b):
    return float(np.mean((a - b) ** 2)) if b is not None else float("nan")


def render(out_path, frames, wrist, gt, expert, fast, meta, stride):
    n_imgs = len(frames) + len(wrist)
    fig = plt.figure(figsize=(22, 12))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.0, 2.5], hspace=0.22)

    gi = gridspec.GridSpecFromSubplotSpec(1, n_imgs, subplot_spec=outer[0], wspace=0.05)
    for j, im in enumerate(frames):
        ax = fig.add_subplot(gi[j])
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(f"cam_high\nt-{(len(frames) - 1 - j) * stride}", fontsize=7)
    for k, im in enumerate(wrist):
        ax = fig.add_subplot(gi[len(frames) + k])
        ax.imshow(im)
        ax.axis("off")
        ax.set_title("wrist(t)", fontsize=8)

    gp = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[1], hspace=0.38, wspace=0.22)
    x = np.arange(gt.shape[0])
    for d in range(gt.shape[1]):
        ax = fig.add_subplot(gp[d // 4, d % 4])
        ax.plot(x, gt[:, d], color="k", lw=2.2, label="ground truth")
        ax.plot(x, expert[:, d], color="tab:blue", lw=1.6, label="action expert")
        if fast is not None:
            ax.plot(x, fast[:, d], color="tab:orange", lw=1.6, ls="--", label="FAST")
        ax.set_title(f"action dim {d}" + (" (gripper)" if d == 6 else ""), fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        if d == 0:
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(meta, fontsize=10, y=0.995)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=None, help="output folder (default: <ckpt>/offline_viz)")
    ap.add_argument("--fast_tok", default=FAST_TOK)
    ap.add_argument("--data_dirs", default=DATA_DIRS,
                    help="dataset dir(s) for norm stats + episodes; MUST match the checkpoint's training data")
    ap.add_argument("--num_flow_steps", type=int, default=10)
    ap.add_argument("--max_fast_new_tokens", type=int, default=96)
    ap.add_argument("--max_subtask_new_tokens", type=int, default=48)
    ap.add_argument("--teacher_force_subtask", action="store_true",
                    help="condition on the GT subtask instead of the model's own generated subtask")
    # These MUST match how the checkpoint was trained.
    ap.add_argument("--no_image_history", action="store_true",
                    help="single current cam_high still instead of the history video")
    ap.add_argument("--subtask_input", action="store_true",
                    help="subtask is the input prompt (predict_subtask=False); no subtask generation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all episodes")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")
    device = "cuda"
    out_dir = Path(args.out) if args.out else Path(args.ckpt) / "offline_viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading dataset (registers FAST tokens, loads norm stats)...")
    ds = build_dataset(args.fast_tok, data_dirs=args.data_dirs, image_history=not args.no_image_history,
                       predict_subtask=not args.subtask_input)
    print(f"  episodes={len(ds)}  vlm_vocab={ds.vlm_vocab_size}  "
          f"fast[start={ds.fast_start_id} end={ds.fast_end_id} base={ds.fast_base_id} V={ds.fast_vocab_size}]")

    print(f"Loading model from {args.ckpt} ...")
    model = load_model(args.ckpt, ds, device)

    im_end_id = ds.tokenizer.convert_tokens_to_ids("<|im_end|>")
    t_rng = random.Random(args.seed)  # reproducible per-episode timestep choice
    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))
    mode = "GT subtask (teacher-forced)" if args.teacher_force_subtask else "generated subtask"
    print(f"Running inference on {n} episodes [{mode}] -> {out_dir}")

    with torch.inference_mode():
        for idx in range(n):
            episode = ds.episodes[idx]
            L = len(episode["states"])
            t = t_rng.randint(0, L - 1)
            name = Path(episode["video_path"]).stem
            try:
                state, top, wrist, gt_subtask, prompt = extract_obs(ds, episode, t)

                # The subtask handling depends on the mode:
                #  - subtask-input (predict_subtask=False): it is already IN the prompt, no
                #    assistant turn, no generation.
                #  - predict-subtask: the model generates it (or teacher-force with GT), and it
                #    becomes the assistant turn both action heads condition on.
                if not ds.data_args.predict_subtask:
                    pred_subtask = gt_subtask  # the input subtask (hard-coded at deploy)
                    mm = templatize(ds, top, wrist, prompt, None, device)
                elif args.teacher_force_subtask:
                    pred_subtask = gt_subtask
                    mm = templatize(ds, top, wrist, prompt, pred_subtask, device)
                else:
                    mm_gen = templatize(ds, top, wrist, prompt, None, device)
                    pred_subtask = generate_subtask(model, mm_gen, ds, im_end_id, args.max_subtask_new_tokens) \
                        or ds.data_args.default_prompt
                    mm = templatize(ds, top, wrist, prompt, pred_subtask, device)

                noise = torch.randn(
                    (1, ACTION_HORIZON, ACTION_DIM), device=device, dtype=torch.float32,
                    generator=torch.Generator(device=device).manual_seed(args.seed + idx),
                )
                expert = predict_expert(model, mm, ds, state, noise, args.num_flow_steps)
                fast, (n_gen, n_valid, exact) = predict_fast(model, mm, ds, state, args.max_fast_new_tokens)
                gt = gt_chunk(episode, t, ACTION_HORIZON)

                match = "match" if pred_subtask.strip() == gt_subtask.strip() else "DIFF"
                meta = (f"ep {idx}  |  {name}  |  t={t}/{L}\n"
                        f"subtask:  pred={pred_subtask!r}   gt={gt_subtask!r}   [{match}]\n"
                        f"MSE(expert)={mse(gt, expert):.4f}   MSE(FAST)={mse(gt, fast):.4f}"
                        f"   [FAST tokens={n_valid}{'' if exact else ' (len approx)'}]")
                render(out_dir / f"episode_{idx:03d}_t{t:04d}.png",
                       top, wrist, gt, expert, fast, meta, ds.stride)
                print(f"[{idx + 1}/{n}] {name} t={t}/{L}  [{match}] "
                      f"pred={pred_subtask!r} gt={gt_subtask!r}  "
                      f"MSE expert={mse(gt, expert):.4f} FAST={mse(gt, fast):.4f}"
                      f"{'' if exact else ' (fast approx len)'}")
            except Exception as e:  # noqa: BLE001
                import traceback
                print(f"[{idx + 1}/{n}] {name} FAILED: {e}")
                traceback.print_exc()

    print(f"Done. PNGs in {out_dir}")


if __name__ == "__main__":
    main()
