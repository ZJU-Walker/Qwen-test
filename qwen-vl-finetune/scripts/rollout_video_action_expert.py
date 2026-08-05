"""Sliding-window rollout VIDEO: watch the policy's predictions track (or drift from) the
ground truth across a whole episode.

For every Nth timestep of an episode this runs the action expert on the SAME observation the
model gets at deploy time (cam_high history + wrist still + state prompt) and renders one video
frame containing:

  * the 10-frame cam_high history + the wrist still  -- exactly what the model sees
  * 7 per-dimension panels: the FULL-episode ground-truth joint trajectory in grey, with the
    50-step GT chunk (black) and the model's predicted chunk (blue) overlaid at the current t
  * an MSE-vs-time panel that accumulates as the episode plays, so you can SEE which part of
    the episode the policy falls apart in

Output: one MP4 per episode. Because the prediction is drawn on the full trajectory, a policy
that is merely noisy looks different from one that is confidently wrong in a specific phase
(e.g. only during the grasp, or only after the block is picked).

This is open-loop: every prediction starts from the GROUND-TRUTH state at t, so errors do not
compound the way they do on the robot. It shows whether the policy knows what to do from a
good state -- not whether it can recover from its own mistakes.

Run (needs a GPU):
  conda activate qwen3vl
  cd /iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune
  python scripts/rollout_video_action_expert.py \
      --ckpt <...>/qwen3_4b_ae_hist_subpred_0717merged_rtc20_subinsul_L18_vis16/checkpoint-15000 \
      --insulated_subtask \
      --fast_tok <...>/checkpoints/fast_tokenizer_trossen_0717merged \
      --data_dirs /iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged \
      --episodes 0,1,2 --t_stride 5
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

import imageio.v2 as imageio
from torchvision.io import read_video

from qwenvl.action_expert.inference import (
    ACTION_DIM, ACTION_HORIZON, DATA_DIRS, DEFAULT_CKPT, FAST_TOK,
    build_dataset, generate_subtask, load_model, load_norm_stats_path, load_visual_budget, make_prompt,
    predict_expert, predict_fast, subtask_token_mask_for, templatize,
)


def decode_whole_video(path):
    """Decode an entire episode video once into PIL frames.

    The dataset decodes a fresh window per __getitem__, which is right for training but
    wasteful here: a sliding window over every timestep would re-decode the same frames
    hundreds of times. One full decode + indexing is ~100x cheaper for this use.
    """
    video, _, _ = read_video(str(path), pts_unit="sec", output_format="THWC")
    return [Image.fromarray(f.numpy()) for f in video]


def resize_to_budget(img, budget):
    """Same pre-resize the dataset applies to wrist stills (robot_data._extract_single_frame)."""
    w, h = img.size
    if w * h > budget:
        s = (budget / (w * h)) ** 0.5
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return img


def history_at(frames, t, num_frames, stride):
    """Replicates robot_data._extract_frames: num_frames frames ending at t, spaced by
    stride, oldest first, clamped at the episode start."""
    n = len(frames)
    return [frames[min(max(t - i * stride, 0), n - 1)] for i in reversed(range(num_frames))]


def gt_chunk(actions, t, horizon):
    """The horizon-step ground-truth action chunk starting at t, edge-padded at the end."""
    gt = actions[t: t + horizon].astype(np.float32)
    if len(gt) < horizon:
        gt = np.concatenate([gt, np.repeat(gt[-1:], horizon - len(gt), axis=0)], axis=0)
    return gt


def render_frame(fig, hist, wrist, ep_actions, t, gt, pred, fast, mse_ts, fast_ts, meta, stride, dpi):
    """Draw one video frame and return it as an RGB array. `fast`/`fast_ts` may be None."""
    fig.clear()
    # height_ratios tuned so the 16:9 image strip does not leave a band of dead space:
    # 11 images across a 20in figure are ~1.7in wide => ~1in tall.
    outer = gridspec.GridSpec(2, 1, height_ratios=[0.5, 2.6], hspace=0.10, figure=fig)

    # --- top strip: exactly the images the model consumed ---
    gi = gridspec.GridSpecFromSubplotSpec(1, len(hist) + 1, subplot_spec=outer[0], wspace=0.04)
    for j, im in enumerate(hist):
        ax = fig.add_subplot(gi[j])
        ax.imshow(im)
        ax.axis("off")
        ax.set_title(f"t-{(len(hist) - 1 - j) * stride}", fontsize=7)
    ax = fig.add_subplot(gi[len(hist)])
    ax.imshow(wrist)
    ax.axis("off")
    ax.set_title("wrist(t)", fontsize=8)

    # --- bottom: 7 action dims + the running-MSE panel ---
    gp = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[1], hspace=0.42, wspace=0.24)
    horizon = gt.shape[0]
    x_chunk = np.arange(t, t + horizon)
    for d in range(gt.shape[1]):
        ax = fig.add_subplot(gp[d // 4, d % 4])
        # full-episode GT in grey = the global context that makes drift visible
        ax.plot(np.arange(len(ep_actions)), ep_actions[:, d], color="0.75", lw=1.0)
        ax.plot(x_chunk, gt[:, d], color="k", lw=2.0, label="GT chunk")
        ax.plot(x_chunk, pred[:, d], color="tab:blue", lw=1.7, label="action expert")
        if fast is not None:
            ax.plot(x_chunk, fast[:, d], color="tab:orange", lw=1.5, ls="--", label="FAST")
        ax.axvline(t, color="tab:red", lw=0.9, ls=":")
        ax.set_title(f"dim {d}" + (" (gripper)" if d == 6 else ""), fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        if d == 0:
            ax.legend(fontsize=7, loc="best")

    ax = fig.add_subplot(gp[1, 3])
    if mse_ts:
        ts, vals = zip(*mse_ts)
        ax.plot(ts, vals, color="tab:blue", lw=1.4, label="expert")
        ax.scatter([ts[-1]], [vals[-1]], color="tab:red", s=18, zorder=3)
    if fast_ts:
        fts, fvals = zip(*fast_ts)
        ax.plot(fts, fvals, color="tab:orange", lw=1.3, ls="--", label="FAST")
        ax.legend(fontsize=7, loc="best")
    ax.set_xlim(0, len(ep_actions))
    ax.set_title("chunk MSE vs episode time", fontsize=9)
    ax.set_xlabel("timestep", fontsize=7)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)

    fig.suptitle(meta, fontsize=10, y=0.995)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=None, help="output folder (default: <ckpt>/rollout_video)")
    ap.add_argument("--fast_tok", default=FAST_TOK)
    ap.add_argument("--data_dirs", default=DATA_DIRS,
                    help="MUST match the checkpoint's training data (norm stats live there)")
    ap.add_argument("--episodes", default="0", help="comma-separated indices, or 'all'")
    ap.add_argument("--t_stride", type=int, default=5,
                    help="run inference every Nth timestep (1 = every step; 5 is a good default)")
    ap.add_argument("--fps", type=int, default=6, help="output video framerate")
    ap.add_argument("--num_flow_steps", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=90, help="lower = smaller/faster video")
    ap.add_argument("--seed", type=int, default=0)
    # These MUST match how the checkpoint was trained.
    ap.add_argument("--no_image_history", action="store_true")
    ap.add_argument("--subtask_input", action="store_true")
    ap.add_argument("--insulated_subtask", action="store_true",
                    help="checkpoint trained with --expert_attends_subtask False")
    ap.add_argument("--show_subtask", action="store_true",
                    help="also decode the subtask each frame for display (slower; the expert "
                         "ignores it when --insulated_subtask)")
    ap.add_argument("--fast", action="store_true",
                    help="ALSO plot the FAST-token head (orange dashed). It is a DEBUG signal -- "
                         "the action expert does not use it -- and it costs ~2s/frame because it "
                         "autoregressively generates ~40-90 tokens. Roughly 4x the total runtime.")
    ap.add_argument("--max_fast_new_tokens", type=int, default=160)  # >= max_fast_len 151 + markers (ee6d artifacts); greedy stops at the end marker earlier
    args = ap.parse_args()

    device = "cuda"
    out_dir = Path(args.out) if args.out else Path(args.ckpt) / "rollout_video"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    budget = load_visual_budget(args.ckpt)
    print(f"[visual budget] {budget or 'not recorded -> legacy defaults (33 tok/frame)'}")
    norm_stats_path = load_norm_stats_path(args.ckpt)
    print(f"[norm stats] {norm_stats_path or 'not stamped in ckpt -> tokenizer-dir/dataset-dir fallback'}")
    ds = build_dataset(args.fast_tok, data_dirs=args.data_dirs,
                       image_history=not args.no_image_history,
                       predict_subtask=not args.subtask_input, budget=budget,
                       norm_stats_path=norm_stats_path)
    print(f"Loading model from {args.ckpt} ...")
    model = load_model(args.ckpt, ds, device, expert_attends_subtask=not args.insulated_subtask)
    im_end_id = ds.tokenizer.convert_tokens_to_ids("<|im_end|>")

    ep_idxs = list(range(len(ds.episodes))) if args.episodes.strip() == "all" \
        else [int(s) for s in args.episodes.split(",") if s.strip()]

    fig = plt.figure(figsize=(20, 9))
    for ei in ep_idxs:
        episode = ds.episodes[ei]
        name = Path(episode["video_path"]).stem
        actions = np.asarray(episode["actions"], dtype=np.float32)
        L = len(episode["states"])
        print(f"\n=== episode {ei} ({name}), {L} steps -> decoding video ...")
        top_frames = decode_whole_video(episode["video_path"])
        wrist_frames = decode_whole_video(episode["wrist_paths"][0]) if episode.get("wrist_paths") else []

        ts = list(range(0, L, args.t_stride))
        mse_ts, fast_ts, subtask_hits, n_sub = [], [], 0, 0
        t_start = time.time()
        mp4 = out_dir / f"episode_{ei:03d}_{name}.mp4"
        writer = imageio.get_writer(mp4, fps=args.fps, codec="libx264",
                                    quality=7, macro_block_size=None)
        try:
            with torch.inference_mode():
                for k, t in enumerate(ts):
                    state = episode["states"][t]
                    gt_subtask = ds._subtask_at(episode, t)
                    if ds.data_args.image_history:
                        hist = history_at(top_frames, t, ds.num_frames, ds.stride)
                    else:
                        hist = [resize_to_budget(top_frames[min(t, len(top_frames) - 1)],
                                                 ds.data_args.max_pixels)]
                    wrist = [resize_to_budget(wrist_frames[min(t, len(wrist_frames) - 1)],
                                              ds.data_args.wrist_max_pixels)] if wrist_frames else []

                    task_text = None if ds.data_args.predict_subtask else gt_subtask
                    prompt = make_prompt(ds, state, task_text)

                    # Build the prefix the same way the SERVER does for this checkpoint's mode.
                    shown_subtask = gt_subtask
                    if args.insulated_subtask or not ds.data_args.predict_subtask:
                        mm = templatize(ds, hist, wrist, prompt, None, device)
                        if args.insulated_subtask and args.show_subtask:
                            mm_gen = templatize(ds, hist, wrist, prompt, None, device)
                            shown_subtask = generate_subtask(model, mm_gen, ds, im_end_id, 16) or ""
                    else:
                        mm_gen = templatize(ds, hist, wrist, prompt, None, device)
                        shown_subtask = generate_subtask(model, mm_gen, ds, im_end_id, 16) \
                            or ds.data_args.default_prompt
                        mm = templatize(ds, hist, wrist, prompt, shown_subtask, device)
                    if ds.data_args.predict_subtask and (args.show_subtask or not args.insulated_subtask):
                        n_sub += 1
                        subtask_hits += int(shown_subtask.strip() == gt_subtask.strip())

                    noise = torch.randn((1, ACTION_HORIZON, ACTION_DIM), device=device,
                                        dtype=torch.float32,
                                        generator=torch.Generator(device=device).manual_seed(args.seed + t))
                    stask_mask = subtask_token_mask_for(mm["input_ids"]) if args.insulated_subtask else None
                    pred = predict_expert(model, mm, ds, state, noise, args.num_flow_steps,
                                          subtask_token_mask=stask_mask)
                    gt = gt_chunk(actions, t, ACTION_HORIZON)
                    m = float(np.mean((gt - pred) ** 2))
                    mse_ts.append((t, m))

                    fast = None
                    if args.fast:
                        fast, _ = predict_fast(model, mm, ds, state, args.max_fast_new_tokens)
                        if fast is not None:
                            fast_ts.append((t, float(np.mean((gt - fast) ** 2))))

                    vg = mm.get("video_grid_thw")
                    res = f"{int(vg[0][2])*16}x{int(vg[0][1])*16}" if vg is not None else "still"
                    fast_str = f"   MSE(FAST)={fast_ts[-1][1]:.4f}" if fast_ts and fast is not None else ""
                    meta = (f"ep {ei}  {name}   t={t}/{L}   frame {k+1}/{len(ts)}   "
                            f"[model sees {res}/frame]\n"
                            f"subtask: {shown_subtask!r}   gt: {gt_subtask!r}    "
                            f"MSE(expert)={m:.4f}{fast_str}   "
                            f"running mean={np.mean([v for _, v in mse_ts]):.4f}")
                    writer.append_data(render_frame(fig, hist, wrist[0] if wrist else hist[-1],
                                                    actions, t, gt, pred, fast, mse_ts, fast_ts,
                                                    meta, ds.stride, args.dpi))
                    if (k + 1) % 10 == 0 or k == len(ts) - 1:
                        el = time.time() - t_start
                        rate = el / (k + 1)
                        print(f"  [{k+1}/{len(ts)}] t={t}  MSE={m:.4f}  "
                              f"running={np.mean([v for _, v in mse_ts]):.4f}  "
                              f"{rate:.1f}s/frame  eta {(len(ts)-k-1)*rate/60:.1f}min")
        finally:
            writer.close()

        vals = [v for _, v in mse_ts]
        worst = max(mse_ts, key=lambda p: p[1]) if mse_ts else (0, 0.0)
        print(f"  -> {mp4}  ({(time.time()-t_start)/60:.1f} min)")
        print(f"     expert: mean MSE {np.mean(vals):.4f} | median {np.median(vals):.4f} | "
              f"worst {worst[1]:.4f} @ t={worst[0]}"
              + (f" | FAST mean {np.mean([v for _, v in fast_ts]):.4f}" if fast_ts else "")
              + (f" | subtask acc {subtask_hits}/{n_sub}" if n_sub else ""))

    plt.close(fig)
    print(f"\nDone. Videos in {out_dir}")


if __name__ == "__main__":
    main()
