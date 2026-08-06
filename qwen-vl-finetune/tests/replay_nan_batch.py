"""Replay a nan_batch.pt saved by the QWEN_NAN_DEBUG trap and bisect the nan.

The trap fires when a training micro-batch produces a non-finite loss and saves the
exact collated inputs. This script replays the CE path (the one that went nan: plain
VLM forward -> lm_head -> CE) on a fresh pretrained VLM and reports:

  1. whole-batch replay: do the losses go non-finite again on healthy weights?
     (yes => data-content trigger; no => the training run's weights were corrupt,
     i.e. an optimizer/gradient-side problem -- check bad_params in the dump)
  2. per-sample replay (batch of 1 each, with the visual tensors correctly sliced
     per sample): WHICH sample triggers it
  3. module bisect on the triggering forward: the first module whose output
     contains a non-finite value

Run:  python tests/replay_nan_batch.py --dump <output_dir>/nan_batch.pt
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, ".")
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

from qwenvl.action_expert.modeling_qwen3vl_with_expert import Qwen3VLWithActionExpert  # noqa: E402,F401
from transformers import Qwen3VLForConditionalGeneration, set_seed  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def language_losses(logits, labels, fast_token_mask):
    """Mirror of Qwen3VLWithActionExpert._language_losses."""
    shift_logits = logits[:, :-1].float().reshape(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].reshape(-1)
    per_tok = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="none")
    valid = shift_labels != -100
    is_fast = fast_token_mask[:, 1:].reshape(-1) if fast_token_mask is not None else torch.zeros_like(valid)
    text_sel, fast_sel = valid & ~is_fast, valid & is_fast
    lm = per_tok[text_sel].mean().item() if text_sel.any() else None
    fast = per_tok[fast_sel].mean().item() if fast_sel.any() else None
    return lm, fast


def visual_slices(grid_thw, per_sample_rows):
    """Row ranges and flat patch ranges for each sample given rows-per-sample."""
    counts = grid_thw.prod(dim=1)  # patches per grid row
    row_edges = [0]
    for n in per_sample_rows:
        row_edges.append(row_edges[-1] + n)
    out = []
    for i in range(len(per_sample_rows)):
        r0, r1 = row_edges[i], row_edges[i + 1]
        p0 = int(counts[:r0].sum())
        p1 = int(counts[:r1].sum())
        out.append((r0, r1, p0, p1))
    return out


def forward_ce(vlm, batch, device):
    kwargs = dict(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        position_ids=batch["position_ids"].to(device),
        pixel_values=batch.get("pixel_values"),
        image_grid_thw=batch.get("image_grid_thw"),
        pixel_values_videos=batch.get("pixel_values_videos"),
        video_grid_thw=batch.get("video_grid_thw"),
        use_cache=False,
    )
    for k in ("pixel_values", "pixel_values_videos"):
        if kwargs[k] is not None:
            kwargs[k] = kwargs[k].to(device=device, dtype=torch.bfloat16)
    for k in ("image_grid_thw", "video_grid_thw"):
        if kwargs[k] is not None:
            kwargs[k] = kwargs[k].to(device)
    out = vlm.model(**{k: v for k, v in kwargs.items() if v is not None})
    logits = vlm.lm_head(out.last_hidden_state)
    lm, fast = language_losses(logits, batch["labels"].to(device),
                               batch.get("fast_token_mask", None).to(device)
                               if batch.get("fast_token_mask") is not None else None)
    finite_hidden = torch.isfinite(out.last_hidden_state).all().item()
    finite_logits = torch.isfinite(logits).all().item()
    return lm, fast, finite_hidden, finite_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--vocab", type=int, default=152695, help="resized tokenizer size from the run log")
    args = ap.parse_args()
    device = "cuda"

    d = torch.load(args.dump, weights_only=False)
    print(f"dump from global_step {d.get('global_step')}: components {d.get('components')}")
    print(f"bad_params recorded at crash: {len(d.get('bad_params') or [])}")
    for p in (d.get("provenance") or [])[-6:]:
        print("  recent sample:", p)
    batch = d["inputs"]
    bsz, seq = batch["input_ids"].shape
    print(f"batch: {bsz} samples, padded seq {seq}")

    set_seed(42)
    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-4B-Instruct", torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir=os.environ["HF_HOME"]).to(device)
    vlm.resize_token_embeddings(args.vocab)
    vlm.eval()

    with torch.no_grad():
        lm, fast, fh, fl = forward_ce(vlm, batch, device)
    print(f"\n[whole batch on FRESH weights] lm={lm} fast={fast} "
          f"hidden_finite={fh} logits_finite={fl}")
    if lm is not None and fast is not None and all(
            map(lambda v: v == v, [lm, fast])) and fh and fl:
        print("=> batch is CLEAN on healthy weights: the training nan came from "
              "corrupted WEIGHTS (optimizer/gradient side). Check bad_params above.")
        return

    # per-sample bisect (2 video rows + 2 image rows per sample in the humanprompt setup)
    vids_per_sample = [2] * bsz
    imgs_per_sample = [2] * bsz
    vslices = visual_slices(batch["video_grid_thw"], vids_per_sample)
    islices = visual_slices(batch["image_grid_thw"], imgs_per_sample)
    for i in range(bsz):
        vr0, vr1, vp0, vp1 = vslices[i]
        ir0, ir1, ip0, ip1 = islices[i]
        sub = {
            "input_ids": batch["input_ids"][i:i + 1],
            "attention_mask": batch["attention_mask"][i:i + 1],
            "position_ids": batch["position_ids"][:, i:i + 1],
            "labels": batch["labels"][i:i + 1],
            "fast_token_mask": batch.get("fast_token_mask", None)[i:i + 1]
            if batch.get("fast_token_mask") is not None else None,
            "pixel_values": batch["pixel_values"][ip0:ip1],
            "image_grid_thw": batch["image_grid_thw"][ir0:ir1],
            "pixel_values_videos": batch["pixel_values_videos"][vp0:vp1],
            "video_grid_thw": batch["video_grid_thw"][vr0:vr1],
        }
        with torch.no_grad():
            lm, fast, fh, fl = forward_ce(vlm, sub, device)
        bad = (lm is not None and lm != lm) or (fast is not None and fast != fast) or not fh or not fl
        tag = "NAN <<<<" if bad else "ok"
        print(f"[sample {i}] lm={lm} fast={fast} hidden_finite={fh} logits_finite={fl}  {tag}")
        if bad:
            # module bisect: first module emitting a non-finite output
            first_bad = []

            def make_hook(name):
                def hook(mod, inp, out):
                    if first_bad:
                        return
                    ts = [t for t in (out if isinstance(out, tuple) else (out,))
                          if torch.is_tensor(t) and t.is_floating_point()]
                    if any(not torch.isfinite(t).all() for t in ts):
                        first_bad.append(name)
                return hook

            handles = [m.register_forward_hook(make_hook(n))
                       for n, m in vlm.named_modules() if n]
            with torch.no_grad():
                forward_ce(vlm, sub, device)
            for h in handles:
                h.remove()
            print(f"    first non-finite module: {first_bad[0] if first_bad else 'NOT FOUND (loss-only?)'}")


if __name__ == "__main__":
    main()
