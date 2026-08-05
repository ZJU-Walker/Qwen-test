"""READ ME TOP TO BOTTOM, THEN RUN ME. A stage-by-stage trace of how a raw camera frame
becomes model tokens, printing the real numbers at every step.

    conda activate qwen3vl
    cd /iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune
    python scripts/trace_image_pipeline.py

Every printed number is followed by the file:line that produced it. Nothing here is
recomputed by hand -- the script calls the SAME functions training calls.

There are two separate pipelines and they behave differently. That is the single most
important thing to understand:

    cam_high history (10 frames) --> VIDEO pipeline --> video_processing_qwen3_vl.py
    wrist still      (1 frame)   --> IMAGE pipeline --> image_processing_qwen2_vl_fast.py
                                     (yes, Qwen3-VL reuses Qwen2's still-image processor)

Vocabulary you need, and nothing more:

  patch      a 16x16 pixel square. The vision transformer's atom.       (patch_size=16)
  merge      after the vision tower, each 2x2 block of patches is       (merge_size=2)
             fused into ONE token. So 1 token spans 32x32 pixels.
  temporal   for VIDEO only, frames are processed in PAIRS: one token   (temporal_patch_size=2)
             covers 32x32 pixels across 2 consecutive frames.
  budget     a maximum PIXEL AREA. Not an edge length. See STAGE 2.

  => exchange rate:   1 video token = 32*32*2 = 2048 px of budget
                      1 still token = 32*32   = 1024 px of budget
"""

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

from qwenvl.action_expert.inference import build_dataset, make_prompt, templatize

FAST_TOK = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717merged"
DATA_DIRS = "/iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged"

RULE = "=" * 78


def head(n, title, where):
    print(f"\n{RULE}\nSTAGE {n}: {title}\n  code: {where}\n{RULE}")


# ============================================================================ STAGE 0
# Build the dataset exactly as training does. This is where the processor gets configured
# (STAGE 2 below), so it must happen first.
print(__doc__)
ds = build_dataset(fast_tok=FAST_TOK, data_dirs=DATA_DIRS, image_history=True, predict_subtask=True)
ep = ds.episodes[0]
t = 60

head(0, "RAW FRAMES OUT OF THE VIDEO FILE",
     "qwenvl/data/robot_data.py:421 _extract_frames  /  :447 _extract_single_frame")

# _extract_frames decodes 10 frames spaced `stride` apart, ending at timestep t.
# It does NO resizing -- frames come out at whatever the camera recorded.
top_frames = ds._extract_frames(ep, t)
nat_w, nat_h = top_frames[0].size
print(f"  cam_high history : {len(top_frames)} frames of {nat_w}x{nat_h} px "
      f"= {nat_w*nat_h:,} px each   <- native camera resolution, untouched")
print(f"                     (num_frames={ds.num_frames}, stride={ds.stride}, "
      f"from --num_frames / --frame_stride)")

# The wrist is different: _extract_single_frame takes a `budget` argument and DOES resize.
# This is OUR code, not the processor -- gate #1 of two for the wrist.
wrist_budget = ds.data_args.wrist_max_pixels
wrist_native = ds._extract_single_frame(ep["wrist_paths"][0], t, budget=10**9)  # 10**9 = no resize
wrist_images = ds._extract_wrist_images(ep, t)                                  # the real call
print(f"\n  wrist native     : {wrist_native.size[0]}x{wrist_native.size[1]} "
      f"= {wrist_native.size[0]*wrist_native.size[1]:,} px")
print(f"  wrist after OUR pre-resize: {wrist_images[0].size[0]}x{wrist_images[0].size[1]} "
      f"= {wrist_images[0].size[0]*wrist_images[0].size[1]:,} px")
print(f"    ^ GATE 1 of 2 for the wrist. robot_data.py:463-465 shrinks it to <= "
      f"wrist_max_pixels={wrist_budget:,}")
print(f"      (--wrist_max_pixels). History frames skip this -- they are passed on raw.")

# ============================================================================ STAGE 1
head(1, "WHAT THE PROCESSOR CONFIG SAYS (and how we overwrote it)",
     "qwenvl/data/data_processor.py:53 update_processor_pixels")

vp, ip = ds.processor.video_processor, ds.processor.image_processor
print("""  NAMING TRAP: the keys are called shortest_edge / longest_edge, but they are NOT
  edge lengths -- they hold PIXEL AREAS. Proof, video_processing_qwen3_vl.py:207-208
  passes them straight into smart_resize's area arguments:

        min_pixels=size.shortest_edge,
        max_pixels=size.longest_edge,

  So: shortest_edge = minimum total pixels (floor), longest_edge = maximum (ceiling).""")

VID_PX_PER_TOK = vp.patch_size * vp.merge_size * vp.patch_size * vp.merge_size * vp.temporal_patch_size
IMG_PX_PER_TOK = ip.patch_size * ip.merge_size * ip.patch_size * ip.merge_size
print(f"\n  exchange rates computed from the live config:")
print(f"    video: {vp.patch_size}*{vp.merge_size} squared * {vp.temporal_patch_size} frames"
      f" = {VID_PX_PER_TOK} px per token")
print(f"    still: {ip.patch_size}*{ip.merge_size} squared"
      f"                = {IMG_PX_PER_TOK} px per token")

# What ships on HuggingFace vs what we run with. The shipped values are hardcoded here
# because update_processor_pixels has already overwritten the live objects by now;
# they come from preprocessor_config.json / video_preprocessor_config.json of
# Qwen/Qwen3-VL-4B-Instruct (verifiable: huggingface.co/Qwen/Qwen3-VL-4B-Instruct/raw/main/...).
SHIPPED = {"video_min": 4096, "video_max": 25165824, "image_min": 65536, "image_max": 16777216}
print(f"\n  {'':22}{'SHIPPED by Qwen':>26} {'OURS after override':>26}")
for name, shipped, live, per_tok in [
    ("video floor", SHIPPED["video_min"], vp.size["shortest_edge"], VID_PX_PER_TOK),
    ("video CEILING", SHIPPED["video_max"], vp.size["longest_edge"], VID_PX_PER_TOK),
    ("image floor", SHIPPED["image_min"], ip.size["shortest_edge"], IMG_PX_PER_TOK),
    ("image ceiling", SHIPPED["image_max"], ip.size["longest_edge"], IMG_PX_PER_TOK),
]:
    print(f"  {name:22}{shipped:>13,} px{shipped/per_tok:>8.0f} tok "
          f"{live:>13,} px{live/per_tok:>8.0f} tok")
print(f"""
  Read the 'image floor' row: Qwen ships a floor of {SHIPPED['image_min']:,} px = 64 tokens --
  a real minimum. data_processor.py:71-79 replaces it with --min_pixels ({ds.data_args.min_pixels}),
  i.e. we removed Qwen's floor. That is why nothing ever warned us about low resolution.

  Read the 'video floor' row: Qwen's own video floor is only {SHIPPED['video_min']:,} px
  = {SHIPPED['video_min']/VID_PX_PER_TOK:.0f} tokens FOR THE WHOLE CLIP -- so the video path has no
  meaningful floor even before we touch it.

  The 'video CEILING' row is the one that decides our resolution (STAGE 2). It comes from
  video_max_pixels in qwenvl/train/argument.py:23, whose default is 1024*28*28. The 28
  is a Qwen2-era constant: Qwen2 used 14px patches (14*2 = 28 px per token). Qwen3 uses
  16px patches (16*2 = 32 px per token), so that arithmetic no longer means what it did.""")

# ============================================================================ STAGE 2
head(2, "THE RESIZE -- the only step that actually destroys pixels",
     "transformers/.../qwen3_vl/video_processing_qwen3_vl.py:34-64 smart_resize")

from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize as video_smart_resize

print("""  The exact source (video_processing_qwen3_vl.py:51-58), with the meaning of each line:

    h_bar = round(height / factor) * factor      # snap height to a multiple of factor
    w_bar = round(width  / factor) * factor      # snap width  to a multiple of factor
    t_bar = round(num_frames / temporal_factor) * temporal_factor   # t_bar counts FRAMES

    if t_bar * h_bar * w_bar > max_pixels:       # <-- frames x height x width, ALL frames
        beta  = sqrt((num_frames * height * width) / max_pixels)    # how much too big we are
        h_bar = max(factor, floor(height / beta / factor) * factor) # shrink + snap DOWN
        w_bar = max(factor, floor(width  / beta / factor) * factor)

  `factor` is patch_size * merge_size = 32: every dimension must be a whole number of
  32px token-squares. `beta` is the linear shrink factor -- note sqrt, because the budget
  is an AREA and we are shrinking two dimensions.

  KEY: t_bar is the FRAME COUNT, so the budget is shared across the whole clip.
  More history frames => each frame gets proportionally fewer pixels. Same budget.""")

factor = vp.patch_size * vp.merge_size
budget = vp.size["longest_edge"]
beta = math.sqrt((len(top_frames) * nat_h * nat_w) / budget)
h_bar, w_bar = video_smart_resize(
    num_frames=len(top_frames), height=nat_h, width=nat_w,
    temporal_factor=vp.temporal_patch_size, factor=factor,
    min_pixels=vp.size["shortest_edge"], max_pixels=budget,
)
print(f"\n  live call with OUR numbers:")
print(f"    input volume  = {len(top_frames)} frames x {nat_h} x {nat_w} "
      f"= {len(top_frames)*nat_h*nat_w:,} px")
print(f"    budget        = {budget:,} px   (over-budget by {len(top_frames)*nat_h*nat_w/budget:.2f}x)")
print(f"    beta          = sqrt({len(top_frames)*nat_h*nat_w:,} / {budget:,}) = {beta:.4f}")
print(f"    height {nat_h} / {beta:.4f} = {nat_h/beta:.1f} -> floor to multiple of {factor} -> {h_bar}")
print(f"    width  {nat_w} / {beta:.4f} = {nat_w/beta:.1f} -> floor to multiple of {factor} -> {w_bar}")
print(f"    => EACH HISTORY FRAME IS RESIZED TO {w_bar} x {h_bar} = {w_bar*h_bar:,} px")
print(f"       ({nat_w*nat_h/(w_bar*h_bar):.1f}x less area than native)")
print(f"\n  Note the flooring loss: the budget allows {budget//len(top_frames):,} px/frame,")
print(f"  but snapping DOWN to multiples of {factor} lands us at {w_bar*h_bar:,} px/frame"
      f" ({100*(1-w_bar*h_bar/(budget/len(top_frames))):.0f}% lost to rounding).")

# ============================================================================ STAGE 3
head(3, "PIXELS -> PATCHES -> TOKENS (pure arithmetic, no pixels lost)",
     "video_processing_qwen3_vl.py:238-259 (the view/permute/reshape)")

grid_t, grid_h, grid_w = len(top_frames) // vp.temporal_patch_size, h_bar // vp.patch_size, w_bar // vp.patch_size
print(f"  cut the {w_bar}x{h_bar} frame into {vp.patch_size}x{vp.patch_size} patches:")
print(f"    across: {w_bar} / {vp.patch_size} = {grid_w} patches")
print(f"    down  : {h_bar} / {vp.patch_size} = {grid_h} patches")
print(f"  pair up frames (temporal_patch_size={vp.temporal_patch_size}):")
print(f"    {len(top_frames)} frames / {vp.temporal_patch_size} = {grid_t} temporal groups")
print(f"\n  => grid_thw = ({grid_t}, {grid_h}, {grid_w})   <- 'grid_thw' means (time, height, width) IN PATCHES")
print(f"     video_processing_qwen3_vl.py:262 returns exactly this")
print(f"\n  then the 2x2 merge (merge_size={vp.merge_size}) fuses each 2x2 patch block into 1 token:")
print(f"    per temporal group: ({grid_h}/{vp.merge_size}) x ({grid_w}/{vp.merge_size}) "
      f"= {grid_h//vp.merge_size} x {grid_w//vp.merge_size} = {(grid_h//vp.merge_size)*(grid_w//vp.merge_size)} tokens")
n_video_tokens = grid_t * (grid_h // vp.merge_size) * (grid_w // vp.merge_size)
print(f"    all {grid_t} groups   : {grid_t} x {(grid_h//vp.merge_size)*(grid_w//vp.merge_size)} "
      f"= {n_video_tokens} tokens TOTAL")
print(f"    per frame          : {n_video_tokens} / {len(top_frames)} = {n_video_tokens/len(top_frames):.0f} TOKENS PER FRAME")

# ============================================================================ STAGE 4
head(4, "THE WRIST, THROUGH THE OTHER PIPELINE",
     "transformers/.../qwen2_vl/image_processing_qwen2_vl_fast.py smart_resize")

from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize as image_smart_resize

pw, ph = wrist_images[0].size
ih, iw = image_smart_resize(ph, pw, factor=ip.patch_size * ip.merge_size,
                            min_pixels=ip.size["shortest_edge"], max_pixels=ip.size["longest_edge"])
n_wrist_tokens = (ih // ip.patch_size // ip.merge_size) * (iw // ip.patch_size // ip.merge_size)
print(f"  Same idea, but NO temporal dimension and the budget is per-image, not per-clip.")
print(f"    native            {wrist_native.size[0]}x{wrist_native.size[1]}")
print(f"    GATE 1 our resize {pw}x{ph}   (robot_data.py:463, --wrist_max_pixels={wrist_budget:,})")
print(f"    GATE 2 processor  {iw}x{ih}   (--max_pixels={ip.size['longest_edge']:,})")
print(f"    tokens            ({ih}/{ip.patch_size}/{ip.merge_size}) x ({iw}/{ip.patch_size}/{ip.merge_size})"
      f" = {n_wrist_tokens}")
print(f"""
  WHY TWO GATES MATTER: gate 1 already shrank the image, so raising ONLY --max_pixels
  changes nothing -- the small image is already under the ceiling. Both must move together
  to give the wrist more resolution.""")

# ============================================================================ STAGE 5
head(5, "TOKENS TAKE THEIR PLACE IN THE SEQUENCE (ground truth, not arithmetic)",
     "qwenvl/action_expert/inference.py:152 templatize -> processor.apply_chat_template")

prompt = make_prompt(ds, ep["states"][t])
mm = templatize(ds, top_frames, wrist_images, prompt, "pick up green block", "cpu")
ids = mm["input_ids"][0]
vid_pad = ds.tokenizer.convert_tokens_to_ids("<|video_pad|>")
img_pad = ds.tokenizer.convert_tokens_to_ids("<|image_pad|>")
n_vid, n_img = int((ids == vid_pad).sum()), int((ids == img_pad).sum())

print(f"  The processor reserves one placeholder token per visual token, which the model")
print(f"  later swaps for the real vision-tower output.  Counting them in the actual tensor:")
print(f"    <|video_pad|> x {n_vid}   <- STAGE 3 predicted {n_video_tokens}   "
      f"{'MATCH' if n_vid == n_video_tokens else 'MISMATCH!'}")
print(f"    <|image_pad|> x {n_img}   <- STAGE 4 predicted {n_wrist_tokens}   "
      f"{'MATCH' if n_img == n_wrist_tokens else 'MISMATCH!'}")
print(f"    grid returned by the processor: video_grid_thw={mm['video_grid_thw'].tolist()}, "
      f"image_grid_thw={mm['image_grid_thw'].tolist()}")
print(f"    total sequence length: {ids.numel()} tokens "
      f"({n_vid} video + {n_img} wrist + {ids.numel()-n_vid-n_img} text)")

# ============================================================================ SUMMARY
head("*", "THE ONE FORMULA, AND WHERE THE KNOBS LIVE", "summary")
print(f"""    tokens per history frame  ~=  video_max_pixels / (num_frames * {VID_PX_PER_TOK})

  check against this run: {budget:,} / ({len(top_frames)} * {VID_PX_PER_TOK})
      = {budget/(len(top_frames)*VID_PX_PER_TOK):.1f} ideal, {n_video_tokens/len(top_frames):.0f} actual after flooring.

  KNOBS, in the order they act:
    --wrist_max_pixels   robot_data.py:463   wrist only, OUR pre-resize      (gate 1)
    --max_pixels         argument.py:19      stills ceiling                  (gate 2)
    --min_pixels         argument.py:20      stills floor (we set it to ~0, erasing Qwen's 64-token floor)
    video_max_pixels     argument.py:23      THE history knob. NOT exposed as a CLI flag today.
    --num_frames         argument.py         divides the video budget: more frames => smaller frames

  WHERE WE STAND: {n_video_tokens/len(top_frames):.0f} tokens/history-frame and {n_wrist_tokens} for the wrist,
  against Qwen3-VL's own shipped image floor of {SHIPPED['image_min']//IMG_PX_PER_TOK} tokens.
""")
