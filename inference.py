import torch
import decord
import numpy as np
import cv2
import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

# ==========================================
# 1. Configuration Setup
# ==========================================
model_checkpoint_path = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/qwen3_4b/checkpoint-6000"
processor_path = "Qwen/Qwen3-VL-4B-Instruct"

# --- CHANGED: Point these to your input and output folders ---
input_video_dir = "/iris/projects/humanoid/trossen_data/yellow_blocks" 
output_video_dir = "./inference_outputs/"
# -------------------------------------------------------------

question = "which colored block did the human hand point to?"

stride = 10
num_frames = 10
eval_step = 1 

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. Load the Fully Fine-Tuned Model
# ==========================================
print(f"Loading full model from {model_checkpoint_path}...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_checkpoint_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

print("Loading processor...")
processor = AutoProcessor.from_pretrained(processor_path)

# Align processor constraints with the 4B training run (sft_qwen3_4b_bk.sh).
# These MUST match update_processor_pixels() at train time, or the model sees a
# different input distribution than it was trained on.
#   - image pixels come from the script's --max_pixels/--min_pixels (no images here, but kept for parity)
#   - video pixels/frames/fps come from the DataArguments defaults (the 4B script does not override them)
if hasattr(processor, "image_processor"):
    ip = processor.image_processor
    ip.max_pixels = 50176
    ip.min_pixels = 784
    if isinstance(getattr(ip, "size", None), dict):
        ip.size["shortest_edge"] = 784
        ip.size["longest_edge"] = 50176

video_proc = getattr(processor, "video_processor", processor.image_processor)
video_proc.fps = 2.0
video_proc.max_pixels = 1024 * 28 * 28   # 802816 (DataArguments default)
video_proc.min_pixels = 256 * 28 * 28    # 200704 (DataArguments default)
video_proc.max_frames = 8
video_proc.min_frames = 4
if isinstance(getattr(video_proc, "size", None), dict):
    video_proc.size["shortest_edge"] = 256 * 28 * 28
    video_proc.size["longest_edge"] = 1024 * 28 * 28

print("✅ Processor aligned with 4B training hyperparameters!")

# ==========================================
# 3. Helper Functions
# ==========================================
def extract_window_frames(vr, t, frame_shape):
    """Extracts history frames with a strict stride. Pads with zero arrays if idx < 0."""
    frame_indices = [t - (i * stride) for i in reversed(range(num_frames))]
    
    # --- PASTE THIS BLOCK ---
    # Clamp negative indices to 0
    clamped_indices = [max(0, idx) for idx in frame_indices]
    
    # Fetch in one batch
    extracted_frames = vr.get_batch(clamped_indices).asnumpy()
    
    pil_images = [Image.fromarray(f) for f in extracted_frames]
    
    # Return original frame_indices so your dashboard still correctly labels "t=-15", etc.
    return pil_images, extracted_frames, frame_indices


def predict_action(video_frames):
    """Passes the frames through Qwen and returns the text string.

    NOTE: this mirrors the training preprocessing in data_processor.preprocess_qwen_visual
    exactly -- single apply_chat_template call with tokenize=True and do_sample_frames=False.
    Using the older two-step processor() path here would let the video processor re-sample
    the 10 frames down to ~4 (and shift the timestamp tokens), creating a train/inference
    mismatch the model was never trained for.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_frames},
                {"type": "text", "text": question}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,   # append the assistant prompt so the model generates
        do_sample_frames=False,       # keep all extracted frames, exactly like training
    )

    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=10)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    return output_text.strip()


def create_dashboard_frame(raw_frames, indices, prediction, t, total_frames):
    """Builds a composite dashboard with the main frame, history grid, and labels."""
    cv_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in raw_frames]
    
    main_w, main_h = 640, 480
    history_count = num_frames - 1
    hist_w = main_w // history_count  
    hist_h = int(hist_w * (main_h / main_w))  
    
    canvas = np.zeros((690, 640, 3), dtype=np.uint8)

    # 1. Place Main Frame (Current timestep)
    main_img = cv2.resize(cv_frames[-1], (main_w, main_h))
    canvas[50:530, 0:640] = main_img

    # 2. Place History Grid
    for i in range(history_count):
        h_img = cv2.resize(cv_frames[i], (hist_w, hist_h))
        x_offset = i * hist_w
        # canvas[530:650, x_offset:x_offset+hist_w] = h_img
        canvas[530:530+hist_h, x_offset:x_offset+hist_w] = h_img

        
        # Explicitly label padded frames versus real frames
        if indices[i] < 0:
            cv2.putText(canvas, "PADDING", (x_offset + 5, 545), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(canvas, "t=0 (Blank)", (x_offset + 5, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
        else:
            cv2.putText(canvas, f"t={indices[i]}", (x_offset + 5, 545), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(canvas, "History", (x_offset + 5, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # 3. Top Label (Time)
    cv2.putText(canvas, f"Time Step: {t} / {total_frames - 1}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 4. Bottom Label (Model Prediction)
    # color = (0, 255, 0) if "green block" in prediction.lower() else (0, 255, 255)
    # if "yellow block" not in prediction.lower() and "green block" not in prediction.lower():
    #     color = (255, 255, 255) 
        
    # cv2.putText(canvas, f"Model Output: {prediction}", (15, 675), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    color = (255, 255, 0) # Cyan in BGR format
    
    # --- REVISED: Moved the text up to Y=645 to avoid media player control overlap ---
    cv2.putText(canvas, f"Model Output: {prediction}", (15, 645), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return canvas

# ==========================================
# 4. The Batch Sliding Window Loop
# ==========================================
input_path = Path(input_video_dir)
output_path = Path(output_video_dir)

# Ensure output directory exists
output_path.mkdir(parents=True, exist_ok=True)

# Find all mp4 files in the target directory
video_files = list(input_path.glob("*.mp4"))

if not video_files:
    print(f"❌ No .mp4 files found in {input_video_dir}")
else:
    print(f"Found {len(video_files)} videos to process.")

for video_file in video_files:
    print(f"\n" + "="*50)
    print(f"Processing: {video_file.name}")
    print("="*50)
    
    try:
        vr = decord.VideoReader(str(video_file))
        total_frames = len(vr)
        native_shape = vr[0].asnumpy().shape
        
        # Dynamically name the output file
        out_filename = f"sliding_eval_{video_file.name}"
        out_filepath = str(output_path / out_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(out_filepath, fourcc, 30.0, (640, 690))
        
        for t in tqdm(range(0, total_frames, eval_step), desc=f"Evaluating {video_file.name}"):
            pil_images, raw_numpy_frames, frame_indices = extract_window_frames(vr, t, native_shape)
            prediction = predict_action(pil_images)
            dashboard = create_dashboard_frame(raw_numpy_frames, frame_indices, prediction, t, total_frames)
            out_writer.write(dashboard)
            
        out_writer.release()
        print(f"✅ Saved to: {out_filepath}")
        
    except Exception as e:
        print(f"❌ Failed to process {video_file.name}. Error: {e}")

print("\n🎉 Batch processing complete!")