"""
Real-time inference server for the fine-tuned Qwen3-VL-4B task planner.

Mirrors the TRAINING preprocessing exactly (see inference.py / data_processor.py):
  - Qwen3VLForConditionalGeneration + the 4B checkpoint
  - processor aligned with update_processor_pixels() (incl. the `size` dict)
  - apply_chat_template(tokenize=True, do_sample_frames=False) so all frames are kept
    and the timestamp tokens match what the model was trained on.

The client (client_qwen_4b.py) must send exactly NUM_FRAMES frames sampled at the
training stride, oldest-first.
"""

import io
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ==========================================
# Configuration  (must match training / inference.py)
# ==========================================
MODEL_PATH = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/qwen3_4b/checkpoint-5000"
PROCESSOR_PATH = "Qwen/Qwen3-VL-4B-Instruct"

# The model was trained on 10 frames at stride 10 (PrototypeRobotDataset).
# The client must send exactly this many frames.
NUM_FRAMES = 10
MAX_NEW_TOKENS = 10
DEVICE = "cuda"

app = FastAPI(title="Qwen3-VL-4B Real-Time Task Planner")

# ==========================================
# Load model + processor
# ==========================================
print(f"Loading fine-tuned model from {MODEL_PATH} ...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
)
model.eval()

print("Loading and aligning processor...")
processor = AutoProcessor.from_pretrained(PROCESSOR_PATH)

# Align processor with the 4B training run (sft_qwen3_4b_bk.sh).
# These MUST match update_processor_pixels() at train time -- including the `size`
# dict, which is what actually drives the resize -- or the model sees a different
# input distribution than it was trained on.
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

print("✅ Server processor aligned with 4B training hyperparameters.")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH, "num_frames": NUM_FRAMES}


@app.post("/predict")
async def predict(
    files: list[UploadFile] = File(...),
    prompt: str = Form(...),
):
    if len(files) != NUM_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Must submit exactly {NUM_FRAMES} sequential frames (oldest first); got {len(files)}.",
        )

    try:
        # Decode the uploaded JPEGs into PIL frames, oldest-first (as sent).
        frame_images = []
        for file in files:
            file_bytes = await file.read()
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            frame_images.append(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frame_images},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # IMPORTANT: identical to training/inference preprocessing.
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            do_sample_frames=False,
        ).to(DEVICE)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return {"status": "success", "prediction": output_text.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Port 8002 so this can run alongside the 3B server (8001) if needed.
    uvicorn.run(app, host="0.0.0.0", port=8002)
