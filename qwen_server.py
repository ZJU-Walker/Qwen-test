import io
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

app = FastAPI(title="Qwen 2.5-VL Real-Time Aligned Server")

MODEL_PATH = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/checkpoint-2000" 
PROCESSOR_PATH = "Qwen/Qwen2.5-VL-3B-Instruct" 

print("Loading fine-tuned model onto GPU...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)
model.eval()

print("Loading and aligning processor...")
processor = AutoProcessor.from_pretrained(PROCESSOR_PATH)

# Fix 1: Align processor constraints exactly with your evaluation script
if hasattr(processor, "image_processor"):
    processor.image_processor.max_pixels = 451584
    processor.image_processor.min_pixels = 12544

video_proc = getattr(processor, "video_processor", processor.image_processor)
video_proc.fps = 2.0
video_proc.max_pixels = 1304576
video_proc.min_pixels = 200704
video_proc.max_frames = 8
video_proc.min_frames = 4

print("✅ Server processor fully aligned with inference hyperparameters.")

@app.post("/predict")
async def predict(
    files: list[UploadFile] = File(...), 
    prompt: str = Form(...)
):
    if len(files) != 5:
        raise HTTPException(status_code=400, detail="Must submit exactly 5 sequential frames.")

    try:
        frame_images = []
        for file in files:
            file_bytes = await file.read()
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            frame_images.append(image)

        # Fix 2: Remove hardcoded chat template properties to let video_proc configs rule
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frame_images},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Fix 3: Nested list wrapper [frame_images] to match your batch inference format
        inputs = processor(
            text=[text],
            images=None,
            videos=[frame_images], 
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            # Fix 4: Set max_new_tokens to 10 for rapid real-time looping
            generated_ids = model.generate(**inputs, max_new_tokens=10)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]

        return {"status": "success", "prediction": output_text.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)