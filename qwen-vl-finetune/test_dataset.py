import os
import json
import torch
from transformers import AutoProcessor

# 1. Cleanly import your datasets from the package
import qwenvl.data.data_processor as dp
from qwenvl.data.data_processor import PrototypeRobotDataset, LazySupervisedDataset

print("\n" + "="*50)
print("🛠️  INITIATING DATASET DISSECTION EXPERIMENT")
print("="*50)

# 2. Mock Arguments matching your bash script constraints
class MockArgs:
    model_type = "qwen2.5vl"
    min_pixels = 12544
    max_pixels = 451584
    video_min_pixels = 200704
    video_max_pixels = 1304576
    video_min_frames = 4
    video_max_frames = 8
    video_fps = 2.0
    data_packing = False
    data_flatten = False
    dataset_use = "mock_dataset"

data_args = MockArgs()

# 3. Point to a REAL video on your server
REAL_VIDEO_PATH = "/iris/projects/humanoid/trossen_data/0528_green_block_mem_copy/videos/chunk-000/observation.images.cam_high/h264_videos/episode_000000.mp4"

# 4. Create a temporary JSON file to mimic LazySupervisedDataset input
dummy_json_path = "temp_mock_dataset.json"
mock_lazy_data = [{
    "id": "eval_001",
    "video": REAL_VIDEO_PATH,
    "conversations": [
        {"from": "human", "value": "<video>\nWhat action should the robot take next?"},
        {"from": "gpt", "value": "pick up green block"}
    ]
}]

with open(dummy_json_path, "w") as f:
    json.dump(mock_lazy_data, f)

# 5. Cleanly mock the data_list function from the OUTSIDE
dp.data_list = lambda x: [{"annotation_path": dummy_json_path, "data_path": "", "sampling_rate": 1.0}]

# 6. Load Processor
print("\nLoading Processor...")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# 7. Initialize both datasets
print("\nInitializing PrototypeRobotDataset...")
proto_dataset = PrototypeRobotDataset(processor, data_args)

print("\nInitializing LazySupervisedDataset...")
lazy_dataset = LazySupervisedDataset(processor, data_args)

# 8. Extract samples
print("\n" + "="*50)
print("🔍 EXTRACTING SAMPLES")
print("="*50)

print("Fetching Sample 0 from Prototype Dataset...")
proto_sample = proto_dataset[0]

print("Fetching Sample 0 from LazySupervised Dataset...")
lazy_sample = lazy_dataset[0]

# 9. Print and Compare Shapes
def print_tensor_shapes(name, sample_dict):
    print(f"\n--- {name} Tensor Shapes ---")
    for key, value in sample_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"{key:<20} | Shape: {value.shape}")
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            print(f"{key:<20} | Shape: List of {len(value)} Tensors (e.g. {value[0].shape})")
        else:
            print(f"{key:<20} | Type: {type(value)}")

print_tensor_shapes("PROTOTYPE DATASET", proto_sample)
print_tensor_shapes("LAZY SUPERVISED DATASET", lazy_sample)

# 10. Cleanup
if os.path.exists(dummy_json_path):
    os.remove(dummy_json_path)
print("\n✅ Dissection Complete.")