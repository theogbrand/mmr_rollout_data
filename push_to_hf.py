from datasets import Dataset, DatasetDict
import json
import os
from PIL import Image

# Load your JSONL file
file_path = "/mnt/fast10/brandon/mmr_rollout_data/prm_training_data_full_v0/final_flattened_trl_format_prm_training_data_500k_mc0.8_v1.jsonl"
# file_path = "/mnt/fast10/brandon/mmr_rollout_data/prm_training_data/train/AI2D_final_mc_rollouts_with_all_models_verification_merged_prm_training_data_final_trl_format_mc0.0.jsonl"

data = []
with open(file_path, 'r') as f:
    for line in f:
        data.append(json.loads(line.strip()))

print(f"Loaded {len(data)} samples")

# def s3_url_to_local_path(s3_url, local_base_dir="/mnt/fast10/brandon/mmr_rollout_data/ai2d_images"):
def s3_url_to_local_path(s3_url, local_base_dir="/mnt/fast10/brandon/mmr_rollout_data/training_data_images"):
    """Convert S3 URL to local file path"""
    # Extract filename from S3 URL
    # s3://arf-share/arf-ob1-mm-reasoning/training_data_images/AI2D/subset_images/706.png
    # -> /tmp/ai2d_images/AI2D/706.png
    cwd_abs_path = os.path.abspath(os.getcwd())
    local_path = s3_url.replace("s3://arf-share/arf-ob1-mm-reasoning/", cwd_abs_path + "/")
    # path_parts = local_path.split('/')
    # filename = '/'.join(path_parts[-4:])  # Get last two parts: AI2D/706.png
    return local_path
    
def process_example_local(example):
    """Load images from local files"""
    pil_images = []
    for s3_url in example['images']:
        local_path = s3_url_to_local_path(s3_url)
        try:
            if os.path.exists(local_path):
                pil_image = Image.open(local_path)
                pil_images.append(pil_image)  # Actually append the loaded image!
            else:
                raise Exception(f"Warning: Local file not found: {local_path}")
        except Exception as e:
            raise Exception(f"Error loading {local_path}: {e}")
    
    # Only update if we successfully loaded at least one image
    if pil_images:
        example['images'] = pil_images
    else:
        print(f"Warning: No images loaded for example")
        example['images'] = []  # Keep it as empty list for consistency
    
    return example

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(data)

# Process images from local files - this will be MUCH faster
print("Converting local images to PIL Images...")
dataset = [process_example_local(sample) for sample in dataset]

# Convert dataset to OAI messages
# need to use list comprehension to keep Pil.Image type, .mape convert image to bytes

# dataset = dataset.map(
#     process_example_local, 
#     num_proc=min(32, os.cpu_count() * 4),  # Can use more processes now
#     desc="Loading local images"
# )

print("🔍 Verifying image types...")
for i in range(min(3, len(dataset))):
    img = dataset[i]['images'][0]
    print(f"Sample {i}: {type(img)}")

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(dataset)

# Push to HuggingFace
username = "ob11"
dataset_name = "ai2d-prm-training-data-v0-full-test"
full_dataset_name = f"{username}/{dataset_name}"

print(f"\n🚀 Pushing to HuggingFace: {full_dataset_name}")

try:
    dataset.push_to_hub(
        full_dataset_name,
        private=True,  # Set to True if you want it private
        token=os.getenv("HF_TOKEN")  # Make sure your HF_TOKEN is set
    )
    print(f"✅ Dataset successfully pushed to: https://huggingface.co/datasets/{full_dataset_name}")
    print(f"🎉 Images are now stored as PIL.Image objects, not byte arrays!")
except Exception as e:
    print(f"❌ Error pushing to HuggingFace: {e}")
    print("Make sure your HF_TOKEN environment variable is set")
