from datasets import Dataset, DatasetDict, Features, Value, Image as HFImage, Sequence, load_from_disk
import json
import os
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

if os.getenv("HF_TOKEN") is None:
    raise Exception("HF_TOKEN is not set")
else:
    print("HF_TOKEN is set: ", os.getenv("HF_TOKEN")[:5] + "...")

# Load your JSONL file
file_path = "data_conversion_scripts/converted_arrow_datasets/qwen_format/prm_training_data_full_v2_custom_token/mc0.01" 

training_dataset = load_from_disk(file_path)

username = "ob11"
dataset_name = "visual-prm-training-data-v2-mc0.01-custom-token-qwen-format" # TODO: to edit to dataset name
full_dataset_name = f"{username}/{dataset_name}"

print(f"\n🚀 Pushing to HuggingFace: {full_dataset_name}")

try:
    training_dataset.push_to_hub(
        full_dataset_name,
        private=True,  # Set to True if you want it private
        token=os.getenv("HF_TOKEN")  # Make sure your HF_TOKEN is set
    )
    print(f"✅ Dataset successfully pushed to: https://huggingface.co/datasets/{full_dataset_name}")
    print(f"🎉 Images are now stored as PIL.Image objects, not byte arrays!")
except Exception as e:
    print(f"❌ Error pushing to HuggingFace: {e}")
    print("Make sure your HF_TOKEN environment variable is set")
