import os
import json
import time
from datasets import DatasetDict, Dataset, Image
from PIL import Image as PILImage

input_jsonl_file = "/mnt/fast10/brandon/mmr_rollout_data/prm_training_data_full_v0/final_flattened_trl_format_prm_training_data_500k_mc0.8_v1.jsonl"

def process_example_local(example):
    """Load images from local files"""
    pil_images = []
    # print(f"example: {example}")
    for s3_url in example['images']:
        cwd_abs_path = os.path.abspath(os.getcwd())
        local_path = s3_url.replace("s3://arf-share/arf-ob1-mm-reasoning/", cwd_abs_path + "/")
        if os.path.exists(local_path):
            print(f"Appending image from {local_path}")
            # pil_image = PILImage.open(local_path)
            # pil_images.append(pil_image)
            # pil_image_rgb = pil_image.convert("RGB")
            # pil_images.append(pil_image_rgb)
            pil_images.append(local_path)
        else: 
            print(f"Warning: Local file not found: {local_path}")
            raise Exception(f"Local file not found: {local_path}")
    
    # Only update if we successfully loaded at least one image
    if pil_images:
        example['images'] = pil_images
    else:
        print("Warning: No images loaded for example")
        example['images'] = []  # Keep it as empty list for consistency

    return example["images"]

messages_flat = []
images_flat = []
with open(input_jsonl_file, 'r', encoding='utf8') as f:
    for line in f:
        data = json.loads(line.strip())
        paths = process_example_local(data)
        for path in paths:
            images_flat.append(path)  # Single path per row
            messages_flat.append(data['messages'])

print(f"images_flat[0]: {images_flat[0]}")
print(f"messages_flat[0]: {messages_flat[0]}")
# create a Dataset instance from dict
hf_ds = Dataset.from_dict({"image": images_flat, "messages": messages_flat})

print(f"hf_ds after casing Dataset.from_dict: {hf_ds}")
for i in range(min(3, len(hf_ds))):
    img = hf_ds[i]['image']
    print(f"Sample {i}: {type(img)}")

print(f"now running cast column to Image() on hf_ds: {hf_ds}")
# cast the content of image column to PIL.Image
hf_ds = hf_ds.cast_column("image", Image())
# create train split
dataset = DatasetDict({"train": hf_ds})

training_dataset = dataset["train"]

for i in range(min(3, len(training_dataset))):
    img = training_dataset[i]['image']
    print(f"Sample {i}: {type(img)}")

# save Arrow files locally
# dataset.save_to_disk("cache")
# set num_proc to save faster with multiprocessing
dataset.save_to_disk("cache", num_proc=4)