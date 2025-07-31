import os
import json
import time
from datasets import DatasetDict, Dataset, Image
from PIL import Image as PILImage

# Usage: in root directory, run:
# python data_conversion_scripts/convert_arrow.py 

# TODO: change input_jsonl_file to the path of the jsonl file you want to convert
input_jsonl_file = "/mnt/fast10/brandon/mmr_rollout_data/prm_training_data/train/mc0.0/final_flattened_trl_format_prm_training_data_500k_mc0.0_v2.jsonl"

# parse mc0.0 from input_jsonl_file
mc_score = input_jsonl_file.split("_mc")[1].split("_v")[0]
print(f"mc_score: {mc_score}")

basename = os.path.basename(input_jsonl_file)
dataset_version = basename.split(".")[-2][-2:]  # Get last 2 chars before extension
print(f"dataset_version: {dataset_version}")

# parse prm_training_data_full_v1 from input_jsonl_file
prm_training_data_full_version = "prm_training_data_full_" + dataset_version
print(f"output directory filename: {prm_training_data_full_version}")

# exit()
def process_example_local(example):
    """Load images from local files"""
    pil_images = []
    # print(f"example: {example}")
    for s3_url in example['images']:
        cwd_abs_path = os.path.abspath(os.getcwd())
        local_path = s3_url.replace("s3://arf-share/arf-ob1-mm-reasoning/", cwd_abs_path + "/")
        if os.path.exists(local_path):
            print(f"Appending image path: {local_path}")
            pil_images.append(local_path) # we cast to Image() later, here we just append the path
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
        for path in paths: # because paths is currently a list, we assume this "hack" for now.
            images_flat.append(path)  # Single path per row
            messages_flat.append(data['messages'])

print(f"images_flat[0]: {images_flat[0]}") # MUST BE A STRING ONLY NOT ARRAY
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
dataset.save_to_disk(f"data_conversion_scripts/converted_arrow_datasets/{prm_training_data_full_version}/mc{mc_score}", num_proc=4)



# TODO (Later): for array of images instead of single images, explore using Sequence before casting to Image()
# from datasets import Features, Sequence, Image
# features = Features({"image": Sequence(Image()), "messages": ...})
# hf_ds = Dataset.from_dict({"image": images, "messages": messages}, features=features)