import os
import json
import math
import copy
import base64
import requests
from io import BytesIO
from datasets import DatasetDict, Dataset, Image as Image_ds
from PIL import Image

input_jsonl_file = "/mnt/fast10/brandon/mmr_rollout_data/prm_training_data/train/mc0.01/final_flattened_trl_format_prm_training_data_500k_mc0.01_v2.jsonl"

# parse mc0.0 from input_jsonl_file
mc_score = input_jsonl_file.split("_mc")[1].split("_v")[0]
print(f"mc_score: {mc_score}")

basename = os.path.basename(input_jsonl_file)
dataset_version = basename.split(".")[-2][-2:]  # Get last 2 chars before extension
print(f"dataset_version: {dataset_version}")

# parse prm_training_data_full_v1 from input_jsonl_file
prm_training_data_full_version = "prm_training_data_full_" + dataset_version
print(f"output directory filename: {prm_training_data_full_version}")

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == 'RGBA':
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")

def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        # fix memory leak issue while using BytesIO
        with requests.get(image, stream=True) as response:
            response.raise_for_status()
            with BytesIO(response.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            # fix memory leak issue while using BytesIO
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))

    return image

def process_example_local(example):
    """Load images from local files"""
    image_path = ""
    # print(f"example: {example}")
    for s3_url in example['images']:
        cwd_abs_path = os.path.abspath(os.getcwd())
        local_path = s3_url.replace("s3://arf-share/arf-ob1-mm-reasoning/", cwd_abs_path + "/")
        if os.path.exists(local_path):
            print(f"Setting image_path: {local_path}")
            image_path = local_path
        else: 
            print(f"Error: Local file not found: {local_path}")
            raise Exception(f"Local file not found: {local_path}")
    
    # Only update if we successfully loaded at least one image
    if image_path:
        example['image'] = image_path
    else:
        print(f"Error: No images loaded for example: {example}")
        raise Exception("Error: No images loaded for example")

    return example # has "messages" -> List[dict] and "image" -> str keys

messages_flat = []
images_flat = []
with open(input_jsonl_file, 'r', encoding='utf8') as f:
    for line in f:
        item = json.loads(line.strip())
        print(f"item: {item}")
        print(f"item['messages']: {item['messages']}")
        messages_flat.append(item['messages']) # just follow original format

        # 1. convert from S3 to local path for the ORIGINAL image.
        local_path_item = process_example_local(item) # path in string format
        print(f"local_path_item: {local_path_item}")

        # 2. Create the new file name first
        base, original_extension = os.path.splitext(local_path_item['image'])
        new_file_name = f"{base}_qwen_resized{original_extension}"
        print(f"new_file_name: {new_file_name}")

        # 3. Check if the file already exists
        if os.path.exists(new_file_name):
            print(f"File already exists at {new_file_name}, skipping processing")
        else:
            exit()
            # Load and Resize the image by Qwen's default recommended size, while maintaining maximum original resolution.
            image = fetch_image(local_path_item)
            print(f"image fetched by Qwen processing functions: {image}")

            # Save this updated image to the new local path
            image.save(new_file_name)
            print(f"image saved to {new_file_name}")
            
            if os.path.exists(new_file_name):
                print(f"image successfully saved and exists at {new_file_name}")
            else:
                print(f"image not saved to {new_file_name}")
                raise Exception(f"image not saved to {new_file_name}")

        # 4. append this updated image path to the images_flat list, to be converted by Image() later
        images_flat.append(new_file_name)

print(f"images_flat[0]: {images_flat[0]}") # MUST BE A STRING ONLY NOT ARRAY
print(f"messages_flat[0]: {messages_flat[0]}") # original message format

# create a Dataset instance from dict
hf_ds = Dataset.from_dict({"image": images_flat, "messages": messages_flat})

print(f"hf_ds after casing Dataset.from_dict: {hf_ds}")
for i in range(min(3, len(hf_ds))):
    img = hf_ds[i]['image']
    print(f"Sample {i}: {type(img)}")

print(f"now running cast column to Image() on hf_ds: {hf_ds}")
# cast the content of image column to PIL.Image
hf_ds = hf_ds.cast_column("image", Image_ds())
# create train split
dataset = DatasetDict({"train": hf_ds})

training_dataset = dataset["train"]

for i in range(min(3, len(training_dataset))):
    img = training_dataset[i]['image']
    print(f"Sample {i}: {type(img)}")

# save Arrow files locally
# dataset.save_to_disk("cache")
# set num_proc to save faster with multiprocessing
dataset.save_to_disk(f"data_conversion_scripts/converted_arrow_datasets/qwen_format/{prm_training_data_full_version}/mc{mc_score}", num_proc=4)



# TODO (Later): for array of images instead of single images, explore using Sequence before casting to Image()
# from datasets import Features, Sequence, Image
# features = Features({"image": Sequence(Image()), "messages": ...})
# hf_ds = Dataset.from_dict({"image": images, "messages": messages}, features=features)