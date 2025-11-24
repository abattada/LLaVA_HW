import json
from datasets import load_dataset
import os
from PIL import Image

# Download Dataset
data_dir = "data"
vqa_raw_dir = os.path.join(data_dir, "vqa")
os.makedirs(vqa_raw_dir, exist_ok = True)
vqa_raw = load_dataset("flaviagiammarino/vqa-rad", cache_dir = vqa_raw_dir)

# Make json format of test and train
# ms-swift will shuffle the dataset, so storing in static format like json is viable

split_names = ["train", "test"]

# for train, test
for name in split_names:

    # make ./data/jsonl/test or train
    jsonl_dir = os.path.join(data_dir, "jsonl")
    os.makedirs(jsonl_dir, exist_ok = True)
    jsonl_path = os.path.join(jsonl_dir, f"{name}.jsonl")

    # write jsonl
    with open(jsonl_path, "w", encoding="utf-8") as f:

        # store image to get image path
        img_dir = os.path.join(data_dir, "images", name)
        os.makedirs(img_dir, exist_ok = True)
        
        # for entry
        for index, entry in enumerate(vqa_raw[name]):

            # construct json messages
            unit = {}
            unit["messages"] = []
            q = entry["question"]
            a = entry["answer"]
            unit["messages"].append({"role": "user", "content": f"<image>\nQuestion: {q}"})
            unit["messages"].append({"role": "assistant", "content": f"{a}"})
            
            # store image in path and construct json image path
            image_raw = entry["image"]
            img_path = os.path.join(img_dir, f"{index:05d}.jpg" )
            image_raw.save(img_path)
            unit["images"] = [img_path]

            # write json line
            f.write(json.dumps(unit, ensure_ascii=False) + "\n")


