import os
import json
from torchvision.datasets import CIFAR10, CIFAR100

# Specify the dataset: 'cifar10' or 'cifar100'
dataset_name = 'cifar100'

def dump_split(split: str, dataset_name: str):
    """
    Saves a split of the specified dataset (CIFAR10/CIFAR100) as images
    and a corresponding JSONL metadata file.

    Args:
        split (str): The dataset split to process, e.g., "train" or "test".
        dataset_name (str): The name of the dataset, 'cifar10' or 'cifar100'.
    """
    print(f"Processing '{split}' split for {dataset_name.upper()}...")

    # --- 1. Select and load the correct dataset ---
    if dataset_name == 'cifar10':
        dataset_class = CIFAR10
    elif dataset_name == 'cifar100':
        dataset_class = CIFAR100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'cifar10' or 'cifar100'.")

    # Load the dataset without any transforms to get PIL images
    is_train = (split == "train")
    ds = dataset_class(root="./data", train=is_train, download=True)
    class_names = ds.classes

    # --- 2. Prepare output directories and files ---
    out_img_dir = f"images/{dataset_name}/{split}"
    os.makedirs(out_img_dir, exist_ok=True)
    out_jsonl_path = f"data/{dataset_name}_{split}.jsonl"

    # --- 3. Dynamically create the prompt for the JSONL file ---
    options_str = ", ".join(class_names)
    human_prompt = f"<image>What is the main object in this image? Choose one word from these options: {options_str}."

    # --- 4. Process and save each item in the dataset ---
    with open(out_jsonl_path, "w", encoding='utf-8') as fw:
        for idx, (img, label) in enumerate(ds):
            # Save the image
            image_filename = f"{split}_{idx:05d}.png"
            img.save(os.path.join(out_img_dir, image_filename))

            # Create the JSONL entry with the updated response format
            entry = {
                "id": f"{dataset_name}-{split}-{idx}",
                "image": f"{split}/{image_filename}",
                "conversations": [
                    {
                        "from": "human",
                        "value": human_prompt
                    },
                    {
                        "from": "gpt",
                        "value": class_names[label]
                    }
                ]
            }
            # Write the JSON object to the file, followed by a newline
            fw.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # Create the main data directory
    os.makedirs("data", exist_ok=True)

    print(f"--- Starting dataset conversion for {dataset_name.upper()} ---")
    
    # Process both the training and testing splits
    dump_split("train", dataset_name)
    dump_split("test", dataset_name)
    
    print(f"\n✅ {dataset_name.upper()} JSONL and images are ready.")
    print(f"   - Images: ./images/{dataset_name}/")
    print(f"   - JSONL files: ./data/{dataset_name}_train.jsonl, ./data/{dataset_name}_test.jsonl")

