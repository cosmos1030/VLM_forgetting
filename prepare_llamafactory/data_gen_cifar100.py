# 파일명: data_gen_cifar10_messages.py
import os
import json
from torchvision.datasets import CIFAR10, CIFAR100
from PIL import Image
from tqdm import tqdm

def main():
    raw_root = './data'
    cifar = CIFAR100(root=raw_root, train=True, download=False)
    classes = cifar.classes
    classes_str = ', '.join(classes)

    images_dir = 'cifar100'
    os.makedirs(images_dir, exist_ok=True)
    
    all_entries = []

    for idx, (img, label) in enumerate(tqdm(cifar, desc='Generating JSON')):
        fname = f'{idx:05d}.png'
        img_path = os.path.join(images_dir, fname)
        img.save(img_path)

        prompt = (
            f"<image>"
            f"What is the main object in this image? "
            f"Choose one word from these options: {classes_str}."
        )

        entry = {
            "messages": [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": "The answer is "+classes[label]}
            ],
            "images": [img_path]
        }
        all_entries.append(entry)

    out_path = './mllm_cifar10.json'
    with open(out_path, 'w', encoding='utf-8') as fout:
        json.dump(all_entries, fout, ensure_ascii=False, indent=2)

    print(f"✅ Generated {len(all_entries)} entries → {out_path}")

if __name__ == "__main__":
    main()
