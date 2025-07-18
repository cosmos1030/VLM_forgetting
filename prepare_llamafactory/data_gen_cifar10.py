# 파일명: data_gen_cifar10_messages.py
import os
import json
from torchvision.datasets import CIFAR10, CIFAR100
from PIL import Image
from tqdm import tqdm

def main():
    # 1) 원본 다운로드 & 저장 경로
    raw_root = './data'
    cifar = CIFAR100(root=raw_root, train=True, download=False)
    classes = cifar.classes
    classes_str = ', '.join(classes)

    # 2) 이미지 저장 폴더
    images_dir = 'cifar100'
    os.makedirs(images_dir, exist_ok=True)
    
    # 3) 전체 결과를 담을 리스트
    all_entries = []

    for idx, (img, label) in enumerate(tqdm(cifar, desc='Generating JSON')):
        # 3-1) 이미지 파일로 저장
        fname = f'{idx:05d}.png'
        img_path = os.path.join(images_dir, fname)
        img.save(img_path)

        # 3-2) 사용자 메시지 + 어시스턴트 메시지(JSON 포맷 포함)
        prompt = (
            f"<image>"
            f"What is the main object in this image? "
            f"Choose one word from these options: {classes_str}."
        )
        # assistant쪽 content를 ```json …``` 로 래핑
        label_json = [{
            "label": "The answer is "+classes[label]
        }]
        assistant_content = "```json\n" + json.dumps(label_json, ensure_ascii=False) + "\n```"

        entry = {
            "messages": [
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": assistant_content}
            ],
            "images": [img_path]
        }
        all_entries.append(entry)

    # 4) 하나의 JSON 파일로 덤프
    out_path = './mllm_cifar100.json'
    with open(out_path, 'w', encoding='utf-8') as fout:
        json.dump(all_entries, fout, ensure_ascii=False, indent=2)

    print(f"✅ Generated {len(all_entries)} entries → {out_path}")

if __name__ == "__main__":
    main()
