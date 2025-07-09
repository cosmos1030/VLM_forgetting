import os
import json
import re
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw
import torch
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

# (build_fewshot_prompt, parse_output 함수는 변경 없이 그대로 사용)
def build_fewshot_prompt(class_names: List[str]) -> str:
    class_names_str = ", ".join(f"'{name}'" for name in class_names)
    sample = class_names[:2] if len(class_names)>=2 else class_names or ['person','car']
    example = "[\n"
    for i,name in enumerate(sample):
        bbox = [150,220,250,580] if i==0 else [400,310,650,450]
        score = 0.92 if i==0 else 0.88
        example += f'  {{ "class_name": "{name}", "bbox": {bbox}, "score": {score} }}'
        if i==0 and len(sample)>1: example += ",\n"
    example += "\n]"
    prompt = (
        "You are an expert object detection assistant.\n"
        f"Objects to find: {class_names_str}\n\n"
        "Provide your answer in a JSON code block like:\n"
        "```json\n" + example + "\n```\n"
        "Now analyze the image and return the JSON."
    )
    return prompt

def parse_output(text: str) -> List[Dict]:
    m = re.search(r"```json\s*(\[.*?\])\s*```", text, re.S)
    js = m.group(1) if m else text[text.find('['):text.rfind(']')+1]
    try:
        return json.loads(js.replace("'",'"'))
    except:
        return []


if __name__ == "__main__":
    # Settings
    HF_MODEL_ID  = "omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321"
    DATA_JSON    = "ovdeval/celebrity.json"
    IMAGE_DIR    = "ovdeval/celebrity"
    INPUT_SIZE   = 384
    
    NUM_TO_TEST = 10
    RESULT_DIR = "results"
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Load dataset
    with open(DATA_JSON, 'r', encoding='utf-8') as f:
        ds = json.load(f)

    # Load model & processor
    print("Loading model and processor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(
        HF_MODEL_ID, trust_remote_code=True,
        min_pixels=INPUT_SIZE*INPUT_SIZE, max_pixels=INPUT_SIZE*INPUT_SIZE
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        HF_MODEL_ID,
        torch_dtype=torch.bfloat16 if device.type == 'cuda' else torch.float32,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    print("Model and processor loaded.")

    images_to_process = ds["images"][:NUM_TO_TEST]
    for i, image_info in enumerate(images_to_process):
        img_id    = image_info["id"]
        file_name = image_info["file_name"]
        img_path  = os.path.join(IMAGE_DIR, file_name)
        
        # ✨ --- 이 부분이 핵심 변경 사항입니다 ---
        # JSON의 'text' 필드를 사용하여 해당 이미지에 대한 특정 쿼리를 만듭니다.
        # 만약 'text' 필드가 없거나 비어있으면, 모든 카테고리를 찾는 이전 방식으로 동작합니다.
        query_classes = image_info.get("text")
        if not query_classes:
            print(f"Warning: No specific 'text' query for {file_name}. Falling back to all categories.")
            query_classes = [c["name"] for c in ds["categories"]]

        # 해당 이미지에 대한 동적 프롬프트 생성
        prompt = build_fewshot_prompt(query_classes)
        # --- 변경 사항 끝 ---

        print(f"\n[{i+1}/{NUM_TO_TEST}] Processing image: {file_name}")
        print(f"Querying for: {query_classes}")

        try:
            orig_img  = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file not found at {img_path}. Skipping.")
            continue

        # (이하 나머지 코드는 동일)
        msg = [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
        text = proc.apply_chat_template([msg], tokenize=False, add_generation_prompt=True)[0]
        inputs = proc(text=[text], images=[orig_img], return_tensors="pt", padding=True).to(device)

        out = model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True)
        gen = out[:, inputs['input_ids'].shape[1]:]
        decoded = proc.batch_decode(gen, skip_special_tokens=True)[0]
        preds = parse_output(decoded)
        print(f"Found {len(preds)} objects.")

        gts = [ann["bbox"] for ann in ds["annotations"] if ann["image_id"] == img_id]

        draw_img = orig_img.copy()
        draw = ImageDraw.Draw(draw_img)

        for x, y, w, h in gts:
            draw.rectangle([x, y, x+w, y+h], outline="red", width=3)

        orig_w, orig_h = orig_img.size
        target_size = INPUT_SIZE
        scale = target_size / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_x = (target_size - new_w) // 2
        pad_y = (target_size - new_h) // 2

        for item in preds:
            if "bbox" not in item: continue
            x1, y1, x2, y2 = item["bbox"]
            x1o = (x1 - pad_x) / scale
            y1o = (y1 - pad_y) / scale
            x2o = (x2 - pad_x) / scale
            y2o = (y2 - pad_y) / scale
            draw.rectangle([x1o, y1o, x2o, y2o], outline="lime", width=2)
            label = f'{item.get("class_name", "N/A")}: {item.get("score", 0):.2f}'
            draw.text((x1o, y1o - 10), label, fill="lime")

        plt.figure(figsize=(8, 8))
        plt.imshow(draw_img)
        plt.axis("off")
        plt.title(f"GT (red) vs Pred (green) on {file_name}", fontsize=12)
        
        save_path = os.path.join(RESULT_DIR, f"result_{file_name}")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Result saved to {save_path}")
        
        plt.close()

    print("\nAll processing finished.")