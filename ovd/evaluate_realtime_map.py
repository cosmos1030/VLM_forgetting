import os
import warnings
import json
from typing import List, Tuple, Dict, Any
import re
from collections import defaultdict
import random

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor
from PIL import Image
import numpy as np

# --------------------------------------------------------------------------
# 1. 유틸리티 함수 (프롬프트 수정)
# --------------------------------------------------------------------------

def build_fewshot_prompt(class_names: List[str]) -> str:
    """
    퓨샷(Few-shot) 예시를 포함하여 모델이 정확한 JSON 형식을 출력하도록 유도하는
    고성능 프롬프트를 생성합니다.
    """
    class_names_str = ", ".join(f"'{name}'" for name in class_names)
    
    # 예시를 위한 클래스 (제공된 클래스 리스트에서 2개 샘플링, 없으면 기본값 사용)
    sample_classes = random.sample(class_names, k=min(2, len(class_names)))
    if len(sample_classes) == 0:
        sample_classes = ['person', 'car']
    
    example_json = "[\n"
    if len(sample_classes) > 0:
        example_json += f'  {{\n    "class_name": "{sample_classes[0]}",\n    "bbox": [150, 220, 250, 580],\n    "score": 0.92\n  }}'
    if len(sample_classes) > 1:
        example_json += f',\n  {{\n    "class_name": "{sample_classes[1]}",\n    "bbox": [400, 310, 650, 450],\n    "score": 0.88\n  }}'
    example_json += "\n]"

    prompt = (
        "You are an expert object detection assistant. Your task is to find all instances of the specified objects in the image. "
        "Provide your answer in a strict JSON format within a ```json code block.\n\n"
        f"Objects to find: {class_names_str}\n\n"
        "Respond with a JSON list, where each object is a dictionary containing 'class_name', 'bbox' (in [x1, y1, x2, y2] format), and a 'score' (from 0.0 to 1.0).\n"
        "Here is an example for detecting 'person' and 'car':\n"
        "```json\n"
        f"{example_json}\n"
        "```\n\n"
        "If you don't find any of the requested objects, return an empty list `[]`.\n"
        "Now, analyze the following image and provide the JSON response."
    )
    return prompt

def save_first_batch(visuals: List[Tuple], filename: str):
    """결과 시각화를 위한 더미 함수."""
    print(f"\n--- Dummy save_first_batch ---")
    print(f"Visualizing first {len(visuals)} results. Would be saved to {filename}")
    if not visuals:
        print("No visuals to save.")
        return
    try:
        image, preds, text = visuals[0]
        print(f"Sample Image Size: {image.size}")
        print(f"Sample Predictions Count: {len(preds)}")
        if preds:
            print(f"First Prediction: {preds[0]}")
        print(f"Sample Raw Text Output:\n---\n{text}\n---")
    except Exception as e:
        print(f"Could not print sample visual info: {e}")
    print(f"-----------------------------\n")

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# 2. 평가자(Evaluator) 클래스
# --------------------------------------------------------------------------

class Evaluator:
    def __init__(self, hf_model_id: str, is_instruct: bool, gpu_ids: str = "0"):
        self.hf_model_id = hf_model_id
        self.multi_gpu = gpu_ids.lower() not in ("", "cpu", "-1")

        if self.multi_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            print(f"Using GPU(s): {gpu_ids}")
            self.device = torch.device("cuda")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")

        cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
        self.actual_model_type_from_config = cfg.model_type
        print(f"HF Config model_type: {self.actual_model_type_from_config} for {hf_model_id}")

        model_loader_map = {
            "qwen2_vl": "Qwen2VLForConditionalGeneration",
            "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration"
        }
        model_class_name = model_loader_map.get(self.actual_model_type_from_config)

        if model_class_name == "Qwen2VLForConditionalGeneration":
            from transformers import Qwen2VLForConditionalGeneration as VLModel
        elif model_class_name == "Qwen2_5_VLForConditionalGeneration":
            from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
        else:
            from transformers import AutoModelForCausalLM as VLModel

        print(f"Loading model with {VLModel.__name__}")
        self.model = VLModel.from_pretrained(
            hf_model_id,
            torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
            device_map="auto" if self.multi_gpu else None,
            trust_remote_code=True
        ).eval()

        # 이미지 토큰 수 고정을 위한 min/max pixels 설정 (384×384)
        self.processor = AutoProcessor.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            min_pixels=384*384,
            max_pixels=384*384
        )
        self.processor.tokenizer.padding_side = 'left'
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        if not self.multi_gpu and self.model.device.type != 'cpu':
            self.model.to(self.device)

    def _parse_model_output(self, text: str, categories: List[Dict]) -> List[Dict]:
        json_block_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if not json_block_match:
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1:
                json_string = text[start:end+1]
            else:
                return []
        else:
            json_string = json_block_match.group(1)

        try:
            data = json.loads(json_string.replace("'", '"'))
            if not isinstance(data, list):
                return []

            parsed = []
            cat_map = {cat['name'].lower(): cat['id'] for cat in categories}
            for item in data:
                if not isinstance(item, dict) or not all(k in item for k in ["class_name", "bbox", "score"]):
                    continue

                class_name = item["class_name"].lower()
                category_id = cat_map.get(class_name)
                if category_id is None:
                    continue
                
                bbox = item["bbox"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue

                try:
                    parsed.append({
                        "image_id": -1,
                        "category_id": int(category_id),
                        "bbox": [int(v) for v in bbox],
                        "score": float(item["score"]),
                    })
                except (ValueError, TypeError):
                    continue

            return parsed
        except json.JSONDecodeError:
            return []

    @torch.no_grad()
    def _predict_batch(self, imgs: List[Image.Image], prompt_text: str, categories: List[Dict]) -> Tuple[List[List[Dict]], List[str]]:
        # PIL로 강제 리사이즈 → 모델 권장 384×384
        resized = [img.resize((384, 384), resample=Image.BILINEAR) for img in imgs]
        rgb_imgs = [img.convert("RGB") for img in resized]

        messages = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]] * len(rgb_imgs)
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
        inputs = self.processor(text=texts, images=rgb_imgs, return_tensors="pt", padding=True).to(self.device)

        gen_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
            "use_cache": False  # 배치 간 캐시 오염 방지
        }
        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
        out_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        final_preds = [self._parse_model_output(text, categories) for text in out_texts]
        return final_preds, out_texts

    def _calculate_iou(self, box_gt: List[int], box_pred: List[int]) -> float:
        gt_x1, gt_y1, gt_w, gt_h = box_gt
        gt_x2, gt_y2 = gt_x1 + gt_w, gt_y1 + gt_h
        pred_x1, pred_y1, pred_x2, pred_y2 = box_pred
        inter_x1 = max(gt_x1, pred_x1)
        inter_y1 = max(gt_y1, pred_y1)
        inter_x2 = min(gt_x2, pred_x2)
        inter_y2 = min(gt_y2, pred_y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        if inter_area == 0:
            return 0.0
        gt_area = gt_w * gt_h
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        union_area = float(gt_area + pred_area - inter_area)
        return inter_area / union_area

    def _calculate_ap(self, gt_for_class: List[Dict], preds_for_class: List[Dict], iou_threshold: float) -> float:
        if not gt_for_class or not preds_for_class:
            return 0.0

        preds_for_class.sort(key=lambda x: x['score'], reverse=True)
        gt_by_image_id = defaultdict(list)
        for gt in gt_for_class:
            gt_by_image_id[gt['image_id']].append(gt)

        num_gt = len(gt_for_class)
        tp = np.zeros(len(preds_for_class))
        fp = np.zeros(len(preds_for_class))
        matched = defaultdict(set)

        for i, pred in enumerate(preds_for_class):
            img_id = pred['image_id']
            best_iou, best_idx = 0, -1
            for idx, gt in enumerate(gt_by_image_id.get(img_id, [])):
                if idx in matched[img_id]:
                    continue
                iou = self._calculate_iou(gt['bbox'], pred['bbox'])
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_iou >= iou_threshold:
                tp[i] = 1
                matched[img_id].add(best_idx)
            else:
                fp[i] = 1

        fp_cum = np.cumsum(fp)
        tp_cum = np.cumsum(tp)
        recalls = tp_cum / num_gt
        denom = tp_cum + fp_cum
        precisions = np.divide(tp_cum, denom, out=np.zeros_like(tp_cum), where=denom!=0)

        recalls = np.concatenate(([0.], recalls, [1.]))
        precisions = np.concatenate(([0.], precisions, [0.]))
        for i in range(len(precisions)-2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i+1])

        ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
        return ap

    def evaluate(self,
                 dataset_json_path: str,
                 image_folder_path: str,
                 output_dir: str,
                 batch_size: int = 8,
                 iou_threshold: float = 0.5,
                 eval_percentage: float = 100.0):
        print(f"Loading dataset from {dataset_json_path}...")
        with open(dataset_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        categories = data.get("categories", [])
        images_full = data.get("images", [])
        ann_full = data.get("annotations", [])

        if eval_percentage < 100.0:
            num = max(1, int(len(images_full) * (eval_percentage / 100.0)))
            images = images_full[:num]
            ids = {img['id'] for img in images}
            annotations = [ann for ann in ann_full if ann['image_id'] in ids]
            print(f"Evaluating on {len(images)} / {len(images_full)} images ({eval_percentage}%)")
        else:
            images = images_full
            annotations = ann_full
            print(f"Evaluating on all {len(images)} images")

        if not annotations:
            print("WARNING: No annotations in selected subset.")
        
        gt_by_cls = defaultdict(list)
        for ann in annotations:
            gt_by_cls[ann['category_id']].append(ann)

        class_names = [c['name'] for c in categories]
        id2name = {c['id']: c['name'] for c in categories}
        prompt = build_fewshot_prompt(class_names)

        all_preds = []
        preds_by_cls = defaultdict(list)
        visuals = []

        pbar = tqdm(total=len(images), desc=f"Processing (BS={batch_size})")
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            imgs, img_ids = [], []
            for info in batch:
                path = os.path.join(image_folder_path, info['file_name'])
                try:
                    img = Image.open(path)
                except:
                    continue
                imgs.append(img)
                img_ids.append(info['id'])
            if not imgs:
                pbar.update(len(batch))
                continue

            try:
                batch_preds, texts = self._predict_batch(imgs, prompt, categories)
            except Exception as e:
                print(f"Batch error at index {i}: {e}")
                pbar.update(len(batch))
                continue

            for j, preds in enumerate(batch_preds):
                img_id = img_ids[j]
                for p in preds:
                    p["image_id"] = img_id
                    all_preds.append(p)
                    preds_by_cls[p['category_id']].append(p)
            if not visuals:
                for j in range(len(imgs)):
                    visuals.append((imgs[j], batch_preds[j], texts[j]))

            pbar.update(len(batch))
        pbar.close()

        os.makedirs(output_dir, exist_ok=True)
        base = os.path.join(output_dir, f"{self.hf_model_id.replace('/', '_')}_perc{int(eval_percentage)}")

        with open(base + "_preds.json", 'w', encoding='utf-8') as f:
            json.dump(all_preds, f, indent=2, ensure_ascii=False)
        print("Predictions saved to", f.name)

        save_first_batch(visuals, base + "_vis.png")

        if not annotations:
            return

        print("Calculating mAP...")
        ap_per_cls = {}
        for cid, gts in gt_by_cls.items():
            ap_per_cls[cid] = self._calculate_ap(gts, preds_by_cls.get(cid, []), iou_threshold)

        valid_aps = [ap for ap in ap_per_cls.values() if not np.isnan(ap)]
        mAP = float(np.mean(valid_aps)) if valid_aps else 0.0

        report = [
            f"mAP: {mAP:.4f}",
            "AP per class:"
        ]
        for cid, ap in ap_per_cls.items():
            report.append(f"- {id2name.get(cid)}: {0.0 if np.isnan(ap) else ap:.4f} "
                          f"(GT {len(gt_by_cls[cid])}, Pred {len(preds_by_cls.get(cid, []))})")

        print("\n".join(report))
        with open(base + "_scores.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        print("Scores saved to", f.name)


# --------------------------------------------------------------------------
# 3. 메인 실행 블록
# --------------------------------------------------------------------------
if __name__ == "__main__":
    HF_MODEL_ID = "omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321"
    IS_INSTRUCT = True
    GPU_IDS = "0,1,2"
    BATCH_SIZE = 1
    EVAL_PERCENTAGE = 0.1  # 빠른 테스트
    JSON_DATASET_PATH = "ovdeval/celebrity.json"
    IMAGE_DIR_PATH = "ovdeval/celebrity"
    OUTPUT_RESULTS_DIR = "bbox_results"

    try:
        import accelerate
    except ImportError:
        print("Warning: accelerate 라이브러리를 설치하세요: pip install accelerate")
    try:
        import numpy
    except ImportError:
        print("Error: numpy가 필요합니다: pip install numpy")
        exit()

    print("="*60)
    print(f"Model: {HF_MODEL_ID}")
    print(f"GPUs: {GPU_IDS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Dataset: {JSON_DATASET_PATH}")
    print(f"Eval %: {EVAL_PERCENTAGE}%")
    print("="*60)

    evaluator = Evaluator(HF_MODEL_ID, IS_INSTRUCT, GPU_IDS)
    evaluator.evaluate(
        dataset_json_path=JSON_DATASET_PATH,
        image_folder_path=IMAGE_DIR_PATH,
        output_dir=OUTPUT_RESULTS_DIR,
        batch_size=BATCH_SIZE,
        iou_threshold=0.5,
        eval_percentage=EVAL_PERCENTAGE
    )
