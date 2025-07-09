import os
import warnings
import json
from typing import List, Tuple, Dict, Any
import re

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor
from PIL import Image

# --------------------------------------------------------------------------
# 1. 유틸리티 함수 (prompt_utils.py가 없어도 되도록 포함)
# --------------------------------------------------------------------------

def build_prompt(class_names: List[str]) -> str:
    """주어진 클래스 이름에 대한 프롬프트를 생성합니다."""
    return (
        "Detect objects and output their bounding boxes and confidence scores in the COCO detection format "
        "(a list of dictionaries, each with 'image_id', 'score', 'category_id', 'bbox'). "
        f"Focus on the classes provided: {', '.join(class_names)}."
    )

def save_first_batch(visuals: List[Tuple], filename: str):
    """결과 시각화를 위한 더미 함수."""
    # 실제 시각화 로직이 필요하다면 여기에 구현해야 합니다. (e.g., matplotlib, PIL.ImageDraw)
    print(f"--- Dummy save_first_batch ---")
    print(f"Visualizing first {len(visuals)} results. Would be saved to {filename}")
    try:
        image, preds, text = visuals[0]
        print(f"Sample Image Size: {image.size}")
        print(f"Sample Predictions: {preds[:2]}")
        print(f"Sample Raw Text Output: {text[:150]}...")
    except Exception as e:
        print(f"Could not print sample visual info: {e}")
    print(f"-----------------------------")


def save_csv(total: int, correct: int, preds: List[Tuple], acc_filename: str, pred_filename: str):
    """결과 저장을 위한 더미 함수."""
    print(f"--- Dummy save_csv ---")
    print(f"Total: {total}, Correct: {correct}")
    print(f"Accuracy and predictions would be saved to {acc_filename} and {pred_filename}")
    print(f"----------------------")


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
            # Fallback for other models like Qwen-VL-Chat or generic Causal LMs
            from transformers import AutoModelForCausalLM as VLModel
        
        print(f"Loading model with {VLModel.__name__}")
        self.model = VLModel.from_pretrained(
            hf_model_id,
            torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
            device_map="auto" if self.multi_gpu else None,
            trust_remote_code=True
        ).eval()

        self.processor = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True)

        # 배치 처리를 위한 왼쪽 패딩 설정 (경고 메시지 해결)
        print("Setting tokenizer padding side to 'left'.")
        self.processor.tokenizer.padding_side = 'left'
        
        # 모델의 토크나이저가 pad_token을 가지고 있는지 확인하고, 없으면 eos_token으로 설정
        if self.processor.tokenizer.pad_token is None:
            print("Tokenizer has no pad_token, setting it to eos_token.")
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        if not self.multi_gpu and self.model.device.type != 'cpu':
             self.model.to(self.device)

    @torch.no_grad()
    def _predict_batch(self, imgs: List[Image.Image], prompt_text: str, categories: List[Dict]) -> Tuple[List[List[Dict]], List[str]]:
        rgb_imgs = [img.convert("RGB") if img.mode != "RGB" else img for img in imgs]
        batch_size = len(rgb_imgs)

        messages = [
            [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt_text}]}]
            for img in rgb_imgs
        ]
        
        texts_with_placeholders = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        inputs = self.processor(
            text=texts_with_placeholders,
            images=rgb_imgs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        if 'input_ids' not in inputs or inputs['input_ids'].shape[-1] == 0:
            print("Warning: Empty input_ids, skipping batch.")
            return [[]] * batch_size, [""] * batch_size

        gen_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
            "use_cache": True
        }

        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        input_ids_len = inputs['input_ids'].shape[1]
        trimmed_generated_ids = generated_ids[:, input_ids_len:]
        
        out_texts_full = self.processor.batch_decode(
            trimmed_generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        final_json_preds: List[List[Dict]] = []
        for out_text_full in out_texts_full:
            parsed_outputs_for_current_image = []
            
            json_block_match = re.search(r"```json\s*(\[.*?\])\s*```", out_text_full, re.DOTALL)
            json_string_to_parse = None

            if json_block_match:
                json_string_to_parse = json_block_match.group(1)
            else:
                json_start_idx = out_text_full.find("[{")
                json_end_idx = out_text_full.rfind("}]")
                if json_start_idx != -1 and json_end_idx != -1 and json_end_idx > json_start_idx:
                    json_string_to_parse = out_text_full[json_start_idx : json_end_idx + 2]
            
            if json_string_to_parse:
                try:
                    json_string_to_parse = json_string_to_parse.replace("'", '"')
                    raw_parsed_data = json.loads(json_string_to_parse)
                    if not isinstance(raw_parsed_data, list):
                        raise ValueError("Parsed JSON is not a list.")

                    for item in raw_parsed_data:
                        converted_item = {}
                        if "bbox" in item and isinstance(item["bbox"], list) and len(item["bbox"]) == 4:
                            converted_item["bbox"] = [int(v) for v in item["bbox"]]
                        else:
                            continue

                        converted_item["score"] = float(item.get("score", 0.85))

                        if "category_id" in item:
                            converted_item["category_id"] = int(item["category_id"])
                        elif "label" in item:
                            category_name_from_model = str(item["label"]).lower()
                            found_category_id = -1
                            for cat in categories:
                                if cat["name"].lower() == category_name_from_model:
                                    found_category_id = cat["id"]
                                    break
                            converted_item["category_id"] = found_category_id
                        else:
                            converted_item["category_id"] = -1
                        converted_item["image_id"] = -1
                        parsed_outputs_for_current_image.append(converted_item)

                except (json.JSONDecodeError, ValueError) as e:
                    parsed_outputs_for_current_image = self._fallback_parse_bbox_regex(out_text_full, categories)
            else:
                parsed_outputs_for_current_image = self._fallback_parse_bbox_regex(out_text_full, categories)
            final_json_preds.append(parsed_outputs_for_current_image)
        return final_json_preds, out_texts_full

    def _fallback_parse_bbox_regex(self, text: str, categories: List[Dict]) -> List[Dict]:
        parsed_results = []
        # This regex is an example. You may need to adjust it for your model's specific output format.
        pattern = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]")
        for match in pattern.finditer(text):
            try:
                bbox = [int(coord) for coord in match.groups()]
                # This fallback is simple and may not correctly infer category or score.
                parsed_results.append({
                    "image_id": -1,
                    "score": 0.5, # Default score
                    "category_id": -1, # Unknown category
                    "bbox": bbox
                })
            except Exception:
                continue
        return parsed_results

    def evaluate(self, dataset_json_path: str, image_folder_path: str, output_dir: str, batch_size: int = 8):
        print(f"Loading dataset from {dataset_json_path}...")
        try:
            with open(dataset_json_path, 'r', encoding='utf-8') as f:
                dataset_data = json.load(f)
            print("Dataset JSON loaded successfully.")
        except Exception as e:
            print(f"Error loading or parsing dataset JSON: {e}")
            return

        categories = dataset_data.get("categories", [])
        images_info = dataset_data.get("images", [])
        
        class_names = [cat['name'] for cat in categories]
        classification_prompt = build_prompt(class_names)
        
        all_predicted_bboxes: List[Dict[str, Any]] = []
        first_batch_visual: List[Tuple] = []
        
        pbar = tqdm(total=len(images_info), desc=f"Processing Images (Batch Size: {batch_size})")

        for i in range(0, len(images_info), batch_size):
            batch_infos = images_info[i:i + batch_size]
            batch_imgs = []
            batch_img_ids = []
            
            for img_info in batch_infos:
                image_id = img_info["id"]
                file_name = img_info["file_name"]
                full_image_path = os.path.join(image_folder_path, file_name)

                try:
                    # Use a 'with' statement to ensure files are closed properly
                    with Image.open(full_image_path) as img:
                        batch_imgs.append(img.copy()) # Use copy to avoid issues with lazy loading
                    batch_img_ids.append(image_id)
                except FileNotFoundError:
                    # print(f"Skipping missing image: {full_image_path}")
                    pbar.update(1)
                except Exception as e:
                    # print(f"Error loading image {full_image_path}: {e}. Skipping.")
                    pbar.update(1)
            
            if not batch_imgs:
                continue

            try:
                predicted_bboxes_batch, full_text_outputs_batch = self._predict_batch(
                    batch_imgs, classification_prompt, categories
                )

                for j, predicted_bboxes_for_one_image in enumerate(predicted_bboxes_batch):
                    current_image_id = batch_img_ids[j]
                    for bbox_pred in predicted_bboxes_for_one_image:
                        bbox_pred["image_id"] = current_image_id
                        if "bbox" in bbox_pred and isinstance(bbox_pred["bbox"], list):
                            bbox_pred["bbox"] = [int(x) for x in bbox_pred["bbox"]]
                        if "score" in bbox_pred:
                            bbox_pred["score"] = float(bbox_pred["score"])
                        all_predicted_bboxes.append(bbox_pred)

                if i == 0:
                    for j in range(len(batch_imgs)):
                        first_batch_visual.append((batch_imgs[j], predicted_bboxes_batch[j], full_text_outputs_batch[j]))
            except Exception as e:
                print(f"Error during batch prediction: {e}")

            pbar.update(len(batch_imgs))

        pbar.close()

        os.makedirs(output_dir, exist_ok=True)
        
        model_name_safe = self.hf_model_id.replace('/', '_').replace('-', '_')
        dataset_name_safe = os.path.basename(dataset_json_path).replace('.json', '')
        output_prefix = os.path.join(output_dir, f"{model_name_safe}_{dataset_name_safe}_bbox_predictions")

        final_output_path = f"{output_prefix}.json"
        try:
            with open(final_output_path, 'w', encoding='utf-8') as outfile:
                json.dump(all_predicted_bboxes, outfile, indent=2, ensure_ascii=False)
            print(f"\nAll predicted bboxes saved to {final_output_path}")
        except Exception as e:
            print(f"Error saving all predicted bboxes to file: {e}")

        save_first_batch(first_batch_visual, f"{output_prefix}_visual_first_batch.png")

        print(f"▶ Finished {self.hf_model_id} on {dataset_name_safe}. Processed {len(images_info)} images.")
        print(f"Total detected objects: {len(all_predicted_bboxes)}. Results saved in {output_dir}")

# --------------------------------------------------------------------------
# 3. 메인 실행 블록
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 수정할 파라미터 ---
    # HF_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
    HF_MODEL_ID = "cosmos1030/Qwen2.5_VL-3B-rec-SFT"
    IS_INSTRUCT = True
    GPU_IDS = "3,4,5"  # 사용할 GPU ID를 콤마로 구분하여 지정
    BATCH_SIZE = 4   # GPU 메모리에 맞춰 배치 크기 조절 (VRAM이 크면 늘리고, 작으면 줄이세요)

    JSON_DATASET_PATH = "ovdeval/celebrity.json" # 실제 데이터셋 경로로 수정
    IMAGE_DIR_PATH = "ovdeval/celebrity"       # 실제 이미지 폴더 경로로 수정
    OUTPUT_RESULTS_DIR = "bbox_results"
    # --- 파라미터 수정 끝 ---

    # 필수 라이브러리 설치 확인
    try:
        import accelerate
    except ImportError:
        print("Warning: 'accelerate' library not found. For multi-GPU, please run: pip install accelerate")


    print("="*60)
    print(f"Starting Evaluation")
    print(f"Model: {HF_MODEL_ID}")
    print(f"GPUs: {GPU_IDS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Dataset: {JSON_DATASET_PATH}")
    print("="*60)

    evaluator = Evaluator(
        hf_model_id=HF_MODEL_ID,
        is_instruct=IS_INSTRUCT,
        gpu_ids=GPU_IDS
    )

    evaluator.evaluate(
        dataset_json_path=JSON_DATASET_PATH,
        image_folder_path=IMAGE_DIR_PATH,
        output_dir=OUTPUT_RESULTS_DIR,
        batch_size=BATCH_SIZE
    )