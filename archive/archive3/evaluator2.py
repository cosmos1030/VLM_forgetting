import os
import warnings
from typing import List, Tuple
import re # 정규 표현식 사용을 위해 추가

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
from PIL import Image

from prompt_utils import build_prompt, save_first_batch, save_csv

warnings.filterwarnings("ignore")


class Evaluator:
    def __init__(self, hf_model_id: str, is_instruct: bool, gpu_ids: str = "0"):
        self.hf_model_id = hf_model_id
        self.multi_gpu = gpu_ids.lower() not in ("", "cpu", "-1")
        if self.multi_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            print("Using GPU(s):", gpu_ids)
        else:
            print("Using CPU")

        cfg = AutoConfig.from_pretrained(hf_model_id)
        self.actual_model_type_from_config = cfg.model_type
        print(f"HF Config model_type: {self.actual_model_type_from_config} for {hf_model_id}")

        # self.is_visualthinker_standalone_model = (hf_model_id == "turningpoint-ai/VisualThinker-R1-Zero" or hf_model_id == "Qwen/Qwen2-VL-2B" or hf_model_id == "cosmos1030/Qwen2_VL-2B-SFT_revised2" or hf_model_id == "cosmos1030/Qwen2_VL-2B-SFT")

        if is_instruct==0:
            print(f"Loading {hf_model_id} using non instruct model logic.")
            self.model = AutoModelForImageTextToText.from_pretrained(
                hf_model_id,
                torch_dtype="auto",
                device_map="auto" if self.multi_gpu else None
            ).eval()
            self.model_family = "qwen_noninstruct_like"
        elif self.actual_model_type_from_config == "qwen2_vl" or \
             "qwen2-vl" in hf_model_id.lower() or \
             "qwen2vl" in hf_model_id.lower():
            print(f"Loading {hf_model_id} using instruct model logic.")
            from transformers import Qwen2VLForConditionalGeneration as VLModel
            self.model = VLModel.from_pretrained(
                hf_model_id,
                torch_dtype=torch.float16 if self.multi_gpu else torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                device_map="auto" if self.multi_gpu else None
            ).eval()
            self.model_family = "qwen_instruct_like"
        elif self.actual_model_type_from_config == "qwen2_5_vl":
            print(f"Loading {hf_model_id} as Qwen2.5-VL family.")
            from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
            self.model = VLModel.from_pretrained(
                hf_model_id,
                torch_dtype=torch.float16 if self.multi_gpu else torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                device_map="auto" if self.multi_gpu else None
            ).eval()
            self.model_family = "qwen_instruct_like"
        else:
            raise ValueError(f"[Evaluator] Unsupported hf_model_id or model_type: {hf_model_id} / {self.actual_model_type_from_config}")

        self.processor = AutoProcessor.from_pretrained(hf_model_id)
        # print(self.processor)
        # print(self.processor.image_processor)
        # print(self.processor.tokenizer)

        # print(f"Using processor: {type(self.processor).__name__}")


    @torch.no_grad()
    def _predict_one(self, img: Image.Image, prompt_text: str, all_class_names: List[str]) -> str:
        if img.mode != "RGB":
            img = img.convert("RGB")

        inputs = None

        if self.model_family == "qwen_noninstruct_like":
            templated_user_assistant_prompt = f"User: {prompt_text}\nAssistant:"
            text_from_template = self.processor.apply_chat_template(
                [{"type": "image", "image": "image"},
                 {"type": "text", "text": templated_user_assistant_prompt}],
                tokenize=False,
                add_generation_prompt=False
            )
            inputs = self.processor(
                text=text_from_template,
                images=img,
                padding="longest",
                return_tensors="pt"
            ).to(device=self.model.device)

        elif self.model_family == "qwen_instruct_like":
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": img}, 
                    {"type": "text", "text": prompt_text}
                ]}
            ]
            text_with_placeholders = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text_with_placeholders],
                images=[img],
                padding=True,
                return_tensors="pt"
            ).to(device=self.model.device)
        else:
            raise ValueError(f"Unknown model family for prediction: {self.model_family}")

        if 'input_ids' not in inputs or inputs['input_ids'].shape[-1] == 0:
            # print(f"CRITICAL ERROR: input_ids is missing or sequence length is 0 for model family {self.model_family}!")
            # text_ref = text_from_template if self.model_family == 'visualthinker_standalone' else text_with_placeholders
            # print(f"  text input to processor was based on: '{text_ref}'")
            # print(f"  input_ids shape: {inputs.get('input_ids', 'N/A')}")
            return "error_empty_input_ids"

        gen_kwargs = {
            "max_new_tokens": 10, 
            "do_sample": False,
            "use_cache": True 
        }

        generated_ids = self.model.generate(**inputs, **gen_kwargs)

        trimmed_generated_ids = []
        for i_input_ids, i_generated_ids_full in zip(inputs['input_ids'], generated_ids):
            if len(i_generated_ids_full) > len(i_input_ids):
                 trimmed_generated_ids.append(i_generated_ids_full[len(i_input_ids):])
            else:
                 trimmed_generated_ids.append(torch.tensor([], dtype=torch.long, device=i_input_ids.device))
        
        out_text_full = self.processor.batch_decode(
            trimmed_generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        # print(out_text_full)
        
        out_text_stripped = out_text_full.strip()
        # print(f"DEBUG: Full generated text: '{out_text_stripped}'")

        cleaned_output_lower = out_text_stripped.lower()
        
        found_class = "unknown"
        best_match_pos = float('inf')

        for cls_name in all_class_names:
            lower_cls_name = cls_name.lower()
            match = re.search(r"\b" + re.escape(lower_cls_name) + r"\b", cleaned_output_lower)
            if match:
                pos = match.start()
                if pos < best_match_pos:
                    best_match_pos = pos
                    found_class = lower_cls_name
        
        if found_class != "unknown":
            # print(f"DEBUG: Extracted class: '{found_class}' from '{out_text_stripped}'")
            return found_class, out_text_full

        else:
            predicted_words = cleaned_output_lower.split()
            fallback_pred = "unknown"
            if predicted_words:
                fallback_pred = predicted_words[0].lower()
            # print(f"DEBUG: No class found in '{out_text_stripped}'. Fallback to first word: '{fallback_pred}'")
            # return fallback_pred
            return fallback_pred, out_text_full


    def evaluate(self,
                 dataset,
                 loader,
                 class_names: List[str],
                 out_dir: str):
        classification_prompt = build_prompt(class_names)
        correct, total = 0, 0
        preds: List[Tuple[str, str]] = []
        first_batch_visual: List[Tuple] = []

        pbar = tqdm(total=len(dataset), desc="Evaluating")
        for batch_idx, (imgs, labels) in enumerate(loader):
            for img, lbl in zip(imgs, labels):
                # pred = self._predict_one(img, classification_prompt, class_names) 
                pred, full_text = self._predict_one(img, classification_prompt, class_names)

                
                # print("Debug:"+pred) 
                
                if pred == "error_empty_input_ids":
                    print(f"Skipping image due to input_ids error. GT: {class_names[lbl].lower()}")
                    gt = class_names[lbl].lower() 
                else:
                    gt = class_names[lbl].lower()

                # preds.append((gt, pred))
                preds.append((gt, pred, full_text))

                if pred != "error_empty_input_ids":
                    correct += int(pred == gt)
                    total += 1

                if batch_idx == 0 and len(first_batch_visual) < 10:
                    first_batch_visual.append((img, gt, pred))
                
                current_acc_str = f"{100*correct/total:.2f}% ({correct}/{total})" if total > 0 else "N/A"
                pbar.set_postfix_str(f"Acc={current_acc_str}")
                pbar.update(1)
            break
        pbar.close()

        os.makedirs(out_dir, exist_ok=True)
        dataset_name_for_file = dataset.__class__.__name__.lower()
        if hasattr(dataset, 'name') and dataset.name:
            dataset_name_for_file = dataset.name
        elif hasattr(dataset, 'root') and dataset.root and os.path.basename(dataset.root):
            dataset_name_for_file = os.path.basename(dataset.root).lower()
            if not dataset_name_for_file and hasattr(dataset, '__class__'):
                 dataset_name_for_file = dataset.__class__.__name__.lower()

        run_name_prefix = self.hf_model_id.replace('/', '_')
        output_prefix = os.path.join(out_dir, f"{run_name_prefix}_{dataset_name_for_file}")

        save_first_batch(first_batch_visual, f"{output_prefix}_visual_first_batch.png")
        save_csv(total, correct, preds, f"{output_prefix}_accuracy.csv", f"{output_prefix}_predictions.csv")

        acc = 100 * correct / total if total > 0 else 0
        print(f"▶ Finished {self.hf_model_id} on {dataset_name_for_file} — Accuracy {acc:.2f}% ({correct}/{total})")
        print(f"Results saved in {out_dir}")