import os
import warnings
from typing import List, Tuple
import re

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from prompt_utils import build_prompt, save_first_batch, save_csv

warnings.filterwarnings("ignore")


class Evaluator:
    def __init__(self, hf_model_id: str, is_instruct: bool, gpu_ids: str = "0"):
        """
        Initializes the Evaluator with a specified Hugging Face model.
        """
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

        # Logic to load the correct model architecture based on config or name
        if not is_instruct:
            print(f"Loading {hf_model_id} using non-instruct model logic.")
            self.model = AutoModelForImageTextToText.from_pretrained(
                hf_model_id,
                torch_dtype="auto",
                device_map="auto" if self.multi_gpu else None
            ).eval()
            self.model_family = "qwen_noninstruct_like"
        elif self.actual_model_type_from_config == "qwen2_vl" or \
             "qwen2-vl" in hf_model_id.lower() or \
             "qwen2vl" in hf_model_id.lower():
            print(f"Loading {hf_model_id} using Qwen2-VL instruct model logic.")
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

    @torch.no_grad()
    def _predict_batch(self, imgs: List[Image.Image], prompt_text: str, all_class_names: List[str]) -> Tuple[List[str], List[str]]:
        """
        Processes a batch of images and returns a batch of predictions.
        """
        # Ensure all images are in RGB format
        rgb_imgs = [img.convert("RGB") if img.mode != "RGB" else img for img in imgs]
        batch_size = len(rgb_imgs)

        inputs = None

        if self.model_family == "qwen_noninstruct_like":
            templated_user_assistant_prompt = f"User: {prompt_text}\nAssistant:"
            # Create a list of templated prompts, one for each image
            texts_from_template = [
                self.processor.apply_chat_template(
                    [{"type": "image", "image": "image"}, {"type": "text", "text": templated_user_assistant_prompt}],
                    tokenize=False, add_generation_prompt=False
                )
            ] * batch_size
            
            inputs = self.processor(
                text=texts_from_template,
                images=rgb_imgs,
                padding="longest",
                return_tensors="pt"
            ).to(device=self.model.device)

        elif self.model_family == "qwen_instruct_like":
            # Build a batch of messages
            messages = [
                [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt_text}]}]
                for img in rgb_imgs
            ]
            
            # Apply chat template to each message list in the batch
            texts_with_placeholders = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in messages
            ]

            inputs = self.processor(
                text=texts_with_placeholders,
                images=rgb_imgs,
                padding=True,
                return_tensors="pt"
            ).to(device=self.model.device)
        else:
            raise ValueError(f"Unknown model family for prediction: {self.model_family}")

        if 'input_ids' not in inputs or inputs['input_ids'].shape[-1] == 0:
            return ["error_empty_input_ids"] * batch_size, [""] * batch_size

        gen_kwargs = {
            "max_new_tokens": 10,
            "do_sample": False,
            "use_cache": True
        }

        # Generate responses for the entire batch at once
        generated_ids = self.model.generate(**inputs, **gen_kwargs)

        # Trim the input tokens from the generated tokens
        input_ids_len = inputs['input_ids'].shape[1]
        trimmed_generated_ids = generated_ids[:, input_ids_len:]
        
        # Decode the entire batch of predictions
        out_texts_full = self.processor.batch_decode(
            trimmed_generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        final_preds = []
        # Post-process each prediction in the batch
        for out_text_full in out_texts_full:
            out_text_stripped = out_text_full.strip()
            cleaned_output_lower = out_text_stripped.lower()
            
            found_class = "unknown"
            best_match_pos = float('inf')

            for cls_name in all_class_names:
                lower_cls_name = cls_name.lower()
                # Use word boundaries for a more robust match
                match = re.search(r"\b" + re.escape(lower_cls_name) + r"\b", cleaned_output_lower)
                if match:
                    pos = match.start()
                    if pos < best_match_pos:
                        best_match_pos = pos
                        found_class = lower_cls_name
            
            if found_class != "unknown":
                final_preds.append(found_class)
            else:
                # Fallback to the first word if no class name is found
                predicted_words = cleaned_output_lower.split()
                fallback_pred = predicted_words[0].lower() if predicted_words else "unknown"
                final_preds.append(fallback_pred)

        return final_preds, out_texts_full

    def evaluate(self,
                 dataset,
                 loader,
                 class_names: List[str],
                 out_dir: str):
        """
        Evaluates the model on a given dataset using batch processing.
        """
        classification_prompt = build_prompt(class_names)
        correct, total = 0, 0
        preds: List[Tuple[str, str, str]] = []
        first_batch_visual: List[Tuple] = []

        pbar = tqdm(total=len(dataset), desc="Evaluating")
        # The loader provides batches of images and labels
        for batch_idx, (imgs, labels) in enumerate(loader):
            # Process the entire batch of images
            batch_preds, batch_full_texts = self._predict_batch(imgs, classification_prompt, class_names)
            
            # Iterate through the results of the batch
            for i in range(len(imgs)):
                pred = batch_preds[i]
                full_text = batch_full_texts[i]
                gt = class_names[labels[i]].lower()

                if pred == "error_empty_input_ids":
                    print(f"Skipping image due to input_ids error. GT: {gt}")
                
                preds.append((gt, pred, full_text))

                if pred != "error_empty_input_ids":
                    correct += int(pred == gt)
                    total += 1

                # Save visuals for the first batch only (up to 10 images)
                if batch_idx == 0 and len(first_batch_visual) < 10:
                    first_batch_visual.append((imgs[i], gt, pred))
                
                current_acc_str = f"{100*correct/total:.2f}% ({correct}/{total})" if total > 0 else "N/A"
                pbar.set_postfix_str(f"Acc={current_acc_str}")
                pbar.update(1)

        pbar.close()

        os.makedirs(out_dir, exist_ok=True)
        # Create a file-safe name for the dataset
        if hasattr(dataset, 'name') and dataset.name:
            dataset_name_for_file = dataset.name
        elif hasattr(dataset, 'root') and dataset.root and os.path.basename(dataset.root):
            dataset_name_for_file = os.path.basename(dataset.root).lower()
        else:
            dataset_name_for_file = dataset.__class__.__name__.lower()

        run_name_prefix = self.hf_model_id.replace('/', '_')
        output_prefix = os.path.join(out_dir, f"{run_name_prefix}_{dataset_name_for_file}")

        # Save artifacts
        save_first_batch(first_batch_visual, f"{output_prefix}_visual_first_batch.png")
        save_csv(total, correct, preds, f"{output_prefix}_accuracy.csv", f"{output_prefix}_predictions.csv")

        acc = 100 * correct / total if total > 0 else 0
        print(f"▶ Finished {self.hf_model_id} on {dataset_name_for_file} — Accuracy {acc:.2f}% ({correct}/{total})")
        print(f"Results saved in {out_dir}")

