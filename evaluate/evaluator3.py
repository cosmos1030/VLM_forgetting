import os
import warnings
from typing import List, Tuple
import re

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor
from PIL import Image

# -- 임시 유틸 함수 (실제 환경에서는 prompt_utils.py에서 import) --
def build_prompt(class_names: List[str]) -> str:
    return (
        f"What is the main object in this image? "
        f"Choose one word from these options: {', '.join(class_names)}. "
        f"Answer in the form: 'The answer is [object name].'"
    )

def save_first_batch(visuals, path: str):
    print(f"[DEBUG] save_first_batch -> saving visuals to {path}")

def save_csv(total: int, correct: int, preds, acc_path: str, preds_path: str):
    print(f"[DEBUG] save_csv -> accuracy saved to {acc_path}, preds to {preds_path}")
# --------------------------------------------------------------

warnings.filterwarnings("ignore")

def extract_answer(text: str, class_names: List[str]) -> str:
    lowered = text.strip().lower()
    print(f"[DEBUG][extract_answer] raw: {text!r} -> lowered: {lowered!r}")

    # 1) "The answer is X" 매칭
    m = re.search(r"the answer is ([\w\-]+)", lowered)
    if m:
        cand = m.group(1)
        print(f"[DEBUG][extract_answer] regex → {cand!r}")
        if cand in [c.lower() for c in class_names]:
            return cand
        else:
            print(f"[DEBUG][extract_answer] but {cand!r} not in class_names")

    # 2) 클래스명 매칭
    for cls in class_names:
        lc = cls.lower()
        if re.search(r"\b"+re.escape(lc)+r"\b", lowered):
            print(f"[DEBUG][extract_answer] matched class_name → {lc!r}")
            return lc

    # 3) fallback
    tokens = lowered.split()
    fb = tokens[0] if tokens else "unknown"
    print(f"[DEBUG][extract_answer] fallback → {fb!r}")
    return fb


class Evaluator:
    def __init__(self, hf_model_id: str, is_instruct: bool, gpu_ids: str = "0"):
        self.hf_model_id = hf_model_id
        self.multi_gpu = gpu_ids.lower() not in ("", "cpu", "-1")
        if self.multi_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            print(f"[DEBUG] Using GPU(s): {gpu_ids}")
        else:
            print("[DEBUG] Using CPU")

        cfg = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
        self.actual_model_type = cfg.model_type
        print(f"[DEBUG] model_type: {self.actual_model_type}")

        model_kwargs = {
            "pretrained_model_name_or_path": hf_model_id,
            "torch_dtype": "auto",
            "device_map": "auto" if self.multi_gpu else None,
            "trust_remote_code": True,
        }

        if not is_instruct:
            print("[DEBUG] non-instruct logic")
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(**model_kwargs).eval()
        elif "qwen2_5_vl" in self.actual_model_type:
            print("[DEBUG] Qwen2.5-VL family")
            from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
            self.model = VLModel.from_pretrained(**model_kwargs).eval()
        elif "qwen2_vl" in self.actual_model_type:
            print("[DEBUG] Qwen2-VL instruct")
            from transformers import Qwen2VLForConditionalGeneration as VLModel
            self.model = VLModel.from_pretrained(**model_kwargs).eval()
        else:
            raise ValueError(f"Unsupported: {self.actual_model_type}")

        self.processor = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True)
        if getattr(self.processor, "image_processor", None).__class__.__name__ == 'Qwen2_5VLImageProcessor':
            self.processor.image_processor.use_fast = False

    @torch.no_grad()
    def _predict_batch(
        self,
        imgs: List[Image.Image],
        prompt_text: str,
        class_names: List[str],
    ) -> Tuple[List[str], List[str]]:
        # RGB로 변환
        rgb = [img.convert("RGB") for img in imgs]
        print(f"[DEBUG][_predict_batch] {len(rgb)} images")

        # 직접 “User: … Assistant:” 형태의 텍스트 생성
        full_prompt = [f"User: <image> {prompt_text}\nAssistant:" for _ in rgb]
        print(f"[DEBUG][_predict_batch] sample prompt: {full_prompt[0]!r}")

        # apply_chat_template로 generation prompt 태깅만 수행
        templates = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": fp}]}],
                tokenize=False,
                add_generation_prompt=True
            )
            for fp in full_prompt
        ]
        print(f"[DEBUG][_predict_batch] applied templates: {templates[0]!r}")

        inputs = self.processor(text=templates, images=rgb, padding=True, return_tensors="pt")
        print(f"[DEBUG][_predict_batch] inputs: {list(inputs.keys())}, shape: {inputs['input_ids'].shape}")
        inputs = inputs.to(self.model.device)

        if inputs['input_ids'].shape[-1] == 0:
            print("[DEBUG][_predict_batch] empty inputs")
            return ["error_empty_input_ids"] * len(rgb), [""] * len(rgb)

        generated = self.model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=True)
        print(f"[DEBUG][_predict_batch] generated shape: {generated.shape}")

        trimmed = generated[:, inputs['input_ids'].shape[1]:]
        out_texts = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(f"[DEBUG][_predict_batch] decoded: {out_texts}")

        preds = []
        for t in out_texts:
            p = extract_answer(t, class_names)
            print(f"[DEBUG][_predict_batch] → pred {p!r}")
            preds.append(p)

        return preds, out_texts

    def evaluate(self, dataset, loader, class_names: List[str], out_dir: str):
        prompt = build_prompt(class_names)
        correct = total = 0
        preds = []
        visuals = []

        pbar = tqdm(total=len(dataset), desc="Evaluating")
        for bidx, (imgs, labels) in enumerate(loader):
            batch_preds, batch_texts = self._predict_batch(imgs, prompt, class_names)
            for i in range(len(imgs)):
                gt = class_names[labels[i]].lower()
                p = batch_preds[i]
                t = batch_texts[i]
                print(f"[DEBUG][evaluate] GT={gt!r}, pred={p!r}, full={t!r}")
                preds.append((gt, p, t))
                if p != "error_empty_input_ids":
                    correct += int(p == gt)
                    total += 1
                if bidx == 0 and len(visuals) < 10:
                    visuals.append((imgs[i], gt, p))
                acc = f"{100*correct/total:.2f}% ({correct}/{total})" if total else "N/A"
                pbar.set_postfix_str(f"Acc={acc}")
                pbar.update(1)
        pbar.close()

        os.makedirs(out_dir, exist_ok=True)
        ds_name = getattr(dataset, 'name', None) or getattr(dataset, 'root', '').split(os.sep)[-1] or dataset.__class__.__name__
        prefix = f"{self.hf_model_id.replace('/', '_')}_{ds_name}"
        save_first_batch(visuals, f"{out_dir}/{prefix}_vis.png")
        save_csv(total, correct, preds, f"{out_dir}/{prefix}_acc.csv", f"{out_dir}/{prefix}_preds.csv")

        final_acc = 100 * correct / total if total else 0
        print(f"▶ Done: {final_acc:.2f}% ({correct}/{total}) on {ds_name}")
