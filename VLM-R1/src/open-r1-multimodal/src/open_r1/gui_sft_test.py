# Copyright 2025 The HuggingFace Team. All rights reserved.
# ... (라이선스 헤더는 동일)

"""
Fine-tuned Vision-Language Model evaluation script for classification tasks.
이 스크립트는 모델의 출력이 주어진 정답과 '완전 일치'하는지를 기준으로
정확도를 계산합니다. UI 결함 탐지와 같은 분류 문제에 최적화되어 있습니다.
"""

import logging
import os
import sys
import json
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import warnings

import transformers
# from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, HfArgumentParser
# from transformers import AutoProcessor, AutoModelForConditionalGeneration, HfArgumentParser
from transformers import AutoConfig, AutoProcessor, HfArgumentParser

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ##################################################################
# 분류 문제에 맞게 인자 클래스를 간소화합니다.
# ##################################################################
@dataclass
class EvalScriptArguments:
    model_name_or_path: str = field(
        metadata={"help": "평가할 파인튜닝된 모델의 경로 또는 허브 주소."}
    )
    dataset_name: str = field(
        metadata={"help": "평가에 사용할 .jsonl 파일 경로."}
    )
    image_root: str = field(
        metadata={"help": "이미지가 저장된 루트 디렉토리."}
    )
    output_file: str = field(
        default="./eval_results_classification.jsonl",
        metadata={"help": "모델의 생성 결과와 정답, 채점 결과를 저장할 파일 경로."}
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "모델 로드 시 사용할 torch dtype. (e.g., 'bfloat16', 'float16', 'float32')"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "모델 로드 시 원격 코드를 신뢰할지 여부."}
    )
    max_new_tokens: int = field(
        default=50,  # 분류 라벨은 짧으므로 토큰 수를 줄여도 됩니다.
        metadata={"help": "모델이 생성할 최대 토큰 수."}
    )

# ##################################################################
# 데이터셋 클래스는 이전과 동일하게 사용합니다.
# ##################################################################
class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, image_root: str):
        super(LazySupervisedDataset, self).__init__()
        self.image_root = image_root
        if not self.image_root:
            raise ValueError("image_root 인자가 반드시 필요합니다. 이미지 폴더 경로를 지정해주세요.")
        self.list_data_dict = []
        logger.info(f"Loading data from {data_path}...")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.list_data_dict.append(json.loads(line))
        logger.info(f"Loaded {len(self.list_data_dict)} examples.")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        example = self.list_data_dict[i].copy()
        human_prompt = example["conversations"][0]["value"]
        gpt_answer = example["conversations"][1]["value"]
        user_content = []
        image_files = example.get("image", [])
        for image_file in image_files:
            image_path = os.path.join(self.image_root, image_file)
            if not os.path.exists(image_path):
                logger.warning(f"Image not found at path: {image_path}")
                continue
            user_content.append({"type": "image", "image": f"file://{image_path}"})
        user_content.append({"type": "text", "text": human_prompt})
        example["messages"] = [{"role": "user", "content": user_content}]
        example["ground_truth"] = gpt_answer
        return example

def main():
    parser = HfArgumentParser((EvalScriptArguments,))
    args, = parser.parse_args_into_dataclasses()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, args.torch_dtype)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO
    )
    logger.info("Loading vision-language model and processor...")
    # 1) config에서 실제 model_type 확인
    cfg = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code
    )
    model_type = cfg.model_type.lower()

    # 2) model_type에 따라 올바른 클래스 import
    if model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
        logger.info("Detected qwen2_5_vl → using Qwen2_5_VLForConditionalGeneration")
    else:
        from transformers import Qwen2VLForConditionalGeneration as VLModel
        logger.info("Detected qwen2_vl → using Qwen2VLForConditionalGeneration")

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code
    )
    model = VLModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        device_map="auto"
    ).eval()
    logger.info(f"VLM model loaded on {model.device}.")

    dataset = LazySupervisedDataset(data_path=args.dataset_name, image_root=args.image_root)
    
    results = []
    total_correct = 0
    log_path = args.output_file.replace(".jsonl", "_per_image.log")
    log_f = open(log_path, "w", encoding="utf-8")
    log_f.write("id\tprediction\tground_truth\n")
    pbar = tqdm(total=len(dataset), desc="Evaluating (Exact Match)")

    for i, example in enumerate(dataset):
        messages = example["messages"]
        ground_truth = example["ground_truth"].strip()
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [Image.open(os.path.join(args.image_root, p)) for p in example.get("image", []) if os.path.exists(os.path.join(args.image_root, p))]
        
        model_inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        
        input_token_len = model_inputs["input_ids"].shape[1]
        generated_ids = generated_ids[:, input_token_len:]
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # 완전 일치(Exact Match)로 채점
        is_correct = 1 if response == ground_truth else 0
        
        total_correct += is_correct
        total_seen = i+1
        accuracy = 100* total_correct/total_seen
        
        pbar.set_postfix_str(f"Acc={accuracy:.2f}% ({total_correct}/{total_seen})")
        pbar.update(1)
        
        img_id = example.get("id", i)
        log_f.write(f"{img_id}\t{response}\t{ground_truth}\n")

        results.append({
            "id": example.get("id", i),
            "prompt": example["conversations"][0]["value"],
            "ground_truth": ground_truth,
            "model_response": response,
            "is_correct": is_correct
        })
    pbar.close()
    log_f.close()

    accuracy = (total_correct / len(dataset)) * 100 if len(dataset) > 0 else 0

    logger.info("\n\n" + "="*50)
    logger.info("          Evaluation Summary")
    logger.info("="*50)
    logger.info(f"Total Samples:         {len(dataset)}")
    logger.info(f"Correct Predictions:   {total_correct}")
    logger.info(f"Accuracy:              {accuracy:.2f}%")
    logger.info("="*50)
    logger.info(f"Detailed results saved to {args.output_file}")

    with open(args.output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
    

# python evaluate_classifier.py \
#     --model_name_or_path ./path/to/your/finetuned_model \
#     --dataset_name /path/to/data/gui_multi-image.jsonl \
#     --image_root /path/to/images \
#     --output_file ./eval_results_classification.jsonl