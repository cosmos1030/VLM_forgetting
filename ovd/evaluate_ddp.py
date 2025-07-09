import os
import warnings
import json
from typing import List, Tuple, Dict, Any
import re
import argparse # << GPU ID 지정을 위해 argparse 추가

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor
from PIL import Image

# --------------------------------------------------------------------------
# 1. 분산 환경 설정 및 유틸리티 함수
# --------------------------------------------------------------------------

warnings.filterwarnings("ignore")

def setup_distributed(rank, world_size):
    # DDP가 통신할 주소와 포트 설정
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' 
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # 현재 프로세스에 해당하는 GPU를 사용하도록 설정
    torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()

# --------------------------------------------------------------------------
# 2. DDP를 위한 커스텀 데이터셋 및 데이터 로더
# --------------------------------------------------------------------------

class ImageDataset(Dataset):
    def __init__(self, images_info: List[Dict], image_folder_path: str):
        self.images_info = images_info
        self.image_folder_path = image_folder_path
        # 데이터 순서 보장을 위해 image_id와 원본 인덱스를 매핑
        self.id_to_original_index = {info['id']: i for i, info in enumerate(images_info)}

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        img_info = self.images_info[idx]
        full_image_path = os.path.join(self.image_folder_path, img_info["file_name"])
        try:
            with Image.open(full_image_path) as img:
                img = img.convert("RGB")
            return {"image": img, "id": img_info["id"]}
        except Exception as e:
            # print(f'Warning: Could not load image {full_image_path}: {e}')
            return None

def collate_fn(batch: List[Dict]) -> Dict:
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return {
        "images": [item['image'] for item in batch],
        "ids": [item['id'] for item in batch]
    }

# --------------------------------------------------------------------------
# 3. 평가자(Evaluator) 클래스 (이전과 동일)
# --------------------------------------------------------------------------
class Evaluator:
    def __init__(self, hf_model_id: str, rank: int):
        self.hf_model_id = hf_model_id
        self.rank = rank
        self.device = torch.device(f"cuda:{rank}")

        cfg = AutoConfig.from_pretrained(hf_model_id)
        model_type = cfg.model_type
        
        if model_type in ["qwen2_vl", "qwen2_5_vl"] or "qwen2" in hf_model_id.lower():
            from transformers import AutoModelForCausalLM as VLModel
        else:
             from transformers import AutoModelForCausalLM as VLModel

        if self.rank == 0: print(f"Loading {hf_model_id} with {VLModel.__name__}")
        
        model = VLModel.from_pretrained(
            hf_model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).eval()
        
        model.to(self.device)
        self.model = DDP(model, device_ids=[self.rank], find_unused_parameters=True)
        self.processor = AutoProcessor.from_pretrained(hf_model_id, trust_remote_code=True)

    @torch.no_grad()
    def _predict_batch(self, imgs: List[Image.Image], prompt_text: str) -> List[List[Dict]]:
        messages = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]] * len(imgs)
        
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

        inputs = self.processor(text=texts, images=imgs, return_tensors="pt", padding=True).to(self.device)

        gen_kwargs = {"max_new_tokens": 512, "do_sample": False}
        
        generated_ids = self.model.module.generate(**inputs, **gen_kwargs)
        
        generated_ids = [ids[len(inputs['input_ids'][i]):] for i, ids in enumerate(generated_ids)]
        out_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        final_json_preds = [self._parse_output_text(text) for text in out_texts]
        return final_json_preds

    def _parse_output_text(self, text: str) -> List[Dict]:
        try:
            # This is a placeholder parsing logic. You should adapt this to your model's specific output format.
            text = text.replace("<bbox>", " [").replace("</bbox>", "] ")
            bbox_texts = re.findall(r'\[\d+,\d+,\d+,\d+\]', text)
            bboxes = [json.loads(b) for b in bbox_texts]
            return [{"bbox": box, "score": 0.9, "category_id": -1} for box in bboxes]
        except Exception:
            return []

    def evaluate(self, dataset_json_path: str, image_folder_path: str, output_dir: str, batch_size: int, world_size: int):
        if self.rank == 0:
            print(f"Starting evaluation on {world_size} GPUs.")
            os.makedirs(output_dir, exist_ok=True)

        with open(dataset_json_path, 'r', encoding='utf-8') as f:
            dataset_data = json.load(f)

        dataset = ImageDataset(dataset_data['images'], image_folder_path)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=self.rank, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=4)
        
        prompt = "Detect all objects in the image."
        predictions_for_this_rank = []
        pbar = tqdm(dataloader, desc=f"Rank {self.rank} Processing", disable=(self.rank != 0))

        for batch in pbar:
            if batch is None: continue
            batch_preds = self._predict_batch(batch["images"], prompt)
            for i, preds_for_one_image in enumerate(batch_preds):
                img_id = batch["ids"][i]
                for pred in preds_for_one_image:
                    pred["image_id"] = img_id
                    predictions_for_this_rank.append(pred)
        
        gathered_predictions = [None] * world_size
        dist.gather_object(
            predictions_for_this_rank,
            gathered_predictions if self.rank == 0 else None,
            dst=0
        )
        
        if self.rank == 0:
            print("\nGathering results on Rank 0...")
            final_all_predictions = [item for sublist in gathered_predictions for item in sublist]
            final_all_predictions.sort(key=lambda x: dataset.id_to_original_index.get(x['image_id'], -1))
            
            model_name_safe = self.hf_model_id.replace('/', '_').replace('-', '_')
            dataset_name_safe = os.path.basename(dataset_json_path).replace('.json', '')
            output_path = os.path.join(output_dir, f"{model_name_safe}_{dataset_name_safe}_predictions.json")

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_all_predictions, f, indent=2)
            
            print(f"✅ Evaluation complete. All predictions are merged, sorted, and saved to:")
            print(output_path)

# --------------------------------------------------------------------------
# 4. 메인 실행 함수 (argparse 추가)
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Distributed Evaluation Script")
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')")
    # 다른 인자들도 필요하다면 여기에 추가할 수 있습니다.
    # parser.add_argument("--model_id", type=str, default="Qwen/Qwen-VL-Chat")
    # parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # <<< 핵심 변경 사항: 지정된 GPU만 보이도록 설정 >>>
    # torchrun이 실행되기 전에 이 환경 변수가 설정되어야 하므로,
    # 실제로는 torchrun 명령어에 직접 삽입하는 것이 더 안정적입니다.
    # 아래는 스크립트 내에서 설정하는 예시이며, 실행 명령어에서 설정하는 것을 권장합니다.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    
    gpu_list = args.gpu_ids.split(',')
    world_size = len(gpu_list)

    # DDP 환경 변수가 있는지 확인
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        # torchrun이 설정한 world_size가 우선순위를 가짐
        world_size_from_torchrun = int(os.environ['WORLD_SIZE'])
        if world_size != world_size_from_torchrun:
             print(f"Warning: --gpu_ids count ({world_size}) mismatches torchrun --nproc_per_node ({world_size_from_torchrun}). Using {world_size_from_torchrun}.")
             world_size = world_size_from_torchrun

        is_ddp = True
        setup_distributed(rank, world_size)
    else:
        rank, world_size, is_ddp = 0, 1, False

    # --- 파라미터 설정 ---
    HF_MODEL_ID = "Qwen/Qwen-VL-Chat"
    BATCH_SIZE = 8
    JSON_DATASET_PATH = "ovdeval/celebrity.json"
    IMAGE_DIR_PATH = "ovdeval/celebrity"
    OUTPUT_RESULTS_DIR = "bbox_results_final"

    # rank 0에서만 파라미터 출력
    if rank == 0:
        print("-" * 50)
        print(f"Starting job with {world_size} GPUs on IDs: {args.gpu_ids}")
        print(f"Model: {HF_MODEL_ID}")
        print(f"Batch Size per GPU: {BATCH_SIZE}")
        print("-" * 50)


    evaluator = Evaluator(hf_model_id=HF_MODEL_ID, rank=rank)
    evaluator.evaluate(
        dataset_json_path=JSON_DATASET_PATH,
        image_folder_path=IMAGE_DIR_PATH,
        output_dir=OUTPUT_RESULTS_DIR,
        batch_size=BATCH_SIZE,
        world_size=world_size
    )

    if is_ddp:
        cleanup_distributed()

if __name__ == "__main__":
    main()