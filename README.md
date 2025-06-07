## VLM Classification Evaluation

This repository evaluates classification performance on CIFAR10, CIFAR100, and MNIST to measure forgetting when fine-tuning vision-language models (VLM) with GRPO and SFT methods.

---

### Environment Setup

1. Create and activate a conda environment:

   ```bash
   conda create -n vlm python=3.10
   conda activate vlm
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

### Supported Models

* `turningpoint-ai/VisualThinker-R1-Zero`

  * Qwen2-VL-2B trained via GRPO on the SAT dataset.
  * Checkpoint provided by [Deep-Agent/R1-V](https://github.com/Deep-Agent/R1-V).

* `cosmos1030/Qwen2_VL-2B-SFT_revised2`

  * Qwen2-VL-2B fine-tuned via supervised finetuning (SFT) on the SAT dataset.
  * Training code from [Deep-Agent/R1-V](https://github.com/Deep-Agent/R1-V).

* `Qwen/Qwen2-VL-2B`

  * Base Qwen2-VL-2B model without additional training.

* `Qwen/Qwen2-VL-2B-Instruct`

  * Instruct-tuned version of Qwen2-VL-2B.

---

### Running the Evaluator

1. Open `scripts/run.sh` and update parameters as needed.
2. Execute the evaluation script:

   ```bash
   python <path of main.py> \
     --model_name <Hugging Face model name> \
     --dataset <CIFAR10|CIFAR100|MNIST> \
     --is_instruct <0 (non-instruct) | 1 (instruct)> \
     --gpus <GPU IDs, e.g. "0" or "0,1"> \
     --batch_size <batch size, e.g. 128> \
     --output_dir <path to save results>
   ```

Example:

```bash
python <path of main.py> \
  --model_name turningpoint-ai/VisualThinker-R1-Zero \
  --dataset CIFAR10 \
  --is_instruct 0 \
  --gpus 0 \
  --batch_size 128 \
  --output_dir ./results/grpo_cifar10
```

---

### Results

Results (accuracy, confusion matrices, etc.) will be saved under the specified `output_dir`. Use these outputs to compare forgetting across models and finetuning methods.
