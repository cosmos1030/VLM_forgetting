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

* `Qwen/Qwen2.5-VL-3B-Instruct`

* `omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321`
  * https://github.com/om-ai-lab/VLM-R1?tab=readme-ov-file

* `omlab/VLM-R1-Qwen2.5VL-3B-Math-0305`

* `omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps`

* `konkazzz/GT-r1`

* `cosmos1030/Qwen2.5_VL-3B-rec-SFT`

* `cosmos1030/Qwen2.5_VL-3B-Instruct-GUI-SFT`

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

| **Dataset** | **Model**                 | **Accuracy (%)** |
|-------------|---------------------------|------------------|
| CIFAR10     | Qwen2-VL-2B (zeroshot)    | 83.48            |
|             | VisualThinker-GRPO        | 84.37            |
|             | VisualThinker-SFT         | 40.45            |
| CIFAR100    | Qwen2-VL-2B (zeroshot)    | 37.00            |
|             | VisualThinker-GRPO        | 36.67            |
|             | VisualThinker-SFT         | 4.73             |
| MNIST       | Qwen2-VL-2B (zeroshot)    | 69.05            |
|             | VisualThinker-GRPO        | 71.41            |
|             | VisualThinker-SFT         | 38.65            |


Qwen2.5-VL-3B-Instruct (zero shot)
cifar10: 85.34
cifar100: 50.80
MNIST: 68.49

Qwen2.5-VL-3B-Instruct-OVD-grpo
cifar10: 85.50
cifar100: 52.45
MNIST: 66.53

Qwen2.5-VL-3B-Instruct-math-grpo
cifar10: 85.64
cifar100: 51.04
MNIST: 75.30

Qwen2.5-VL-3B-Instruct-rec-grpo
cifar10: 85.56
cifar100: 51.12
MNIST: 66.39

Qwen2.5-VL-3B-Instruct-gui-grpo
cifar10: 88.22
cifar100: 64.37
MNIST: 96.04

Qwen2.5-VL-3B-Instruct-gui-sft (3epochs)
cifar10: 83.44
cifar100: 47.92
MNIST: 16.95