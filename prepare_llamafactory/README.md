# 📚 SFT Training with LLaMA-Factory

This guide explains how to train a Supervised Fine-Tuning (SFT) model using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), applied to the CIFAR-10 and CIFAR-100 datasets.

---

## 🔧 Setup

### 1. Clone the LLaMA-Factory repository

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

### 2. Install dependencies

```bash
pip install -e ".[torch,metrics]"
```

---

## 📁 Data Preparation

### 1. Generate CIFAR-style multi-modal data

```bash
python data_gen_cifar10.py
python data_gen_cifar100.py
```

This will generate:

- `mllm_cifar10.json`
- `mllm_cifar100.json`
- `cifar10/` folder
- `cifar100/` folder

### 2. Organize files

Move the following into the appropriate directories:

```bash
# Move data-related files
mv mllm_cifar10.json mllm_cifar100.json dataset_info.json data/
mv cifar10 data/
mv cifar100 data/

# Move training config YAMLs
mv qwen2_5_vl_full_sft_cifar10.yaml examples/train_full/
mv qwen2_5_vl_full_sft_cifar100.yaml examples/train_full/
mv qwen2_vl_full_sft_cifar10.yaml examples/train_full/
mv qwen2_vl_full_sft_cifar100.yaml examples/train_full/
```

---

## 🚀 Training

Run the following command to start training. For example:

```bash
llamafactory-cli train examples/train_full/qwen2_5_vl_full_sft_cifar10.yaml
```

Replace the YAML config with any of the others for different models or datasets:

- `qwen2_5_vl_full_sft_cifar100.yaml`
- `qwen2_vl_full_sft_cifar10.yaml`
- `qwen2_vl_full_sft_cifar100.yaml`
