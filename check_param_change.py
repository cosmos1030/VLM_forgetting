#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_model_changes.py

This script compares parameter changes between:
  • Qwen/Qwen2-VL-2B (base)
  • cosmos1030/Qwen2_VL-2B-SFT_revised2 (SFT)
  • turningpoint-ai/VisualThinker-R1-Zero (RL)

It loads each model one at a time (to avoid OOM), extracts the state_dict to CPU,
computes per-parameter and global change ratios, logs messages to both console and
a log file under the `logs/` directory, prints top-changed params, and produces/saves:
  • Histogram of change ratios
  • Layer-wise mean change bar chart (if layers are detected)
"""

import os
import logging
import torch
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoModel

# ---- Configuration ----
# MODEL_IDS = {
#     "base": "Qwen/Qwen2-VL-2B",
#     "sft":  "cosmos1030/Qwen2_VL-2B-SFT_revised2",
#     "rl":   "turningpoint-ai/VisualThinker-R1-Zero",
# }
# MODEL_IDS = {
#     "base": "Qwen/Qwen2.5-VL-3B-Instruct",
#     "sft":  "cosmos1030/Qwen2.5_VL-3B-Instruct-GUI-SFT",
#     "rl":   "konkazzz/GT-r1",
# }
MODEL_IDS = {
    "base": "Qwen/Qwen2.5-VL-3B-Instruct",
    "sft":  "cosmos1030/Qwen2.5_VL-3B-rec-SFT-500steps",
    "rl":   "omlab/Qwen2.5VL-3B-VLM-R1-REC-500steps",
}
OFFLOAD_FOLDER = "offload"
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "compare_model_changes.log")

# ---- Setup logging ----
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("compare_model_changes")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

# File handler
fh = logging.FileHandler(LOG_FILE, mode="w")
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

def load_model(model_id: str):
    """
    Load a model with:
      - device_map="auto" to distribute across GPUs
      - max_memory per device to limit usage
      - offload_state_dict to spill weights to disk if needed
      - float16 and low_cpu_mem_usage to reduce footprint
    """
    n_gpus = torch.cuda.device_count()
    max_mem = {i: "15GiB" for i in range(n_gpus)}
    max_mem["cpu"] = "100GiB"

    logger.info(f"Loading model '{model_id}' with {n_gpus} GPUs, max_memory={max_mem}")
    return AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory=max_mem,
        offload_folder=OFFLOAD_FOLDER,
        offload_state_dict=True,
    )

def extract_state_dict_cpu(model: torch.nn.Module):
    """
    Move all parameters to CPU and return state_dict dict
    """
    return {name: param.detach().cpu() for name, param in model.state_dict().items()}

def collect_state_dicts():
    """
    Load each model one by one, extract its state_dict to CPU,
    then delete the model and clear cache to free GPU memory.
    Returns a dict of name -> state_dict.
    """
    state_dicts = {}
    for name, model_id in MODEL_IDS.items():
        logger.info(f"[{name.upper()}] Loading & extracting state_dict...")
        model = load_model(model_id)
        state_dicts[name] = extract_state_dict_cpu(model)
        # free memory
        del model
        torch.cuda.empty_cache()
        logger.info(f"[{name.upper()}] state_dict extracted, GPU cache cleared")
    return state_dicts

def compute_change_dataframe(sd_base, sd_target, tol=1e-5):
    """
    Compare two state_dicts:
      - Compute absolute difference per parameter
      - Count elements where diff > tol
      - Return DataFrame with columns: param, changed, total, change_ratio
    """
    rows = []
    for name, base_param in sd_base.items():
        tgt_param = sd_target.get(name)
        if tgt_param is None or base_param.shape != tgt_param.shape:
            continue
        diff = (base_param.float() - tgt_param.float()).abs()
        changed = int((diff > tol).sum().item())
        total = diff.numel()
        rows.append({
            "param": name,
            "changed": changed,
            "total": total,
            "change_ratio": changed / total,
        })
    df = pd.DataFrame(rows)
    logger.info(f"Computed change DataFrame: {len(df)} parameters compared")
    return df

def compute_global_change(df: pd.DataFrame) -> float:
    """
    Compute global change ratio = sum(changed) / sum(total)
    """
    total_changed = df["changed"].sum()
    total_params = df["total"].sum()
    ratio = total_changed / total_params if total_params > 0 else 0.0
    logger.info(f"Global change: {total_changed}/{total_params} = {ratio:.4f}")
    return ratio

def plot_histograms(dfs):
    """
    For each DataFrame in dfs (name->df), plot and save a histogram of change_ratios.
    """
    for name, df in dfs.items():
        plt.figure(figsize=(8, 6))
        plt.hist(df["change_ratio"], bins=50)
        plt.title(f"{name.upper()} Change Ratio Distribution")
        plt.xlabel("Fraction of Changed Elements")
        plt.ylabel("Number of Parameters")
        plt.tight_layout()
        out_file = f"{name}_change_histogram.png"
        plt.savefig(out_file)
        plt.close()
        logger.info(f"Saved histogram: {out_file}")

def plot_layer_changes(dfs):
    """
    For each DataFrame in dfs (name->df), extract 'layer.X' from param names,
    compute mean change_ratio per layer, and plot a bar chart if any layers found.
    """
    for name, df in dfs.items():
        df = df.copy()
        df["layer"] = df["param"].str.extract(r"(layer\.\d+)", expand=False)
        layer_means = df.dropna(subset=["layer"]).groupby("layer")["change_ratio"].mean().sort_index()
        if layer_means.empty:
            logger.warning(f"No 'layer.X' parameters found in {name.upper()}, skipping layer plot.")
            continue
        plt.figure(figsize=(12, 6))
        layer_means.plot(kind="bar")
        plt.title(f"Layer-wise Mean Change Ratio ({name.upper()})")
        plt.xlabel("Layer")
        plt.ylabel("Mean Change Ratio")
        plt.tight_layout()
        out_file = f"{name}_layer_change.png"
        plt.savefig(out_file)
        plt.close()
        logger.info(f"Saved layer-wise bar chart: {out_file}")

def main():
    logger.info("=== Starting parameter change comparison ===")
    state_dicts = collect_state_dicts()

    df_sft = compute_change_dataframe(state_dicts["base"], state_dicts["sft"])
    df_rl = compute_change_dataframe(state_dicts["base"], state_dicts["rl"])

    logger.info("=== Global change ratios ===")
    logger.info(f"SFT: {compute_global_change(df_sft) * 100:.2f}%")
    logger.info(f"RL:  {compute_global_change(df_rl) * 100:.2f}%")

    logger.info("=== Top 10 changed parameters ===")
    logger.info("SFT:\n" + df_sft.nlargest(10, "change_ratio")[["param", "change_ratio"]].to_string(index=False))
    logger.info("RL:\n" + df_rl.nlargest(10, "change_ratio")[["param", "change_ratio"]].to_string(index=False))

    plot_histograms({"sft": df_sft, "rl": df_rl})
    plot_layer_changes({"sft": df_sft, "rl": df_rl})

    logger.info("=== Completed parameter change comparison ===")

if __name__ == "__main__":
    main()
