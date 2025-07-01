import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from transformers import AutoProcessor, AutoConfig, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm
from typing import List
from PIL import Image
import os
from attentive_pooler import AttentivePooler, AttentiveClassifier
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    choices=["CIFAR10", "CIFAR100", "MNIST"],
    default="CIFAR10",
    help="dataset to use"
)
parser.add_argument(
    "--model_id",
    type=str,
    default="Qwen/Qwen2-VL-2B",
    help="model on huggingface"
)
parser.add_argument(
    "--cuda_device",
    type=int,
    help="cuda device you want to use ex. 2"
)
args = parser.parse_args()

# 1. Env setting and hyperparam
# ==============================================================================
MODEL_ID = args.model_id
DATASET = args.dataset # ["CIFAR10", "CIFAR100", "MNIST"]
sanitized_model_id = MODEL_ID.replace("/", "_")

BATCH_SIZE = 512
LEARNING_RATE = 1e-3
EPOCHS = 15
NUM_CLASSES = 10
VAL_SPLIT = 0.1
CHECKPOINT_PATH = f"{DATASET}_{sanitized_model_id}_Instruct.pth"
if DATASET == "CIFAR10":
    NUM_CLASSES= 10
    DatasetClass = CIFAR10
if DATASET == "CIFAR100":
    NUM_CLASSES= 100
    DatasetClass = CIFAR100
if DATASET == "MNIST":
    NUM_CLASSES= 10
    DatasetClass = MNIST

# GPU setting
device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Wandb initialization
wandb_config = {
    "model_id": MODEL_ID,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "num_classes": NUM_CLASSES,
    "val_split": VAL_SPLIT,
    "device": device,
    "attentive_pooler_depth": 1,
    "attentive_pooler_complete_block": True
}
wandb.init(project=f"Qwen_AttentiveProbing", name=f"{DATASET}_{MODEL_ID}", config=wandb_config)

# 2. load model and processor
# ==============================================================================
print("Loading Qwen model and processor...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(MODEL_ID)

vision_encoder = model.model.visual
del model
vision_encoder.to(device).eval()
print("Model and processor loaded.")

# 3. define model for linear probing
# ==============================================================================
class AttentiveProbingQwenVL(nn.Module):
    def __init__(self, vision_encoder, num_classes):
        super().__init__()
        self.vision_encoder = vision_encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        vision_hidden_size = self.vision_encoder.config.out_hidden_size
        
        self.attentive_classifier = AttentiveClassifier(
            embed_dim=vision_hidden_size,
            num_heads=vision_encoder.config.num_heads,
            mlp_ratio=4.0,
            depth=1,
            norm_layer=nn.LayerNorm,
            init_std=0.02,
            qkv_bias=True,
            num_classes=num_classes,
            complete_block=True
        )

    def forward(self, images: List[Image.Image]):
        current_batch_size = len(images)
        if current_batch_size == 0:
            return torch.tensor([], device=self.attentive_classifier.linear.weight.device)
            
        processed = processor.image_processor(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device=device, dtype=torch.bfloat16)
        grid_thw = processed["image_grid_thw"].to(device)
        
        with torch.no_grad():
            flat_embeddings = self.vision_encoder(pixel_values, grid_thw)
        reshaped = flat_embeddings.view(current_batch_size, flat_embeddings.shape[0] // current_batch_size, flat_embeddings.shape[1])
        logits = self.attentive_classifier(reshaped)
        return logits

# 4. prepare dataset
# ==============================================================================
print(f"Loading and preparing {DATASET} dataset...")
train_dataset_full = DatasetClass(root='./data', train=True, download=True)
test_dataset = DatasetClass(root='./data', train=False, download=True)

num_train = len(train_dataset_full)
num_val = int(num_train * VAL_SPLIT)
num_train = num_train - num_val
train_dataset, val_dataset = random_split(train_dataset_full, [num_train, num_val])

def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return images, labels

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"Dataset prepared: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")

# 5. prepare training
# ==============================================================================
probing_model = AttentiveProbingQwenVL(vision_encoder, NUM_CLASSES).to(device).to(dtype=torch.bfloat16)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(probing_model.attentive_classifier.parameters(), lr=LEARNING_RATE)

print("Starting training...")

best_val_accuracy = 0.0
best_epoch = 0

# 6. training and validation loop
# ==============================================================================
for epoch in range(EPOCHS):
    # --- Training ---
    probing_model.attentive_classifier.train()
    train_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
    for images, labels in progress_bar:
        labels = labels.to(device)
        outputs = probing_model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)

    # --- Validation ---
    probing_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]")
        for images, labels in progress_bar_val:
            labels = labels.to(device)
            outputs = probing_model(images)
            loss = criterion(outputs.float(), labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            val_acc = 100 * correct / total
            progress_bar_val.set_postfix(accuracy=f"{val_acc:.2f}%")

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    wandb.log({
        "epoch": epoch+1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_accuracy": val_accuracy
    })

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1
        torch.save(probing_model.state_dict(), CHECKPOINT_PATH)
        print(f"🎉 New best model saved at Epoch {best_epoch}! Validation Accuracy: {val_accuracy:.2f}%")

print("Training finished.")

# 7. Final test and analysis
# ==============================================================================
print(f"\nLoading best model from '{CHECKPOINT_PATH}' for final evaluation...")
if os.path.exists(CHECKPOINT_PATH):
    probing_model.load_state_dict(torch.load(CHECKPOINT_PATH))
    print("Best model loaded successfully.")
else:
    print("Warning: No checkpoint file found. Evaluating the model from the last epoch.")
    best_epoch = "N/A" # 저장된 파일이 없을 경우

probing_model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="[Testing]"):
        labels = labels.to(device)
        outputs = probing_model(images)
        
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"==========================================")
print(f"🏆 Best model was from Epoch {best_epoch} with Validation Accuracy: {best_val_accuracy:.2f}%")
print(f"✅ Final Test Accuracy (on best model): {test_accuracy:.2f}%")
print(f"==========================================")

wandb.log({"final_test_accuracy": test_accuracy})


wandb.finish()