import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from tqdm import tqdm
from typing import List
from PIL import Image
import os

# 1. 환경 설정 및 하이퍼파라미터
# ==============================================================================
MODEL_ID = "Qwen/Qwen2-VL-2B"
# MODEL_ID = "cosmos1030/Qwen2_VL-2B-SFT_revised2"
# MODEL_ID = "turningpoint-ai/VisualThinker-R1-Zero"
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
EPOCHS = 15
NUM_CLASSES = 10
VAL_SPLIT = 0.1
CHECKPOINT_PATH = "best_model_grpo.pth"

# GPU 사용 설정
device = "cuda:2" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. 모델 및 프로세서 로드
# ==============================================================================
print("Loading Qwen model and processor...")
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(MODEL_ID)

vision_encoder = model.model.visual
del model
vision_encoder.to(device).eval()
print("Model and processor loaded.")

# 3. Linear Probing을 위한 모델 정의
# ==============================================================================
class LinearProbingQwenVL(nn.Module):
    def __init__(self, vision_encoder, num_classes):
        super().__init__()
        self.vision_encoder = vision_encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        vision_hidden_size = self.vision_encoder.config.hidden_size
        self.classifier = nn.Linear(vision_hidden_size, num_classes)

    def forward(self, images: List[Image.Image]):
        current_batch_size = len(images)
        if current_batch_size == 0:
            return torch.tensor([], device=self.classifier.weight.device)
            
        processed = processor.image_processor(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device=device, dtype=torch.bfloat16)
        grid_thw = processed["image_grid_thw"].to(device)
        
        with torch.no_grad():
            flat_embeddings = self.vision_encoder(pixel_values, grid_thw)
        
        vision_hidden_size = self.vision_encoder.config.hidden_size
        
        tokens = flat_embeddings.view(current_batch_size, -1, vision_hidden_size)
        pooled = tokens.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

# 4. 데이터셋 준비
# ==============================================================================
print("Loading and preparing CIFAR-10 dataset...")
train_dataset_full = CIFAR10(root='./data', train=True, download=True)
test_dataset = CIFAR10(root='./data', train=False, download=True)

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

# 5. 학습 준비
# ==============================================================================
probing_model = LinearProbingQwenVL(vision_encoder, NUM_CLASSES).to(device).to(dtype=torch.bfloat16)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(probing_model.classifier.parameters(), lr=LEARNING_RATE)

print("Starting training...")

# ✅ 1. 최고 성능 및 에포크 저장을 위한 변수 초기화
best_val_accuracy = 0.0
best_epoch = 0

# 6. 학습 및 검증 루프
# ==============================================================================
for epoch in range(EPOCHS):
    # --- Training ---
    probing_model.classifier.train()
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

    # ✅ 2. 검증 성능 비교 및 최고 성능 모델/에포크 저장
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1 # 에포크 번호는 1부터 시작하므로 +1
        torch.save(probing_model.state_dict(), CHECKPOINT_PATH)
        # 저장 메시지에 에포크 번호도 함께 출력
        print(f"🎉 New best model saved at Epoch {best_epoch}! Validation Accuracy: {val_accuracy:.2f}%")

print("Training finished.")

# 7. 최종 테스트 및 성능 평가
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
# ✅ 3. 최종 결과에 최고 성능 에포크 정보 추가
print(f"🏆 Best model was from Epoch {best_epoch} with Validation Accuracy: {best_val_accuracy:.2f}%")
print(f"✅ Final Test Accuracy (on best model): {test_accuracy:.2f}%")
print(f"==========================================")