import os
import torch

# 사용할 GPU 인덱스 (문자열 형태로 쉼표로 구분)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 예시: GPU 0만 사용
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,2' # 예시: GPU 0과 2 사용

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} name: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")