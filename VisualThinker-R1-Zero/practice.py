from transformers import AutoModelForImageTextToText

# trust_remote_code=True 로 remote custom 코드를 통째로 불러와서 모델 로드
model = AutoModelForImageTextToText.from_pretrained(
    "turningpoint-ai/VisualThinker-R1-Zero",
    trust_remote_code=True
)

# 이걸로 config에 접근하면 절대 에러 안 납니다
cfg = model.config
print(cfg.model_type)   # → qwen2_vl
