from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText

# cfg = AutoConfig.from_pretrained("turningpoint-ai/VisualThinker-R1-Zero")

cfg = AutoConfig.from_pretrained("turningpoint-ai/VisualThinker-R1-Zero",
                                 force_download=True,      # 캐시 무시
    resume_download=False,
    trust_remote_code=True )