"""
Model alias ↔ HuggingFace Model ID Mapping Utility
You should add to MODEL_ALIASES if you want to use new model
"""

MODEL_ALIASES = {
    "qwen2.5vl-3b-instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2vl-2b-instruct"  : "Qwen/Qwen2-VL-2B-Instruct",
    "qwen2vl-2b"  : "Qwen/Qwen2-VL-2B",
    "visualthinker"        : "turningpoint-ai/VisualThinker-R1-Zero",
    "visualthinker-sft" : "cosmos1030/Qwen2_VL-2B-SFT_revised2"
}

def resolve_model_name(name_or_alias: str) -> str:
    """
    When the alias is given, covert to real HF ID, if it's HF ID itself, just return it
    """
    key = name_or_alias.lower().strip()
    return MODEL_ALIASES.get(key, name_or_alias)
