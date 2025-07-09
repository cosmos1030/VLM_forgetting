from huggingface_hub import HfApi, create_repo

repo_id = "cosmos1030/Qwen2.5_VL-3B-GUI-SFT"

# 1. Public 저장소로 생성 (이미 생성되어 있다면 이 줄은 생략 가능)
create_repo(repo_id, repo_type="model", private=False, exist_ok=True)

# 2. 대용량 폴더 업로드
api = HfApi()
api.upload_large_folder(
    folder_path="qwen2.5-vl-3b-sft-gui-defect",
    repo_id=repo_id,
    repo_type="model"
)
