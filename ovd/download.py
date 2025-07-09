from huggingface_hub import snapshot_download

# 원하는 경로로 다운로드
snapshot_download(
    repo_id="omlab/OVDEval",
    repo_type="dataset",
    local_dir="./ovdeval",  # 원하는 다운로드 위치
    local_dir_use_symlinks=False       # 실제 파일 복사 (심볼릭 링크 아님)
)
