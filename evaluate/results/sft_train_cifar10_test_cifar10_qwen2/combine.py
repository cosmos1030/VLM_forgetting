import os
import pandas as pd

base_dir = "."
acc_data = []

for subdir in os.listdir(base_dir):
    subpath = os.path.join(base_dir, subdir)
    if not os.path.isdir(subpath):
        continue

    # 정확히 step 폴더인 경우만 (예: 5000_steps)
    if not subdir.endswith("_steps"):
        continue

    step_str = subdir.replace("_steps", "")
    try:
        step = int(step_str)
    except ValueError:
        continue

    # accuracy 파일 찾기
    for filename in os.listdir(subpath):
        if filename.endswith("_accuracy.csv"):
            csv_path = os.path.join(subpath, filename)
            try:
                df = pd.read_csv(csv_path)
                acc = df["accuracy_percent"].iloc[0]
                acc_data.append({"step": step, "accuracy": acc})
            except Exception as e:
                print(f"[Error] {csv_path}: {e}")
            break

# 정렬 및 저장
df_all = pd.DataFrame(acc_data).sort_values("step")
print(df_all)
df_all.to_csv("all_steps_accuracy.csv", index=False)

