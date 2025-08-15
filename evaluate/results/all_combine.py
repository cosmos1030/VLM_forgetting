#!/usr/bin/env python3
import os
import re
import pandas as pd

# 사용: python combine.py [BASE_DIR]
import sys
BASE_DIR = sys.argv[1] if len(sys.argv) > 1 else "."

step_dir_pat = re.compile(r"^(\d+)_steps$")  # 예: 5000_steps

rows = []
for root, dirs, files in os.walk(BASE_DIR):
    base = os.path.basename(root)
    m = step_dir_pat.match(base)
    if not m:
        continue

    step = int(m.group(1))
    # "실험 디렉토리"는 BASE_DIR 바로 아래 1레벨 폴더명으로 정의
    # (…/results/<EXP>/<STEP>_steps/… 형태 가정)
    rel = os.path.relpath(root, BASE_DIR)
    parts = rel.split(os.sep)
    exp = parts[0] if len(parts) >= 2 else "(unknown_exp)"

    # accuracy csv 찾기
    csv_path = None
    for f in files:
        if f.endswith("_accuracy.csv"):
            csv_path = os.path.join(root, f)
            break
    if csv_path is None:
        # 못 찾으면 스킵
        continue

    try:
        df = pd.read_csv(csv_path)
        # 컬럼 방어적 처리
        if "accuracy_percent" in df.columns:
            acc = float(df["accuracy_percent"].iloc[0])
        elif "accuracy" in df.columns:
            acc = float(df["accuracy"].iloc[0])
        else:
            raise ValueError(f"No accuracy column in {csv_path}")

        total = None
        correct = None
        for c in ["total_samples","total","n","num_samples"]:
            if c in df.columns:
                total = int(df[c].iloc[0]); break
        for c in ["correct_predictions","correct","num_correct"]:
            if c in df.columns:
                correct = int(df[c].iloc[0]); break

        rows.append({
            "exp": exp,
            "step": step,
            "accuracy_percent": acc,
            "total_samples": total,
            "correct_predictions": correct,
            "csv_path": csv_path,
        })
    except Exception as e:
        print(f"[Error] {csv_path}: {e}")

# 합치기
if not rows:
    print("No results found."); sys.exit(0)

df_all = pd.DataFrame(rows).sort_values(["exp","step"]).reset_index(drop=True)
print(df_all)

# 전체 저장
out_path = os.path.join(BASE_DIR, "all_steps_accuracy.csv")
df_all.to_csv(out_path, index=False)

# 실험별 best 저장(선택)
best = df_all.sort_values(["exp","accuracy_percent","step"], ascending=[True,False,True]) \
             .groupby("exp", as_index=False).first()
best.to_csv(os.path.join(BASE_DIR, "best_accuracy_per_exp.csv"), index=False)

