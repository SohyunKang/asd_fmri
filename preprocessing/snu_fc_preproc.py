import os
import glob
import numpy as np
import pandas as pd

# ----- 경로 설정 -----
data_dir = "/storage/shared/SNUH_fMRI_project/C-PAC/0831_result_rerun/FC_upload/aal"   # <-- sub-*.npy 파일들이 있는 폴더
save_csv = "/home/sohyunkang/asd_preliminary/SNU_ASD/data/snu_subjects_fc_upper2.csv"

# # ----- 파일 찾기 -----
files = sorted(glob.glob(os.path.join(data_dir, "sub-*.npy")))
if not files:
    print("No .npy files found")
    exit()

all_rows = []
cols = None
triu_idx = None
n_roi_ref = None

for f in files:
    subj = os.path.basename(f).replace(".npy", "")
    mat = np.load(f)  # (ROI x ROI) 행렬

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        print(f"⚠️ Warning: {subj} invalid shape {mat.shape}, skipped.")
        continue

    n_roi = mat.shape[0]
    if n_roi_ref is None:
        n_roi_ref = n_roi
        triu_idx = np.triu_indices(n_roi_ref, k=1)  # upper triangle index (대각 제외)
        cols = ["Subject"] + [f"ROI{i+1}_ROI{j+1}" for i, j in zip(*triu_idx)]
    elif n_roi != n_roi_ref:
        print(f"⚠️ Warning: {subj} ROI mismatch ({n_roi} vs {n_roi_ref}), skipped.")
        continue

    fc_flat = mat[triu_idx]
    row = [subj] + fc_flat.tolist()
    all_rows.append(row)

# ----- CSV 저장 -----
if all_rows:
    df = pd.DataFrame(all_rows, columns=cols)
    df.to_csv(save_csv, index=False)
    print(f"✅ Saved {save_csv}, shape={df.shape}")
else:
    print("⚠️ No valid matrices processed.")

import pandas as pd
import re

# ----- 경로 -----
meta_excel = "/home/sohyunkang/asd_preliminary/SNU_ASD/data/integrated_basic_demo_sohyun.csv"                  # 메타데이터
protocol_excel = '/home/sohyunkang/asd_preliminary/SNU_ASD/data/demo_snu.xlsx'
new_save_csv = "/home/sohyunkang/asd_preliminary/SNU_ASD/data/snu_subjects_fc_with_demo2.csv"

# ----- 데이터 불러오기 -----
df_fc = pd.read_csv(save_csv)
df_meta = pd.read_csv(meta_excel, encoding='cp949')
df_protocol = pd.read_excel(protocol_excel)


print(df_fc.head(10))
# ----- FC Subject 전처리 -----
# sub-asi9 → asi9

import re
import unicodedata

# "sub-asi9" → "asi9"

# 원본 확인
print("Before clean:", df_fc["Subject"].head().tolist())

# asi009 → asi9 (뒤의 숫자 0 제거)
def normalize_id(x):
    x = str(x).strip().lower()
    m = re.match(r"([a-zA-Z]+)(0*)(\d+)$", str(x))
    if m:
        prefix, _, num = m.groups()
        return f"{prefix}{int(num)}"   # 숫자를 int로 바꿔 앞의 0 제거
    return str(x)

# 진짜로 sub- 제거
df_fc["match_id"] = df_fc["Subject"].str.replace("sub-", "", regex=False).apply(normalize_id)

# 결과 확인
print("After clean:", df_fc["match_id"].head().tolist())

# ----- Meta Subject 전처리 -----
df_meta["match_id"] = df_meta["new_id"].apply(normalize_id)
df_protocol["match_id"] = df_protocol["MRI list"].apply(normalize_id)

# Group → Label (ASD=1, TD=0), protocol (구:1, 신:2)
df_meta["Label"] = df_meta["group"]
df_protocol["protocol"] = df_protocol["구:1, 신:2"]

# ----- Merge -----
print(df_fc.head(5), df_meta.head(5), df_protocol.head(5))

df_merged = df_fc.merge(df_meta[["match_id", "Label", "SEX", "age_MRI"]], on="match_id", how="left")
df_merged = df_merged.merge(df_protocol[["match_id", "protocol"]], on="match_id", how="left")

# 매칭 안된 케이스 확인
unmatched = df_merged[(df_merged["Label"].isna() | df_merged["age_MRI"].isna() | df_merged["protocol"].isna())]

if not unmatched.empty:
    print("⚠️ 매칭 실패 subjects:", len(unmatched), '개')
    print(unmatched["match_id"])

# Label 없는 행 제거 (원하면 keep 가능)
df_merged = df_merged.dropna(subset=["Label"])
df_merged = df_merged.dropna(subset=["age_MRI"])
df_merged = df_merged.dropna(subset=["protocol"])
df_merged["Label"] = df_merged["Label"].astype(int)
df_merged["age_MRI"] = df_merged["age_MRI"].astype(int)
df_merged["protocol"] = df_merged["protocol"].astype(int)

print(df_merged.head(5))
# ----- 저장 -----
df_merged.to_csv(new_save_csv, index=False)
print(f"✅ Saved merged file with demos → {new_save_csv}, shape={df_merged.shape}")
