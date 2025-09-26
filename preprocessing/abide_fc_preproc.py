import os
import glob
import pandas as pd
import numpy as np

# 데이터 경로
data_dir = "/storage/shared/fMRI/ABIDE1/preprocessed_dataset/Outputs/cpac/filt_global/rois_aal"
save_all_csv = "/home/sohyunkang/asd_preliminary/SNU_ASD/data/all_subjects_fc_upper.csv"
save_clean_csv = "/home/sohyunkang/asd_preliminary/SNU_ASD/data/all_subjects_fc_upper_clean.csv"
# save_qc_csv = "/home/sohyunkang/asd_preliminary/SNU_ASD/data/qc_report.csv"
# save_qc_names_csv = "/home/sohyunkang/asd_preliminary/SNU_ASD/data/qc_report_with_names.csv"

# # ----- Atlas 이름을 txt에서 불러오기 -----
# atlas_file = "/home/sohyunkang/asd_preliminary/SNU_ASD/data/aal116_labels.txt"
# with open(atlas_file, "r") as f:
#     AAL116_ROI = [line.strip() for line in f if line.strip()]

# files = sorted(glob.glob(os.path.join(data_dir, "*_rois_aal.1D")))
# if not files:
#     print("No files found")
#     exit()

# all_rows = []
# clean_rows = []
# qc_rows = []
# roi_problem_counts = np.zeros(116, dtype=int)  # ROI별 std=0 subject 수
# cols = None
# triu_idx = None
# n_roi_ref = None

# for f in files:
#     subj = os.path.basename(f).replace("_rois_aal.1D", "")
#     ts = np.genfromtxt(f, comments="#")

#     if ts.ndim != 2:
#         print(f"⚠️ Warning: {subj} invalid shape {ts.shape}, skipped.")
#         continue

#     n_time, n_roi = ts.shape
#     if n_roi != 116:
#         print(f"⚠️ Warning: {subj} invalid roi {n_roi}, skipped.")
#         continue

#     # ----- QC: 표준편차 0 ROI -----
#     stds = np.std(ts, axis=0)
#     zero_std_idx = np.where(stds < 1e-8)[0]

#     qc_rows.append({
#         "Subject": subj,
#         "n_zero_roi": len(zero_std_idx),
#         "zero_roi_indices": zero_std_idx.tolist()
#     })

#     # ROI별 카운트
#     for idx in zero_std_idx:
#         roi_problem_counts[idx] += 1

#     if len(zero_std_idx) > 0:
#         print(f"⚠️ {subj}: {len(zero_std_idx)} ROI(s) have std=0")

#     # 첫 valid 파일에서 upper-triangle index 정의
#     if n_roi_ref is None:
#         n_roi_ref = n_roi
#         triu_idx = np.triu_indices(n_roi_ref, k=1)
#         cols = ["Subject"] + [f"ROI{i+1}_ROI{j+1}" for i, j in zip(*triu_idx)]
#     elif n_roi != n_roi_ref:
#         print(f"⚠️ Warning: {subj} ROI mismatch ({n_roi} vs {n_roi_ref}), skipped.")
#         continue

#     # FC 계산
#     fc = np.corrcoef(ts.T)
#     fc_flat = fc[triu_idx]

#     row = [subj] + fc_flat.tolist()
#     all_rows.append(row)
#     if len(zero_std_idx) == 0:
#         clean_rows.append(row)

# # ----- CSV 저장 -----
# if all_rows:
#     df_all = pd.DataFrame(all_rows, columns=cols)
#     df_all.to_csv(save_all_csv, index=False)
#     print(f"✅ All subjects saved → {save_all_csv}, shape={df_all.shape}")

# if clean_rows:
#     df_clean = pd.DataFrame(clean_rows, columns=cols)
#     df_clean.to_csv(save_clean_csv, index=False)
#     print(f"✅ Clean subjects saved → {save_clean_csv}, shape={df_clean.shape}")

# if qc_rows:
#     df_qc = pd.DataFrame(qc_rows)
#     df_qc.to_csv(save_qc_csv, index=False)
#     print(f"✅ QC report saved → {save_qc_csv}, shape={df_qc.shape}")


# # ----- Summary 텍스트 저장 -----
# summary_file = "/home/sohyunkang/asd_preliminary/SNU_ASD/data/qc_summary.txt"

# total_subjects = len(all_rows)
# problem_subjects = sum(1 for r in qc_rows if r["n_zero_roi"] > 0)
# mean_problem_rois = (
#     np.mean([r["n_zero_roi"] for r in qc_rows if r["n_zero_roi"] > 0])
#     if problem_subjects > 0 else 0
# )

# lines = []
# lines.append(f"총 subject 수: {total_subjects}")
# lines.append(f"std=0 ROI가 있는 subject 수: {problem_subjects} ({problem_subjects/total_subjects*100:.2f}%)")
# lines.append(f"해당 subject들의 평균 문제 ROI 개수: {mean_problem_rois:.2f}\n")
# lines.append("ROI별 std=0 발생 빈도:")

# for i, count in enumerate(roi_problem_counts, 1):
#     if count > 0:
#         roi_name = AAL116_ROI[i-1] if i <= len(AAL116_ROI) else f"ROI{i}"
#         lines.append(f" - ROI {i} ({roi_name}): {count} subjects")

# with open(summary_file, "w") as f:
#     f.write("\n".join(lines))

# print(f"✅ QC summary text saved → {summary_file}")


# ----- 경로 -----
meta_excel = "/home/sohyunkang/asd_preliminary/SNU_ASD/data/Phenotypic_V1_0b_preprocessed1.csv"                  # 메타데이터
new_save_csv = "/home/sohyunkang/asd_preliminary/SNU_ASD/data/abide_subjects_fc_with_demo.csv"

# ----- 데이터 불러오기 -----
df_fc = pd.read_csv(save_clean_csv)
df_meta = pd.read_csv(meta_excel, encoding='cp949')

print(df_fc.head(10))
# ----- FC Subject 전처리 -----
# sub-asi9 → asi9

import re
import unicodedata

# 진짜로 sub- 제거
df_fc["match_id"] = df_fc["Subject"]

# ----- Meta Subject 전처리 -----
df_meta["match_id"] = df_meta["FILE_ID"]

# Group → Label (ASD=1, TD=0), protocol (구:1, 신:2)
df_meta["Label"] = df_meta["DX_GROUP"]
df_meta["Site"] = df_meta["SITE_ID"].astype('category').cat.codes

# ----- Merge -----
print(df_fc.head(5), df_meta.head(5))

df_merged = df_fc.merge(df_meta[["match_id", "Label", "SEX", "AGE_AT_SCAN", "Site", "DSM_IV_TR"]], on="match_id", how="left")

# 매칭 안된 케이스 확인
unmatched = df_merged[(df_merged["Label"].isna() | df_merged["SEX"].isna()| df_merged["AGE_AT_SCAN"].isna() | df_merged["Site"].isna())]

if not unmatched.empty:
    print("⚠️ 매칭 실패 subjects:", len(unmatched), '개')
    print(unmatched["match_id"])

# Label 없는 행 제거 (원하면 keep 가능)
df_merged = df_merged.dropna(subset=["Label"])
df_merged = df_merged.dropna(subset=["SEX"])
df_merged = df_merged.dropna(subset=["AGE_AT_SCAN"])
df_merged = df_merged.dropna(subset=["Site"])
df_merged["Label"] = df_merged["Label"].astype(int)
df_merged["SEX"] = df_merged["SEX"].astype(int)
df_merged["AGE_AT_SCAN"] = df_merged["AGE_AT_SCAN"].astype(int)
df_merged["Site"] = df_merged["Site"].astype(int)

print(df_merged.head(5))
# ----- 저장 -----
df_merged.to_csv(new_save_csv, index=False)
print(f"✅ Saved merged file with demos → {new_save_csv}, shape={df_merged.shape}")
