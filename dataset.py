import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess_csv(csv_path, fixed_cols=None, feature_prefixes=None, extra_cols=None, label_col=None):
    """
    CSV 파일에서 feature 선택만 담당 (scaler는 get_dataloaders에서 적용)
    """
    df = pd.read_csv(csv_path)

    # ----- feature 선택 -----
    fixed_cols = fixed_cols if fixed_cols else []

    prefix_cols = []
    if feature_prefixes:
        for prefix in feature_prefixes:
            prefix_cols.extend([c for c in df.columns if c.startswith(prefix)])

    extra_cols = extra_cols if extra_cols else []

    feature_cols = fixed_cols + prefix_cols + extra_cols
    feature_cols = list(dict.fromkeys(feature_cols))  # 중복 제거

    # ----- 숫자형 변환 -----
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # ----- label 처리 -----
    if label_col is not None and label_col in df.columns:
        df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
        df[label_col] = df[label_col] - 1   # 1→0, 2→1

    return df, feature_cols, label_col

import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, df, feature_cols, label_col=None):
        self.df = df
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.num_classes = self.df[label_col].nunique() if label_col else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(row[self.feature_cols].values.astype(float), dtype=torch.float)
        if self.label_col:
            label = torch.tensor(row[self.label_col], dtype=torch.long)
            return features, label
        else:
            return features
