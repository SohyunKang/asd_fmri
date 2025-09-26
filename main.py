# ----- 표준 라이브러리 -----
import os
import argparse
import yaml
import numpy as np

# ----- PyTorch 관련 -----
import torch
from torch.utils.data import DataLoader

# ----- Scikit-learn -----
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

# ----- 사용자 정의 모듈 -----
from dataset import CSVDataset, preprocess_csv
from models import get_model
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Training Pipeline")
    parser.add_argument("--config", type=str, default="./config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name to override config (e.g., mlp)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--folds", type=int, default=5,
                        help="Total number of CV folds")
    return parser.parse_args()


def main():
    # ----- Parse args -----
    args = parse_args()

    # ----- Load config -----
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.model is not None:
        config["model"]["name"] = args.model

    # 실행할 fold 번호 (config에서 가져오기)
    target_fold = config["experiment"].get("fold_num", 0)

    # ----- Data Load -----
    df, feature_cols, label_col = preprocess_csv(
        csv_path=config["data"]["csv_path"],
        fixed_cols=config["data"].get("fixed_cols"),
        feature_prefixes=config["data"].get("feature_prefixes"),
        extra_cols=config["data"].get("extra_cols"),
        label_col=config["data"].get("label_col"),
    )

    X = df[feature_cols].values
    y = df[label_col].values if label_col else None

    # ----- K-Fold -----
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        if fold != target_fold:
            continue
        print(f"\n========== Running Fold {fold+1}/{args.folds} ==========")

        # split
        train_df, val_df = df.iloc[train_idx].copy(), df.iloc[val_idx].copy()

        # normalization (train 기준 fit)
        scaler_type = config["data"].get("scaler", "standard")
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = None

        if scaler:
            scaler.fit(train_df[feature_cols])
            train_df[feature_cols] = scaler.transform(train_df[feature_cols])
            val_df[feature_cols] = scaler.transform(val_df[feature_cols])

        # Dataset & Dataloader
        train_set = CSVDataset(train_df, feature_cols, label_col)
        val_set = CSVDataset(val_df, feature_cols, label_col)

        train_loader = DataLoader(train_set,
                                  batch_size=config["data"]["batch_size"],
                                  shuffle=True,
                                  num_workers=config["data"]["num_workers"],
                                  pin_memory=True)

        val_loader = DataLoader(val_set,
                                batch_size=config["data"]["batch_size"],
                                shuffle=False,
                                num_workers=config["data"]["num_workers"],
                                pin_memory=True)

        # input/output dim 자동
        input_dim = len(feature_cols)
        output_dim = train_set.num_classes
        config["model"]["input_dim"] = input_dim
        config["model"]["output_dim"] = output_dim

        # Model
        device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        model = get_model(config["model"], input_dim, output_dim).to(device)

        # Optimizer & Scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"].get("lr", 1e-4),
            weight_decay=config["training"].get("weight_decay", 1e-5)
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Trainer (fold 번호 포함)
        trainer = Trainer(
            model=model,
            device=device,
            config=config,
            fold=fold   # 👈 fold 번호 전달
        )

        # Train & Validate (index도 넘겨서 log.json에 저장)
        trainer.fit(train_loader, val_loader, train_idx=train_idx, val_idx=val_idx)

        # 마지막 fold 결과 출력
        val_loss, val_acc, val_auc = trainer.validate(val_loader)
        print(f"Fold {fold+1} Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")


if __name__ == "__main__":
    main()
