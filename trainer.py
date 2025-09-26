import os
import json
import torch

class Trainer:
    def __init__(self, model, config, device, fold=None):
        self.model = model
        self.config = config
        self.device = device
        self.fold = fold

        # optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"]
        )

        # loss function 선택
        if config["training"]["loss"] == "bce":
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

        # 로그 저장할 디렉토리
        self.save_dir = os.path.join(config["experiment"]["save_dir"], f"fold{fold}")
        os.makedirs(self.save_dir, exist_ok=True)

        # 학습 기록용
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_auc": []}

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(x)

            loss = self.criterion(out, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        return total_loss / total, correct / total

    def validate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                loss = self.criterion(out, y)

                total_loss += loss.item() * x.size(0)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        acc = correct / total

        # ROC-AUC 추가 (binary/multi-class 자동 처리)
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(all_labels, all_preds, multi_class="ovr")
        except ValueError:
            auc = None

        return total_loss / total, acc, auc

    def fit(self, train_loader, val_loader, train_idx=None, val_idx=None):
        best_acc, best_auc = 0, 0
        for epoch in range(self.config["training"]["epochs"]):
            tr_loss, tr_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc, val_auc = self.validate(val_loader)

            # 기록 저장
            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_auc"].append(val_auc)

            print(f"[Fold {self.fold}] Epoch {epoch+1}: "
                  f"Train loss {tr_loss:.4f}, acc {tr_acc:.4f} | "
                  f"Val loss {val_loss:.4f}, acc {val_acc:.4f}, auc {val_auc}")

            # best model 저장
            if val_acc > best_acc:
                print('The best!')
                best_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, "best_model.pt"))

        # ---- 최종 로그/인덱스 저장 ----
        log_path = os.path.join(self.save_dir, "log.json")
        with open(log_path, "w") as f:
            json.dump({
                "history": self.history,
                "best_val_acc": best_acc,
                "best_val_auc": max([x for x in self.history["val_auc"] if x is not None]),
                "train_idx": train_idx.tolist() if train_idx is not None else None,
                "val_idx": val_idx.tolist() if val_idx is not None else None
            }, f, indent=2)
