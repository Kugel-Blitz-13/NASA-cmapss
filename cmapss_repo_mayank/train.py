
import argparse
import json
import math
import os
import random
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
DATA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"

COL_NAMES = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def maybe_download_dataset(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    wanted = ["train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"]

    if all((data_dir / fname).exists() for fname in wanted):
        print("Dataset already exists.")
        return

    zip_path = data_dir / "cmapss.zip"
    print(f"Downloading dataset from {DATA_URL} ...")
    urllib.request.urlretrieve(DATA_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    for fname in wanted:
        dst = data_dir / fname
        if dst.exists():
            continue
        matches = list(data_dir.rglob(fname))
        if not matches:
            raise FileNotFoundError(f"Could not find {fname} after extracting {zip_path}")
        shutil.copy(matches[0], dst)
    print("Dataset ready.")

def load_cmapss_split(data_dir: Path, split: str) -> pd.DataFrame:
    if split == "train":
        path = data_dir / "train_FD001.txt"
    elif split == "test":
        path = data_dir / "test_FD001.txt"
    else:
        raise ValueError("split must be train or test")

    df = pd.read_csv(path, sep=r"\s+", header=None)
    # Some distributions contain 26 real columns + 2 empty columns at the end.
    if df.shape[1] > 26:
        df = df.iloc[:, :26]
    df.columns = COL_NAMES
    return df

def load_rul_truth(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "RUL_FD001.txt"
    rul = pd.read_csv(path, sep=r"\s+", header=None)
    rul = rul.iloc[:, :1]
    rul.columns = ["RUL"]
    rul["unit"] = np.arange(1, len(rul) + 1)
    return rul

def compute_train_rul(train_df: pd.DataFrame, max_rul: int) -> pd.DataFrame:
    max_cycle = train_df.groupby("unit")["cycle"].max().rename("max_cycle")
    out = train_df.merge(max_cycle, on="unit")
    out["RUL"] = out["max_cycle"] - out["cycle"]
    out["RUL"] = out["RUL"].clip(upper=max_rul)
    return out.drop(columns=["max_cycle"])

def choose_features(train_df: pd.DataFrame, std_threshold: float = 1e-8):
    feature_cols = [c for c in train_df.columns if c.startswith("op_setting_") or c.startswith("sensor_")]
    stds = train_df[feature_cols].std()
    keep = stds[stds > std_threshold].index.tolist()
    return keep

@dataclass
class PreparedData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    test_rul: pd.DataFrame
    feature_cols: list
    scaler: StandardScaler

def split_by_unit(train_df: pd.DataFrame, val_frac: float = 0.2, seed: int = SEED):
    units = sorted(train_df["unit"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = max(1, int(len(units) * val_frac))
    val_units = set(units[:n_val])
    train_units = set(units[n_val:])
    tr = train_df[train_df["unit"].isin(train_units)].copy()
    va = train_df[train_df["unit"].isin(val_units)].copy()
    return tr, va

def prepare_data(data_dir: Path, max_rul: int = 125, val_frac: float = 0.2) -> PreparedData:
    train_df = load_cmapss_split(data_dir, "train")
    test_df = load_cmapss_split(data_dir, "test")
    test_rul = load_rul_truth(data_dir)

    train_df = compute_train_rul(train_df, max_rul=max_rul)

    # For test, RUL is only defined at the end of each unit. We keep raw test_df and use truth later.
    feature_cols = choose_features(train_df)

    tr_df, va_df = split_by_unit(train_df, val_frac=val_frac, seed=SEED)

    scaler = StandardScaler()
    scaler.fit(tr_df[feature_cols])

    for df in [tr_df, va_df, test_df]:
        df.loc[:, feature_cols] = scaler.transform(df[feature_cols])

    return PreparedData(
        train_df=tr_df,
        val_df=va_df,
        test_df=test_df,
        test_rul=test_rul,
        feature_cols=feature_cols,
        scaler=scaler,
    )

class SequenceWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list, window_size: int, stride: int = 1, train_mode: bool = True):
        self.samples = []
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.train_mode = train_mode

        for unit_id, unit_df in df.groupby("unit"):
            unit_df = unit_df.sort_values("cycle").reset_index(drop=True)
            features = unit_df[feature_cols].values.astype(np.float32)

            if train_mode:
                targets = unit_df["RUL"].values.astype(np.float32)
                if len(unit_df) < window_size:
                    pad_len = window_size - len(unit_df)
                    pad_feats = np.repeat(features[:1], pad_len, axis=0)
                    x = np.concatenate([pad_feats, features], axis=0)
                    y = targets[-1]
                    self.samples.append((x, y))
                else:
                    for end in range(window_size - 1, len(unit_df), stride):
                        start = end - window_size + 1
                        x = features[start : end + 1]
                        y = targets[end]
                        self.samples.append((x, y))
            else:
                # One sample per engine: final window only
                if len(unit_df) < window_size:
                    pad_len = window_size - len(unit_df)
                    pad_feats = np.repeat(features[:1], pad_len, axis=0)
                    x = np.concatenate([pad_feats, features], axis=0)
                else:
                    x = features[-window_size:]
                self.samples.append((unit_id, x))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        if self.train_mode:
            x, y = item
            return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
        unit_id, x = item
        return unit_id, torch.from_numpy(x)

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred = self.head(last).squeeze(-1)
        return pred

class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNRegressor(nn.Module):
    def __init__(self, input_dim: int, channels=(64, 64, 64), kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size=kernel_size, dilation=2**i, dropout=dropout))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        feat = self.tcn(x)
        last = feat[:, :, -1]
        pred = self.head(last).squeeze(-1)
        return pred

def make_model(model_name: str, input_dim: int):
    if model_name == "lstm":
        return LSTMRegressor(input_dim=input_dim)
    if model_name == "tcn":
        return TCNRegressor(input_dim=input_dim)
    raise ValueError("model must be lstm or tcn")

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        batch_n = x.size(0)
        total_loss += loss.item() * batch_n
        n += batch_n
    return total_loss / max(1, n)

@torch.no_grad()
def evaluate_regression(model, loader, device):
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        preds.append(pred)
        ys.append(y.numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    return {
        "rmse": rmse(y_true, y_pred),
        "phm_score": phm_score(y_true, y_pred),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }

@torch.no_grad()
def evaluate_test_last_window(model, dataset: SequenceWindowDataset, test_rul: pd.DataFrame, device):
    model.eval()
    unit_ids, preds = [], []
    for unit_id, x in dataset:
        x = x.unsqueeze(0).to(device)
        pred = model(x).item()
        unit_ids.append(unit_id)
        preds.append(pred)

    pred_df = pd.DataFrame({"unit": unit_ids, "pred_rul": preds})
    merged = pred_df.merge(test_rul, on="unit", how="inner")
    y_true = merged["RUL"].values.astype(float)
    y_pred = merged["pred_rul"].values.astype(float)
    return {
        "rmse": rmse(y_true, y_pred),
        "phm_score": phm_score(y_true, y_pred),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }

def plot_learning_curves(history: dict, output_path: Path):
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_loss"], label="Train MSE")
    plt.plot(history["val_rmse"], label="Val RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_scatter(y_true, y_pred, output_path: Path):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("True vs Predicted RUL")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/CMAPSS")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["lstm", "tcn"], required=True)
    parser.add_argument("--window_size", type=int, default=30)
    parser.add_argument("--max_rul", type=int, default=125)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    maybe_download_dataset(data_dir)
    prepared = prepare_data(data_dir, max_rul=args.max_rul, val_frac=args.val_frac)

    train_ds = SequenceWindowDataset(prepared.train_df, prepared.feature_cols, window_size=args.window_size, stride=args.stride, train_mode=True)
    val_ds = SequenceWindowDataset(prepared.val_df, prepared.feature_cols, window_size=args.window_size, stride=1, train_mode=True)
    test_ds = SequenceWindowDataset(prepared.test_df, prepared.feature_cols, window_size=args.window_size, train_mode=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = make_model(args.model, input_dim=len(prepared.feature_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_rmse": [], "val_phm_score": []}
    best_state = None
    best_val_rmse = math.inf
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate_regression(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_phm_score"].append(val_metrics["phm_score"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f} | "
            f"val_phm={val_metrics['phm_score']:.2f}"
        )

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "feature_cols": prepared.feature_cols,
                "window_size": args.window_size,
                "max_rul": args.max_rul,
                "model_name": args.model,
            }
            torch.save(best_state, output_dir / "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_metrics = evaluate_regression(model, val_loader, device)
    test_metrics = evaluate_test_last_window(model, test_ds, prepared.test_rul, device)

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)

    plot_learning_curves(history, output_dir / "learning_curve.png")
    plot_scatter(test_metrics["y_true"], test_metrics["y_pred"], output_dir / "test_scatter.png")

    summary = {
        "model": args.model,
        "device": str(device),
        "window_size": args.window_size,
        "feature_count": len(prepared.feature_cols),
        "feature_cols": prepared.feature_cols,
        "val_rmse": val_metrics["rmse"],
        "val_phm_score": val_metrics["phm_score"],
        "test_rmse": test_metrics["rmse"],
        "test_phm_score": test_metrics["phm_score"],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
