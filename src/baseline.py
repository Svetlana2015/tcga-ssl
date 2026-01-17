import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# dataset
class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Train one run

def train_one_run(
    X_sub, y_sub,
    X_test, y_test,
    device,
    nb_classes,
    epochs=25,
    batch_size=32,
    lr=1e-3,
    dropout=0.0
):
    # Scaling
    scaler = StandardScaler()
    X_sub = scaler.fit_transform(X_sub)
    X_test = scaler.transform(X_test)

    # Datasets
    train_ds = NumpyDataset(X_sub.astype(np.float32), y_sub)
    test_ds = NumpyDataset(X_test.astype(np.float32), y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = MLP(
        input_dim=X_sub.shape[1],
        num_classes=nb_classes,
        dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

    # Test
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.append(preds)
            all_true.append(yb.numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    acc = accuracy_score(all_true, all_preds)
    return acc

def run_baseline(
    finetune_path,
    test_path,
    epochs=25,
    batch_size=32,
    lr=1e-3,
    dropout=0.0,
    n_repeats=5,
    out_path="baseline_results.npz",
):
    # Load data
    df_finetune = pd.read_parquet(finetune_path)
    df_test = pd.read_parquet(test_path)

    feature_cols = [c for c in df_test.columns if c != "cancer_type"]

    X_train = df_finetune[feature_cols].values.astype(np.float32)
    y_train = df_finetune["cancer_type"].values

    X_test = df_test[feature_cols].values.astype(np.float32)
    y_test = df_test["cancer_type"].values

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    nb_classes = len(le.classes_)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_sizes = list(range(100, len(X_train) + 1, 100))
    if train_sizes[-1] != len(X_train):
        train_sizes.append(len(X_train))

    results = {}

    for n in train_sizes:
        accs = []

        for r in range(n_repeats):
            if n == len(X_train):
                X_sub = X_train
                y_sub = y_train_enc
            else:
                X_sub, _, y_sub, _ = train_test_split(
                    X_train,
                    y_train_enc,
                    train_size=n,
                    stratify=y_train_enc,
                    random_state=42 + r
                )

            acc = train_one_run(
                X_sub, y_sub,
                X_test, y_test_enc,
                device=device,
                nb_classes=nb_classes,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                dropout=dropout
            )

            accs.append(acc)

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)

        results[n] = (mean_acc, std_acc)
        print(f"n={n}: mean_acc={mean_acc:.4f} ± {std_acc:.4f}")

    
    np.savez(
        out_path,
        train_sizes=list(results.keys()),
        mean_accs=[v[0] for v in results.values()],
        std_accs=[v[1] for v in results.values()],
    )

    print(f"Baseline results saved to {out_path}")

    return results


