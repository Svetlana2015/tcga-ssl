"""
Fine-tuning / linear probing on top of a pretrained SSL encoder.

This module automatically encodes string labels (e.g. "TCGA-KIRP") into integers
using a shared mapping built from finetune_df + test_df.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


# 1. DataFrame to (X, y) tensors
def df_to_tensor(df, label_col="cancer_type"):
    """
    Converts DataFrame to (X, y) torch tensors.
    Assumes label_col is already integer-encoded (0..K-1).
    """
    df = df.copy()

    # X: numeric features only
    X_df = df.drop(columns=[label_col], errors="ignore")
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X = torch.tensor(X_df.values, dtype=torch.float32)

    # y: integer labels
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataframe.")
    y = torch.tensor(df[label_col].values, dtype=torch.long)

    return X, y


# 2. Classifier head
class SSLClassifier(nn.Module):
    def __init__(self, encoder, latent_dim, num_classes, freeze_encoder=True):
        super().__init__()

        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        # Optionally freeze encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Linear classifier
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                z = self.encoder(x)
        else:
            z = self.encoder(x)

        return self.classifier(z)


# 3. Single-run fine-tuning
def finetune_single_run(
    encoder, X_finetune, y_finetune, X_test, y_test,
    num_classes, epochs=50, lr=5e-4, batch_size=32,
    freeze_encoder=True
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move encoder to device
    encoder = encoder.to(device)
    encoder.eval() if freeze_encoder else encoder.train()

    # Move tensors to device
    X_finetune, y_finetune = X_finetune.to(device), y_finetune.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # Detect latent dimension
    with torch.no_grad():
        latent_dim = encoder(X_finetune[:1]).shape[1]

    model = SSLClassifier(
        encoder=encoder,
        latent_dim=latent_dim,
        num_classes=num_classes,
        freeze_encoder=freeze_encoder,
    ).to(device)

    loader = DataLoader(
        TensorDataset(X_finetune, y_finetune),
        batch_size=batch_size,
        shuffle=True
    )

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test.cpu().numpy(), preds)
    return acc, model


# 4. Multi-proportion fine-tuning (+ LabelEncoder)
def run_finetune(
    encoder,
    finetune_df,
    test_df,
    proportions=None,
    runs=5,
    epochs=50,
    batch_size=32,
    freeze_encoder=True,
    save_dir="experiments/ssl_finetune",
    res_dir="results/ssl",
    label_col="cancer_type",
    save_label_encoder=True
):
    """
    Fine-tunes a linear classifier on top of a pretrained encoder.

    Adds label encoding so label_col can be string labels.
    """

    finetune_df = finetune_df.copy()
    test_df = test_df.copy()

    # Create folders
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # ---------------------------
    # Label encoding (NEW)
    # ---------------------------
    if label_col not in finetune_df.columns or label_col not in test_df.columns:
        raise ValueError(f"Both dataframes must contain label column '{label_col}'.")

    le = LabelEncoder()
    le.fit(finetune_df[label_col].astype(str))

    # Ensure test has no unseen labels
    test_labels = set(test_df[label_col].astype(str).unique())
    train_labels = set(le.classes_)
    unseen = sorted(list(test_labels - train_labels))
    if len(unseen) > 0:
        raise ValueError(
            "Test set contains labels not present in fine-tune set. "
            f"Unseen labels: {unseen[:10]}{'...' if len(unseen) > 10 else ''}"
        )

    finetune_df[label_col] = le.transform(finetune_df[label_col].astype(str))
    test_df[label_col] = le.transform(test_df[label_col].astype(str))

    if save_label_encoder:
        np.save(os.path.join(save_dir, "label_encoder_classes.npy"), le.classes_)

    # ---------------------------
    # Convert to tensors
    # ---------------------------
    X_finetune, y_finetune = df_to_tensor(finetune_df, label_col=label_col)
    X_test, y_test = df_to_tensor(test_df, label_col=label_col)

    # Determine mode name
    mode = "frozen" if freeze_encoder else "unfrozen"

    if proportions is None:
        proportions = np.linspace(0.1, 1.0, 10)

    # Prepare numpy arrays for stratified split
    X_np = X_finetune.numpy()
    y_np = y_finetune.numpy()

    num_classes = len(np.unique(y_np))

    results = {}
    best_acc = -1.0
    best_state = None

    for p in proportions:
        results[p] = []

        if float(p) == 1.0:
            # Full dataset training
            for run_id in range(1, runs + 1):
                acc, model = finetune_single_run(
                    encoder, X_finetune, y_finetune, X_test, y_test,
                    num_classes=num_classes,
                    epochs=epochs,
                    freeze_encoder=freeze_encoder,
                    batch_size=batch_size
                )

                results[p].append(acc)

                ckpt = os.path.join(save_dir, f"ssl_prop_{int(p*100)}_run{run_id}_{mode}.pth")
                torch.save(model.state_dict(), ckpt)

                if acc > best_acc:
                    best_acc = acc
                    best_state = model.state_dict()

        else:
            splitter = StratifiedShuffleSplit(
                n_splits=runs, train_size=float(p), random_state=42
            )

            for run_id, (idx, _) in enumerate(splitter.split(X_np, y_np), start=1):
                X_sub = X_finetune[idx]
                y_sub = y_finetune[idx]

                acc, model = finetune_single_run(
                    encoder, X_sub, y_sub, X_test, y_test,
                    num_classes=num_classes,
                    epochs=epochs,
                    freeze_encoder=freeze_encoder,
                    batch_size=batch_size
                )

                results[p].append(acc)

                ckpt = os.path.join(save_dir, f"ssl_prop_{int(p*100)}_run{run_id}_{mode}.pth")
                torch.save(model.state_dict(), ckpt)

                if acc > best_acc:
                    best_acc = acc
                    best_state = model.state_dict()

        print(f"Prop {p:.1f} → Mean Acc = {np.mean(results[p]):.4f}")

    # Save best model
    torch.save(best_state, os.path.join(save_dir, f"best_model_{mode}.pth"))

    # Save results csv
    rows = []
    for p, acc_list in results.items():
        row = {
            "proportion": float(p),
            "mean": float(np.mean(acc_list)),
            "std": float(np.std(acc_list)),
        }
        # Save up to first 5 runs
        for i, v in enumerate(acc_list[:5], start=1):
            row[f"run{i}"] = float(v)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(res_dir, f"ssl_finetune_{mode}.csv"), index=False)

    return df_out