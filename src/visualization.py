import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_baseline_performance(
    results_path="baseline_results.npz",
    figsize=(8, 5),
    show=True,
    save_path=None
):
    """
    Plot baseline MLP performance vs training set size.

    Parameters
    ----------
    results_path : str
        Path to .npz file produced by run_baseline
    figsize : tuple
        Figure size
    show : bool
        Whether to call plt.show()
    save_path : str or None
        If provided, saves the figure to this path
    """

    data = np.load(results_path)

    train_sizes = data["train_sizes"]
    mean_accs = data["mean_accs"]
    std_accs = data["std_accs"]

    plt.figure(figsize=figsize)

    plt.plot(
        train_sizes,
        mean_accs,
        marker="o",
        label="Mean accuracy"
    )

    plt.fill_between(
        train_sizes,
        mean_accs - std_accs,
        mean_accs + std_accs,
        alpha=0.2,
        label="±1 std"
    )

    plt.xlabel("Training subsample size")
    plt.ylabel("Accuracy on test")
    plt.title("MLP quality vs train size (stratification)")
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()




def _load_curve(csv_path: str) -> pd.DataFrame:
    """
    Compare baseline and SSL fine-tuning curves (frozen / unfrozen encoder)
    Reads CSV results and draws a single summary plot.
    
    Load a results CSV that contains:
      - proportion
      - mean
      - (optional) std
    Returns dataframe sorted by proportion.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"proportion", "mean"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{csv_path} must contain columns {sorted(required)}. "
            f"Found: {sorted(df.columns)}"
        )

    df = df.sort_values("proportion").reset_index(drop=True)
    return df


def load_baseline_npz(npz_path: str, full_train_size: int = 1000) -> pd.DataFrame:
    data = np.load(npz_path)

    train_sizes = data["train_sizes"].astype(float)
    proportions = train_sizes / float(full_train_size)

    df = pd.DataFrame({
        "proportion": proportions,
        "mean": data["mean_accs"],
        "std": data["std_accs"],
    })

    return df.sort_values("proportion").reset_index(drop=True)

def compare_training_curves(
    baseline_npz: str = "baseline_results.npz",
    ssl_frozen_csv: str = "results/ssl/ssl_finetune_frozen.csv",
    ssl_unfrozen_csv: str = "results/ssl/ssl_finetune_unfrozen.csv",
    title: str = "Baseline vs SSL Fine-tuning Performance",
    save_path: str | None = None,
    show: bool = True,
):
    # --- Load results ---
    df_base = load_baseline_npz(baseline_npz)
    df_frz = _load_curve(ssl_frozen_csv)
    df_unf = _load_curve(ssl_unfrozen_csv)

    # --- Plot ---
    plt.figure(figsize=(8, 5))

    plt.plot(
        df_base["proportion"],
        df_base["mean"],
        marker="o",
        label="Baseline MLP"
    )

    plt.plot(
        df_frz["proportion"],
        df_frz["mean"],
        marker="s",
        label="SSL Frozen Encoder"
    )

    plt.plot(
        df_unf["proportion"],
        df_unf["mean"],
        marker="^",
        label="SSL Unfrozen Encoder"
    )

    plt.xlabel("Training Proportion")
    plt.ylabel("Mean Test Accuracy")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[Saved] {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    print("[Done] Baseline vs SSL curves plotted.")

