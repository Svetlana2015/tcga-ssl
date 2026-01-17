from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.ssl_pretrain import SSLPretrainConfig, run_pretrain


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    elif suf == ".csv":
        return pd.read_csv(path, index_col=0)
    else:
        raise ValueError("Supported formats: .parquet, .csv")


def _align_by_index(df_a: pd.DataFrame, df_b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align two tables by sample IDs stored in the index.

    Robust to common pitfalls:
    - int vs str indices (e.g., 0..N vs "0".."N")
    - leading/trailing whitespace
    - duplicated indices
    """
    df_a = df_a.copy()
    df_b = df_b.copy()

    # Normalize indices (fixes int vs str mismatches)
    df_a.index = df_a.index.astype(str).str.strip()
    df_b.index = df_b.index.astype(str).str.strip()

    # Drop duplicate indices (keep first)
    if df_a.index.has_duplicates:
        df_a = df_a[~df_a.index.duplicated(keep="first")]
    if df_b.index.has_duplicates:
        df_b = df_b[~df_b.index.duplicated(keep="first")]

    # If the indices are the same, then ok.
    if df_a.index.equals(df_b.index):
        return df_a, df_b

    # Otherwise, align at the intersection
    common = df_a.index.intersection(df_b.index)
    if len(common) == 0:
        raise ValueError(
            "Genes and pathways have no common sample IDs in index (after normalization).\n"
            "Make sure both tables use the same sample IDs as index."
        )

    df_a2 = df_a.loc[common].copy()
    df_b2 = df_b.loc[common].copy()

    # Important: same order
    df_b2 = df_b2.reindex(df_a2.index)

    return df_a2, df_b2


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SSL pretraining (genes + pathways).")
    parser.add_argument("--genes", required=True, help="Path to genes table (.parquet or .csv)")
    parser.add_argument("--pathways", required=True, help="Path to pathways table (.parquet or .csv)")

    # training params
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)

    # ssl params
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.5)

    # augmentation
    parser.add_argument("--noise_sigma", type=float, default=0.1)
    parser.add_argument("--drop_prob", type=float, default=0.1)

    # model dims
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)

    # output
    parser.add_argument("--save_dir", type=str, default="experiments/ssl_pretrain")

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    genes_path = (root / args.genes).resolve()
    pathways_path = (root / args.pathways).resolve()

    genes_df = _read_table(genes_path)
    pathways_df = _read_table(pathways_path)

    # alignment by sample-id (index)
    genes_df, pathways_df = _align_by_index(genes_df, pathways_df)

    cfg = SSLPretrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mask_ratio=args.mask_ratio,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        temperature=args.temperature,
        noise_sigma=args.noise_sigma,
        drop_prob=args.drop_prob,
        latent_dim=args.latent_dim,
        proj_dim=args.proj_dim,
        dropout=args.dropout,
        save_dir=args.save_dir,
    )

    print("Genes shape:", genes_df.shape)
    print("Pathways shape:", pathways_df.shape)
    print("Saving to:", cfg.save_dir)

    run_pretrain(genes_df, pathways_df, config=cfg)


if __name__ == "__main__":
    main()
