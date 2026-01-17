rom __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.ssl_pretrain import load_pretrained_encoder
from src.ssl_finetune import run_finetune


def read_table(path: Path) -> pd.DataFrame:
    """Read parquet or CSV table."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError("Input table must be .parquet or .csv")


def resolve_path(root: Path, p: str) -> Path:
    """Resolve path relative to repo root unless already absolute."""
    path = Path(p)
    return path if path.is_absolute() else (root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune / linear probe a pretrained SSL encoder."
    )

    parser.add_argument(
        "--finetune",
        required=True,
        help="Path to finetune dataset (features + label column)"
    )
    parser.add_argument(
        "--test",
        required=True,
        help="Path to test dataset (features + label column)"
    )

    parser.add_argument(
        "--weights",
        required=True,
        help="Path to pretrained SSL weights (.pth)"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to SSL config.json"
    )

    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder (linear probing). Default: frozen."
    )
    parser.add_argument(
        "--unfreeze_encoder",
        dest="freeze_encoder",
        action="store_false",
        help="Unfreeze encoder (full fine-tuning)."
    )
    parser.set_defaults(freeze_encoder=True)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--runs", type=int, default=5)

    parser.add_argument(
        "--save_dir",
        default="experiments/ssl_finetune",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--res_dir",
        default="results/ssl",
        help="Directory to save CSV results"
    )

    parser.add_argument(
        "--label_col",
        default="cancer_type",
        help="Label column name"
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    finetune_path = resolve_path(root, args.finetune)
    test_path = resolve_path(root, args.test)
    weights_path = resolve_path(root, args.weights)
    config_path = resolve_path(root, args.config)

    # ---------------------------
    # Load config (for input_dim check)
    # ---------------------------
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    expected_input_dim = cfg.get("input_dim", None)

    # ---------------------------
    # Load data
    # ---------------------------
    finetune_df = read_table(finetune_path)
    test_df = read_table(test_path)

    # ---------------------------
    # Load pretrained encoder
    # ---------------------------
    encoder = load_pretrained_encoder(
        weights_path=str(weights_path),
        config_path=str(config_path),
    )

    # ---------------------------
    # Run fine-tuning
    # ---------------------------
    run_finetune(
        encoder=encoder,
        finetune_df=finetune_df,
        test_df=test_df,
        epochs=args.epochs,
        batch_size=args.batch_size,
        runs=args.runs,
        freeze_encoder=args.freeze_encoder,
        save_dir=args.save_dir,
        res_dir=args.res_dir,
        label_col=args.label_col,
        expected_input_dim=expected_input_dim,
    )

    print("SSL fine-tuning completed successfully.")


if __name__ == "__main__":
    main()

