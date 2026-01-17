from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

from src.baseline import run_baseline


def _check_inputs(parquet_path: Path, label_col: str = "cancer_type") -> None:
    if not parquet_path.exists():
        raise FileNotFoundError(f"File not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if label_col not in df.columns:
        raise ValueError(
            f"Column '{label_col}' is missing in {parquet_path}.\n"
            f"Columns example: {list(df.columns[:30])}\n\n"
            f"Your parquet must include '{label_col}' column."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", default="data/finetune.parquet")
    parser.add_argument("--test", default="data/test.parquet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--out", default="baseline_results.npz")

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    finetune_path = (root / args.finetune).resolve()
    test_path = (root / args.test).resolve()
    out_path = (root / args.out).resolve()

    _check_inputs(finetune_path)
    _check_inputs(test_path)

    run_baseline(
        finetune_path=str(finetune_path),
        test_path=str(test_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        n_repeats=args.n_repeats,
        out_path=str(out_path),
    )

    print(f"Done. Results saved to: {out_path}")


if __name__ == "__main__":
    main()