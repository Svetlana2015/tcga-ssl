import argparse
from pathlib import Path
from src.preprocessing import preprocess_to_parquet, SplitSizes


def main():
    parser = argparse.ArgumentParser(description="Generate pretrain/finetune/test parquet files.")
    parser.add_argument("--expr", required=True, help="Path to raw expression parquet (must contain caseID)")
    parser.add_argument("--labels", required=True, help="Path to raw labels parquet (must contain cases,cancer_type)")
    parser.add_argument("--out_dir", default="data", help="Output dir (default: data)")
    parser.add_argument("--test_n", type=int, default=1000)
    parser.add_argument("--finetune_n", type=int, default=1000)
    parser.add_argument("--pretrain_n", type=int, default=7000,
                        help="Pretrain size. Use -1 to take all remaining.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    expr_path = Path(args.expr)
    labels_path = Path(args.labels)

    if not expr_path.is_absolute():
        expr_path = (root / expr_path).resolve()
    if not labels_path.is_absolute():
        labels_path = (root / labels_path).resolve()

    out_dir = (root / args.out_dir).resolve()

    pretrain_n = None if args.pretrain_n == -1 else args.pretrain_n
    sizes = SplitSizes(test=args.test_n, finetune=args.finetune_n, pretrain=pretrain_n)

    preprocess_to_parquet(
        expr_path=str(expr_path),
        label_path=str(labels_path),
        output_dir=str(out_dir),
        sizes=sizes,
        seed=args.seed,
        scale=True,
        save_full=True,
        save_unused=True,
    )


if __name__ == "__main__":
    main()
