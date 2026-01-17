from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from src.mapping import run_mapping
from src.pathways import run_pathway


def read_table(path: Path) -> pd.DataFrame:
    """Read input table from .parquet or .csv."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        # If your CSV has an index column, you may set index_col=0 here.
        return pd.read_csv(path)

    raise ValueError("Input must be a .parquet or .csv file")


def save_parquet(df: pd.DataFrame, out_path: Path) -> None:
    """Save dataframe as parquet (always with index)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=True)


def split_genes_and_meta(
    mapped: pd.DataFrame,
    label_col: str = "cancer_type",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split mapped dataframe into:
    - gene_df: numeric gene expression matrix (samples x genes)
    - meta_df: metadata columns (label_col + object dtype columns)
    """
    # Keep label column and any object columns as metadata
    meta_cols: list[str] = []
    for c in mapped.columns:
        if c == label_col:
            meta_cols.append(c)
        elif mapped[c].dtype == "object":
            meta_cols.append(c)

    meta_df = mapped[meta_cols].copy() if meta_cols else pd.DataFrame(index=mapped.index)

    # Genes are everything else (usually numeric)
    gene_df = mapped.drop(columns=meta_cols, errors="ignore")

    if gene_df.empty:
        raise ValueError(
            "After mapping, gene matrix is empty. "
            "Check that input contains ENSG columns and mapping succeeded."
        )

    return gene_df, meta_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run mapping (ENSG->SYMBOL) + ssGSEA pathways and save outputs to fixed folders."
    )

    # Inputs
    parser.add_argument("--input", required=True, help="Expression table (.parquet or .csv)")
    parser.add_argument("--gmt", required=True, help="Path to GMT file (e.g., data/kegg.gmt)")

    # Naming
    parser.add_argument(
        "--name",
        required=True,
        help="Dataset name used for output files (e.g., finetune, pretrain, test)",
    )

    # Mapping options
    parser.add_argument("--case_id_col", default="caseID", help="Column with sample IDs (will become index if present).")
    parser.add_argument("--label_col", default="cancer_type", help="Label column to keep (if present).")

    parser.add_argument("--drop_unmapped", action="store_true", help="Drop genes that cannot be mapped to symbols.")
    parser.add_argument("--no_drop_unmapped", dest="drop_unmapped", action="store_false")
    parser.set_defaults(drop_unmapped=True)

    parser.add_argument("--aggregate_duplicates", action="store_true", help="Aggregate duplicated symbols by mean.")
    parser.add_argument("--no_aggregate_duplicates", dest="aggregate_duplicates", action="store_false")
    parser.set_defaults(aggregate_duplicates=True)

    # Pathways options
    parser.add_argument("--batch_size", type=int, default=200, help="Batch size for ssGSEA loop.")

    args = parser.parse_args()

    # Resolve repo root: scripts/make_pathways.py -> parents[1]
    root = Path(__file__).resolve().parents[1]

    input_path = (root / args.input).resolve()
    gmt_path = (root / args.gmt).resolve()

    # Fixed output folders (as you want in your screenshot)
    mapped_dir = root / "data" / "results_mapped"
    pathways_dir = root / "data" / "pathways"

    mapped_out = mapped_dir / f"{args.name}_mapped.parquet"
    pathways_out = pathways_dir / f"{args.name}_pathways.parquet"

    # 1) Read input expression
    expr_df = read_table(input_path)

    # 2) Mapping ENSG -> SYMBOL (keeps metadata columns intact)
    mapped = run_mapping(
        expr_df,
        case_id_col=args.case_id_col,
        drop_unmapped=args.drop_unmapped,
        aggregate_duplicates=args.aggregate_duplicates,
    )

    # 3) Split mapped into gene matrix + metadata
    gene_df, meta_df = split_genes_and_meta(mapped, label_col=args.label_col)

    # 4) Save mapped result (genes + metadata)
    mapped_to_save = gene_df.join(meta_df, how="left")
    save_parquet(mapped_to_save, mapped_out)
    print(f"Saved mapped matrix to: {mapped_out}")
    print("Mapped shape:", mapped_to_save.shape)

    # 5) Compute pathways from gene matrix (do NOT include labels/metadata in ssGSEA input)
    pathway_scores = run_pathway(
        gene_df,
        gene_set="CUSTOM",
        gmt_path=gmt_path,
        batch_size=args.batch_size,
    )

    # 6) Add metadata back (so downstream baseline can read cancer_type)
    out_df = pathway_scores.join(meta_df, how="left")

    # 7) Save pathways result
    save_parquet(out_df, pathways_out)
    print(f"Saved pathways matrix to: {pathways_out}")
    print("Pathways shape:", out_df.shape)

    if args.label_col in out_df.columns:
        print(f"Label column '{args.label_col}' is present ")
    else:
        print(f"Label column '{args.label_col}' NOT present  (baseline needs it)")


if __name__ == "__main__":
    main()
