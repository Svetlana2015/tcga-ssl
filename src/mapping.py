import pandas as pd
from mygene import MyGeneInfo


def run_mapping(
    expr_df: pd.DataFrame,
    case_id_col: str = "caseID",
    drop_unmapped: bool = True,
    aggregate_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Map Ensembl gene IDs (ENSG...) to gene symbols using mygene.

    This version preserves non-ENSG columns (metadata/labels), e.g. `cancer_type`.

    Pipeline:
    1) Detect ENSG columns
    2) Move caseID to index if present
    3) Split into: gene matrix + metadata
    4) Strip ENSG version suffix
    5) Query mygene for ENSG -> SYMBOL
    6) Rename gene columns
    7) (Optional) drop unmapped genes
    8) (Optional) aggregate duplicate symbols by mean
    9) Normalize symbols (UPPER + strip)
    10) Concatenate mapped genes + metadata
    """

    df = expr_df.copy()

    # Identify ENSG columns
    gene_cols = [c for c in df.columns if str(c).startswith("ENSG")]

    # Ensure sample IDs are in index (if caseID exists as a column)
    if case_id_col in df.columns:
        df = df.set_index(case_id_col)

    # Split into genes and metadata (KEEP metadata!)
    genes_df = df[gene_cols].copy()
    meta_cols = [c for c in df.columns if c not in gene_cols]
    meta_df = df[meta_cols].copy() if len(meta_cols) > 0 else None

    # Strip ENSG version suffix (ENSG... .12 -> ENSG...)
    genes_df.columns = [str(c).split(".")[0] for c in genes_df.columns]
    gene_cols_clean = genes_df.columns.tolist()

    # Query mygene for ENSG -> SYMBOL
    mg = MyGeneInfo()
    query_res = mg.querymany(
        gene_cols_clean,
        scopes="ensembl.gene",
        fields="symbol",
        species="human"
    )

    # Build mapping dictionary
    mapping_dict = {}
    for item in query_res:
        q = item.get("query")
        if item.get("notfound", False):
            mapping_dict[q] = None
        else:
            mapping_dict[q] = item.get("symbol", None)

    # Rename gene columns
    genes_mapped = genes_df.rename(columns=mapping_dict)

    # Drop unmapped genes if requested
    if drop_unmapped:
        genes_mapped = genes_mapped.loc[:, genes_mapped.columns.notna()]
    else:
        genes_mapped.columns = [
            (c if c is not None else "UNMAPPED") for c in genes_mapped.columns
        ]

    # Aggregate duplicated symbols (multiple ENSG -> same SYMBOL)
    if aggregate_duplicates:
        # group duplicates by column name
        genes_mapped = genes_mapped.T.groupby(genes_mapped.columns).mean().T

    # Normalize symbols
    genes_mapped.columns = genes_mapped.columns.astype(str).str.upper().str.strip()

    # Concatenate back metadata (if present)
    if meta_df is not None and meta_df.shape[1] > 0:
        out = pd.concat([genes_mapped, meta_df], axis=1)
    else:
        out = genes_mapped

    return out
