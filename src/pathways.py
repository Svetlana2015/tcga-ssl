import pandas as pd
import gseapy as gp
from pathlib import Path
from typing import Dict, List, Union, Optional

"""
Pathway profiling utilities (ssGSEA).

This module computes pathway activity profiles from gene expression matrices
using ssGSEA (single-sample Gene Set Enrichment Analysis).

Expected input:
- rows = samples
- columns = gene symbols (e.g., TP53, EGFR, ...)

Output:
- rows = samples
- columns = pathways (e.g., KEGG_APOPTOSIS, ...)
- values = pathway activity scores (NES by default)
"""

def read_gmt(gmt_path: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Read a GMT file and return a dict: pathway_name -> list of genes.

    GMT format:
    pathway_name <TAB> description <TAB> gene1 <TAB> gene2 <TAB> ...

    Parameters
    ----------
    gmt_path : str or Path
        Path to a .gmt file.

    Returns
    -------
    dict
        Dictionary mapping pathway names to gene symbol lists.
    """
    gmt_path = Path(gmt_path)
    gene_sets = {}

    with gmt_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = [g for g in parts[2:] if g]
            gene_sets[name] = genes

    return gene_sets
    

# Main function
def run_pathway(
    expr_df,
    gene_set: str = "KEGG",
    gmt_path: Optional[Union[str, Path]] = None,
    batch_size=200) -> pd.DataFrame:

    

    """
    Compute pathway profiles for all samples using ssGSEA.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Expression matrix with shape (n_samples, n_genes).
        Index should be sample IDs; columns should be gene symbols.
     gene_set : str
        Name of the gene set collection (only used for documentation/logging).
        Example: "KEGG".
    gmt_path : str or Path, optional
        Path to a GMT file. If None, a default path is expected to be provided
        by the user. (Recommended: provide gmt_path explicitly.)
    batch_size : int
        Gene set size.

    Returns
    -------
    pd.DataFrame
        Pathway score matrix with shape (n_samples, n_pathways).
    """

    # Validate input
    if expr_df is None or expr_df.empty:
        raise ValueError("Input expression dataframe is empty.")
    
    # Load gene sets dict from GMT 
    if gmt_path is None:
        raise ValueError(
            "gmt_path is None. Please provide a GMT file path, e.g. "
            "run_pathway_batch(expr, gene_set='KEGG', gmt_path='path/to/kegg.gmt')"
        )

    gene_set = read_gmt(gmt_path)

    # Drop cancer_type if present
    expr = expr_df.drop(columns=["cancer_type"], errors="ignore")

    print("Expression matrix shape:", expr.shape)

    # Transpose: genes = rows, samples = columns
    expr_t = expr.T
    expr_t.columns = expr_t.columns.astype(str)

    all_scores = []

    # Run ssGSEA
    for i in range(0, expr_t.shape[1], batch_size):
        batch = expr_t.iloc[:, i:i+batch_size].astype(float)
        print(f"Running ssGSEA batch {i}-{i + batch.shape[1]} using {gene_set}")

        ss = gp.ssgsea(
            data=batch,
            gene_sets=gene_set,
            sample_norm_method="rank",
            scale=True,
            outdir=None,
            min_size=1,
            max_size=10000
        )

        # ss.res2d is long-format table: Name (sample), Term (pathway), ES, NES
        res2d = ss.res2d

        needed_cols = {"Name", "Term", "NES"}
        
        if not needed_cols.issubset(set(res2d.columns)):
            raise ValueError(
            f"Expected columns {needed_cols} in ssGSEA output, got: {list(res2d.columns)}"
        )


        # Pivot to wide matrix: rows=samples, cols=pathways
        batch_scores = res2d.pivot(index="Term", columns="Name", values="NES")
        all_scores.append(batch_scores)

    # Combine all batches into one matrix
    pathway_scores = pd.concat(all_scores, axis=1)

    # Transpose back to Sample × Pathway
    pathway_scores = pathway_scores.T

    print("Final pathway matrix:", pathway_scores.shape)

    return pathway_scores

def save_pathway_profiles(
    pathway_scores: pd.DataFrame,
    out_path: Union[str, Path],
    fmt: str = "parquet"
) -> None:
    """
    Save pathway profile matrix to disk.

    Parameters
    ----------
    pathway_scores : pd.DataFrame
        Output of run_pathway_batch()
    out_path : str or Path
        Output path (without forcing extension).
    fmt : str
        "parquet" or "csv"
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt.lower() == "parquet":
        pathway_scores.to_parquet(out_path.with_suffix(".parquet"))
    elif fmt.lower() == "csv":
        pathway_scores.to_csv(out_path.with_suffix(".csv"))
    else:
        raise ValueError("fmt must be 'parquet' or 'csv'")

    
