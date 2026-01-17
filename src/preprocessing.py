from __future__ import annotations

import os
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitSizes:
    test: int = 1000
    finetune: int = 1000
    pretrain: int | None = 7000  # None => "всё оставшееся"


def _assert_columns(expr: pd.DataFrame, labels: pd.DataFrame) -> None:
    if "caseID" not in expr.columns:
        raise ValueError("Expression parquet must contain column 'caseID'.")
    if "cases" not in labels.columns:
        raise ValueError("Labels parquet must contain column 'cases'.")
    if "cancer_type" not in labels.columns:
        raise ValueError("Labels parquet must contain column 'cancer_type'.")


def _extract_case_id(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.copy()
    labels["caseID"] = labels["cases"].astype(str).str.split("|").str[1]
    return labels[["caseID", "cancer_type"]].copy()


def _stratified_take(
    df: pd.DataFrame,
    y_col: str,
    n_take: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_take <= 0:
        raise ValueError("n_take must be > 0")
    if n_take >= len(df):
        raise ValueError(f"Requested n_take={n_take} but df has only {len(df)} rows")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_take, random_state=seed)
    # test_size
    idx_rem, idx_take = next(sss.split(df, df[y_col]))
    taken = df.iloc[idx_take].copy()
    remaining = df.iloc[idx_rem].copy()
    return taken, remaining


def preprocess_to_parquet(
    expr_path: str,
    label_path: str,
    output_dir: str = "data",
    sizes: SplitSizes = SplitSizes(),
    seed: int = 42,
    scale: bool = True,
    save_full: bool = True,
    save_unused: bool = True,
) -> None:
    
    os.makedirs(output_dir, exist_ok=True)

    expr = pd.read_parquet(expr_path)
    labels = pd.read_parquet(label_path)
    _assert_columns(expr, labels)

    labels_ = _extract_case_id(labels)

    merged = pd.merge(labels_, expr, on="caseID", how="inner")
    if merged.empty:
        raise ValueError("Merge resulted in empty dataframe. Check caseID matching.")

    # y as a string for baseline compatibility
    merged["cancer_type"] = merged["cancer_type"].astype(str)

    # we separate the signs
    feature_cols = [c for c in merged.columns if c not in ("caseID", "cancer_type")]
    if not feature_cols:
        raise ValueError("No feature columns found after merge (besides caseID/cancer_type).")

    # first test (exactly sizes.test)
    test_df, rest = _stratified_take(merged, y_col="cancer_type", n_take=sizes.test, seed=seed)

    # finetune (exactly sizes.finetune)
    finetune_df, rest2 = _stratified_take(rest, y_col="cancer_type", n_take=sizes.finetune, seed=seed)

    # pretrain
    if sizes.pretrain is None:
        pretrain_df = rest2.copy()
        unused_df = None
    else:
        if sizes.pretrain > len(rest2):
            raise ValueError(
                f"Requested pretrain={sizes.pretrain} but only {len(rest2)} rows remain "
                f"after taking test={sizes.test} and finetune={sizes.finetune}."
            )
        # We take exactly pretrained stratified
        pretrain_df, unused_df = _stratified_take(rest2, y_col="cancer_type", n_take=sizes.pretrain, seed=seed + 1)

    # scaling (fit ONLY on pretrain)
    if scale:
        scaler = StandardScaler()
        scaler.fit(pretrain_df[feature_cols])

        for d in (pretrain_df, finetune_df, test_df):
            d.loc[:, feature_cols] = scaler.transform(d[feature_cols])

 
    pretrain_out = pretrain_df.drop(columns=["caseID"]).reset_index(drop=True)
    finetune_out = finetune_df.drop(columns=["caseID"]).reset_index(drop=True)
    test_out = test_df.drop(columns=["caseID"]).reset_index(drop=True)

    pretrain_out.to_parquet(os.path.join(output_dir, "pretrain.parquet"), index=False)
    finetune_out.to_parquet(os.path.join(output_dir, "finetune.parquet"), index=False)
    test_out.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)

    if save_full:
        full_out = merged.drop(columns=["caseID"]).reset_index(drop=True)
        full_out.to_parquet(os.path.join(output_dir, "full.parquet"), index=False)

    if save_unused and unused_df is not None and len(unused_df) > 0:
        unused_out = unused_df.drop(columns=["caseID"]).reset_index(drop=True)
        unused_out.to_parquet(os.path.join(output_dir, "unused.parquet"), index=False)

    print(f"Saved to: {output_dir}")
    print(f"Sizes: pretrain={len(pretrain_out)}, finetune={len(finetune_out)}, test={len(test_out)}")
    if unused_df is not None:
        print(f"Unused saved: {len(unused_df)} rows")
