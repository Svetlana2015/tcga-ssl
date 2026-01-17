import torch
import torch.nn as nn
import json
from torchsummary import summary
import pandas as pd


# 1. Baseline Summary
def summarize_baseline(
    finetune_path="finetune.parquet",
    baseline_model_class=None   # pass baseline.MLP
):
    """
    Print torchsummary for the Baseline MLP.
    """

    if baseline_model_class is None:
        raise ValueError("You must pass baseline.MLP as baseline_model_class")

    print("\n==================== BASELINE MODEL SUMMARY ====================")

    # Load finetune dataset to detect dimensions
    finetune_df = pd.read_parquet(finetune_path)
    input_dim = finetune_df.drop(columns=["cancer_type"]).shape[1]
    num_classes = finetune_df["cancer_type"].nunique()

    # Build model on CPU
    model = baseline_model_class(input_dim=input_dim, num_classes=num_classes).cpu()

    # Print summary
    try:
        summary(model, input_size=(input_dim,), device="cpu")
    except Exception as e:
        print("[torchsummary failed]", e)

    print("================================================================\n")


# 2. SSL Encoder Summary
def summarize_ssl_encoder(
    ssl_config_path="experiments/ssl_pretrain_run1/config.json",
    ssl_weight_path="experiments/ssl_pretrain_run1/ssl_model.pth",
    ssl_model_class=None    # pass SSL_MLP
):
    """
    Print torchsummary for the SSL encoder (encoder + LayerNorm).
    """

    if ssl_model_class is None:
        raise ValueError("You must pass ssl_pretrain.SSL_MLP as ssl_model_class")

    print("\n==================== SSL ENCODER SUMMARY ====================")

    # Load config
    with open(ssl_config_path, "r") as f:
        cfg = json.load(f)

    # Build SSL model
    model = ssl_model_class(
        input_dim=cfg["input_dim"],
        pathway_dim=cfg["pathway_dim"],
        latent_dim=cfg["latent_dim"],
        proj_dim=cfg["proj_dim"],
        dropout=0.3
    ).cpu()

    # Load weights
    state = torch.load(ssl_weight_path, map_location="cpu")
    model.load_state_dict(state)

    # Extract encoder
    # encoder_only = nn.Sequential(model.encoder, model.norm).cpu()
    encoder_only = nn.Sequential(model.encoder, model.latent_norm).cpu()

    # Print summary
    try:
        summary(encoder_only, input_size=(cfg["input_dim"],), device="cpu")
    except Exception as e:
        print("[torchsummary failed]", e)

    print("===============================================================\n")
