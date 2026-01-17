"""
Self-supervised pretraining for transcriptomic data.

This implementation combines three self-supervised objectives:
1) Pathway profile prediction (regression to pathway scores)
2) Masked gene modeling (MAE-style reconstruction on masked positions)
3) Contrastive learning (SimCLR NT-Xent loss on two augmented views)

Design notes:
- The encoder is a simple MLP.
- Three heads share the encoder:
  (a) pathway regression head
  (b) MAE decoder head
  (c) projection head for contrastive loss

Input conventions:
- X_gene_df: samples x genes (mapped gene symbols recommended)
- X_pathway_df: samples x pathways (e.g., ssGSEA NES scores)
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------
# Utilities: data conversion
# ----------------------------

def dataframe_to_float_tensor(df: pd.DataFrame, drop_label: Optional[str] = "cancer_type") -> torch.Tensor:
    """
    Convert a pandas DataFrame to a float32 torch tensor.
    Non-numeric values are coerced to NaN, then filled with 0.0.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    drop_label : str or None
        If provided, attempts to drop this column (useful if labels are present).

    Returns
    -------
    torch.Tensor
        Tensor with shape (n_samples, n_features).
    """
    safe_df = df.copy()
    safe_df = safe_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if drop_label is not None:
        safe_df = safe_df.drop(columns=[drop_label], errors="ignore")

    return torch.tensor(safe_df.values, dtype=torch.float32)


# ----------------------------
# SSL components: masking & augmentation
# ----------------------------

def apply_random_gene_mask(x: torch.Tensor, ratio: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask a fraction of gene features per sample (set to 0.0).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, G).
    ratio : float
        Fraction of genes to mask (0..1).

    Returns
    -------
    x_masked : torch.Tensor
        Masked version of x.
    mask : torch.Tensor (bool)
        Boolean mask indicating masked positions (True = masked).
    """
    B, G = x.shape
    k = int(G * ratio)

    mask = torch.zeros((B, G), dtype=torch.bool, device=x.device)
    x_masked = x.clone()

    # Keep the original per-sample random masking behavior
    for i in range(B):
        idx = torch.randperm(G, device=x.device)[:k]
        mask[i, idx] = True
        x_masked[i, idx] = 0.0

    return x_masked, mask


def gene_expression_augmentation(x: torch.Tensor, noise_sigma: float = 0.1, drop_prob: float = 0.1) -> torch.Tensor:
    """
    Create an augmented view of the gene expression vector:
    - Add Gaussian noise
    - Random feature dropout (set some features to 0)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (B, G).
    noise_sigma : float
        Std of Gaussian noise.
    drop_prob : float
        Probability of dropping each gene.

    Returns
    -------
    torch.Tensor
        Augmented tensor (B, G).
    """
    x_aug = x + torch.randn_like(x) * noise_sigma

    if drop_prob > 0:
        drop_mask = (torch.rand_like(x) < drop_prob).float()
        x_aug = x_aug * (1.0 - drop_mask)

    return x_aug


# ----------------------------
# Losses
# ----------------------------

def nt_xent_contrastive_loss(z_a: torch.Tensor, z_b: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """
    NT-Xent loss (SimCLR-style) for two batches of representations.

    Parameters
    ----------
    z_a, z_b : torch.Tensor
        Embeddings of shape (B, D) for two augmented views.
    tau : float
        Temperature.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)

    B = z_a.size(0)
    z = torch.cat([z_a, z_b], dim=0)  # (2B, D)

    sim = (z @ z.T) / tau             # (2B, 2B)
    diag = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, -1e9) # remove self-similarity

    # For i in [0..2B-1], positive index is (i + B) % (2B)
    pos_idx = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return F.cross_entropy(sim, pos_idx)


def masked_reconstruction_mse(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    MSE computed only on masked positions.

    Parameters
    ----------
    recon : torch.Tensor
        Reconstruction (B, G)
    target : torch.Tensor
        Original input (B, G)
    mask : torch.Tensor (bool)
        Mask positions (B, G)

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    diff = (recon - target) ** 2
    return diff[mask].mean()


# ----------------------------
# Model
# ----------------------------

class GeneSSLModel(nn.Module):
    """
    Encoder + three heads:
    - pathway regression head
    - MAE reconstruction head
    - contrastive projection head
    """

    def __init__(
        self,
        input_dim: int,
        pathway_dim: int,
        latent_dim: int = 256,
        proj_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Shared encoder (ReLU stays as requested)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, latent_dim),
            nn.ReLU(),
        )

        self.latent_norm = nn.LayerNorm(latent_dim)

        # Pathway regression head
        self.pathway_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, pathway_dim),
        )

        # MAE decoder head
        self.mae_decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
        )

        # Contrastive projection head
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.latent_norm(z)

    def forward(self, x_full: torch.Tensor, x_masked: torch.Tensor):
        """
        Returns:
        - pathway prediction from x_full
        - reconstruction from masked input
        - projection from x_full
        - latent embedding from x_full
        """
        z_full = self.encode(x_full)
        y_path = self.pathway_head(z_full)

        z_mask = self.encode(x_masked)
        x_recon = self.mae_decoder(z_mask)

        z_proj = self.projection_head(z_full)

        return y_path, x_recon, z_proj, z_full


# ----------------------------
# Config
# ----------------------------

@dataclass
class SSLPretrainConfig:
    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3

    mask_ratio: float = 0.4

    # Loss weights
    alpha: float = 1.0   # pathway loss weight
    beta: float = 0.3    # MAE loss weight
    gamma: float = 0.1   # contrastive loss weight

    temperature: float = 0.5

    # Augmentation defaults (kept same spirit as original)
    noise_sigma: float = 0.1
    drop_prob: float = 0.1

    # Model dims
    latent_dim: int = 256
    proj_dim: int = 128
    dropout: float = 0.3

    save_dir: str = "experiments/ssl_pretrain"


# ----------------------------
# Pretraining routine
# ----------------------------

def run_pretrain(
    X_gene_df: pd.DataFrame,
    X_pathway_df: pd.DataFrame,
    config: SSLPretrainConfig = SSLPretrainConfig(),
) -> GeneSSLModel:
    """
    Train the SSL model.

    Parameters
    ----------
    X_gene_df : pd.DataFrame
        Gene expression matrix (samples x genes).
    X_pathway_df : pd.DataFrame
        Pathway score matrix (samples x pathways).
    config : SSLPretrainConfig
        Training and model configuration.

    Returns
    -------
    GeneSSLModel
        Trained model (weights also saved to disk).
    """
    os.makedirs(config.save_dir, exist_ok=True)

    # Convert dataframes to tensors
    X_genes = dataframe_to_float_tensor(X_gene_df, drop_label="cancer_type")
    X_paths = dataframe_to_float_tensor(X_pathway_df, drop_label="cancer_type")

    if X_genes.shape[0] != X_paths.shape[0]:
        raise ValueError(
            f"Sample count mismatch: genes={X_genes.shape[0]} vs pathways={X_paths.shape[0]}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_genes = X_genes.to(device)
    X_paths = X_paths.to(device)

    input_dim = X_genes.shape[1]
    pathway_dim = X_paths.shape[1]

    model = GeneSSLModel(
        input_dim=input_dim,
        pathway_dim=pathway_dim,
        latent_dim=config.latent_dim,
        proj_dim=config.proj_dim,
        dropout=config.dropout,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    reg_criterion = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_genes, X_paths),
        batch_size=config.batch_size,
        shuffle=True,
    )

    print(f"[SSL] Pretraining on {device} for {config.epochs} epochs")
    model.train()

    for epoch in range(config.epochs):
        epoch_loss = 0.0

        for xb, pb in loader:
            xb = xb.to(device)
            pb = pb.to(device)

            # 1) Masked input for MAE-style reconstruction
            xb_masked, mask = apply_random_gene_mask(xb, ratio=config.mask_ratio)

            # 2) Forward pass for pathway + reconstruction
            pred_path, recon, _, _ = model(xb, xb_masked)

            loss_path = reg_criterion(pred_path, pb)
            loss_mae = masked_reconstruction_mse(recon, xb, mask)

            # 3) Contrastive views
            view1 = gene_expression_augmentation(xb, noise_sigma=config.noise_sigma, drop_prob=config.drop_prob)
            view2 = gene_expression_augmentation(xb, noise_sigma=config.noise_sigma, drop_prob=config.drop_prob)

            z1 = model.encode(view1)
            z2 = model.encode(view2)

            p1 = model.projection_head(z1)
            p2 = model.projection_head(z2)

            loss_ctr = nt_xent_contrastive_loss(p1, p2, tau=config.temperature)

            # 4) Total loss
            loss = config.alpha * loss_path + config.beta * loss_mae + config.gamma * loss_ctr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0 or (epoch + 1) == config.epochs:
            avg_loss = epoch_loss / max(1, len(loader))
            print(f"[SSL] Epoch {epoch+1:03d}/{config.epochs} | avg_loss={avg_loss:.4f}")

    # Save weights
    model_path = os.path.join(config.save_dir, "ssl_model.pth")
    torch.save(model.state_dict(), model_path)

    # Save config
    cfg_path = os.path.join(config.save_dir, "config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dim": input_dim,
                "pathway_dim": pathway_dim,
                **asdict(config),
            },
            f,
            indent=4,
        )

    print(f"[SSL] Done. Saved model to: {model_path}")
    print(f"[SSL] Saved config to: {cfg_path}")

    return model


# ----------------------------
# Encoder loading for fine-tuning
# ----------------------------

def load_pretrained_encoder(weights_path: str, config_path: str) -> nn.Module:
    """
    Load ONLY the encoder part (encoder + LayerNorm) from a pretrained SSL model.

    Parameters
    ----------
    weights_path : str
        Path to saved model weights (.pth).
    config_path : str
        Path to saved config (.json).

    Returns
    -------
    nn.Module
        Encoder module ready for downstream fine-tuning.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = GeneSSLModel(
        input_dim=cfg["input_dim"],
        pathway_dim=cfg["pathway_dim"],
        latent_dim=cfg.get("latent_dim", 256),
        proj_dim=cfg.get("proj_dim", 128),
        dropout=cfg.get("dropout", 0.3),
    ).to(device)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    encoder = nn.Sequential(model.encoder, model.latent_norm).to(device)
    encoder.eval()

    print(f"[SSL] Loaded encoder from: {weights_path}")
    return encoder

