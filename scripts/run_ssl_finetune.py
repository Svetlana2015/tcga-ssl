from pathlib import Path
import pandas as pd

from src.ssl_pretrain import load_pretrained_encoder
from src.ssl_finetune import run_finetune

ROOT = Path(__file__).resolve().parents[1]

def main():
    data_dir = ROOT / "data"

    # Данные (genes, после mapping)
    finetune_df = pd.read_parquet(data_dir / "finetune.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")

    # Encoder weights
    weights_path = ROOT / "experiments" / "ssl_pretrain" / "ssl_model.pth"
    config_path = ROOT / "experiments" / "ssl_pretrain" / "config.json"

    encoder = load_pretrained_encoder(str(weights_path), str(config_path))

    # Frozen
    run_finetune(
        encoder=encoder,
        finetune_df=finetune_df,
        test_df=test_df,
        epochs=50,
        freeze_encoder=True,
        res_dir=str(ROOT / "results" / "ssl"),
        save_dir=str(ROOT / "experiments" / "ssl_finetune"),
    )

    # Unfrozen
    run_finetune(
        encoder=encoder,
        finetune_df=finetune_df,
        test_df=test_df,
        epochs=50,
        freeze_encoder=False,
        res_dir=str(ROOT / "results" / "ssl"),
        save_dir=str(ROOT / "experiments" / "ssl_finetune"),
    )

if __name__ == "__main__":
    main()
