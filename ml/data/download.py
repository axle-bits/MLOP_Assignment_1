"""Download the UCI Heart Disease (Cleveland) dataset to a local CSV.

Kept separate from cleaning: acquisition and preprocessing are independent,
individually re-runnable pipeline steps.
"""
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

UCI_DATASET_ID = 45
DEFAULT_RAW_PATH = Path("data/raw/heart_cleveland_raw.csv")


def download_raw(out_path: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Fetch UCI dataset 45 and write features + raw target to CSV."""
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    frame = download_raw()
    print(f"Wrote {DEFAULT_RAW_PATH} shape={frame.shape}")
