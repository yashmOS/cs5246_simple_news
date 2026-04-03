from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _has_any_csv(data_dir: Path) -> bool:
    return any(data_dir.glob("*.csv"))


def _standardize_hf_split(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["article"] = df["article"].fillna("").astype(str)
    out["highlights"] = df["highlights"].fillna("").astype(str)
    if "id" in df.columns:
        out["source_id"] = df["id"].astype(str)
    return out


def download_cnn_dailymail(
    data_dir: Path,
    dataset_name: str = "cnn_dailymail",
    dataset_version: str = "3.0.0",
    force: bool = False,
    sample_train: Optional[int] = None,
    sample_validation: Optional[int] = None,
    sample_test: Optional[int] = None,
) -> None:
    """
    Download CNN/DailyMail via Hugging Face datasets and save local CSV files.

    Files written:
    - data/train.csv
    - data/validation.csv
    - data/test.csv
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    target_files = [data_dir / "train.csv", data_dir / "validation.csv", data_dir / "test.csv"]
    if not force and all(path.exists() for path in target_files):
        return

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Missing dependency: datasets. Install requirements.txt before running dataset download."
        ) from e

    dataset = load_dataset(dataset_name, dataset_version)

    split_limits = {
        "train": sample_train,
        "validation": sample_validation,
        "test": sample_test,
    }

    for split_name, limit in split_limits.items():
        split_df = dataset[split_name].to_pandas()
        if limit is not None:
            split_df = split_df.iloc[:limit].copy()
        split_df = _standardize_hf_split(split_df)
        split_df.to_csv(data_dir / f"{split_name}.csv", index=False)


def ensure_dataset(
    data_dir: Path,
    auto_download: bool = True,
    force_download: bool = False,
    sample_train: Optional[int] = None,
    sample_validation: Optional[int] = None,
    sample_test: Optional[int] = None,
) -> str:
    """Ensure at least one local CSV dataset is available.

    Returns a short status string for logging.
    """
    if _has_any_csv(data_dir) and not force_download:
        return "Found existing local CSV dataset."

    if not auto_download:
        raise FileNotFoundError(
            f"No CSV dataset found in {data_dir.resolve()} and auto_download is disabled."
        )

    download_cnn_dailymail(
        data_dir=data_dir,
        force=force_download,
        sample_train=sample_train,
        sample_validation=sample_validation,
        sample_test=sample_test,
    )
    return "Downloaded CNN/DailyMail and saved local CSV files."
