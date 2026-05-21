"""
Patient-level train/val/test splits for TCGA-BRCA.

Splits are done at the patient (case) level — never at the slide level —
to prevent label leakage between folds (one patient may have multiple slides).

Default ratio: 70 / 15 / 15 with stratification on ER_status.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple


SPLIT_RATIOS = (0.70, 0.15, 0.15)
RANDOM_SEED = 42


def make_patient_splits(
    labels_csv: str,
    output_dir: str,
    stratify_col: str = "ER_status",
    ratios: Tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = RANDOM_SEED,
) -> Dict[str, pd.Index]:
    """
    Create and save patient-level splits, stratified by `stratify_col`.

    Returns dict with keys 'train', 'val', 'test' and values = case_id arrays.
    """
    df = pd.read_csv(labels_csv, index_col="case_id")
    case_ids = df.index.to_numpy()
    strat = df[stratify_col].fillna("Unknown").to_numpy() if stratify_col else None

    train_ratio, val_ratio, _ = ratios
    train_ids, temp_ids, train_strat, temp_strat = train_test_split(
        case_ids, strat, test_size=(1 - train_ratio), random_state=seed, stratify=strat
    )
    val_size_relative = val_ratio / (1 - train_ratio)
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=(1 - val_size_relative), random_state=seed, stratify=temp_strat
    )

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        pd.Series(ids, name="case_id").to_csv(output_dir / f"{name}.csv", index=False)

    print(f"Splits — train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")
    return splits


def load_split(split_csv: str) -> list:
    """Load a saved split CSV and return case IDs as a list."""
    return pd.read_csv(split_csv)["case_id"].tolist()
