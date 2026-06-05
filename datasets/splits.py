"""
Patient-level train/val/test splits for TCGA-BRCA.

Splits are done at the patient (case) level — never at the slide level —
to prevent label leakage between folds (one patient may have multiple slides).

Default ratio: 70 / 15 / 15 with stratification on ER_status.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
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
    if stratify_col:
        df = df[df[stratify_col].notna()]
    case_ids = df.index.to_numpy()
    strat = df[stratify_col].astype(str).to_numpy() if stratify_col else None

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


def make_cv_folds(
    labels_csv: str,
    output_dir: str,
    stratify_col: str = "ER_status",
    n_folds: int = 5,
    val_frac: float = 0.15,
    seed: int = RANDOM_SEED,
) -> int:
    # patient-level stratified k-fold
    # every patient in test EXACTLY once
    # output_dir/fold{i}/{train,val,test}.csv
    # stratify_col = target evaluating 
    df = pd.read_csv(labels_csv, index_col="case_id")
    df = df[df[stratify_col].notna()]
    case_ids = df.index.to_numpy()
    strat = df[stratify_col].astype(str).to_numpy()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    output_dir = Path(output_dir)
    for fold, (trainval_idx, test_idx) in enumerate(skf.split(case_ids, strat)):
        tv_ids = case_ids[trainval_idx]
        tv_strat = strat[trainval_idx]
        # val set out of the train portion for early stopping 
        train_ids, val_ids = train_test_split(
            tv_ids, test_size=val_frac, random_state=seed, stratify=tv_strat
        )
        test_ids = case_ids[test_idx]
        fold_dir = output_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
            pd.Series(ids, name="case_id").to_csv(fold_dir / f"{name}.csv", index=False)
        print(f"fold{fold} — train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    return n_folds
