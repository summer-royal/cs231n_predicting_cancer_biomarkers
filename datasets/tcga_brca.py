"""
PyTorch Dataset for TCGA-BRCA precomputed patch features.

Each slide is represented as a bag of patch embeddings stored in an .h5 file.
Labels are loaded from a CSV with one row per patient (TCGA case ID).

Binary targets  : ER_status, PR_status, HER2_status  (0 / 1)
Multiclass      : PAM50_subtype  (LumA, LumB, HER2E, Basal, Normal)
Continuous      : estrogen_score, proliferation_score, immune_score  (z-scored)
"""

import h5py
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


BINARY_TARGETS = ["ER_status", "PR_status", "HER2_status"]
PAM50_CLASSES = ["LumA", "LumB", "HER2E", "Basal", "Normal"]
CONTINUOUS_TARGETS = ["estrogen_score", "proliferation_score", "immune_score"]
ALL_TARGETS = BINARY_TARGETS + ["PAM50_subtype"] + CONTINUOUS_TARGETS


class TCGABRCADataset(Dataset):
    """
    Args:
        feature_dir: directory of .h5 files, one per slide (stem = TCGA case ID).
        labels_csv:  path to CSV with columns: case_id, ER_status, PR_status,
                     HER2_status, PAM50_subtype, estrogen_score, ...
        case_ids:    list of case IDs to include (e.g., one split's worth).
        target:      which label column to return; None returns all.
        max_patches: if set, randomly subsample patches to this count.
    """

    def __init__(
        self,
        feature_dir: str,
        labels_csv: str,
        case_ids: List[str],
        target: Optional[str] = None,
        max_patches: Optional[int] = None,
    ):
        self.feature_dir = Path(feature_dir)
        self.max_patches = max_patches
        self.target = target

        labels_df = pd.read_csv(labels_csv, index_col="case_id")
        self.labels_df = labels_df.loc[labels_df.index.intersection(case_ids)]

        # Drop cases missing the requested target label
        if target is not None and target in self.labels_df.columns:
            before = len(self.labels_df)
            self.labels_df = self.labels_df[self.labels_df[target].notna()]
            dropped = before - len(self.labels_df)
            if dropped:
                print(f"[TCGABRCADataset] Skipping {dropped} case(s) with no '{target}' label.")

        # Drop cases whose .h5 feature file is absent
        self.case_ids = [
            cid for cid in self.labels_df.index
            if (self.feature_dir / f"{cid}.h5").exists()
        ]
        missing = len(self.labels_df) - len(self.case_ids)
        if missing:
            print(f"[TCGABRCADataset] Skipping {missing} case(s) with no .h5 file.")

        if len(self.case_ids) == 0:
            raise ValueError("No matching case IDs found between CSV and feature directory.")

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        case_id = self.case_ids[idx]
        features = self._load_features(case_id)
        labels = self._load_labels(case_id)
        return features, labels

    def _load_features(self, case_id: str) -> torch.Tensor:
        h5_path = self.feature_dir / f"{case_id}.h5"
        with h5py.File(h5_path, "r") as f:
            feats = f["features"][:]  # (N_patches, D)

        if self.max_patches and len(feats) > self.max_patches:
            idx = np.random.choice(len(feats), self.max_patches, replace=False)
            feats = feats[idx]

        return torch.from_numpy(feats.astype(np.float32))

    def _load_labels(self, case_id: str) -> Dict:
        row = self.labels_df.loc[case_id]
        labels: Dict = {}

        for col in BINARY_TARGETS:
            if col in row.index and not pd.isna(row[col]):
                labels[col] = torch.tensor(int(row[col]), dtype=torch.long)

        if "PAM50_subtype" in row.index and not pd.isna(row["PAM50_subtype"]):
            labels["PAM50_subtype"] = torch.tensor(
                PAM50_CLASSES.index(row["PAM50_subtype"]), dtype=torch.long
            )

        for col in CONTINUOUS_TARGETS:
            if col in row.index and not pd.isna(row[col]):
                labels[col] = torch.tensor(float(row[col]), dtype=torch.float32)

        if self.target is not None:
            return {self.target: labels[self.target]}
        return labels
