"""
Evaluate a saved CLAM checkpoint on the test split.

Usage:
    python scripts/eval_checkpoint.py \
        --feature_dir data/features \
        --labels_csv data/labels/tcga_brca_labels.csv \
        --splits_dir data/splits \
        --target ER_status \
        --checkpoint checkpoints/ER_status_clam_sb_best.pt \
        --in_dim 2048
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import TCGABRCADataset, load_split
from evaluation.metrics import binary_metrics
from models import CLAM_SB


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_dir",  required=True)
    p.add_argument("--labels_csv",   required=True)
    p.add_argument("--splits_dir",   required=True)
    p.add_argument("--target",       required=True)
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--in_dim",  type=int, default=2048)
    p.add_argument("--hidden_dim", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_ids = load_split(f"{args.splits_dir}/test.csv")
    test_ds  = TCGABRCADataset(args.feature_dir, args.labels_csv,
                                test_ids, target=args.target)
    loader   = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    model = CLAM_SB(in_dim=args.in_dim, hidden_dim=args.hidden_dim, n_classes=2)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device).eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for features, labels in loader:
            features = features.squeeze(0).to(device)
            label = int(labels[args.target].item())
            logits, _ = model(features)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            all_probs.append(prob)
            all_labels.append(label)

    metrics = binary_metrics(np.array(all_labels), np.array(all_probs))
    print(f"\n=== {args.target} — test set ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
