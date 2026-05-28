"""
Mean-pool + logistic regression baseline.

Aggregates patch features by mean-pooling per slide, then fits a
logistic regression classifier on the slide-level feature vectors.
This is the simplest possible MIL baseline: no attention, no learned
aggregation — just an average bag embedding.

Usage:
    python scripts/run_baseline.py \
        --feature_dir data/features \
        --labels_csv data/labels/tcga_brca_labels.csv \
        --splits_dir data/splits

Optional: restrict to specific targets:
    python scripts/run_baseline.py ... --targets ER_status HER2_status
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_split
from evaluation.metrics import binary_metrics, summarize_results

BINARY_TARGETS = ["ER_status", "PR_status", "HER2_status"]


def load_mean_features(feature_dir: Path, case_ids: list):
    """Return (X, valid_ids) where X is (n_valid, D) mean-pooled features."""
    X, valid_ids = [], []
    for cid in case_ids:
        h5_path = feature_dir / f"{cid}.h5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as f:
            feats = f["features"][:]          # (N_patches, D)
        X.append(feats.mean(axis=0))
        valid_ids.append(cid)
    return np.array(X, dtype=np.float32), valid_ids


def run_one_target(feature_dir, labels_df, train_ids, val_ids, test_ids, target):
    target_df = labels_df[target].dropna()
    tr_ids = [i for i in train_ids if i in target_df.index]
    va_ids = [i for i in val_ids   if i in target_df.index]
    te_ids = [i for i in test_ids  if i in target_df.index]

    X_tr, tr_ids = load_mean_features(feature_dir, tr_ids)
    X_va, va_ids = load_mean_features(feature_dir, va_ids)
    X_te, te_ids = load_mean_features(feature_dir, te_ids)

    if len(X_tr) == 0 or len(X_te) == 0:
        print(f"  {target}: skipped (no feature files found)")
        return None

    y_tr = target_df.loc[tr_ids].values.astype(int)
    y_va = target_df.loc[va_ids].values.astype(int)
    y_te = target_df.loc[te_ids].values.astype(int)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, solver="lbfgs")
    clf.fit(X_tr, y_tr)

    results = {}
    for split_name, X_s, y_s in [("val", X_va, y_va), ("test", X_te, y_te)]:
        if len(X_s) == 0:
            continue
        probs = clf.predict_proba(X_s)[:, 1]
        results[split_name] = binary_metrics(y_s, probs)

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_dir", required=True)
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--splits_dir", required=True)
    p.add_argument("--targets", nargs="+", default=BINARY_TARGETS)
    args = p.parse_args()

    feature_dir = Path(args.feature_dir)
    labels_df = pd.read_csv(args.labels_csv, index_col="case_id")
    train_ids = load_split(f"{args.splits_dir}/train.csv")
    val_ids   = load_split(f"{args.splits_dir}/val.csv")
    test_ids  = load_split(f"{args.splits_dir}/test.csv")

    test_summary = {}
    for target in args.targets:
        print(f"\n=== {target} ===")
        results = run_one_target(feature_dir, labels_df, train_ids, val_ids, test_ids, target)
        if results is None:
            continue
        for split, m in results.items():
            print(f"  {split:4s}: AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  "
                  f"bal-acc={m['balanced_acc']:.4f}  Brier={m['brier']:.4f}")
        if "test" in results:
            test_summary[target] = results["test"]

    if test_summary:
        print("\n=== Baseline summary (test set) ===")
        print(summarize_results(test_summary).to_string())


if __name__ == "__main__":
    main()
