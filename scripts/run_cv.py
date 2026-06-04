"""
k-fold cross-validation for the ablation (mean-pool / clam_sb / tumor_aware).

usage:
    python scripts/run_cv.py \\
        --feature_dir data/features_resnet50 \\
        --labels_csv  data/labels/tcga_brca_labels.csv \\
        --target      ER_status \\
        --models      clam_sb tumor_aware \\
        --include_baseline \\
        --n_folds 5 --epochs 20 \\
        --out_dir data/cv/resnet50
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import TCGABRCADataset, infer_feature_dim, load_split, make_cv_folds
from evaluation.metrics import binary_metrics_with_ci, bootstrap_ci, fmt_ci
from scripts.plot_ablation import build_model, get_baseline_predictions, get_mil_predictions
from sklearn.metrics import average_precision_score, roc_auc_score

MODEL_LABELS = {
    "baseline": "Mean-pool + LR",
    "clam_sb": "CLAM-SB",
    "tumor_aware": "Tumor-Aware CLAM",
}


def train_fold(feature_dir, labels_csv, fold_dir, target, model, in_dim, hidden_dim, epochs, save_dir):
    # reuse run_training so cv matches the main pipeline exactly
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(Path(__file__).parent / "run_training.py"),
        "--feature_dir", str(feature_dir),
        "--labels_csv", str(labels_csv),
        "--splits_dir", str(fold_dir),
        "--target", target,
        "--model", model,
        "--in_dim", str(in_dim),
        "--hidden_dim", str(hidden_dim),
        "--epochs", str(epochs),
        "--save_dir", str(save_dir),
    ]
    subprocess.run(cmd, check=True)
    return save_dir / f"{target}_{model}_best.pt"


def fold_predictions(model_name, feature_dir, labels_csv, labels_df, fold_dir,
                     target, in_dim, hidden_dim, epochs, ckpt_dir, device):
    train_ids = load_split(fold_dir / "train.csv")
    test_ids = load_split(fold_dir / "test.csv")

    if model_name == "baseline":
        return get_baseline_predictions(feature_dir, labels_df, train_ids, test_ids, target)

    ckpt = train_fold(feature_dir, labels_csv, fold_dir, target, model_name,
                      in_dim, hidden_dim, epochs, ckpt_dir)
    dataset = TCGABRCADataset(str(feature_dir), str(labels_csv), test_ids, target=target)
    model = build_model(model_name, in_dim, hidden_dim)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device).eval()
    return get_mil_predictions(model, dataset, device, target)


def aggregate(model_name, fold_metrics, pooled_y, pooled_p, n_boot, seed):
    # mean/std over folds + bootstrap ci on the pooled oof predictions
    fa = np.array([m["auroc"] for m in fold_metrics])
    fp = np.array([m["auprc"] for m in fold_metrics])
    pooled_y = np.array(pooled_y)
    pooled_p = np.array(pooled_p)

    lo_roc, hi_roc = bootstrap_ci(pooled_y, pooled_p, roc_auc_score, n_boot, seed)
    lo_pr, hi_pr = bootstrap_ci(pooled_y, pooled_p, average_precision_score, n_boot, seed)
    return {
        "model": MODEL_LABELS.get(model_name, model_name),
        "n_folds": len(fold_metrics),
        "auroc_mean": round(float(fa.mean()), 4),
        "auroc_std": round(float(fa.std()), 4),
        "auprc_mean": round(float(fp.mean()), 4),
        "auprc_std": round(float(fp.std()), 4),
        "pooled_auroc": round(float(roc_auc_score(pooled_y, pooled_p)), 4),
        "pooled_auroc_lo": round(lo_roc, 4),
        "pooled_auroc_hi": round(hi_roc, 4),
        "pooled_auprc": round(float(average_precision_score(pooled_y, pooled_p)), 4),
        "pooled_auprc_lo": round(lo_pr, 4),
        "pooled_auprc_hi": round(hi_pr, 4),
    }


def parse_args():
    p = argparse.ArgumentParser(description="k-fold cv ablation with bootstrap ci.")
    p.add_argument("--feature_dir", required=True)
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--target", required=True,
                   choices=["ER_status", "PR_status", "HER2_status"])
    p.add_argument("--models", nargs="*", default=["clam_sb", "tumor_aware"],
                   choices=["clam_sb", "clam_mb", "tumor_aware"])
    p.add_argument("--include_baseline", action="store_true")
    p.add_argument("--in_dim", default="auto")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="data/cv")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_dir = Path(args.feature_dir)
    labels_df = pd.read_csv(args.labels_csv, index_col="case_id")

    # folds stratified on the target so each fold keeps class balance
    splits_dir = out_dir / "cv_splits" / args.target
    n_folds = make_cv_folds(args.labels_csv, splits_dir, stratify_col=args.target,
                            n_folds=args.n_folds, seed=args.seed)

    in_dim = (infer_feature_dim(str(feature_dir), load_split(splits_dir / "fold0" / "train.csv"))
              if args.in_dim == "auto" else int(args.in_dim))
    print(f"target: {args.target} | feature dim: {in_dim} | folds: {n_folds} | device: {device}")

    models = (["baseline"] if args.include_baseline else []) + list(args.models)

    # per model
    # list of fold metrics + pooled oof predictions
    fold_metrics = {m: [] for m in models}
    pooled = {m: ([], []) for m in models}

    for fold in range(n_folds):
        fold_dir = splits_dir / f"fold{fold}"
        ckpt_dir = out_dir / "cv_ckpts" / args.target / f"fold{fold}"
        for m in models:
            y, p = fold_predictions(m, feature_dir, args.labels_csv, labels_df, fold_dir,
                                    args.target, in_dim, args.hidden_dim, args.epochs,
                                    ckpt_dir, device)
            if y is None:
                print(f"[skip] {m} fold{fold}: no predictions")
                continue
            fold_metrics[m].append(binary_metrics_with_ci(y, p, n_boot=0))
            pooled[m][0].extend(list(y))
            pooled[m][1].extend(list(p))
            print(f"fold{fold} {m}: auroc={roc_auc_score(y, p):.3f} (n={len(y)})")

    rows = []
    for m in models:
        if not fold_metrics[m]:
            continue
        rows.append(aggregate(m, fold_metrics[m], pooled[m][0], pooled[m][1],
                              args.n_boot, args.seed))

    summary = pd.DataFrame(rows)
    csv_path = out_dir / f"cv_summary_{args.target}.csv"
    summary.to_csv(csv_path, index=False)

    print(f"\n=== {args.n_folds}-fold cv ({args.target}) ===")
    for r in rows:
        print(f"{r['model']:<18} "
              f"auroc {r['auroc_mean']:.3f} +/- {r['auroc_std']:.3f} | "
              f"pooled {fmt_ci(r['pooled_auroc'], r['pooled_auroc_lo'], r['pooled_auroc_hi'])}")
    print(f"  -> {csv_path}")


if __name__ == "__main__":
    main()
