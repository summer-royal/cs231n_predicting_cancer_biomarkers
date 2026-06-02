"""
Ablation: overlay ROC/PR curves and build a comparison table across models.

Compares any subset of: mean-pool + LR baseline, CLAM_SB, TumorAwareMIL.
All models are evaluated on the same split and target.

Usage:
    python scripts/plot_ablation.py \\
        --feature_dir data/features \\
        --labels_csv  data/labels/tcga_brca_labels.csv \\
        --splits_dir  data/splits \\
        --target      ER_status \\
        --split       test \\
        --checkpoints checkpoints/ER_status_clam_sb_best.pt \\
                      checkpoints/ER_status_tumor_aware_best.pt \\
        --models      clam_sb tumor_aware \\
        --model_labels "CLAM-SB" "Tumor-Aware CLAM" \\
        --in_dim      2048 \\
        --include_baseline \\
        --out_dir     figures

Outputs (in --out_dir):
    ablation_roc_pr_{target}_{split}.pdf   – overlaid ROC and PR curves
    ablation_table_{target}_{split}.csv    – numeric comparison table
    ablation_table_{target}_{split}.pdf    – table figure
"""

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import TCGABRCADataset, load_split
from evaluation.metrics import binary_metrics
from models import CLAM_MB, CLAM_SB, TumorAwareMIL

BINARY_N_CLASSES = 2


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_model(model_name: str, in_dim: int, hidden_dim: int) -> torch.nn.Module:
    if model_name == "clam_sb":
        return CLAM_SB(in_dim=in_dim, hidden_dim=hidden_dim, n_classes=BINARY_N_CLASSES)
    if model_name == "clam_mb":
        return CLAM_MB(in_dim=in_dim, hidden_dim=hidden_dim, n_classes=BINARY_N_CLASSES)
    backbone = CLAM_SB(in_dim=in_dim, hidden_dim=hidden_dim, n_classes=BINARY_N_CLASSES)
    return TumorAwareMIL(backbone=backbone, in_dim=in_dim)


def get_mil_predictions(model, dataset, device, target):
    """Return (labels, probs) arrays for a trained MIL model."""
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    labels, probs = [], []
    with torch.no_grad():
        for features, label_dict in loader:
            features = features.squeeze(0).to(device)
            label = int(label_dict[target].item())
            out = model(features)
            logits = out[0]  # first element is always logits for all model types
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            labels.append(label)
            probs.append(prob)
    return np.array(labels), np.array(probs)


def get_baseline_predictions(feature_dir, labels_df, train_ids, eval_ids, target):
    """Mean-pool + logistic regression baseline (re-fits on train, predicts on eval)."""

    def _load_mean(ids):
        X, valid = [], []
        for cid in ids:
            h5 = Path(feature_dir) / f"{cid}.h5"
            if not h5.exists():
                continue
            with h5py.File(h5, "r") as f:
                feats = f["features"][:]
            X.append(feats.mean(axis=0))
            valid.append(cid)
        return np.array(X, dtype=np.float32), valid

    target_series = labels_df[target].dropna()
    tr_ids = [i for i in train_ids if i in target_series.index]
    ev_ids = [i for i in eval_ids  if i in target_series.index]

    X_tr, tr_valid = _load_mean(tr_ids)
    X_ev, ev_valid = _load_mean(ev_ids)

    if len(X_tr) == 0 or len(X_ev) == 0:
        return None, None

    y_tr = target_series.loc[tr_valid].values.astype(int)
    y_ev = target_series.loc[ev_valid].values.astype(int)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_ev = scaler.transform(X_ev)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, solver="lbfgs")
    clf.fit(X_tr, y_tr)

    return y_ev, clf.predict_proba(X_ev)[:, 1]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overlay(entries, target, split, out_dir):
    """
    Overlay ROC and PR curves from multiple models on the same figure.

    entries: list of (display_name, labels_arr, probs_arr)
    """
    colors = plt.cm.tab10.colors
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))

    for i, (name, labels, probs) in enumerate(entries):
        c = colors[i % len(colors)]
        fpr, tpr, _ = roc_curve(labels, probs)
        auroc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=2, color=c, label=f"{name}  AUROC={auroc:.3f}")

        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = auc(recall, precision)
        ax_pr.plot(recall, precision, lw=2, color=c, label=f"{name}  AP={ap:.3f}")

    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC — {target}")
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1.05])

    baseline_prev = float(entries[0][1].mean())
    ax_pr.axhline(baseline_prev, color="k", linestyle="--", lw=1,
                  label=f"Prevalence = {baseline_prev:.2f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"PR — {target}")
    ax_pr.legend(fontsize=8, loc="upper right")
    ax_pr.set_xlim([0, 1])
    ax_pr.set_ylim([0, 1.05])

    fig.suptitle(f"Ablation — {target}  ({split} set)", fontsize=11)
    fig.tight_layout()
    path = out_dir / f"ablation_roc_pr_{target}_{split}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def save_comparison_table(entries, target, split, out_dir):
    """entries: list of (display_name, labels_arr, probs_arr)"""
    rows = {name: binary_metrics(y, p) for name, y, p in entries}
    df = pd.DataFrame(rows).T.round(4)
    df.index.name = "model"

    csv_path = out_dir / f"ablation_table_{target}_{split}.csv"
    df.to_csv(csv_path)
    print(f"\n=== Ablation ({target} / {split}) ===")
    print(df.to_string())
    print(f"  -> {csv_path}")

    fig, ax = plt.subplots(
        figsize=(max(6, len(df.columns) * 2), max(2, len(df) * 0.7 + 1.5))
    )
    ax.axis("off")
    table = ax.table(
        cellText=df.values.tolist(),
        rowLabels=list(df.index),
        colLabels=list(df.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    fig.suptitle(f"Ablation — {target}  ({split} set)", fontsize=10)
    fig.tight_layout()
    fig_path = out_dir / f"ablation_table_{target}_{split}.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {fig_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Overlay ROC/PR curves across models for ablation comparison."
    )
    p.add_argument("--feature_dir",   required=True)
    p.add_argument("--labels_csv",    required=True)
    p.add_argument("--splits_dir",    required=True)
    p.add_argument("--target",        required=True,
                   choices=["ER_status", "PR_status", "HER2_status"],
                   help="Binary target to compare (PAM50 multiclass not supported here)")
    p.add_argument("--split",         default="test", choices=["train", "val", "test"])
    p.add_argument("--checkpoints",   nargs="*", default=[],
                   help="One .pt checkpoint per MIL model (in the same order as --models)")
    p.add_argument("--models",        nargs="*", default=[],
                   choices=["clam_sb", "clam_mb", "tumor_aware"],
                   help="Model type for each checkpoint")
    p.add_argument("--model_labels",  nargs="*", default=None,
                   help="Display labels for the legend (defaults to --models values)")
    p.add_argument("--in_dim",        type=int, default=2048,
                   help="Encoder feature dim (ResNet-50: 2048, UNI: 1024, CONCH: 512)")
    p.add_argument("--hidden_dim",    type=int, default=256)
    p.add_argument("--include_baseline", action="store_true",
                   help="Also run the mean-pool + LR baseline and include it in the plot")
    p.add_argument("--out_dir",       default="figures")
    return p.parse_args()


def main():
    args = parse_args()
    if len(args.checkpoints) != len(args.models):
        raise ValueError("--checkpoints and --models must have the same number of entries.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(args.labels_csv, index_col="case_id")
    train_ids = load_split(f"{args.splits_dir}/train.csv")
    eval_ids  = load_split(f"{args.splits_dir}/{args.split}.csv")

    entries = []

    if args.include_baseline:
        print("Running mean-pool + LR baseline ...")
        y, p = get_baseline_predictions(
            args.feature_dir, labels_df, train_ids, eval_ids, args.target
        )
        if y is not None:
            entries.append(("Mean-pool + LR", y, p))
        else:
            print("  [warn] baseline skipped — no feature files found")

    display_names = args.model_labels if args.model_labels else args.models
    for ckpt, model_name, label in zip(args.checkpoints, args.models, display_names):
        print(f"Loading {model_name} from {ckpt} ...")
        dataset = TCGABRCADataset(
            args.feature_dir, args.labels_csv, eval_ids, target=args.target
        )
        model = build_model(model_name, args.in_dim, args.hidden_dim)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model = model.to(device).eval()
        y, p = get_mil_predictions(model, dataset, device, args.target)
        entries.append((label, y, p))

    if not entries:
        print("Nothing to plot. Pass --include_baseline and/or --checkpoints.")
        return

    plot_overlay(entries, args.target, args.split, out_dir)
    save_comparison_table(entries, args.target, args.split, out_dir)


if __name__ == "__main__":
    main()
