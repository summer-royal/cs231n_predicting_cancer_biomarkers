"""
combine per-target cv summaries into one figure + table.

reads cv_summary_<target>.csv (from run_cv.py) and plots pooled auroc with
95% bootstrap ci error bars, grouped by target, one bar per model. also
writes a merged csv.

usage:
    python scripts/plot_cv_summary.py \\
        --cv_dir data/cv/resnet50 \\
        --targets ER_status PR_status HER2_status \\
        --out_dir data/cv/resnet50
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_ORDER = ["Mean-pool + LR", "CLAM-SB", "Tumor-Aware CLAM"]


def load(cv_dir, targets):
    rows = []
    for t in targets:
        path = Path(cv_dir) / f"cv_summary_{t}.csv"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        df = pd.read_csv(path)
        df["target"] = t
        rows.append(df)
    if not rows:
        raise SystemExit("no cv_summary_*.csv files found")
    return pd.concat(rows, ignore_index=True)


def plot(df, targets, out_dir):
    targets = [t for t in targets if t in set(df["target"])]
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    x = np.arange(len(targets))
    width = 0.8 / max(len(models), 1)
    colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(1.8 * len(targets) + 3, 4.5))
    for i, m in enumerate(models):
        means, lo_err, hi_err = [], [], []
        for t in targets:
            r = df[(df["target"] == t) & (df["model"] == m)]
            if r.empty:
                means.append(np.nan); lo_err.append(0); hi_err.append(0); continue
            r = r.iloc[0]
            means.append(r["pooled_auroc"])
            # asymmetric error bars from the bootstrap ci
            lo_err.append(r["pooled_auroc"] - r["pooled_auroc_lo"])
            hi_err.append(r["pooled_auroc_hi"] - r["pooled_auroc"])
        ax.bar(x + i * width, means, width, label=m, color=colors[i % len(colors)],
               yerr=[lo_err, hi_err], capsize=4, error_kw={"elinewidth": 1})

    ax.axhline(0.5, color="k", linestyle="--", lw=1, label="chance")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([t.replace("_status", "") for t in targets])
    ax.set_ylabel("Pooled AUROC (5-fold, 95% CI)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Cross-validated ablation — pooled out-of-fold AUROC")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    path = Path(out_dir) / "cv_summary_auroc.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def save_table(df, out_dir):
    cols = ["target", "model", "auroc_mean", "auroc_std",
            "pooled_auroc", "pooled_auroc_lo", "pooled_auroc_hi",
            "pooled_auprc", "pooled_auprc_lo", "pooled_auprc_hi"]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()
    path = Path(out_dir) / "cv_summary_combined.csv"
    out.to_csv(path, index=False)
    print(f"  -> {path}")
    print("\n=== combined cv summary ===")
    print(out.to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(description="combine cv summaries into a figure + table.")
    p.add_argument("--cv_dir", required=True)
    p.add_argument("--targets", nargs="+", default=["ER_status", "PR_status", "HER2_status"])
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.cv_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load(args.cv_dir, args.targets)
    save_table(df, out_dir)
    plot(df, args.targets, out_dir)


if __name__ == "__main__":
    main()
