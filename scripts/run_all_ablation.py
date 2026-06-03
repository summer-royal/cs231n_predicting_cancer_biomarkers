"""
Run ablation plots/tables for all encoders and merge a cross-encoder summary.

This is intended for Modal's /data volume after training has produced:
    /data/features_<encoder>/
    /data/splits/<encoder>/
    /data/labels/<encoder>/tcga_brca_labels.csv
    /data/checkpoints/<encoder>/

It also works locally if the same directory layout exists under --root.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


DEFAULT_ENCODERS = ["resnet50", "uni", "conch"]
DEFAULT_TARGETS = ["ER_status", "PR_status", "HER2_status"]
DEFAULT_MODELS = ["clam_sb", "tumor_aware"]
DEFAULT_MODEL_LABELS = ["CLAM-SB", "Tumor-Aware CLAM"]


def run_ablation(root: Path, encoder: str, target: str, split: str) -> Path | None:
    feature_dir = root / f"features_{encoder}"
    labels_csv = root / "labels" / encoder / "tcga_brca_labels.csv"
    splits_dir = root / "splits" / encoder
    ckpt_dir = root / "checkpoints" / encoder
    out_dir = root / "figures" / encoder

    checkpoints = [ckpt_dir / f"{target}_{model}_best.pt" for model in DEFAULT_MODELS]
    required = [feature_dir, labels_csv, splits_dir, *checkpoints]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print(f"[skip] {encoder} / {target} / {split}: missing {missing}")
        return None

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "plot_ablation.py"),
        "--feature_dir", str(feature_dir),
        "--labels_csv", str(labels_csv),
        "--splits_dir", str(splits_dir),
        "--target", target,
        "--split", split,
        "--checkpoints", *(str(p) for p in checkpoints),
        "--models", *DEFAULT_MODELS,
        "--model_labels", *DEFAULT_MODEL_LABELS,
        "--in_dim", "auto",
        "--include_baseline",
        "--out_dir", str(out_dir),
    ]
    print(f"[run] {encoder} / {target} / {split}")
    subprocess.run(cmd, check=True)
    return out_dir / f"ablation_table_{target}_{split}.csv"


def merge_tables(root: Path, csv_paths: list[tuple[str, str, str, Path]], out_dir: Path) -> None:
    rows = []
    for encoder, target, split, csv_path in csv_paths:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        model_col = "model" if "model" in df.columns else df.columns[0]
        for _, row in df.iterrows():
            record = row.to_dict()
            record["model"] = record.pop(model_col)
            record["encoder"] = encoder
            record["target"] = target
            record["split"] = split
            rows.append(record)

    if not rows:
        print("[warn] no ablation CSVs found to merge")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    leading = ["encoder", "target", "split", "model"]
    summary = summary[leading + [c for c in summary.columns if c not in leading]]
    summary = summary.sort_values(["target", "split", "model", "encoder"])
    csv_path = out_dir / "cross_encoder_ablation_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"[ok] wrote {csv_path}")

    for split in sorted(summary["split"].unique()):
        split_df = summary[summary["split"] == split]
        pivot = split_df.pivot_table(
            index=["target", "model"],
            columns="encoder",
            values="auroc",
            aggfunc="first",
        ).round(4)
        pivot_path = out_dir / f"cross_encoder_auroc_{split}.csv"
        pivot.to_csv(pivot_path)
        print(f"[ok] wrote {pivot_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Run all encoder ablation figures and merge summaries.")
    p.add_argument("--root", default="data", help="Root containing features_*, splits, labels, checkpoints")
    p.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--summary_dir", default=None,
                   help="Where merged tables should be written. Defaults to <root>/figures/summary.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    summary_dir = Path(args.summary_dir) if args.summary_dir else root / "figures" / "summary"

    completed = []
    for encoder in args.encoders:
        for split in args.splits:
            for target in args.targets:
                csv_path = run_ablation(root, encoder, target, split)
                if csv_path is not None:
                    completed.append((encoder, target, split, csv_path))

    merge_tables(root, completed, summary_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
