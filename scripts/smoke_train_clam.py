"""
Real-data shape-and-debug script for CLAM models on TCGA-BRCA features.

This is the first thing to run *after* `extract_features.py` has produced
.h5 files under data/features/. It:

  1. Loads one sample from TCGABRCADataset and prints feature/label info.
  2. Infers `in_dim` from that sample (do not hardcode 1024 vs 2048).
  3. Builds the requested model (clam_sb, clam_mb, or tumor_aware).
  4. Runs a single forward pass and prints output shapes.
  5. Optionally runs a 1-epoch training loop end-to-end.

Example (binary ER prediction):

    python scripts/smoke_train_clam.py \
        --features_dir data/features \
        --labels_csv data/labels/tcga_brca_labels.csv \
        --split_csv data/splits/train.csv \
        --target ER_status \
        --task_type binary \
        --model clam_sb \
        --max_patches 512 \
        --epochs 1

Example (PAM50 multi-class):

    python scripts/smoke_train_clam.py \
        --target PAM50_subtype --task_type multiclass --model clam_mb
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import TCGABRCADataset, load_split
from models import CLAM_MB, CLAM_SB, TumorAwareMIL
from training import Trainer


PAM50_N_CLASSES = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", required=True, type=str)
    p.add_argument("--labels_csv", required=True, type=str)
    p.add_argument("--split_csv", required=True, type=str)
    p.add_argument("--target", required=True, type=str,
                   help="Label column, e.g. ER_status, PR_status, HER2_status, PAM50_subtype.")
    p.add_argument("--task_type", choices=["binary", "multiclass"], default="binary")
    p.add_argument("--model", choices=["clam_sb", "clam_mb", "tumor_aware"], default="clam_sb")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--max_patches", type=int, default=None,
                   help="Cap bag size for memory. None = use all patches.")
    p.add_argument("--epochs", type=int, default=0,
                   help="0 = forward-pass debug only. >0 = train this many epochs.")
    p.add_argument("--batch_size", type=int, default=1,
                   help="MIL on variable-sized bags expects batch_size=1.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--instance_eval", action="store_true",
                   help="Enable CLAM instance-clustering aux loss.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_model(args: argparse.Namespace, in_dim: int, n_classes: int) -> torch.nn.Module:
    if args.model == "clam_sb":
        return CLAM_SB(in_dim=in_dim, hidden_dim=args.hidden_dim, n_classes=n_classes)
    if args.model == "clam_mb":
        return CLAM_MB(in_dim=in_dim, hidden_dim=args.hidden_dim, n_classes=n_classes)
    # tumor_aware: use SB backbone for binary, MB for multiclass
    backbone_cls = CLAM_SB if n_classes == 2 else CLAM_MB
    backbone = backbone_cls(in_dim=in_dim, hidden_dim=args.hidden_dim, n_classes=n_classes)
    return TumorAwareMIL(backbone=backbone, in_dim=in_dim)


def main() -> int:
    args = parse_args()

    # 1) Load dataset & pull one sample to inspect.
    case_ids = load_split(args.split_csv)
    dataset = TCGABRCADataset(
        feature_dir=args.features_dir,
        labels_csv=args.labels_csv,
        case_ids=case_ids,
        target=args.target,
        max_patches=args.max_patches,
    )
    print(f"\n[dataset] {len(dataset)} slides match split + features.")

    features, labels = dataset[0]
    if args.target not in labels:
        raise RuntimeError(
            f"First sample is missing label '{args.target}'. "
            f"Available: {list(labels.keys())}"
        )
    in_dim = int(features.shape[-1])
    print(f"[sample]  feature_tensor={tuple(features.shape)}  dtype={features.dtype}")
    print(f"[sample]  label[{args.target}]={labels[args.target].item()}  in_dim inferred={in_dim}")

    # 2) Build model.
    n_classes = 2 if args.task_type == "binary" else PAM50_N_CLASSES
    model = build_model(args, in_dim=in_dim, n_classes=n_classes)
    model = model.to(args.device)
    print(f"[model]   {args.model}(in_dim={in_dim}, hidden_dim={args.hidden_dim}, n_classes={n_classes})")

    # 3) One forward pass for shape verification.
    model.eval()
    with torch.no_grad():
        x = features.to(args.device).float()
        out = model(x)
        logits, A = out[0], out[1]
        print(f"[forward] logits={tuple(logits.shape)}  attention={tuple(A.shape)}")
        if len(out) >= 3 and args.model == "tumor_aware":
            print(f"[forward] tumor_probs={tuple(out[-1].shape)}  "
                  f"range=[{float(out[-1].min()):.3f}, {float(out[-1].max()):.3f}]")

    # 4) Optional: train for a few epochs.
    if args.epochs <= 0:
        print("\n[done] forward-pass debug only (--epochs 0).")
        return 0

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        target=args.target,
        task_type=args.task_type,
        device=args.device,
        instance_eval=args.instance_eval,
    )

    for epoch in range(args.epochs):
        avg_loss = trainer.train_epoch(loader)
        print(f"[epoch {epoch + 1}/{args.epochs}] train_loss={avg_loss:.4f}")

    print("\n[done] training smoke run finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
