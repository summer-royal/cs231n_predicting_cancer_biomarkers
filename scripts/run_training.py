"""
End-to-end training script for CLAM_SB / CLAM_MB / TumorAwareMIL.

Usage (binary):
    python scripts/run_training.py \
        --feature_dir data/features_resnet50 \
        --labels_csv data/labels/tcga_brca_labels.csv \
        --splits_dir data/splits \
        --target ER_status \
        --model clam_sb \
        --epochs 20 \
        --in_dim auto

Usage (PAM50 multi-class):
    python scripts/run_training.py ... \
        --target PAM50_subtype --task_type multiclass --model clam_mb
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import TCGABRCADataset, infer_feature_dim, load_split
from models import CLAM_MB, CLAM_SB, TumorAwareMIL
from training import Trainer

PAM50_N_CLASSES = 5
BINARY_N_CLASSES = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature_dir", required=True)
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--splits_dir", required=True)
    p.add_argument("--target", default="ER_status",
                   choices=["ER_status", "PR_status", "HER2_status", "PAM50_subtype"])
    p.add_argument("--task_type", default="binary", choices=["binary", "multiclass"])
    p.add_argument("--model", default="clam_sb",
                   choices=["clam_sb", "clam_mb", "tumor_aware"])
    p.add_argument("--in_dim", default="auto",
                   help="Feature dim or 'auto' to infer from feature files.")
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--max_patches", type=int, default=None)
    p.add_argument("--instance_eval", action="store_true",
                   help="Enable CLAM instance-clustering auxiliary loss")
    p.add_argument("--instance_loss_weight", type=float, default=0.3)
    p.add_argument("--gate_mode", default="mult", choices=["mult", "residual"])
    p.add_argument("--gate_alpha", type=float, default=1.0)
    p.add_argument("--learn_temp", action="store_true")
    p.add_argument("--reg_mode", default="none", choices=["none", "entropy", "budget", "both"])
    p.add_argument("--reg_weight", type=float, default=0.0)
    p.add_argument("--gate_budget", type=float, default=0.25)
    p.add_argument("--checkpoint_suffix", default="")
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args()


def build_model(args, n_classes):
    if args.model == "clam_sb":
        return CLAM_SB(in_dim=args.in_dim, hidden_dim=args.hidden_dim, n_classes=n_classes)
    if args.model == "clam_mb":
        return CLAM_MB(in_dim=args.in_dim, hidden_dim=args.hidden_dim, n_classes=n_classes)
    # tumor_aware
    backbone_cls = CLAM_SB if n_classes == BINARY_N_CLASSES else CLAM_MB
    backbone = backbone_cls(in_dim=args.in_dim, hidden_dim=args.hidden_dim, n_classes=n_classes)
    return TumorAwareMIL(
        backbone=backbone,
        in_dim=args.in_dim,
        gate_mode=args.gate_mode,
        alpha=args.gate_alpha,
        learn_temp=args.learn_temp,
        reg_mode=args.reg_mode,
        reg_weight=args.reg_weight,
        budget=args.gate_budget,
    )


def checkpoint_stem(args):
    if args.checkpoint_suffix:
        return f"{args.target}_{args.model}_{args.checkpoint_suffix}"
    if args.model != "tumor_aware":
        return f"{args.target}_{args.model}"
    if (
        args.gate_mode == "mult"
        and not args.learn_temp
        and args.reg_mode == "none"
        and args.reg_weight == 0.0
    ):
        return f"{args.target}_{args.model}"
    parts = [args.model, args.gate_mode]
    if args.gate_mode == "residual":
        parts.append(f"a{args.gate_alpha:g}")
    if args.reg_mode != "none" and args.reg_weight > 0.0:
        parts.append(args.reg_mode)
        parts.append(f"rw{args.reg_weight:g}")
        if args.reg_mode in ("budget", "both"):
            parts.append(f"b{args.gate_budget:g}")
    if args.learn_temp:
        parts.append("temp")
    return f"{args.target}_{'_'.join(parts)}"


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = PAM50_N_CLASSES if args.task_type == "multiclass" else BINARY_N_CLASSES
    print(f"Device: {device} | target: {args.target} | model: {args.model} | n_classes: {n_classes}")

    train_ids = load_split(f"{args.splits_dir}/train.csv")
    val_ids   = load_split(f"{args.splits_dir}/val.csv")
    in_dim = infer_feature_dim(args.feature_dir, train_ids) if args.in_dim == "auto" else int(args.in_dim)
    print(f"Feature dim: {in_dim}")

    train_ds = TCGABRCADataset(
        args.feature_dir, args.labels_csv, train_ids,
        target=args.target, max_patches=args.max_patches
    )
    val_ds = TCGABRCADataset(
        args.feature_dir, args.labels_csv, val_ids,
        target=args.target, max_patches=args.max_patches
    )
    # batch_size=1: bags have variable patch counts; can't stack across slides
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=2)

    args.in_dim = in_dim
    model = build_model(args, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = Trainer(
        model, optimizer,
        target=args.target,
        task_type=args.task_type,
        device=device,
        use_wandb=args.wandb,
        instance_eval=args.instance_eval,
        instance_loss_weight=args.instance_loss_weight,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"{checkpoint_stem(args)}_best.pt"

    best_primary = 0.0  # AUROC for binary, macro_auroc for multiclass
    primary_key  = "auroc" if args.task_type == "binary" else "macro_auroc"

    for epoch in range(1, args.epochs + 1):
        train_loss  = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)
        primary     = val_metrics.get(primary_key, 0.0)
        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        print(f"Epoch {epoch:3d}/{args.epochs} | loss {train_loss:.4f} | {metrics_str}")
        if primary > best_primary:
            best_primary = primary
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> saved best checkpoint ({primary_key}={best_primary:.4f})")

    print(f"\nDone. Best val {primary_key}: {best_primary:.4f}  |  checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
