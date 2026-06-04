"""
Generate ROC/PR curves, a metrics table, and attention heatmaps for a trained checkpoint.

Outputs (written to --out_dir):
    roc_pr_{target}_{model}.pdf        – ROC and PR curves side-by-side (binary targets)
    metrics_{model}_{split}.csv        – numeric metrics table
    metrics_table_{model}_{split}.pdf  – metrics as a figure
    heatmaps/heatmap_{case_id}_{target}.pdf  – per-slide attention heatmaps

Usage (basic):
    python scripts/generate_figures.py \\
        --feature_dir data/features \\
        --labels_csv  data/labels/tcga_brca_labels.csv \\
        --splits_dir  data/splits \\
        --checkpoint  checkpoints/ER_status_clam_sb_best.pt \\
        --target      ER_status \\
        --model       clam_sb \\
        --in_dim      2048 \\
        --out_dir     figures

With WSI thumbnail overlays (requires raw SVS files):
    python scripts/generate_figures.py ... \\
        --slide_dir data/raw \\
        --n_heatmaps 5

Encoder dim reference:
    ResNet-50 → 2048  |  UNI → 1024  |  CONCH → 512
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
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import TCGABRCADataset, infer_feature_dim, load_split
from evaluation.metrics import binary_metrics, multiclass_metrics
from models import CLAM_MB, CLAM_SB, TumorAwareMIL

PAM50_N_CLASSES = 5
BINARY_N_CLASSES = 2


# ---------------------------------------------------------------------------
# Model construction (mirrors run_training.py)
# ---------------------------------------------------------------------------

def build_model(model_name: str, in_dim: int, hidden_dim: int, n_classes: int) -> torch.nn.Module:
    if model_name == "clam_sb":
        return CLAM_SB(in_dim=in_dim, hidden_dim=hidden_dim, n_classes=n_classes)
    if model_name == "clam_mb":
        return CLAM_MB(in_dim=in_dim, hidden_dim=hidden_dim, n_classes=n_classes)
    # tumor_aware wraps an SB backbone for binary, MB for multi-class
    backbone_cls = CLAM_SB if n_classes == BINARY_N_CLASSES else CLAM_MB
    backbone = backbone_cls(in_dim=in_dim, hidden_dim=hidden_dim, n_classes=n_classes)
    return TumorAwareMIL(backbone=backbone, in_dim=in_dim)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def collect_predictions(model, dataset, device, target, task_type="binary"):
    """
    Run inference on every slide in dataset (batch_size=1, no shuffle).

    Returns:
        labels       : np.ndarray (N,)
        probs        : np.ndarray (N,) for binary (class-1 prob),
                       np.ndarray (N, C) for multiclass
        attentions   : list[np.ndarray]  one (K_i,) array per slide
        tumor_scores : list[np.ndarray]  one (K_i,) array per slide, or None
        case_ids     : list[str]         in dataset order
    """
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    is_tumor_aware = isinstance(model, TumorAwareMIL)
    model.eval()

    all_labels, all_probs, attentions, tumor_scores = [], [], [], []

    with torch.no_grad():
        for features, label_dict in loader:
            features = features.squeeze(0).to(device)
            label = int(label_dict[target].item())
            out = model(features)

            if is_tumor_aware:
                logits, A, tp = out
                tumor_scores.append(tp.squeeze(-1).cpu().numpy())
            else:
                logits, A = out

            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()  # (C,)

            # Flatten attention to (K,)
            # CLAM_SB returns A shape (K, 1); CLAM_MB returns (C, K).
            # Distinguish them by the last dim: (K,1) has shape[-1]==1.
            A_np = A.cpu().numpy()
            if A_np.ndim == 2 and A_np.shape[-1] != 1:
                # Multi-branch (C, K): use the attention of the predicted class
                A_np = A_np[probs.argmax()]
            else:
                A_np = A_np.flatten()

            all_labels.append(label)
            all_probs.append(probs)
            attentions.append(A_np)

    all_labels = np.array(all_labels)
    all_probs_mat = np.stack(all_probs)  # (N, C)
    out_probs = all_probs_mat[:, 1] if task_type == "binary" else all_probs_mat

    return (
        all_labels,
        out_probs,
        attentions,
        tumor_scores if is_tumor_aware else None,
        dataset.case_ids,
    )


# ---------------------------------------------------------------------------
# ROC / PR curves
# ---------------------------------------------------------------------------

def plot_roc_pr(labels, probs, target_name, model_name, out_dir):
    fpr, tpr, _ = roc_curve(labels, probs)
    auroc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = auc(recall, precision)
    baseline_rate = float(labels.mean())

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4))

    ax_roc.plot(fpr, tpr, lw=2, color="steelblue", label=f"AUROC = {auroc:.3f}")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC — {target_name}")
    ax_roc.legend(loc="lower right")
    ax_roc.set_xlim([0, 1])
    ax_roc.set_ylim([0, 1.05])

    ax_pr.plot(recall, precision, lw=2, color="tomato", label=f"AP = {ap:.3f}")
    ax_pr.axhline(baseline_rate, color="k", linestyle="--", lw=1,
                  label=f"Prevalence = {baseline_rate:.2f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"PR — {target_name}")
    ax_pr.legend(loc="upper right")
    ax_pr.set_xlim([0, 1])
    ax_pr.set_ylim([0, 1.05])

    fig.suptitle(f"{target_name}  ·  {model_name}", fontsize=11)
    fig.tight_layout()
    path = out_dir / f"roc_pr_{target_name}_{model_name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ---------------------------------------------------------------------------
# Metrics table
# ---------------------------------------------------------------------------

def save_metrics(results: dict, out_dir: Path, split_name: str, model_name: str):
    """Save a CSV and a PDF figure of the metrics table."""
    df = pd.DataFrame(results).T.round(4)
    df.index.name = "target"

    csv_path = out_dir / f"metrics_{model_name}_{split_name}.csv"
    df.to_csv(csv_path)
    print(f"\n=== Metrics ({model_name} / {split_name} set) ===")
    print(df.to_string())
    print(f"  -> {csv_path}")

    fig, ax = plt.subplots(
        figsize=(max(5, len(df.columns) * 1.8), max(2, len(df) * 0.7 + 1.5))
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
    fig.suptitle(f"{model_name}  ·  {split_name} set", fontsize=10)
    fig.tight_layout()
    fig_path = out_dir / f"metrics_table_{model_name}_{split_name}.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {fig_path}")


# ---------------------------------------------------------------------------
# Attention heatmaps
# ---------------------------------------------------------------------------

def plot_attention_heatmap(
    case_id: str,
    feature_h5: str,
    attention: np.ndarray,
    save_path: str,
    slide_path: str = None,
    tumor_probs: np.ndarray = None,
    patch_size: int = 256,
):
    """
    Render per-patch attention as a 2D spatial heatmap.

    If slide_path is given and openslide is importable, the heatmap is
    overlaid on the WSI thumbnail. Otherwise falls back to a scatter plot
    on raw pixel coordinates from the feature .h5 file.
    """
    with h5py.File(feature_h5, "r") as f:
        if "coords" not in f:
            print(f"  [warn] no coords in {feature_h5} — skipping heatmap for {case_id}")
            return
        coords = f["coords"][:]  # (K, 2) in level-0 pixel units

    # Trim if there's a length mismatch (shouldn't happen in normal runs)
    n = min(len(coords), len(attention))
    coords = coords[:n]
    attention = attention[:n]
    if tumor_probs is not None:
        tumor_probs = tumor_probs[:n]

    n_cols = 2 if tumor_probs is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    def _draw(ax, weights, title):
        # Attempt WSI thumbnail overlay
        if slide_path and Path(slide_path).exists():
            try:
                import openslide
                slide = openslide.OpenSlide(slide_path)
                thumb = slide.get_thumbnail((512, 512))
                slide_w, slide_h = slide.dimensions
                slide.close()
                thumb_w, thumb_h = thumb.size
                sx, sy = thumb_w / slide_w, thumb_h / slide_h
                ax.imshow(thumb, extent=[0, thumb_w, thumb_h, 0])
                xs = coords[:, 0] * sx
                ys = coords[:, 1] * sy
                dot = max(1.0, (sx * patch_size) ** 2)
                sc = ax.scatter(xs, ys, c=weights, cmap="magma", alpha=0.6,
                                s=dot, vmin=0, vmax=weights.max())
                plt.colorbar(sc, ax=ax, fraction=0.046)
                ax.set_title(f"{title}\n{case_id}", fontsize=8)
                return
            except Exception:
                pass

        # Fallback: render as a 2D grid image so tiles are always visible
        # regardless of the raw pixel coordinate scale.
        xs, ys = coords[:, 0], coords[:, 1]
        x_min, y_min = xs.min(), ys.min()

        # Infer tile step from the smallest gap between unique x positions.
        ux = np.unique(xs)
        step = int(np.diff(np.sort(ux)).min()) if len(ux) > 1 else patch_size

        grid_w = int((xs.max() - x_min) / step) + 1
        grid_h = int((ys.max() - y_min) / step) + 1

        grid = np.full((grid_h, grid_w), np.nan)
        for idx, (x, y) in enumerate(coords):
            gx = int((x - x_min) / step)
            gy = int((y - y_min) / step)
            if 0 <= gy < grid_h and 0 <= gx < grid_w:
                grid[gy, gx] = weights[idx]

        # Use percentile-based normalization so subtle differences are visible.
        # vmin=0 would collapse everything to the top of the colourmap when
        # all softmax weights are tiny (e.g. 1/K ~ 1e-4).
        vmin = np.nanpercentile(grid[~np.isnan(grid)], 10)
        vmax = np.nanpercentile(grid[~np.isnan(grid)], 99)
        im = ax.imshow(grid, cmap="magma", interpolation="nearest",
                       vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xlabel("tile col")
        ax.set_ylabel("tile row")
        ax.set_title(f"{title}\n{case_id}", fontsize=8)

    _draw(axes[0], attention, "Attention")
    if tumor_probs is not None:
        _draw(axes[1], tumor_probs, "Tumor score")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate ROC/PR curves, metrics tables, and attention heatmaps."
    )
    p.add_argument("--feature_dir", required=True, help="Dir of per-slide .h5 feature files")
    p.add_argument("--labels_csv",  required=True, help="CSV with case_id + label columns")
    p.add_argument("--splits_dir",  required=True, help="Dir with train/val/test.csv split files")
    p.add_argument("--checkpoint",  required=True, help="Path to .pt checkpoint")
    p.add_argument("--target",      required=True,
                   choices=["ER_status", "PR_status", "HER2_status", "PAM50_subtype"])
    p.add_argument("--model",       default="clam_sb",
                   choices=["clam_sb", "clam_mb", "tumor_aware"])
    p.add_argument("--task_type",   default="binary", choices=["binary", "multiclass"])
    p.add_argument("--in_dim",      default="auto",
                   help="Encoder feature dim, or 'auto' to infer from feature files (default: auto)")
    p.add_argument("--hidden_dim",  type=int, default=256)
    p.add_argument("--split",       default="test", choices=["train", "val", "test"])
    p.add_argument("--out_dir",     default="figures")
    p.add_argument("--slide_dir",   default=None,
                   help="Path to raw .svs files; enables WSI thumbnail overlays in heatmaps")
    p.add_argument("--n_heatmaps",  type=int, default=5,
                   help="Number of slides to render attention heatmaps for (0 to skip)")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = PAM50_N_CLASSES if args.task_type == "multiclass" else BINARY_N_CLASSES

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    case_ids = load_split(f"{args.splits_dir}/{args.split}.csv")
    dataset = TCGABRCADataset(
        args.feature_dir, args.labels_csv, case_ids, target=args.target
    )
    in_dim = infer_feature_dim(args.feature_dir, case_ids) if args.in_dim == "auto" else int(args.in_dim)
    print(f"Feature dim: {in_dim}")
    model = build_model(args.model, in_dim, args.hidden_dim, n_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device).eval()
    print(f"Loaded {args.model} | {len(dataset)} slides | device: {device}")

    labels, probs, attentions, tumor_scores, slide_ids = collect_predictions(
        model, dataset, device, args.target, args.task_type
    )

    # Metrics table
    if args.task_type == "binary":
        metrics = binary_metrics(labels, probs)
    else:
        metrics = multiclass_metrics(labels, probs)
    save_metrics({args.target: metrics}, out_dir, args.split, args.model)

    # ROC / PR curves (binary targets only)
    if args.task_type == "binary":
        plot_roc_pr(labels, probs, args.target, args.model, out_dir)

    # Attention heatmaps
    if args.n_heatmaps > 0:
        heatmap_dir = out_dir / "heatmaps"
        heatmap_dir.mkdir(exist_ok=True)
        for i, case_id in enumerate(slide_ids[: args.n_heatmaps]):
            h5_path = Path(args.feature_dir) / f"{case_id}.h5"
            if not h5_path.exists():
                continue
            slide_path = None
            if args.slide_dir:
                cand = Path(args.slide_dir) / f"{case_id}.svs"
                if cand.exists():
                    slide_path = str(cand)
            tp = tumor_scores[i] if tumor_scores is not None else None
            plot_attention_heatmap(
                case_id,
                str(h5_path),
                attentions[i],
                str(heatmap_dir / f"heatmap_{case_id}_{args.target}.pdf"),
                slide_path=slide_path,
                tumor_probs=tp,
            )
        print(f"  -> heatmaps saved to {heatmap_dir}/")


if __name__ == "__main__":
    main()
