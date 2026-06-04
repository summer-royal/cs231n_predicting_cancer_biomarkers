"""
Evaluation metrics for biomarker prediction.
Owner: Luke Zhao

Binary targets  : AUROC, AUPRC, balanced accuracy, Brier score
Continuous targets: Spearman r, Pearson r, RMSE
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from typing import Dict


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Args:
        y_true: binary labels (0/1), shape (N,)
        y_prob: predicted probability for class 1, shape (N,)
    """
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
    }


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=1000, seed=0):
    # resample patients w/ replacement, recompute metric each time
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            # need both classes or auroc/auprc undefined
            continue
        vals.append(metric_fn(yt, y_prob[idx]))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def binary_metrics_with_ci(y_true, y_prob, n_boot=1000, seed=0) -> Dict[str, float]:
    # point estimates + 95% bootstrap ci on the ranking metrics
    out = dict(binary_metrics(y_true, y_prob))
    out["auroc_lo"], out["auroc_hi"] = bootstrap_ci(
        y_true, y_prob, roc_auc_score, n_boot, seed
    )
    out["auprc_lo"], out["auprc_hi"] = bootstrap_ci(
        y_true, y_prob, average_precision_score, n_boot, seed
    )
    return out


def fmt_ci(point, lo, hi) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return f"{point:.3f}"
    return f"{point:.3f} [{lo:.2f}, {hi:.2f}]"


def multiclass_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Args:
        y_true: integer class labels, shape (N,)
        y_prob: predicted class probabilities, shape (N, C)
    """
    y_pred = y_prob.argmax(axis=1)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }
    # Macro AUROC requires at least one positive example per class present.
    try:
        metrics["macro_auroc"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["macro_auroc"] = float("nan")
    return metrics


def continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Args:
        y_true: ground truth values, shape (N,)
        y_pred: predicted values, shape (N,)
    """
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    return {
        "rmse": rmse,
        "spearman_r": float(spearman_r),
        "pearson_r": float(pearson_r),
    }


def summarize_results(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Format a dict of {target: metrics_dict} into a DataFrame."""
    return pd.DataFrame(results).T.round(4)
