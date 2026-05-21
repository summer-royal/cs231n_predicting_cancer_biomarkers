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
