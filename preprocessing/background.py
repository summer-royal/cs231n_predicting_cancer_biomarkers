"""
Background / tissue detection for H&E tiles.

Strategy: convert to HSV, threshold saturation channel with Otsu's method.
H&E-stained tissue has high saturation; glass/fat/white space does not.
"""

import numpy as np
from PIL import Image


def is_tissue(tile_rgb: np.ndarray, threshold: float = 0.5) -> bool:
    """
    Return True if at least `threshold` fraction of the tile is tissue.

    Args:
        tile_rgb: uint8 array of shape (H, W, 3).
        threshold: minimum fraction of pixels that must be tissue.
    """
    tissue_mask = _otsu_tissue_mask(tile_rgb)
    return tissue_mask.mean() >= threshold


def _otsu_tissue_mask(tile_rgb: np.ndarray) -> np.ndarray:
    """Binary mask: True = tissue pixel, using Otsu on HSV saturation."""
    img = Image.fromarray(tile_rgb).convert("HSV")
    hsv = np.array(img)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0

    # Otsu threshold on the saturation channel
    thresh = _otsu_threshold(saturation)
    return saturation > thresh


def _otsu_threshold(channel: np.ndarray) -> float:
    """Compute Otsu's threshold for a [0,1] float array."""
    hist, bin_edges = np.histogram(channel, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    mean_bg = np.cumsum(hist * bin_centers) / np.maximum(weight_bg, 1)
    mean_fg = (np.sum(hist * bin_centers) - np.cumsum(hist * bin_centers)) / np.maximum(weight_fg, 1)

    variance_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    best_idx = int(np.argmax(variance_between))
    return float(bin_centers[best_idx])
