"""
Macenko stain normalization for H&E tiles.

Reference: Macenko et al., "A method for normalizing histology slides for
quantitative analysis," ISBI 2009.

Normalizes each tile to a reference stain matrix so that batch effects from
different scanners/labs don't confound the model's feature extraction.
"""

import numpy as np
from PIL import Image


TCGA_REFERENCE_MATRIX = np.array(
    [[0.5626, 0.7201, 0.4062],
     [0.7201, 0.4062, 0.5626]],
    dtype=np.float32,
)


class MacenkoNormalizer:
    """
    Fit a reference stain matrix from a target image, then normalize tiles.

    Usage:
        normalizer = MacenkoNormalizer()
        normalizer.fit(reference_tile_rgb)          # once, on a representative tile
        normalized = normalizer.transform(tile_rgb)  # for each tile
    """

    def __init__(self, beta: float = 0.15, alpha: float = 1.0):
        self.beta = beta
        self.alpha = alpha
        self.stain_matrix_target: np.ndarray | None = None
        self.max_concentrations: np.ndarray | None = None

    def fit(self, target_rgb: np.ndarray) -> "MacenkoNormalizer":
        """Estimate stain matrix from a reference tile."""
        OD = self._rgb_to_od(target_rgb)
        self.stain_matrix_target = self._get_stain_matrix(OD)
        self.max_concentrations = self._get_concentrations(OD, self.stain_matrix_target).max(axis=0)
        return self

    def transform(self, tile_rgb: np.ndarray) -> np.ndarray:
        """Normalize a tile to the reference stain matrix."""
        if self.stain_matrix_target is None:
            raise RuntimeError("Call fit() before transform().")

        OD = self._rgb_to_od(tile_rgb)
        stain_matrix_source = self._get_stain_matrix(OD)
        concentrations = self._get_concentrations(OD, stain_matrix_source)

        max_conc = concentrations.max(axis=0)
        concentrations = concentrations / np.maximum(max_conc, 1e-6)
        concentrations *= self.max_concentrations

        OD_normalized = concentrations @ self.stain_matrix_target
        rgb_normalized = np.exp(-OD_normalized).reshape(tile_rgb.shape)
        return np.clip(rgb_normalized * 255, 0, 255).astype(np.uint8)

    def _rgb_to_od(self, rgb: np.ndarray) -> np.ndarray:
        rgb = rgb.reshape(-1, 3).astype(np.float32)
        rgb = np.maximum(rgb, 1.0)
        return -np.log(rgb / 255.0)

    def _get_stain_matrix(self, OD: np.ndarray) -> np.ndarray:
        OD = OD[(OD > self.beta).all(axis=1)]
        if len(OD) < 10:
            return TCGA_REFERENCE_MATRIX

        _, _, Vt = np.linalg.svd(OD, full_matrices=False)
        plane = Vt[:2]

        projected = OD @ plane.T
        angles = np.arctan2(projected[:, 1], projected[:, 0])
        lo, hi = np.percentile(angles, [self.alpha, 100 - self.alpha])

        v1 = np.array([np.cos(lo), np.sin(lo)]) @ plane
        v2 = np.array([np.cos(hi), np.sin(hi)]) @ plane

        if v1[0] > v2[0]:
            v1, v2 = v2, v1

        return np.stack([v1, v2], axis=0)

    def _get_concentrations(self, OD: np.ndarray, stain_matrix: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(stain_matrix.T, OD.T, rcond=None)[0].T
