"""
Tile WSIs into non-overlapping patches and save coordinates.

Pipeline per slide:
  1. Open slide at target magnification (default 20x → 256x256 px tiles).
  2. Generate a grid of tile coordinates.
  3. Filter out background tiles via is_tissue().
  4. Save tissue tile coordinates to an HDF5 file for downstream feature extraction.
"""

import os
import h5py
import numpy as np
from pathlib import Path
from typing import List, Tuple

import openslide
from PIL import Image
from tqdm import tqdm

from .background import is_tissue


TILE_SIZE_PX = 256
TARGET_MAG = 20
TISSUE_THRESHOLD = 0.5  # fraction of tile that must be tissue


class TilePipeline:
    def __init__(
        self,
        tile_size: int = TILE_SIZE_PX,
        target_mag: float = TARGET_MAG,
        tissue_thresh: float = TISSUE_THRESHOLD,
    ):
        self.tile_size = tile_size
        self.target_mag = target_mag
        self.tissue_thresh = tissue_thresh

    def process_slide(self, slide_path: str, output_h5: str) -> int:
        """
        Tile a single WSI and write tissue tile coordinates to output_h5.
        Returns the number of tissue tiles kept.
        """
        slide = openslide.OpenSlide(slide_path)
        level, downsample = self._find_level(slide)
        tile_size_at_level = int(self.tile_size * downsample)

        w, h = slide.level_dimensions[level]
        coords: List[Tuple[int, int]] = []

        for y in range(0, h, tile_size_at_level):
            for x in range(0, w, tile_size_at_level):
                tile = slide.read_region((x, y), level, (tile_size_at_level, tile_size_at_level))
                tile = tile.convert("RGB").resize((self.tile_size, self.tile_size), Image.BILINEAR)
                if is_tissue(np.array(tile), threshold=self.tissue_thresh):
                    coords.append((x, y))

        coords_arr = np.array(coords, dtype=np.int32)
        with h5py.File(output_h5, "w") as f:
            f.create_dataset("coords", data=coords_arr)
            f.attrs["slide_path"] = str(slide_path)
            f.attrs["patch_size"] = self.tile_size
            f.attrs["patch_level"] = level
            f.attrs["target_mag"] = self.target_mag

        slide.close()
        return len(coords)

    def _find_level(self, slide: openslide.OpenSlide) -> Tuple[int, float]:
        """Return (level_idx, downsample_factor) for the target magnification."""
        native_mag = float(slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40))
        target_downsample = native_mag / self.target_mag
        best_level = slide.get_best_level_for_downsample(target_downsample)
        actual_downsample = slide.level_downsamples[best_level]
        return best_level, target_downsample / actual_downsample


def tile_cohort(
    slide_dir: str,
    output_dir: str,
    tile_size: int = TILE_SIZE_PX,
    target_mag: float = TARGET_MAG,
) -> None:
    """Tile all .svs files in slide_dir and save coordinates to output_dir."""
    slide_dir = Path(slide_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = TilePipeline(tile_size=tile_size, target_mag=target_mag)
    slides = sorted(slide_dir.glob("*.svs"))

    for slide_path in tqdm(slides, desc="Tiling slides"):
        h5_path = output_dir / (slide_path.stem + ".h5")
        if h5_path.exists():
            continue
        n_tiles = pipeline.process_slide(str(slide_path), str(h5_path))
        print(f"  {slide_path.name}: {n_tiles} tissue tiles")
