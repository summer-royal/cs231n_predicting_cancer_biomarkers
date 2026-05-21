"""
Tile all WSIs and save tissue patch coordinates.

Usage:
    python scripts/preprocess_slides.py \
        --slide_dir data/raw \
        --output_dir data/tiles \
        --tile_size 256 \
        --target_mag 20
"""

import argparse
from preprocessing import tile_cohort


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--slide_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--tile_size", type=int, default=256)
    p.add_argument("--target_mag", type=float, default=20.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tile_cohort(args.slide_dir, args.output_dir, args.tile_size, args.target_mag)
