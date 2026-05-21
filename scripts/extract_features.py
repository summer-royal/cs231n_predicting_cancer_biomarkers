"""
Extract patch embeddings from tiled slides using a pretrained encoder.

For each slide, reads tile coordinates from data/tiles/<case_id>.h5,
extracts patch embeddings in batches, and writes them to data/features/<case_id>.h5.

Usage:
    python scripts/extract_features.py \
        --tile_dir data/tiles \
        --slide_dir data/raw \
        --output_dir data/features \
        --encoder resnet50 \
        --batch_size 256
"""

import argparse
import h5py
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import get_encoder
from preprocessing import MacenkoNormalizer


class PatchDataset(Dataset):
    def __init__(self, slide, coords, patch_size, patch_level, normalizer=None):
        self.slide = slide
        self.coords = coords
        self.patch_size = patch_size
        self.patch_level = patch_level
        self.normalizer = normalizer

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        import torchvision.transforms.functional as TF
        x, y = self.coords[idx]
        tile = self.slide.read_region((int(x), int(y)), self.patch_level, (self.patch_size, self.patch_size))
        tile = tile.convert("RGB")
        arr = np.array(tile)
        if self.normalizer:
            arr = self.normalizer.transform(arr)
        return TF.to_tensor(Image.fromarray(arr))


def extract_slide(slide_path, tile_h5, output_h5, encoder, device, batch_size=256, normalize=True):
    import openslide
    slide = openslide.OpenSlide(slide_path)

    with h5py.File(tile_h5, "r") as f:
        coords = f["coords"][:]
        patch_size = int(f.attrs["patch_size"])
        patch_level = int(f.attrs["patch_level"])

    normalizer = MacenkoNormalizer() if normalize else None

    dataset = PatchDataset(slide, coords, patch_size, patch_level, normalizer)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    all_feats = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            feats = encoder(batch).cpu().numpy()
            all_feats.append(feats)

    all_feats = np.concatenate(all_feats, axis=0)
    with h5py.File(output_h5, "w") as f:
        f.create_dataset("features", data=all_feats)
        f.create_dataset("coords", data=coords)
    slide.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tile_dir", required=True)
    p.add_argument("--slide_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--encoder", default="resnet50")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--no_stain_norm", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = get_encoder(args.encoder, device)

    tile_dir = Path(args.tile_dir)
    slide_dir = Path(args.slide_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for tile_h5 in tqdm(sorted(tile_dir.glob("*.h5")), desc="Extracting features"):
        case_id = tile_h5.stem
        slide_path = slide_dir / f"{case_id}.svs"
        output_h5 = output_dir / f"{case_id}.h5"
        if output_h5.exists() or not slide_path.exists():
            continue
        extract_slide(str(slide_path), str(tile_h5), str(output_h5),
                      encoder, device, args.batch_size, not args.no_stain_norm)


if __name__ == "__main__":
    main()
