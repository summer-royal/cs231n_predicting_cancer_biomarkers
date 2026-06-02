"""
Extract patch embeddings from tiled slides using a pretrained encoder.

For each slide, reads tile coordinates from data/tiles/<case_id>.h5,
extracts patch embeddings in batches, and writes them to an encoder-specific
feature directory such as data/features_resnet50/<case_id>.h5.

Usage:
    python scripts/extract_features.py \
        --tile_dir data/tiles \
        --slide_dir data/raw \
        --output_dir data/features_resnet50 \
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

from models import get_encoder, list_encoders
from preprocessing.stain_norm import MacenkoNormalizer


class PatchDataset(Dataset):
    def __init__(self, slide, coords, patch_size, patch_level, transform, normalizer=None):
        self.slide = slide
        self.coords = coords
        self.patch_size = patch_size
        self.patch_level = patch_level
        self.transform = transform
        self.normalizer = normalizer

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]
        tile = self.slide.read_region((int(x), int(y)), self.patch_level, (self.patch_size, self.patch_size))
        tile = tile.convert("RGB")
        arr = np.array(tile)
        if self.normalizer:
            arr = self.normalizer.transform(arr)
        return self.transform(Image.fromarray(arr).convert("RGB"))


def extract_slide(
    slide_path,
    tile_h5,
    output_h5,
    encoder,
    device,
    batch_size=256,
    normalize=True,
    num_workers=4,
):
    import openslide
    slide = openslide.OpenSlide(slide_path)

    with h5py.File(tile_h5, "r") as f:
        coords = f["coords"][:]
        patch_size = int(f.attrs["patch_size"])
        patch_level = int(f.attrs["patch_level"])

    if len(coords) == 0:
        slide.close()
        raise ValueError(f"No tissue coordinates found in {tile_h5}")

    normalizer = None
    if normalize:
        normalizer = MacenkoNormalizer()
        # Fit on the first patch of this slide as the reference stain
        ref_x, ref_y = int(coords[0][0]), int(coords[0][1])
        ref_tile = slide.read_region((ref_x, ref_y), patch_level, (patch_size, patch_size))
        normalizer.fit(np.array(ref_tile.convert("RGB")))

    dataset = PatchDataset(slide, coords, patch_size, patch_level, encoder.transform, normalizer)
    pin_memory = device.startswith("cuda")
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

    all_feats = []
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            feats = encoder(batch).cpu().numpy()
            all_feats.append(feats)

    all_feats = np.concatenate(all_feats, axis=0)
    write_feature_h5(
        output_h5,
        all_feats,
        coords,
        encoder_name=encoder.name,
        feature_dim=encoder.feature_dim,
        checkpoint_source=encoder.checkpoint_source,
        stain_norm=normalize,
        patch_size=patch_size,
        patch_level=patch_level,
    )
    slide.close()


def write_feature_h5(
    output_h5,
    features,
    coords,
    encoder_name,
    feature_dim,
    checkpoint_source,
    stain_norm,
    patch_size,
    patch_level,
):
    """Write feature bags using the schema consumed by downstream MIL code."""
    with h5py.File(output_h5, "w") as f:
        f.create_dataset("features", data=features)
        f.create_dataset("coords", data=coords)
        f.attrs["encoder"] = encoder_name
        f.attrs["feature_dim"] = int(feature_dim)
        f.attrs["checkpoint_source"] = checkpoint_source
        f.attrs["stain_norm"] = bool(stain_norm)
        f.attrs["patch_size"] = int(patch_size)
        f.attrs["patch_level"] = int(patch_level)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tile_dir", required=True)
    p.add_argument("--slide_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--encoder", default="resnet50", choices=sorted(list_encoders()))
    p.add_argument("--checkpoint_path", default=None,
                   help="Optional local checkpoint path for gated pathology encoders.")
    p.add_argument("--hf_token", default=None,
                   help="Optional Hugging Face token. Prefer --hf_token_env for normal use.")
    p.add_argument("--hf_token_env", default="HF_TOKEN",
                   help="Environment variable containing a Hugging Face token.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_stain_norm", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = get_encoder(
        args.encoder,
        device,
        checkpoint_path=args.checkpoint_path,
        hf_token=args.hf_token,
        hf_token_env=args.hf_token_env,
    )
    print(
        f"Encoder: {encoder.name} | feature_dim={encoder.feature_dim} | "
        f"source={encoder.checkpoint_source}"
    )

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
                      encoder, device, args.batch_size,
                      not args.no_stain_norm, args.num_workers)


if __name__ == "__main__":
    main()
