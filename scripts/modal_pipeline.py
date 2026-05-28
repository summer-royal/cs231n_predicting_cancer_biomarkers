"""
Modal pipeline for TCGA-BRCA feature extraction and MIL training.

Runs the full pipeline on Modal cloud GPUs with full parallelism:
  download (parallel) → tile+extract (parallel, GPU) → train

Prerequisites:
    pip install modal
    modal token new          # authenticate (use the account you want)
    python scripts/prepare_labels.py   # creates data/manifest.txt + labels CSV

Usage:
    modal run scripts/modal_pipeline.py              # full pipeline
    modal run scripts/modal_pipeline.py --step download
    modal run scripts/modal_pipeline.py --step process
    modal run scripts/modal_pipeline.py --step train
"""

import sys
from pathlib import Path

import modal

app = modal.App("tcga-brca-pipeline")

# Persistent volume — survives across runs and steps
vol = modal.Volume.from_name("tcga-brca-data", create_if_missing=True)
REMOTE_ROOT = Path("/data")

# Container image: PyTorch base + OpenSlide + project code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install([
        "libgl1", "libglib2.0-0",
        "libopenslide0", "openslide-tools",
    ])
    .pip_install(
        "torch", "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "openslide-python",
        "h5py", "pandas", "scikit-learn", "scipy",
        "tqdm", "requests", "Pillow", "numpy", "timm",
    )
    # Copy project source into the container
    .add_local_dir("preprocessing", "/app/preprocessing")
    .add_local_dir("models",        "/app/models")
    .add_local_dir("datasets",      "/app/datasets")
    .add_local_dir("evaluation",    "/app/evaluation")
    .add_local_dir("training",      "/app/training")
    .add_local_dir("scripts",       "/app/scripts")
)


def _case_id(filename: str) -> str:
    """TCGA-A1-A0SB-01Z-... → TCGA-A1-A0SB"""
    return "-".join(filename.split("-")[:3])


# ---------------------------------------------------------------------------
# Step 1: Download one slide directly from GDC (no gdc-client needed)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={str(REMOTE_ROOT): vol},
    timeout=60 * 90,
    retries=2,
)
def download_slide(file_id: str, case_id: str) -> str:
    import requests

    dest = REMOTE_ROOT / "raw" / f"{case_id}.svs"
    (REMOTE_ROOT / "raw").mkdir(parents=True, exist_ok=True)

    if dest.exists():
        return f"skip  {case_id}"

    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)

    vol.commit()
    return f"ok    {case_id}"


# ---------------------------------------------------------------------------
# Step 2: Tile + extract features for one slide (GPU, parallel across slides)
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={str(REMOTE_ROOT): vol},
    gpu="T4",
    timeout=60 * 60,
    retries=1,
)
def process_slide(case_id: str) -> str:
    sys.path.insert(0, "/app")
    import torch
    from preprocessing import TilePipeline
    from models import get_encoder
    from scripts.extract_features import extract_slide

    slide_path = REMOTE_ROOT / "raw"      / f"{case_id}.svs"
    tile_h5    = REMOTE_ROOT / "tiles"    / f"{case_id}.h5"
    feat_h5    = REMOTE_ROOT / "features" / f"{case_id}.h5"

    for d in ("tiles", "features"):
        (REMOTE_ROOT / d).mkdir(parents=True, exist_ok=True)

    if not slide_path.exists():
        return f"missing_slide {case_id}"

    if not tile_h5.exists():
        n = TilePipeline().process_slide(str(slide_path), str(tile_h5))
        if n == 0:
            return f"no_tissue {case_id}"

    if not feat_h5.exists():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = get_encoder("resnet50", device)
        extract_slide(str(slide_path), str(tile_h5), str(feat_h5),
                      encoder, device, batch_size=256)

    vol.commit()
    return f"ok    {case_id}"


# ---------------------------------------------------------------------------
# Step 3: Create splits, run baseline, train CLAM on all three targets
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={str(REMOTE_ROOT): vol},
    gpu="T4",
    timeout=60 * 60 * 4,
    cpu=4,
)
def run_train(labels_csv_content: str) -> None:
    sys.path.insert(0, "/app")
    import subprocess

    labels_csv   = REMOTE_ROOT / "labels" / "tcga_brca_labels.csv"
    splits_dir   = REMOTE_ROOT / "splits"
    features_dir = REMOTE_ROOT / "features"
    ckpt_dir     = REMOTE_ROOT / "checkpoints"

    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    labels_csv.write_text(labels_csv_content)

    # Patient-level splits
    subprocess.run([
        "python", "-c",
        "import sys; sys.path.insert(0, '/app'); "
        "from datasets import make_patient_splits; "
        f"make_patient_splits('{labels_csv}', '{splits_dir}')"
    ], check=True)

    # Mean-pool + LR baseline
    subprocess.run([
        "python", "/app/scripts/run_baseline.py",
        "--feature_dir", str(features_dir),
        "--labels_csv",  str(labels_csv),
        "--splits_dir",  str(splits_dir),
    ], check=True)

    # CLAM_SB for each binary target
    for target in ["ER_status", "PR_status", "HER2_status"]:
        subprocess.run([
            "python", "/app/scripts/run_training.py",
            "--feature_dir", str(features_dir),
            "--labels_csv",  str(labels_csv),
            "--splits_dir",  str(splits_dir),
            "--target",      target,
            "--model",       "clam_sb",
            "--in_dim",      "2048",
            "--epochs",      "20",
            "--save_dir",    str(ckpt_dir),
        ], check=True)

    vol.commit()


# ---------------------------------------------------------------------------
# Local entrypoint — runs on your laptop, dispatches to Modal
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(step: str = "all"):
    import pandas as pd

    manifest_path   = Path("data/manifest.txt")
    labels_csv_path = Path("data/labels/tcga_brca_labels.csv")

    if not manifest_path.exists():
        print("Run python scripts/prepare_labels.py first.")
        return

    manifest   = pd.read_csv(manifest_path, sep="\t")
    file_ids   = manifest["id"].tolist()
    file_names = manifest["filename"].tolist()
    case_ids   = [_case_id(fn) for fn in file_names]

    if step in ("all", "download"):
        print(f"Downloading {len(file_ids)} slides in parallel …")
        for res in download_slide.starmap(zip(file_ids, case_ids)):
            print(f"  {res}")

    if step in ("all", "process"):
        print(f"Tiling + extracting features for {len(case_ids)} slides …")
        for res in process_slide.map(case_ids):
            print(f"  {res}")

    if step in ("all", "train"):
        print("Running baseline + CLAM training …")
        run_train.remote(labels_csv_path.read_text())
        print("Training dispatched — follow logs at modal.com/apps")
