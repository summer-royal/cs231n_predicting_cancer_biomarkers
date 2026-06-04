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
    modal run scripts/modal_pipeline.py --encoder uni
    modal run scripts/modal_pipeline.py --step download
    modal run scripts/modal_pipeline.py --step process
    modal run scripts/modal_pipeline.py --step train
    modal run scripts/modal_pipeline.py --step cv --encoder uni
    modal run scripts/modal_pipeline.py --step cvb --encoder resnet50
    modal run scripts/modal_pipeline.py --step figures

CONCH note:
    CONCH is installed in the Modal image from the MahmoodLab GitHub repo.
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
    .apt_install(["git", "libgl1", "libglib2.0-0"])
    .pip_install(
        "torch", "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "openslide-python>=1.3.0", "openslide-bin",  # 1.3+ auto-loads bundled lib
        "h5py", "pandas", "scikit-learn", "scipy",
        "tqdm", "requests", "Pillow", "numpy", "timm", "huggingface_hub",
        "matplotlib", "seaborn",
    )
    .pip_install("git+https://github.com/mahmoodlab/CONCH.git")
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

    # SVS/TIFF files start with II (little-endian) or MM (big-endian).
    # If we got an HTML/JSON error page instead of a real slide, delete it.
    with open(dest, "rb") as f:
        magic = f.read(4)
    if magic[:2] not in (b"\x49\x49", b"\x4D\x4D"):
        size = dest.stat().st_size
        dest.unlink()
        return f"bad_download {case_id} size={size} magic={magic.hex()}"

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
def process_slide(case_id: str, encoder_name: str = "resnet50", hf_token: str | None = None) -> str:
    sys.path.insert(0, "/app")
    import torch
    from preprocessing import TilePipeline
    from models import get_encoder
    from scripts.extract_features import extract_slide

    if encoder_name in ("uni", "conch") and not hf_token:
        return f"missing_hf_token {case_id}: set HF_TOKEN before running gated encoder '{encoder_name}'"

    slide_path = REMOTE_ROOT / "raw"      / f"{case_id}.svs"
    tile_h5    = REMOTE_ROOT / "tiles"    / f"{case_id}.h5"
    features_subdir = f"features_{encoder_name}"
    feat_h5    = REMOTE_ROOT / features_subdir / f"{case_id}.h5"

    for d in ("tiles", features_subdir):
        (REMOTE_ROOT / d).mkdir(parents=True, exist_ok=True)

    if not slide_path.exists():
        return f"missing_slide {case_id}"

    # Validate file is a real SVS/TIFF before handing it to OpenSlide
    with open(slide_path, "rb") as f:
        magic = f.read(4)
    if magic[:2] not in (b"\x49\x49", b"\x4D\x4D"):
        return f"invalid_file {case_id} size={slide_path.stat().st_size} magic={magic.hex()}"

    if not tile_h5.exists():
        try:
            n = TilePipeline().process_slide(str(slide_path), str(tile_h5))
        except Exception as exc:
            return f"tile_error {case_id}: {exc}"
        if n == 0:
            return f"no_tissue {case_id}"

    if not feat_h5.exists():
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            encoder = get_encoder(encoder_name, device, hf_token=hf_token)
            extract_slide(str(slide_path), str(tile_h5), str(feat_h5),
                          encoder, device, batch_size=256)
        except Exception as exc:
            return f"extract_error {case_id}: {exc}"

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
def run_train(labels_csv_content: str, encoder_name: str = "resnet50") -> None:
    sys.path.insert(0, "/app")
    import subprocess

    labels_csv   = REMOTE_ROOT / "labels" / encoder_name / "tcga_brca_labels.csv"
    splits_dir   = REMOTE_ROOT / "splits" / encoder_name
    features_dir = REMOTE_ROOT / f"features_{encoder_name}"
    ckpt_dir     = REMOTE_ROOT / "checkpoints" / encoder_name
    figures_dir  = REMOTE_ROOT / "figures" / encoder_name

    # Backward compatibility with earlier ResNet-50 runs that wrote to
    # /data/features before encoder-specific feature directories existed.
    legacy_resnet_features = REMOTE_ROOT / "features"
    if encoder_name == "resnet50" and not features_dir.exists() and legacy_resnet_features.exists():
        features_dir = legacy_resnet_features

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

    # Train clam_sb and tumor_aware for each binary target (ablation).
    # Skip any model whose checkpoint already exists so the step is
    # idempotent/resumable and we don't re-burn credits on prior runs.
    for target in ["ER_status", "PR_status", "HER2_status"]:
        for model in ["clam_sb", "tumor_aware"]:
            ckpt = ckpt_dir / f"{target}_{model}_best.pt"
            if ckpt.exists():
                print(f"[skip] {ckpt.name} already exists — not retraining")
                continue
            subprocess.run([
                "python", "/app/scripts/run_training.py",
                "--feature_dir", str(features_dir),
                "--labels_csv",  str(labels_csv),
                "--splits_dir",  str(splits_dir),
                "--target",      target,
                "--model",       model,
                "--in_dim",      "auto",
                "--epochs",      "20",
                "--save_dir",    str(ckpt_dir),
            ], check=True)

    # Persist final ablation artifacts for report analysis. This re-fits the
    # mean-pool baseline on the same split and overlays it with both MIL models.
    for split in ["val", "test"]:
        for target in ["ER_status", "PR_status", "HER2_status"]:
            subprocess.run([
                "python", "/app/scripts/plot_ablation.py",
                "--feature_dir",  str(features_dir),
                "--labels_csv",   str(labels_csv),
                "--splits_dir",   str(splits_dir),
                "--target",       target,
                "--split",        split,
                "--checkpoints",
                str(ckpt_dir / f"{target}_clam_sb_best.pt"),
                str(ckpt_dir / f"{target}_tumor_aware_best.pt"),
                "--models",       "clam_sb", "tumor_aware",
                "--model_labels", "CLAM-SB", "Tumor-Aware CLAM",
                "--in_dim",       "auto",
                "--include_baseline",
                "--out_dir",      str(figures_dir),
            ], check=True)

    vol.commit()


@app.function(
    image=image,
    volumes={str(REMOTE_ROOT): vol},
    gpu="T4",
    timeout=60 * 60 * 6,
    cpu=4,
)
def run_cv(labels_csv_content: str, encoder_name: str = "resnet50",
           n_folds: int = 5, epochs: int = 20) -> None:
    sys.path.insert(0, "/app")
    import subprocess

    labels_csv   = REMOTE_ROOT / "labels" / encoder_name / "tcga_brca_labels.csv"
    features_dir = REMOTE_ROOT / f"features_{encoder_name}"
    cv_dir       = REMOTE_ROOT / "cv" / encoder_name

    legacy_resnet_features = REMOTE_ROOT / "features"
    if encoder_name == "resnet50" and not features_dir.exists() and legacy_resnet_features.exists():
        features_dir = legacy_resnet_features

    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    labels_csv.write_text(labels_csv_content)

    for target in ["ER_status", "PR_status", "HER2_status"]:
        subprocess.run([
            "python", "/app/scripts/run_cv.py",
            "--feature_dir", str(features_dir),
            "--labels_csv",  str(labels_csv),
            "--target",      target,
            "--models",      "clam_sb", "tumor_aware",
            "--include_baseline",
            "--in_dim",      "auto",
            "--n_folds",     str(n_folds),
            "--epochs",      str(epochs),
            "--out_dir",     str(cv_dir),
        ], check=True)

    subprocess.run([
        "python", "/app/scripts/plot_cv_summary.py",
        "--cv_dir",  str(cv_dir),
        "--targets", "ER_status", "PR_status", "HER2_status",
        "--out_dir", str(cv_dir),
    ], check=True)

    vol.commit()


# tumor aware gate variants (extension)
# (tag, display label, extra CLI flags)
GATE_VARIANTS = [
    ("residual", "Tumor-Aware (residual)",
     ["--gate_mode", "residual", "--gate_alpha", "1.0"]),
    ("residual_reg", "Tumor-Aware (residual+reg)",
     ["--gate_mode", "residual", "--gate_alpha", "1.0",
      "--reg_mode", "both", "--reg_weight", "0.1", "--gate_budget", "0.25"]),
    ("residual_reg_temp", "Tumor-Aware (residual+reg+temp)",
     ["--gate_mode", "residual", "--gate_alpha", "1.0",
      "--reg_mode", "both", "--reg_weight", "0.1", "--gate_budget", "0.25",
      "--learn_temp"]),
]


@app.function(
    image=image,
    volumes={str(REMOTE_ROOT): vol},
    gpu="T4",
    timeout=60 * 60 * 6,
    cpu=4,
)
def run_cv_variants(labels_csv_content: str, encoder_name: str = "resnet50",
                    n_folds: int = 5, epochs: int = 20) -> None:
    sys.path.insert(0, "/app")
    import subprocess

    labels_csv   = REMOTE_ROOT / "labels" / encoder_name / "tcga_brca_labels.csv"
    features_dir = REMOTE_ROOT / f"features_{encoder_name}"
    cv_dir       = REMOTE_ROOT / "cv" / encoder_name

    legacy_resnet_features = REMOTE_ROOT / "features"
    if encoder_name == "resnet50" and not features_dir.exists() and legacy_resnet_features.exists():
        features_dir = legacy_resnet_features

    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    labels_csv.write_text(labels_csv_content)

    for target in ["ER_status", "PR_status", "HER2_status"]:
        for tag, label, flags in GATE_VARIANTS:
            subprocess.run([
                "python", "/app/scripts/run_cv.py",
                "--feature_dir", str(features_dir),
                "--labels_csv",  str(labels_csv),
                "--target",      target,
                "--models",      "tumor_aware",
                "--in_dim",      "auto",
                "--n_folds",     str(n_folds),
                "--epochs",      str(epochs),
                "--out_dir",     str(cv_dir),
                "--tag",         tag,
                "--model_label", label,
            ] + flags, check=True)

    vol.commit()


@app.function(
    image=image,
    volumes={str(REMOTE_ROOT): vol},
    gpu="T4",
    timeout=60 * 60 * 4,
    cpu=4,
)
def run_figures() -> None:
    sys.path.insert(0, "/app")
    import subprocess

    subprocess.run([
        "python", "/app/scripts/run_all_ablation.py",
        "--root", str(REMOTE_ROOT),
        "--encoders", "resnet50", "uni", "conch",
        "--targets", "ER_status", "PR_status", "HER2_status",
        "--splits", "val", "test",
        "--summary_dir", str(REMOTE_ROOT / "figures" / "summary"),
    ], check=True)
    vol.commit()


# ---------------------------------------------------------------------------
# Local entrypoint — runs on your laptop, dispatches to Modal
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(step: str = "all", encoder: str = "resnet50"):
    import pandas as pd
    import os

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
        hf_token = os.environ.get("HF_TOKEN")
        if encoder in ("uni", "conch") and not hf_token:
            raise RuntimeError(
                f"Encoder '{encoder}' requires gated Hugging Face access. "
                "Set HF_TOKEN in this shell before running Modal."
            )
        if encoder in ("uni", "conch"):
            print("HF_TOKEN found locally; passing it to Modal workers.")
        for res in process_slide.starmap((case_id, encoder, hf_token) for case_id in case_ids):
            print(f"  {res}")

    if step in ("all", "train"):
        print("Running baseline + CLAM training …")
        run_train.remote(labels_csv_path.read_text(), encoder)
        print("Training dispatched — follow logs at modal.com/apps")

    if step == "cv":
        print(f"Running 5-fold cross-validation ({encoder}) …")
        run_cv.remote(labels_csv_path.read_text(), encoder)
        print("CV dispatched — follow logs at modal.com/apps")

    if step == "cvb":
        print(f"Running 5-fold CV for tumor-aware gate variants ({encoder}) …")
        run_cv_variants.remote(labels_csv_path.read_text(), encoder)
        print("Tumor-aware variant CV dispatched — follow logs at modal.com/apps")

    if step == "figures":
        print("Generating cross-encoder ablation figures and summary tables …")
        run_figures.remote()
        print("Figure generation dispatched — follow logs at modal.com/apps")
