# Predicting Breast Cancer Molecular Biomarkers from H&E Histopathology

CS 231N Project — Jen Ho, Summer Royal, Luke Zhao (Spring 2026)

## Overview

We predict clinically meaningful molecular phenotypes from routine H&E-stained
whole-slide images (WSIs) of breast cancer using weakly supervised
multiple-instance learning (MIL). Targets include ER, PR, HER2 receptor status,
PAM50 intrinsic subtype, and transcriptomic signatures (estrogen signaling,
proliferation, immune infiltration).

**Dataset**: TCGA-BRCA — ~1,000 diagnostic H&E WSIs from GDC.

## Method

```
WSI (.svs)
  └─ Tiling (256×256 px @ 20x)
       └─ Background removal (Otsu on HSV saturation)
            └─ Stain normalization (Macenko)
                 └─ Patch embeddings (ResNet-50 / UNI)
                      └─ Attention MIL (CLAM)
                           └─ Biomarker predictions
```

**Key modification**: Tumor-aware MIL — a lightweight first-pass scorer
up-weights patches in likely invasive tumor regions before attention
aggregation, reducing confounding from stroma, fat, and slide artifacts.

## Repo Structure

```
preprocessing/      # Tiling, background removal, stain normalization  [Summer]
datasets/           # TCGA-BRCA dataset, patient-level splits          [Summer]
models/             # CLAM, encoder, tumor-aware wrapper               [Jen]
evaluation/         # AUROC, AUPRC, Spearman, Brier score              [Luke]
training/           # Training loop, losses                            [Luke]
scripts/            # End-to-end runnable scripts
data/               # Labels and splits (raw slides not committed)
```

## Setup

```bash
conda create -n cs231n python=3.11
conda activate cs231n
pip install -r requirements.txt

# OpenSlide (macOS)
brew install openslide
```

## Pipeline

### 1. Download data

See [data/README.md](data/README.md) for GDC download instructions.

### 2. Tile slides

```bash
python scripts/preprocess_slides.py \
    --slide_dir data/raw \
    --output_dir data/tiles
```

### 3. Extract patch features

```bash
python scripts/extract_features.py \
    --tile_dir data/tiles \
    --slide_dir data/raw \
    --output_dir data/features \
    --encoder resnet50
```

### 4. Create patient splits

```python
from datasets import make_patient_splits
make_patient_splits("data/labels/tcga_brca_labels.csv", "data/splits")
```

### 5. Train

```bash
python scripts/smoke_train_clam.py \
    --features_dir data/features \
    --labels_csv data/labels/tcga_brca_labels.csv \
    --split_csv data/splits/train.csv \
    --target ER_status \
    --task_type binary \
    --model clam_sb \
    --max_patches 512 \
    --epochs 1
```

## Debug / smoke tests

Before launching a real training run, verify that CLAM consumes the
features we produce and outputs the expected shapes.

**Synthetic smoke test (no data required):**

```bash
python scripts/smoke_test_clam.py
```

Checks:
- `CLAM_SB(in_dim=2048).forward(x_{K=128})` → `logits (1, 2)`, `A (K, 1)`, `sum(A) ≈ 1`.
- `CLAM_MB(in_dim=2048, n_classes=5).forward(x)` → `logits (1, 5)`, `A (5, K)`, each row sums to 1.
- Small-bag edge case (`K=4`) with instance clustering — clamps `k_sample` safely.
- `TumorAwareMIL` returns `(logits, A, tumor_probs)` with `tumor_probs ∈ [0, 1]` of shape `(K, 1)`.
- Gradients flow through the tumor scorer end-to-end.

**Real-data forward-pass debug (one slide only):**

```bash
python scripts/smoke_train_clam.py \
    --features_dir data/features \
    --labels_csv data/labels/tcga_brca_labels.csv \
    --split_csv data/splits/train.csv \
    --target PAM50_subtype --task_type multiclass --model clam_mb \
    --epochs 0
```

`--epochs 0` prints the feature/label/logits/attention shapes for one
slide and exits without training. Useful for sanity-checking the encoder
output dim (`in_dim` is inferred from the actual feature tensor).

## Evaluation

| Target type | Metrics |
|---|---|
| Binary (ER, PR, HER2) | AUROC, AUPRC, balanced accuracy, Brier score |
| Multi-class (PAM50) | Macro AUROC, balanced accuracy |
| Continuous (transcriptomic) | Spearman r, Pearson r, RMSE |

## References

1. Liu et al., "Breast Cancer Molecular Subtype Prediction... with Discriminative Patch Selection and MIL," arXiv 2022.
2. Lu et al., "Data-efficient and weakly supervised computational pathology on WSIs" (CLAM), Nature BME 2021.
3. Akbarnejad et al., "Toward Accurate Deep Learning-Based Prediction of Ki67, ER, PR, HER2 from H&E," 2024.
4. Schmauch et al., "A deep learning model to predict RNA-Seq expression from WSIs" (HE2RNA), Nature Comm. 2020.
5. TCGA Network, "Comprehensive molecular portraits of human breast tumours," Nature 2012.
