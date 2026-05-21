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
python scripts/run_training.py --target ER_status --model clam_sb
```

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
