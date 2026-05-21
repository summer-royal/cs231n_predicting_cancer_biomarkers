# Data

## Dataset: TCGA-BRCA

Source: [GDC Data Portal](https://portal.gdc.cancer.gov/projects/TCGA-BRCA)
~1,000 diagnostic H&E whole-slide images (.svs) from breast invasive carcinoma cases.

### Expected directory layout after download

```
data/
├── raw/          # .svs files (not committed — see .gitignore)
├── features/     # extracted patch embeddings as .h5 files (not committed)
├── labels/
│   └── tcga_brca_labels.csv   # ER, PR, HER2, PAM50, transcriptomic targets
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

### Downloading slides

```bash
# Install GDC client first: https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
bash data/download.sh
```

### Label sources

- ER/PR/HER2 IHC status: GDC clinical endpoint files
- PAM50 subtype: TCGA supplementary tables (Nature 2012)
- Transcriptomic signatures: TCGA RNA-seq (log-normalized TPM)
