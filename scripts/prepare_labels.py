"""
Prepare TCGA-BRCA labels CSV + GDC download manifest.

Clinical labels (ER, PR, HER2, PAM50) come from cBioPortal — they curate the
TCGA receptor status data in a clean, queryable form. The slide file manifest
still uses the GDC REST API since that's where the .svs files live.

No login or API key required for either service.

Usage (200 well-labeled slides):
    python scripts/prepare_labels.py \
        --out_csv data/labels/tcga_brca_labels.csv \
        --out_manifest data/manifest.txt \
        --max_slides 200

Full cohort (~1,000 slides):
    python scripts/prepare_labels.py --max_slides 0

Outputs:
    data/labels/tcga_brca_labels.csv  — case_id, ER_status, PR_status,
                                        HER2_status, PAM50_subtype
    data/manifest.txt                 — GDC manifest for gdc-client download
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

GDC_FILES_URL      = "https://api.gdc.cancer.gov/files"
CBIOPORTAL_API     = "https://www.cbioportal.org/api"
CBIOPORTAL_STUDIES = ["brca_tcga", "brca_tcga_pan_can_atlas_2018"]

# cBioPortal attribute IDs to try, in priority order, for each target
ATTR_CANDIDATES = {
    "ER_status":     ["ER_STATUS_BY_IHC", "ER_STATUS",  "IHC_ER"],
    "PR_status":     ["PR_STATUS_BY_IHC", "PR_STATUS",  "IHC_PR"],
    "HER2_status":   ["IHC_HER2", "HER2_STATUS", "HER2_STATUS_BY_IHC", "HER_2_STATUS"],
    "PAM50_subtype": ["CLAUDIN_SUBTYPE", "SUBTYPE", "PAM50_RNASEQ_SUBTYPE"],
}

# Binary IHC value → 0/1
IHC_MAP = {"positive": 1, "negative": 0}

# cBioPortal PAM50 label → our PAM50_CLASSES
PAM50_MAP = {
    "luma": "LumA",        "luminal a": "LumA",
    "lumb": "LumB",        "luminal b": "LumB",
    "her2": "HER2E",       "her2-enriched": "HER2E",  "her2e": "HER2E",
    "basal": "Basal",      "basal-like": "Basal",
    "normal": "Normal",    "normal-like": "Normal",
}


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get(url: str, params: dict = None, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [{attempt+1}/{retries}] {exc} — retry in {wait}s")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# cBioPortal: clinical labels
# ---------------------------------------------------------------------------

def _fetch_cbio_study(study_id: str) -> pd.DataFrame:
    """Return a pivot DataFrame (patientId × attributeId) for one study."""
    url = f"{CBIOPORTAL_API}/studies/{study_id}/clinical-data"
    rows = []
    for data_type in ("PATIENT", "SAMPLE"):
        data = _get(url, params={
            "clinicalDataType": data_type,
            "projection": "DETAILED",
            "pageSize": 10000,
            "pageNumber": 0,
        })
        rows.extend(data)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Normalise the patient-ID column (samples use sampleId, patients use patientId)
    id_col = "patientId" if "patientId" in df.columns else "sampleId"
    # For sample-level data the patient id is embedded in sampleId (TCGA-XX-XXXX-01)
    df["_patient"] = df[id_col].str[:12]        # e.g. TCGA-A1-A0SB
    pivot = df.pivot_table(
        index="_patient", columns="clinicalAttributeId", values="value", aggfunc="first"
    )
    pivot.index.name = "case_id"
    return pivot


def fetch_clinical() -> pd.DataFrame:
    """
    Query cBioPortal for TCGA-BRCA ER/PR/HER2/PAM50 labels.

    Tries multiple study IDs and attribute name variants so the script is
    robust to cBioPortal schema changes.
    """
    pivot = pd.DataFrame()
    for study_id in CBIOPORTAL_STUDIES:
        print(f"  Trying cBioPortal study '{study_id}' …")
        try:
            pivot = _fetch_cbio_study(study_id)
            if not pivot.empty:
                print(f"    Got {len(pivot)} patients, {len(pivot.columns)} attributes")
                break
        except Exception as exc:
            print(f"    Failed: {exc}")

    if pivot.empty:
        print("ERROR: could not fetch clinical data from cBioPortal.", file=sys.stderr)
        sys.exit(1)

    result = pd.DataFrame(index=pivot.index)
    result.index.name = "case_id"

    # --- Binary IHC targets ---
    for col, candidates in ATTR_CANDIDATES.items():
        if col == "PAM50_subtype":
            continue
        matched = next((c for c in candidates if c in pivot.columns), None)
        if matched:
            result[col] = pivot[matched].str.lower().map(IHC_MAP)
            print(f"    {col} ← '{matched}'  "
                  f"(pos={int(result[col].eq(1).sum())}  "
                  f"neg={int(result[col].eq(0).sum())}  "
                  f"missing={int(result[col].isna().sum())})")
        else:
            result[col] = float("nan")
            print(f"    {col}: no matching attribute found "
                  f"(tried {candidates})")

    # --- PAM50 ---
    pam_attr = next(
        (c for c in ATTR_CANDIDATES["PAM50_subtype"] if c in pivot.columns), None
    )
    if pam_attr:
        result["PAM50_subtype"] = (
            pivot[pam_attr].str.lower()
            .map(PAM50_MAP)
        )
        vc = result["PAM50_subtype"].value_counts().to_dict()
        print(f"    PAM50_subtype ← '{pam_attr}'  {vc}")
    else:
        result["PAM50_subtype"] = float("nan")
        print(f"    PAM50_subtype: not found (tried {ATTR_CANDIDATES['PAM50_subtype']})")

    return result


# ---------------------------------------------------------------------------
# GDC: slide file manifest
# ---------------------------------------------------------------------------

def fetch_slides(case_ids: list) -> pd.DataFrame:
    """Return DataFrame (indexed by case_id) with file_id, file_name, file_size."""
    print(f"Querying GDC for diagnostic slides ({len(case_ids)} cases) …")
    chunk_size, all_rows = 500, []
    for i in range(0, len(case_ids), chunk_size):
        chunk = case_ids[i : i + chunk_size]
        params = {
            "filters": json.dumps({
                "op": "and",
                "content": [
                    {"op": "in", "content": {
                        "field": "cases.submitter_id", "value": chunk}},
                    {"op": "=",  "content": {
                        "field": "data_type", "value": "Slide Image"}},
                    {"op": "=",  "content": {
                        "field": "experimental_strategy",
                        "value": "Diagnostic Slide"}},
                ],
            }),
            "fields": "file_id,file_name,file_size,cases.submitter_id",
            "expand": "cases",
            "format": "JSON",
            "size": str(len(chunk) * 3),
        }
        hits = _get(GDC_FILES_URL, params)["data"]["hits"]
        for f in hits:
            cases = f.get("cases") or [{}]
            all_rows.append({
                "case_id":   (cases[0].get("submitter_id") or ""),
                "file_id":   f["file_id"],
                "file_name": f["file_name"],
                "file_size": f.get("file_size", 0),
            })

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # Keep one slide per case (largest = highest-resolution)
    df = (df.sort_values("file_size", ascending=False)
            .drop_duplicates("case_id")
            .set_index("case_id"))
    print(f"  {len(df)} unique slides found")
    return df


def write_manifest(slide_df: pd.DataFrame, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["id\tfilename\tmd5\tsize\tstate"]
    for _, row in slide_df.iterrows():
        lines.append(
            f"{row['file_id']}\t{row['file_name']}\t\t{int(row['file_size'])}\treleased"
        )
    out.write_text("\n".join(lines) + "\n")
    print(f"  Manifest → {out_path}  ({len(slide_df)} files)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_csv",      default="data/labels/tcga_brca_labels.csv")
    p.add_argument("--out_manifest", default="data/manifest.txt")
    p.add_argument("--max_slides",   type=int, default=200,
                   help="Cap number of slides. 0 = full cohort.")
    args = p.parse_args()

    print("=== Step 1: clinical labels (cBioPortal) ===")
    labels_df = fetch_clinical()
    print(f"  {len(labels_df)} patients total\n")

    # Prioritise fully-labeled cases
    has_er   = labels_df["ER_status"].notna()
    has_all  = has_er & labels_df["PR_status"].notna() & labels_df["HER2_status"].notna()
    ordered  = (labels_df[has_all].index.tolist()
                + labels_df[has_er & ~has_all].index.tolist()
                + labels_df[~has_er].index.tolist())

    if args.max_slides > 0:
        ordered = ordered[: args.max_slides]
    labels_df = labels_df.loc[ordered]

    print(f"=== Step 2: slide file IDs (GDC) ===")
    slide_df = fetch_slides(ordered)
    if slide_df.empty:
        print("No slides found — check network.", file=sys.stderr)
        sys.exit(1)

    common    = labels_df.index.intersection(slide_df.index)
    labels_df = labels_df.loc[common]
    slide_df  = slide_df.loc[common]
    print(f"\n{len(common)} cases have labels + a diagnostic slide")

    for col in ["ER_status", "PR_status", "HER2_status"]:
        vc = labels_df[col].value_counts().to_dict()
        n_miss = int(labels_df[col].isna().sum())
        print(f"  {col}: pos={int(vc.get(1.0,0))}  neg={int(vc.get(0.0,0))}  missing={n_miss}")

    total_gb = slide_df["file_size"].sum() / 1e9
    print(f"\nEstimated download size: {total_gb:.1f} GB")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(out_csv)
    print(f"Labels CSV → {out_csv}")
    write_manifest(slide_df, args.out_manifest)

    print(f"""
Next steps:
  bash data/download.sh {args.out_manifest} data/raw/
  python scripts/preprocess_slides.py --slide_dir data/raw --output_dir data/tiles
  python scripts/extract_features.py --tile_dir data/tiles --slide_dir data/raw \\
      --output_dir data/features --encoder resnet50
  python -c "from datasets import make_patient_splits; \\
             make_patient_splits('{args.out_csv}', 'data/splits')"
  python scripts/run_baseline.py --feature_dir data/features \\
      --labels_csv {args.out_csv} --splits_dir data/splits
  python scripts/run_training.py --feature_dir data/features \\
      --labels_csv {args.out_csv} --splits_dir data/splits --target ER_status
""")


if __name__ == "__main__":
    main()
