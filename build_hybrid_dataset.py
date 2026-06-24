"""
build_hybrid_dataset.py

Reads the raw Mendeley blood-cancer CSV, applies the agreed filtering and
label-mapping rules, then augments with 12 synthetic CBC columns sampled from
published clinical reference ranges conditioned on each patient's class.

Label rules (approved 2026-06-24):
  - Drop header-leak junk rows (1 row where Cancer_Type == 'Cancer_Type')
  - Drop all rows where Diagnosis_Result == 'Suspected'
  - 'Ruled Out'  → label 'Normal'
  - 'Confirmed'  → label = Cancer_Type value (ALL / AML / CLL / CML /
                    Lymphoma / Multiple Myeloma)
  Result: 7-class problem, 1 579 rows.

Reference ranges documented in docs/cbc_reference_ranges.md
Sources: Wintrobe's 13th ed.; Harrison's 21st ed.; WHO 5th ed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data"

RAW_FILENAME = "Blood Cancer Diseases dataset  - Sheet1.csv"
raw_path     = DATA_DIR / RAW_FILENAME
out_path     = DATA_DIR / "blood_cancer_hybrid.csv"

if not raw_path.exists():
    sys.exit(
        f"\n[ERROR] Raw CSV not found at:\n  {raw_path}\n"
        "Place the Mendeley CSV in the data/ folder and retry.\n"
    )

# ---------------------------------------------------------------------------
# Load & clean column names
# ---------------------------------------------------------------------------
df = pd.read_csv(raw_path)
df.columns = df.columns.str.strip()

CT_COL  = "Cancer_Type(AML, ALL, CLL)"
DR_COL  = "Diagnosis_Result"
WBC_COL = "Total WBC count(/cumm)"
PLT_COL = "Platelet Count( (/cumm)"

# Drop header-leak junk rows
df = df[df[DR_COL] != "Diagnosis_Result"]
df = df[df[CT_COL] != "Cancer_Type"]

# Decision 2: drop Suspected
df = df[df[DR_COL] != "Suspected"].copy()
df.reset_index(drop=True, inplace=True)

# ---------------------------------------------------------------------------
# Derive 7-class target label
# ---------------------------------------------------------------------------
def derive_label(row) -> str:
    if row[DR_COL] == "Ruled Out":
        return "Normal"
    return row[CT_COL]          # ALL / AML / CLL / CML / Lymphoma / Multiple Myeloma

df["label"] = df.apply(derive_label, axis=1)

# Convert real numeric columns from str → float
for col in [WBC_COL, PLT_COL]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("=" * 60)
print(f"Loaded : {RAW_FILENAME}")
print(f"Shape  : {df.shape[0]} rows × {df.shape[1]} columns (before adding CBC)")
print()
print("Label distribution:")
print(df["label"].value_counts().to_string())

# ---------------------------------------------------------------------------
# CBC reference ranges (mean, std) per class
# Full table: docs/cbc_reference_ranges.md
# ---------------------------------------------------------------------------
CBC_PARAMS: dict[str, dict[str, tuple[float, float]]] = {
    "RBC_M_per_uL": {
        "Normal":           (5.00, 0.40),
        "ALL":              (3.00, 0.40),
        "AML":              (2.50, 0.40),
        "CLL":              (4.00, 0.30),
        "CML":              (3.50, 0.40),
        "Lymphoma":         (3.80, 0.40),
        "Multiple Myeloma": (3.00, 0.40),
    },
    "Hemoglobin_g_dL": {
        "Normal":           (14.0, 1.5),
        "ALL":              (8.5,  1.0),
        "AML":              (7.5,  1.0),
        "CLL":              (11.5, 1.0),
        "CML":              (10.0, 1.5),
        "Lymphoma":         (11.5, 1.0),
        "Multiple Myeloma": (9.5,  1.0),
    },
    "Hematocrit_pct": {
        "Normal":           (43.0, 4.0),
        "ALL":              (27.0, 4.0),
        "AML":              (23.0, 4.0),
        "CLL":              (35.0, 3.0),
        "CML":              (30.0, 4.0),
        "Lymphoma":         (35.0, 3.0),
        "Multiple Myeloma": (29.0, 3.0),
    },
    "MCV_fL": {
        "Normal":           (90.0, 5.0),
        "ALL":              (85.0, 6.0),
        "AML":              (90.0, 6.0),
        "CLL":              (90.0, 5.0),
        "CML":              (88.0, 5.0),
        "Lymphoma":         (87.0, 5.0),
        "Multiple Myeloma": (87.0, 5.0),
    },
    "MCH_pg": {
        "Normal":           (30.0, 2.0),
        "ALL":              (28.0, 2.0),
        "AML":              (29.0, 2.0),
        "CLL":              (30.0, 2.0),
        "CML":              (29.0, 2.0),
        "Lymphoma":         (29.0, 2.0),
        "Multiple Myeloma": (29.0, 2.0),
    },
    "MCHC_g_dL": {
        "Normal":           (34.0, 1.0),
        "ALL":              (32.5, 1.0),
        "AML":              (32.0, 1.0),
        "CLL":              (33.5, 1.0),
        "CML":              (33.0, 1.0),
        "Lymphoma":         (33.0, 1.0),
        "Multiple Myeloma": (33.0, 1.0),
    },
    "Neutrophils_pct": {
        "Normal":           (60.0,  7.0),
        "ALL":              (20.0,  8.0),
        "AML":              (12.0,  6.0),
        "CLL":              (30.0,  8.0),
        "CML":              (45.0, 10.0),
        "Lymphoma":         (52.0,  8.0),
        "Multiple Myeloma": (52.0,  8.0),
    },
    "Lymphocytes_pct": {
        "Normal":           (30.0,  5.0),
        "ALL":              (75.0, 10.0),
        "AML":              (20.0,  7.0),
        "CLL":              (75.0,  8.0),
        "CML":              (25.0,  7.0),
        "Lymphoma":         (40.0,  8.0),
        "Multiple Myeloma": (30.0,  6.0),
    },
    "Monocytes_pct": {
        "Normal":           (5.0,  1.5),
        "ALL":              (4.0,  1.5),
        "AML":              (10.0, 3.0),
        "CLL":              (4.0,  1.5),
        "CML":              (5.0,  2.0),
        "Lymphoma":         (6.0,  2.0),
        "Multiple Myeloma": (5.0,  1.5),
    },
    "Eosinophils_pct": {
        "Normal":           (2.5, 1.0),
        "ALL":              (2.0, 1.0),
        "AML":              (2.0, 1.0),
        "CLL":              (2.0, 1.0),
        "CML":              (3.0, 1.5),
        "Lymphoma":         (4.0, 2.0),
        "Multiple Myeloma": (2.0, 1.0),
    },
    "Basophils_pct": {
        "Normal":           (0.5, 0.2),
        "ALL":              (0.5, 0.2),
        "AML":              (0.8, 0.3),
        "CLL":              (0.5, 0.2),
        "CML":              (2.0, 0.8),
        "Lymphoma":         (0.5, 0.2),
        "Multiple Myeloma": (0.5, 0.2),
    },
    "Blasts_pct": {
        "Normal":           (0.2,  0.2),
        "ALL":              (55.0, 20.0),
        "AML":              (55.0, 20.0),
        "CLL":              (3.0,  2.0),
        "CML":              (5.0,  3.0),
        "Lymphoma":         (2.0,  1.5),
        "Multiple Myeloma": (5.0,  3.0),
    },
}

CLIP_BOUNDS: dict[str, tuple[float, float]] = {
    "RBC_M_per_uL":    (0.5,  8.0),
    "Hemoglobin_g_dL": (2.0, 20.0),
    "Hematocrit_pct":  (5.0, 60.0),
    "MCV_fL":          (60.0, 120.0),
    "MCH_pg":          (15.0,  45.0),
    "MCHC_g_dL":       (25.0,  40.0),
    "Neutrophils_pct": (0.0,   95.0),
    "Lymphocytes_pct": (0.0,   95.0),
    "Monocytes_pct":   (0.0,   30.0),
    "Eosinophils_pct": (0.0,   20.0),
    "Basophils_pct":   (0.0,   10.0),
    "Blasts_pct":      (0.0,   99.0),
}

# ---------------------------------------------------------------------------
# Generate synthetic CBC columns
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)

for col, class_params in CBC_PARAMS.items():
    values = np.zeros(len(df), dtype=float)
    for cls, (mean, std) in class_params.items():
        mask = (df["label"] == cls).values
        if mask.any():
            values[mask] = rng.normal(mean, std, mask.sum())
    lo, hi = CLIP_BOUNDS[col]
    df[col] = np.round(np.clip(values, lo, hi), 2)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
df.to_csv(out_path, index=False)

print()
print("=" * 60)
print(f"Hybrid dataset saved -> {out_path.name}")
print(f"Shape  : {df.shape[0]} rows × {df.shape[1]} columns")
print()
print("Final label distribution:")
print(df["label"].value_counts().to_string())
print()
print("Columns in output:")
for c in df.columns:
    print(f"  {c}")
print("=" * 60)
