"""
make_preprocessing_notebook.py  —  generates 02_preprocessing.ipynb
"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.8.0"},
}

cells = []

# ------------------------------------------------------------------
# 0. Header
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "# Phase 3: Preprocessing Pipeline\n"
    "## Blood Cancer Detection — Hybrid Dataset\n\n"
    "Steps covered here:\n"
    "1. Feature selection (exclude leakage and post-diagnosis columns)\n"
    "2. Type coercion (WBC, Platelets, Age stored as str in CSV)\n"
    "3. Stratified 80/20 train/test split\n"
    "4. `ColumnTransformer`: `StandardScaler` (numeric) + `OneHotEncoder` (categorical)\n"
    "   — fit on **train only**, transform both sets\n"
    "5. **SMOTE on training set only** — test set is never resampled\n"
    "6. Save preprocessed arrays + fitted preprocessor for Phase 4\n\n"
    "> **Synthetic-data caveat:** 12 CBC columns are synthetic "
    "(see `docs/cbc_reference_ranges.md`). "
    "All other columns are real Mendeley patient records."
))

# ------------------------------------------------------------------
# 1. Imports
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""\
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

PROJECT = Path('..')
DATA    = PROJECT / 'data'
MODELS  = PROJECT / 'models'
REPORTS = PROJECT / 'reports'
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(DATA / 'blood_cancer_hybrid.csv')

# Coerce numeric columns stored as strings
df['Total WBC count(/cumm)']   = pd.to_numeric(df['Total WBC count(/cumm)'],   errors='coerce')
df['Platelet Count( (/cumm)']  = pd.to_numeric(df['Platelet Count( (/cumm)'],  errors='coerce')
df['Age']                       = pd.to_numeric(df['Age'],                       errors='coerce')

print('Loaded:', df.shape)
"""))

# ------------------------------------------------------------------
# 2. Feature selection
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "### Feature Selection\n\n"
    "**Excluded columns (label leakage or post-diagnosis):**\n\n"
    "| Column | Reason |\n"
    "|---|---|\n"
    "| `Cancer_Type(AML, ALL, CLL)` | IS the label for Confirmed rows — direct leak |\n"
    "| `Diagnosis_Result` | Used to derive `label` — direct leak |\n"
    "| `Treatment_Type` | Post-diagnosis — not available at prediction time |\n"
    "| `Treatment_Outcome` | Post-diagnosis |\n"
    "| `Side_Effects` | Post-diagnosis |\n"
    "| `Comments` | Free-text noise |\n"
    "| `label` | Target itself |"
))
cells.append(nbf.v4.new_code_cell("""\
NUMERIC_COLS = [
    'Age',
    'Total WBC count(/cumm)',
    'Platelet Count( (/cumm)',
    'RBC_M_per_uL', 'Hemoglobin_g_dL', 'Hematocrit_pct',
    'MCV_fL', 'MCH_pg', 'MCHC_g_dL',
    'Neutrophils_pct', 'Lymphocytes_pct', 'Monocytes_pct',
    'Eosinophils_pct', 'Basophils_pct', 'Blasts_pct',
]

CATEGORICAL_COLS = [
    'Gender',
    'Bone Marrow Aspiration(Positive / Negative / Not Done)',
    'Serum Protein Electrophoresis (SPEP)(Normal / Abnormal)',
    'Lymph Node Biopsy(Positive / Negative / Not Done)',
    'Lumbar Puncture (Spinal Tap)',
    'Genetic_Data(BCR-ABL, FLT3)',
]

TARGET = 'label'
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()

print(f'Features : {len(FEATURE_COLS)}  ({len(NUMERIC_COLS)} numeric, {len(CATEGORICAL_COLS)} categorical)')
print(f'Target   : {TARGET}')
print(f'Rows     : {len(X)}')
"""))

# ------------------------------------------------------------------
# 3. Stratified split
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("### Stratified 80/20 Train/Test Split"))
cells.append(nbf.v4.new_code_cell("""\
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f'Train rows : {len(X_train_raw)}')
print(f'Test rows  : {len(X_test_raw)}')

counts_train_before = pd.Series(y_train).value_counts().sort_index()
counts_test_before  = pd.Series(y_test).value_counts().sort_index()

print()
print('Train class counts (BEFORE SMOTE):')
print(counts_train_before.to_string())
print()
print('Test class counts (BEFORE SMOTE — final hold-out, never resampled):')
print(counts_test_before.to_string())
"""))

# ------------------------------------------------------------------
# 4. Preprocessor — fit on train only
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "### ColumnTransformer\n"
    "Fit **only on the training set**. "
    "The test set is transformed with the same fitted object — no data leaks."
))
cells.append(nbf.v4.new_code_cell("""\
numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])

categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipe,       NUMERIC_COLS),
    ('cat', categorical_pipe,   CATEGORICAL_COLS),
])

X_train_t = preprocessor.fit_transform(X_train_raw)   # fit + transform train
X_test_t  = preprocessor.transform(X_test_raw)        # transform only — no refit

print(f'X_train_t shape : {X_train_t.shape}')
print(f'X_test_t  shape : {X_test_t.shape}')
"""))

# ------------------------------------------------------------------
# 5. Label encoding
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""\
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc  = le.transform(y_test)

print('Label encoding:')
for i, cls in enumerate(le.classes_):
    print(f'  {i} -> {cls}')
"""))

# ------------------------------------------------------------------
# 6. SMOTE — training set only
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "### SMOTE — Applied to Training Set Only\n\n"
    "SMOTE is fit **after** the preprocessor so it operates on scaled, "
    "encoded features. The test set is never touched."
))
cells.append(nbf.v4.new_code_cell("""\
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_t, y_train_enc)

counts_train_after = pd.Series(
    le.inverse_transform(y_train_res)
).value_counts().sort_index()

counts_test_after = pd.Series(
    le.inverse_transform(y_test_enc)
).value_counts().sort_index()

print(f'Train rows after SMOTE : {len(X_train_res)}  (was {len(X_train_t)})')
print(f'Test rows  (unchanged) : {len(X_test_t)}')
print()
print('Train class counts (AFTER SMOTE):')
print(counts_train_after.to_string())
print()
print('Test class counts (AFTER SMOTE — should be IDENTICAL to before):')
print(counts_test_after.to_string())
print()
print('Test set unchanged:', counts_test_before.equals(counts_test_after))
"""))

# ------------------------------------------------------------------
# 7. Write report
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""\
report_path = REPORTS / 'phase3_preprocessing_summary.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('=== PHASE 3: PREPROCESSING SUMMARY ===\\n\\n')
    f.write(f'Feature columns : {len(FEATURE_COLS)}\\n')
    f.write(f'  Numeric        : {len(NUMERIC_COLS)}\\n')
    f.write(f'  Categorical    : {len(CATEGORICAL_COLS)}\\n')
    f.write(f'Total rows       : {len(X)}\\n')
    f.write(f'Train (raw)      : {len(X_train_raw)}\\n')
    f.write(f'Test  (raw)      : {len(X_test_raw)}\\n\\n')

    f.write('--- TRAIN class counts BEFORE SMOTE ---\\n')
    f.write(counts_train_before.to_string())
    f.write('\\n\\n--- TEST class counts BEFORE SMOTE ---\\n')
    f.write(counts_test_before.to_string())

    f.write(f'\\n\\nX_train_t shape (preprocessed, pre-SMOTE) : {X_train_t.shape}\\n')
    f.write(f'X_test_t  shape (preprocessed)             : {X_test_t.shape}\\n\\n')

    f.write(f'Train rows after SMOTE : {len(X_train_res)}\\n\\n')
    f.write('--- TRAIN class counts AFTER SMOTE ---\\n')
    f.write(counts_train_after.to_string())
    f.write('\\n\\n--- TEST class counts AFTER SMOTE (must match BEFORE) ---\\n')
    f.write(counts_test_after.to_string())
    f.write(f'\\n\\nTest set unchanged: {counts_test_before.equals(counts_test_after)}\\n')

    f.write('\\n\\n--- LABEL ENCODING ---\\n')
    for i, cls in enumerate(le.classes_):
        f.write(f'  {i} -> {cls}\\n')

print(f'Report -> {report_path}')
"""))

# ------------------------------------------------------------------
# 8. Save artifacts
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("### Save Preprocessed Artifacts for Phase 4"))
cells.append(nbf.v4.new_code_cell("""\
np.save(MODELS / 'X_train.npy', X_train_res)
np.save(MODELS / 'y_train.npy', y_train_res)
np.save(MODELS / 'X_test.npy',  X_test_t)
np.save(MODELS / 'y_test.npy',  y_test_enc)

joblib.dump(preprocessor, MODELS / 'preprocessor.pkl')
joblib.dump(le,           MODELS / 'label_encoder.pkl')

cat_feature_names = (
    preprocessor.named_transformers_['cat']['encoder']
    .get_feature_names_out(CATEGORICAL_COLS)
    .tolist()
)
all_feature_names = NUMERIC_COLS + cat_feature_names
joblib.dump(all_feature_names, MODELS / 'feature_names.pkl')

print('Saved:')
print('  models/X_train.npy        ', X_train_res.shape)
print('  models/y_train.npy        ', y_train_res.shape)
print('  models/X_test.npy         ', X_test_t.shape)
print('  models/y_test.npy         ', y_test_enc.shape)
print('  models/preprocessor.pkl')
print('  models/label_encoder.pkl')
print(f'  models/feature_names.pkl  ({len(all_feature_names)} features after encoding)')
"""))

# ------------------------------------------------------------------
# 9. Summary markdown
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "## Phase 3 Summary\n\n"
    "| Step | Detail |\n"
    "|---|---|\n"
    "| Split | 80/20 stratified, `random_state=42` |\n"
    "| Preprocessor | `ColumnTransformer` fit on train only |\n"
    "| Numeric | `SimpleImputer(median)` → `StandardScaler` |\n"
    "| Categorical | `SimpleImputer(most_frequent)` → `OneHotEncoder(handle_unknown='ignore')` |\n"
    "| Resampling | `SMOTE(random_state=42)` on **train only** — test set untouched |\n"
    "| Artifacts | Saved to `models/` — load in Phase 4 |\n\n"
    "See `reports/phase3_preprocessing_summary.txt` for full class counts."
))

nb.cells = cells

out_nb = Path(__file__).parent / '02_preprocessing.ipynb'
with open(out_nb, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Notebook written -> {out_nb}')
