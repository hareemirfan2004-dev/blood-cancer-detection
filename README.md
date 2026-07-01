# Big Data-Driven Predictive Analytics for Early Blood Cancer Detection

A supervised multi-class classification project for the Big Data Essentials course.
Classifies patients into: **Normal / ALL / AML / CLL / CML** from clinical CBC data.

---

## Dataset

This project uses a **hybrid dataset**:

- **Real patient records** sourced from:
  > Md. Aminul Islam et al. "Comprehensive Hematological and Clinical Profiles Related to Blood Cancer in Bangladesh."
  > Mendeley Data, v2. https://data.mendeley.com/datasets/d3vs7sbpt2/2 — CC BY 4.0.
  > (2,296 patient rows; columns include Age, Gender, Cancer_Type, Total WBC count, Platelet Count, Diagnosis_Result, and others.)

- **Synthetic augmentation** — the following 12 columns were **not present** in the original dataset
  and were **generated synthetically** by sampling Gaussian distributions whose parameters are
  derived from published clinical reference ranges (Wintrobe's Clinical Hematology 13th ed.;
  Harrison's Principles 21st ed.; WHO Classification of Haematopoietic Tumours):

  `RBC_M_per_uL`, `Hemoglobin_g_dL`, `Hematocrit_pct`, `MCV_fL`, `MCH_pg`, `MCHC_g_dL`,
  `Neutrophils_pct`, `Lymphocytes_pct`, `Monocytes_pct`, `Eosinophils_pct`, `Basophils_pct`,
  `Blasts_pct`

  Each synthetic value was sampled conditioned on the patient's mapped class label so that
  distributions reflect known haematological patterns for each leukaemia subtype.

> **Caveat for graders / readers:**
> The augmented CBC columns are synthetic. Any model performance reported on those features
> should be interpreted as a demonstration of the pipeline methodology, not as a clinical
> validation. The real Mendeley columns (WBC, Platelets, demographics, genetic markers) are
> unmodified and carry their original provenance.

---

## Project Structure

```
blood_cancer_detection/
├── data/                    # raw + hybrid CSVs (gitignored)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_spark_pipeline.ipynb
├── models/                  # saved model files (gitignored)
├── outputs/                 # charts and figures
├── build_hybrid_dataset.py  # data augmentation script
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt

# 1. Download the Mendeley CSV and place it in data/
# 2. Build the hybrid dataset:
python build_hybrid_dataset.py

# 3. Launch Jupyter:
jupyter notebook
```

## Pipeline Phases

| Phase | Notebook / File | Description |
|-------|-----------------|-------------|
| 1 | `01_eda.ipynb` | Exploratory data analysis |
| 2 | `02_preprocessing.ipynb` | Cleaning, encoding, scaling, train/test split |
| 3 | `03_modeling.ipynb` | Train 5 classifiers with cross-validation |
| 4 | `04_evaluation.ipynb` | Metrics, confusion matrix, required course visuals |
| 5 | `05_spark_pipeline.ipynb` | PySpark/MLlib replication of the pipeline |
| 6 | `src/app.py` | Streamlit prediction app (Phase 7) |

## Streamlit Prediction App (Phase 7)

Runs a local web UI that accepts CBC inputs and returns the predicted cancer class plus a per-class probability bar chart.

**Requirements:** models must already be trained and saved to `models/` (run notebooks 1–4 first).

```bash
# From the blood_cancer_detection/ directory, using the Python 3.11 venv:
C:\Users\<your-username>\Desktop\<your-project-folder>\venv311\Scripts\python.exe -m streamlit run src/app.py
Replace <your-username> with your Windows account name and <your-project-folder> with the folder where you cloned this project.
```  
All 21 input fields are pre-filled with Normal-class CBC reference values — click **Predict** without changing anything to verify the pipeline loads correctly.
