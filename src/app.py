"""
Blood Cancer Detection — Streamlit Prediction App (Phase 7)
Academic project. NOT a diagnostic tool.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so the app works from any cwd
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).parent.parent / "models"

# ---------------------------------------------------------------------------
# Artifact loading (cached so reload only happens once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    model        = joblib.load(MODELS_DIR / "best_model.pkl")
    le           = joblib.load(MODELS_DIR / "label_encoder.pkl")

    # Recover column lists directly from the fitted ColumnTransformer
    # so we never have to hardcode them here.
    num_cols = list(preprocessor.transformers_[0][2])
    cat_cols = list(preprocessor.transformers_[1][2])

    ohe         = preprocessor.named_transformers_["cat"]["encoder"]
    cat_options = {
        col: list(cats)
        for col, cats in zip(cat_cols, ohe.categories_)
    }

    return preprocessor, model, le, num_cols, cat_cols, cat_options

preprocessor, model, le, NUMERIC_COLS, CATEGORICAL_COLS, CAT_OPTIONS = load_artifacts()

# ---------------------------------------------------------------------------
# Display labels (strip parenthetical suffixes for readability)
# ---------------------------------------------------------------------------
LABELS = {
    "Age":                                                          "Age (years)",
    "Total WBC count(/cumm)":                                      "Total WBC (/cumm)",
    "Platelet Count( (/cumm)":                                     "Platelet Count (/cumm)",
    "RBC_M_per_uL":                                                "RBC (M/µL)",
    "Hemoglobin_g_dL":                                             "Hemoglobin (g/dL)",
    "Hematocrit_pct":                                              "Hematocrit (%)",
    "MCV_fL":                                                      "MCV (fL)",
    "MCH_pg":                                                      "MCH (pg)",
    "MCHC_g_dL":                                                   "MCHC (g/dL)",
    "Neutrophils_pct":                                             "Neutrophils (%)",
    "Lymphocytes_pct":                                             "Lymphocytes (%)",
    "Monocytes_pct":                                               "Monocytes (%)",
    "Eosinophils_pct":                                             "Eosinophils (%)",
    "Basophils_pct":                                               "Basophils (%)",
    "Blasts_pct":                                                  "Blasts (%)",
    "Gender":                                                      "Gender",
    "Bone Marrow Aspiration(Positive / Negative / Not Done)":      "Bone Marrow Aspiration",
    "Serum Protein Electrophoresis (SPEP)(Normal / Abnormal)":     "SPEP Result",
    "Lymph Node Biopsy(Positive / Negative / Not Done)":           "Lymph Node Biopsy",
    "Lumbar Puncture (Spinal Tap)":                                "Lumbar Puncture",
    "Genetic_Data(BCR-ABL, FLT3)":                                "Genetic Marker",
}

# Normal-class CBC reference defaults (Wintrobe's / Harrison's ranges)
NUMERIC_DEFAULTS = {
    "Age":                      35.0,
    "Total WBC count(/cumm)":   7000.0,
    "Platelet Count( (/cumm)":  250000.0,
    "RBC_M_per_uL":             5.0,
    "Hemoglobin_g_dL":          14.0,
    "Hematocrit_pct":           42.0,
    "MCV_fL":                   90.0,
    "MCH_pg":                   30.0,
    "MCHC_g_dL":                33.0,
    "Neutrophils_pct":          60.0,
    "Lymphocytes_pct":          30.0,
    "Monocytes_pct":            6.0,
    "Eosinophils_pct":          2.0,
    "Basophils_pct":            0.5,
    "Blasts_pct":               0.0,
}

NUMERIC_RANGES = {
    "Age":                      (0.0,    120.0,  1.0),
    "Total WBC count(/cumm)":   (100.0,  100000.0, 100.0),
    "Platelet Count( (/cumm)":  (10000.0, 1000000.0, 1000.0),
    "RBC_M_per_uL":             (0.5,    12.0,   0.1),
    "Hemoglobin_g_dL":          (2.0,    25.0,   0.1),
    "Hematocrit_pct":           (5.0,    70.0,   0.1),
    "MCV_fL":                   (50.0,   130.0,  0.1),
    "MCH_pg":                   (10.0,   50.0,   0.1),
    "MCHC_g_dL":                (20.0,   40.0,   0.1),
    "Neutrophils_pct":          (0.0,    100.0,  0.1),
    "Lymphocytes_pct":          (0.0,    100.0,  0.1),
    "Monocytes_pct":            (0.0,    50.0,   0.1),
    "Eosinophils_pct":          (0.0,    50.0,   0.1),
    "Basophils_pct":            (0.0,    20.0,   0.1),
    "Blasts_pct":               (0.0,    100.0,  0.1),
}

# Sensible defaults for a "Normal" patient
CAT_DEFAULTS = {
    "Gender":                                                      "Male",
    "Bone Marrow Aspiration(Positive / Negative / Not Done)":      "Not Done",
    "Serum Protein Electrophoresis (SPEP)(Normal / Abnormal)":     "Normal",
    "Lymph Node Biopsy(Positive / Negative / Not Done)":           "Not Done",
    "Lumbar Puncture (Spinal Tap)":                                "Not Done",
    "Genetic_Data(BCR-ABL, FLT3)":                                "TP53",
}

# Colour per predicted class
CLASS_COLOURS = {
    "Normal":           "#2ecc71",
    "ALL":              "#e74c3c",
    "AML":              "#c0392b",
    "CLL":              "#e67e22",
    "CML":              "#d35400",
    "Lymphoma":         "#9b59b6",
    "Multiple Myeloma": "#8e44ad",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Blood Cancer Detection",
    page_icon="🩸",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------
st.warning(
    "⚠️ **ACADEMIC PROJECT — NOT A DIAGNOSTIC TOOL.**  "
    "This app was built for a Big Data Essentials course. "
    "The model was trained partly on **synthetic CBC data** (12 of 15 numeric features "
    "were generated from published reference ranges, not real patient measurements). "
    "Predictions have no clinical validity and must **never** be used for medical decisions.",
    icon="🚨",
)

st.title("🩸 Blood Cancer Detection")
st.caption(
    "Predicts one of 7 classes — Normal / ALL / AML / CLL / CML / Lymphoma / Multiple Myeloma — "
    "from CBC values and diagnostic markers, using a scikit-learn Random Forest "
    f"(31 features after OHE, trained on {len(le.classes_)} classes)."
)

st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("cbc_form"):
    st.subheader("CBC & Diagnostic Inputs")
    st.caption("Pre-filled with typical Normal-class reference values. Adjust as needed.")

    col_num, col_cat = st.columns([3, 2], gap="large")

    numeric_values: dict = {}
    with col_num:
        st.markdown("**Haematology (CBC)**")
        # Two sub-columns inside numeric
        sub1, sub2 = st.columns(2)
        for i, col in enumerate(NUMERIC_COLS):
            mn, mx, step = NUMERIC_RANGES[col]
            target_col = sub1 if i % 2 == 0 else sub2
            numeric_values[col] = target_col.number_input(
                label=LABELS[col],
                min_value=mn,
                max_value=mx,
                value=NUMERIC_DEFAULTS[col],
                step=step,
                format="%.1f",
            )

    cat_values: dict = {}
    with col_cat:
        st.markdown("**Diagnostic Markers**")
        for col in CATEGORICAL_COLS:
            options = CAT_OPTIONS[col]
            default_idx = options.index(CAT_DEFAULTS[col]) if CAT_DEFAULTS[col] in options else 0
            cat_values[col] = st.selectbox(
                label=LABELS[col],
                options=options,
                index=default_idx,
            )

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True, type="primary")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
if submitted:
    # Build a single-row DataFrame with the exact column order the preprocessor expects
    row = {col: [numeric_values[col]] for col in NUMERIC_COLS}
    row.update({col: [cat_values[col]] for col in CATEGORICAL_COLS})
    df_input = pd.DataFrame(row)

    X_processed = preprocessor.transform(df_input)
    pred_idx    = int(model.predict(X_processed)[0])
    pred_class  = le.classes_[pred_idx]
    proba       = model.predict_proba(X_processed)[0]  # shape (n_classes,)

    st.divider()
    st.subheader("Prediction Result")

    colour = CLASS_COLOURS.get(pred_class, "#3498db")
    st.markdown(
        f"<div style='background:{colour}22; border-left:6px solid {colour}; "
        f"padding:16px 20px; border-radius:6px; margin-bottom:12px;'>"
        f"<span style='font-size:1.1rem; color:{colour}; font-weight:700;'>Predicted class</span><br>"
        f"<span style='font-size:2.2rem; font-weight:800; color:{colour};'>{pred_class}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Per-class probability bar chart (Altair)
    prob_df = pd.DataFrame({
        "Class":       le.classes_,
        "Probability": proba,
    }).sort_values("Probability", ascending=False)

    chart = (
        alt.Chart(prob_df)
        .mark_bar()
        .encode(
            x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%", title="Probability")),
            y=alt.Y("Class:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.Class == pred_class,
                alt.value(colour),
                alt.value("#bdc3c7"),
            ),
            tooltip=[
                alt.Tooltip("Class:N"),
                alt.Tooltip("Probability:Q", format=".2%"),
            ],
        )
        .properties(height=220, title="Per-class probabilities")
    )
    st.altair_chart(chart, use_container_width=True)

    # Raw probability table
    with st.expander("Full probability table"):
        prob_display = prob_df.copy()
        prob_display["Probability"] = prob_display["Probability"].map("{:.4f}".format)
        st.dataframe(prob_display.reset_index(drop=True), use_container_width=True)

    st.caption(
        "Reminder: this model was trained on a hybrid dataset with synthetic CBC values. "
        "These outputs carry no clinical meaning."
    )
