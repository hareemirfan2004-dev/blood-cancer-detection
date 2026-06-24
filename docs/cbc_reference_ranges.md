# CBC Reference Ranges Used for Synthetic Augmentation

These values define the Gaussian distributions (mean ± SD) used to generate the
12 synthetic CBC columns in `blood_cancer_hybrid.csv`. Each row was sampled
independently as `clip(Normal(mean, std), lo, hi)` with `random_seed=42`.

**Sources**
1. Greer J.P. et al. *Wintrobe's Clinical Hematology*, 13th ed. Lippincott Williams & Wilkins, 2014.
2. Loscalzo J. et al. *Harrison's Principles of Internal Medicine*, 21st ed. McGraw-Hill, 2022.
3. Swerdlow S.H. et al. *WHO Classification of Tumours of Haematopoietic and Lymphoid Tissues*, 5th ed. IARC Press, 2022.

---

## Normal (healthy reference)

| Column | Mean | SD | Physiological clip |
|---|---|---|---|
| RBC (M/µL) | 5.00 | 0.40 | [0.5, 8.0] |
| Hemoglobin (g/dL) | 14.0 | 1.5 | [2.0, 20.0] |
| Hematocrit (%) | 43.0 | 4.0 | [5.0, 60.0] |
| MCV (fL) | 90.0 | 5.0 | [60.0, 120.0] |
| MCH (pg) | 30.0 | 2.0 | [15.0, 45.0] |
| MCHC (g/dL) | 34.0 | 1.0 | [25.0, 40.0] |
| Neutrophils (%) | 60.0 | 7.0 | [0.0, 95.0] |
| Lymphocytes (%) | 30.0 | 5.0 | [0.0, 95.0] |
| Monocytes (%) | 5.0 | 1.5 | [0.0, 30.0] |
| Eosinophils (%) | 2.5 | 1.0 | [0.0, 20.0] |
| Basophils (%) | 0.5 | 0.2 | [0.0, 10.0] |
| Blasts (%) | 0.2 | 0.2 | [0.0, 99.0] |

*Source: Wintrobe's Table 2-1 (adult reference intervals).*

---

## ALL — Acute Lymphoblastic Leukemia

| Column | Mean | SD | Clinical rationale |
|---|---|---|---|
| RBC (M/µL) | 3.00 | 0.40 | Bone marrow replacement → normocytic anemia |
| Hemoglobin (g/dL) | 8.5 | 1.0 | Anaemia at presentation in ~80% of cases |
| Hematocrit (%) | 27.0 | 4.0 | Mirrors haemoglobin decline |
| MCV (fL) | 85.0 | 6.0 | Normocytic; slight variation |
| MCH (pg) | 28.0 | 2.0 | Slightly reduced |
| MCHC (g/dL) | 32.5 | 1.0 | Low-normal |
| Neutrophils (%) | 20.0 | 8.0 | Neutropenia from marrow failure |
| Lymphocytes (%) | 75.0 | 10.0 | Lymphoblast predominance |
| Monocytes (%) | 4.0 | 1.5 | Suppressed |
| Eosinophils (%) | 2.0 | 1.0 | Normal |
| Basophils (%) | 0.5 | 0.2 | Normal |
| Blasts (%) | 55.0 | 20.0 | Hallmark: circulating lymphoblasts |

*Source: Harrison's Ch. 108; WHO 5th ed. p. 245.*

---

## AML — Acute Myeloid Leukemia

| Column | Mean | SD | Clinical rationale |
|---|---|---|---|
| RBC (M/µL) | 2.50 | 0.40 | Severe anaemia; marrow replaced by blasts |
| Hemoglobin (g/dL) | 7.5 | 1.0 | Often <8 g/dL at diagnosis |
| Hematocrit (%) | 23.0 | 4.0 | Reflects severe anaemia |
| MCV (fL) | 90.0 | 6.0 | Normocytic |
| MCH (pg) | 29.0 | 2.0 | Normal |
| MCHC (g/dL) | 32.0 | 1.0 | Low-normal |
| Neutrophils (%) | 12.0 | 6.0 | Profound neutropenia |
| Lymphocytes (%) | 20.0 | 7.0 | Reduced |
| Monocytes (%) | 10.0 | 3.0 | Elevated in M4/M5 subtypes |
| Eosinophils (%) | 2.0 | 1.0 | Normal |
| Basophils (%) | 0.8 | 0.3 | Slightly elevated |
| Blasts (%) | 55.0 | 20.0 | ≥20% blasts required for AML diagnosis (WHO) |

*Source: Harrison's Ch. 107; WHO 5th ed. p. 139.*

---

## CLL — Chronic Lymphocytic Leukemia

| Column | Mean | SD | Clinical rationale |
|---|---|---|---|
| RBC (M/µL) | 4.00 | 0.30 | Mild anaemia due to autoimmune haemolysis |
| Hemoglobin (g/dL) | 11.5 | 1.0 | Mild anaemia, often Coombs-positive |
| Hematocrit (%) | 35.0 | 3.0 | Mild decrease |
| MCV (fL) | 90.0 | 5.0 | Normocytic |
| MCH (pg) | 30.0 | 2.0 | Normal |
| MCHC (g/dL) | 33.5 | 1.0 | Normal |
| Neutrophils (%) | 30.0 | 8.0 | Relatively decreased (lymphocytes dominate) |
| Lymphocytes (%) | 75.0 | 8.0 | Absolute lymphocytosis — defining feature |
| Monocytes (%) | 4.0 | 1.5 | Normal |
| Eosinophils (%) | 2.0 | 1.0 | Normal |
| Basophils (%) | 0.5 | 0.2 | Normal |
| Blasts (%) | 3.0 | 2.0 | Low; rise indicates Richter transformation |

*Source: Wintrobe's Ch. 79; WHO 5th ed. p. 265.*

---

## CML — Chronic Myeloid Leukemia

| Column | Mean | SD | Clinical rationale |
|---|---|---|---|
| RBC (M/µL) | 3.50 | 0.40 | Mild anaemia |
| Hemoglobin (g/dL) | 10.0 | 1.5 | Normocytic anaemia common |
| Hematocrit (%) | 30.0 | 4.0 | Mild decrease |
| MCV (fL) | 88.0 | 5.0 | Normocytic |
| MCH (pg) | 29.0 | 2.0 | Normal |
| MCHC (g/dL) | 33.0 | 1.0 | Normal |
| Neutrophils (%) | 45.0 | 10.0 | Absolute neutrophilia; left shift (bands/myelocytes) |
| Lymphocytes (%) | 25.0 | 7.0 | Relatively decreased |
| Monocytes (%) | 5.0 | 2.0 | Normal to slightly elevated |
| Eosinophils (%) | 3.0 | 1.5 | Elevated — characteristic of CML |
| Basophils (%) | 2.0 | 0.8 | Elevated basophilia — diagnostic hallmark |
| Blasts (%) | 5.0 | 3.0 | <10% in chronic phase; rise = accelerated phase |

*Source: Harrison's Ch. 109; WHO 5th ed. p. 175.*

---

## Lymphoma (Non-Hodgkin's / Hodgkin's — peripheral blood findings)

| Column | Mean | SD | Clinical rationale |
|---|---|---|---|
| RBC (M/µL) | 3.80 | 0.40 | Mild normocytic anaemia (anaemia of chronic disease) |
| Hemoglobin (g/dL) | 11.5 | 1.0 | Mild anaemia common at diagnosis |
| Hematocrit (%) | 35.0 | 3.0 | Matches haemoglobin decline |
| MCV (fL) | 87.0 | 5.0 | Normocytic — no megaloblastic change |
| MCH (pg) | 29.0 | 2.0 | Normal |
| MCHC (g/dL) | 33.0 | 1.0 | Normal |
| Neutrophils (%) | 52.0 | 8.0 | Normal to mildly decreased |
| Lymphocytes (%) | 40.0 | 8.0 | Mildly elevated; circulating tumour cells possible |
| Monocytes (%) | 6.0 | 2.0 | Slightly elevated |
| Eosinophils (%) | 4.0 | 2.0 | Elevated — classic Hodgkin's association (Reed-Sternberg) |
| Basophils (%) | 0.5 | 0.2 | Normal |
| Blasts (%) | 2.0 | 1.5 | Low; rise indicates high-grade transformation |

*Source: Wintrobe's Ch. 82–84; Harrison's Ch. 112–113.*

---

## Multiple Myeloma (peripheral blood findings)

| Column | Mean | SD | Clinical rationale |
|---|---|---|---|
| RBC (M/µL) | 3.00 | 0.40 | Normocytic anaemia — hallmark; rouleaux on smear |
| Hemoglobin (g/dL) | 9.5 | 1.0 | Anaemia present in ~70% at diagnosis |
| Hematocrit (%) | 29.0 | 3.0 | Matches haemoglobin |
| MCV (fL) | 87.0 | 5.0 | Normocytic normochromic |
| MCH (pg) | 29.0 | 2.0 | Normal |
| MCHC (g/dL) | 33.0 | 1.0 | Normal |
| Neutrophils (%) | 52.0 | 8.0 | Often normal; neutropenia in advanced disease |
| Lymphocytes (%) | 30.0 | 6.0 | Normal range |
| Monocytes (%) | 5.0 | 1.5 | Normal |
| Eosinophils (%) | 2.0 | 1.0 | Normal |
| Basophils (%) | 0.5 | 0.2 | Normal |
| Blasts (%) | 5.0 | 3.0 | Represents circulating plasma cells; elevated vs. normal |

*Source: Harrison's Ch. 118; WHO 5th ed. p. 321.*

---

## Clip bounds applied after sampling

| Column | Lower | Upper |
|---|---|---|
| RBC (M/µL) | 0.5 | 8.0 |
| Hemoglobin (g/dL) | 2.0 | 20.0 |
| Hematocrit (%) | 5.0 | 60.0 |
| MCV (fL) | 60.0 | 120.0 |
| MCH (pg) | 15.0 | 45.0 |
| MCHC (g/dL) | 25.0 | 40.0 |
| Neutrophils (%) | 0.0 | 95.0 |
| Lymphocytes (%) | 0.0 | 95.0 |
| Monocytes (%) | 0.0 | 30.0 |
| Eosinophils (%) | 0.0 | 20.0 |
| Basophils (%) | 0.0 | 10.0 |
| Blasts (%) | 0.0 | 99.0 |
