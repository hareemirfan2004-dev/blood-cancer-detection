"""
make_spark_notebook.py  —  generates 05_spark_pipeline.ipynb
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
    "# Phase 6: PySpark + MLlib Pipeline\n"
    "## Blood Cancer Detection — Hybrid Dataset\n\n"
    "This notebook reproduces the classification pipeline using **Apache Spark MLlib**, "
    "running in local mode. Each step is annotated with its production equivalent on a "
    "HDFS + YARN + Spark cluster.\n\n"
    "**Pipeline stages:**\n"
    "1. Load hybrid CSV into a Spark DataFrame\n"
    "2. `StringIndexer` — encode categorical features and label\n"
    "3. `Imputer` — handle missing numeric values\n"
    "4. `VectorAssembler` — combine all features into a dense vector\n"
    "5. `StandardScaler` — zero-mean, unit-variance scaling\n"
    "6. `RandomForestClassifier` (MLlib)\n"
    "7. `MulticlassClassificationEvaluator` — accuracy and F1\n\n"
    "> **Synthetic-data caveat:** 12 of 15 numeric features are synthetic. "
    "See `docs/LIMITATIONS.md`."
))

# ------------------------------------------------------------------
# 1. SparkSession
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("### 1. Create SparkSession"))
cells.append(nbf.v4.new_code_cell("""\
import sys, os
from pathlib import Path

# Ensure PySpark uses the same Python interpreter as this notebook
os.environ['PYSPARK_PYTHON']        = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession

# LOCAL MODE — runs on all available CPU cores.
# PRODUCTION EQUIVALENT:
#   master("yarn") + spark.hadoop.fs.defaultFS = hdfs://namenode:9000
#   Resources allocated via YARN ResourceManager; executors launched on NodeManagers.
spark = (
    SparkSession.builder
    .appName("BloodCancerDetection")
    .master("local[*]")
    # Production: .master("yarn")
    # Production: .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    .config("spark.sql.shuffle.partitions", "8")   # keep low for local runs
    .config("spark.driver.memory", "1g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print(f"Spark version : {spark.version}")
print(f"Master        : {spark.sparkContext.master}")
print(f"App name      : {spark.sparkContext.appName}")
"""))

# ------------------------------------------------------------------
# 2. Load CSV into Spark DataFrame
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("### 2. Load Data into Spark DataFrame"))
cells.append(nbf.v4.new_code_cell("""\
PROJECT  = Path('..')
CSV_PATH = str(PROJECT / 'data' / 'blood_cancer_hybrid.csv')

# LOCAL MODE — reads from local filesystem.
# PRODUCTION EQUIVALENT:
#   CSV_PATH = "hdfs://namenode:9000/user/hadoop/blood_cancer/hybrid.csv"
#   Data would be ingested from S3/Kafka via Spark Structured Streaming or
#   batch load, stored on HDFS, and partitioned across DataNodes.
df_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(CSV_PATH)
)

print(f"Rows    : {df_raw.count()}")
print(f"Columns : {len(df_raw.columns)}")
df_raw.printSchema()
"""))

# ------------------------------------------------------------------
# 3. Rename & clean columns
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "### 3. Rename Columns & Cast Types\n\n"
    "Column names with parentheses and slashes must be sanitised "
    "before passing to MLlib transformers."
))
cells.append(nbf.v4.new_code_cell("""\
from pyspark.sql import functions as F

RENAME = {
    "Cancer_Type(AML, ALL, CLL)":                            "cancer_type",
    "Treatment_Type(Chemotherapy, Radiation)":               "treatment_type",
    "Bone Marrow Aspiration(Positive / Negative / Not Done)":"bone_marrow_aspiration",
    "Serum Protein Electrophoresis (SPEP)(Normal / Abnormal)":"spep",
    "Lymph Node Biopsy(Positive / Negative / Not Done)":     "lymph_node_biopsy",
    "Lumbar Puncture (Spinal Tap)":                          "lumbar_puncture",
    "Total WBC count(/cumm)":                                "wbc_count",
    "Platelet Count( (/cumm)":                               "platelet_count",
    "Genetic_Data(BCR-ABL, FLT3)":                           "genetic_data",
}

df = df_raw
for old, new in RENAME.items():
    df = df.withColumnRenamed(old, new)

# Cast numeric columns (inferSchema may read some as string)
NUMERIC_COLS = [
    "Age", "wbc_count", "platelet_count",
    "RBC_M_per_uL", "Hemoglobin_g_dL", "Hematocrit_pct",
    "MCV_fL", "MCH_pg", "MCHC_g_dL",
    "Neutrophils_pct", "Lymphocytes_pct", "Monocytes_pct",
    "Eosinophils_pct", "Basophils_pct", "Blasts_pct",
]
for c in NUMERIC_COLS:
    df = df.withColumn(c, F.col(c).cast("double"))

# Drop post-diagnosis / leakage columns (same exclusion list as scikit-learn Phase 3)
DROP_COLS = [
    "cancer_type", "Diagnosis_Result", "treatment_type",
    "Treatment_Outcome", "Side_Effects", "Comments",
]
df = df.drop(*DROP_COLS)

CATEGORICAL_COLS = [
    "Gender", "bone_marrow_aspiration", "spep",
    "lymph_node_biopsy", "lumbar_puncture", "genetic_data",
]

print("Retained columns:")
for c in df.columns:
    print(f"  {c}")
print(f"\\nTotal: {len(df.columns)} columns, {df.count()} rows")
"""))

# ------------------------------------------------------------------
# 4. Train / test split
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "### 4. Train / Test Split\n\n"
    "`randomSplit` is not stratified — Spark MLlib has no built-in stratified split. "
    "For a production pipeline, stratification can be achieved by splitting per-class "
    "and unioning the results, or handled upstream in the data preparation job."
))
cells.append(nbf.v4.new_code_cell("""\
# PRODUCTION EQUIVALENT:
#   Data would be pre-partitioned on HDFS into train/test directories,
#   written by a separate Spark data-preparation job and versioned in a
#   data catalogue (e.g. Apache Atlas or AWS Glue).
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print(f"Train rows : {train_df.count()}")
print(f"Test  rows : {test_df.count()}")
print()
print("Label distribution in test split:")
test_df.groupBy("label").count().orderBy("count", ascending=False).show()
"""))

# ------------------------------------------------------------------
# 5. Build MLlib Pipeline
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "### 5. Build MLlib Pipeline\n\n"
    "Stages: `StringIndexer` (×6 categorical + label) → `Imputer` → "
    "`VectorAssembler` → `StandardScaler` → `RandomForestClassifier`\n\n"
    "**Production note:** In a cluster deployment this `Pipeline` object would be "
    "serialised and submitted via `spark-submit` to YARN. The fitted `PipelineModel` "
    "would be written to HDFS (`model.write().overwrite().save('hdfs://...')`) and "
    "served by a Spark Structured Streaming job or a REST API backed by MLflow."
))
cells.append(nbf.v4.new_code_cell("""\
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, Imputer, VectorAssembler, StandardScaler
)
from pyspark.ml.classification import RandomForestClassifier

# --- Stage 1: StringIndexer for each categorical feature ---
# PRODUCTION: these mappings would be learned on the full training partition
# stored on HDFS and reused for online inference.
cat_indexers = [
    StringIndexer(
        inputCol=c, outputCol=f"{c}_idx",
        handleInvalid="keep",   # unseen categories → extra index bucket
    )
    for c in CATEGORICAL_COLS
]

# --- Stage 2: StringIndexer for the target label ---
label_indexer = StringIndexer(
    inputCol="label", outputCol="label_idx", handleInvalid="keep"
)

# --- Stage 3: Imputer for numeric columns (median strategy) ---
# PRODUCTION: median statistics computed over the full HDFS training dataset.
imputed_cols = [f"{c}_imp" for c in NUMERIC_COLS]
imputer = Imputer(
    inputCols=NUMERIC_COLS,
    outputCols=imputed_cols,
    strategy="median",
)

# --- Stage 4: VectorAssembler — merge all features into one vector ---
# PRODUCTION: this dense vector is the unit of data exchange between
# Spark executors during shuffle operations across the cluster.
feature_cols = [f"{c}_idx" for c in CATEGORICAL_COLS] + imputed_cols
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features_raw",
    handleInvalid="keep",
)

# --- Stage 5: StandardScaler ---
# PRODUCTION: mean and std computed over the distributed training RDD;
# broadcast to all executors for the transform step.
scaler = StandardScaler(
    inputCol="features_raw", outputCol="features",
    withMean=True, withStd=True,
)

# --- Stage 6: RandomForestClassifier ---
# PRODUCTION: tree construction is parallelised across executors via
# YARN; each node handles a subset of the data partition on its local
# HDFS DataNode block (data locality). numTrees controls parallelism.
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label_idx",
    numTrees=100,
    maxDepth=10,
    seed=42,
)

pipeline = Pipeline(
    stages=cat_indexers + [label_indexer, imputer, assembler, scaler, rf]
)

print("Pipeline stages:")
for i, s in enumerate(pipeline.getStages()):
    print(f"  {i:>2}. {type(s).__name__}")
"""))

# ------------------------------------------------------------------
# 6. Fit pipeline
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("### 6. Fit Pipeline on Training Data"))
cells.append(nbf.v4.new_code_cell("""\
import time

# PRODUCTION EQUIVALENT:
#   spark-submit --master yarn --deploy-mode cluster pipeline_job.py
#   The Driver coordinates the DAG; tasks execute on YARN containers.
#   Intermediate RDDs are cached on executor memory (or spill to HDFS).
t0 = time.time()
model = pipeline.fit(train_df)
elapsed = time.time() - t0

print(f"Pipeline fitted in {elapsed:.1f}s")
"""))

# ------------------------------------------------------------------
# 7. Predict & evaluate
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell("### 7. Predict & Evaluate on Test Set"))
cells.append(nbf.v4.new_code_cell("""\
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

predictions = model.transform(test_df)

# PRODUCTION EQUIVALENT:
#   Predictions written back to HDFS as Parquet, then read by
#   a reporting job (e.g. Spark SQL → Hive → QuickSight / Tableau).
predictions.select("label", "label_idx", "prediction").show(10)

evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label_idx", predictionCol="prediction", metricName="accuracy"
)
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label_idx", predictionCol="prediction", metricName="f1"
)
evaluator_wf1 = MulticlassClassificationEvaluator(
    labelCol="label_idx", predictionCol="prediction", metricName="weightedFMeasure"
)

spark_accuracy = evaluator_acc.evaluate(predictions)
spark_f1_macro = evaluator_f1.evaluate(predictions)
spark_f1_wtd   = evaluator_wf1.evaluate(predictions)

print(f"Spark MLlib RandomForest — test-set results")
print(f"  Accuracy         : {spark_accuracy:.4f}")
print(f"  F1 (macro)       : {spark_f1_macro:.4f}")
print(f"  F1 (weighted)    : {spark_f1_wtd:.4f}")
"""))

# ------------------------------------------------------------------
# 8. Per-class breakdown via Spark SQL
# ------------------------------------------------------------------
cells.append(nbf.v4.new_markdown_cell(
    "### 8. Per-Class Breakdown via Spark SQL\n\n"
    "PRODUCTION EQUIVALENT: this query would run as a Hive/Spark SQL job "
    "on the predictions Parquet file stored in HDFS, surfaced in a BI tool."
))
cells.append(nbf.v4.new_code_cell("""\
predictions.createOrReplaceTempView("predictions_view")

per_class = spark.sql(\"\"\"
    SELECT
        label,
        COUNT(*)                                                        AS support,
        SUM(CASE WHEN label_idx = prediction THEN 1 ELSE 0 END)        AS correct,
        ROUND(
            SUM(CASE WHEN label_idx = prediction THEN 1 ELSE 0 END)
            / COUNT(*), 4
        )                                                               AS per_class_accuracy
    FROM predictions_view
    GROUP BY label
    ORDER BY support DESC
\"\"\")
per_class.show()
"""))

# ------------------------------------------------------------------
# 9. Write results
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""\
REPORTS = PROJECT / 'reports'
result_path = REPORTS / 'spark_results.txt'

per_class_rows = per_class.collect()

with open(result_path, 'w', encoding='utf-8') as f:
    f.write('=== PHASE 6: SPARK MLLIB RESULTS ===\\n\\n')
    f.write(f'Mode            : local[*] (all CPU cores)\\n')
    f.write(f'Spark version   : {spark.version}\\n')
    f.write(f'Pipeline stages : StringIndexer x6 + LabelIndexer + Imputer '
            f'+ VectorAssembler + StandardScaler + RandomForestClassifier\\n')
    f.write(f'numTrees        : 100   maxDepth : 10   seed : 42\\n\\n')
    f.write(f'Train rows : {train_df.count()}\\n')
    f.write(f'Test  rows : {test_df.count()}\\n\\n')
    f.write(f'--- OVERALL METRICS ---\\n')
    f.write(f'Accuracy      : {spark_accuracy:.4f}\\n')
    f.write(f'F1 (macro)    : {spark_f1_macro:.4f}\\n')
    f.write(f'F1 (weighted) : {spark_f1_wtd:.4f}\\n\\n')
    f.write(f'--- PER-CLASS ACCURACY ---\\n')
    f.write(f'{\"Label\":<20} {\"Support\":>8} {\"Correct\":>8} {\"Accuracy\":>10}\\n')
    f.write('-' * 50 + '\\n')
    for row in per_class_rows:
        f.write(f'{str(row.label):<20} {row.support:>8} {row.correct:>8} {row.per_class_accuracy:>10.4f}\\n')
    f.write('\\n--- PRODUCTION ARCHITECTURE MAPPING ---\\n')
    f.write('Local step                  ->  Production equivalent\\n')
    f.write('SparkSession local[*]       ->  YARN ResourceManager + NodeManagers\\n')
    f.write('spark.read.csv (local path) ->  spark.read.csv (HDFS path)\\n')
    f.write('randomSplit                 ->  Pre-partitioned HDFS train/test dirs\\n')
    f.write('StringIndexer               ->  Fitted on full cluster dataset; saved to HDFS\\n')
    f.write('VectorAssembler             ->  Dense vectors shuffled across executors\\n')
    f.write('StandardScaler              ->  Mean/std broadcast to all executors\\n')
    f.write('RandomForestClassifier      ->  Trees built in parallel on YARN containers (data-local)\\n')
    f.write('model.transform             ->  Batch scoring job or Structured Streaming\\n')
    f.write('Spark SQL per-class query   ->  Hive metastore / QuickSight dashboard\\n')
    f.write('PipelineModel (local)       ->  model.write().save("hdfs://...") + MLflow tracking\\n')

print(f'Results written -> {result_path}')
"""))

# ------------------------------------------------------------------
# 10. Shutdown & architecture summary
# ------------------------------------------------------------------
cells.append(nbf.v4.new_code_cell("""\
spark.stop()
print("SparkSession stopped.")
"""))

cells.append(nbf.v4.new_markdown_cell(
    "## Architecture Mapping Summary\n\n"
    "| This notebook (local) | Production cluster equivalent |\n"
    "|---|---|\n"
    "| `SparkSession.master('local[*]')` | YARN ResourceManager allocates executors across NodeManagers |\n"
    "| `spark.read.csv('../data/...')` | `spark.read.csv('hdfs://namenode:9000/...')` — data on HDFS DataNodes |\n"
    "| `randomSplit([0.8, 0.2])` | Pre-partitioned train/test directories on HDFS, versioned in data catalogue |\n"
    "| `StringIndexer` fit on local data | Fit on full distributed dataset; mapping saved to HDFS |\n"
    "| `VectorAssembler` | Combines features into dense vectors; shuffled across executor JVMs |\n"
    "| `StandardScaler` | Mean/std computed via distributed aggregation; broadcast to executors |\n"
    "| `RandomForestClassifier(numTrees=100)` | Trees built in parallel across YARN containers with data locality (blocks on same NodeManager) |\n"
    "| `model.transform(test_df)` | Batch scoring via `spark-submit`; or real-time via Structured Streaming + Kafka |\n"
    "| Spark SQL per-class query | Hive metastore query; results surfaced in QuickSight / Tableau |\n"
    "| `Pipeline` object (in-memory) | Serialised and tracked in MLflow; `PipelineModel.write().save('hdfs://...')` |\n\n"
    "**Data flow in production:**\n"
    "```\n"
    "Kafka (streaming) / S3 (batch)\n"
    "        ↓  Spark Structured Streaming / Glue ETL\n"
    "      HDFS  (raw + processed zones)\n"
    "        ↓  spark-submit (training job)\n"
    "  MLlib Pipeline → PipelineModel → HDFS / MLflow\n"
    "        ↓  spark-submit (scoring job) or Structured Streaming\n"
    "  Predictions → Hive / Parquet → QuickSight / Athena\n"
    "```"
))

nb.cells = cells

out_nb = Path(__file__).parent / '05_spark_pipeline.ipynb'
with open(out_nb, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Notebook written -> {out_nb}')
