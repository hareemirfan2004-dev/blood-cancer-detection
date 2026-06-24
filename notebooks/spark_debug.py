"""Quick smoke-test to find the Spark failure point."""
import sys, os
os.environ['PYSPARK_PYTHON']        = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession

try:
    spark = (
        SparkSession.builder
        .appName("DebugSession")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"OK: SparkSession created — version {spark.version}")
except Exception as e:
    print(f"FAIL SparkSession: {e}")
    sys.exit(1)

try:
    from pathlib import Path
    csv_path = str(Path(__file__).parent.parent / "data" / "blood_cancer_hybrid.csv")
    df = spark.read.option("header","true").option("inferSchema","true").csv(csv_path)
    print(f"OK: CSV loaded — {df.count()} rows, {len(df.columns)} cols")
except Exception as e:
    print(f"FAIL CSV load: {e}")
    spark.stop()
    sys.exit(1)

try:
    from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, Imputer
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.ml import Pipeline
    print("OK: MLlib imports successful")
except Exception as e:
    print(f"FAIL MLlib import: {e}")
    spark.stop()
    sys.exit(1)

spark.stop()
print("Debug complete — no errors.")
