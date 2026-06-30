# Databricks notebook source
# MAGIC %md
# MAGIC # Churn Prediction – Feature Engineering (Refactored)
# MAGIC
# MAGIC This notebook is a **thin orchestration wrapper** around the production-grade
# MAGIC Python module `churn.feature_store_pipeline`.
# MAGIC
# MAGIC ## Data Source Strategy
# MAGIC
# MAGIC The pipeline supports three source types configured via YAML:
# MAGIC
# MAGIC | `data_source.type` | When to use | Example |
# MAGIC |---|---|---|
# MAGIC | `unity_catalog_table` | **Production** – read directly from a LOB source catalog | `telco_catalog.customer360.base` |
# MAGIC | `volume_csv` | **Demo / Dev** – CSV uploaded to a UC Volume | `/Volumes/main/shared_data/telco/file.csv` |
# MAGIC | `http_csv` | **Local unit tests only** | IBM public dataset URL |
# MAGIC
# MAGIC All logic lives in the importable packages – this notebook only handles:
# MAGIC 1. Dependency installation
# MAGIC 2. Config loading (env-specific YAML from a UC Volume)
# MAGIC 3. A single function call to run the full 7-stage pipeline

# COMMAND ----------

# MAGIC %pip install --quiet \
# MAGIC   databricks-feature-engineering>=0.13.0 \
# MAGIC   mlflow \
# MAGIC   pandera \
# MAGIC   pyyaml \
# MAGIC   /Volumes/main/dbdemos_mlops/wheels/mlops_utils-0.1.0-py3-none-any.whl \
# MAGIC   /Volumes/main/dbdemos_mlops/wheels/mlops_churn-0.1.0-py3-none-any.whl \
# MAGIC   --upgrade
# MAGIC
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration
# MAGIC
# MAGIC Load the environment-specific config YAML from a Unity Catalog Volume.
# MAGIC Any value can be overridden via `MLOPS_*` environment variables.
# MAGIC
# MAGIC **Demo setup** – run the companion upload cell below to put the IBM Telco CSV
# MAGIC into the Volume before executing the pipeline.

# COMMAND ----------

import os
from churn.config import load_churn_config

# Config YAML is stored in a UC Volume (set via job parameter or env var)
config_path = os.environ.get(
    "MLOPS_CONFIG_PATH",
    "/Volumes/main/dbdemos_mlops/configs/dev.yaml",   # ← swap to prod.yaml for production
)

cfg = load_churn_config(config_path)

print(f"Pipeline configuration:")
print(f"  Catalog:           {cfg.catalog}")
print(f"  Training schema:   {cfg.schemas.training_datasets}")
print(f"  Feature schema:    {cfg.schemas.offline_features}")
print(f"  Source type:       {cfg.data_source.type}")
if cfg.data_source.type == "unity_catalog_table":
    print(f"  Source table:      {cfg.data_source.source_table}")
elif cfg.data_source.type == "volume_csv":
    print(f"  Volume path:       {cfg.data_source.volume_path}")
print(f"  Bronze table:      {cfg.full_bronze_table}")
print(f"  Feature table:     {cfg.full_feature_table}")
print(f"  Label table:       {cfg.full_label_table}")
print(f"  Online store:      enabled={cfg.online_store.enabled}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## (Demo only) Upload Source CSV to the UC Volume
# MAGIC
# MAGIC Skip this cell in production (`unity_catalog_table` source reads directly
# MAGIC from the LOB catalog – no upload needed).
# MAGIC
# MAGIC For the demo (`volume_csv` source), run this cell once to place the IBM
# MAGIC Telco dataset in the configured Volume path.

# COMMAND ----------

if cfg.data_source.type == "volume_csv":
    import os
    import requests

    volume_path = cfg.data_source.volume_path
    volume_dir  = "/".join(volume_path.split("/")[:-1])   # strip filename

    # Ensure the Volume directory exists (it should, but just in case)
    dbutils.fs.mkdirs(volume_dir.replace("/Volumes/", "dbfs:/Volumes/"))

    # Check if the file is already there
    try:
        files = dbutils.fs.ls(volume_dir)
        already_present = any(f.name == volume_path.split("/")[-1] for f in files)
    except Exception:
        already_present = False

    if not already_present:
        print(f"Downloading IBM Telco CSV to {volume_path} …")
        csv_url = (
            "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
            "master/data/Telco-Customer-Churn.csv"
        )
        # Fallback to S3 mirror if GitHub is rate-limited
        for url in [csv_url, "https://dbdemos-dataset.s3.amazonaws.com/"
                    "retail/lakehouse-retail-churn/telco-customer-churn/Telco-Customer-Churn.csv"]:
            try:
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                # Write via dbutils (handles the /Volumes path correctly)
                dbutils.fs.put(
                    volume_path.replace("/Volumes/", "dbfs:/Volumes/"),
                    r.text,
                    overwrite=True,
                )
                print(f"✓ CSV uploaded to {volume_path} ({len(r.content):,} bytes)")
                break
            except Exception as exc:
                print(f"  Failed from {url}: {exc}. Trying fallback…")
    else:
        print(f"✓ File already present at {volume_path} – skipping download.")
else:
    print(f"Source type='{cfg.data_source.type}' – no upload needed.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the Full 7-Stage Pipeline
# MAGIC
# MAGIC The pipeline stages are:
# MAGIC
# MAGIC 1. **Load source** – UC cross-catalog table *or* Volume CSV (from config)
# MAGIC 2. **Write bronze** – persist raw data to Delta for lineage & replay
# MAGIC 3. **Feature engineering** – `num_optional_services`, type coercions, `transaction_ts`
# MAGIC 4. **Split labels** – separate ground-truth `churn` column + `train/test` assignment
# MAGIC 5. **Write label table** – Delta table with PK constraints for point-in-time lookups
# MAGIC 6. **Create feature table** – Unity Catalog Feature Table (drop & recreate)
# MAGIC 7. **Write features** – write feature rows to the UC Feature Table

# COMMAND ----------

from churn.feature_store_pipeline import run_feature_engineering_pipeline

run_feature_engineering_pipeline(
    spark=spark,
    config=cfg,
    reset_feature_table=True,   # Drop & recreate; set False in prod to use merge mode
    publish_online=None,        # Respect cfg.online_store.enabled
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inspect Results

# COMMAND ----------

print("=== Bronze table (raw source data) ===")
display(spark.table(cfg.full_bronze_table))

# COMMAND ----------

print("=== Feature table ===")
display(spark.table(cfg.full_feature_table))

# COMMAND ----------

print("=== Label table (with train/test split) ===")
display(
    spark.table(cfg.full_label_table)
    .groupBy("split")
    .count()
    .orderBy("split")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define On-Demand Feature Function
# MAGIC
# MAGIC `avg_price_increase` is defined in Unity Catalog and used during model training
# MAGIC via a `FeatureFunction` lookup spec.  It computes the average monthly price
# MAGIC increase for long-tenured customers.

# COMMAND ----------

spark.sql(f"""
  CREATE OR REPLACE FUNCTION {cfg.catalog}.{cfg.schemas.offline_features}.avg_price_increase(
    monthly_charges_in DOUBLE,
    tenure_in         DOUBLE,
    total_charges_in  DOUBLE
  )
  RETURNS FLOAT
  LANGUAGE PYTHON
  COMMENT '[Feature Function] Calculate potential average price increase for tenured customers'
  AS $$
  if tenure_in > 0:
      return monthly_charges_in - total_charges_in / tenure_in
  else:
      return 0
  $$
""")

print(f"Feature function created: {cfg.catalog}.{cfg.schemas.offline_features}.avg_price_increase")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Next Step: Model Training with HPO
# MAGIC
# MAGIC Next: [Train a model using HPO with Optuna]($./02_model_training_hpo_optuna)
