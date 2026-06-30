# Databricks notebook source
# MAGIC %md
# MAGIC # Refactored HPO Model Training using Optuna & MLflow

# COMMAND ----------

# MAGIC %run ../_resources/00-setup $adv_mlops=true

# COMMAND ----------

import os
import sys

# Ensure src and packages/mlops_utils/src are in the path for the Databricks notebook
repo_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.append(os.path.join(repo_root, "src"))
sys.path.append(os.path.join(repo_root, "packages", "mlops_utils", "src"))

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from churn.config import load_churn_config
from churn.training import run_model_training_pipeline
from mlops_utils.spark_utils import get_or_create_spark

spark = get_or_create_spark()
config = load_churn_config("configs/dev.yaml")
fe_client = FeatureEngineeringClient()

# COMMAND ----------

# Run MLflow Optuna Study
# Leverage shared utils which is generic and model/ml framework agnostic
run_model_training_pipeline(spark=spark, config=config, fe_client=fe_client)

# COMMAND ----------
# MAGIC %md
# MAGIC # Done
