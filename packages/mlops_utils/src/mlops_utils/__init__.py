"""
mlops_utils – Shared MLOps utility library for Databricks pipelines.

Re-exports the most commonly used helpers so callers can do:

    from mlops_utils import get_or_create_spark, read_delta, write_delta
"""

from mlops_utils.catalog import (
    drop_and_recreate_schema,
    ensure_mlops_schemas,
    setup_catalog_and_schema,
)
from mlops_utils.config_loader import load_config, merge_configs
from mlops_utils.data_io import read_delta, upsert_delta, write_delta
from mlops_utils.data_validation import DataCheckResult, DataValidator
from mlops_utils.feature_store import FeatureStoreManager
from mlops_utils.logger import get_logger
from mlops_utils.model_logging import log_and_evaluate_model
from mlops_utils.optimization import NoneValuePruner, run_mlflow_optuna_study
from mlops_utils.spark_utils import get_or_create_spark, table_exists, wait_for_table
from mlops_utils.validation import CheckResult, ModelValidator

__all__ = [
    "get_or_create_spark",
    "table_exists",
    "wait_for_table",
    "read_delta",
    "write_delta",
    "upsert_delta",
    "setup_catalog_and_schema",
    "drop_and_recreate_schema",
    "ensure_mlops_schemas",
    "load_config",
    "merge_configs",
    "FeatureStoreManager",
    "ModelValidator",
    "CheckResult",
    "DataValidator",
    "DataCheckResult",
    "get_logger",
    "run_mlflow_optuna_study",
    "NoneValuePruner",
    "log_and_evaluate_model",
]
