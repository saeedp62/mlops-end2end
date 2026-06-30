"""
mlops_utils – Shared MLOps utility library for Databricks pipelines.

Re-exports the most commonly used helpers so callers can do:

    from mlops_utils import get_or_create_spark, read_delta, write_delta
"""

from mlops_utils.spark_utils import get_or_create_spark, table_exists, wait_for_table
from mlops_utils.data_io import read_delta, write_delta, upsert_delta
from mlops_utils.catalog import setup_catalog_and_schema, drop_and_recreate_schema, ensure_mlops_schemas
from mlops_utils.config_loader import load_config, merge_configs
from mlops_utils.feature_store import FeatureStoreManager
from mlops_utils.validation import ModelValidator, CheckResult

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
]
