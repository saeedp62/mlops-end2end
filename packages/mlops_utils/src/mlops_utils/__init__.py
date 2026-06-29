"""
mlops_utils – Shared MLOps utility library for Databricks pipelines.

Re-exports the most commonly used helpers so callers can do:

    from mlops_utils import get_or_create_spark, read_delta, write_delta
"""

from mlops_utils.spark_utils import get_or_create_spark, table_exists, wait_for_table
from mlops_utils.data_io import read_delta, write_delta, upsert_delta
from mlops_utils.catalog import setup_catalog_and_schema, drop_and_recreate_schema
from mlops_utils.config_loader import load_config, merge_configs

__all__ = [
    "get_or_create_spark",
    "table_exists",
    "wait_for_table",
    "read_delta",
    "write_delta",
    "upsert_delta",
    "setup_catalog_and_schema",
    "drop_and_recreate_schema",
    "load_config",
    "merge_configs",
]
