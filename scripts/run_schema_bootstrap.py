"""
scripts/run_schema_bootstrap.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Entry-point script called by the Databricks Asset Bundle ``bootstrap_schemas``
job task (spark_python_task).

This script loads the ChurnConfig from the YAML specified by ``--config-path``
and calls ``ensure_schemas`` to idempotently create all 6 MLOps schemas in
Unity Catalog.

Usage (invoked automatically by the DAB job task)::

    spark-submit run_schema_bootstrap.py \
        --config-path /Volumes/lighthouse_bkk6_analytics/training_datasets/bundle/configs/dev.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse args, load config, and bootstrap schemas."""
    parser = argparse.ArgumentParser(
        description="Idempotently create MLOps Unity Catalog schemas."
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Absolute path to the YAML config file (e.g. configs/dev.yaml).",
    )
    args = parser.parse_args()

    logger.info("Schema bootstrap starting (config=%s)...", args.config_path)

    # SparkSession is injected by the Databricks runtime; import lazily so
    # this script can be imported in unit tests without a live Spark cluster.
    from pyspark.sql import SparkSession  # type: ignore[import]

    spark = SparkSession.builder.getOrCreate()

    from churn.config import load_churn_config
    # Shim calls mlops_utils.catalog.ensure_mlops_schemas under the hood.
    # For other use-cases, call ensure_mlops_schemas directly with a custom
    # {schema_name: comment} dict instead of loading a ChurnConfig.
    from churn.schema_bootstrap import ensure_schemas

    cfg = load_churn_config(args.config_path)
    ensure_schemas(spark, cfg)

    logger.info("Schema bootstrap finished successfully.")


if __name__ == "__main__":
    main()
    sys.exit(0)
