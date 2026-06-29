"""
churn.feature_store_pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Orchestrates the full data → feature-store pipeline for the churn use case.

This is the **main entry point** called by:
- Databricks notebooks (thin wrapper)
- DLT pipeline definitions
- CI integration tests

The pipeline runs these stages:
1. Read bronze customers table
2. Build features (compute + clean + timestamp)
3. Split labels from features
4. Write label table to Delta (with PK constraints)
5. Create / replace UC feature table
6. Write features to UC feature table
7. Optionally publish features to online store

All stages are orchestrated via injected clients (SparkSession,
FeatureEngineeringClient, etc.) for full testability.

Usage::

    from churn.config import load_churn_config
    from churn.feature_store_pipeline import run_feature_engineering_pipeline
    from mlops_utils.spark_utils import get_or_create_spark

    spark = get_or_create_spark()
    cfg = load_churn_config("configs/dev.yaml")
    run_feature_engineering_pipeline(spark, cfg)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

from churn.config import ChurnConfig

logger = logging.getLogger(__name__)


def run_feature_engineering_pipeline(
    spark: "SparkSession",
    config: ChurnConfig,
    *,
    fe_client: Optional[object] = None,
    reset_feature_table: bool = True,
    publish_online: Optional[bool] = None,
) -> None:
    """Full Bronze → Feature-Store pipeline.

    Parameters
    ----------
    spark:
        Active SparkSession.
    config:
        Churn pipeline configuration.
    fe_client:
        Optional pre-constructed ``FeatureEngineeringClient``.  A new one
        is created if ``None`` (requires Databricks runtime).
    reset_feature_table:
        Drop and recreate the feature table on each run (useful during
        development; set to ``False`` in production to use merge mode).
    publish_online:
        Override ``config.online_store.enabled``.  Pass ``True`` to force
        online publish, ``False`` to skip, or ``None`` to respect the config.

    Raises
    ------
    Exception
        Re-raises any error from a pipeline stage after logging it.
    """
    from databricks.feature_engineering import FeatureEngineeringClient  # type: ignore[import]

    from churn.data_source import get_source_dataframe, ingest_bronze_table
    from churn.feature_engineering import (
        build_feature_df,
        split_label_from_features,
    )
    from mlops_utils.data_io import write_delta, add_primary_key_constraint
    from mlops_utils.feature_store import (
        create_or_replace_feature_table,
        write_feature_table,
        publish_to_online_store,
    )

    fe = fe_client or FeatureEngineeringClient()
    do_online = config.online_store.enabled if publish_online is None else publish_online

    # ------------------------------------------------------------------
    # Stage 1: Load source data (cross-LOB UC table, Volume CSV, or HTTP)
    # ------------------------------------------------------------------
    source_type = config.data_source.type
    logger.info("[1/7] Loading source data (type='%s')…", source_type)
    raw_df = get_source_dataframe(spark, config)

    # ------------------------------------------------------------------
    # Stage 2: Persist to bronze Delta table
    # (enables downstream replay, audit, and data lineage in UC)
    # ------------------------------------------------------------------
    logger.info("[2/7] Writing bronze table '%s'…", config.full_bronze_table)
    ingest_bronze_table(
        spark,
        raw_df,
        config.full_bronze_table,
        mode="overwrite",
        validate_schema=True,
    )
    bronze_df = spark.table(config.full_bronze_table)

    # ------------------------------------------------------------------
    # Stage 3: Feature engineering
    # ------------------------------------------------------------------
    logger.info("[3/7] Computing features…")
    feature_and_label_df = build_feature_df(bronze_df)

    # ------------------------------------------------------------------
    # Stage 4: Split labels from features
    # ------------------------------------------------------------------
    logger.info("[4/7] Splitting labels from features…")
    feature_df, label_df = split_label_from_features(
        feature_and_label_df,
        label_col=config.label_col,
        train_ratio=config.train_ratio,
        seed=config.rng_seed,
    )

    # ------------------------------------------------------------------
    # Stage 5: Write label table
    # ------------------------------------------------------------------
    logger.info("[5/7] Writing label table '%s'…", config.full_label_table)
    write_delta(
        label_df,
        config.full_label_table,
        mode="overwrite",
        comment=(
            f"Ground-truth {config.label_col} labels with train/test split "
            f"for {config.full_feature_table}."
        ),
    )
    # Add PK constraints so Unity Catalog can link the label table to the feature table
    add_primary_key_constraint(
        spark,
        config.full_label_table,
        config.label_table_pk_constraint,
        key_columns=list(config.primary_keys),
        timeseries_columns=[config.timeseries_col],
    )

    # ------------------------------------------------------------------
    # Stage 6: Create / replace UC feature table
    # ------------------------------------------------------------------
    logger.info("[6/7] Creating feature table '%s'…", config.full_feature_table)
    if reset_feature_table:
        create_or_replace_feature_table(
            fe=fe,
            name=config.full_feature_table,
            df=feature_df,
            primary_keys=list(config.primary_keys),
            timeseries_columns=config.timeseries_col,
            description=(
                f"Features derived from {config.full_bronze_table}. "
                "Includes service counts and cleaned charge columns."
            ),
        )

    # ------------------------------------------------------------------
    # Stage 7: Write features
    # ------------------------------------------------------------------
    logger.info("[7/7] Writing features to '%s'…", config.full_feature_table)
    write_feature_table(
        fe=fe,
        name=config.full_feature_table,
        df=feature_df,
        mode="merge" if not reset_feature_table else "overwrite",
    )

    # ------------------------------------------------------------------
    # Optional: Publish to online store
    # ------------------------------------------------------------------
    if do_online:
        _publish_online(config, fe)

    logger.info("Feature engineering pipeline completed successfully.")


# ---------------------------------------------------------------------------
# Online store publish helper
# ---------------------------------------------------------------------------

def _publish_online(config: ChurnConfig, fe: object) -> None:
    """Publish the offline feature table to the configured online store."""
    from mlops_utils.feature_store import (
        publish_to_online_store,
        create_feature_serving_endpoint,
    )

    backend = config.online_store.backend.lower()
    logger.info("Publishing features to online store (backend=%s)…", backend)

    online_store_spec = _build_online_store_spec(config)

    publish_to_online_store(
        fe=fe,
        feature_table_name=config.full_feature_table,
        online_store_spec=online_store_spec,
        mode="merge",
    )

    # Optionally create a Feature Serving endpoint
    endpoint_name = config.online_store.endpoint_name
    if endpoint_name:
        try:
            from databricks.feature_engineering.entities.feature_serving_endpoint import (  # type: ignore[import]
                ServedEntity,
            )

            served_entities = [
                ServedEntity(
                    feature_spec_name=config.full_feature_table,
                    workload_size="Small",
                    scale_to_zero_enabled=True,
                )
            ]
            create_feature_serving_endpoint(
                endpoint_name=endpoint_name,
                served_entities=served_entities,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not create serving endpoint '%s': %s", endpoint_name, exc)


def _build_online_store_spec(config: ChurnConfig) -> object:
    """Build the backend-specific online store spec object."""
    backend = config.online_store.backend.lower()
    extra = config.online_store.extra

    if backend == "databricks":
        # Databricks-managed online store (no external config needed)
        try:
            from databricks.feature_store.online_store_spec import (  # type: ignore[import]
                AzureCosmosDBSpec,
            )

            return AzureCosmosDBSpec(**extra) if extra else None
        except ImportError:
            return None  # Will use the FE client's default

    if backend == "dynamodb":
        from databricks.feature_store.online_store_spec import (  # type: ignore[import]
            AmazonDynamoDBSpec,
        )

        return AmazonDynamoDBSpec(**extra)

    if backend in {"cosmosdb", "cosmos"}:
        from databricks.feature_store.online_store_spec import (  # type: ignore[import]
            AzureCosmosDBSpec,
        )

        return AzureCosmosDBSpec(**extra)

    raise ValueError(
        f"Unknown online_store.backend='{backend}'. "
        "Choose from: databricks, dynamodb, cosmosdb."
    )
