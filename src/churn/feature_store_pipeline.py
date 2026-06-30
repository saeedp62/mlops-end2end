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

from typing import TYPE_CHECKING

from mlops_utils.logger import get_logger

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

from churn.config import ChurnConfig

logger = get_logger(__name__)


def run_feature_engineering_pipeline(
    spark: SparkSession,
    config: ChurnConfig,
    *,
    fe_client: object | None = None,
    reset_feature_table: bool = True,
    publish_online: bool | None = None,
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

    from mlops_utils.data_io import add_primary_key_constraint, write_delta
    from mlops_utils.feature_store import FeatureStoreManager

    from churn.data_source import get_source_dataframe, ingest_bronze_table
    from churn.feature_engineering import (
        build_feature_df,
        split_label_from_features,
    )

    if fe_client:
        fsm = FeatureStoreManager(
            fe=fe_client,
            catalog=config.catalog,
            offline_schema=config.schemas.offline_features,
            online_schema=config.schemas.online_features,
        )
    else:
        fsm = FeatureStoreManager.from_config(config)

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
    # Stage 3.5: Data Quality Validation
    # ------------------------------------------------------------------
    logger.info("[3.5/7] Running data quality validations…")
    from mlops_utils.data_validation import DataValidator

    validator = DataValidator(feature_and_label_df)
    for pk in config.primary_keys:
        validator.add_check(DataValidator.check_no_nulls(pk))
        validator.add_check(DataValidator.check_unique(pk))

    validator.add_check(DataValidator.check_allowed_values(config.label_col, [0, 1]))
    validator.add_check(
        DataValidator.check_custom_sql("total_charges_non_negative", "total_charges >= 0")
    )

    # Run validations and fail the pipeline if they don't pass
    validator.run(raise_on_fail=True)

    # ------------------------------------------------------------------
    # Stage 4: Split labels from features
    # ------------------------------------------------------------------
    logger.info("[4/7] Splitting labels from features…")
    feature_df, label_df = split_label_from_features(
        feature_and_label_df,
        label_col=config.label_col,
        train_ratio=config.train_ratio,
        seed=config.rng_seed,
        strategy="time",
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
    # Stage 6 & 7: Create / replace UC feature table and write features
    # ------------------------------------------------------------------
    logger.info("[6/7] Writing features to '%s'...", config.full_feature_table)
    if reset_feature_table:
        # Drops and recreates the table, then writes in merge mode
        fsm.reset_and_write(
            table_name=config.feature_table,
            df=feature_df,
            primary_keys=list(config.primary_keys),
            timeseries_columns=config.timeseries_col,
            description=(
                f"Features derived from {config.full_bronze_table}. "
                "Includes service counts and cleaned charge columns."
            ),
        )
    else:
        fsm.write(
            table_name=config.feature_table,
            df=feature_df,
            mode="merge",
        )

    # ------------------------------------------------------------------
    # Optional: Publish to online store
    # ------------------------------------------------------------------
    if do_online:
        from mlops_utils.feature_store import publish_online_if_enabled
        publish_online_if_enabled(config, fsm, force_publish=True)

    logger.info("Feature engineering pipeline completed successfully.")
