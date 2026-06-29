"""
pipelines/dlt_feature_engineering.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Delta Live Tables (DLT) pipeline: Silver feature computation.

Pipeline DAG
------------
::

    bronze_customers (dlt_ingestion.py)
        └─► bronze_customers_validated
                ├─► silver_churn_features   (feature table – no label col)
                └─► silver_churn_labels     (label + train/test split)
                      ↑ also reads silver_churn_features for transaction_ts key

``silver_churn_labels`` reads **transaction_ts** from the already-computed
``silver_churn_features`` table (DLT dependency) and the **churn label** from
``bronze_customers_validated``.  This avoids re-running the full feature
pipeline a second time.

DLT pipeline parameters:
    config_path: /Volumes/main/dbdemos_mlops/configs/prod.yaml

This pipeline coexists with the notebook-based workflow – the same pure
functions from ``churn.feature_engineering`` are called by both paths.
"""

from __future__ import annotations

import dlt  # type: ignore[import]

from churn.config import load_churn_config
from churn.feature_engineering import (
    compute_service_features,
    clean_churn_features,
    add_transaction_timestamp,
)


# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

def _get_config():  # type: ignore[no-untyped-def]
    """Load ChurnConfig from pipeline parameter or fall back to env/defaults."""
    try:
        config_path = (
            spark.conf.get("config_path", "")  # type: ignore[name-defined]  # noqa: F821
            or spark.conf.get("pipelines.config_path", "")  # type: ignore[name-defined]  # noqa: F821
        )
    except Exception:  # noqa: BLE001
        config_path = ""

    if config_path:
        return load_churn_config(config_path)

    from churn.config import ChurnConfig
    return ChurnConfig()


cfg = _get_config()

# DLT table names referenced across both functions
_BRONZE_VALIDATED = f"{cfg.bronze_table}_validated"


# ---------------------------------------------------------------------------
# Silver feature table
# ---------------------------------------------------------------------------

@dlt.expect_or_drop("customer_id_not_null", "customer_id IS NOT NULL")
@dlt.expect_or_drop("transaction_ts_not_null", "transaction_ts IS NOT NULL")
@dlt.expect("num_optional_services_range", "num_optional_services BETWEEN 0 AND 6")
@dlt.expect("tenure_non_negative", "tenure >= 0 OR tenure IS NULL")
@dlt.expect("monthly_charges_non_negative", "monthly_charges >= 0 OR monthly_charges IS NULL")
@dlt.table(
    name=cfg.feature_table,
    comment=(
        f"Silver churn feature table derived from {cfg.full_bronze_table}. "
        "Includes num_optional_services (Pandas UDF), cleaned charge columns, "
        "and a transaction_ts snapshot timestamp. Label column excluded."
    ),
    table_properties={
        "quality": "silver",
        "delta.enableChangeDataFeed": "true",   # Required for streaming FS reads
    },
)
def silver_churn_features():  # type: ignore[no-untyped-def]
    """Apply all feature-engineering transforms to the validated bronze table.

    Steps:
    1. ``compute_service_features``  – Pandas UDF counting optional 'Yes' services
    2. ``clean_churn_features``      – senior_citizen int→str, total_charges parse, null fill
    3. ``add_transaction_timestamp`` – append ``transaction_ts`` snapshot column
    4. Drop ``churn`` label column   – prevents label leakage into the feature table
    """
    bronze_df = dlt.read(_BRONZE_VALIDATED)

    df = compute_service_features(bronze_df)
    df = clean_churn_features(df)
    df = add_transaction_timestamp(df)

    return df.drop(cfg.label_col)


# ---------------------------------------------------------------------------
# Label / ground-truth table
# ---------------------------------------------------------------------------

@dlt.expect_or_drop("customer_id_not_null", "customer_id IS NOT NULL")
@dlt.expect_or_drop("transaction_ts_not_null", "transaction_ts IS NOT NULL")
@dlt.expect_or_drop("valid_split", "split IN ('train', 'test')")
@dlt.expect("label_not_null", f"{cfg.label_col} IN ('Yes', 'No') OR {cfg.label_col} IS NULL")
@dlt.table(
    name=cfg.label_table,
    comment=(
        f"Ground-truth churn labels joined from {cfg.full_bronze_table}. "
        "Keyed on (customer_id, transaction_ts) for point-in-time Feature Store lookups. "
        "Includes train/test split assignments."
    ),
    table_properties={
        "quality": "silver",
        "delta.enableChangeDataFeed": "true",
    },
)
def silver_churn_labels():  # type: ignore[no-untyped-def]
    """Build the label table by joining features (for keys) with bronze (for label).

    Reads ``silver_churn_features`` – already computed above – for
    ``customer_id`` and ``transaction_ts`` (avoids re-running the full
    feature pipeline). Joins with ``bronze_customers_validated`` to retrieve
    the raw ``churn`` label column, then assigns a deterministic train/test split.

    The join is performed on ``customer_id`` only (transaction_ts is not present
    in the bronze table; it is added during feature engineering).
    """
    import pyspark.sql.functions as F

    # Keys + timestamp from the already-computed silver feature table
    keys_df = dlt.read(cfg.feature_table).select(
        "customer_id", "transaction_ts"
    )

    # Raw label from validated bronze (churn column preserved there)
    label_source_df = dlt.read(_BRONZE_VALIDATED).select(
        "customer_id", cfg.label_col
    )

    # Join on customer_id to produce (customer_id, transaction_ts, churn)
    joined = keys_df.join(label_source_df, on="customer_id", how="left")

    # Deterministic train / test split
    label_df = (
        joined
        .withColumn("_rand", F.rand(seed=cfg.rng_seed))
        .withColumn(
            "split",
            F.when(F.col("_rand") < cfg.train_ratio, "train").otherwise("test"),
        )
        .drop("_rand")
    )

    return label_df
