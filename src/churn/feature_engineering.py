"""
churn.feature_engineering
~~~~~~~~~~~~~~~~~~~~~~~~~
Pure feature-engineering transforms for the Telco churn dataset.

All functions are **stateless** (SparkDataFrame → SparkDataFrame) with no
side effects, making them:
- Independently unit-testable with a local SparkSession
- Directly callable from DLT ``@dlt.table`` definitions
- Composable into any pipeline (batch, streaming, DLT)

Public API
----------
::

    from churn.feature_engineering import (
        compute_service_features,
        clean_churn_features,
        add_transaction_timestamp,
        build_feature_df,
        split_label_from_features,
    )
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from mlops_utils.logger import get_logger

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame

logger = get_logger(__name__)

# Optional services checked when computing num_optional_services
_OPTIONAL_SERVICE_COLS = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
]

# Numerical columns to fill nulls with 0.0
_NULL_FILL_COLS = {
    "tenure": 0.0,
    "monthly_charges": 0.0,
    "total_charges": 0.0,
}


def compute_service_features(df: SparkDataFrame) -> SparkDataFrame:
    """Add ``num_optional_services`` column – count of optional services enabled.

    Uses a Pandas UDF so the computation scales across the full Spark cluster.

    Parameters
    ----------
    df:
        Input Spark DataFrame (must contain the six optional-service columns).

    Returns
    -------
    SparkDataFrame with an additional ``num_optional_services`` (double) column.
    """
    from pyspark.sql.functions import pandas_udf

    @pandas_udf("double")
    def _num_optional_services(*cols):  # type: ignore[no-untyped-def]
        """Count the number of 'Yes' values across optional service columns."""
        return sum(map(lambda s: (s == "Yes").astype("double"), cols))

    return df.withColumn(
        "num_optional_services",
        _num_optional_services(*_OPTIONAL_SERVICE_COLS),
    )


def clean_churn_features(df: SparkDataFrame) -> SparkDataFrame:
    """Clean and type-cast raw bronze columns using the Pandas-on-Spark API.

    Transformations applied:
    - ``senior_citizen`` int → ``"Yes"`` / ``"No"`` string mapping
    - ``total_charges`` parsed to float (handles whitespace-only strings)
    - Null-fill for ``tenure``, ``monthly_charges``, ``total_charges``

    Parameters
    ----------
    df:
        Input Spark DataFrame (bronze schema).

    Returns
    -------
    Cleaned SparkDataFrame.
    """
    # Work via pandas-on-spark (koalas) for familiar pandas syntax at scale
    psdf = df.pandas_api()

    # senior_citizen: int 0/1 → string "No"/"Yes"
    psdf = psdf.astype({"senior_citizen": "string"})
    psdf["senior_citizen"] = psdf["senior_citizen"].map({"1": "Yes", "0": "No"})

    # total_charges: parse to float (source has whitespace-only strings)
    psdf["total_charges"] = psdf["total_charges"].apply(
        lambda x: float(x) if str(x).strip() else 0.0
    )

    # Fill numerical nulls
    psdf = psdf.fillna(_NULL_FILL_COLS)

    return psdf.to_spark()


def add_transaction_timestamp(
    df: SparkDataFrame,
    *,
    timestamp: datetime | None = None,
    col_name: str = "transaction_ts",
) -> SparkDataFrame:
    """Append a ``transaction_ts`` timestamp column to *df*.

    Parameters
    ----------
    df:
        Input Spark DataFrame.
    timestamp:
        Timestamp value to use.  Defaults to ``datetime.now()`` if ``None``.
    col_name:
        Name of the new column (default: ``"transaction_ts"``).

    Returns
    -------
    SparkDataFrame with the new timestamp column appended.
    """
    from pyspark.sql.functions import lit

    ts = timestamp or datetime.now()
    return df.withColumn(col_name, lit(ts.timestamp()).cast("timestamp"))


def build_feature_df(bronze_df: SparkDataFrame) -> SparkDataFrame:
    """Compose all feature-engineering steps into a single transformation.

    Pipeline:
    1. :func:`compute_service_features` – add ``num_optional_services``
    2. :func:`clean_churn_features` – type cast + null fill
    3. :func:`add_transaction_timestamp` – append ``transaction_ts``

    Parameters
    ----------
    bronze_df:
        Raw bronze Spark DataFrame.

    Returns
    -------
    Feature-engineered Spark DataFrame (without the ``churn`` label column).
    """
    logger.info("Running full feature-engineering pipeline…")
    df = compute_service_features(bronze_df)
    df = clean_churn_features(df)
    df = add_transaction_timestamp(df)
    logger.info("Feature engineering complete.")
    return df


def split_label_from_features(
    df: SparkDataFrame,
    label_col: str = "churn",
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    strategy: str = "random",
) -> tuple[SparkDataFrame, SparkDataFrame]:
    """Split *df* into a feature DataFrame and a label DataFrame.

    Adds a ``split`` column (``"train"`` / ``"test"``) to the label table.
    The ``churn`` column is dropped from the feature DataFrame to prevent
    label leakage.

    Parameters
    ----------
    df:
        Combined feature + label Spark DataFrame.
    label_col:
        Name of the label column to extract.
    train_ratio:
        Fraction of rows assigned to ``"train"`` split.
    seed:
        Random seed for reproducible splitting (if strategy="random").
    strategy:
        "random" or "time". If "time", splits out-of-time using `transaction_ts`.

    Returns
    -------
    (feature_df, label_df)
        ``feature_df`` – all columns except *label_col*
        ``label_df``   – ``customer_id``, ``transaction_ts``, *label_col*, ``split``
    """
    import pyspark.sql.functions as F

    label_df_raw = df.select("customer_id", "transaction_ts", label_col)

    if strategy == "time":
        from mlops_utils.data_splitting import time_based_split_spark
        label_df = time_based_split_spark(
            label_df_raw, time_col="transaction_ts", train_ratio=train_ratio
        )
    else:
        label_df = (
            label_df_raw
            .withColumn("random", F.rand(seed=seed))
            .withColumn(
                "split",
                F.when(F.col("random") < train_ratio, "train").otherwise("test"),
            )
            .drop("random")
        )

    feature_df = df.drop(label_col)

    logger.info(
        "Split (strategy=%s): %s rows total (%.0f%% train target).",
        strategy,
        df.count() if logger.isEnabledFor(logging.DEBUG) else "?",
        train_ratio * 100,
    )
    return feature_df, label_df


def get_latest_label_per_customer(
    labels_df: SparkDataFrame,
    label_col: str = "churn",
) -> SparkDataFrame:
    """Return the most recent label row per customer.

    Used before building the training set to ensure a single row per
    ``customer_id`` (point-in-time lookup keys must be unique).

    Parameters
    ----------
    labels_df:
        Label table Spark DataFrame with columns
        ``customer_id``, ``transaction_ts``, *label_col*, ``split``.
    label_col:
        Name of the label column.

    Returns
    -------
    Spark DataFrame with one row per ``customer_id``.
    """
    from pyspark.sql.functions import last
    from pyspark.sql.functions import max as spark_max

    return (
        labels_df
        .groupBy("customer_id")
        .agg(
            spark_max("transaction_ts").alias("transaction_ts"),
            last(label_col).alias(label_col),
            last("split").alias("split"),
        )
    )


def build_churn_feature_lookups(cfg: Any, fsm: Any = None) -> list[Any]:
    """Return standard FeatureLookup + FeatureFunction specs from a ChurnConfig."""
    from databricks.feature_engineering import FeatureFunction

    lookups = []
    if fsm:
        lookups.extend(
            fsm.create_lookups(
                table_name=cfg.feature_table,
                lookup_keys=list(cfg.primary_keys),
                timestamp_lookup_key=cfg.timeseries_col,
            )
        )
    else:
        from databricks.feature_engineering import FeatureLookup
        lookups.append(
            FeatureLookup(
                table_name=f"{cfg.catalog}.{cfg.schemas.offline_features}.{cfg.feature_table}",
                lookup_key=list(cfg.primary_keys),
                timestamp_lookup_key=cfg.timeseries_col,
            )
        )

    lookups.append(
        FeatureFunction(
            udf_name=f"{cfg.catalog}.{cfg.schemas.offline_features}.avg_price_increase",
            output_name="avg_price_increase",
            input_bindings={
                "monthly_charges_in": "monthly_charges",
                "tenure_in": "tenure",
                "total_charges_in": "total_charges",
            },
        )
    )

    return lookups
