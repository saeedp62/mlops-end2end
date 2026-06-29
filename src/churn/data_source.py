"""
churn.data_source
~~~~~~~~~~~~~~~~~
Bronze layer ingestion with a pluggable, config-driven source strategy.

Three source types are supported (controlled by ``DataSourceConfig.type``):

``unity_catalog_table``  *(production default)*
    Read directly from a Delta table in **another LOB catalog** using the
    Unity Catalog cross-catalog reader.  No data movement or internet access
    required – Databricks streams the data under the caller's identity.

``volume_csv``  *(demo / dev)*
    Load a CSV file stored in a Unity Catalog Volume path
    (``/Volumes/<catalog>/<schema>/<volume>/<file>``) via the Spark CSV reader.
    Use this for demo data that lives in a Volume instead of a table.

``http_csv``  *(local unit tests only)*
    Download from a public HTTP URL (falls back to an S3 mirror) and
    load via pandas.  **Not for production or Databricks clusters** – intended
    solely for local development and CI unit tests that cannot access Volumes.

Public API
----------
::

    from churn.data_source import get_source_dataframe, ingest_bronze_table

    # Load from whichever source is configured
    df = get_source_dataframe(spark, config)

    # Optionally persist to the bronze Delta table
    ingest_bronze_table(spark, df, config.full_bronze_table)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame, SparkSession

    from churn.config import ChurnConfig, DataSourceConfig

logger = logging.getLogger(__name__)

# Fallback HTTP URL (S3 mirror – avoids GitHub rate limits on CI runners)
_DEFAULT_HTTP_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
_S3_FALLBACK_URL = (
    "https://dbdemos-dataset.s3.amazonaws.com/"
    "retail/lakehouse-retail-churn/telco-customer-churn/Telco-Customer-Churn.csv"
)


# ===========================================================================
# Public API
# ===========================================================================

def get_source_dataframe(
    spark: "SparkSession",
    config: "ChurnConfig",
) -> "SparkDataFrame":
    """Dispatch to the correct source reader based on ``config.data_source.type``.

    Parameters
    ----------
    spark:
        Active SparkSession.
    config:
        ``ChurnConfig`` whose ``data_source`` field controls which reader is used.

    Returns
    -------
    pyspark.sql.DataFrame
        Raw (un-engineered) Spark DataFrame ready for feature transforms.
        Column normalisation is applied when ``config.data_source.normalize_columns``
        is ``True``.

    Raises
    ------
    ValueError
        If ``config.data_source.type`` is not a recognised strategy.
    """
    ds = config.data_source
    logger.info("Data source type: '%s'.", ds.type)

    if ds.type == "unity_catalog_table":
        return read_from_unity_catalog(spark, ds)

    if ds.type == "volume_csv":
        return read_from_volume_csv(spark, ds)

    if ds.type == "http_csv":
        return read_from_http_csv(spark, ds)

    raise ValueError(
        f"Unknown data_source.type='{ds.type}'. "
        "Expected one of: unity_catalog_table, volume_csv, http_csv."
    )


def ingest_bronze_table(
    spark: "SparkSession",
    df: "SparkDataFrame",
    full_table_name: str,
    *,
    mode: str = "overwrite",
    validate_schema: bool = True,
) -> None:
    """Persist *df* to a bronze Delta table.

    This is a **thin persistence wrapper** – all source-reading logic lives in
    :func:`get_source_dataframe`.  Call it after :func:`get_source_dataframe`
    when you want to materialise the source data into the bronze layer.

    Parameters
    ----------
    spark:
        Active SparkSession.
    df:
        Spark DataFrame to persist (already loaded and normalised).
    full_table_name:
        Destination Delta table (``catalog.schema.table``).
    mode:
        Write mode – ``"overwrite"`` or ``"append"``.
    validate_schema:
        Run pandera schema validation on a pandas sample before writing.
        Validation is done on a 10k-row sample to avoid driver OOM.
    """
    if validate_schema:
        _validate_sample(df)

    from mlops_utils.data_io import write_delta

    write_delta(df, full_table_name, mode=mode)
    logger.info(
        "Ingested bronze table '%s' (mode=%s).", full_table_name, mode
    )


# ===========================================================================
# Source-specific readers
# ===========================================================================

def read_from_unity_catalog(
    spark: "SparkSession",
    ds: "DataSourceConfig",
) -> "SparkDataFrame":
    """Read a Delta table from another LOB Unity Catalog.

    Uses the Spark ``spark.table()`` API which Databricks routes transparently
    across catalog boundaries using the caller's Unity Catalog identity (no
    credential copy required).

    Parameters
    ----------
    spark:
        Active SparkSession.
    ds:
        ``DataSourceConfig`` with ``type == "unity_catalog_table"`` and a
        non-empty ``source_table``.

    Returns
    -------
    pyspark.sql.DataFrame

    Raises
    ------
    ValueError
        If ``source_table`` is not set.
    """
    if not ds.source_table:
        raise ValueError(
            "data_source.source_table must be set when type='unity_catalog_table'. "
            "Example: 'retail_catalog.crm.customers'"
        )

    logger.info(
        "Reading source from Unity Catalog table '%s'.", ds.source_table
    )
    df = spark.table(ds.source_table)

    if ds.normalize_columns:
        df = _normalize_spark_columns(df)

    logger.info(
        "Loaded %s columns from '%s'.", len(df.columns), ds.source_table
    )
    return df


def read_from_volume_csv(
    spark: "SparkSession",
    ds: "DataSourceConfig",
) -> "SparkDataFrame":
    """Read a CSV file from a Unity Catalog Volume path.

    Parameters
    ----------
    spark:
        Active SparkSession.
    ds:
        ``DataSourceConfig`` with ``type == "volume_csv"`` and a non-empty
        ``volume_path`` pointing to a file inside a UC Volume
        (e.g. ``/Volumes/main/shared_data/telco/Telco-Customer-Churn.csv``).

    Returns
    -------
    pyspark.sql.DataFrame

    Raises
    ------
    ValueError
        If ``volume_path`` is not set.
    """
    if not ds.volume_path:
        raise ValueError(
            "data_source.volume_path must be set when type='volume_csv'. "
            "Example: '/Volumes/main/shared_data/telco/Telco-Customer-Churn.csv'"
        )

    logger.info("Reading source CSV from Volume path '%s'.", ds.volume_path)

    # Apply caller-provided Spark CSV options (header, inferSchema, etc.)
    reader = spark.read.format("csv")
    for k, v in ds.csv_options.items():
        reader = reader.option(k, v)

    df = reader.load(ds.volume_path)

    if ds.normalize_columns:
        df = _normalize_spark_columns(df)

    logger.info(
        "Loaded %s columns from Volume path '%s'.", len(df.columns), ds.volume_path
    )
    return df


def read_from_http_csv(
    spark: "SparkSession",
    ds: "DataSourceConfig",
) -> "SparkDataFrame":
    """Download a CSV from an HTTP URL and convert to a Spark DataFrame.

    Intended **only for local development and CI unit tests** that cannot
    access a Databricks Volume.  In production, always use
    ``unity_catalog_table`` or ``volume_csv``.

    Parameters
    ----------
    spark:
        Active SparkSession.
    ds:
        ``DataSourceConfig`` with ``type == "http_csv"``.  ``url`` is optional;
        defaults to the IBM Telco dataset public URL with an S3 fallback.

    Returns
    -------
    pyspark.sql.DataFrame
    """
    import io

    import pandas as pd
    import requests

    primary_url = ds.url or _DEFAULT_HTTP_URL

    pdf: Optional[pd.DataFrame] = None
    for attempt_url in filter(None, [primary_url, _S3_FALLBACK_URL]):
        try:
            response = requests.get(attempt_url, timeout=60)
            response.raise_for_status()
            pdf = pd.read_csv(io.StringIO(response.text))
            logger.info(
                "Downloaded CSV from %s (%d rows).", attempt_url, len(pdf)
            )
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch %s: %s. Trying fallback…", attempt_url, exc)

    if pdf is None:
        raise RuntimeError(
            f"Could not download CSV from any URL. Last attempted: {_S3_FALLBACK_URL}"
        )

    if ds.normalize_columns:
        pdf = _normalize_pandas_columns(pdf)

    return spark.createDataFrame(pdf)


# ===========================================================================
# Column normalisation helpers
# ===========================================================================

def normalize_column_names(df: "import pandas as pd; pd.DataFrame") -> "import pandas as pd; pd.DataFrame":  # type: ignore[name-defined]
    """Public alias for :func:`_normalize_pandas_columns` (kept for backwards compat)."""
    import pandas as pd
    return _normalize_pandas_columns(df)


def _normalize_pandas_columns(df: "import pandas as pd; pd.DataFrame") -> "import pandas as pd; pd.DataFrame":  # type: ignore[name-defined]
    """Normalise column names of a **pandas** DataFrame to snake_case."""
    import re
    import pandas as pd

    def _norm(name: str) -> str:
        name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)  # CamelCase → snake
        name = name.lower()
        name = re.sub(r"[ \-]", "_", name)
        name = re.sub(r"[()]", "", name)
        name = re.sub(r"_+", "_", name)
        return name.strip("_")

    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    # Known IBM dataset renames
    return df.rename(columns={
        "streaming_t_v": "streaming_tv",
        "customer_i_d": "customer_id",
    })


def _normalize_spark_columns(df: "SparkDataFrame") -> "SparkDataFrame":
    """Normalise column names of a **Spark** DataFrame to snake_case.

    Applied when the source table uses PascalCase or other non-snake_case
    naming (e.g. ``CustomerID`` → ``customer_id``).
    """
    import re

    def _norm(name: str) -> str:
        name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
        name = name.lower()
        name = re.sub(r"[ \-]", "_", name)
        name = re.sub(r"[()]", "", name)
        name = re.sub(r"_+", "_", name)
        return name.strip("_")

    rename_map = {
        "streaming_t_v": "streaming_tv",
        "customer_i_d": "customer_id",
    }

    for old_col in df.columns:
        new_col = rename_map.get(_norm(old_col), _norm(old_col))
        if new_col != old_col:
            df = df.withColumnRenamed(old_col, new_col)

    return df


# ===========================================================================
# Schema validation helper (internal)
# ===========================================================================

def _validate_sample(df: "SparkDataFrame", sample_size: int = 10_000) -> None:
    """Run pandera validation on a small pandas sample of *df*."""
    from churn.schemas import BronzeCustomerSchema

    sample_pdf = df.limit(sample_size).toPandas()
    try:
        BronzeCustomerSchema.validate(sample_pdf, lazy=True)
        logger.info(
            "Bronze schema validation passed on %d-row sample.", len(sample_pdf)
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Bronze schema validation failed: %s", exc)
        raise
