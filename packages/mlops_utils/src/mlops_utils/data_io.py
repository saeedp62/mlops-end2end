"""
mlops_utils.data_io
~~~~~~~~~~~~~~~~~~~
Generic Delta Lake read / write / upsert wrappers.

Design goals
------------
* Accept **fully-qualified** table names (``catalog.schema.table``) so the
  caller never has to SET CATALOG / USE DATABASE first.
* Expose a simple, opinionated API while still allowing callers to pass
  arbitrary Spark options.
* All functions are pure (no side effects beyond writing to Delta).

Public API
----------
::

    from mlops_utils.data_io import read_delta, write_delta, upsert_delta
"""

from __future__ import annotations

from mlops_utils.logger import get_logger
from typing import Optional

logger = get_logger(__name__)


def read_delta(
    spark: "pyspark.sql.SparkSession",  # type: ignore[name-defined]  # noqa: F821
    full_table_name: str,
    *,
    filters: Optional[str] = None,
    columns: Optional[list[str]] = None,
) -> "pyspark.sql.DataFrame":  # type: ignore[name-defined]  # noqa: F821
    """Read a Delta table into a Spark DataFrame.

    Parameters
    ----------
    spark:
        Active SparkSession.
    full_table_name:
        Fully-qualified table name, e.g. ``"main.dbdemos_mlops.my_table"``.
    filters:
        Optional SQL WHERE clause string applied via ``.filter()``,
        e.g. ``"split = 'train' AND tenure > 0"``.
    columns:
        Optional list of column names to select.  ``None`` returns all columns.

    Returns
    -------
    pyspark.sql.DataFrame
    """
    logger.debug("Reading table '%s' (filters=%r, columns=%r).", full_table_name, filters, columns)
    df = spark.table(full_table_name)

    if filters:
        df = df.filter(filters)

    if columns:
        df = df.select(*columns)

    return df


def write_delta(
    df: "pyspark.sql.DataFrame",  # type: ignore[name-defined]  # noqa: F821
    full_table_name: str,
    *,
    mode: str = "overwrite",
    partition_by: Optional[list[str]] = None,
    options: Optional[dict[str, str]] = None,
    comment: Optional[str] = None,
) -> None:
    """Write a DataFrame to a Delta table.

    Parameters
    ----------
    df:
        Source DataFrame.
    full_table_name:
        Destination table (will be created if it doesn't exist).
    mode:
        Write mode – ``"overwrite"``, ``"append"``, ``"ignore"``, or
        ``"error"`` (Spark default).
    partition_by:
        Optional list of column names to partition by.
    options:
        Additional ``DataFrameWriter`` options, e.g.
        ``{"overwriteSchema": "true"}``.
    comment:
        Optional table comment set after write via ``COMMENT ON TABLE``.
    """
    defaults: dict[str, str] = {"mergeSchema": "true"}
    if mode == "overwrite":
        defaults["overwriteSchema"] = "true"

    merged_options = {**defaults, **(options or {})}

    writer = df.write.format("delta").mode(mode)
    for k, v in merged_options.items():
        writer = writer.option(k, v)

    if partition_by:
        writer = writer.partitionBy(*partition_by)

    writer.saveAsTable(full_table_name)
    logger.info("Wrote %s rows to '%s' (mode=%s).", df.count() if logger.isEnabledFor(logging.DEBUG) else "?", full_table_name, mode)

    if comment:
        # Requires an active SparkSession – grab from DataFrame
        spark = df.sparkSession
        spark.sql(f"COMMENT ON TABLE {full_table_name} IS '{comment}'")


def upsert_delta(
    df: "pyspark.sql.DataFrame",  # type: ignore[name-defined]  # noqa: F821
    full_table_name: str,
    merge_keys: list[str],
    *,
    update_columns: Optional[list[str]] = None,
    insert_all: bool = True,
) -> None:
    """MERGE (upsert) *df* into an existing Delta table.

    Uses the Delta Lake ``MERGE INTO`` syntax.  The table must already exist.

    Parameters
    ----------
    df:
        Source DataFrame with new / updated rows.
    full_table_name:
        Target Delta table (must already exist).
    merge_keys:
        Column name(s) used as join keys for MATCH detection.
    update_columns:
        Columns to update on match.  ``None`` updates all columns.
    insert_all:
        When ``True``, inserts the source row if no match is found.
    """
    from delta.tables import DeltaTable  # type: ignore[import]

    spark = df.sparkSession
    delta_table = DeltaTable.forName(spark, full_table_name)

    # Build MERGE condition
    match_condition = " AND ".join(
        f"target.{k} = source.{k}" for k in merge_keys
    )

    merge_builder = (
        delta_table.alias("target")
        .merge(df.alias("source"), match_condition)
    )

    # Update matched rows
    if update_columns:
        update_map = {c: f"source.{c}" for c in update_columns}
        merge_builder = merge_builder.whenMatchedUpdate(set=update_map)
    else:
        merge_builder = merge_builder.whenMatchedUpdateAll()

    # Insert unmatched rows
    if insert_all:
        merge_builder = merge_builder.whenNotMatchedInsertAll()

    merge_builder.execute()
    logger.info("Upserted into '%s' on keys %s.", full_table_name, merge_keys)


def add_primary_key_constraint(
    spark: "pyspark.sql.SparkSession",  # type: ignore[name-defined]  # noqa: F821
    full_table_name: str,
    constraint_name: str,
    key_columns: list[str],
    *,
    timeseries_columns: Optional[list[str]] = None,
) -> None:
    """Add a PRIMARY KEY constraint to a Delta table (required for UC Feature Tables).

    Parameters
    ----------
    full_table_name:
        Fully-qualified table name.
    constraint_name:
        Name of the constraint (must be unique within the table).
    key_columns:
        Primary key column(s).
    timeseries_columns:
        Subset of *key_columns* to mark as ``TIMESERIES`` for point-in-time
        Feature Store lookups.
    """
    # Mark key columns NOT NULL (required for PK constraints in UC)
    for col in key_columns:
        _safe_sql(spark, f"ALTER TABLE {full_table_name} ALTER COLUMN {col} SET NOT NULL")

    # Drop existing constraint if present
    _safe_sql(
        spark,
        f"ALTER TABLE {full_table_name} DROP CONSTRAINT IF EXISTS {constraint_name}",
    )

    # Build PK definition
    pk_cols = []
    for col in key_columns:
        if timeseries_columns and col in timeseries_columns:
            pk_cols.append(f"{col} TIMESERIES")
        else:
            pk_cols.append(col)

    pk_def = ", ".join(pk_cols)
    spark.sql(
        f"ALTER TABLE {full_table_name} ADD CONSTRAINT {constraint_name} "
        f"PRIMARY KEY ({pk_def})"
    )
    logger.info("Added PK constraint '%s' on '%s'.", constraint_name, full_table_name)


def _safe_sql(spark: "pyspark.sql.SparkSession", statement: str) -> None:  # type: ignore[name-defined]  # noqa: F821
    try:
        spark.sql(statement)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SQL warning (non-fatal): %s | Statement: %s", exc, statement)
