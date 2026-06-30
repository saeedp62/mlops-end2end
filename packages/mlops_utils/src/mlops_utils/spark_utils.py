"""
mlops_utils.spark_utils
~~~~~~~~~~~~~~~~~~~~~~~
SparkSession helpers that work transparently in three environments:

1. **Local / CI**   – creates a minimal standalone SparkSession for unit tests.
2. **Databricks**   – returns the pre-existing ``spark`` binding injected by the
                      Databricks runtime (no new session is created).
3. **Databricks Connect** – handled automatically by ``pyspark``.

Public API
----------
::

    from mlops_utils.spark_utils import get_or_create_spark, table_exists, wait_for_table
"""

from __future__ import annotations

import time

from mlops_utils.logger import get_logger

logger = get_logger(__name__)


def get_or_create_spark(
    app_name: str = "mlops",
    *,
    master: str = "local[*]",
    extra_configs: dict[str, str] | None = None,
) -> pyspark.sql.SparkSession:  # type: ignore[name-defined]  # noqa: F821
    """Return a SparkSession, reusing any session that already exists.

    On Databricks the pre-existing ``spark`` session is always returned
    (``getOrCreate`` is a no-op there).  In local/CI environments a minimal
    standalone session is created.

    Parameters
    ----------
    app_name:
        Application name shown in Spark UI / logs.
    master:
        Spark master URL.  Ignored when running on Databricks.
    extra_configs:
        Additional ``spark.conf.set`` key-value pairs applied after session
        creation (e.g. ``{"spark.sql.ansi.enabled": "false"}``).

    Returns
    -------
    pyspark.sql.SparkSession
    """
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        # Needed for pandas-on-Spark (koalas) API
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.ansi.enabled", "false")
    )

    spark = builder.getOrCreate()

    if extra_configs:
        for k, v in extra_configs.items():
            spark.conf.set(k, v)

    logger.debug(
        "SparkSession '%s' obtained (version=%s, master=%s)",
        app_name,
        spark.version,
        spark.sparkContext.master,
    )
    return spark


def table_exists(spark: pyspark.sql.SparkSession, full_table_name: str) -> bool:  # type: ignore[name-defined]  # noqa: F821
    """Return ``True`` if *full_table_name* exists in the metastore.

    Parameters
    ----------
    spark:
        Active SparkSession.
    full_table_name:
        Three-part Unity Catalog name ``catalog.schema.table`` or two-part
        ``schema.table`` name.
    """
    try:
        return spark.catalog.tableExists(full_table_name)
    except Exception:  # noqa: BLE001
        return False


def wait_for_table(
    spark: pyspark.sql.SparkSession,  # type: ignore[name-defined]  # noqa: F821
    full_table_name: str,
    *,
    timeout_seconds: int = 120,
    poll_interval_seconds: float = 1.0,
) -> None:
    """Block until *full_table_name* exists and is non-empty.

    Useful when a streaming pipeline writes to a table that a downstream
    notebook needs to read.

    Parameters
    ----------
    spark:
        Active SparkSession.
    full_table_name:
        Fully-qualified table name.
    timeout_seconds:
        Maximum number of seconds to wait before raising ``TimeoutError``.
    poll_interval_seconds:
        Sleep interval between existence checks.

    Raises
    ------
    TimeoutError
        If the table does not appear within *timeout_seconds*.
    """
    elapsed = 0.0
    while elapsed < timeout_seconds:
        if table_exists(spark, full_table_name):
            try:
                count = spark.table(full_table_name).count()
                if count > 0:
                    logger.info(
                        "Table '%s' is ready (%d rows, waited %.1fs).",
                        full_table_name,
                        count,
                        elapsed,
                    )
                    return
            except Exception:  # noqa: BLE001
                pass
        time.sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds

    raise TimeoutError(
        f"Table '{full_table_name}' did not become non-empty within {timeout_seconds}s."
    )
