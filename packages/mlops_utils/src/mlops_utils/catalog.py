"""
mlops_utils.catalog
~~~~~~~~~~~~~~~~~~~
Unity Catalog setup helpers: create/use catalogs and schemas, manage
permissions, and reset schemas for CI/CD or demo purposes.

All functions are idempotent – calling them on an already-existing resource
is safe.

Public API
----------
::

    from mlops_utils.catalog import setup_catalog_and_schema, drop_and_recreate_schema
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def setup_catalog_and_schema(
    spark: "pyspark.sql.SparkSession",  # type: ignore[name-defined]  # noqa: F821
    catalog: str,
    schema: str,
    *,
    volume_name: Optional[str] = None,
    grant_all_to: Optional[str] = None,
) -> None:
    """Create (if missing) and USE a Unity Catalog catalog + schema.

    Parameters
    ----------
    spark:
        Active SparkSession (must have Unity Catalog enabled).
    catalog:
        Target catalog name.  Must **not** be ``hive_metastore`` or
        ``spark_catalog`` (Unity Catalog is required).
    schema:
        Target schema / database name inside *catalog*.
    volume_name:
        Optional Unity Catalog Volume to create inside the schema.
    grant_all_to:
        Optional principal (group name or user e-mail) to grant
        ``CREATE`` and ``USAGE`` on the schema.

    Raises
    ------
    ValueError
        If *catalog* is one of the legacy non-UC catalog names.
    """
    if catalog in {"hive_metastore", "spark_catalog"}:
        raise ValueError(
            f"Catalog '{catalog}' is not a Unity Catalog. "
            "Please provide a Unity Catalog name."
        )

    # Create catalog if it doesn't exist
    spark.sql(f"CREATE CATALOG IF NOT EXISTS `{catalog}`")
    spark.sql(f"USE CATALOG `{catalog}`")
    logger.info("Using catalog '%s'.", catalog)

    # Create schema
    spark.sql(f"CREATE DATABASE IF NOT EXISTS `{schema}`")
    spark.sql(f"USE `{catalog}`.`{schema}`")
    logger.info("Using schema '%s.%s'.", catalog, schema)

    # Optional: create volume
    if volume_name:
        spark.sql(f"CREATE VOLUME IF NOT EXISTS `{catalog}`.`{schema}`.`{volume_name}`")
        logger.info("Volume '%s.%s.%s' ensured.", catalog, schema, volume_name)

    # Optional: grant permissions
    if grant_all_to:
        _safe_sql(
            spark,
            f"GRANT CREATE, USAGE ON DATABASE `{catalog}`.`{schema}` TO `{grant_all_to}`",
        )
        logger.info("Granted CREATE/USAGE on '%s.%s' to '%s'.", catalog, schema, grant_all_to)


def drop_and_recreate_schema(
    spark: "pyspark.sql.SparkSession",  # type: ignore[name-defined]  # noqa: F821
    catalog: str,
    schema: str,
    *,
    volume_name: Optional[str] = None,
) -> None:
    """Drop (CASCADE) and recreate a schema – for CI/CD reset or demo purposes.

    .. warning::
        This **permanently deletes** all tables and data in *schema*.
        Only call in non-production environments.

    Parameters
    ----------
    spark:
        Active SparkSession.
    catalog:
        Target catalog.
    schema:
        Schema to drop and recreate.
    volume_name:
        Optional volume to drop before dropping the schema.
    """
    logger.warning(
        "Dropping schema '%s.%s' (CASCADE) – all data will be lost!", catalog, schema
    )
    if volume_name:
        _safe_sql(
            spark,
            f"DROP VOLUME IF EXISTS `{catalog}`.`{schema}`.`{volume_name}`",
        )
    spark.sql(f"DROP SCHEMA IF EXISTS `{catalog}`.`{schema}` CASCADE")
    setup_catalog_and_schema(spark, catalog, schema, volume_name=volume_name)


def set_table_owner(
    spark: "pyspark.sql.SparkSession",  # type: ignore[name-defined]  # noqa: F821
    full_table_name: str,
    owner: str,
) -> None:
    """Transfer ownership of a table to *owner*."""
    _safe_sql(spark, f"ALTER TABLE {full_table_name} OWNER TO `{owner}`")
    logger.debug("Transferred ownership of '%s' to '%s'.", full_table_name, owner)


def grant_table_privileges(
    spark: "pyspark.sql.SparkSession",  # type: ignore[name-defined]  # noqa: F821
    full_table_name: str,
    principal: str,
    privileges: list[str] | None = None,
) -> None:
    """Grant one or more privileges on *full_table_name* to *principal*.

    Parameters
    ----------
    privileges:
        List of SQL privilege strings, e.g. ``["SELECT", "MODIFY"]``.
        Defaults to ``["ALL PRIVILEGES"]``.
    """
    privs = ", ".join(privileges) if privileges else "ALL PRIVILEGES"
    _safe_sql(spark, f"GRANT {privs} ON TABLE {full_table_name} TO `{principal}`")
    logger.debug("Granted %s on '%s' to '%s'.", privs, full_table_name, principal)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_sql(spark: "pyspark.sql.SparkSession", statement: str) -> None:  # type: ignore[name-defined]  # noqa: F821
    """Execute *statement*, logging and swallowing non-critical errors."""
    try:
        spark.sql(statement)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SQL warning (non-fatal): %s | Statement: %s", exc, statement)
