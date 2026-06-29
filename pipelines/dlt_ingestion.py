"""
pipelines/dlt_ingestion.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Delta Live Tables (DLT) pipeline: Bronze layer ingestion.

Source strategy is controlled entirely by the ``data_source`` section of the
YAML config – no code changes are needed when switching between environments:

* **Production**  (``unity_catalog_table``) – reads directly from a LOB Unity
  Catalog table across catalog boundaries using the pipeline service principal's
  UC identity.

* **Demo / Dev**  (``volume_csv``) – reads a CSV file that has been uploaded to
  a Unity Catalog Volume path.

* **Local tests** (``http_csv``) – downloads from a public HTTP URL with an S3
  fallback.  Not intended for DLT; use ``volume_csv`` or ``unity_catalog_table``
  in all Databricks pipelines.

Deploy as a DLT pipeline targeting this file.  Pass the config path as a
pipeline parameter in the DLT UI, via Databricks Asset Bundles, or via the
Databricks CLI:

    databricks pipelines create --settings pipelines/bundle.yaml

DLT pipeline parameters (set in the DLT UI or ``bundle.yaml``):
    config_path: /Volumes/main/dbdemos_mlops/configs/prod.yaml

Individual config values can be overridden via cluster environment variables:
    MLOPS_CATALOG=prod_catalog
    MLOPS_DB=dbdemos_mlops

Grant required for cross-catalog reads (run once as catalog owner):
    GRANT SELECT ON TABLE <source_catalog>.<schema>.<table>
    TO `<pipeline-service-principal>@company.com`;
"""

from __future__ import annotations

import dlt  # type: ignore[import]
import pyspark.sql.functions as F

from churn.config import load_churn_config
from churn.data_source import get_source_dataframe


# ---------------------------------------------------------------------------
# Load config from DLT pipeline parameter (falls back to env-var defaults)
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

    # Env-vars (MLOPS_CATALOG, MLOPS_DB, …) are applied by ChurnConfig defaults
    from churn.config import ChurnConfig
    return ChurnConfig()


cfg = _get_config()

# Human-readable source description used in table comments
_SOURCE_DESCRIPTION = {
    "unity_catalog_table": f"cross-catalog Unity Catalog table '{cfg.data_source.source_table}'",
    "volume_csv":          f"Volume CSV at '{cfg.data_source.volume_path}'",
    "http_csv":            "public HTTP CSV (IBM Telco dataset)",
}.get(cfg.data_source.type, cfg.data_source.type)


# ---------------------------------------------------------------------------
# Bronze table – raw source data
# ---------------------------------------------------------------------------

@dlt.table(
    name=cfg.bronze_table,
    comment=(
        f"Raw customer data ingested from {_SOURCE_DESCRIPTION}. "
        f"Managed as Unity Catalog table {cfg.full_bronze_table}."
    ),
    table_properties={
        "quality": "bronze",
        "pipelines.reset.allowed": "true",
        "data_source.type": cfg.data_source.type,
    },
)
def bronze_customers():  # type: ignore[no-untyped-def]
    """Ingest the raw source data using the config-driven source strategy.

    The ``get_source_dataframe`` dispatcher routes to the correct reader:
    - ``unity_catalog_table`` → ``spark.table(<source_table>)``
    - ``volume_csv``          → ``spark.read.csv(<volume_path>)``
    - ``http_csv``            → HTTP download → ``spark.createDataFrame``

    Column names are normalised to snake_case when
    ``data_source.normalize_columns: true`` (default).
    """
    raw_df = get_source_dataframe(
        spark,  # type: ignore[name-defined]  # noqa: F821
        cfg,
    )
    return raw_df.withColumn("_ingested_at", F.current_timestamp())


# ---------------------------------------------------------------------------
# Bronze validated – data quality gate
# ---------------------------------------------------------------------------

@dlt.expect_or_drop("customer_id_not_null", "customer_id IS NOT NULL")
@dlt.expect("monthly_charges_non_negative", "monthly_charges >= 0 OR monthly_charges IS NULL")
@dlt.expect("total_charges_non_negative", "total_charges >= 0 OR total_charges IS NULL")
@dlt.expect("valid_churn_label", "churn IN ('Yes', 'No') OR churn IS NULL")
@dlt.expect("valid_contract_type",
            "contract IN ('Month-to-month', 'One year', 'Two year') OR contract IS NULL")
@dlt.table(
    name=f"{cfg.bronze_table}_validated",
    comment=(
        "Bronze customer data with DLT data quality expectations applied. "
        "Rows failing the 'customer_id_not_null' check are dropped; "
        "remaining violations are tracked as metrics."
    ),
    table_properties={"quality": "bronze_validated"},
)
def bronze_customers_validated():  # type: ignore[no-untyped-def]
    """Apply DLT data quality expectations to the raw bronze table."""
    return dlt.read(cfg.bronze_table)
