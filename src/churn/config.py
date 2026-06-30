"""
churn.config
~~~~~~~~~~~~
Configuration dataclass for the churn pipeline.

Config values are loaded from a YAML/JSON file via ``mlops_utils.config_loader``
and mapped into a typed ``ChurnConfig`` dataclass.  Environment variables
prefixed with ``MLOPS_`` override file values.

Data source strategy
--------------------
Three source types are supported, controlled by ``data_source.type``:

``unity_catalog_table`` *(production default)*
    Read directly from a fully-qualified Delta table in another LOB catalog.
    Set ``data_source.source_table`` to the three-part name::

        data_source:
          type: unity_catalog_table
          source_table: lob_catalog.lob_schema.customers

``volume_csv`` *(demo / dev)*
    Load a CSV file stored in a Unity Catalog Volume::

        data_source:
          type: volume_csv
          volume_path: /Volumes/main/shared_data/telco/Telco-Customer-Churn.csv
          csv_options:
            header: "true"
            inferSchema: "true"

``http_csv`` *(local unit tests only – not for Databricks)*
    Download from a public HTTP URL (falls back to an S3 mirror)::

        data_source:
          type: http_csv
          url: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

Full example YAML (new multi-schema layout)::

    catalog: lighthouse_bkk6_analytics
    schemas:
      training_datasets: training_datasets   # bronze + label tables
      offline_features:  offline_features    # feature table
      online_features:   online_features     # online serving
      ml_models:         ml_models           # registered models
      model_predictions: model_predictions   # batch inference outputs
      ml_monitoring:     ml_monitoring       # monitoring metrics
    bronze_table: advanced_churn_bronze_customers
    feature_table: advanced_churn_feature_table
    label_table: advanced_churn_label_table
    model_name: advanced_mlops_churn
    label_col: churn
    pos_label: "Yes"
    primary_keys:
      - customer_id
      - transaction_ts
    timeseries_col: transaction_ts
    train_ratio: 0.8
    rng_seed: 2025
    experiment_path: /Shared/lighthouse_bkk6_analytics/experiments/mlops-end2end
    experiment_name: advanced_mlops_churn_experiment
    data_source:
      type: volume_csv
      volume_path: /Volumes/lighthouse_bkk6_analytics/training_datasets/telco/Telco-Customer-Churn.csv
    online_store:
      enabled: false
      backend: databricks
      endpoint_name: churn_online_features

Usage::

    from churn.config import ChurnConfig, load_churn_config

    cfg = load_churn_config("configs/dev.yaml")
    print(cfg.full_feature_table)   # "lighthouse_bkk6_analytics.offline_features.advanced_churn_feature_table"
    print(cfg.full_bronze_table)    # "lighthouse_bkk6_analytics.training_datasets.advanced_churn_bronze_customers"
    print(cfg.data_source.type)     # "volume_csv"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from mlops_utils.config_loader import load_config, merge_configs


# Valid source type identifiers
_SOURCE_TYPES = frozenset({"unity_catalog_table", "volume_csv", "http_csv"})


@dataclass
class DataSourceConfig:
    """Data source strategy for the bronze ingestion layer.

    Attributes
    ----------
    type:
        One of ``"unity_catalog_table"`` (production), ``"volume_csv"`` (demo),
        or ``"http_csv"`` (local unit tests only).
    source_table:
        Fully-qualified UC table name used when ``type == "unity_catalog_table"``.
        Supports cross-catalog reads from any LOB, e.g.
        ``"retail_catalog.crm.customers"``.
    volume_path:
        Absolute path to a CSV file inside a Unity Catalog Volume when
        ``type == "volume_csv"``, e.g.
        ``"/Volumes/main/shared_data/telco/Telco-Customer-Churn.csv"``.
    url:
        HTTP URL used when ``type == "http_csv"`` (local tests / internet-enabled
        envs).  A fallback S3 mirror is tried automatically.
    csv_options:
        Extra Spark CSV reader options applied for ``volume_csv`` and
        ``http_csv`` sources (e.g. ``{"header": "true", "inferSchema": "true"}``).
    normalize_columns:
        When ``True`` (default), apply snake_case normalisation and IBM-dataset
        column renames to the raw DataFrame before any downstream processing.
        Set to ``False`` if the source table already has normalised column names.
    """

    type: str = "volume_csv"          # unity_catalog_table | volume_csv | http_csv
    source_table: str = ""            # used when type == unity_catalog_table
    volume_path: str = ""             # used when type == volume_csv
    url: str = ""                     # used when type == http_csv
    csv_options: dict[str, str] = field(
        default_factory=lambda: {"header": "true", "inferSchema": "true"}
    )
    normalize_columns: bool = True

    def __post_init__(self) -> None:
        if self.type not in _SOURCE_TYPES:
            raise ValueError(
                f"data_source.type='{self.type}' is not valid. "
                f"Choose from: {sorted(_SOURCE_TYPES)}"
            )


@dataclass
class SchemaConfig:
    """Unity Catalog schema names for each MLOps data lifecycle stage.

    Schema naming follows business-purpose conventions (lowercase snake_case).
    The same schema names are used across all environments; environment
    isolation is achieved via catalog/workspace separation.

    Attributes
    ----------
    training_datasets:
        Stores bronze ingestion tables and curated label/split datasets
        used for model training and validation.
    offline_features:
        Stores historical feature tables.  Supports model training and
        batch inference use cases.
    online_features:
        Stores features required for real-time, low-latency model serving.
    ml_models:
        Stores registered model assets and supports lifecycle management.
    model_predictions:
        Stores batch inference outputs and model scoring results.
    ml_monitoring:
        Stores model performance metrics, drift monitoring, and operational
        ML metrics.
    """

    training_datasets: str = "training_datasets"
    offline_features:  str = "offline_features"
    online_features:   str = "online_features"
    ml_models:         str = "ml_models"
    model_predictions: str = "model_predictions"
    ml_monitoring:     str = "ml_monitoring"

    @property
    def all_schemas(self) -> list[str]:
        """Return all schema names (used by the bootstrap utility)."""
        return [
            self.training_datasets,
            self.offline_features,
            self.online_features,
            self.ml_models,
            self.model_predictions,
            self.ml_monitoring,
        ]

    def as_comments_dict(self) -> dict[str, str]:
        """Return ``{schema_name: comment}`` for use with ``mlops_utils.catalog.ensure_mlops_schemas``.

        This keeps schema descriptions co-located with the config (single source
        of truth) while keeping ``mlops_utils`` decoupled from any use-case.

        Example::

            from mlops_utils.catalog import ensure_mlops_schemas

            ensure_mlops_schemas(spark, cfg.catalog, cfg.schemas.as_comments_dict())
        """
        return {
            self.training_datasets: (
                "Bronze ingestion tables and curated label/split datasets "
                "used for model training and validation."
            ),
            self.offline_features: (
                "Historical feature tables supporting model training "
                "and batch inference use cases."
            ),
            self.online_features: (
                "Features required for real-time, low-latency model serving."
            ),
            self.ml_models: (
                "Model-related assets supporting lifecycle management "
                "and model registration."
            ),
            self.model_predictions: (
                "Batch inference outputs and model scoring results."
            ),
            self.ml_monitoring: (
                "Model performance metrics, drift monitoring, and "
                "operational ML metrics."
            ),
        }


@dataclass
class OnlineStoreConfig:
    """Online Feature Store serving configuration."""

    enabled: bool = False
    backend: str = "databricks"      # databricks | dynamodb | cosmosdb
    endpoint_name: str = "churn_online_features"
    # Additional backend-specific keys are stored here
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChurnConfig:
    """All configuration for the churn MLOps pipeline.

    Attributes
    ----------
    catalog:
        Unity Catalog name.  All schemas below are created inside this catalog.
        Environment isolation is achieved by pointing dev/staging/prod at
        different catalogs (or workspaces), NOT by varying schema names.
    schemas:
        Per-lifecycle-stage schema names (see :class:`SchemaConfig`).
        The same schema names are used in every environment.
    bronze_table:
        Name of the raw (bronze) ingestion table (no catalog/schema prefix).
        Lands in ``catalog.schemas.training_datasets``.
    feature_table:
        Name of the offline feature table (no catalog/schema prefix).
        Lands in ``catalog.schemas.offline_features``.
    label_table:
        Name of the label / train-test split table (no catalog/schema prefix).
        Lands in ``catalog.schemas.training_datasets``.
    model_name:
        Registered model name in Unity Catalog.
        Lands in ``catalog.schemas.ml_models``.
    label_col:
        Target label column name.
    pos_label:
        Positive class value for binary metrics (``"Yes"``).
    primary_keys:
        Primary key column(s) for the feature table.
    timeseries_col:
        Timestamp column used for point-in-time lookups.
    train_ratio:
        Fraction of data used for training (remainder goes to test).
    rng_seed:
        Random seed for reproducibility.
    experiment_path:
        Workspace path for the MLflow experiment folder.
    experiment_name:
        Name of the MLflow experiment.
    data_source:
        Data source strategy – cross-LOB UC table, Volume CSV, or HTTP CSV.
    online_store:
        Online Feature Store configuration.
    """

    catalog: str = "lighthouse_bkk6_analytics"
    schemas: SchemaConfig = field(default_factory=SchemaConfig)
    bronze_table: str = "advanced_churn_bronze_customers"
    feature_table: str = "advanced_churn_feature_table"
    label_table: str = "advanced_churn_label_table"
    model_name: str = "advanced_mlops_churn"
    label_col: str = "churn"
    pos_label: str = "Yes"
    primary_keys: list[str] = field(default_factory=lambda: ["customer_id", "transaction_ts"])
    timeseries_col: str = "transaction_ts"
    train_ratio: float = 0.8
    rng_seed: int = 2025
    experiment_path: str = "/Shared/lighthouse_bkk6_analytics/experiments/mlops-end2end"
    experiment_name: str = "advanced_mlops_churn_experiment"
    data_source: DataSourceConfig = field(default_factory=DataSourceConfig)
    online_store: OnlineStoreConfig = field(default_factory=OnlineStoreConfig)

    # -----------------------------------------------------------------------
    # Derived properties — each table resolves to its correct schema
    # -----------------------------------------------------------------------

    @property
    def full_bronze_table(self) -> str:
        """Fully-qualified bronze table name (lives in training_datasets schema)."""
        return f"{self.catalog}.{self.schemas.training_datasets}.{self.bronze_table}"

    @property
    def full_feature_table(self) -> str:
        """Fully-qualified offline feature table name (lives in offline_features schema)."""
        return f"{self.catalog}.{self.schemas.offline_features}.{self.feature_table}"

    @property
    def full_label_table(self) -> str:
        """Fully-qualified label table name (lives in training_datasets schema)."""
        return f"{self.catalog}.{self.schemas.training_datasets}.{self.label_table}"

    @property
    def full_model_name(self) -> str:
        """Fully-qualified model name registered in Unity Catalog (lives in ml_models schema)."""
        return f"{self.catalog}.{self.schemas.ml_models}.{self.model_name}"

    @property
    def full_experiment_name(self) -> str:
        """Full MLflow experiment path."""
        return f"{self.experiment_path}/{self.experiment_name}"

    @property
    def label_table_pk_constraint(self) -> str:
        """Primary key constraint name for the label table."""
        return f"{self.label_table}_pk"

    @property
    def feature_table_pk_constraint(self) -> str:
        """Primary key constraint name for the feature table."""
        return f"{self.feature_table}_pk"


# ---------------------------------------------------------------------------
# Loader function
# ---------------------------------------------------------------------------

def load_churn_config(
    config_path: str | Path,
    *,
    base_path: Optional[str | Path] = None,
) -> ChurnConfig:
    """Load a YAML/JSON config file and return a validated ``ChurnConfig``.

    Parameters
    ----------
    config_path:
        Path to the environment-specific YAML/JSON file
        (e.g. ``"configs/dev.yaml"``).
    base_path:
        Optional base config that is deep-merged *before* the env-specific
        config (env-specific values win).

    Returns
    -------
    ChurnConfig
    """
    if base_path:
        raw = merge_configs(base_path, config_path)
    else:
        raw = load_config(config_path)

    # Parse schemas sub-dict.
    # 'schemas:' block is required for the multi-schema layout.
    # If absent, all six schemas fall back to their SchemaConfig dataclass defaults.
    schemas_raw = raw.pop("schemas", {})
    schemas_cfg = SchemaConfig(
        training_datasets=schemas_raw.get("training_datasets", SchemaConfig.training_datasets),
        offline_features=schemas_raw.get("offline_features", SchemaConfig.offline_features),
        online_features=schemas_raw.get("online_features", SchemaConfig.online_features),
        ml_models=schemas_raw.get("ml_models", SchemaConfig.ml_models),
        model_predictions=schemas_raw.get("model_predictions", SchemaConfig.model_predictions),
        ml_monitoring=schemas_raw.get("ml_monitoring", SchemaConfig.ml_monitoring),
    )

    # Parse data_source sub-dict
    ds_raw = raw.pop("data_source", {})
    ds_cfg = DataSourceConfig(
        type=ds_raw.get("type", "volume_csv"),
        source_table=ds_raw.get("source_table", ""),
        volume_path=ds_raw.get("volume_path", ""),
        url=ds_raw.get("url", ""),
        csv_options=ds_raw.get("csv_options", {"header": "true", "inferSchema": "true"}),
        normalize_columns=ds_raw.get("normalize_columns", True),
    )

    # Parse online_store sub-dict
    online_raw = raw.pop("online_store", {})
    online_cfg = OnlineStoreConfig(
        enabled=online_raw.get("enabled", False),
        backend=online_raw.get("backend", "databricks"),
        endpoint_name=online_raw.get("endpoint_name", "churn_online_features"),
        extra={k: v for k, v in online_raw.items() if k not in {"enabled", "backend", "endpoint_name"}},
    )

    # Primary keys may come as a list or comma-separated string
    pk_raw = raw.get("primary_keys", ["customer_id", "transaction_ts"])
    if isinstance(pk_raw, str):
        pk_raw = [k.strip() for k in pk_raw.split(",")]

    return ChurnConfig(
        catalog=raw.get("catalog", "lighthouse_bkk6_analytics"),
        schemas=schemas_cfg,
        bronze_table=raw.get("bronze_table", "advanced_churn_bronze_customers"),
        feature_table=raw.get("feature_table", "advanced_churn_feature_table"),
        label_table=raw.get("label_table", "advanced_churn_label_table"),
        model_name=raw.get("model_name", "advanced_mlops_churn"),
        label_col=raw.get("label_col", "churn"),
        pos_label=raw.get("pos_label", "Yes"),
        primary_keys=pk_raw,
        timeseries_col=raw.get("timeseries_col", "transaction_ts"),
        train_ratio=float(raw.get("train_ratio", 0.8)),
        rng_seed=int(raw.get("rng_seed", 2025)),
        experiment_path=raw.get(
            "experiment_path",
            "/Shared/lighthouse_bkk6_analytics/experiments/mlops-end2end",
        ),
        experiment_name=raw.get("experiment_name", "advanced_mlops_churn_experiment"),
        data_source=ds_cfg,
        online_store=online_cfg,
    )
