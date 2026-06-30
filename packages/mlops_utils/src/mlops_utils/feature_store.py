"""
mlops_utils.feature_store
~~~~~~~~~~~~~~~~~~~~~~~~~
Thin, typed wrappers around ``databricks.feature_engineering.FeatureEngineeringClient``
for both **offline (batch)** and **online (real-time)** feature serving.

Design goals
------------
* All functions accept ``fe`` (FeatureEngineeringClient) as a dependency-injected
  argument so they are mockable in unit tests.
* Online-store helpers are gated behind a lazy import so the module can be
  imported without ``databricks-feature-engineering`` installed (e.g. in unit test
  environments that use a mock).
* Both ``FeatureLookup`` and ``FeatureFunction`` specs are supported.

Public API
----------
::

    from mlops_utils.feature_store import (
        FeatureStoreManager,
        create_or_replace_feature_table,
        write_feature_table,
        build_training_set,
        score_batch_wrapper,
        publish_to_online_store,
    )
"""

from __future__ import annotations

from mlops_utils.logger import get_logger
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from mlops_utils._retry import with_retry

if TYPE_CHECKING:
    # These are only imported for type checking; the actual runtime imports
    # are deferred so the module is importable without Databricks SDK.
    from databricks.feature_engineering import FeatureEngineeringClient
    from pyspark.sql import DataFrame

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# FeatureStoreManager Facade
# ---------------------------------------------------------------------------

@dataclass
class FeatureStoreManager:
    """Config-aware facade over FeatureEngineeringClient.

    Binds a config (or any object with .catalog + .schemas) so callers
    never have to manually compose fully-qualified names.

    Usage
    -----
    ::

        from mlops_utils.feature_store import FeatureStoreManager

        fsm = FeatureStoreManager.from_config(cfg)

        fsm.create_or_replace(cfg.feature_table, df, primary_keys=cfg.primary_keys)
        fsm.write(cfg.feature_table, df)
        training_set = fsm.build_training_set(labels_df, lookups, label_col="churn")
    """

    fe: "FeatureEngineeringClient"
    catalog: str
    offline_schema: str
    online_schema: str
    dry_run: bool = False

    @classmethod
    def from_config(cls, cfg: Any, dry_run: bool = False) -> "FeatureStoreManager":
        from databricks.feature_engineering import FeatureEngineeringClient
        return cls(
            fe=FeatureEngineeringClient(),
            catalog=cfg.catalog,
            offline_schema=cfg.schemas.offline_features,
            online_schema=cfg.schemas.online_features,
            dry_run=dry_run,
        )

    def fqn(self, table_name: str, *, use_online_schema: bool = False) -> str:
        """Return fully-qualified name: catalog.schema.table_name."""
        s = self.online_schema if use_online_schema else self.offline_schema
        return f"{self.catalog}.{s}.{table_name}"

    def create_or_replace(
        self,
        table_name: str,
        df: "DataFrame",
        primary_keys: list[str],
        *,
        timeseries_columns: Optional[str | list[str]] = None,
        description: str = "",
        tags: Optional[dict[str, str]] = None,
        raise_on_drop_error: bool = False,
    ) -> Any:
        if self.dry_run:
            logger.info("[DRY RUN] Would create/replace feature table '%s'", self.fqn(table_name))
            return None
            
        return create_or_replace_feature_table(
            self.fe,
            name=self.fqn(table_name),
            df=df,
            primary_keys=primary_keys,
            timeseries_columns=timeseries_columns,
            description=description,
            tags=tags,
            raise_on_drop_error=raise_on_drop_error,
        )

    def write(self, table_name: str, df: "DataFrame", *, mode: str = "merge") -> None:
        if self.dry_run:
            logger.info("[DRY RUN] Would write feature table '%s' (mode=%s)", self.fqn(table_name), mode)
            return
            
        write_feature_table(self.fe, name=self.fqn(table_name), df=df, mode=mode)

    def reset_and_write(self, table_name: str, df: "DataFrame", primary_keys: list[str], **kwargs: Any) -> None:
        """Drop-and-recreate then write — the safe 'overwrite' pattern."""
        self.create_or_replace(table_name, df, primary_keys, **kwargs)
        self.write(table_name, df)

    def build_training_set(
        self,
        labels_df: "DataFrame",
        feature_lookups: list[Any],
        label_col: str,
        *,
        exclude_columns: Optional[list[str]] = None,
    ) -> "DataFrame":
        """Build training set and materialise it immediately."""
        ts = build_training_set(
            self.fe, 
            labels_df, 
            feature_lookups, 
            label_col,
            exclude_columns=exclude_columns
        )
        return ts.load_df()

    def score_batch(self, df: "DataFrame", model_uri: str, **kwargs: Any) -> "DataFrame":
        return score_batch_wrapper(self.fe, df, model_uri, **kwargs)

    def create_lookups(
        self,
        table_name: str,
        lookup_keys: list[str],
        feature_names: Optional[list[str]] = None,
        *,
        rename_features: Optional[dict[str, str]] = None,
        timestamp_lookup_key: Optional[str] = None,
    ) -> list[Any]:
        """Create a list of FeatureLookups for the given table."""
        from databricks.feature_engineering import FeatureLookup
        
        return [
            FeatureLookup(
                table_name=self.fqn(table_name),
                lookup_key=lookup_keys,
                feature_names=feature_names,
                rename_features=rename_features,
                timestamp_lookup_key=timestamp_lookup_key,
            )
        ]

    def drop_table(self, table_name: str) -> None:
        """Drop an offline feature table."""
        if self.dry_run:
            logger.info("[DRY RUN] Would drop feature table '%s'", self.fqn(table_name))
            return
        self.fe.drop_table(name=self.fqn(table_name))
        logger.info("Dropped feature table '%s'.", self.fqn(table_name))

    def read_table(self, table_name: str) -> "DataFrame":
        """Read the offline feature table as a Spark DataFrame."""
        return self.fe.read_table(name=self.fqn(table_name))
        
    def set_tags(self, table_name: str, tags: dict[str, str]) -> None:
        """Set tags on the feature table."""
        if self.dry_run:
            logger.info("[DRY RUN] Would set tags %s on '%s'", tags, self.fqn(table_name))
            return
        for k, v in tags.items():
            self.fe.set_feature_table_tag(name=self.fqn(table_name), key=k, value=v)
        logger.info("Set tags on feature table '%s'.", self.fqn(table_name))

    def update_description(self, table_name: str, description: str) -> None:
        """Update the description of the feature table."""
        if self.dry_run:
            logger.info("[DRY RUN] Would update description on '%s'", self.fqn(table_name))
            return
        from mlops_utils.spark_utils import get_or_create_spark
        spark = get_or_create_spark()
        safe_desc = description.replace("'", "\\'")
        spark.sql(f"COMMENT ON TABLE {self.fqn(table_name)} IS '{safe_desc}'")
        logger.info("Updated description on feature table '%s'.", self.fqn(table_name))

    @with_retry(max_attempts=3, backoff_base=2.0)
    def sync_online_table(self, table_name: str, primary_keys: list[str]) -> None:
        """Sync the offline feature table to a Unity Catalog Online Table."""
        if self.dry_run:
            logger.info("[DRY RUN] Would create/update online table '%s'", self.fqn(table_name, use_online_schema=True))
            return
            
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.catalog import OnlineTableSpec, OnlineTableSpecContinuousSchedulingPolicy
        
        w = WorkspaceClient()
        online_fqn = self.fqn(table_name, use_online_schema=True)
        offline_fqn = self.fqn(table_name, use_online_schema=False)
        
        try:
            w.online_tables.get(online_fqn)
            logger.info("Online table '%s' already exists (sync is continuous).", online_fqn)
        except Exception:
            spec = OnlineTableSpec(
                primary_key_columns=primary_keys,
                source_table_full_name=offline_fqn,
                run_continuously=OnlineTableSpecContinuousSchedulingPolicy(),
            )
            w.online_tables.create(name=online_fqn, spec=spec)
            logger.info("Created online table '%s' mirroring '%s'.", online_fqn, offline_fqn)

    def query_endpoint(self, endpoint_name: str, records: list[dict[str, Any]]) -> Any:
        """Query an online Feature Serving endpoint for real-time inference."""
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        response = w.serving_endpoints.query(
            name=endpoint_name,
            dataframe_records=records,
        )
        return response.predictions


# ---------------------------------------------------------------------------
# Offline / batch helpers
# ---------------------------------------------------------------------------

def create_or_replace_feature_table(
    fe: "FeatureEngineeringClient",
    name: str,
    df: "DataFrame",
    primary_keys: list[str],
    *,
    timeseries_columns: Optional[str | list[str]] = None,
    description: str = "",
    tags: Optional[dict[str, str]] = None,
    raise_on_drop_error: bool = False,
) -> Any:
    """Create a Unity Catalog Feature Table (drop first if exists).
    
    Parameters
    ----------
    fe:
        ``FeatureEngineeringClient`` instance.
    name:
        Fully-qualified feature table name, e.g.
        ``"main.dbdemos_mlops.churn_feature_table"``.
    df:
        Source DataFrame whose schema defines the feature table schema.
    primary_keys:
        Column(s) used as the primary key.
    timeseries_columns:
        Column(s) to mark as timeseries (enables point-in-time lookups).
        Accepts a single column name or a list.
    description:
        Human-readable description logged to UC.
    tags:
        Optional metadata tags (key-value pairs).
    raise_on_drop_error:
        If True, raise exception if dropping the table fails (e.g., due to permissions).
        Otherwise, swallow the exception (assuming table didn't exist).

    Returns
    -------
    FeatureTable metadata object returned by the FE client.
    """
    # Drop existing table to allow schema changes
    try:
        fe.drop_table(name=name)
        logger.info("Dropped existing feature table '%s'.", name)
    except Exception as exc:  # noqa: BLE001
        if raise_on_drop_error:
            raise
        logger.debug("Drop failed for feature table '%s' (may not exist): %s", name, exc)

    kwargs: dict[str, Any] = dict(
        name=name,
        primary_keys=primary_keys,
        schema=df.schema,
        description=description,
    )
    if timeseries_columns:
        kwargs["timeseries_columns"] = (
            timeseries_columns
            if isinstance(timeseries_columns, str)
            else timeseries_columns[0]
            if len(timeseries_columns) == 1
            else timeseries_columns
        )
    if tags:
        kwargs["tags"] = tags

    table_meta = fe.create_table(**kwargs)
    logger.info("Created feature table '%s'.", name)
    return table_meta


def write_feature_table(
    fe: "FeatureEngineeringClient",
    name: str,
    df: "DataFrame",
    *,
    mode: str = "merge",
) -> None:
    """Write (or merge) features into an existing Unity Catalog Feature Table.

    Parameters
    ----------
    fe:
        ``FeatureEngineeringClient`` instance.
    name:
        Fully-qualified feature table name.
    df:
        DataFrame of feature values to write.
    mode:
        Write mode.  The Databricks Feature Engineering client only supports
        ``"merge"`` (upsert by primary key).  ``"overwrite"`` is **not**
        supported — to reset a feature table, call
        :func:`create_or_replace_feature_table` first (which drops and
        recreates the table), then write with ``mode="merge"`` into the
        now-empty table.

    Raises
    ------
    Exception
        Propagates any error raised by ``fe.write_table``.
    """
    if mode != "merge":
        raise ValueError(
            f"write_feature_table only supports mode='merge'; got mode={mode!r}. "
            "To reset a feature table, call create_or_replace_feature_table() first "
            "(drops + recreates the table), then write with mode='merge'."
        )
    fe.write_table(name=name, df=df, mode=mode)
    logger.info("Wrote features to '%s' (mode=%s).", name, mode)


def build_training_set(
    fe: "FeatureEngineeringClient",
    labels_df: "DataFrame",
    feature_lookups: list[Any],
    label_col: str,
    *,
    exclude_columns: Optional[list[str]] = None,
    exclude_null_labels: bool = True,
) -> Any:
    """Create a training-set specification via point-in-time feature lookups.

    Parameters
    ----------
    fe:
        ``FeatureEngineeringClient`` instance.
    labels_df:
        DataFrame containing at minimum the lookup key(s) and label column.
    feature_lookups:
        List of ``FeatureLookup`` / ``FeatureFunction`` specs.
    label_col:
        Name of the label column in *labels_df*.
    exclude_columns:
        Columns to exclude from the assembled training set (e.g. internal
        timestamp keys).
    exclude_null_labels:
        Drop rows where *label_col* is null.

    Returns
    -------
    TrainingSet specification (call ``.load_df()`` to materialise it).
    """
    training_set = fe.create_training_set(
        df=labels_df,
        label=label_col,
        feature_lookups=feature_lookups,
        exclude_columns=exclude_columns or [],
        exclude_null_labels=exclude_null_labels,
    )
    logger.info("Built training set specification for label '%s'.", label_col)
    return training_set


def score_batch_wrapper(
    fe: "FeatureEngineeringClient",
    df: "DataFrame",
    model_uri: str,
    *,
    result_type: str = "string",
    env_manager: str = "virtualenv",
) -> "DataFrame":
    """Run batch inference using the Feature Engineering client.

    Parameters
    ----------
    fe:
        ``FeatureEngineeringClient`` instance.
    df:
        DataFrame of lookup keys (feature values are pulled from the feature
        store automatically).
    model_uri:
        MLflow model URI, e.g. ``"models:/catalog.schema.model@Champion"``.
    result_type:
        Spark SQL type string for the prediction column, e.g. ``"string"`` or
        ``"double"``.
    env_manager:
        Virtual environment manager used during inference – ``"virtualenv"``,
        ``"uv"``, or ``"local"``.

    Returns
    -------
    pyspark.sql.DataFrame with an appended ``prediction`` column.
    """
    preds_df = fe.score_batch(
        df=df,
        model_uri=model_uri,
        result_type=result_type,
        env_manager=env_manager,
    )
    logger.info("Batch scoring complete (model_uri='%s').", model_uri)
    return preds_df


# ---------------------------------------------------------------------------
# Online store helpers
# ---------------------------------------------------------------------------

def publish_online_if_enabled(cfg: Any, fsm: FeatureStoreManager) -> bool:
    """Publish to online store (UC Online Table) only if cfg.online_store.enabled is True."""
    if not cfg.online_store.enabled:
        logger.info("Online store disabled – skipping publish.")
        return False
        
    fsm.sync_online_table(
        table_name=cfg.feature_table,
        primary_keys=list(cfg.primary_keys)
    )
    return True


@with_retry(max_attempts=3, backoff_base=2.0)
def create_feature_serving_endpoint(
    endpoint_name: str,
    served_entities: list[Any],
    *,
    workspace_client: Optional[Any] = None,
) -> Any:
    """Create or update a Databricks Feature Serving endpoint.

    Parameters
    ----------
    endpoint_name:
        Name of the online serving endpoint.
    served_entities:
        List of ``ServedEntity`` objects defining which feature spec(s) to serve.
    workspace_client:
        Optional ``databricks.sdk.WorkspaceClient``.  If ``None``, a new client
        is created from the ambient Databricks environment.

    Returns
    -------
    The created/updated endpoint object.
    """
    if workspace_client is None:
        from databricks.sdk import WorkspaceClient  # type: ignore[import]

        workspace_client = WorkspaceClient()

    from databricks.sdk.service.serving import EndpointCoreConfigInput  # type: ignore[import]

    try:
        endpoint = workspace_client.serving_endpoints.create_and_wait(
            name=endpoint_name,
            config=EndpointCoreConfigInput(served_entities=served_entities),
        )
        logger.info("Created feature serving endpoint '%s'.", endpoint_name)
    except Exception:  # noqa: BLE001
        endpoint = workspace_client.serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_entities=served_entities,
        )
        logger.info("Updated feature serving endpoint '%s'.", endpoint_name)

    return endpoint
