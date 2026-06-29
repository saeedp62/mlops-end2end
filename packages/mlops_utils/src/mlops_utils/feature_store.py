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
        create_or_replace_feature_table,
        write_feature_table,
        build_training_set,
        score_batch_wrapper,
        publish_to_online_store,
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    # These are only imported for type checking; the actual runtime imports
    # are deferred so the module is importable without Databricks SDK.
    from databricks.feature_engineering import FeatureEngineeringClient
    from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


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

    Returns
    -------
    FeatureTable metadata object returned by the FE client.
    """
    # Drop existing table to allow schema changes
    try:
        fe.drop_table(name=name)
        logger.info("Dropped existing feature table '%s'.", name)
    except Exception:  # noqa: BLE001
        pass  # Table didn't exist – that's fine

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
        Write mode – ``"merge"`` (default, supports schema evolution) or
        ``"overwrite"``.
    """
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

def publish_to_online_store(
    fe: "FeatureEngineeringClient",
    feature_table_name: str,
    online_store_spec: Any,
    *,
    mode: str = "merge",
) -> None:
    """Publish features from an offline Delta table to an online (key-value) store.

    The ``online_store_spec`` controls which backend to target.  Example for
    Databricks-managed online store:

    .. code-block:: python

        from databricks.feature_engineering.entities.feature_serving_endpoint import (
            ServedEntity,
        )
        from databricks.feature_store.online_store_spec import AmazonDynamoDBSpec

        spec = AmazonDynamoDBSpec(
            region="us-east-1",
            write_secret_prefix="ml/feature-store",
            read_secret_prefix="ml/feature-store",
            table_name="churn_features_online",
        )

    Parameters
    ----------
    fe:
        ``FeatureEngineeringClient`` instance.
    feature_table_name:
        Fully-qualified offline feature table name.
    online_store_spec:
        Platform-specific online store specification object.
    mode:
        Publish mode – ``"merge"`` or ``"overwrite"``.
    """
    fe.publish_table(
        name=feature_table_name,
        online_store=online_store_spec,
        mode=mode,
    )
    logger.info(
        "Published feature table '%s' to online store (mode=%s).",
        feature_table_name,
        mode,
    )


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
