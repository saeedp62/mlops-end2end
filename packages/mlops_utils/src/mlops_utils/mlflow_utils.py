"""
mlops_utils.mlflow_utils
~~~~~~~~~~~~~~~~~~~~~~~~
MLflow experiment, run, and model-registry helpers.

All functions accept an ``MlflowClient`` as a dependency-injected argument so
they can be mocked in unit tests without touching the MLflow server.

Public API
----------
::

    from mlops_utils.mlflow_utils import (
        get_or_create_experiment,
        get_champion_metric,
        promote_model_alias,
        set_model_version_tags,
        log_classification_metrics,
    )
"""

from __future__ import annotations

from mlops_utils.logger import get_logger
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from mlflow import MlflowClient
    from mlflow.entities import Run
    import pandas as pd

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Experiment helpers
# ---------------------------------------------------------------------------

def get_or_create_experiment(
    experiment_name: str,
    *,
    tags: Optional[dict[str, str]] = None,
) -> str:
    """Return the experiment ID for *experiment_name*, creating it if absent.

    Parameters
    ----------
    experiment_name:
        Full MLflow experiment path (e.g. ``"/Users/user@co/my_experiment"``).
    tags:
        Tags to set when **creating** a new experiment (ignored if the
        experiment already exists).

    Returns
    -------
    str
        Experiment ID (opaque string).
    """
    import mlflow

    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is not None:
        logger.info("Found existing experiment '%s' (id=%s).", experiment_name, existing.experiment_id)
        return existing.experiment_id

    experiment_id = mlflow.create_experiment(experiment_name, tags=tags or {})
    logger.info("Created experiment '%s' (id=%s).", experiment_name, experiment_id)
    return experiment_id


def ensure_workspace_path(path: str) -> None:
    """Create a Databricks workspace folder *path* if it doesn't exist."""
    try:
        from databricks.sdk import WorkspaceClient  # type: ignore[import]

        WorkspaceClient().workspace.mkdirs(path=path)
        logger.debug("Workspace path '%s' ensured.", path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not create workspace path '%s': %s", path, exc)


# ---------------------------------------------------------------------------
# Model registry helpers
# ---------------------------------------------------------------------------

def get_champion_metric(
    client: "MlflowClient",
    model_name: str,
    metric_key: str,
    *,
    alias: str = "Champion",
) -> Optional[float]:
    """Fetch a logged metric from the run backing the *alias* model version.

    Parameters
    ----------
    client:
        ``MlflowClient`` instance.
    model_name:
        Fully-qualified UC model name (``catalog.schema.model``).
    metric_key:
        Name of the metric to retrieve (e.g. ``"test_f1_score"``).
    alias:
        Model alias to look up.  Defaults to ``"Champion"``.

    Returns
    -------
    float or ``None``
        The metric value, or ``None`` if the alias or metric does not exist.
    """
    import mlflow

    try:
        mv = client.get_model_version_by_alias(model_name, alias)
        run = mlflow.get_run(mv.run_id)
        value = run.data.metrics.get(metric_key)
        logger.info(
            "Champion metric '%s' = %s (model=%s, alias=%s, run=%s).",
            metric_key,
            value,
            model_name,
            alias,
            mv.run_id,
        )
        return value
    except Exception:  # noqa: BLE001
        logger.info("No '%s' alias found for model '%s'.", alias, model_name)
        return None


def promote_model_alias(
    client: "MlflowClient",
    model_name: str,
    version: str | int,
    alias: str,
) -> None:
    """Set *alias* on the given model *version* (creates or moves the alias).

    Parameters
    ----------
    client:
        ``MlflowClient`` instance.
    model_name:
        Fully-qualified UC model name.
    version:
        Model version number (integer or string).
    alias:
        Alias to set (e.g. ``"Champion"`` or ``"Challenger"``).
    """
    client.set_registered_model_alias(
        name=model_name, alias=alias, version=str(version)
    )
    logger.info("Set alias '%s' → version %s for model '%s'.", alias, version, model_name)


def set_model_version_tags(
    client: "MlflowClient",
    model_name: str,
    version: str | int,
    tags: dict[str, Any],
) -> None:
    """Batch-set multiple tags on a model version.

    Parameters
    ----------
    client:
        ``MlflowClient`` instance.
    model_name:
        Fully-qualified UC model name.
    version:
        Model version number.
    tags:
        Dictionary of ``{key: value}`` pairs.  Values are coerced to strings.
    """
    for key, value in tags.items():
        client.set_model_version_tag(
            name=model_name,
            version=str(version),
            key=key,
            value=str(value),
        )
    logger.info("Set %d tag(s) on '%s' v%s.", len(tags), model_name, version)


def find_run_by_name(
    client: "MlflowClient",
    experiment_id: str,
    run_name: str,
) -> Optional["Run"]:
    """Search for the most recent run with *run_name* in *experiment_id*.

    Returns ``None`` if not found.
    """
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def log_classification_metrics(
    model: Any,
    X: "pd.DataFrame",
    y: "pd.Series",
    *,
    label_col: str,
    metric_prefix: str = "",
    pos_label: Optional[str] = None,
    log_explainability: bool = False,
) -> Any:
    """Evaluate a model and log metrics into the active MLflow run.

    Wraps ``mlflow.evaluate`` to standardise prefix conventions across
    training and validation pipelines.

    Parameters
    ----------
    model:
        A fitted ``PyFuncModel`` or ``sklearn`` model.
    X:
        Feature DataFrame.
    y:
        Label Series.
    label_col:
        Name to assign to the label column in the evaluation dataset.
    metric_prefix:
        String prepended to every logged metric name (e.g. ``"test_"``).
    pos_label:
        Positive class label for binary classification metrics.
    log_explainability:
        Whether to compute and log SHAP-based explainability artifacts.

    Returns
    -------
    mlflow.models.EvaluationResult
    """
    import mlflow
    from mlflow.models import Model
    from mlflow.pyfunc import PyFuncModel
    from mlflow import pyfunc

    # Wrap sklearn model as pyfunc if needed
    if not isinstance(model, PyFuncModel):
        mlflow_model = Model()
        pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
        pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model)
    else:
        pyfunc_model = model

    eval_data = X.assign(**{label_col: y})

    evaluator_config: dict[str, Any] = {
        "log_model_explainability": log_explainability,
        "metric_prefix": metric_prefix,
    }
    if pos_label is not None:
        evaluator_config["pos_label"] = pos_label

    result = mlflow.evaluate(
        model=pyfunc_model,
        data=eval_data,
        targets=label_col,
        model_type="classifier",
        evaluator_config=evaluator_config,
    )
    logger.info("Logged %s metrics with prefix '%s'.", len(result.metrics), metric_prefix)
    return result
