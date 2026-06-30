"""
model_logging.py - Shared MLflow model evaluation and registration utilities.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from databricks.feature_engineering import FeatureEngineeringClient
    from databricks.feature_engineering.training_set import TrainingSet
    from sklearn.pipeline import Pipeline

import pandas as pd

from mlops_utils.logger import get_logger

logger = get_logger(__name__)


def log_and_evaluate_model(
    model_pipeline: Pipeline,
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    X_test: pd.DataFrame,
    Y_test: pd.Series,
    label_col: str,
    pos_label: str,
    training_set_specs: TrainingSet,
    run_id: str,
    experiment_id: str,
    artifact_path: str = "model",
    fe_client: FeatureEngineeringClient | None = None,
) -> None:
    """
    Evaluates the final model using mlflow.evaluate and logs it to MLflow
    via the Feature Engineering client.
    """
    import mlflow
    fe = fe_client
    if fe is None:
        logger.warning(
            "FeatureEngineeringClient not provided. Will use mlflow.sklearn.log_model instead."
        )

    from mlops_utils.mlflow_utils import managed_mlflow_run

    with managed_mlflow_run(run_id=run_id, experiment_id=experiment_id):
        from mlops_utils.mlflow_utils import log_classification_metrics, log_feature_importance

        # Log metrics for the training set
        logger.info("Evaluating on training set...")
        log_classification_metrics(
            model=model_pipeline,
            X=X_train,
            y=Y_train,
            label_col=str(label_col),
            metric_prefix="training_",
            pos_label=pos_label,
            log_explainability=False,
        )

        # Log metrics for the test set
        logger.info("Evaluating on test set...")
        log_classification_metrics(
            model=model_pipeline,
            X=X_test,
            y=Y_test,
            label_col=str(label_col),
            metric_prefix="test_",
            pos_label=pos_label,
            log_explainability=True,
        )

        # Log the feature importances
        feature_names = list(X_train.columns)
        logger.info("Logging feature importance plot...")
        log_feature_importance(model_pipeline, feature_names)

        # Log the model
        if fe is not None and training_set_specs is not None:
            logger.info("Logging model using FeatureEngineeringClient...")
            fe.log_model(
                model=model_pipeline,
                artifact_path=artifact_path,
                flavor=mlflow.sklearn,
                training_set=training_set_specs,
                serialization_format="cloudpickle",
            )
        else:
            logger.info("Logging model using mlflow.sklearn...")
            mlflow.sklearn.log_model(
                sk_model=model_pipeline,
                artifact_path=artifact_path,
                serialization_format="cloudpickle",
            )

        logger.info(f"Model logged to artifact path: {artifact_path}")
