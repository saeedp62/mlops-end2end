"""
optimization.py - Shared MLflow and Optuna Hyperparameter Optimization utilities.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    import optuna
    from optuna.pruners import BasePruner
except ImportError:
    optuna = None

from mlops_utils.logger import get_logger

logger = get_logger(__name__)


class NoneValuePruner:
    """Custom Pruner to ignore failed trials with None value."""

    def prune(self, study: Any, trial: Any) -> bool:
        # If the trial's value is None, prune it
        if trial.value is None:
            return True
        return False


def run_mlflow_optuna_study(
    objective_fn: Callable[[Any], float | None],
    experiment_id: str,
    run_name: str,
    n_trials: int = 8,
    n_jobs: int = 1,
    direction: str = "maximize",
    sampler: Any | None = None,
    pruner: Any | None = None,
) -> Any:
    """
    Run an Optuna study distributed via Spark and backed by MLflow.

    Requires `mlflow` and `optuna` packages.
    """
    if optuna is None:
        raise ImportError("optuna is required for hyperparameter optimization.")

    from mlflow.optuna.storage import MlflowStorage
    try:
        from mlflow.pyspark.optuna.study import MlflowSparkStudy
    except ImportError:
        # Fallback for environments lacking the MlflowSparkStudy extension
        logger.warning("MlflowSparkStudy not found. Falling back to local optuna study.")
        MlflowSparkStudy = None

    if pruner is None:
        pruner = NoneValuePruner()

    mlflow_storage = MlflowStorage(experiment_id=experiment_id)

    if MlflowSparkStudy is not None:
        logger.info(f"Starting distributed MlflowSparkStudy '{run_name}' with {n_trials} trials.")
        study = MlflowSparkStudy(
            pruner=pruner,
            sampler=sampler,
            study_name=run_name,
            storage=mlflow_storage,
        )
        study._directions = [direction]
    else:
        logger.info(f"Starting local optuna study '{run_name}' with {n_trials} trials.")
        study = optuna.create_study(
            pruner=pruner,
            sampler=sampler,
            study_name=run_name,
            storage=mlflow_storage,
            direction=direction,
            load_if_exists=True,
        )

    from mlops_utils.mlflow_utils import managed_mlflow_run

    with managed_mlflow_run(experiment_id=experiment_id, run_name=f"{run_name}_study"):
        study.optimize(objective_fn, n_trials=n_trials, n_jobs=n_jobs)

    logger.info("Optuna study complete.")
    logger.info(f"Best trial value: {study.best_trial.value}")

    return study
