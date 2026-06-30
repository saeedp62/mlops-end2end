from unittest import mock

from mlops_utils.optimization import NoneValuePruner, run_mlflow_optuna_study


def test_none_value_pruner():
    pruner = NoneValuePruner()

    # Mock study and trial
    study = mock.Mock()
    trial = mock.Mock()

    # When value is None
    trial.value = None
    assert pruner.prune(study, trial) is True

    # When value is not None
    trial.value = 0.85
    assert pruner.prune(study, trial) is False

@mock.patch("mlops_utils.optimization.optuna")
@mock.patch("mlflow.pyspark.optuna.study.MlflowSparkStudy")
@mock.patch("mlflow.optuna.storage.MlflowStorage")
@mock.patch("mlops_utils.mlflow_utils.managed_mlflow_run")
def test_run_mlflow_optuna_study_spark(mock_managed_run, mock_mlflow_storage, mock_spark_study, mock_optuna):
    def dummy_objective(trial):
        return 1.0

    study_instance = mock.Mock()
    study_instance.best_trial.value = 1.0
    mock_spark_study.return_value = study_instance

    result = run_mlflow_optuna_study(
        objective_fn=dummy_objective,
        experiment_id="test_exp_id",
        run_name="test_run",
        n_trials=2,
        direction="maximize"
    )

    assert result is study_instance
    mock_mlflow_storage.assert_called_once_with(experiment_id="test_exp_id")
    mock_spark_study.assert_called_once()
    study_instance.optimize.assert_called_once_with(dummy_objective, n_trials=2, n_jobs=1)
