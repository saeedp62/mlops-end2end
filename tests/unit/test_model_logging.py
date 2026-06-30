from unittest import mock

import pandas as pd
from mlops_utils.model_logging import log_and_evaluate_model


@mock.patch("mlops_utils.mlflow_utils.managed_mlflow_run")
@mock.patch("mlops_utils.mlflow_utils.log_classification_metrics")
def test_log_and_evaluate_model_with_fe(mock_log_classification_metrics, mock_managed_mlflow_run):
    # Mock FeatureEngineeringClient
    fe_mock = mock.Mock()
    with mock.patch("mlops_utils.model_logging.FeatureEngineeringClient", return_value=fe_mock, create=True):
        classifier_mock = mock.Mock()
        classifier_mock.feature_importances_ = [0.5, 0.5]
        pipeline_mock = mock.Mock()
        pipeline_mock.steps = [("classifier", classifier_mock)]
        X_train = pd.DataFrame({"f1": [1, 2]})
        Y_train = pd.Series([0, 1])
        X_test = pd.DataFrame({"f1": [3]})
        Y_test = pd.Series([1])

        log_and_evaluate_model(
            model_pipeline=pipeline_mock,
            X_train=X_train,
            Y_train=Y_train,
            X_test=X_test,
            Y_test=Y_test,
            label_col="target",
            pos_label="1",
            training_set_specs=mock.Mock(),
            run_id="test_run",
            experiment_id="test_exp",
            artifact_path="my_model",
            fe_client=fe_mock
        )

        # Verify managed_mlflow_run context was entered
        mock_managed_mlflow_run.assert_called_once_with(run_id="test_run", experiment_id="test_exp")

        # Verify log_classification_metrics was called twice (train and test)
        assert mock_log_classification_metrics.call_count == 2

        # Verify FE log_model was called
        fe_mock.log_model.assert_called_once()
