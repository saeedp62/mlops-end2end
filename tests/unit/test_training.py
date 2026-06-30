from unittest import mock

import pandas as pd

from churn.training import ChurnOptunaObjective


def test_churn_optuna_objective_initialization():
    X_train = pd.DataFrame({
        "gender": ["Male", "Female"],
        "monthly_charges": [50.0, 60.0]
    })
    Y_train = pd.Series(["Yes", "No"])

    # Needs a mock of the preprocessor since it attempts to build it
    with mock.patch("churn.training.build_churn_preprocessor", return_value="mock_preprocessor"):
        obj = ChurnOptunaObjective(X_train, Y_train, rng_seed=42)

        assert obj.preprocessor == "mock_preprocessor"
        assert len(obj.X_train) + len(obj.X_val) == 2
        assert len(obj.Y_train) + len(obj.Y_val) == 2
        assert obj.pos_label == "Yes"

@mock.patch("churn.training.Pipeline")
@mock.patch("churn.training.LogisticRegression")
def test_churn_optuna_objective_call(mock_lr, mock_pipeline):
    X_train = pd.DataFrame({"f1": [1, 2, 3, 4]})
    Y_train = pd.Series(["Yes", "No", "Yes", "No"])

    with mock.patch("churn.training.build_churn_preprocessor"):
        obj = ChurnOptunaObjective(X_train, Y_train, rng_seed=42)

        # Mock trial
        trial = mock.Mock()
        trial.suggest_categorical.return_value = "LogisticRegression"
        trial.suggest_float.side_effect = [0.1, 1e-4]

        # Mock pipeline
        pipeline_instance = mock.Mock()
        pipeline_instance.predict.return_value = ["Yes"] * len(obj.X_val)
        mock_pipeline.return_value = pipeline_instance

        with mock.patch("mlflow.sklearn.autolog"):
            score = obj(trial)

            # LR was suggested and instantiated
            mock_lr.assert_called_once_with(
                C=0.1,
                tol=1e-4,
                random_state=42,
                class_weight={"No": 1.5, "Yes": 0.75}
            )

            # Pipeline was fit
            pipeline_instance.fit.assert_called_once()
            pipeline_instance.predict.assert_called_once_with(obj.X_val)

            # Since mock predicts all "Yes" and val has size ~1, score is deterministic
            assert isinstance(score, float)
