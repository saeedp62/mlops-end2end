"""
tests/unit/test_mlflow_utils.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ``mlops_utils.mlflow_utils``.

MLflow client is always mocked so these tests run without an MLflow server.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestGetOrCreateExperiment:
    def test_returns_existing_experiment_id(self):
        from mlops_utils.mlflow_utils import get_or_create_experiment

        fake_experiment = MagicMock()
        fake_experiment.experiment_id = "existing-exp-id"

        with patch("mlflow.get_experiment_by_name", return_value=fake_experiment) as mock_get:
            result = get_or_create_experiment("/Users/user/my_experiment")

        assert result == "existing-exp-id"
        mock_get.assert_called_once_with("/Users/user/my_experiment")

    def test_creates_new_experiment_when_not_found(self):
        from mlops_utils.mlflow_utils import get_or_create_experiment

        with patch("mlflow.get_experiment_by_name", return_value=None):
            with patch("mlflow.create_experiment", return_value="new-exp-id") as mock_create:
                result = get_or_create_experiment("/Users/user/new_experiment")

        assert result == "new-exp-id"
        mock_create.assert_called_once()

    def test_tags_passed_on_create(self):
        from mlops_utils.mlflow_utils import get_or_create_experiment

        with patch("mlflow.get_experiment_by_name", return_value=None):
            with patch("mlflow.create_experiment", return_value="id-123") as mock_create:
                get_or_create_experiment(
                    "/Users/user/exp", tags={"env": "dev", "team": "ml"}
                )

        _, call_kwargs = mock_create.call_args
        assert call_kwargs["tags"] == {"env": "dev", "team": "ml"}


class TestGetChampionMetric:
    def test_returns_metric_value(self, mock_mlflow_client):
        from mlops_utils.mlflow_utils import get_champion_metric

        mock_run = MagicMock()
        mock_run.data.metrics = {"test_f1_score": 0.85}

        with patch("mlflow.get_run", return_value=mock_run):
            result = get_champion_metric(
                mock_mlflow_client, "catalog.schema.model", "test_f1_score"
            )

        assert result == 0.85

    def test_returns_none_when_no_champion(self):
        from mlops_utils.mlflow_utils import get_champion_metric

        client = MagicMock()
        client.get_model_version_by_alias.side_effect = Exception("No Champion")

        result = get_champion_metric(client, "catalog.schema.model", "test_f1_score")
        assert result is None

    def test_returns_none_when_metric_not_logged(self, mock_mlflow_client):
        from mlops_utils.mlflow_utils import get_champion_metric

        mock_run = MagicMock()
        mock_run.data.metrics = {}  # metric not present

        with patch("mlflow.get_run", return_value=mock_run):
            result = get_champion_metric(
                mock_mlflow_client, "catalog.schema.model", "missing_metric"
            )

        assert result is None


class TestPromoteModelAlias:
    def test_calls_set_registered_model_alias(self, mock_mlflow_client):
        from mlops_utils.mlflow_utils import promote_model_alias

        promote_model_alias(mock_mlflow_client, "catalog.schema.model", 3, "Champion")

        mock_mlflow_client.set_registered_model_alias.assert_called_once_with(
            name="catalog.schema.model",
            alias="Champion",
            version="3",
        )

    def test_version_coerced_to_string(self, mock_mlflow_client):
        from mlops_utils.mlflow_utils import promote_model_alias

        promote_model_alias(mock_mlflow_client, "m.n.model", 5, "Challenger")

        call_args = mock_mlflow_client.set_registered_model_alias.call_args
        assert call_args[1]["version"] == "5"


class TestSetModelVersionTags:
    def test_all_tags_written(self, mock_mlflow_client):
        from mlops_utils.mlflow_utils import set_model_version_tags

        tags = {"has_description": True, "metric_f1_passed": False, "env": "test"}
        set_model_version_tags(mock_mlflow_client, "m.n.model", 1, tags)

        assert mock_mlflow_client.set_model_version_tag.call_count == 3

    def test_values_coerced_to_string(self, mock_mlflow_client):
        from mlops_utils.mlflow_utils import set_model_version_tags

        set_model_version_tags(mock_mlflow_client, "m.n.model", 1, {"passed": True})

        call_kwargs = mock_mlflow_client.set_model_version_tag.call_args[1]
        assert call_kwargs["value"] == "True"  # Python bool → string


class TestFindRunByName:
    def test_returns_first_matching_run(self, mock_mlflow_client):
        from mlops_utils.mlflow_utils import find_run_by_name

        fake_run = MagicMock()
        mock_mlflow_client.search_runs.return_value = [fake_run]

        result = find_run_by_name(mock_mlflow_client, "exp-id", "my-run")
        assert result is fake_run

    def test_returns_none_when_no_runs_found(self, mock_mlflow_client):
        from mlops_utils.mlflow_utils import find_run_by_name

        mock_mlflow_client.search_runs.return_value = []
        result = find_run_by_name(mock_mlflow_client, "exp-id", "missing-run")
        assert result is None
