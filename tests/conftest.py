"""
tests/conftest.py
~~~~~~~~~~~~~~~~~
Shared pytest fixtures for all test levels.

Fixtures provided
-----------------
spark
    A local SparkSession usable in unit tests (no Databricks dependency).
bronze_pdf
    A small pandas DataFrame mimicking the raw bronze table.
bronze_sdf
    Spark version of ``bronze_pdf``.
feature_pdf
    A small pandas DataFrame representing the feature table output.
mock_mlflow_client
    A MagicMock pre-configured with the most common MlflowClient call patterns.
"""

from __future__ import annotations

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# SparkSession fixture  (scope=session → created once per test run)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    """Local SparkSession for unit tests – no Databricks dependency."""
    from mlops_utils.spark_utils import get_or_create_spark

    session = get_or_create_spark(
        app_name="churn_unit_tests",
        master="local[2]",
        extra_configs={"spark.sql.ansi.enabled": "false"},
    )
    yield session
    session.stop()


# ---------------------------------------------------------------------------
# Data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def bronze_pdf() -> pd.DataFrame:
    """Minimal bronze-table pandas DataFrame (3 rows, all required columns)."""
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "gender": ["Male", "Female", "Male"],
            "senior_citizen": [0, 1, 0],
            "partner": ["Yes", "No", "Yes"],
            "dependents": ["No", "No", "Yes"],
            "tenure": [12.0, None, 5.0],
            "phone_service": ["Yes", "Yes", "No"],
            "multiple_lines": ["No", "Yes", "No phone service"],
            "internet_service": ["DSL", "Fiber optic", "No"],
            "online_security": ["No", "Yes", "No internet service"],
            "online_backup": ["Yes", "No", "No internet service"],
            "device_protection": ["No", "Yes", "No internet service"],
            "tech_support": ["No", "No", "No internet service"],
            "streaming_tv": ["No", "Yes", "No internet service"],
            "streaming_movies": ["No", "Yes", "No internet service"],
            "contract": ["Month-to-month", "One year", "Two year"],
            "paperless_billing": ["Yes", "No", "Yes"],
            "payment_method": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
            "monthly_charges": [29.85, 56.95, None],
            "total_charges": ["358.5", " ", "42.3"],
            "churn": ["No", "No", "Yes"],
        }
    )


@pytest.fixture
def bronze_sdf(spark, bronze_pdf):
    """Spark version of ``bronze_pdf``."""
    return spark.createDataFrame(bronze_pdf)


@pytest.fixture
def feature_sdf(spark, bronze_sdf):
    """Feature-engineered Spark DataFrame (after apply of build_feature_df)."""
    from churn.feature_engineering import build_feature_df

    return build_feature_df(bronze_sdf)


# ---------------------------------------------------------------------------
# Mock MLflow client fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_mlflow_client(mocker):
    """Return a MagicMock configured like MlflowClient for common call patterns."""
    from unittest.mock import MagicMock

    client = MagicMock()

    # get_model_version → fake ModelVersion
    mv = MagicMock()
    mv.version = "1"
    mv.run_id = "test-run-id-001"
    mv.description = "A test model for churn prediction with >20 characters."
    client.get_model_version.return_value = mv

    # get_model_version_by_alias → champion version
    champion_mv = MagicMock()
    champion_mv.version = "1"
    champion_mv.run_id = "champion-run-id-001"
    client.get_model_version_by_alias.return_value = champion_mv

    # search_runs → empty list (no existing runs)
    client.search_runs.return_value = []

    return client


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def churn_config():
    """Default ChurnConfig for tests (no file I/O required)."""
    from churn.config import ChurnConfig, SchemaConfig

    return ChurnConfig(
        catalog="test_catalog",
        schemas=SchemaConfig(
            training_datasets="test_training_datasets",
            offline_features="test_offline_features",
            online_features="test_online_features",
            ml_models="test_ml_models",
            model_predictions="test_model_predictions",
            ml_monitoring="test_ml_monitoring",
        ),
        bronze_table="test_bronze",
        feature_table="test_features",
        label_table="test_labels",
    )
