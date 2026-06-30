"""
tests/integration/test_feature_store_pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for ``churn.feature_store_pipeline``.

These tests require a **live Databricks cluster** with:
- Unity Catalog enabled
- ``lighthouse_bkk6_analytics.training_datasets_ci`` and
  ``lighthouse_bkk6_analytics.offline_features_ci`` schemas
  (created automatically if missing)
- The ``databricks-feature-engineering`` package installed

Run locally with Databricks Connect or on a CI Databricks cluster.
Skip in pure unit test runs with: ``pytest -m "not integration"``

Set the following environment variables for CI:
    DATABRICKS_HOST      = https://your-workspace.azuredatabricks.net
    DATABRICKS_TOKEN     = dapi...
    MLOPS_CATALOG        = lighthouse_bkk6_analytics
"""

from __future__ import annotations

import pytest


# All tests in this module require a live Databricks cluster
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def ci_spark():
    """Databricks-connected SparkSession for integration tests."""
    from mlops_utils.spark_utils import get_or_create_spark

    return get_or_create_spark(app_name="churn_integration_tests")


@pytest.fixture(scope="module")
def ci_config(tmp_path_factory):
    """ChurnConfig pointing to isolated CI schemas inside lighthouse_bkk6_analytics.

    Schemas used:
      training_datasets_ci  – bronze + label tables for CI run
      offline_features_ci   – feature table for CI run

    All other schemas (online_features, ml_models, etc.) use the shared
    schema names – CI tests do not write to those.
    """
    from churn.config import ChurnConfig
    from churn.config import SchemaConfig

    return ChurnConfig(
        catalog="lighthouse_bkk6_analytics",
        schemas=SchemaConfig(
            training_datasets="training_datasets_ci",
            offline_features="offline_features_ci",
            online_features="online_features",
            ml_models="ml_models",
            model_predictions="model_predictions",
            ml_monitoring="ml_monitoring",
        ),
        bronze_table="ci_churn_bronze_customers",
        feature_table="ci_churn_feature_table",
        label_table="ci_churn_label_table",
        model_name="ci_mlops_churn",
    )


@pytest.fixture(scope="module", autouse=True)
def setup_ci_schema(ci_spark, ci_config):
    """Create all CI schemas (and clean up after all tests in this module)."""
    from mlops_utils.catalog import ensure_mlops_schemas, drop_and_recreate_schema

    # Set up – create the two CI-isolated schemas
    ci_schemas = {
        ci_config.schemas.training_datasets: "CI bronze + label tables.",
        ci_config.schemas.offline_features:  "CI feature tables.",
    }
    ensure_mlops_schemas(ci_spark, ci_config.catalog, ci_schemas)

    yield

    # Tear down – drop the CI schemas after integration tests
    for schema in [ci_config.schemas.training_datasets, ci_config.schemas.offline_features]:
        drop_and_recreate_schema(ci_spark, ci_config.catalog, schema)


@pytest.fixture(scope="module")
def bronze_table_created(ci_spark, ci_config):
    """Download and ingest the Telco CSV into the CI bronze table."""
    from churn.data_source import download_telco_csv, normalize_column_names, ingest_bronze_table

    raw_df = download_telco_csv()
    normalised_df = normalize_column_names(raw_df)
    ingest_bronze_table(
        spark=ci_spark,
        df=normalised_df,
        full_table_name=ci_config.full_bronze_table,
        validate_schema=True,
    )
    return ci_config.full_bronze_table


class TestBronzeIngestion:
    def test_bronze_table_exists_after_ingest(self, ci_spark, ci_config, bronze_table_created):
        from mlops_utils.spark_utils import table_exists

        assert table_exists(ci_spark, ci_config.full_bronze_table)

    def test_bronze_table_has_expected_row_count(self, ci_spark, ci_config, bronze_table_created):
        count = ci_spark.table(ci_config.full_bronze_table).count()
        assert count > 7_000  # IBM Telco dataset has ~7,043 rows

    def test_bronze_table_has_customer_id_column(self, ci_spark, ci_config, bronze_table_created):
        cols = ci_spark.table(ci_config.full_bronze_table).columns
        assert "customer_id" in cols


class TestFeatureEngineeringPipeline:
    @pytest.fixture(scope="class", autouse=True)
    def run_pipeline(self, ci_spark, ci_config, bronze_table_created):
        """Run the full feature-engineering pipeline once for this test class."""
        from churn.feature_store_pipeline import run_feature_engineering_pipeline

        run_feature_engineering_pipeline(
            spark=ci_spark,
            config=ci_config,
            reset_feature_table=True,
            publish_online=False,  # Skip online store in CI
        )

    def test_label_table_created(self, ci_spark, ci_config):
        from mlops_utils.spark_utils import table_exists

        assert table_exists(ci_spark, ci_config.full_label_table)

    def test_label_table_has_split_column(self, ci_spark, ci_config):
        cols = ci_spark.table(ci_config.full_label_table).columns
        assert "split" in cols

    def test_label_table_split_values(self, ci_spark, ci_config):
        from pyspark.sql.functions import col

        splits = {
            r["split"]
            for r in ci_spark.table(ci_config.full_label_table).select("split").distinct().collect()
        }
        assert splits.issubset({"train", "test"})

    def test_feature_table_created(self, ci_spark, ci_config):
        from mlops_utils.spark_utils import table_exists

        assert table_exists(ci_spark, ci_config.full_feature_table)

    def test_feature_table_has_num_optional_services(self, ci_spark, ci_config):
        cols = ci_spark.table(ci_config.full_feature_table).columns
        assert "num_optional_services" in cols

    def test_feature_table_does_not_contain_label(self, ci_spark, ci_config):
        cols = ci_spark.table(ci_config.full_feature_table).columns
        assert ci_config.label_col not in cols

    def test_feature_table_senior_citizen_is_string(self, ci_spark, ci_config):
        dtype_map = dict(ci_spark.table(ci_config.full_feature_table).dtypes)
        assert dtype_map.get("senior_citizen") == "string"

    def test_train_test_ratio_is_approximately_80_20(self, ci_spark, ci_config):
        from pyspark.sql.functions import col, count

        label_df = ci_spark.table(ci_config.full_label_table)
        total = label_df.count()
        train_count = label_df.filter(col("split") == "train").count()
        ratio = train_count / total
        # Allow 5% tolerance around the 80% target
        assert 0.75 <= ratio <= 0.85
