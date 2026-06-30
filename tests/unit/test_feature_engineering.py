"""
tests/unit/test_feature_engineering.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ``churn.feature_engineering``.

All tests use the local SparkSession fixture from conftest.py.
No Databricks or external dependencies required.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# compute_service_features
# ---------------------------------------------------------------------------

class TestComputeServiceFeatures:
    def test_all_yes_gives_six(self, spark):
        from churn.feature_engineering import compute_service_features

        row = {
            "online_security": "Yes", "online_backup": "Yes",
            "device_protection": "Yes", "tech_support": "Yes",
            "streaming_tv": "Yes", "streaming_movies": "Yes",
        }
        df = spark.createDataFrame([row])
        result = compute_service_features(df).collect()[0]
        assert result["num_optional_services"] == 6.0

    def test_all_no_gives_zero(self, spark):
        from churn.feature_engineering import compute_service_features

        row = {
            "online_security": "No", "online_backup": "No",
            "device_protection": "No", "tech_support": "No",
            "streaming_tv": "No", "streaming_movies": "No",
        }
        df = spark.createDataFrame([row])
        result = compute_service_features(df).collect()[0]
        assert result["num_optional_services"] == 0.0

    def test_partial_yes_counts_correctly(self, spark):
        from churn.feature_engineering import compute_service_features

        row = {
            "online_security": "Yes", "online_backup": "No",
            "device_protection": "Yes", "tech_support": "No internet service",
            "streaming_tv": "Yes", "streaming_movies": "No",
        }
        df = spark.createDataFrame([row])
        result = compute_service_features(df).collect()[0]
        assert result["num_optional_services"] == 3.0

    def test_column_is_double_type(self, spark):
        from churn.feature_engineering import compute_service_features

        row = dict.fromkeys(["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"], "No")
        df = spark.createDataFrame([row])
        result = compute_service_features(df)
        dtype = dict(result.dtypes)["num_optional_services"]
        assert dtype == "double"

    def test_existing_columns_preserved(self, spark, bronze_sdf):
        from churn.feature_engineering import compute_service_features

        result = compute_service_features(bronze_sdf)
        # Original columns should still be present
        assert "customer_id" in result.columns
        assert "num_optional_services" in result.columns


# ---------------------------------------------------------------------------
# clean_churn_features
# ---------------------------------------------------------------------------

class TestCleanChurnFeatures:
    def test_senior_citizen_int_to_string(self, spark):
        from churn.feature_engineering import clean_churn_features, compute_service_features

        # Must run compute_service_features first (adds required cols)
        row = {
            "senior_citizen": 1,
            "total_charges": "100.5",
            "tenure": 5.0,
            "monthly_charges": 29.85,
            "online_security": "No", "online_backup": "No",
            "device_protection": "No", "tech_support": "No",
            "streaming_tv": "No", "streaming_movies": "No",
        }
        df = spark.createDataFrame([row])
        df = compute_service_features(df)
        result = clean_churn_features(df).collect()[0]
        assert result["senior_citizen"] == "Yes"

    def test_senior_citizen_zero_maps_to_no(self, spark):
        from churn.feature_engineering import clean_churn_features, compute_service_features

        row = {
            "senior_citizen": 0,
            "total_charges": "50.0",
            "tenure": 10.0,
            "monthly_charges": 50.0,
            "online_security": "No", "online_backup": "No",
            "device_protection": "No", "tech_support": "No",
            "streaming_tv": "No", "streaming_movies": "No",
        }
        df = spark.createDataFrame([row])
        df = compute_service_features(df)
        result = clean_churn_features(df).collect()[0]
        assert result["senior_citizen"] == "No"

    def test_whitespace_total_charges_becomes_zero(self, spark):
        from churn.feature_engineering import clean_churn_features, compute_service_features

        row = {
            "senior_citizen": 0,
            "total_charges": " ",  # whitespace only
            "tenure": 0.0,
            "monthly_charges": 29.85,
            "online_security": "No", "online_backup": "No",
            "device_protection": "No", "tech_support": "No",
            "streaming_tv": "No", "streaming_movies": "No",
        }
        df = spark.createDataFrame([row])
        df = compute_service_features(df)
        result = clean_churn_features(df).collect()[0]
        assert result["total_charges"] == 0.0

    def test_null_tenure_filled_with_zero(self, spark):
        from churn.feature_engineering import clean_churn_features, compute_service_features

        row = {
            "senior_citizen": 0,
            "total_charges": "100.0",
            "tenure": None,
            "monthly_charges": 30.0,
            "online_security": "No", "online_backup": "No",
            "device_protection": "No", "tech_support": "No",
            "streaming_tv": "No", "streaming_movies": "No",
        }
        df = spark.createDataFrame([row])
        df = compute_service_features(df)
        result = clean_churn_features(df).collect()[0]
        assert result["tenure"] == 0.0

    def test_null_monthly_charges_filled(self, spark):
        from churn.feature_engineering import clean_churn_features, compute_service_features

        row = {
            "senior_citizen": 0,
            "total_charges": "100.0",
            "tenure": 5.0,
            "monthly_charges": None,
            "online_security": "No", "online_backup": "No",
            "device_protection": "No", "tech_support": "No",
            "streaming_tv": "No", "streaming_movies": "No",
        }
        df = spark.createDataFrame([row])
        df = compute_service_features(df)
        result = clean_churn_features(df).collect()[0]
        assert result["monthly_charges"] == 0.0


# ---------------------------------------------------------------------------
# add_transaction_timestamp
# ---------------------------------------------------------------------------

class TestAddTransactionTimestamp:
    def test_timestamp_column_added(self, spark, bronze_sdf):
        from churn.feature_engineering import add_transaction_timestamp

        result = add_transaction_timestamp(bronze_sdf)
        assert "transaction_ts" in result.columns

    def test_timestamp_type_is_timestamp(self, spark, bronze_sdf):
        from churn.feature_engineering import add_transaction_timestamp

        result = add_transaction_timestamp(bronze_sdf)
        dtype = dict(result.dtypes)["transaction_ts"]
        assert dtype == "timestamp"

    def test_custom_column_name(self, spark, bronze_sdf):
        from churn.feature_engineering import add_transaction_timestamp

        result = add_transaction_timestamp(bronze_sdf, col_name="scored_at")
        assert "scored_at" in result.columns
        assert "transaction_ts" not in result.columns

    def test_row_count_unchanged(self, spark, bronze_sdf):
        from churn.feature_engineering import add_transaction_timestamp

        original_count = bronze_sdf.count()
        result = add_transaction_timestamp(bronze_sdf)
        assert result.count() == original_count


# ---------------------------------------------------------------------------
# split_label_from_features
# ---------------------------------------------------------------------------

class TestSplitLabelFromFeatures:
    def test_feature_df_does_not_contain_label(self, feature_sdf):
        # Manually add churn back for testing the split function
        from pyspark.sql.functions import lit

        from churn.feature_engineering import split_label_from_features
        df_with_label = feature_sdf.withColumn("churn", lit("No"))
        feature_df, label_df = split_label_from_features(df_with_label, "churn")

        assert "churn" not in feature_df.columns
        assert "churn" in label_df.columns

    def test_label_df_contains_split_column(self, feature_sdf):
        from pyspark.sql.functions import lit

        from churn.feature_engineering import split_label_from_features

        df_with_label = feature_sdf.withColumn("churn", lit("Yes"))
        _feature_df, label_df = split_label_from_features(df_with_label, "churn")

        assert "split" in label_df.columns

    def test_split_column_only_has_train_or_test(self, feature_sdf):
        from pyspark.sql.functions import lit

        from churn.feature_engineering import split_label_from_features

        df_with_label = feature_sdf.withColumn("churn", lit("No"))
        _feature_df, label_df = split_label_from_features(df_with_label, "churn", seed=42)

        splits = {r["split"] for r in label_df.select("split").collect()}
        assert splits.issubset({"train", "test"})

    def test_total_rows_preserved(self, feature_sdf):
        from pyspark.sql.functions import lit

        from churn.feature_engineering import split_label_from_features

        df_with_label = feature_sdf.withColumn("churn", lit("No"))
        original_count = df_with_label.count()
        feature_df, label_df = split_label_from_features(df_with_label, "churn")

        assert feature_df.count() == original_count
        assert label_df.count() == original_count


# ---------------------------------------------------------------------------
# build_feature_df (integration of all transforms)
# ---------------------------------------------------------------------------

class TestBuildFeatureDf:
    def test_output_has_num_optional_services(self, bronze_sdf):
        from churn.feature_engineering import build_feature_df

        result = build_feature_df(bronze_sdf)
        assert "num_optional_services" in result.columns

    def test_output_has_transaction_ts(self, bronze_sdf):
        from churn.feature_engineering import build_feature_df

        result = build_feature_df(bronze_sdf)
        assert "transaction_ts" in result.columns

    def test_row_count_preserved(self, bronze_sdf):
        from churn.feature_engineering import build_feature_df

        original_count = bronze_sdf.count()
        result = build_feature_df(bronze_sdf)
        assert result.count() == original_count

    def test_senior_citizen_is_string_after_pipeline(self, bronze_sdf):
        from churn.feature_engineering import build_feature_df

        result = build_feature_df(bronze_sdf)
        dtype = dict(result.dtypes)["senior_citizen"]
        assert dtype == "string"
