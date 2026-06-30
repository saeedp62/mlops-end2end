"""
tests/unit/test_data_source.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ``churn.data_source``.

Covers all three source strategies (unity_catalog_table, volume_csv, http_csv)
plus column normalisation helpers.  No Databricks, no live HTTP – all I/O
is mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ===========================================================================
# Column normalisation helpers
# ===========================================================================

class TestNormalizeSparkColumns:
    """_normalize_spark_columns renames Spark DataFrame columns to snake_case."""

    def test_camel_case_converted(self, spark):
        from churn.data_source import _normalize_spark_columns

        df = spark.createDataFrame([{"MonthlyCharges": 29.85, "TotalCharges": "358.5"}])
        result = _normalize_spark_columns(df)
        assert "monthly_charges" in result.columns
        assert "total_charges" in result.columns

    def test_streaming_tv_alias_applied(self, spark):
        from churn.data_source import _normalize_spark_columns

        df = spark.createDataFrame([{"StreamingTV": "Yes"}])
        result = _normalize_spark_columns(df)
        assert "streaming_tv" in result.columns

    def test_customer_id_alias_applied(self, spark):
        from churn.data_source import _normalize_spark_columns

        df = spark.createDataFrame([{"CustomerID": "C001"}])
        result = _normalize_spark_columns(df)
        assert "customer_id" in result.columns

    def test_already_snake_case_unchanged(self, spark):
        from churn.data_source import _normalize_spark_columns

        df = spark.createDataFrame([{"customer_id": "C001", "tenure": 12}])
        result = _normalize_spark_columns(df)
        assert "customer_id" in result.columns
        assert "tenure" in result.columns

    def test_original_columns_dropped_after_rename(self, spark):
        from churn.data_source import _normalize_spark_columns

        df = spark.createDataFrame([{"MonthlyCharges": 10.0}])
        result = _normalize_spark_columns(df)
        assert "MonthlyCharges" not in result.columns


class TestNormalizePandasColumns:
    """_normalize_pandas_columns covers the pandas path (http_csv source)."""

    def test_camel_case_converted(self):
        from churn.data_source import _normalize_pandas_columns

        df = pd.DataFrame(columns=["MonthlyCharges", "TotalCharges"])
        result = _normalize_pandas_columns(df)
        assert list(result.columns) == ["monthly_charges", "total_charges"]

    def test_spaces_replaced(self):
        from churn.data_source import _normalize_pandas_columns

        df = pd.DataFrame(columns=["Payment Method"])
        result = _normalize_pandas_columns(df)
        assert "payment_method" in result.columns

    def test_known_renames_applied(self):
        from churn.data_source import _normalize_pandas_columns

        df = pd.DataFrame(columns=["StreamingTV", "CustomerID"])
        result = _normalize_pandas_columns(df)
        assert "streaming_tv" in result.columns
        assert "customer_id" in result.columns

    def test_original_df_not_mutated(self):
        from churn.data_source import _normalize_pandas_columns

        original_cols = ["MonthlyCharges"]
        df = pd.DataFrame(columns=original_cols)
        _normalize_pandas_columns(df)
        assert list(df.columns) == original_cols  # original unchanged

    def test_public_alias_normalize_column_names(self):
        """normalize_column_names is a backwards-compat alias."""
        from churn.data_source import normalize_column_names

        df = pd.DataFrame(columns=["MonthlyCharges"])
        result = normalize_column_names(df)
        assert "monthly_charges" in result.columns


# ===========================================================================
# unity_catalog_table source
# ===========================================================================

class TestReadFromUnityCatalog:
    def test_calls_spark_table_with_source_table(self, spark, churn_config):
        from churn.data_source import read_from_unity_catalog

        churn_config.data_source.type = "unity_catalog_table"
        churn_config.data_source.source_table = "lob_cat.lob_schema.customers"
        churn_config.data_source.normalize_columns = False

        mock_df = spark.createDataFrame([{"customer_id": "C001", "tenure": 12}])
        with patch.object(spark, "table", return_value=mock_df) as mock_table:
            result = read_from_unity_catalog(spark, churn_config.data_source)
            mock_table.assert_called_once_with("lob_cat.lob_schema.customers")

    def test_raises_when_source_table_empty(self, spark, churn_config):
        from churn.data_source import read_from_unity_catalog

        churn_config.data_source.type = "unity_catalog_table"
        churn_config.data_source.source_table = ""

        with pytest.raises(ValueError, match="source_table must be set"):
            read_from_unity_catalog(spark, churn_config.data_source)

    def test_normalisation_applied_when_enabled(self, spark, churn_config):
        from churn.data_source import read_from_unity_catalog

        churn_config.data_source.type = "unity_catalog_table"
        churn_config.data_source.source_table = "lob.sch.tbl"
        churn_config.data_source.normalize_columns = True

        # Source table has PascalCase column
        mock_df = spark.createDataFrame([{"MonthlyCharges": 29.85}])
        with patch.object(spark, "table", return_value=mock_df):
            result = read_from_unity_catalog(spark, churn_config.data_source)
            assert "monthly_charges" in result.columns
            assert "MonthlyCharges" not in result.columns

    def test_normalisation_skipped_when_disabled(self, spark, churn_config):
        from churn.data_source import read_from_unity_catalog

        churn_config.data_source.type = "unity_catalog_table"
        churn_config.data_source.source_table = "lob.sch.tbl"
        churn_config.data_source.normalize_columns = False

        mock_df = spark.createDataFrame([{"MonthlyCharges": 29.85}])
        with patch.object(spark, "table", return_value=mock_df):
            result = read_from_unity_catalog(spark, churn_config.data_source)
            # Without normalisation, original names preserved
            assert "MonthlyCharges" in result.columns


# ===========================================================================
# volume_csv source
# ===========================================================================

class TestReadFromVolumeCsv:
    def test_raises_when_volume_path_empty(self, spark, churn_config):
        from churn.data_source import read_from_volume_csv

        churn_config.data_source.type = "volume_csv"
        churn_config.data_source.volume_path = ""

        with pytest.raises(ValueError, match="volume_path must be set"):
            read_from_volume_csv(spark, churn_config.data_source)

    def test_calls_spark_csv_reader_with_volume_path(self, spark, churn_config):
        from churn.data_source import read_from_volume_csv

        churn_config.data_source.type = "volume_csv"
        churn_config.data_source.volume_path = "/Volumes/main/data/telco.csv"
        churn_config.data_source.csv_options = {"header": "true"}
        churn_config.data_source.normalize_columns = False

        fake_df = spark.createDataFrame([{"customer_id": "C001"}])
        mock_reader = MagicMock()
        mock_reader.option.return_value = mock_reader
        mock_reader.load.return_value = fake_df

        with patch.object(spark.read, "format", return_value=mock_reader):
            result = read_from_volume_csv(spark, churn_config.data_source)
            mock_reader.load.assert_called_once_with("/Volumes/main/data/telco.csv")

    def test_csv_options_applied(self, spark, churn_config):
        from churn.data_source import read_from_volume_csv

        churn_config.data_source.type = "volume_csv"
        churn_config.data_source.volume_path = "/Volumes/main/data/telco.csv"
        churn_config.data_source.csv_options = {"header": "true", "inferSchema": "true"}
        churn_config.data_source.normalize_columns = False

        fake_df = spark.createDataFrame([{"customer_id": "C001"}])
        mock_reader = MagicMock()
        mock_reader.option.return_value = mock_reader
        mock_reader.load.return_value = fake_df

        with patch.object(spark.read, "format", return_value=mock_reader):
            read_from_volume_csv(spark, churn_config.data_source)
            # Both options should have been set
            option_calls = [c[0][0] for c in mock_reader.option.call_args_list]
            assert "header" in option_calls
            assert "inferSchema" in option_calls


# ===========================================================================
# http_csv source
# ===========================================================================

class TestReadFromHttpCsv:
    _SAMPLE_CSV = (
        "customerID,gender,SeniorCitizen,Partner,Dependents,tenure,"
        "PhoneService,MultipleLines,InternetService,OnlineSecurity,"
        "OnlineBackup,DeviceProtection,TechSupport,StreamingTV,"
        "StreamingMovies,Contract,PaperlessBilling,PaymentMethod,"
        "MonthlyCharges,TotalCharges,Churn\n"
        "7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,"
        "Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No\n"
    )

    def test_downloads_and_creates_spark_df(self, spark, churn_config):
        from churn.data_source import read_from_http_csv

        churn_config.data_source.type = "http_csv"
        churn_config.data_source.url = "http://example.com/data.csv"
        churn_config.data_source.normalize_columns = False

        mock_resp = MagicMock()
        mock_resp.text = self._SAMPLE_CSV
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            result = read_from_http_csv(spark, churn_config.data_source)

        assert result.count() == 1

    def test_fallback_url_tried_on_failure(self, spark, churn_config):
        import requests as req_lib

        from churn.data_source import read_from_http_csv

        churn_config.data_source.type = "http_csv"
        churn_config.data_source.url = "http://primary.example.com/data.csv"
        churn_config.data_source.normalize_columns = False

        call_count = 0

        def _side_effect(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise req_lib.ConnectionError("primary down")
            mock_resp = MagicMock()
            mock_resp.text = self._SAMPLE_CSV
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        with patch("requests.get", side_effect=_side_effect):
            result = read_from_http_csv(spark, churn_config.data_source)

        assert result.count() == 1
        assert call_count == 2  # primary + S3 fallback

    def test_raises_when_all_urls_fail(self, spark, churn_config):
        import requests as req_lib

        from churn.data_source import read_from_http_csv

        churn_config.data_source.type = "http_csv"
        churn_config.data_source.url = "http://bad.example.com/data.csv"
        churn_config.data_source.normalize_columns = False

        with patch("requests.get", side_effect=req_lib.ConnectionError("all down")):
            with pytest.raises(RuntimeError, match="Could not download"):
                read_from_http_csv(spark, churn_config.data_source)

    def test_normalisation_applied(self, spark, churn_config):
        from churn.data_source import read_from_http_csv

        churn_config.data_source.type = "http_csv"
        churn_config.data_source.normalize_columns = True

        mock_resp = MagicMock()
        mock_resp.text = self._SAMPLE_CSV
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            result = read_from_http_csv(spark, churn_config.data_source)

        # IBM dataset has "customerID" which should become "customer_id"
        assert "customer_id" in result.columns


# ===========================================================================
# get_source_dataframe dispatcher
# ===========================================================================

class TestGetSourceDataframe:
    def test_dispatches_to_unity_catalog_reader(self, spark, churn_config):
        from churn.data_source import get_source_dataframe

        churn_config.data_source.type = "unity_catalog_table"
        churn_config.data_source.source_table = "lob.sch.tbl"

        fake_df = spark.createDataFrame([{"customer_id": "C001"}])
        with patch("churn.data_source.read_from_unity_catalog", return_value=fake_df) as mock_fn:
            get_source_dataframe(spark, churn_config)
            mock_fn.assert_called_once()

    def test_dispatches_to_volume_csv_reader(self, spark, churn_config):
        from churn.data_source import get_source_dataframe

        churn_config.data_source.type = "volume_csv"
        churn_config.data_source.volume_path = "/Volumes/main/data/f.csv"

        fake_df = spark.createDataFrame([{"customer_id": "C001"}])
        with patch("churn.data_source.read_from_volume_csv", return_value=fake_df) as mock_fn:
            get_source_dataframe(spark, churn_config)
            mock_fn.assert_called_once()

    def test_dispatches_to_http_csv_reader(self, spark, churn_config):
        from churn.data_source import get_source_dataframe

        churn_config.data_source.type = "http_csv"

        fake_df = spark.createDataFrame([{"customer_id": "C001"}])
        with patch("churn.data_source.read_from_http_csv", return_value=fake_df) as mock_fn:
            get_source_dataframe(spark, churn_config)
            mock_fn.assert_called_once()

    def test_raises_on_unknown_source_type(self, spark, churn_config):
        from churn.config import DataSourceConfig
        from churn.data_source import get_source_dataframe

        # Bypass __post_init__ validation to test the dispatcher's own guard
        churn_config.data_source = object.__new__(DataSourceConfig)
        churn_config.data_source.type = "unknown_source"

        with pytest.raises(ValueError, match="Unknown data_source.type"):
            get_source_dataframe(spark, churn_config)


# ===========================================================================
# DataSourceConfig validation
# ===========================================================================

class TestDataSourceConfig:
    def test_valid_types_accepted(self):
        from churn.config import DataSourceConfig

        for t in ["unity_catalog_table", "volume_csv", "http_csv"]:
            cfg = DataSourceConfig(type=t)
            assert cfg.type == t

    def test_invalid_type_raises(self):
        from churn.config import DataSourceConfig

        with pytest.raises(ValueError, match="not valid"):
            DataSourceConfig(type="s3_parquet")

    def test_default_type_is_volume_csv(self):
        from churn.config import DataSourceConfig

        cfg = DataSourceConfig()
        assert cfg.type == "volume_csv"

    def test_normalize_columns_defaults_true(self):
        from churn.config import DataSourceConfig

        cfg = DataSourceConfig(type="volume_csv")
        assert cfg.normalize_columns is True
