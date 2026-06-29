"""
tests/unit/test_dlt_ingestion.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ``pipelines/dlt_ingestion.py``.

Since the ``dlt`` module only exists inside the Databricks runtime, we:
1. Build a transparent no-op ``dlt`` mock (decorators pass functions through).
2. Load the pipeline file via ``importlib`` with the mock injected into
   ``sys.modules['dlt']`` and a fake ``spark`` bound in the module globals.
3. Call the decorated table functions directly and assert on their output.

What is tested here (not covered elsewhere):
- ``_get_config()`` – resolves config from spark conf param or falls back
- ``bronze_customers()`` – calls ``get_source_dataframe``, appends ``_ingested_at``
- ``bronze_customers_validated()`` – delegates to ``dlt.read``
- ``_SOURCE_DESCRIPTION`` – correct human-readable label per source type
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Infrastructure helpers
# ---------------------------------------------------------------------------

_PIPELINES_DIR = Path(__file__).parents[2] / "pipelines"


def _make_dlt_mock() -> MagicMock:
    """Build a mock ``dlt`` module whose decorators are transparent no-ops."""
    m = MagicMock(name="dlt")
    m.table.return_value = lambda fn: fn          # @dlt.table(...) passes fn through
    m.expect.return_value = lambda fn: fn         # @dlt.expect(...) passes fn through
    m.expect_or_drop.return_value = lambda fn: fn # @dlt.expect_or_drop(...) passes fn through
    m.read = MagicMock(name="dlt.read")
    return m


def _load_ingestion_module(
    spark_mock: MagicMock,
    dlt_mock: MagicMock,
    *,
    config_path: str = "",
) -> object:
    """Load ``dlt_ingestion.py`` with mocked runtime globals.

    Parameters
    ----------
    spark_mock:
        Fake ``spark`` session injected as a global (simulates DLT runtime).
    dlt_mock:
        Mock ``dlt`` module registered in ``sys.modules``.
    config_path:
        Value returned by ``spark.conf.get("config_path", "")``; empty string
        triggers fallback to ``ChurnConfig`` defaults.
    """
    spark_mock.conf.get.return_value = config_path

    path = _PIPELINES_DIR / "dlt_ingestion.py"
    spec = importlib.util.spec_from_file_location("dlt_ingestion", path)
    mod = importlib.util.module_from_spec(spec)
    mod.spark = spark_mock  # inject DLT runtime global

    old = sys.modules.get("dlt")
    sys.modules["dlt"] = dlt_mock
    try:
        spec.loader.exec_module(mod)
    finally:
        if old is None:
            sys.modules.pop("dlt", None)
        else:
            sys.modules["dlt"] = old

    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def dlt_mock() -> MagicMock:
    return _make_dlt_mock()


@pytest.fixture()
def spark_conf_mock() -> MagicMock:
    """Minimal SparkSession mock that returns '' for all conf reads."""
    m = MagicMock(name="spark")
    m.conf.get.return_value = ""
    return m


# ---------------------------------------------------------------------------
# _get_config()
# ---------------------------------------------------------------------------

class TestGetConfig:
    def test_falls_back_to_defaults_when_no_config_path(
        self, spark_conf_mock, dlt_mock
    ):
        """Empty spark conf → ChurnConfig() defaults used."""
        mod = _load_ingestion_module(spark_conf_mock, dlt_mock, config_path="")

        # The module-level cfg should be a ChurnConfig with default values
        assert mod.cfg.catalog == "main"
        assert mod.cfg.db == "dbdemos_mlops"

    def test_uses_config_path_from_spark_conf(
        self, spark_conf_mock, dlt_mock, tmp_path
    ):
        """When config_path is set in spark conf, that YAML is loaded."""
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("catalog: ci_catalog\ndb: ci_db\n")

        # Only the first conf.get call matters here
        spark_conf_mock.conf.get.return_value = str(cfg_file)

        mod = _load_ingestion_module(
            spark_conf_mock, dlt_mock, config_path=str(cfg_file)
        )
        assert mod.cfg.catalog == "ci_catalog"
        assert mod.cfg.db == "ci_db"

    def test_falls_back_gracefully_when_spark_raises(self, dlt_mock):
        """NameError on spark (DLT global unavailable) → still returns defaults."""
        # spark.conf.get raises AttributeError → caught by try/except in _get_config
        bad_spark = MagicMock()
        bad_spark.conf.get.side_effect = Exception("spark not available")

        mod = _load_ingestion_module(bad_spark, dlt_mock, config_path="")
        assert mod.cfg.catalog == "main"  # defaults intact


# ---------------------------------------------------------------------------
# _SOURCE_DESCRIPTION
# ---------------------------------------------------------------------------

class TestSourceDescription:
    def _load_with_source_type(
        self, source_type: str, spark_conf_mock, dlt_mock, tmp_path
    ) -> object:
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(
            f"catalog: main\ndb: ml\n"
            f"data_source:\n  type: {source_type}\n"
            f"  source_table: lob.sch.tbl\n"
            f"  volume_path: /Volumes/main/data/file.csv\n"
        )
        spark_conf_mock.conf.get.return_value = str(cfg_file)
        return _load_ingestion_module(
            spark_conf_mock, dlt_mock, config_path=str(cfg_file)
        )

    def test_unity_catalog_table_description(
        self, spark_conf_mock, dlt_mock, tmp_path
    ):
        mod = self._load_with_source_type(
            "unity_catalog_table", spark_conf_mock, dlt_mock, tmp_path
        )
        assert "lob.sch.tbl" in mod._SOURCE_DESCRIPTION
        assert "cross-catalog" in mod._SOURCE_DESCRIPTION

    def test_volume_csv_description(self, spark_conf_mock, dlt_mock, tmp_path):
        mod = self._load_with_source_type(
            "volume_csv", spark_conf_mock, dlt_mock, tmp_path
        )
        assert "/Volumes/main/data/file.csv" in mod._SOURCE_DESCRIPTION

    def test_http_csv_description(self, spark_conf_mock, dlt_mock, tmp_path):
        mod = self._load_with_source_type(
            "http_csv", spark_conf_mock, dlt_mock, tmp_path
        )
        assert "HTTP" in mod._SOURCE_DESCRIPTION or "http" in mod._SOURCE_DESCRIPTION.lower()


# ---------------------------------------------------------------------------
# bronze_customers()
# ---------------------------------------------------------------------------

class TestBronzeCustomers:
    def test_calls_get_source_dataframe(self, spark, spark_conf_mock, dlt_mock):
        """bronze_customers() must delegate to get_source_dataframe."""
        mod = _load_ingestion_module(spark_conf_mock, dlt_mock)

        fake_df = spark.createDataFrame([{"customer_id": "C001"}])
        with patch("churn.data_source.get_source_dataframe", return_value=fake_df) as mock_fn:
            mod.bronze_customers()
            mock_fn.assert_called_once()

    def test_adds_ingested_at_column(self, spark, spark_conf_mock, dlt_mock):
        """bronze_customers() must append a ``_ingested_at`` timestamp column."""
        mod = _load_ingestion_module(spark_conf_mock, dlt_mock)

        fake_df = spark.createDataFrame([{"customer_id": "C001", "tenure": 12}])
        with patch("churn.data_source.get_source_dataframe", return_value=fake_df):
            result = mod.bronze_customers()

        assert "_ingested_at" in result.columns

    def test_ingested_at_is_timestamp_type(self, spark, spark_conf_mock, dlt_mock):
        """``_ingested_at`` must be a timestamp column, not a string."""
        mod = _load_ingestion_module(spark_conf_mock, dlt_mock)

        fake_df = spark.createDataFrame([{"customer_id": "C001"}])
        with patch("churn.data_source.get_source_dataframe", return_value=fake_df):
            result = mod.bronze_customers()

        dtype_map = dict(result.dtypes)
        assert dtype_map["_ingested_at"] == "timestamp"

    def test_source_columns_preserved(self, spark, spark_conf_mock, dlt_mock):
        """All original source columns must still be present alongside _ingested_at."""
        mod = _load_ingestion_module(spark_conf_mock, dlt_mock)

        fake_df = spark.createDataFrame(
            [{"customer_id": "C001", "tenure": 12, "monthly_charges": 29.85}]
        )
        with patch("churn.data_source.get_source_dataframe", return_value=fake_df):
            result = mod.bronze_customers()

        assert "customer_id" in result.columns
        assert "tenure" in result.columns
        assert "monthly_charges" in result.columns

    def test_row_count_unchanged(self, spark, spark_conf_mock, dlt_mock):
        """Adding _ingested_at must not alter the row count."""
        mod = _load_ingestion_module(spark_conf_mock, dlt_mock)

        rows = [{"customer_id": f"C{i:03d}"} for i in range(5)]
        fake_df = spark.createDataFrame(rows)
        with patch("churn.data_source.get_source_dataframe", return_value=fake_df):
            result = mod.bronze_customers()

        assert result.count() == 5


# ---------------------------------------------------------------------------
# bronze_customers_validated()
# ---------------------------------------------------------------------------

class TestBronzeCustomersValidated:
    def test_reads_from_bronze_table(self, spark, spark_conf_mock, dlt_mock):
        """bronze_customers_validated() must call dlt.read with the bronze table name."""
        mod = _load_ingestion_module(spark_conf_mock, dlt_mock)

        expected_name = mod.cfg.bronze_table
        fake_df = spark.createDataFrame([{"customer_id": "C001"}])
        dlt_mock.read.return_value = fake_df

        mod.bronze_customers_validated()

        dlt_mock.read.assert_called_with(expected_name)

    def test_returns_dlt_read_result(self, spark, spark_conf_mock, dlt_mock):
        """The function must return the DataFrame produced by dlt.read unchanged."""
        mod = _load_ingestion_module(spark_conf_mock, dlt_mock)

        fake_df = spark.createDataFrame([{"customer_id": "C001"}])
        dlt_mock.read.return_value = fake_df

        result = mod.bronze_customers_validated()
        assert result is fake_df


# ---------------------------------------------------------------------------
# DLT decorator wiring sanity checks
# ---------------------------------------------------------------------------

class TestDltDecoratorWiring:
    def test_dlt_table_called_for_bronze(self, spark_conf_mock, dlt_mock):
        """@dlt.table must be applied to bronze_customers."""
        _load_ingestion_module(spark_conf_mock, dlt_mock)
        # dlt.table() is called twice (once per table) during module load
        assert dlt_mock.table.call_count >= 2

    def test_dlt_expect_or_drop_applied(self, spark_conf_mock, dlt_mock):
        """@dlt.expect_or_drop must be applied to the validated table."""
        _load_ingestion_module(spark_conf_mock, dlt_mock)
        assert dlt_mock.expect_or_drop.call_count >= 1

    def test_dlt_expect_applied(self, spark_conf_mock, dlt_mock):
        """@dlt.expect must be applied (advisory checks)."""
        _load_ingestion_module(spark_conf_mock, dlt_mock)
        assert dlt_mock.expect.call_count >= 1
