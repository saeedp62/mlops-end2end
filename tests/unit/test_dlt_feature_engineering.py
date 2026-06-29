"""
tests/unit/test_dlt_feature_engineering.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ``pipelines/dlt_feature_engineering.py``.

Uses the same importlib + dlt-mock approach as ``test_dlt_ingestion.py``.

What is tested here (not covered in test_feature_engineering.py):
- ``silver_churn_features()`` – dlt.read dependency, transforms applied, label dropped
- ``silver_churn_labels()``   – correct dlt.read calls (feature table + bronze),
                                join produces the right columns, split values valid
- DLT decorator wiring        – table / expect / expect_or_drop call counts
- Config resolution            – same _get_config logic as ingestion pipeline
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from pyspark.sql import functions as F

# ---------------------------------------------------------------------------
# Infrastructure helpers (same pattern as test_dlt_ingestion.py)
# ---------------------------------------------------------------------------

_PIPELINES_DIR = Path(__file__).parents[2] / "pipelines"


def _make_dlt_mock() -> MagicMock:
    """Build a mock ``dlt`` module whose decorators are transparent no-ops."""
    m = MagicMock(name="dlt")
    m.table.return_value = lambda fn: fn
    m.expect.return_value = lambda fn: fn
    m.expect_or_drop.return_value = lambda fn: fn
    m.read = MagicMock(name="dlt.read")
    return m


def _load_fe_module(
    spark_mock: MagicMock,
    dlt_mock: MagicMock,
    *,
    config_path: str = "",
) -> object:
    """Load ``dlt_feature_engineering.py`` with mocked runtime globals."""
    spark_mock.conf.get.return_value = config_path

    path = _PIPELINES_DIR / "dlt_feature_engineering.py"
    spec = importlib.util.spec_from_file_location("dlt_feature_engineering", path)
    mod = importlib.util.module_from_spec(spec)
    mod.spark = spark_mock

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
    m = MagicMock(name="spark")
    m.conf.get.return_value = ""
    return m


@pytest.fixture()
def fe_module(spark_conf_mock, dlt_mock) -> object:
    """Pre-loaded dlt_feature_engineering module with default config."""
    return _load_fe_module(spark_conf_mock, dlt_mock)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_bronze_validated_name(self, fe_module):
        """_BRONZE_VALIDATED must be '<bronze_table>_validated'."""
        expected = f"{fe_module.cfg.bronze_table}_validated"
        assert fe_module._BRONZE_VALIDATED == expected

    def test_default_catalog_is_main(self, fe_module):
        assert fe_module.cfg.catalog == "main"

    def test_feature_table_name_accessible(self, fe_module):
        assert fe_module.cfg.feature_table == "advanced_churn_feature_table"

    def test_label_col_is_churn(self, fe_module):
        assert fe_module.cfg.label_col == "churn"


# ---------------------------------------------------------------------------
# silver_churn_features()
# ---------------------------------------------------------------------------

class TestSilverChurnFeatures:
    def test_reads_from_validated_bronze(self, spark, fe_module, dlt_mock):
        """Must call dlt.read with the validated bronze table name."""
        fake_df = spark.createDataFrame(
            [{"customer_id": "C001", "tenure": 12, "senior_citizen": 0,
              "total_charges": "358.5", "monthly_charges": 29.85,
              "online_security": "No", "online_backup": "Yes",
              "device_protection": "No", "tech_support": "No",
              "streaming_tv": "No", "streaming_movies": "No",
              "churn": "No"}]
        )
        dlt_mock.read.return_value = fake_df

        fe_module.silver_churn_features()

        dlt_mock.read.assert_called_with(fe_module._BRONZE_VALIDATED)

    def test_label_col_dropped(self, spark, fe_module, dlt_mock):
        """``churn`` column must not appear in the silver feature table."""
        fake_df = spark.createDataFrame(
            [{"customer_id": "C001", "tenure": 12.0, "senior_citizen": 0,
              "total_charges": "358.5", "monthly_charges": 29.85,
              "online_security": "No", "online_backup": "Yes",
              "device_protection": "No", "tech_support": "No",
              "streaming_tv": "No", "streaming_movies": "No",
              "churn": "No"}]
        )
        dlt_mock.read.return_value = fake_df

        result = fe_module.silver_churn_features()

        assert fe_module.cfg.label_col not in result.columns

    def test_num_optional_services_added(self, spark, fe_module, dlt_mock):
        """Feature transform must compute ``num_optional_services``."""
        fake_df = spark.createDataFrame(
            [{"customer_id": "C001", "tenure": 12.0, "senior_citizen": 0,
              "total_charges": "358.5", "monthly_charges": 29.85,
              "online_security": "Yes", "online_backup": "Yes",
              "device_protection": "No", "tech_support": "No",
              "streaming_tv": "Yes", "streaming_movies": "No",
              "churn": "No"}]
        )
        dlt_mock.read.return_value = fake_df

        result = fe_module.silver_churn_features()

        assert "num_optional_services" in result.columns
        row = result.collect()[0]
        assert row["num_optional_services"] == 3.0  # Yes for online_security, online_backup, streaming_tv

    def test_transaction_ts_added(self, spark, fe_module, dlt_mock):
        """``transaction_ts`` snapshot column must be appended."""
        fake_df = spark.createDataFrame(
            [{"customer_id": "C001", "tenure": 12.0, "senior_citizen": 0,
              "total_charges": "100.0", "monthly_charges": 29.85,
              "online_security": "No", "online_backup": "No",
              "device_protection": "No", "tech_support": "No",
              "streaming_tv": "No", "streaming_movies": "No",
              "churn": "No"}]
        )
        dlt_mock.read.return_value = fake_df

        result = fe_module.silver_churn_features()

        assert "transaction_ts" in result.columns
        assert dict(result.dtypes)["transaction_ts"] == "timestamp"

    def test_senior_citizen_cast_to_string(self, spark, fe_module, dlt_mock):
        """senior_citizen integer must be converted to 'Yes'/'No' string."""
        fake_df = spark.createDataFrame(
            [{"customer_id": "C001", "tenure": 5.0, "senior_citizen": 1,
              "total_charges": "50.0", "monthly_charges": 50.0,
              "online_security": "No", "online_backup": "No",
              "device_protection": "No", "tech_support": "No",
              "streaming_tv": "No", "streaming_movies": "No",
              "churn": "No"}]
        )
        dlt_mock.read.return_value = fake_df

        result = fe_module.silver_churn_features()

        assert dict(result.dtypes).get("senior_citizen") == "string"
        assert result.collect()[0]["senior_citizen"] == "Yes"

    def test_row_count_unchanged(self, spark, fe_module, dlt_mock):
        """Feature transforms must not change the row count."""
        rows = [
            {"customer_id": f"C{i:03d}", "tenure": float(i), "senior_citizen": 0,
             "total_charges": "100.0", "monthly_charges": 29.85,
             "online_security": "No", "online_backup": "No",
             "device_protection": "No", "tech_support": "No",
             "streaming_tv": "No", "streaming_movies": "No",
             "churn": "No"}
            for i in range(4)
        ]
        fake_df = spark.createDataFrame(rows)
        dlt_mock.read.return_value = fake_df

        result = fe_module.silver_churn_features()

        assert result.count() == 4


# ---------------------------------------------------------------------------
# silver_churn_labels()
# ---------------------------------------------------------------------------

class TestSilverChurnLabels:
    def _make_feature_df(self, spark, customer_ids: list[str]):
        """Minimal silver feature DataFrame (keys + transaction_ts)."""
        from datetime import datetime

        ts = datetime(2024, 1, 1)
        return spark.createDataFrame(
            [{"customer_id": cid, "transaction_ts": ts} for cid in customer_ids]
        )

    def _make_bronze_df(self, spark, customer_ids: list[str], label: str = "No"):
        """Minimal bronze DataFrame with just customer_id and churn."""
        return spark.createDataFrame(
            [{"customer_id": cid, "churn": label} for cid in customer_ids]
        )

    def _configure_dlt_reads(
        self,
        dlt_mock: MagicMock,
        feature_df,
        bronze_df,
        feature_table_name: str,
        bronze_validated_name: str,
    ) -> None:
        """Route dlt.read calls to the correct fake DataFrame by table name."""
        def _side_effect(table_name):
            if table_name == feature_table_name:
                return feature_df
            if table_name == bronze_validated_name:
                return bronze_df
            raise ValueError(f"Unexpected dlt.read call: {table_name!r}")

        dlt_mock.read.side_effect = _side_effect

    def test_reads_feature_table_for_keys(self, spark, fe_module, dlt_mock):
        """Must call dlt.read(cfg.feature_table) to get customer_id + transaction_ts."""
        feature_df = self._make_feature_df(spark, ["C001", "C002"])
        bronze_df = self._make_bronze_df(spark, ["C001", "C002"])

        self._configure_dlt_reads(
            dlt_mock, feature_df, bronze_df,
            fe_module.cfg.feature_table, fe_module._BRONZE_VALIDATED,
        )

        fe_module.silver_churn_labels()

        # dlt.read must have been called with the feature table
        read_calls = [c.args[0] for c in dlt_mock.read.call_args_list]
        assert fe_module.cfg.feature_table in read_calls

    def test_reads_bronze_for_label(self, spark, fe_module, dlt_mock):
        """Must call dlt.read(_BRONZE_VALIDATED) to get the churn label."""
        feature_df = self._make_feature_df(spark, ["C001"])
        bronze_df = self._make_bronze_df(spark, ["C001"])

        self._configure_dlt_reads(
            dlt_mock, feature_df, bronze_df,
            fe_module.cfg.feature_table, fe_module._BRONZE_VALIDATED,
        )

        fe_module.silver_churn_labels()

        read_calls = [c.args[0] for c in dlt_mock.read.call_args_list]
        assert fe_module._BRONZE_VALIDATED in read_calls

    def test_output_has_split_column(self, spark, fe_module, dlt_mock):
        """Output must include a ``split`` column."""
        feature_df = self._make_feature_df(spark, ["C001", "C002", "C003"])
        bronze_df = self._make_bronze_df(spark, ["C001", "C002", "C003"])

        self._configure_dlt_reads(
            dlt_mock, feature_df, bronze_df,
            fe_module.cfg.feature_table, fe_module._BRONZE_VALIDATED,
        )

        result = fe_module.silver_churn_labels()

        assert "split" in result.columns

    def test_split_values_only_train_or_test(self, spark, fe_module, dlt_mock):
        """``split`` column must only contain 'train' or 'test'."""
        # Use enough rows to get both split values with high probability
        ids = [f"C{i:04d}" for i in range(50)]
        feature_df = self._make_feature_df(spark, ids)
        bronze_df = self._make_bronze_df(spark, ids)

        self._configure_dlt_reads(
            dlt_mock, feature_df, bronze_df,
            fe_module.cfg.feature_table, fe_module._BRONZE_VALIDATED,
        )

        result = fe_module.silver_churn_labels()

        split_values = {r["split"] for r in result.select("split").collect()}
        assert split_values.issubset({"train", "test"})

    def test_output_has_label_col(self, spark, fe_module, dlt_mock):
        """Label column (``churn``) must be present in the output."""
        feature_df = self._make_feature_df(spark, ["C001"])
        bronze_df = self._make_bronze_df(spark, ["C001"], label="Yes")

        self._configure_dlt_reads(
            dlt_mock, feature_df, bronze_df,
            fe_module.cfg.feature_table, fe_module._BRONZE_VALIDATED,
        )

        result = fe_module.silver_churn_labels()

        assert fe_module.cfg.label_col in result.columns

    def test_output_has_transaction_ts(self, spark, fe_module, dlt_mock):
        """``transaction_ts`` from the feature table must appear in the label table."""
        feature_df = self._make_feature_df(spark, ["C001"])
        bronze_df = self._make_bronze_df(spark, ["C001"])

        self._configure_dlt_reads(
            dlt_mock, feature_df, bronze_df,
            fe_module.cfg.feature_table, fe_module._BRONZE_VALIDATED,
        )

        result = fe_module.silver_churn_labels()

        assert "transaction_ts" in result.columns

    def test_join_matches_on_customer_id(self, spark, fe_module, dlt_mock):
        """Customers in the feature table that have no bronze row get null label."""
        # Feature table has C001, C002; bronze only has C001
        feature_df = self._make_feature_df(spark, ["C001", "C002"])
        bronze_df = self._make_bronze_df(spark, ["C001"])

        self._configure_dlt_reads(
            dlt_mock, feature_df, bronze_df,
            fe_module.cfg.feature_table, fe_module._BRONZE_VALIDATED,
        )

        result = fe_module.silver_churn_labels()

        # Both rows should be present (left join)
        assert result.count() == 2

        # C002 should have null churn
        c002_row = result.filter(F.col("customer_id") == "C002").collect()
        assert len(c002_row) == 1
        assert c002_row[0][fe_module.cfg.label_col] is None

    def test_row_count_matches_feature_table(self, spark, fe_module, dlt_mock):
        """Output row count must match the feature table (left join semantics)."""
        ids = [f"C{i:03d}" for i in range(10)]
        feature_df = self._make_feature_df(spark, ids)
        bronze_df = self._make_bronze_df(spark, ids)

        self._configure_dlt_reads(
            dlt_mock, feature_df, bronze_df,
            fe_module.cfg.feature_table, fe_module._BRONZE_VALIDATED,
        )

        result = fe_module.silver_churn_labels()

        assert result.count() == 10

    def test_split_is_deterministic_with_seed(self, spark, fe_module, dlt_mock):
        """Same data + same rng_seed must produce the same train/test assignment."""
        ids = [f"C{i:04d}" for i in range(20)]
        feature_df = self._make_feature_df(spark, ids)
        bronze_df = self._make_bronze_df(spark, ids)

        def _read_side_effect(table_name):
            if table_name == fe_module.cfg.feature_table:
                return feature_df
            return bronze_df

        dlt_mock.read.side_effect = _read_side_effect
        result1 = fe_module.silver_churn_labels()
        splits1 = {r["customer_id"]: r["split"] for r in result1.collect()}

        dlt_mock.read.side_effect = _read_side_effect
        result2 = fe_module.silver_churn_labels()
        splits2 = {r["customer_id"]: r["split"] for r in result2.collect()}

        assert splits1 == splits2


# ---------------------------------------------------------------------------
# DLT decorator wiring sanity checks
# ---------------------------------------------------------------------------

class TestDltDecoratorWiring:
    def test_dlt_table_called_twice(self, fe_module, dlt_mock):
        """Two @dlt.table decorators should be applied (features + labels)."""
        assert dlt_mock.table.call_count >= 2

    def test_dlt_expect_or_drop_applied(self, fe_module, dlt_mock):
        """At least one @dlt.expect_or_drop applied to each table (4 total)."""
        assert dlt_mock.expect_or_drop.call_count >= 4

    def test_dlt_expect_applied(self, fe_module, dlt_mock):
        """At least one @dlt.expect advisory check applied."""
        assert dlt_mock.expect.call_count >= 1

    def test_feature_table_name_in_dlt_table_calls(self, fe_module, dlt_mock):
        """@dlt.table must be called with the feature table name."""
        table_names = [
            kw.get("name")
            for _, kw in [c.args, c.kwargs]
            if isinstance(kw, dict)
            for c in dlt_mock.table.call_args_list
        ]
        # Just verify dlt.table was called with keyword args that contain names
        all_kwargs = [c.kwargs for c in dlt_mock.table.call_args_list]
        names_used = [kw.get("name", "") for kw in all_kwargs]
        assert fe_module.cfg.feature_table in names_used

    def test_label_table_name_in_dlt_table_calls(self, fe_module, dlt_mock):
        """@dlt.table must be called with the label table name."""
        all_kwargs = [c.kwargs for c in dlt_mock.table.call_args_list]
        names_used = [kw.get("name", "") for kw in all_kwargs]
        assert fe_module.cfg.label_table in names_used
