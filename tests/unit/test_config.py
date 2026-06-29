"""
tests/unit/test_config.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ``churn.config``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest


class TestChurnConfigProperties:
    def test_full_bronze_table(self):
        from churn.config import ChurnConfig

        cfg = ChurnConfig(catalog="cat", db="sch", bronze_table="raw_customers")
        assert cfg.full_bronze_table == "cat.sch.raw_customers"

    def test_full_feature_table(self):
        from churn.config import ChurnConfig

        cfg = ChurnConfig(catalog="main", db="ml_db", feature_table="feat_tbl")
        assert cfg.full_feature_table == "main.ml_db.feat_tbl"

    def test_full_model_name(self):
        from churn.config import ChurnConfig

        cfg = ChurnConfig(catalog="prod", db="ml", model_name="churn_model")
        assert cfg.full_model_name == "prod.ml.churn_model"

    def test_full_experiment_name(self):
        from churn.config import ChurnConfig

        cfg = ChurnConfig(
            experiment_path="/Users/joe/mlops",
            experiment_name="churn_exp",
        )
        assert cfg.full_experiment_name == "/Users/joe/mlops/churn_exp"

    def test_pk_constraint_name(self):
        from churn.config import ChurnConfig

        cfg = ChurnConfig(label_table="my_labels")
        assert cfg.label_table_pk_constraint == "my_labels_pk"


class TestLoadChurnConfig:
    def _write_config(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "test_config.yaml"
        p.write_text(textwrap.dedent(content))
        return p

    def test_loads_catalog_and_db(self, tmp_path):
        from churn.config import load_churn_config

        cfg_file = self._write_config(tmp_path, """
            catalog: test_catalog
            db: test_schema
        """)
        cfg = load_churn_config(cfg_file)
        assert cfg.catalog == "test_catalog"
        assert cfg.db == "test_schema"

    def test_primary_keys_loaded_as_list(self, tmp_path):
        from churn.config import load_churn_config

        cfg_file = self._write_config(tmp_path, """
            catalog: main
            db: ml
            primary_keys:
              - customer_id
              - transaction_ts
        """)
        cfg = load_churn_config(cfg_file)
        assert cfg.primary_keys == ["customer_id", "transaction_ts"]

    def test_primary_keys_from_comma_string(self, tmp_path):
        from churn.config import load_churn_config

        cfg_file = self._write_config(tmp_path, """
            catalog: main
            db: ml
            primary_keys: "customer_id, transaction_ts"
        """)
        cfg = load_churn_config(cfg_file)
        assert cfg.primary_keys == ["customer_id", "transaction_ts"]

    def test_online_store_disabled_by_default(self, tmp_path):
        from churn.config import load_churn_config

        cfg_file = self._write_config(tmp_path, """
            catalog: main
            db: ml
        """)
        cfg = load_churn_config(cfg_file)
        assert cfg.online_store.enabled is False

    def test_online_store_enabled_from_yaml(self, tmp_path):
        from churn.config import load_churn_config

        cfg_file = self._write_config(tmp_path, """
            catalog: main
            db: ml
            online_store:
              enabled: true
              backend: dynamodb
              endpoint_name: my_endpoint
        """)
        cfg = load_churn_config(cfg_file)
        assert cfg.online_store.enabled is True
        assert cfg.online_store.backend == "dynamodb"
        assert cfg.online_store.endpoint_name == "my_endpoint"

    def test_env_var_override(self, tmp_path, monkeypatch):
        from churn.config import load_churn_config

        monkeypatch.setenv("MLOPS_CATALOG", "env_catalog")
        cfg_file = self._write_config(tmp_path, """
            catalog: yaml_catalog
            db: ml
        """)
        cfg = load_churn_config(cfg_file)
        assert cfg.catalog == "env_catalog"

    def test_file_not_found_raises(self):
        from churn.config import load_churn_config

        with pytest.raises(FileNotFoundError):
            load_churn_config("/nonexistent/path/config.yaml")

    def test_train_ratio_defaults_to_0_8(self, tmp_path):
        from churn.config import load_churn_config

        cfg_file = self._write_config(tmp_path, "catalog: main\ndb: ml\n")
        cfg = load_churn_config(cfg_file)
        assert cfg.train_ratio == 0.8
