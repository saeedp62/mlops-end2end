"""
tests/unit/test_preprocessing.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ``mlops_utils.preprocessing`` and ``churn.preprocessing``.

These tests run pure sklearn logic – no Spark session required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# mlops_utils.preprocessing builders
# ---------------------------------------------------------------------------

class TestBuildBooleanPipeline:
    def test_returns_correct_tuple_structure(self):
        from mlops_utils.preprocessing import build_boolean_pipeline

        name, pipeline, columns = build_boolean_pipeline(["gender", "partner"])
        assert name == "boolean"
        assert isinstance(pipeline, Pipeline)
        assert columns == ["gender", "partner"]

    def test_transforms_yes_no_to_binary(self):
        from mlops_utils.preprocessing import build_boolean_pipeline

        _name, pipeline, cols = build_boolean_pipeline(["col_a"])
        df = pd.DataFrame({"col_a": ["Yes", "No", "Yes", "No"]})
        result = pipeline.fit_transform(df)
        # OneHotEncoder with drop='first' on 2 categories → 1 column
        assert result.shape == (4, 1)

    def test_handles_missing_values(self):
        from mlops_utils.preprocessing import build_boolean_pipeline

        _name, pipeline, cols = build_boolean_pipeline(["col_a"])
        df = pd.DataFrame({"col_a": ["Yes", None, "No"]})
        # Should not raise
        result = pipeline.fit_transform(df)
        assert result.shape[0] == 3


class TestBuildNumericalPipeline:
    def test_returns_correct_tuple_structure(self):
        from mlops_utils.preprocessing import build_numerical_pipeline

        name, pipeline, columns = build_numerical_pipeline(["tenure", "monthly_charges"])
        assert name == "numerical"
        assert isinstance(pipeline, Pipeline)
        assert columns == ["tenure", "monthly_charges"]

    def test_scales_output_when_scale_true(self):
        from mlops_utils.preprocessing import build_numerical_pipeline

        _name, pipeline, cols = build_numerical_pipeline(["a", "b"], scale=True)
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        result = pipeline.fit_transform(df)
        # After StandardScaler: mean ≈ 0
        assert abs(result.mean()) < 1e-6

    def test_no_scale_when_scale_false(self):
        from mlops_utils.preprocessing import build_numerical_pipeline

        _name, pipeline, cols = build_numerical_pipeline(["a"], scale=False)
        df = pd.DataFrame({"a": [10.0, 20.0, 30.0]})
        result = pipeline.fit_transform(df)
        # Values should be close to original (only imputed, not scaled)
        assert result.mean() == pytest.approx(20.0, abs=1e-3)

    def test_imputes_null_with_mean(self):
        from mlops_utils.preprocessing import build_numerical_pipeline

        _name, pipeline, cols = build_numerical_pipeline(["a"], scale=False)
        df = pd.DataFrame({"a": [10.0, None, 30.0]})
        result = pipeline.fit_transform(df)
        # Mean of [10, 30] = 20 → null should be filled with 20
        assert result[1, 0] == pytest.approx(20.0, abs=1e-3)


class TestBuildCategoricalOHEPipeline:
    def test_returns_correct_tuple_structure(self):
        from mlops_utils.preprocessing import build_categorical_ohe_pipeline

        name, pipeline, columns = build_categorical_ohe_pipeline(["contract"])
        assert name == "categorical_ohe"
        assert isinstance(pipeline, Pipeline)
        assert columns == ["contract"]

    def test_one_hot_encodes_categories(self):
        from mlops_utils.preprocessing import build_categorical_ohe_pipeline

        _name, pipeline, cols = build_categorical_ohe_pipeline(["contract"])
        df = pd.DataFrame({"contract": ["Month-to-month", "One year", "Two year"]})
        result = pipeline.fit_transform(df)
        # 3 unique categories → 3 OHE columns
        assert result.shape == (3, 3)

    def test_handle_unknown_does_not_raise(self):
        from mlops_utils.preprocessing import build_categorical_ohe_pipeline

        _name, pipeline, cols = build_categorical_ohe_pipeline(["contract"])
        df_train = pd.DataFrame({"contract": ["Month-to-month", "One year"]})
        pipeline.fit(df_train)
        df_test = pd.DataFrame({"contract": ["UNKNOWN_CONTRACT"]})
        # Should not raise with handle_unknown="ignore"
        result = pipeline.transform(df_test)
        assert result.shape[1] == 2  # 2 trained categories


class TestBuildColumnTransformer:
    def test_returns_column_transformer(self):
        from mlops_utils.preprocessing import build_column_transformer

        ct = build_column_transformer(
            bool_cols=["gender"],
            num_cols=["tenure"],
            cat_cols=["contract"],
        )
        assert isinstance(ct, ColumnTransformer)

    def test_empty_bool_cols_skipped(self):
        from mlops_utils.preprocessing import build_column_transformer

        ct = build_column_transformer(bool_cols=[], num_cols=["tenure"], cat_cols=[])
        # Only 1 transformer (numerical)
        assert len(ct.transformers) == 1

    def test_full_transformer_fits_churn_like_data(self):
        from mlops_utils.preprocessing import build_column_transformer

        ct = build_column_transformer(
            bool_cols=["gender"],
            num_cols=["tenure", "monthly_charges"],
            cat_cols=["contract"],
        )
        df = pd.DataFrame({
            "gender": ["Male", "Female", "Male"],
            "tenure": [12.0, 5.0, 0.0],
            "monthly_charges": [29.85, 56.95, 29.85],
            "contract": ["Month-to-month", "One year", "Two year"],
        })
        result = ct.fit_transform(df)
        assert result.shape[0] == 3  # 3 rows
        assert result.shape[1] > 0   # at least one feature column


# ---------------------------------------------------------------------------
# churn.preprocessing
# ---------------------------------------------------------------------------

class TestBuildChurnPreprocessor:
    def test_returns_column_transformer(self):
        from churn.preprocessing import build_churn_preprocessor

        ct = build_churn_preprocessor()
        assert isinstance(ct, ColumnTransformer)

    def test_override_bool_cols_respected(self):
        from churn.preprocessing import build_churn_preprocessor

        ct = build_churn_preprocessor(bool_cols=["gender"])
        # Only 1 boolean column → boolean transformer should have 1 column
        bool_transformer = next(
            t for name, t, cols in ct.transformers if name == "boolean"
        )
        assert bool_transformer is not None

    def test_fits_on_sample_churn_data(self):
        from churn.preprocessing import build_churn_preprocessor

        ct = build_churn_preprocessor()
        df = pd.DataFrame({
            "gender": ["Male", "Female"],
            "phone_service": ["Yes", "No"],
            "dependents": ["No", "Yes"],
            "senior_citizen": ["No", "Yes"],
            "paperless_billing": ["Yes", "No"],
            "partner": ["Yes", "No"],
            "monthly_charges": [29.85, 56.95],
            "total_charges": [358.5, 1889.5],
            "avg_price_increase": [0.5, -1.2],
            "tenure": [12.0, 34.0],
            "num_optional_services": [2.0, 4.0],
            "contract": ["Month-to-month", "One year"],
            "device_protection": ["No", "Yes"],
            "internet_service": ["DSL", "Fiber optic"],
            "multiple_lines": ["No", "Yes"],
            "online_backup": ["Yes", "No"],
            "online_security": ["No", "Yes"],
            "payment_method": ["Electronic check", "Mailed check"],
            "streaming_movies": ["No", "Yes"],
            "streaming_tv": ["No", "No"],
            "tech_support": ["No", "Yes"],
        })
        result = ct.fit_transform(df)
        assert result.shape[0] == 2
