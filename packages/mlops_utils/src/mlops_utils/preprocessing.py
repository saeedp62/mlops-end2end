"""
mlops_utils.preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~
Generic scikit-learn pipeline component builders.

These builder functions return ``(name, pipeline, columns)`` tuples that can be
plugged directly into ``sklearn.compose.ColumnTransformer``.  The builders are
intentionally parameterised so they can serve any tabular ML project, not just
the churn demo.

Public API
----------
::

    from mlops_utils.preprocessing import (
        build_boolean_pipeline,
        build_numerical_pipeline,
        build_categorical_ohe_pipeline,
        build_column_transformer,
    )
"""

from __future__ import annotations

from mlops_utils.logger import get_logger
from typing import Optional

import numpy as np

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Individual pipeline builders
# ---------------------------------------------------------------------------

def build_boolean_pipeline(
    columns: list[str],
    *,
    imputer_strategy: str = "most_frequent",
    drop_first: bool = True,
) -> tuple[str, "Pipeline", list[str]]:  # type: ignore[name-defined]  # noqa: F821
    """Build a pipeline for binary (Yes/No) columns.

    Steps:
    1. Cast to ``object`` dtype so ``OneHotEncoder`` treats them as strings.
    2. ``SimpleImputer`` with *imputer_strategy*.
    3. ``OneHotEncoder`` (``drop='first'`` to avoid multicollinearity).

    Parameters
    ----------
    columns:
        List of column names to apply this pipeline to.
    imputer_strategy:
        Strategy passed to ``SimpleImputer``.
    drop_first:
        Whether to drop one dummy column per feature (avoid dummy trap).

    Returns
    -------
    ``(name, pipeline, columns)`` – ready for ``ColumnTransformer``.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

    pipeline = Pipeline(
        steps=[
            ("cast_type", FunctionTransformer(lambda df: df.astype(object))),
            (
                "imputer",
                ColumnTransformer(
                    [("imputer", SimpleImputer(strategy=imputer_strategy), columns)],
                    remainder="passthrough",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first" if drop_first else None,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ("boolean", pipeline, columns)


def build_numerical_pipeline(
    columns: list[str],
    *,
    imputer_strategy: str = "mean",
    scale: bool = True,
) -> tuple[str, "Pipeline", list[str]]:  # type: ignore[name-defined]  # noqa: F821
    """Build a pipeline for numerical columns.

    Steps:
    1. ``FunctionTransformer`` coercing to numeric (handles object columns).
    2. ``ColumnTransformer``-based imputation.
    3. Optionally ``StandardScaler``.

    Parameters
    ----------
    columns:
        List of column names.
    imputer_strategy:
        Strategy for ``SimpleImputer`` (``"mean"``, ``"median"``, etc.).
    scale:
        Whether to apply ``StandardScaler`` after imputation.

    Returns
    -------
    ``(name, pipeline, columns)`` – ready for ``ColumnTransformer``.
    """
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    steps: list[tuple] = [
        (
            "to_numeric",
            FunctionTransformer(
                lambda df: df.apply(pd.to_numeric, errors="coerce"), validate=False
            ),
        ),
        (
            "imputer",
            ColumnTransformer(
                [("impute", SimpleImputer(strategy=imputer_strategy), list(range(len(columns))))],
                remainder="passthrough",
            ),
        ),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))

    return ("numerical", Pipeline(steps=steps), columns)


def build_categorical_ohe_pipeline(
    columns: list[str],
    *,
    imputer_strategy: str = "most_frequent",
    handle_unknown: str = "ignore",
) -> tuple[str, "Pipeline", list[str]]:  # type: ignore[name-defined]  # noqa: F821
    """Build a pipeline for low-cardinality categorical columns (one-hot encoding).

    Parameters
    ----------
    columns:
        List of column names.
    imputer_strategy:
        Strategy for ``SimpleImputer``.
    handle_unknown:
        Passed to ``OneHotEncoder`` – ``"ignore"`` keeps unseen categories as
        zeros at inference time.

    Returns
    -------
    ``(name, pipeline, columns)`` – ready for ``ColumnTransformer``.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                ColumnTransformer(
                    [("impute", SimpleImputer(strategy=imputer_strategy), list(range(len(columns))))],
                    remainder="passthrough",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False),
            ),
        ]
    )
    return ("categorical_ohe", pipeline, columns)


# ---------------------------------------------------------------------------
# Composite builder
# ---------------------------------------------------------------------------

def build_column_transformer(
    bool_cols: list[str],
    num_cols: list[str],
    cat_cols: list[str],
    *,
    remainder: str = "drop",
    sparse_threshold: float = 0.0,
    # Per-pipeline overrides
    bool_kwargs: Optional[dict] = None,
    num_kwargs: Optional[dict] = None,
    cat_kwargs: Optional[dict] = None,
) -> "ColumnTransformer":  # type: ignore[name-defined]  # noqa: F821
    """Compose a full ``ColumnTransformer`` from three column groups.

    Parameters
    ----------
    bool_cols:
        Boolean / binary columns (Yes/No, 0/1).
    num_cols:
        Numerical columns.
    cat_cols:
        Low-cardinality categorical columns.
    remainder:
        How to handle columns not in any of the three lists.
        ``"drop"`` (default) or ``"passthrough"``.
    sparse_threshold:
        Passed to ``ColumnTransformer`` – set to ``0.0`` to force dense output.
    bool_kwargs / num_kwargs / cat_kwargs:
        Optional keyword-argument overrides forwarded to the respective
        builder function.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Configured (but **not** fitted) transformer.
    """
    from sklearn.compose import ColumnTransformer

    transformers: list[tuple] = []

    if bool_cols:
        transformers.append(build_boolean_pipeline(bool_cols, **(bool_kwargs or {})))
    if num_cols:
        transformers.append(build_numerical_pipeline(num_cols, **(num_kwargs or {})))
    if cat_cols:
        transformers.append(build_categorical_ohe_pipeline(cat_cols, **(cat_kwargs or {})))

    ct = ColumnTransformer(
        transformers=transformers,
        remainder=remainder,
        sparse_threshold=sparse_threshold,
    )
    logger.debug(
        "Built ColumnTransformer: %d bool, %d num, %d cat cols.",
        len(bool_cols),
        len(num_cols),
        len(cat_cols),
    )
    return ct
