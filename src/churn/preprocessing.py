"""
churn.preprocessing
~~~~~~~~~~~~~~~~~~~
Scikit-learn column definitions for the churn dataset.

This module maps the churn domain's concrete column names onto the generic
pipeline builders in ``mlops_utils.preprocessing``.

Usage::

    from churn.preprocessing import build_churn_preprocessor

    preprocessor = build_churn_preprocessor()
    # → sklearn.compose.ColumnTransformer (unfitted)
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer

from mlops_utils.preprocessing import build_column_transformer

# ---------------------------------------------------------------------------
# Column group definitions
# ---------------------------------------------------------------------------

#: Boolean (Yes/No) columns
BOOL_COLS: list[str] = [
    "gender",
    "phone_service",
    "dependents",
    "senior_citizen",
    "paperless_billing",
    "partner",
]

#: Numerical columns
NUM_COLS: list[str] = [
    "monthly_charges",
    "total_charges",
    "avg_price_increase",   # on-demand feature function output
    "tenure",
    "num_optional_services",
]

#: Low-cardinality categorical columns (one-hot encoded)
CAT_COLS: list[str] = [
    "contract",
    "device_protection",
    "internet_service",
    "multiple_lines",
    "online_backup",
    "online_security",
    "payment_method",
    "streaming_movies",
    "streaming_tv",
    "tech_support",
]


def build_churn_preprocessor(
    *,
    bool_cols: list[str] | None = None,
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
    scale_numerics: bool = True,
) -> ColumnTransformer:
    """Build the churn sklearn preprocessing pipeline.

    Callers may override any column list to accommodate schema changes or
    ablation experiments.

    Parameters
    ----------
    bool_cols:
        Override for boolean column list.  Defaults to ``BOOL_COLS``.
    num_cols:
        Override for numerical column list.  Defaults to ``NUM_COLS``.
    cat_cols:
        Override for categorical column list.  Defaults to ``CAT_COLS``.
    scale_numerics:
        Whether to apply ``StandardScaler`` to numeric columns.

    Returns
    -------
    sklearn.compose.ColumnTransformer (unfitted).
    """
    return build_column_transformer(
        bool_cols=bool_cols if bool_cols is not None else BOOL_COLS,
        num_cols=num_cols if num_cols is not None else NUM_COLS,
        cat_cols=cat_cols if cat_cols is not None else CAT_COLS,
        num_kwargs={"scale": scale_numerics},
    )
