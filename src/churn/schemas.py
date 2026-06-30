"""
churn.schemas
~~~~~~~~~~~~~
Pandera schemas for validating the bronze customer table and the churn
feature table.  Schema validation runs at the start of each pipeline stage
to catch data quality issues early.

Usage::

    from churn.schemas import BronzeCustomerSchema, ChurnFeatureSchema

    # Validate a pandas DataFrame before processing
    BronzeCustomerSchema.validate(pandas_df)

    # Validate a Spark DataFrame (converts to pandas internally)
    BronzeCustomerSchema.validate(spark_df.toPandas())
"""

from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema

# ---------------------------------------------------------------------------
# Bronze / raw customer table schema
# ---------------------------------------------------------------------------

BronzeCustomerSchema = DataFrameSchema(
    columns={
        "customer_id": Column(
            str,
            nullable=False,
            checks=Check.str_length(min_value=1),
            description="Unique customer identifier.",
        ),
        "gender": Column(
            str,
            nullable=True,
            checks=Check.isin(["Male", "Female"]),
        ),
        "senior_citizen": Column(
            nullable=True,
            # Raw data has integer 0/1; after cleaning it becomes "Yes"/"No"
            checks=Check.isin([0, 1, "0", "1", "Yes", "No"]),
        ),
        "partner": Column(
            str,
            nullable=True,
            checks=Check.isin(["Yes", "No"]),
        ),
        "dependents": Column(
            str,
            nullable=True,
            checks=Check.isin(["Yes", "No"]),
        ),
        "tenure": Column(
            nullable=True,
            checks=[Check.greater_than_or_equal_to(0)],
        ),
        "phone_service": Column(
            str,
            nullable=True,
            checks=Check.isin(["Yes", "No"]),
        ),
        "multiple_lines": Column(
            str,
            nullable=True,
        ),
        "internet_service": Column(
            str,
            nullable=True,
            checks=Check.isin(["Fiber optic", "DSL", "No"]),
        ),
        "online_security": Column(str, nullable=True),
        "online_backup": Column(str, nullable=True),
        "device_protection": Column(str, nullable=True),
        "tech_support": Column(str, nullable=True),
        "streaming_tv": Column(str, nullable=True),
        "streaming_movies": Column(str, nullable=True),
        "contract": Column(
            str,
            nullable=True,
            checks=Check.isin(["Month-to-month", "One year", "Two year"]),
        ),
        "paperless_billing": Column(
            str,
            nullable=True,
            checks=Check.isin(["Yes", "No"]),
        ),
        "payment_method": Column(str, nullable=True),
        "monthly_charges": Column(
            nullable=True,
            checks=Check.greater_than_or_equal_to(0),
        ),
        "total_charges": Column(
            nullable=True,
        ),
        "churn": Column(
            str,
            nullable=True,
            checks=Check.isin(["Yes", "No"]),
            description="Ground-truth churn label.",
        ),
    },
    coerce=True,     # Attempt type coercion before validation
    strict=False,    # Allow extra columns (e.g. ingestion metadata)
    name="BronzeCustomerSchema",
)


# ---------------------------------------------------------------------------
# Silver / feature table schema
# ---------------------------------------------------------------------------

ChurnFeatureSchema = DataFrameSchema(
    columns={
        "customer_id": Column(str, nullable=False),
        "transaction_ts": Column(
            "datetime64[ns]",
            nullable=False,
            description="Timestamp of the feature snapshot.",
        ),
        "gender": Column(str, nullable=True),
        "senior_citizen": Column(
            str,
            nullable=True,
            checks=Check.isin(["Yes", "No"]),
        ),
        "partner": Column(str, nullable=True),
        "dependents": Column(str, nullable=True),
        "tenure": Column(
            float,
            nullable=True,
            checks=Check.greater_than_or_equal_to(0),
        ),
        "phone_service": Column(str, nullable=True),
        "multiple_lines": Column(str, nullable=True),
        "internet_service": Column(str, nullable=True),
        "online_security": Column(str, nullable=True),
        "online_backup": Column(str, nullable=True),
        "device_protection": Column(str, nullable=True),
        "tech_support": Column(str, nullable=True),
        "streaming_tv": Column(str, nullable=True),
        "streaming_movies": Column(str, nullable=True),
        "contract": Column(str, nullable=True),
        "paperless_billing": Column(str, nullable=True),
        "payment_method": Column(str, nullable=True),
        "monthly_charges": Column(
            float,
            nullable=True,
            checks=Check.greater_than_or_equal_to(0),
        ),
        "total_charges": Column(
            float,
            nullable=True,
            checks=Check.greater_than_or_equal_to(0),
        ),
        "num_optional_services": Column(
            float,
            nullable=True,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than_or_equal_to(6),
            ],
            description="Count of optional services enabled (0–6).",
        ),
    },
    coerce=True,
    strict=False,
    name="ChurnFeatureSchema",
)


# ---------------------------------------------------------------------------
# Label table schema
# ---------------------------------------------------------------------------

ChurnLabelSchema = DataFrameSchema(
    columns={
        "customer_id": Column(str, nullable=False),
        "transaction_ts": Column("datetime64[ns]", nullable=False),
        "churn": Column(
            str,
            nullable=True,
            checks=Check.isin(["Yes", "No"]),
        ),
        "split": Column(
            str,
            nullable=False,
            checks=Check.isin(["train", "test"]),
            description="Train/test assignment.",
        ),
    },
    coerce=True,
    strict=False,
    name="ChurnLabelSchema",
)
