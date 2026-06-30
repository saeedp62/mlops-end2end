"""
churn.schema_bootstrap
~~~~~~~~~~~~~~~~~~~~~~
Churn-specific schema bootstrap shim.

The actual schema creation logic lives in the **shared** utility:

    mlops_utils.catalog.ensure_mlops_schemas

This module is a thin adapter that bridges the churn ``ChurnConfig`` to that
generic function, so the churn pipeline never needs to import from
``mlops_utils.catalog`` directly.

Usage::

    from churn.config import load_churn_config
    from churn.schema_bootstrap import ensure_schemas

    cfg = load_churn_config("/Volumes/.../configs/dev.yaml")
    ensure_schemas(spark, cfg)

Or call the shared utility directly (for custom schema sets)::

    from mlops_utils.catalog import ensure_mlops_schemas

    ensure_mlops_schemas(spark, "lighthouse_bkk6_analytics", {
        "training_datasets": "Bronze + label tables.",
        "offline_features":  "Feature tables.",
        ...
    })
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlops_utils.logger import get_logger

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

from mlops_utils.catalog import ensure_mlops_schemas

from churn.config import ChurnConfig

logger = get_logger(__name__)


def ensure_schemas(spark: SparkSession, cfg: ChurnConfig) -> None:
    """Bootstrap all MLOps schemas required by the churn pipeline.

    Delegates to :func:`mlops_utils.catalog.ensure_mlops_schemas` — the
    shared, use-case-agnostic implementation.  Schema descriptions are
    sourced from :meth:`~churn.config.SchemaConfig.as_comments_dict` so
    the churn config remains the single source of truth.

    Parameters
    ----------
    spark:
        Active SparkSession with Unity Catalog enabled.
    cfg:
        Loaded :class:`~churn.config.ChurnConfig`.
    """
    logger.info(
        "Churn schema bootstrap: ensuring schemas in catalog '%s'...",
        cfg.catalog,
    )
    ensure_mlops_schemas(
        spark,
        catalog=cfg.catalog,
        schemas=cfg.schemas.as_comments_dict(),
    )
