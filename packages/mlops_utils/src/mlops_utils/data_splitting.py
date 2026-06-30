"""
data_splitting.py - Utilities for advanced data splitting (e.g. out-of-time validation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlops_utils.logger import get_logger

if TYPE_CHECKING:
    from pyspark.sql import DataFrame as SparkDataFrame

logger = get_logger(__name__)


def time_based_split_spark(
    df: SparkDataFrame,
    time_col: str,
    train_ratio: float = 0.8,
) -> SparkDataFrame:
    """Perform an out-of-time (OOT) split on a Spark DataFrame.

    Sorts the data by `time_col` and assigns the first `train_ratio` fraction
    to the "train" split and the remainder to the "test" split.

    Parameters
    ----------
    df:
        Input Spark DataFrame.
    time_col:
        The column containing timestamps to sort by.
    train_ratio:
        The fraction of rows (0.0 to 1.0) to assign to "train".

    Returns
    -------
    Spark DataFrame with an added `split` column containing "train" or "test".
    """
    import pyspark.sql.functions as F
    from pyspark.sql.window import Window

    logger.info("Performing time-based split on column '%s' (train_ratio=%.2f)", time_col, train_ratio)

    # Use percent_rank to determine the relative position of each row ordered by time_col.
    # Note: If time_col has many duplicates, this can place them in the same rank.
    # To ensure a strict split, we order by time_col and a tie-breaker if available, but for now we just use time_col.
    window_spec = Window.orderBy(F.col(time_col).asc())

    df_with_rank = df.withColumn("time_rank", F.percent_rank().over(window_spec))

    df_split = df_with_rank.withColumn(
        "split",
        F.when(F.col("time_rank") <= train_ratio, "train").otherwise("test"),
    ).drop("time_rank")

    return df_split
