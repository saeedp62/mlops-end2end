"""
mlops_utils.data_validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Generic PySpark data quality validation framework.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from mlops_utils.logger import get_logger

logger = get_logger(__name__)

# A check is a callable that returns a (passed, message) tuple.
DataCheckFn = Callable[[DataFrame], tuple[bool, str]]

@dataclass
class DataCheckResult:
    """Result of a single data validation check."""
    name: str
    passed: bool
    message: str

@dataclass
class DataValidator:
    """Orchestrates a configurable set of PySpark data quality checks.
    
    Parameters
    ----------
    df:
        The PySpark DataFrame to validate.
    """
    df: DataFrame
    _checks: list[tuple[str, DataCheckFn]] = field(default_factory=list, init=False)

    def add_check(self, name: str, fn: DataCheckFn) -> DataValidator:
        """Register a check under *name*.
        
        Parameters
        ----------
        name:
            Short identifier for the check (e.g. "customer_id_no_nulls").
        fn:
            Callable returning (bool, str).
            
        Returns
        -------
        self (for chaining)
        """
        self._checks.append((name, fn))
        return self

    def run(self, raise_on_fail: bool = True) -> tuple[bool, list[DataCheckResult]]:
        """Execute all registered checks against the DataFrame.
        
        Parameters
        ----------
        raise_on_fail:
            If True, raises a ValueError if any check fails.
            
        Returns
        -------
        (all_passed, results)
        """
        results: list[DataCheckResult] = []
        for name, fn in self._checks:
            try:
                passed, message = fn(self.df)
            except Exception as exc:  # noqa: BLE001
                passed, message = False, f"Exception: {exc}"
                logger.exception("Data check '%s' raised an exception.", name)

            results.append(DataCheckResult(name=name, passed=passed, message=message))
            logger.info("[%s] %s — %s", "PASS" if passed else "FAIL", name, message)

        overall = all(r.passed for r in results)

        if not overall and raise_on_fail:
            failed_checks = [r for r in results if not r.passed]
            error_msg = "\n".join(f"- {r.name}: {r.message}" for r in failed_checks)
            raise ValueError(f"Data Quality Validation failed for {len(failed_checks)} checks:\n{error_msg}")

        return overall, results

    # ------------------------------------------------------------------
    # Built-in check factories (return (name, fn) ready for add_check)
    # ------------------------------------------------------------------

    @staticmethod
    def check_no_nulls(column: str) -> tuple[str, DataCheckFn]:
        """Check that the column has no null values."""
        def _fn(df: DataFrame) -> tuple[bool, str]:
            null_count = df.filter(F.col(column).isNull()).count()
            passed = null_count == 0
            msg = f"Found {null_count} nulls in '{column}'" if not passed else f"No nulls in '{column}'"
            return passed, msg
        return f"{column}_no_nulls", _fn

    @staticmethod
    def check_unique(column: str) -> tuple[str, DataCheckFn]:
        """Check that all values in the column are unique (no duplicates)."""
        def _fn(df: DataFrame) -> tuple[bool, str]:
            total_count = df.count()
            distinct_count = df.select(column).distinct().count()
            passed = total_count == distinct_count
            msg = f"Total: {total_count}, Distinct: {distinct_count}"
            return passed, msg
        return f"{column}_is_unique", _fn

    @staticmethod
    def check_allowed_values(column: str, allowed_values: list[Any]) -> tuple[str, DataCheckFn]:
        """Check that all values in the column are within the allowed list."""
        def _fn(df: DataFrame) -> tuple[bool, str]:
            invalid_count = df.filter(~F.col(column).isin(allowed_values) & F.col(column).isNotNull()).count()
            passed = invalid_count == 0
            msg = f"Found {invalid_count} invalid values for '{column}'" if not passed else f"All values valid in '{column}'"
            return passed, msg
        return f"{column}_allowed_values", _fn

    @staticmethod
    def check_range(column: str, min_val: float, max_val: float) -> tuple[str, DataCheckFn]:
        """Check that all values in the column are between min_val and max_val (inclusive)."""
        def _fn(df: DataFrame) -> tuple[bool, str]:
            out_of_bounds = df.filter((F.col(column) < min_val) | (F.col(column) > max_val)).count()
            passed = out_of_bounds == 0
            msg = f"Found {out_of_bounds} values out of bounds [{min_val}, {max_val}] in '{column}'" if not passed else f"All values in range for '{column}'"
            return passed, msg
        return f"{column}_range", _fn

    @staticmethod
    def check_custom_sql(name: str, expression: str) -> tuple[str, DataCheckFn]:
        """Check using a custom Spark SQL expression that evaluates to a boolean."""
        def _fn(df: DataFrame) -> tuple[bool, str]:
            failed_count = df.filter(~F.expr(expression)).count()
            passed = failed_count == 0
            msg = f"Found {failed_count} rows failing expression: {expression}" if not passed else f"All rows passed expression: {expression}"
            return passed, msg
        return name, _fn
