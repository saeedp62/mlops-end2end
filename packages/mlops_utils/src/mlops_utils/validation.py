"""
mlops_utils.validation
~~~~~~~~~~~~~~~~~~~~~~
Generic model-validation framework for champion-challenger workflows.

The ``ModelValidator`` class orchestrates a sequence of pluggable checks.
Each check is a callable that returns ``(passed: bool, message: str)``.
After running all checks the validator produces a summary dict of tags that
can be written to the MLflow model registry.

Usage
-----
::

    from mlops_utils.validation import ModelValidator, CheckResult
    from mlflow import MlflowClient

    client = MlflowClient()
    validator = ModelValidator(client, model_name="catalog.schema.model", version="3")

    validator.add_check(validator.check_description())
    validator.add_check(validator.check_artifacts())
    validator.add_check(
        validator.check_metric_vs_threshold(metric_key="test_f1_score", min_value=0.70)
    )
    validator.add_check(
        validator.check_champion_challenger(
            metric_key="test_f1_score", alias="Champion"
        )
    )

    passed, tags = validator.run()
    if passed:
        client.set_registered_model_alias(model_name, "Champion", version)
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mlops_utils.logger import get_logger

if TYPE_CHECKING:
    from mlflow import MlflowClient

logger = get_logger(__name__)

# A check is a zero-argument callable that returns a (passed, message) tuple.
CheckFn = Callable[[], tuple[bool, str]]


@dataclass
class CheckResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    message: str


@dataclass
class ModelValidator:
    """Orchestrates a configurable set of model validation checks.

    Parameters
    ----------
    client:
        ``MlflowClient`` instance.
    model_name:
        Fully-qualified UC model name (``catalog.schema.model``).
    version:
        Model version to validate (string or int).
    """

    client: MlflowClient
    model_name: str
    version: str | int
    _checks: list[tuple[str, CheckFn]] = field(default_factory=list, init=False)

    def add_check(self, name: str, fn: CheckFn) -> ModelValidator:
        """Register a check under *name*.

        Parameters
        ----------
        name:
            Short identifier logged to model tags (e.g. ``"has_description"``).
        fn:
            Zero-argument callable returning ``(bool, str)``.

        Returns
        -------
        self  (for chaining)
        """
        self._checks.append((name, fn))
        return self

    def run(self, *, write_tags: bool = True) -> tuple[bool, dict[str, str]]:
        """Execute all registered checks and optionally write tags.

        Parameters
        ----------
        write_tags:
            If ``True``, tag results are written to the model version in MLflow.

        Returns
        -------
        (all_passed, tags_dict)
            ``all_passed`` is ``True`` only if every check passed.
            ``tags_dict`` maps each check name to ``"True"`` / ``"False"``.
        """
        results: list[CheckResult] = []
        for name, fn in self._checks:
            try:
                passed, message = fn()
            except Exception as exc:  # noqa: BLE001
                passed, message = False, f"Exception: {exc}"
                logger.exception("Check '%s' raised an exception.", name)

            results.append(CheckResult(name=name, passed=passed, message=message))
            logger.info("[%s] %s — %s", "PASS" if passed else "FAIL", name, message)

        tags = {r.name: str(r.passed) for r in results}
        overall = all(r.passed for r in results)

        if write_tags:
            from mlops_utils.mlflow_utils import set_model_version_tags

            set_model_version_tags(self.client, self.model_name, self.version, tags)
            approval_tag = "Approved" if overall else "Failed"
            set_model_version_tags(
                self.client, self.model_name, self.version,
                {"Approval_Check": approval_tag}
            )

        return overall, tags

    # ------------------------------------------------------------------
    # Built-in check factories (return (name, fn) ready for add_check)
    # ------------------------------------------------------------------

    def check_description(self, *, min_chars: int = 20) -> tuple[str, CheckFn]:
        """Check that the model version has a sufficiently long description."""
        model_name, version, client = self.model_name, self.version, self.client

        def _fn() -> tuple[bool, str]:
            details = client.get_model_version(model_name, str(version))
            desc = details.description or ""
            passed = len(desc) >= min_chars
            msg = (
                f"Description has {len(desc)} chars (min={min_chars})."
                if passed
                else f"Description too short or missing ({len(desc)} chars, min={min_chars})."
            )
            return passed, msg

        return "has_description", _fn

    def check_artifacts(
        self,
        *,
        local_dir: str = "/tmp/model_artifacts_validation",
    ) -> tuple[str, CheckFn]:
        """Check that the training run has logged at least one artifact."""
        model_name, version, client = self.model_name, self.version, self.client

        def _fn() -> tuple[bool, str]:
            import mlflow

            details = client.get_model_version(model_name, str(version))
            run_info = client.get_run(run_id=details.run_id)
            os.makedirs(local_dir, exist_ok=True)
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_info.info.run_id, dst_path=local_dir
            )
            files = os.listdir(local_path)
            passed = bool(files)
            msg = (
                f"Found {len(files)} artifact(s): {files}"
                if passed
                else "No artifacts found – please log metrics plots or data profiles."
            )
            return passed, msg

        return "has_artifacts", _fn

    def check_metric_vs_threshold(
        self,
        metric_key: str,
        *,
        min_value: float = 0.0,
    ) -> tuple[str, CheckFn]:
        """Check that *metric_key* meets a minimum threshold."""
        model_name, version, client = self.model_name, self.version, self.client

        def _fn() -> tuple[bool, str]:
            import mlflow

            details = client.get_model_version(model_name, str(version))
            metric_value = mlflow.get_run(details.run_id).data.metrics.get(metric_key)
            if metric_value is None:
                return False, f"Metric '{metric_key}' was not logged."
            passed = metric_value >= min_value
            msg = f"{metric_key}={metric_value:.4f} (threshold={min_value})."
            return passed, msg

        return f"metric_{metric_key}_passed", _fn

    def check_champion_challenger(
        self,
        metric_key: str,
        *,
        alias: str = "Champion",
        accept_if_no_champion: bool = True,
    ) -> tuple[str, CheckFn]:
        """Check challenger metric ≥ champion metric.

        Parameters
        ----------
        accept_if_no_champion:
            If ``True``, passing is automatic when no Champion exists yet.
        """
        model_name, version, client = self.model_name, self.version, self.client

        def _fn() -> tuple[bool, str]:
            import mlflow

            details = client.get_model_version(model_name, str(version))
            challenger_value = mlflow.get_run(details.run_id).data.metrics.get(metric_key)

            if challenger_value is None:
                return False, f"Challenger metric '{metric_key}' not found."

            try:
                champion_mv = client.get_model_version_by_alias(model_name, alias)
                champion_value = mlflow.get_run(champion_mv.run_id).data.metrics.get(metric_key)
            except Exception:  # noqa: BLE001
                champion_value = None

            if champion_value is None:
                if accept_if_no_champion:
                    return True, f"No '{alias}' model found – accepting challenger automatically."
                return False, f"No '{alias}' model found and accept_if_no_champion=False."

            passed = challenger_value >= champion_value
            msg = (
                f"Challenger {metric_key}={challenger_value:.4f} vs "
                f"Champion {metric_key}={champion_value:.4f}."
            )
            return passed, msg

        return "champion_challenger_passed", _fn

    def check_inference_runs(
        self,
        inference_df: Any,
        *,
        fe: Any | None = None,
        label_col: str,
        env_manager: str = "virtualenv",
    ) -> tuple[str, CheckFn]:
        """Check that the model can produce predictions on *inference_df*.

        Parameters
        ----------
        inference_df:
            Spark DataFrame with at minimum the lookup keys + label column.
        fe:
            FeatureEngineeringClient instance. Created automatically if None.
        label_col:
            Name of the label column (used for result_type lookup).
        env_manager:
            Passed to ``fe.score_batch``.
        """
        model_name, version, client = self.model_name, self.version, self.client

        def _fn() -> tuple[bool, str]:
            if fe is None:
                from databricks.feature_engineering import FeatureEngineeringClient
                fe_client = FeatureEngineeringClient()
            else:
                fe_client = fe

            model_uri = f"models:/{model_name}/{version}"
            try:
                result_type = inference_df.schema[label_col].dataType
                preds = fe_client.score_batch(
                    df=inference_df,
                    model_uri=model_uri,
                    result_type=result_type,
                    env_manager=env_manager,
                )
                count = preds.count()
                return True, f"Scored {count} rows successfully."
            except Exception as exc:  # noqa: BLE001
                return False, f"Inference failed: {exc}"

        return "predicts", _fn
