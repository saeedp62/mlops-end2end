"""
mlops_utils.config_loader
~~~~~~~~~~~~~~~~~~~~~~~~~
YAML / JSON configuration loader with environment-variable override support.

Usage
-----
::

    from mlops_utils.config_loader import load_config, merge_configs

    # Load a single YAML file
    cfg = load_config("configs/dev.yaml")

    # Merge base + env-specific layer (env-specific values win)
    cfg = merge_configs("configs/base.yaml", "configs/prod.yaml")

    # Access values
    catalog = cfg["catalog"]
    schemas = cfg["schemas"]

Environment overrides
~~~~~~~~~~~~~~~~~~~~~
Any top-level key in the config can be overridden at runtime by setting an
environment variable prefixed with ``MLOPS_``.  The prefix is stripped and
the remainder lowercased:

    MLOPS_CATALOG=prod_catalog  →  cfg["catalog"] = "prod_catalog"
"""

from __future__ import annotations

import json
from mlops_utils.logger import get_logger
import os
from pathlib import Path
from typing import Any

import yaml

logger = get_logger(__name__)

# Environment variable prefix that triggers overrides
_ENV_PREFIX = "MLOPS_"


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config file and apply environment-variable overrides.

    Parameters
    ----------
    path:
        Absolute or relative path to a ``.yaml``, ``.yml``, or ``.json`` file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary with environment-variable overrides
        applied on top.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file extension is not ``.yaml``, ``.yml``, or ``.json``.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")

    suffix = p.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with p.open("r", encoding="utf-8") as fh:
            cfg: dict[str, Any] = yaml.safe_load(fh) or {}
    elif suffix == ".json":
        with p.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    else:
        raise ValueError(f"Unsupported config format '{suffix}'. Use .yaml/.yml or .json")

    _apply_env_overrides(cfg)
    logger.debug("Loaded config from %s: %s", p, cfg)
    return cfg


def merge_configs(*paths: str | Path) -> dict[str, Any]:
    """Load and deep-merge multiple config files left-to-right (last wins).

    Parameters
    ----------
    *paths:
        One or more paths to YAML/JSON config files.  Later files override
        keys from earlier ones.  Environment-variable overrides are applied
        after all file merges.

    Returns
    -------
    dict[str, Any]
        Merged configuration dictionary.
    """
    merged: dict[str, Any] = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            logger.warning("Config file not found, skipping: %s", p)
            continue
        layer = load_config(p)
        merged = _deep_merge(merged, layer)
    return merged


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _apply_env_overrides(cfg: dict[str, Any]) -> None:
    """Mutate *cfg* in-place with values from ``MLOPS_*`` environment variables."""
    for key, value in os.environ.items():
        if key.startswith(_ENV_PREFIX):
            cfg_key = key[len(_ENV_PREFIX):].lower()
            logger.debug("Env override: %s -> cfg['%s'] = %r", key, cfg_key, value)
            cfg[cfg_key] = value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base* (override wins on conflicts)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
