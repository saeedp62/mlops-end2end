#!/usr/bin/env python3
"""
scripts/bootstrap_volumes.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One-time bootstrap script – creates the Unity Catalog Volumes and uploads
the initial config YAMLs so that ``databricks bundle deploy`` can reference
them immediately.

Run once per target environment by a catalog admin before the first deploy:

    # Dev
    DATABRICKS_HOST=https://... DATABRICKS_TOKEN=dapi... \
        python scripts/bootstrap_volumes.py --target dev

    # Staging / Prod
    python scripts/bootstrap_volumes.py --target staging
    python scripts/bootstrap_volumes.py --target prod

The volume is always created inside the ``training_datasets`` schema of the
``lighthouse_bkk6_analytics`` catalog, producing the path::

    /Volumes/lighthouse_bkk6_analytics/training_datasets/bundle

This matches the ``artifacts_volume`` variable in ``databricks.yml``.

Requires: ``databricks-sdk`` (pip install databricks-sdk)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Target → Volume mapping
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# The bundle volume always lives at:
#   /Volumes/<catalog>/<schema>/<volume>
# = /Volumes/lighthouse_bkk6_analytics/training_datasets/bundle
#
# This matches artifacts_volume in databricks.yml:
#   /Volumes/lighthouse_bkk6_analytics/training_datasets/bundle
# ---------------------------------------------------------------------------
TARGETS = {
    "dev": {
        "catalog": "lighthouse_bkk6_analytics",
        "schema": "training_datasets",   # bundle volume lives here
        "volume": "bundle",
        "config_src": "configs/dev.yaml",
        "config_dest": "dev.yaml",
    },
    "staging": {
        "catalog": "lighthouse_bkk6_analytics",
        "schema": "training_datasets",   # same schema; CI environment is isolated by workspace
        "volume": "bundle",
        "config_src": "configs/dev.yaml",   # staging uses dev config as base; CI overwrites
        "config_dest": "staging.yaml",
    },
    "prod": {
        "catalog": "lighthouse_bkk6_analytics",
        "schema": "training_datasets",
        "volume": "bundle",
        "config_src": "configs/prod.yaml",
        "config_dest": "prod.yaml",
    },
}

_REPO_ROOT = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap UC Volumes for a bundle target.")
    parser.add_argument("--target", required=True, choices=list(TARGETS), help="Bundle target.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing.")
    args = parser.parse_args()

    t = TARGETS[args.target]
    catalog, schema, volume_name = t["catalog"], t["schema"], t["volume"]
    volume_path = f"/Volumes/{catalog}/{schema}/{volume_name}"

    print(f"Bootstrap target: {args.target!r}")
    print(f"  Catalog  : {catalog}")
    print(f"  Schema   : {schema}")
    print(f"  Volume   : {volume_name}")
    print(f"  Volume path: {volume_path}")
    print()

    if args.dry_run:
        print("[DRY RUN] No changes made.")
        return

    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.catalog import VolumeType
    except ImportError:
        sys.exit(
            "databricks-sdk is required. Install with: pip install databricks-sdk"
        )

    w = WorkspaceClient()   # reads DATABRICKS_HOST + DATABRICKS_TOKEN from env

    # ── 1. Create schema if missing ─────────────────────────────────────────
    try:
        w.schemas.create(name=schema, catalog_name=catalog)
        print(f"  ✓ Created schema: {catalog}.{schema}")
    except Exception as exc:
        if "already exists" in str(exc).lower():
            print(f"  · Schema already exists: {catalog}.{schema}")
        else:
            raise

    # ── 2. Create volume if missing ─────────────────────────────────────────
    try:
        w.volumes.create(
            catalog_name=catalog,
            schema_name=schema,
            name=volume_name,
            volume_type=VolumeType.MANAGED,
        )
        print(f"  ✓ Created volume: {volume_path}")
    except Exception as exc:
        if "already exists" in str(exc).lower():
            print(f"  · Volume already exists: {volume_path}")
        else:
            raise

    # ── 3. Create sub-directories by uploading placeholder files ────────────
    for subdir in ["configs", "wheels"]:
        placeholder = f"{volume_path}/{subdir}/.gitkeep"
        try:
            w.files.upload(placeholder, b"")
            print(f"  ✓ Created directory: {volume_path}/{subdir}/")
        except Exception:
            print(f"  · Directory already exists: {volume_path}/{subdir}/")

    # ── 4. Upload config YAML ────────────────────────────────────────────────
    config_src = _REPO_ROOT / t["config_src"]
    config_dest = f"{volume_path}/configs/{t['config_dest']}"
    if not config_src.exists():
        print(f"  ✗ Config file not found: {config_src}")
        sys.exit(1)

    w.files.upload(config_dest, config_src.read_bytes(), overwrite=True)
    print(f"  ✓ Uploaded config: {config_src.name} → {config_dest}")

    print()
    print("Bootstrap complete.  Next steps:")
    print(f"  1. Run:  databricks bundle deploy --target {args.target}")
    print(f"  2. Upload wheels after building:")
    print(f"       databricks fs cp dist/*.whl {volume_path}/wheels/ --overwrite")


if __name__ == "__main__":
    main()
