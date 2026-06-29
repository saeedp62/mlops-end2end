# MLOps End-to-End Churn Prediction

[![CI/CD](https://github.com/your-org/mlops-end2end/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/mlops-end2end/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Databricks](https://img.shields.io/badge/Databricks-Runtime%2015.4%2B-orange.svg)](https://databricks.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-grade MLOps reference implementation built on Databricks with Unity Catalog, Delta Live Tables, and MLflow. The project demonstrates a **two-package architecture** that separates reusable infrastructure utilities from domain-specific ML logic, enabling the pattern to be applied to other ML projects across your organisation.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Components](#components)
  - [mlops\_utils – Shared Infrastructure](#mlops_utils--shared-infrastructure)
  - [churn – Domain Package](#churn--domain-package)
  - [DLT Pipelines](#dlt-pipelines)
  - [Databricks Asset Bundle](#databricks-asset-bundle)
  - [Configuration System](#configuration-system)
- [Data Source Strategy](#data-source-strategy)
- [Data Flow](#data-flow)
- [Version Control & Branching Strategy](#version-control--branching-strategy)
- [CI/CD Pipeline](#cicd-pipeline)
  - [GitHub Actions](#github-actions)
  - [Bitbucket Pipelines](#bitbucket-pipelines)
  - [Key differences](#key-differences-github-actions-vs-bitbucket-pipelines)
- [Deployment Guide](#deployment-guide)
- [Local Development](#local-development)
- [Testing](#testing)
- [Extending to Other ML Projects](#extending-to-other-ml-projects)
- [Required Secrets & Permissions](#required-secrets--permissions)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Git Repository                               │
│                                                                  │
│  packages/mlops_utils/   ◄── Separate wheel, reusable anywhere  │
│  src/churn/              ◄── Domain logic, depends on mlops_utils│
│  pipelines/dlt_*.py      ◄── Thin DLT wrappers (call churn pkg) │
│  notebooks/              ◄── Thin notebook wrappers             │
│  databricks.yml          ◄── Infrastructure as Code (DAB)       │
│  resources/*.yml         ◄── DLT pipelines, jobs, experiments   │
│  configs/{env}.yaml      ◄── Per-environment configuration      │
│  tests/                  ◄── Unit + integration tests            │
│  .github/workflows/      ◄── CI/CD via GitHub Actions           │
└─────────────────────────────────────────────────────────────────┘
                              │  bundle deploy
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Databricks Workspace                           │
│                                                                  │
│  DLT Pipeline: Bronze Ingestion                                  │
│    └► bronze_customers (raw)                                     │
│    └► bronze_customers_validated (DQ gates)                      │
│                │                                                 │
│  DLT Pipeline: Feature Engineering                               │
│    └► silver_churn_features  (UC Feature Table)                  │
│    └► silver_churn_labels    (train/test split)                  │
│                │                                                 │
│  Batch Job (Model Training – future)                             │
│    └► MLflow Experiment → UC Model Registry → Serving Endpoint   │
└─────────────────────────────────────────────────────────────────┘
```

The design follows three core principles:

1. **Dependency Injection** – `SparkSession`, `FeatureEngineeringClient`, and `MlflowClient` are always passed as arguments, never imported as globals. This keeps every function unit-testable without a live cluster.
2. **Config-driven behaviour** – switching data sources, catalogs, or online store backends requires only a YAML change, not a code change.
3. **DLT as an optional path** – the same pure Python functions are called by DLT pipeline files, orchestration notebooks, and integration tests. There is no DLT-specific business logic.

---

## Repository Structure

```
mlops-end2end/
│
├── packages/
│   └── mlops_utils/                  # Separate distributable wheel
│       ├── pyproject.toml
│       └── src/mlops_utils/
│           ├── catalog.py            # Unity Catalog setup helpers
│           ├── config_loader.py      # YAML/JSON loader + env-var overrides
│           ├── data_io.py            # Delta read/write/upsert
│           ├── feature_store.py      # Batch + online Feature Store wrappers
│           ├── mlflow_utils.py       # Experiment, registry, metrics helpers
│           ├── preprocessing.py      # Generic sklearn pipeline builders
│           ├── spark_utils.py        # get_or_create_spark, table_exists
│           └── validation.py         # Pluggable model validation framework
│
├── src/
│   └── churn/                        # Domain-specific package
│       ├── config.py                 # ChurnConfig dataclass + YAML loader
│       ├── data_source.py            # Three-strategy source dispatcher
│       ├── feature_engineering.py    # Pure Spark transform functions
│       ├── feature_store_pipeline.py # 7-stage orchestrator
│       ├── preprocessing.py          # Churn sklearn column definitions
│       └── schemas.py                # Pandera schema validation
│
├── pipelines/
│   ├── dlt_ingestion.py              # DLT: Bronze layer (source → validated bronze)
│   └── dlt_feature_engineering.py   # DLT: Silver layer (features + labels)
│
├── notebooks/
│   └── 02-mlops-advanced/
│       └── 01_feature_engineering_refactored.py  # Thin orchestration notebook
│
├── configs/
│   ├── dev.yaml                      # Dev: volume_csv source, online store off
│   └── prod.yaml                     # Prod: unity_catalog_table source, online store on
│
├── resources/                        # Databricks Asset Bundle resource definitions
│   ├── dlt_pipelines.yml             # DLT pipeline resource declarations
│   ├── jobs.yml                      # Batch job definitions
│   └── experiments.yml               # MLflow experiment declarations
│
├── tests/
│   ├── conftest.py                   # Shared Spark session, config fixtures
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_data_source.py       # All 3 source strategies + dispatcher
│   │   ├── test_dlt_ingestion.py     # DLT pipeline unit tests (dlt mocked)
│   │   ├── test_dlt_feature_engineering.py
│   │   ├── test_feature_engineering.py
│   │   ├── test_mlflow_utils.py
│   │   └── test_preprocessing.py
│   └── integration/
│       └── test_feature_store_pipeline.py  # Live cluster tests
│
├── scripts/
│   └── bootstrap_volumes.py          # One-time UC Volume setup (admin)
│
├── databricks.yml                    # Bundle root: variables, targets, includes
├── pyproject.toml                    # churn package metadata
└── .github/
    └── workflows/
        └── ci.yml                    # 6-job CI/CD pipeline
```

---

## Components

### `mlops_utils` – Shared Infrastructure

> **Installable as a standalone wheel** — `pip install mlops_utils-*.whl`

Provides generic, reusable utilities that any ML project on Databricks can use. Has no dependency on the `churn` package.

| Module | Key functions | Purpose |
|--------|--------------|---------|
| `spark_utils` | `get_or_create_spark`, `table_exists`, `wait_for_table` | Safe SparkSession management across local, CI, and Databricks environments |
| `config_loader` | `load_config`, `merge_configs` | YAML/JSON loading with deep merge and `MLOPS_*` env-var overrides |
| `catalog` | `setup_catalog_and_schema`, `grant_table_privileges`, `drop_and_recreate_schema` | Idempotent Unity Catalog setup |
| `data_io` | `read_delta`, `write_delta`, `upsert_delta`, `add_primary_key_constraint` | Delta table I/O with lineage comments |
| `feature_store` | `create_or_replace_feature_table`, `write_feature_table`, `score_batch_wrapper`, `publish_to_online_store`, `create_feature_serving_endpoint` | Databricks Feature Engineering Client wrappers for batch **and** online serving |
| `mlflow_utils` | `get_or_create_experiment`, `promote_model_alias`, `log_classification_metrics`, `get_champion_metric` | MLflow experiment and Unity Catalog model registry helpers |
| `preprocessing` | `build_boolean_pipeline`, `build_numerical_pipeline`, `build_categorical_ohe_pipeline`, `build_column_transformer` | Composable sklearn preprocessing pipeline builders |
| `validation` | `ModelValidator`, `check_metric_vs_threshold`, `check_champion_challenger`, `check_inference_runs` | Pluggable model validation with pass/fail reporting |

### `churn` – Domain Package

> **Installable as a separate wheel** — `pip install mlops_churn-*.whl`

Contains all churn-specific business logic. Depends on `mlops_utils` but not vice versa.

| Module | Key functions | Purpose |
|--------|--------------|---------|
| `config` | `ChurnConfig`, `DataSourceConfig`, `load_churn_config` | Typed configuration dataclasses with YAML loader |
| `data_source` | `get_source_dataframe`, `read_from_unity_catalog`, `read_from_volume_csv`, `read_from_http_csv` | Config-driven source dispatcher |
| `feature_engineering` | `compute_service_features`, `clean_churn_features`, `add_transaction_timestamp`, `split_label_from_features` | Pure Spark transforms (no side effects) |
| `feature_store_pipeline` | `run_feature_engineering_pipeline` | 7-stage orchestrator: source → bronze → features → labels → UC Feature Table |
| `preprocessing` | `build_churn_preprocessor` | Churn-specific sklearn column definitions |
| `schemas` | `BronzeCustomerSchema`, `FeatureSchema`, `LabelSchema` | Pandera schema validation for data quality |

### DLT Pipelines

Two Delta Live Tables pipelines implement the same logic as the notebook path, using the same `churn` package functions:

**`pipelines/dlt_ingestion.py`**

Reads from the configured source (cross-catalog UC table or Volume CSV) and materialises a validated bronze Delta Live Table with data quality expectations:

| Expectation | Action |
|---|---|
| `customer_id IS NOT NULL` | Drop row |
| `monthly_charges >= 0` | Warn (metric tracked) |
| `total_charges >= 0` | Warn |
| `churn IN ('Yes', 'No')` | Warn |
| `contract IN (...)` | Warn |

**`pipelines/dlt_feature_engineering.py`**

DLT dependency graph:

```
bronze_customers_validated
       │
       ├─► silver_churn_features   (UC Feature Table – label col excluded)
       │         ↑ also read by:
       └─► silver_churn_labels     (keys from features + label from bronze + split)
```

`silver_churn_labels` reads `customer_id` and `transaction_ts` from the already-computed `silver_churn_features` table to avoid re-running the full feature pipeline a second time.

### Databricks Asset Bundle

The bundle (`databricks.yml` + `resources/*.yml`) manages all Databricks resources as Infrastructure as Code:

```
databricks.yml
├── variables            # catalog, db, source_type, node_type, …
├── artifacts            # mlops_utils wheel + churn wheel (auto-built on deploy)
├── resources/
│   ├── dlt_pipelines.yml   → 2 DLT pipelines
│   ├── jobs.yml            → full pipeline job + feature refresh job
│   └── experiments.yml     → MLflow experiment (created automatically)
└── targets
    ├── dev      (mode: development – resources prefixed "[dev <username>]")
    ├── staging  (deployed on push to develop)
    └── prod     (deployed on push to main, requires reviewer approval)
```

`databricks bundle deploy --target <env>` will:
1. Build both Python wheels
2. Upload wheels to `${artifacts_volume}/wheels/` in a UC Volume
3. Sync all pipeline files to the workspace via the Files API
4. Create or update DLT pipelines, jobs, and experiments via REST API (idempotent)

### Configuration System

Configs are YAML files stored in `configs/` (version-controlled) and also uploaded to a UC Volume at deploy time so pipelines can read them at runtime.

```yaml
# Example: configs/prod.yaml
catalog: main
db: dbdemos_mlops

data_source:
  type: unity_catalog_table              # reads cross-LOB Delta table directly
  source_table: telco_catalog.customer360.base_customers
  normalize_columns: true

online_store:
  enabled: true
  backend: databricks
  endpoint_name: churn_online_features
```

Any value can be overridden with `MLOPS_*` environment variables at runtime:

```bash
export MLOPS_CATALOG=main
export MLOPS_DB=my_schema
```

---

## Data Source Strategy

Three source types are supported, controlled by `data_source.type` in the YAML config. **No code changes are needed** when switching between them.

| Type | Environment | How it reads | Config key |
|------|-------------|--------------|-----------|
| `unity_catalog_table` | **Production** | `spark.table("lob.schema.table")` — cross-catalog UC read using pipeline service principal identity. No data copy. | `source_table: "telco_catalog.customer360.base_customers"` |
| `volume_csv` | **Demo / Dev** | `spark.read.csv("/Volumes/…")` — reads a CSV uploaded to a UC Volume | `volume_path: "/Volumes/main/shared_data/telco/file.csv"` |
| `http_csv` | **Local unit tests** | `requests.get(url)` → `spark.createDataFrame(pdf)` — HTTP download with S3 fallback. Not for Databricks clusters. | `url: "https://…"` |

The dispatcher lives in `churn/data_source.py`:

```python
from churn.data_source import get_source_dataframe

df = get_source_dataframe(spark, config)   # routes to correct reader automatically
```

For production cross-catalog reads, grant access once per source table:

```sql
GRANT SELECT ON TABLE telco_catalog.customer360.base_customers
TO `pipeline-service-principal@company.com`;
```

---

## Data Flow

```
Source Data (LOB catalog / Volume CSV)
          │
          ▼ Stage 1: get_source_dataframe()
          │          (unity_catalog_table | volume_csv | http_csv)
          │
          ▼ Stage 2: ingest_bronze_table()
          │          Delta bronze table  main.db.advanced_churn_bronze_customers
          │          + Pandera schema validation
          │
          ▼ Stage 3: compute_service_features()
          │          + clean_churn_features()
          │          + add_transaction_timestamp()
          │
          ▼ Stage 4: split_label_from_features()
          │
     ┌────┴─────┐
     ▼           ▼
Stage 5:      Stage 5:
Label table   Feature table (no label col)
(train/test)
     │              │
     └──────┬───────┘
            ▼ Stage 6/7: Unity Catalog Feature Table
                         main.db.advanced_churn_feature_table
                         (+ optional online store publish)
```

---

## Version Control & Branching Strategy

```
main  ──────────────────────────────────────────────────────►  Production
  ▲                                                            (approval gate)
  │ PR + review
  │
develop  ───────────────────────────────────────────────────►  Staging
  ▲                                                            (auto-deploy)
  │ PR + review
  │
feature/xxx  ─── (unit tests pass locally) ──────────────────  Developer sandbox
```

### Branch rules

| Branch | Protection | Deploy target |
|--------|-----------|---------------|
| `main` | Require PR + 1 reviewer + CI green | `prod` (manual approval gate – GitHub Environment or Bitbucket Deployment) |
| `develop` | Require PR + CI green | `staging` (automatic on merge) |
| `feature/*` | None | `dev` (manual `bundle deploy`) |

### What lives in Git

Every artifact that defines the system's behaviour is version-controlled:

| Git-tracked | Not in Git |
|-------------|-----------|
| All Python source code | Wheel binaries (`dist/`, `*.egg-info/`) |
| Config YAMLs (`configs/dev.yaml`, `configs/prod.yaml`) | UC Volume contents (uploaded at deploy time) |
| Bundle definition (`databricks.yml`, `resources/*.yml`) | MLflow run artefacts |
| CI/CD workflow (`.github/workflows/ci.yml`) | Databricks workspace notebooks (managed by bundle) |
| Tests (`tests/`) | Databricks cluster logs |

### Tagging releases

Every production deploy creates an annotated Git tag automatically:

```
deploy-prod-20240115-031200
```

To roll back, redeploy from a previous tag:

```bash
git checkout deploy-prod-20240114-031200
databricks bundle deploy --target prod
```

---

## CI/CD Pipeline

Two CI/CD systems are supported. Both implement identical logic: lint → unit test → build/validate → staging deploy → integration test → prod deploy (with approval gate).

### GitHub Actions

**Config:** `.github/workflows/ci.yml`

```
PR / push
    │
    ├─[1] lint ──────────────► ruff check + ruff format + mypy
    │
    ├─[2] unit-tests ────────► pytest tests/unit/ (Python 3.10 + 3.11 matrix)
    │                          local PySpark, no Databricks auth needed
    │
    └─[3] build-and-validate ► python -m build (both wheels)
               │               databricks bundle validate --target dev
               │
    ┌──────────┘ (on push to develop)
    │
    ├─[4] deploy-staging ────► databricks bundle deploy --target staging
    │                          uploads wheels + syncs pipeline files
    │
    └─[5] integration-tests ► pytest tests/integration/ via Databricks Connect
                               (runs on staging cluster)

    ┌──────────────────────── (on push to main + manual approval via GitHub Environment)
    │
    └─[6] deploy-prod ───────► databricks bundle deploy --target prod
                               + git tag deploy-prod-<timestamp>
```

**Approval gate:** Create a GitHub Environment named `databricks-prod` with **Required reviewers** enabled. The workflow pauses at job 6 until a reviewer clicks Approve.

### Bitbucket Pipelines

**Config:** `bitbucket-pipelines.yml`

```
PR (any branch)
    │
    ├─ parallel: lint + unit-tests (3.11) + unit-tests (3.10)
    └─ build-and-validate

branch: develop
    │
    ├─ parallel: lint + unit-tests (3.11) + unit-tests (3.10)
    ├─ build-and-validate
    ├─ deploy-staging  (automatic, Deployment env: staging)
    └─ integration-tests

branch: main
    │
    ├─ parallel: lint + unit-tests (3.11) + unit-tests (3.10)
    ├─ build-and-validate
    └─ deploy-prod  (trigger: manual → Deployment env: production)

custom pipelines (on-demand):
    deploy-staging-manual
    deploy-prod-manual
    integration-tests-manual
```

**Approval gate:** Create a Bitbucket **Deployment environment** named `production`. Any step linked to that environment with `trigger: manual` pauses and requires a pipeline operator to click **Run step**. Add **Required reviewers** on the environment in Settings → Deployments to enforce peer approval.

### Key differences: GitHub Actions vs Bitbucket Pipelines

| Feature | GitHub Actions | Bitbucket Pipelines |
|---------|---------------|--------------------|
| Config file | `.github/workflows/ci.yml` | `bitbucket-pipelines.yml` |
| Secrets | GitHub Secrets (per environment) | Repository variables + Deployment variables |
| Python matrix builds | Native `strategy.matrix` | Separate steps with different `image:` |
| Artifact passing | `upload-artifact` / `download-artifact` | `artifacts:` key (within same pipeline run) |
| Manual approval | GitHub Environment with required reviewer | Deployment environment + `trigger: manual` |
| Release tagging | `git push origin <tag>` via GITHUB_TOKEN | `git push` via `BITBUCKET_REPO_TOKEN` |
| Databricks CLI install | `databricks/setup-cli` action | `curl` install script |
| Java (for PySpark) | `actions/setup-java` | `apt-get install openjdk-17-jdk-headless` |

### CI is safe to run without Databricks credentials

Jobs/steps 1–3 (lint, unit tests, bundle validate) require **zero Databricks secrets**. They run on every PR with a local PySpark session and a placeholder host for bundle validation.

Deploy steps only trigger on branch merges and require workspace credentials configured in the respective secrets store.

---

## Deployment Guide

### Prerequisites

- Databricks CLI ≥ 0.200 (supports Asset Bundles): `brew install databricks/tap/databricks`
- Unity Catalog enabled on your workspace
- Service principal with `CREATE TABLE`, `CREATE VOLUME` on the target catalog

### Step 1 – Bootstrap Volumes (once per environment, admin only)

```bash
# Creates UC schemas, Volumes, and uploads initial config YAMLs
DATABRICKS_HOST=https://your-workspace.azuredatabricks.net \
DATABRICKS_TOKEN=dapi... \
python scripts/bootstrap_volumes.py --target dev

# Preview what will happen without executing
python scripts/bootstrap_volumes.py --target prod --dry-run
```

### Step 2 – Deploy

```bash
# Deploy to dev (uses your own identity)
databricks bundle deploy --target dev

# Deploy to staging (requires staging secrets in env)
DATABRICKS_HOST=... DATABRICKS_TOKEN=... \
databricks bundle deploy --target staging

# Deploy to prod (requires prod secrets, creates git tag via CI)
# → triggered automatically on merge to main via GitHub Actions
```

### Step 3 – Run a pipeline manually

```bash
# Trigger the full pipeline job
databricks bundle run churn_full_pipeline_job --target dev

# Or trigger just the DLT ingestion pipeline
databricks bundle run --resource pipelines.churn_bronze_ingestion --target dev
```

### Step 4 – Validate the bundle without deploying

```bash
databricks bundle validate --target staging
databricks bundle summary --target staging    # shows current deployed state
```

---

## Local Development

### Setup

```bash
# Clone and install both packages in editable mode
git clone https://github.com/your-org/mlops-end2end.git
cd mlops-end2end

pip install -e "packages/mlops_utils[dev]"
pip install -e ".[dev]"

# Java is required for local PySpark
# macOS: brew install openjdk@17
# Ubuntu: sudo apt install openjdk-17-jdk
```

### Running the feature pipeline locally (Databricks Connect)

```bash
# Configure Databricks Connect to route Spark to a remote cluster
databricks configure
databricks-connect configure --cluster-id <your-cluster-id>

# Run against the dev schema
MLOPS_CONFIG_PATH=configs/dev.yaml python -c "
from churn.config import load_churn_config
from churn.feature_store_pipeline import run_feature_engineering_pipeline
from mlops_utils.spark_utils import get_or_create_spark

spark = get_or_create_spark()
cfg = load_churn_config('configs/dev.yaml')
run_feature_engineering_pipeline(spark, cfg)
"
```

---

## Testing

### Unit tests (no Databricks needed)

```bash
# Run all unit tests with coverage
pytest tests/unit/ -v -m "not integration" \
  --cov=src/churn \
  --cov=packages/mlops_utils/src/mlops_utils \
  --cov-report=term-missing

# Run a specific test class
pytest tests/unit/test_dlt_ingestion.py::TestBronzeCustomers -v
```

### Test coverage breakdown

| Test file | What it covers |
|-----------|---------------|
| `test_config.py` | `ChurnConfig`, `DataSourceConfig`, `load_churn_config`, env-var overrides |
| `test_data_source.py` | All 3 source readers, column normalisation (Spark + pandas), dispatcher routing |
| `test_feature_engineering.py` | All transform functions with 20+ assertions |
| `test_preprocessing.py` | sklearn pipeline builders, churn preprocessor column schema |
| `test_mlflow_utils.py` | Experiment creation, alias promotion, metric helpers (mocked client) |
| `test_dlt_ingestion.py` | DLT pipeline unit tests with `dlt` module mocked via `importlib` |
| `test_dlt_feature_engineering.py` | `silver_churn_labels` join logic, split determinism, decorator wiring |

### How DLT pipeline unit tests work

The `dlt` Python module only exists inside a Databricks runtime. The tests use `importlib.util.spec_from_file_location` to load the pipeline files *after* injecting a mock `dlt` module with transparent no-op decorators:

```python
sys.modules['dlt'] = MagicMock(
    table=lambda **_: (lambda fn: fn),
    expect=lambda *_: (lambda fn: fn),
    expect_or_drop=lambda *_: (lambda fn: fn),
    read=MagicMock(),
)
```

This lets you call `bronze_customers()` and `silver_churn_features()` directly and assert on their Spark DataFrame outputs — identical to any other unit test.

### Integration tests (requires Databricks cluster)

```bash
# Requires DATABRICKS_HOST + DATABRICKS_TOKEN + Databricks Connect configured
pytest tests/integration/ -v -m integration
```

Integration tests create an isolated `dbdemos_mlops_ci` schema, run the full pipeline end-to-end, assert on table existence and row counts, then drop the schema on teardown.

---

## Extending to Other ML Projects

The `mlops_utils` package is designed to be domain-agnostic. To create a new ML project:

### 1. Install the shared wheel

```bash
pip install mlops_utils-*.whl
```

### 2. Create a domain config

```python
from dataclasses import dataclass
from mlops_utils.config_loader import load_config

@dataclass
class MyProjectConfig:
    catalog: str = "main"
    db: str = "my_project"
    # … add your fields

def load_my_config(path):
    raw = load_config(path)
    return MyProjectConfig(**raw)
```

### 3. Use the shared utilities

```python
from mlops_utils.spark_utils import get_or_create_spark
from mlops_utils.data_io import write_delta, upsert_delta
from mlops_utils.feature_store import create_or_replace_feature_table
from mlops_utils.mlflow_utils import get_or_create_experiment, log_classification_metrics
from mlops_utils.validation import ModelValidator

spark = get_or_create_spark()
write_delta(my_df, "main.my_project.my_table")
```

### 4. Replicate the bundle structure

Copy `databricks.yml`, `resources/`, `.github/workflows/ci.yml`, and `bitbucket-pipelines.yml` — update variable defaults, resource names, and the `churn_wheel` artifact path to point to your new package.

---

## Required Secrets & Permissions

### GitHub Secrets

Set these under **Settings → Secrets → Actions** in your GitHub repository, scoped to the appropriate GitHub Environment:

| Secret | Environment | Description |
|--------|-------------|-------------|
| `DATABRICKS_HOST_STAGING` | `databricks-staging` | Staging workspace URL |
| `DATABRICKS_TOKEN_STAGING` | `databricks-staging` | Staging PAT or SP token |
| `DATABRICKS_HOST_PROD` | `databricks-prod` | Production workspace URL |
| `DATABRICKS_TOKEN_PROD` | `databricks-prod` | Production PAT or SP token |
| `PIPELINE_SP_CLIENT_ID_STAGING` | `databricks-staging` | Service principal for staging pipelines |
| `PIPELINE_SP_CLIENT_ID_PROD` | `databricks-prod` | Service principal for prod pipelines |
| `CI_CLUSTER_ID_STAGING` | `databricks-staging` | Cluster ID for integration tests |

> **Tip:** Create GitHub Environments (`databricks-staging`, `databricks-prod`) with **required reviewer** protection on `databricks-prod` to enforce a manual approval gate before every production deployment.

### Bitbucket Repository Variables

Set these under **Repository settings → Repository variables** (available to all pipelines) and **Settings → Deployments** (scoped to `staging` / `production` environments):

| Variable | Scope | Description |
|----------|-------|-------------|
| `DATABRICKS_HOST_STAGING` | Repository | Staging workspace URL |
| `DATABRICKS_TOKEN_STAGING` | Deployment: `staging` | Staging PAT or SP token |
| `DATABRICKS_HOST_PROD` | Repository | Production workspace URL |
| `DATABRICKS_TOKEN_PROD` | Deployment: `production` | Production PAT or SP token |
| `PIPELINE_SP_CLIENT_ID_STAGING` | Deployment: `staging` | Service principal client ID |
| `PIPELINE_SP_CLIENT_ID_PROD` | Deployment: `production` | Service principal client ID |
| `CI_CLUSTER_ID_STAGING` | Deployment: `staging` | Cluster ID for Databricks Connect integration tests |
| `BITBUCKET_REPO_TOKEN` | Repository | App password with **write** scope for pushing release git tags |

> **Tip:** Mark `*_TOKEN` and `*_CLIENT_ID` variables as **Secured** in Bitbucket so they are masked in logs and not accessible to fork pipelines.

> **Approval gate:** In **Settings → Deployments**, add **Required reviewers** to the `production` environment. Any pipeline step with `deployment: production` will pause and wait for an authorised reviewer to approve before the deploy step runs.

### Unity Catalog Permissions

Run once per source table as a catalog admin before the first production deploy:

```sql
-- Grant the pipeline service principal read access to the LOB source table
GRANT SELECT ON TABLE telco_catalog.customer360.base_customers
TO `pipeline-sp@your-company.com`;

-- Grant write access to the destination catalog
GRANT CREATE TABLE, CREATE VOLUME ON SCHEMA main.dbdemos_mlops
TO `pipeline-sp@your-company.com`;
```
