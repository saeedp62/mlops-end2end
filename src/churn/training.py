"""
training.py - Domain-specific model training pipeline for the churn use-case.
"""

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import optuna
    from pyspark.sql import SparkSession

from lightgbm import LGBMClassifier
from mlops_utils.class_weights import compute_class_weights
from mlops_utils.feature_store import FeatureStoreManager
from mlops_utils.logger import get_logger
from mlops_utils.mlflow_utils import get_or_create_experiment
from mlops_utils.model_logging import log_and_evaluate_model
from mlops_utils.optimization import run_mlflow_optuna_study
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from churn.config import ChurnConfig
from churn.preprocessing import build_churn_preprocessor

logger = get_logger(__name__)


class ChurnOptunaObjective:
    """
    Optuna objective wrapper that defines the hyperparameter search spaces
    for LogisticRegression, RandomForest, and LightGBM for the Churn model.
    """

    def __init__(
        self,
        X_train_in: pd.DataFrame,
        Y_train_in: pd.Series,
        rng_seed: int = 2025,
        pos_label: str = "Yes",
    ) -> None:
        self.preprocessor = build_churn_preprocessor()
        self.rng_seed = rng_seed
        self.pos_label = pos_label

        # Split into training and validation sets for Optuna internal evaluation
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train_in, Y_train_in, test_size=0.1, random_state=rng_seed
        )
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val

        self.class_weights = compute_class_weights(self.Y_train, verbose=False)

    def __call__(self, trial: "optuna.Trial") -> float:
        classifier_name = trial.suggest_categorical(
            "classifier", ["LogisticRegression", "RandomForest", "LightGBM"]
        )

        if classifier_name == "LogisticRegression":
            lr_C = trial.suggest_float("C", 1e-2, 1, log=True)
            lr_tol = trial.suggest_float("tol", 1e-6, 1e-3, step=1e-6)
            classifier_obj = LogisticRegression(
                C=lr_C, tol=lr_tol, random_state=self.rng_seed, class_weight=self.class_weights
            )
        elif classifier_name == "RandomForest":
            n_estimators = trial.suggest_int("n_estimators", 10, 200, log=True)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
            classifier_obj = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                random_state=self.rng_seed,
                class_weight=self.class_weights,
            )
        elif classifier_name == "LightGBM":
            n_estimators = trial.suggest_int("n_estimators", 10, 200, log=True)
            max_depth = trial.suggest_int("max_depth", 3, 10)
            learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.9)
            max_bin = trial.suggest_int("max_bin", 2, 256)
            num_leaves = trial.suggest_int("num_leaves", 2, 256)
            classifier_obj = LGBMClassifier(
                force_row_wise=True,
                verbose=-1,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                max_bin=max_bin,
                num_leaves=num_leaves,
                random_state=self.rng_seed,
                class_weight=self.class_weights,
            )
        else:
            raise ValueError(f"Unknown classifier: {classifier_name}")

        model = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("classifier", classifier_obj)]
        )

        # We don't want nested MLflow tracking for every fit inside trials
        import mlflow
        mlflow.sklearn.autolog(disable=True)

        model.fit(self.X_train, self.Y_train)
        y_val_pred = model.predict(self.X_val)

        return f1_score(
            self.Y_val, y_val_pred, average="binary", pos_label=self.pos_label
        )


def _reproduce_best_model(best_params: dict, rng_seed: int, preprocessor: Any) -> Pipeline:
    classifier_type = best_params.pop("classifier")
    best_params["random_state"] = rng_seed

    if classifier_type == "LogisticRegression":
        clf = LogisticRegression(**best_params)
    elif classifier_type == "RandomForest":
        clf = RandomForestClassifier(**best_params)
    elif classifier_type == "LightGBM":
        clf = LGBMClassifier(force_row_wise=True, verbose=-1, **best_params)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def run_model_training_pipeline(
    spark: "SparkSession", 
    config: ChurnConfig,
    fe_client: Any | None = None
) -> None:
    """
    Main orchestrator for model training.
    """
    logger.info("Initializing Feature Engineering Client...")
    
    if fe_client:
        fsm = FeatureStoreManager(
            fe=fe_client,
            catalog=config.catalog,
            offline_schema=config.schemas.offline_features,
            online_schema=config.schemas.online_features,
        )
    else:
        fsm = FeatureStoreManager.from_config(config)
    
    fe = fsm.fe

    # 1. Define Feature Lookups
    logger.info("Defining Feature Lookups and Functions...")
    from churn.feature_engineering import build_churn_feature_lookups
    feature_lookups_n_functions = build_churn_feature_lookups(config, fsm)

    # 2. Extract Data using Feature Store Point-In-Time Joins
    logger.info("Pulling latest labels...")
    labels_df = spark.read.table(config.full_label_table)

    from churn.feature_engineering import get_latest_label_per_customer
    latest_customer_ids_df = get_latest_label_per_customer(labels_df, config.label_col)

    logger.info("Creating training set specification...")
    training_set_specs = fe.create_training_set(
        df=latest_customer_ids_df,
        label=config.label_col,
        feature_lookups=feature_lookups_n_functions,
        exclude_columns=["customer_id", config.timeseries_col],
        exclude_null_labels=True,
    )

    logger.info("Loading training and test sets as pandas dataframes...")
    training_df = training_set_specs.load_df()
    training_pdf = training_df.filter("split == 'train'").drop("split").toPandas()
    test_pdf = training_df.filter("split == 'test'").drop("split").toPandas()

    X_train = training_pdf.drop(config.label_col, axis=1)
    Y_train = training_pdf[config.label_col]
    X_test = test_pdf.drop(config.label_col, axis=1)
    Y_test = test_pdf[config.label_col]

    # 3. Setup Experiment
    logger.info(f"Setting experiment to {config.full_experiment_name}")
    experiment_id = get_or_create_experiment(
        config.full_experiment_name, tags={"dbdemos": "advanced"}
    )

    # 4. Run HPO
    logger.info("Starting Hyperparameter Optimization...")
    objective_fn = ChurnOptunaObjective(X_train, Y_train, config.rng_seed, config.pos_label)

    try:
        import optuna
        optuna_sampler = optuna.samplers.TPESampler(seed=config.rng_seed)
    except ImportError:
        optuna_sampler = None

    study = run_mlflow_optuna_study(
        objective_fn=objective_fn,
        experiment_id=experiment_id,
        run_name=config.hpo.run_name,
        n_trials=config.hpo.n_trials,
        n_jobs=config.hpo.n_jobs,
        sampler=optuna_sampler,
    )

    # 5. Extract Best Model and Retrain on full train set
    logger.info("Retraining best model on full training set...")
    best_params = study.best_params
    best_model_pipeline = _reproduce_best_model(
        best_params, config.rng_seed, objective_fn.preprocessor
    )

    best_model_pipeline.fit(X_train, Y_train)

    # 6. Evaluate and Log
    # To log, we must be under a specific run so we'll fetch the run ID of the best trial
    import mlflow
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{config.hpo.run_name}'",
        order_by=["start_time DESC"],
        max_results=1
    )

    run_id = runs[0].info.run_id if runs else None

    if run_id is None:
        # Create a new run if parent doesn't exist
        with mlflow.start_run(experiment_id=experiment_id) as new_run:
            run_id = new_run.info.run_id

    log_and_evaluate_model(
        model_pipeline=best_model_pipeline,
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        label_col=config.label_col,
        pos_label=config.pos_label,
        training_set_specs=training_set_specs,
        run_id=run_id,
        experiment_id=experiment_id,
        artifact_path="model",
        fe_client=fe,
    )

    logger.info("Model training pipeline completed successfully.")
