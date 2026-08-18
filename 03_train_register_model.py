# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "environment.yaml"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 03 · Train and Register the Production Model
# MAGIC
# MAGIC This is the service-principal training entry point. It tunes the reviewed XGBoost
# MAGIC recipe, evaluates the winner on an untouched test split, and registers one model
# MAGIC version. Production monitoring starts later, from inference events written by notebook 04.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

dbutils.widgets.dropdown("environment", "dev", ["dev", "staging", "prod"])
dbutils.widgets.text("experiment_name", "")
dbutils.widgets.text("train_season", "2024")
dbutils.widgets.text("model_alias", "champion")
dbutils.widgets.dropdown("training_device", "auto", ["auto", "gpu", "cpu"])
dbutils.widgets.text("tuning_trials", "8")

ENVIRONMENT = dbutils.widgets.get("environment")
TRAIN_SEASON = int(dbutils.widgets.get("train_season"))
MODEL_ALIAS = dbutils.widgets.get("model_alias").strip()
TRAINING_DEVICE = dbutils.widgets.get("training_device")
TUNING_TRIALS = int(dbutils.widgets.get("tuning_trials"))
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name").strip() or (
    f"/Shared/{SCHEMA}_{ENVIRONMENT}"
)

SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_pitches"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.fastball_stuff_rv"

# COMMAND ----------

import json
import shutil

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import optuna
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT_NAME)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Serverless/shared compute blocks the py4j call MLflow uses to resolve some run-context
# tags; silence that benign WARNING so it doesn't spam the output (runs still log fine).
import logging

logging.getLogger("mlflow.tracking.context.registry").setLevel(logging.ERROR)

TARGET = "pitch_rv"
NUMERIC_FEATURES = [
    "release_speed",
    "spin_rate",
    "spin_direction",
    "pfx_x_hnorm",
    "pfx_z",
    "release_x_hnorm",
    "release_z",
    "extension",
    "vaa",
]
CATEGORICAL_FEATURES = ["p_throws", "stand"]
MODEL_INPUTS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

assert spark.catalog.tableExists(SILVER_TABLE), f"{SILVER_TABLE} does not exist. Run notebook 00."
assert TUNING_TRIALS > 0, "tuning_trials must be positive."
# "auto" uses a GPU when one is present and falls back to CPU otherwise; "gpu" requires
# one and errors if it is missing; "cpu" forces CPU.
if TRAINING_DEVICE == "gpu":
    assert shutil.which("nvidia-smi"), "training_device=gpu needs NVIDIA GPU compute; use auto or cpu."
    XGBOOST_DEVICE = "cuda"
elif TRAINING_DEVICE == "cpu":
    XGBOOST_DEVICE = "cpu"
else:
    XGBOOST_DEVICE = "cuda" if shutil.which("nvidia-smi") else "cpu"
    if XGBOOST_DEVICE == "cpu":
        print("No NVIDIA GPU detected; training on CPU (training_device=auto).")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Reserve an untouched test split
# MAGIC
# MAGIC The model is tuned on one portion of 2024. A separate 20% holdout is never used for
# MAGIC fitting and provides an honest final quality check before registration.

# COMMAND ----------

season_pdf = (
    spark.table(SILVER_TABLE)
    .where(f"season = {TRAIN_SEASON} AND pitch_type IN ('FF', 'SI')")
    .selectExpr("pitch_type", *MODEL_INPUTS, f"CAST({TARGET} AS DOUBLE) AS {TARGET}")
    .toPandas()
    .dropna(subset=MODEL_INPUTS + [TARGET])
    .reset_index(drop=True)
)
assert not season_pdf.empty, f"No usable rows were found for {TRAIN_SEASON}."

training_pdf, test_pdf = train_test_split(season_pdf, test_size=0.20, random_state=42)
train_core, tuning_validation_pdf = train_test_split(
    training_pdf, test_size=0.20, random_state=42
)


def make_model(params):
    preprocessing = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    estimator = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        device=XGBOOST_DEVICE,
        n_jobs=-1,
        random_state=42,
        **params,
    )
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def regression_metrics(actual, prediction, prefix):
    return {
        f"{prefix}_rmse": float(root_mean_squared_error(actual, prediction)),
        f"{prefix}_mae": float(mean_absolute_error(actual, prediction)),
        f"{prefix}_r2": float(r2_score(actual, prediction)),
    }


def suggest_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.16, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.70, 1.00),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.00),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
    }


print(
    f"experiment={EXPERIMENT_NAME}\n"
    f"training={len(training_pdf):,} tuning_holdout={len(tuning_validation_pdf):,} "
    f"test={len(test_pdf):,}\n"
    f"device={XGBOOST_DEVICE} trials={TUNING_TRIALS}"
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Tune as nested MLflow runs, then register one winner

# COMMAND ----------

with mlflow.start_run(run_name=f"canonical-{ENVIRONMENT}-xgboost") as parent_run:
    mlflow.set_tags(
        {
            "workflow": "canonical_training",
            "environment": ENVIRONMENT,
            "algorithm": "xgboost",
            "feature_set": "engineered",
        }
    )

    def objective(trial):
        params = suggest_params(trial)
        with mlflow.start_run(nested=True):
            candidate = make_model(params)
            candidate.fit(train_core[MODEL_INPUTS], train_core[TARGET])
            prediction = candidate.predict(tuning_validation_pdf[MODEL_INPUTS])
            metrics = regression_metrics(tuning_validation_pdf[TARGET], prediction, "validation")
            mlflow.log_params({"trial_number": trial.number, **params})
            mlflow.log_metrics(metrics)
            return metrics["validation_rmse"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=TUNING_TRIALS, gc_after_trial=True)

    final_model = make_model(study.best_params)
    final_model.fit(training_pdf[MODEL_INPUTS], training_pdf[TARGET])
    test_prediction = final_model.predict(test_pdf[MODEL_INPUTS])
    test_metrics = regression_metrics(test_pdf[TARGET], test_prediction, "test")

    mlflow.log_params({f"best_{name}": value for name, value in study.best_params.items()})
    mlflow.log_param("best_trial_number", study.best_trial.number)
    mlflow.log_metrics(test_metrics)
    mlflow.log_dict(
        {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "ordered_inputs": MODEL_INPUTS,
            "target": TARGET,
        },
        "feature_contract.json",
    )

    # Training may use CUDA, but downstream batch and serving inference use a portable CPU artifact.
    final_model.named_steps["model"].set_params(device="cpu")
    final_model.named_steps["model"].get_booster().set_param({"device": "cpu"})
    input_example = training_pdf[MODEL_INPUTS].head(10)
    model_info = mlflow.sklearn.log_model(
        sk_model=final_model,
        name="model",
        input_example=input_example,
        signature=infer_signature(input_example, final_model.predict(input_example)),
        registered_model_name=MODEL_NAME,
        # cloudpickle instead of MLflow 3's skops default so the registered model logs
        # (and later loads, e.g. in the app) without skops trusted-type errors.
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )

    reloaded = mlflow.pyfunc.load_model(model_info.model_uri)
    assert len(reloaded.predict(input_example.head(2))) == 2

VERSION = str(model_info.registered_model_version)
registry = MlflowClient(registry_uri="databricks-uc")
registry.set_model_version_tag(MODEL_NAME, VERSION, "environment", ENVIRONMENT)
registry.set_model_version_tag(MODEL_NAME, VERSION, "evaluation_season", str(TRAIN_SEASON))
if MODEL_ALIAS:
    registry.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, VERSION)

print(f"registered {MODEL_NAME} version {VERSION}")
dbutils.notebook.exit(
    json.dumps(
        {
            "model_name": MODEL_NAME,
            "model_version": VERSION,
            "model_alias": MODEL_ALIAS,
            "training_run_id": parent_run.info.run_id,
            "test_rows": len(test_pdf),
            **test_metrics,
        }
    )
)
