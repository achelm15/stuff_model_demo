# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "environment.yaml"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 07 · Bootstrapping the Stuff Model with Genie Code
# MAGIC
# MAGIC This notebook was scaffolded by **Databricks Genie Code** (the in-workspace agent-mode
# MAGIC coding assistant) from a single natural-language prompt, using a shared **workspace skill**
# MAGIC that encodes our MLflow + Unity Catalog conventions. The point of this notebook is not the
# MAGIC model — notebook 03 remains the governed, service-principal training entry point — it is
# MAGIC how fast an agent gets you to a *correct first draft* that already follows house style.
# MAGIC
# MAGIC **The prompt given to Genie Code (@-mentioning the workspace skill):**
# MAGIC
# MAGIC > `@rockies-mlflow-conventions` Train an XGBoost "stuff" regressor that predicts
# MAGIC > `pitch_rv` for four-seam and sinker fastballs in
# MAGIC > `<catalog>.<schema>.silver_pitches` (the target you set on the `_config` widgets).
# MAGIC > Use the engineered feature
# MAGIC > set, tune with Optuna as nested MLflow runs, evaluate on an untouched holdout, log the
# MAGIC > model with a signature and input example, and register a **candidate** version to Unity
# MAGIC > Catalog without moving the `@champion` alias.
# MAGIC
# MAGIC The `@rockies-mlflow-conventions` skill lives at
# MAGIC `Workspace/.assistant/skills/rockies-mlflow-conventions/SKILL.md` and is available to
# MAGIC everyone in the workspace, so every teammate's agent produces the same shape of code.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

dbutils.widgets.text("train_season", "2024")
dbutils.widgets.text("n_trials", "8")
dbutils.widgets.dropdown("training_device", "gpu", ["gpu", "cpu"])

TRAIN_SEASON = int(dbutils.widgets.get("train_season"))
N_TRIALS = int(dbutils.widgets.get("n_trials"))
TRAINING_DEVICE = dbutils.widgets.get("training_device")
CURRENT_USER = spark.sql("SELECT current_user()").first()[0]

SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_pitches"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.fastball_stuff_rv"
EXPERIMENT_NAME = f"/Users/{CURRENT_USER}/rockies-mlflow-demo/genie-code-stuff-model"

# COMMAND ----------

import json
import shutil

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import optuna
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
if TRAINING_DEVICE == "gpu":
    assert shutil.which("nvidia-smi"), "Attach GPU compute or select training_device=cpu."
XGBOOST_DEVICE = "cuda" if TRAINING_DEVICE == "gpu" else "cpu"

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load the governed fastball rows and reserve a holdout
# MAGIC
# MAGIC The skill told the agent to read from `silver_pitches`, filter to fastballs, and hold out
# MAGIC 20% that is never used for fitting — the same honest-evaluation pattern as notebook 03.

# COMMAND ----------

season_pdf = (
    spark.table(SILVER_TABLE)
    .where(f"season = {TRAIN_SEASON} AND pitch_type IN ('FF', 'SI')")
    .selectExpr(*MODEL_INPUTS, f"CAST({TARGET} AS DOUBLE) AS {TARGET}")
    .toPandas()
    .dropna(subset=MODEL_INPUTS + [TARGET])
    .reset_index(drop=True)
)
assert not season_pdf.empty, f"No usable rows for season {TRAIN_SEASON}."

training_pdf, test_pdf = train_test_split(season_pdf, test_size=0.20, random_state=42)
train_core, validation_pdf = train_test_split(training_pdf, test_size=0.20, random_state=42)
print(f"train={len(train_core):,} validation={len(validation_pdf):,} test={len(test_pdf):,}")


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

# COMMAND ----------
# MAGIC %md
# MAGIC ## Tune with Optuna as nested MLflow runs, then register a candidate
# MAGIC
# MAGIC Genie Code produced the whole tracking scaffold — nested runs, signature, input example,
# MAGIC feature contract — because the workspace skill described exactly how this team logs models.
# MAGIC It deliberately does **not** touch the `@champion` alias: this is a prototype, not a
# MAGIC promotion.

# COMMAND ----------

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


with mlflow.start_run(run_name="genie-code-xgboost-stuff") as parent_run:
    mlflow.set_tags({
        "workflow": "genie_code_prototype",
        "algorithm": "xgboost",
        "feature_set": "engineered",
        "authored_by": "genie_code",
    })

    def objective(trial):
        params = suggest_params(trial)
        with mlflow.start_run(nested=True, run_name=f"trial-{trial.number:03d}"):
            candidate = make_model(params)
            candidate.fit(train_core[MODEL_INPUTS], train_core[TARGET])
            prediction = candidate.predict(validation_pdf[MODEL_INPUTS])
            metrics = regression_metrics(validation_pdf[TARGET], prediction, "validation")
            mlflow.log_params({"trial_number": trial.number, **params})
            mlflow.log_metrics(metrics)
            return metrics["validation_rmse"]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

    final_model = make_model(study.best_params)
    final_model.fit(training_pdf[MODEL_INPUTS], training_pdf[TARGET])
    test_prediction = final_model.predict(test_pdf[MODEL_INPUTS])
    test_metrics = regression_metrics(test_pdf[TARGET], test_prediction, "test")

    mlflow.log_params({f"best_{name}": value for name, value in study.best_params.items()})
    mlflow.log_metrics(test_metrics)
    mlflow.log_dict(
        {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES, "target": TARGET},
        "feature_contract.json",
    )

    # Portable CPU artifact for downstream batch and serving inference.
    final_model.named_steps["model"].set_params(device="cpu")
    final_model.named_steps["model"].get_booster().set_param({"device": "cpu"})
    input_example = training_pdf[MODEL_INPUTS].head(10)
    model_info = mlflow.sklearn.log_model(
        sk_model=final_model,
        name="model",
        input_example=input_example,
        signature=infer_signature(input_example, final_model.predict(input_example)),
        registered_model_name=MODEL_NAME,
    )

    reloaded = mlflow.pyfunc.load_model(model_info.model_uri)
    assert len(reloaded.predict(input_example.head(2))) == 2

VERSION = str(model_info.registered_model_version)
registry = MlflowClient(registry_uri="databricks-uc")
registry.set_model_version_tag(MODEL_NAME, VERSION, "workflow", "genie_code_prototype")
registry.set_model_version_tag(MODEL_NAME, VERSION, "promotion_status", "candidate")

print(f"registered {MODEL_NAME} version {VERSION} as a candidate (champion alias untouched)")
print("held-out test metrics:", json.dumps(test_metrics, indent=2))
print("Review, then move approved recipes into notebook 03 for governed promotion.")
