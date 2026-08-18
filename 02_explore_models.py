# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "environment.yaml"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 02 · Explore Models in a Personal MLflow Experiment
# MAGIC
# MAGIC This is the development notebook. It is intentionally owned by the person doing
# MAGIC the modeling work: compare features and algorithms, inspect MLflow runs, and tune
# MAGIC promising candidates. It logs model artifacts, but it **never registers a model**.
# MAGIC
# MAGIC The reviewed algorithm, feature list, and parameters are copied into notebook 03
# MAGIC through normal code review. Notebook 03 is the service-principal entry point.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

dbutils.widgets.text("n_trials", "16")
dbutils.widgets.text("experiment_name", "")
dbutils.widgets.dropdown("training_device", "gpu", ["gpu", "cpu"])

N_TRIALS = int(dbutils.widgets.get("n_trials"))
TRAINING_DEVICE = dbutils.widgets.get("training_device")
CURRENT_USER = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT_NAME = dbutils.widgets.get("experiment_name").strip() or (
    f"/Users/{CURRENT_USER}/{SCHEMA}/stuff-model-development"
)

SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_pitches"

# COMMAND ----------

import json
import shutil
import subprocess

import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import optuna
import pandas as pd
import shap
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

mlflow.set_experiment(EXPERIMENT_NAME)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def configure_training_devices(requested_device):
    if requested_device == "cpu":
        return "cpu", "cpu", None

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        raise RuntimeError(
            "training_device=gpu requires NVIDIA GPU-backed Databricks compute. "
            "Attach this notebook to GPU compute or set training_device=cpu."
        )
    probe = subprocess.run(
        [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0 or not probe.stdout.strip():
        raise RuntimeError(
            "training_device=gpu requires NVIDIA GPU-backed Databricks compute. "
            "Attach this notebook to GPU compute or set training_device=cpu."
        )
    gpu_name = probe.stdout.strip().splitlines()[0]
    # AI Runtime provides CUDA for prebuilt packages such as XGBoost, but does not expose
    # an OpenCL device or the nvcc compiler needed to build LightGBM's CUDA backend.
    return "cpu", "cuda", gpu_name


LIGHTGBM_DEVICE, XGBOOST_DEVICE, GPU_NAME = configure_training_devices(TRAINING_DEVICE)
print(
    f"training_device={TRAINING_DEVICE}; "
    f"lightgbm_device={LIGHTGBM_DEVICE}; xgboost_device={XGBOOST_DEVICE}; "
    f"gpu={GPU_NAME or 'none'}"
)

TARGET = "pitch_rv"
CATEGORICAL_FEATURES = ["p_throws", "stand"]
FEATURE_SETS = {
    "basic": [
        "release_speed",
        "spin_rate",
        "pfx_x_hnorm",
        "pfx_z",
        "extension",
    ],
    "engineered": [
        "release_speed",
        "spin_rate",
        "spin_direction",
        "pfx_x_hnorm",
        "pfx_z",
        "release_x_hnorm",
        "release_z",
        "extension",
        "vaa",
    ],
}
ALL_MODEL_INPUTS = sorted(set(sum(FEATURE_SETS.values(), []) + CATEGORICAL_FEATURES))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Inspect the governed training data

# COMMAND ----------

assert spark.catalog.tableExists(SILVER_TABLE), f"{SILVER_TABLE} does not exist. Run notebook 00 first."

display(spark.sql(f"""
  SELECT
    season,
    pitch_type,
    count(*) AS pitches,
    round(avg(pitch_rv), 5) AS avg_pitch_rv,
    round(avg(release_speed), 2) AS avg_release_speed,
    round(avg(spin_rate), 0) AS avg_spin_rate
  FROM {SILVER_TABLE}
  WHERE pitch_type IN ('FF', 'SI')
  GROUP BY season, pitch_type
  ORDER BY season, pitch_type
"""))

pdf = (
    spark.table(SILVER_TABLE)
    .where("pitch_type IN ('FF', 'SI')")
    .selectExpr("season", *ALL_MODEL_INPUTS, f"CAST({TARGET} AS DOUBLE) AS {TARGET}")
    .toPandas()
    .dropna(subset=ALL_MODEL_INPUTS + [TARGET])
    .reset_index(drop=True)
)
assert len(pdf) > 0, "No complete fastball rows are available for modeling."

train_pdf = pdf[pdf["season"] == 2024].copy()
test_pdf = pdf[pdf["season"] == 2025].copy()
split_strategy = "season_2024_train_2025_test"
if train_pdf.empty or test_pdf.empty:
    train_pdf, test_pdf = train_test_split(pdf, test_size=0.25, random_state=42)
    split_strategy = "random_75_25_fallback"

train_core, validation_pdf = train_test_split(train_pdf, test_size=0.20, random_state=42)
print(
    f"experiment={EXPERIMENT_NAME}\n"
    f"split={split_strategy} train={len(train_core):,} "
    f"validation={len(validation_pdf):,} test={len(test_pdf):,}"
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Track the first baseline with MLflow autologging
# MAGIC
# MAGIC Notebook 01 established the untracked-workflow problem. Here, one line of LightGBM
# MAGIC autologging captures the parameters, metrics, model artifact, input example, and
# MAGIC signature for the first reproducible baseline.

# COMMAND ----------

basic_numeric = FEATURE_SETS["basic"]
mlflow.lightgbm.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    silent=True,
)
with mlflow.start_run(run_name="autolog-lightgbm-baseline"):
    tracked_baseline = LGBMRegressor(
        objective="regression",
        device_type=LIGHTGBM_DEVICE,
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=40,
        random_state=42,
        verbosity=-1,
    )
    tracked_baseline.fit(train_core[basic_numeric], train_core[TARGET].astype(float))
    tracked_prediction = tracked_baseline.predict(validation_pdf[basic_numeric])
    tracked_rmse = float(root_mean_squared_error(validation_pdf[TARGET], tracked_prediction))
    mlflow.log_metric("validation_rmse", tracked_rmse)
    mlflow.set_tags({
        "workflow": "personal_model_development",
        "algorithm": "lightgbm",
        "feature_set": "basic_numeric",
        "training_device": LIGHTGBM_DEVICE,
        "compute_accelerator": TRAINING_DEVICE,
        "gpu_name": GPU_NAME or "none",
    })

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.scatter(validation_pdf[TARGET], tracked_prediction, alpha=0.20, s=8)
    axis.set_title("Autologged baseline: predicted vs actual")
    axis.set_xlabel("Actual run value")
    axis.set_ylabel("Predicted run value")
    mlflow.log_figure(figure, "predicted_vs_actual.png")
    plt.close(figure)

mlflow.lightgbm.autolog(disable=True)
print(f"Tracked baseline validation RMSE: {tracked_rmse:.5f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Compare algorithms and feature sets
# MAGIC
# MAGIC Each row below becomes an independent MLflow run. Select the four runs in the
# MAGIC experiment UI and use **Compare → Show diff only**.

# COMMAND ----------

LGBM_PARAMS = {
    "n_estimators": 350,
    "learning_rate": 0.045,
    "num_leaves": 48,
    "min_child_samples": 80,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_lambda": 0.10,
}
XGB_PARAMS = {
    "n_estimators": 350,
    "learning_rate": 0.045,
    "max_depth": 6,
    "min_child_weight": 6,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "reg_lambda": 1.0,
}


def make_pipeline(algorithm, numeric_features, params):
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if algorithm == "lightgbm":
        estimator = LGBMRegressor(
            objective="regression",
            device_type=LIGHTGBM_DEVICE,
            n_jobs=-1,
            random_state=42,
            verbosity=-1,
            **params,
        )
    elif algorithm == "xgboost":
        estimator = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            device=XGBOOST_DEVICE,
            n_jobs=-1,
            random_state=42,
            **params,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def regression_metrics(actual, prediction, prefix):
    return {
        f"{prefix}_rmse": float(root_mean_squared_error(actual, prediction)),
        f"{prefix}_mae": float(mean_absolute_error(actual, prediction)),
        f"{prefix}_r2": float(r2_score(actual, prediction)),
    }


def log_calibration_plot(actual, prediction, dataset_label):
    calibration = pd.DataFrame({
        "actual": pd.Series(actual).astype(float).to_numpy(),
        "predicted": pd.Series(prediction).astype(float).to_numpy(),
    })
    calibration["prediction_decile"] = pd.qcut(
        calibration["predicted"],
        q=min(10, calibration["predicted"].nunique()),
        labels=False,
        duplicates="drop",
    )
    binned = (
        calibration.dropna(subset=["prediction_decile"])
        .groupby("prediction_decile", as_index=False)
        .agg(
            mean_predicted=("predicted", "mean"),
            mean_actual=("actual", "mean"),
            pitches=("actual", "size"),
        )
    )
    assert not binned.empty, "Calibration plot requires at least one prediction bin."

    lower = min(binned["mean_predicted"].min(), binned["mean_actual"].min())
    upper = max(binned["mean_predicted"].max(), binned["mean_actual"].max())
    if lower == upper:
        lower, upper = lower - 1e-6, upper + 1e-6

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([lower, upper], [lower, upper], "--", color="gray", label="perfect calibration")
    axis.plot(
        binned["mean_predicted"],
        binned["mean_actual"],
        "o-",
        color="#33006F",
        label="prediction-decile mean",
    )
    axis.set(
        xlabel="Mean predicted run value",
        ylabel="Mean actual run value",
        title=f"XGBoost calibration on {dataset_label} data",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    mlflow.log_figure(figure, "calibration_deciles.png")
    plt.close(figure)


def log_shap_plots(fitted_pipeline, raw_features, max_rows=2_000):
    sample = raw_features.sample(min(max_rows, len(raw_features)), random_state=42)
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    estimator = fitted_pipeline.named_steps["model"]
    transformed = pd.DataFrame(
        preprocessor.transform(sample),
        columns=preprocessor.get_feature_names_out(),
        index=sample.index,
    )
    shap_values = shap.TreeExplainer(estimator).shap_values(transformed)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        transformed,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    figure = plt.gcf()
    figure.tight_layout()
    mlflow.log_figure(figure, "shap_feature_importance.png")
    plt.close(figure)

    plt.figure(figsize=(8, 6))
    shap.summary_plot(
        shap_values,
        transformed,
        max_display=15,
        show=False,
    )
    figure = plt.gcf()
    figure.tight_layout()
    mlflow.log_figure(figure, "shap_beeswarm.png")
    plt.close(figure)


comparison_rows = []
for algorithm, params in [("lightgbm", LGBM_PARAMS), ("xgboost", XGB_PARAMS)]:
    for feature_set, numeric_features in FEATURE_SETS.items():
        model_inputs = numeric_features + CATEGORICAL_FEATURES
        run_name = f"{algorithm}-{feature_set}"
        effective_device = LIGHTGBM_DEVICE if algorithm == "lightgbm" else XGBOOST_DEVICE
        with mlflow.start_run(run_name=run_name) as run:
            candidate = make_pipeline(algorithm, numeric_features, params)
            candidate.fit(train_core[model_inputs], train_core[TARGET].astype(float))
            validation_prediction = candidate.predict(validation_pdf[model_inputs])
            test_prediction = candidate.predict(test_pdf[model_inputs])
            metrics = {
                **regression_metrics(validation_pdf[TARGET], validation_prediction, "validation"),
                **regression_metrics(test_pdf[TARGET], test_prediction, "test"),
            }

            mlflow.set_tags({
                "workflow": "personal_model_development",
                "algorithm": algorithm,
                "feature_set": feature_set,
                "split_strategy": split_strategy,
                "training_device": effective_device,
                "compute_accelerator": TRAINING_DEVICE,
                "gpu_name": GPU_NAME or "none",
            })
            mlflow.log_params({
                "algorithm": algorithm,
                "feature_set": feature_set,
                "training_device": effective_device,
                **params,
            })
            mlflow.log_metrics(metrics)
            mlflow.log_dict(
                {"numeric": numeric_features, "categorical": CATEGORICAL_FEATURES, "target": TARGET},
                "feature_contract.json",
            )
            training_dataset = mlflow.data.from_pandas(
                train_core[model_inputs + [TARGET]],
                source=SILVER_TABLE,
                name="fastball_training_data",
                targets=TARGET,
            )
            mlflow.log_input(training_dataset, context="training")
            input_example = train_core[model_inputs].head(10)
            signature = infer_signature(input_example, candidate.predict(input_example))
            mlflow.sklearn.log_model(
                sk_model=candidate,
                name="model",
                input_example=input_example,
                signature=signature,
            )
            comparison_rows.append({
                "run_id": run.info.run_id,
                "algorithm": algorithm,
                "feature_set": feature_set,
                "training_device": effective_device,
                **metrics,
            })

comparison = pd.DataFrame(comparison_rows).sort_values("validation_rmse")
display(comparison)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Tune the reviewed XGBoost + engineered-features candidate
# MAGIC
# MAGIC Optuna trials are nested MLflow runs. They remain development evidence and are not
# MAGIC registered. This parent run selects and records the winning recipe. The next run retrains
# MAGIC that recipe, logs the final development model and diagnostics, but still does not register it.

# COMMAND ----------

REVIEWED_NUMERIC_FEATURES = FEATURE_SETS["engineered"]
REVIEWED_MODEL_INPUTS = REVIEWED_NUMERIC_FEATURES + CATEGORICAL_FEATURES


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.16, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.70, 1.00),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.70, 1.00),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
    }
    with mlflow.start_run(nested=True, run_name=f"trial-{trial.number:03d}"):
        candidate = make_pipeline("xgboost", REVIEWED_NUMERIC_FEATURES, params)
        candidate.fit(train_core[REVIEWED_MODEL_INPUTS], train_core[TARGET].astype(float))
        prediction = candidate.predict(validation_pdf[REVIEWED_MODEL_INPUTS])
        rmse = float(root_mean_squared_error(validation_pdf[TARGET], prediction))
        mlflow.log_params({"training_device": XGBOOST_DEVICE, **params})
        mlflow.log_metric("validation_rmse", rmse)
        return rmse


with mlflow.start_run(run_name="optuna-xgboost-engineered") as tuning_run:
    mlflow.set_tags({
        "workflow": "personal_model_development",
        "algorithm": "xgboost",
        "feature_set": "engineered",
        "run_type": "hpo_parent",
        "training_device": XGBOOST_DEVICE,
        "compute_accelerator": TRAINING_DEVICE,
        "gpu_name": GPU_NAME or "none",
    })
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)
    mlflow.log_metric("best_validation_rmse", float(study.best_value))
    mlflow.log_params({f"best_{key}": value for key, value in study.best_params.items()})
    mlflow.log_dict(
        {
            "algorithm": "xgboost",
            "numeric_features": REVIEWED_NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET,
            "suggested_params": study.best_params,
            "note": "Review and copy approved code into notebook 03; do not register this run.",
        },
        "selected_recipe.json",
    )

tuning_parent_run_id = tuning_run.info.run_id

# COMMAND ----------
# MAGIC %md
# MAGIC ## Retrain and log the best development candidate
# MAGIC
# MAGIC This is the final output of personal model development. It retrains the selected recipe
# MAGIC on all development-year rows, evaluates once on the held-out test data, and logs a model
# MAGIC artifact with the diagnostics used during review. It is deliberately **not registered**;
# MAGIC notebook 03 independently retrains reviewed code under the job identity and registers that
# MAGIC new artifact.

# COMMAND ----------

with mlflow.start_run(run_name="best-xgboost-engineered") as best_run:
    best_model = make_pipeline(
        "xgboost",
        REVIEWED_NUMERIC_FEATURES,
        study.best_params,
    )
    best_model.fit(
        train_pdf[REVIEWED_MODEL_INPUTS],
        train_pdf[TARGET].astype(float),
    )
    test_prediction = best_model.predict(test_pdf[REVIEWED_MODEL_INPUTS])
    test_metrics = regression_metrics(test_pdf[TARGET], test_prediction, "test")

    mlflow.set_tags({
        "workflow": "personal_model_development",
        "algorithm": "xgboost",
        "feature_set": "engineered",
        "run_type": "best_candidate",
        "tuning_parent_run_id": tuning_parent_run_id,
        "split_strategy": split_strategy,
        "training_device": XGBOOST_DEVICE,
        "compute_accelerator": TRAINING_DEVICE,
        "gpu_name": GPU_NAME or "none",
        "registration_status": "development_only",
    })
    mlflow.log_params({
        "algorithm": "xgboost",
        "feature_set": "engineered",
        "training_device": XGBOOST_DEVICE,
        **study.best_params,
    })
    mlflow.log_metric("selected_validation_rmse", float(study.best_value))
    mlflow.log_metrics(test_metrics)
    mlflow.log_dict(
        {
            "numeric": REVIEWED_NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "target": TARGET,
            "tuning_parent_run_id": tuning_parent_run_id,
        },
        "feature_contract.json",
    )

    final_training_dataset = mlflow.data.from_pandas(
        train_pdf[REVIEWED_MODEL_INPUTS + [TARGET]],
        source=SILVER_TABLE,
        name="fastball_final_development_training_data",
        targets=TARGET,
    )
    mlflow.log_input(final_training_dataset, context="training")

    input_example = train_pdf[REVIEWED_MODEL_INPUTS].head(10)
    signature = infer_signature(input_example, best_model.predict(input_example))
    log_calibration_plot(test_pdf[TARGET], test_prediction, "held-out test")
    log_shap_plots(best_model, test_pdf[REVIEWED_MODEL_INPUTS])
    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="model",
        input_example=input_example,
        signature=signature,
    )

print("Suggested XGBoost parameters:", json.dumps(study.best_params, indent=2))
print(f"Best validation RMSE: {study.best_value:.5f}")
print(f"Final development run: {best_run.info.run_id}")
print("Held-out test metrics:", json.dumps(test_metrics, indent=2))
print("No model was registered. Continue to notebook 03 only after code review.")
