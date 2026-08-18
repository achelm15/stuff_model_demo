# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · The Current Workflow: Models Without a System of Record
# MAGIC
# MAGIC This notebook intentionally models the fragile workflow many teams start with:
# MAGIC
# MAGIC 1. train a model;
# MAGIC 2. print a metric;
# MAGIC 3. change features or algorithms in place;
# MAGIC 4. overwrite the Python variables; and
# MAGIC 5. later try to remember which result was actually best.
# MAGIC
# MAGIC **This is an anti-pattern demonstration.** Nothing here uses MLflow, persists a
# MAGIC model, records the environment, or creates a reproducible handoff.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------


SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_pitches"

# COMMAND ----------

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

BASIC_FEATURES = [
    "release_speed",
    "spin_rate",
    "pfx_x_hnorm",
    "pfx_z",
    "extension",
]
ENGINEERED_FEATURES = [
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
TARGET = "pitch_rv"

assert spark.catalog.tableExists(SILVER_TABLE), f"{SILVER_TABLE} does not exist. Run notebook 00 first."

pdf = (
    spark.table(SILVER_TABLE)
    .where("pitch_type IN ('FF', 'SI')")
    .selectExpr("season", *ENGINEERED_FEATURES, f"CAST({TARGET} AS DOUBLE) AS {TARGET}")
    .toPandas()
    .dropna(subset=ENGINEERED_FEATURES + [TARGET])
    .reset_index(drop=True)
)
assert pd.api.types.is_numeric_dtype(pdf[TARGET]), f"{TARGET} must be numeric, got {pdf[TARGET].dtype}."
train_pdf = pdf[pdf["season"] == 2024].copy()
test_pdf = pdf[pdf["season"] == 2025].copy()
assert not train_pdf.empty and not test_pdf.empty, "This workshop expects both 2024 and 2025 data."

print(f"train rows={len(train_pdf):,}; test rows={len(test_pdf):,}")
display(pdf.groupby("season")[TARGET].agg(["count", "mean", "std"]).reset_index())

# COMMAND ----------
# MAGIC %md
# MAGIC ## Attempt 1: a reasonable LightGBM baseline
# MAGIC
# MAGIC We print two metrics. If this output is cleared or the notebook is rerun, the record
# MAGIC is gone. The model exists only in this Python session.

# COMMAND ----------

features = BASIC_FEATURES
params = {
    "n_estimators": 250,
    "learning_rate": 0.05,
    "num_leaves": 40,
}
model = LGBMRegressor(
    objective="regression",
    random_state=42,
    verbosity=-1,
    **params,
)
model.fit(train_pdf[features], train_pdf[TARGET])
prediction = model.predict(test_pdf[features])

print("LightGBM basic features")
print("RMSE:", round(root_mean_squared_error(test_pdf[TARGET], prediction), 6))
print("MAE: ", round(mean_absolute_error(test_pdf[TARGET], prediction), 6))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Attempt 2: add features and try XGBoost
# MAGIC
# MAGIC Notice what happens: `features`, `params`, `model`, and `prediction` are overwritten.
# MAGIC The first fitted model is no longer accessible unless we rerun the earlier cell.

# COMMAND ----------

features = ENGINEERED_FEATURES
params = {
    "n_estimators": 350,
    "learning_rate": 0.04,
    "max_depth": 6,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
}
model = XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    **params,
)
model.fit(train_pdf[features], train_pdf[TARGET])
prediction = model.predict(test_pdf[features])

print("XGBoost engineered features")
print("RMSE:", round(root_mean_squared_error(test_pdf[TARGET], prediction), 6))
print("MAE: ", round(mean_absolute_error(test_pdf[TARGET], prediction), 6))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Attempt 3: hand-tune a few more models
# MAGIC
# MAGIC The loop prints results, but it does not retain a comparison table, model artifacts,
# MAGIC input examples, signatures, plots, package versions, or a durable link to the data.

# COMMAND ----------

for leaves, learning_rate in [(24, 0.08), (48, 0.045), (96, 0.025), (128, 0.018)]:
    params = {
        "n_estimators": 400,
        "learning_rate": learning_rate,
        "num_leaves": leaves,
    }
    model = LGBMRegressor(
        objective="regression",
        random_state=42,
        verbosity=-1,
        **params,
    )
    model.fit(train_pdf[ENGINEERED_FEATURES], train_pdf[TARGET])
    prediction = model.predict(test_pdf[ENGINEERED_FEATURES])
    rmse = root_mean_squared_error(test_pdf[TARGET], prediction)
    print(f"leaves={leaves:>3} learning_rate={learning_rate:.3f} RMSE={rmse:.6f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## A week later: “Which one did we choose?”
# MAGIC
# MAGIC Python can only tell us what survived in the current session. The variable below is
# MAGIC the **last** loop iteration, not necessarily the best experiment. There is no run ID,
# MAGIC immutable artifact, registered version, approver, or deployment target.

# COMMAND ----------

print("Model still in memory:", type(model).__name__)
print("Parameters still in memory:", params)
print("Features we think it used:", ENGINEERED_FEATURES)
print("Last RMSE still in memory:", round(rmse, 6))

# COMMAND ----------
# MAGIC %md
# MAGIC ## What is missing?
# MAGIC
# MAGIC - Which attempt was best, and against which split?
# MAGIC - What exact code, parameters, features, data version, and packages produced it?
# MAGIC - Where is the fitted model artifact?
# MAGIC - Can another analyst compare results without rerunning every cell?
# MAGIC - Which model was approved, registered, deployed, and used for predictions?
# MAGIC - Can a scheduled job reproduce it under a service principal?
# MAGIC
# MAGIC Notebook 02 repeats this development loop with MLflow. The algorithms are not the
# MAGIC lesson—the system of record is.
