# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "xgboost==3.1.1",
#   "lightgbm==4.6.0",
#   "optuna==3.6.1",
#   "shap==0.49.1",
#   "databricks-sdk>=0.68.0",  # 05 needs the dataquality API (missing in the v5 base sdk 0.67.0)
# ]
# ///
# MAGIC %md
# MAGIC # 04 · Batch Inference
# MAGIC
# MAGIC Score a season of pitches with the `@champion` model and write the predictions to one
# MAGIC Delta table in Unity Catalog. That table is what monitoring (notebook 05) reads and what
# MAGIC dashboards query. Re-running the notebook overwrites the table, so it always holds the
# MAGIC latest scoring.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

dbutils.widgets.text("model_alias", "champion")
dbutils.widgets.text("inference_season", "2025")

MODEL_ALIAS = dbutils.widgets.get("model_alias").strip()
INFERENCE_SEASON = int(dbutils.widgets.get("inference_season"))

SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_pitches"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.fastball_stuff_rv"
PREDICTIONS_TABLE = f"{CATALOG}.{SCHEMA}.gold_pitch_predictions"

# COMMAND ----------

import json

import mlflow
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F

mlflow.set_registry_uri("databricks-uc")
registry = MlflowClient(registry_uri="databricks-uc")
registered_version = registry.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
MODEL_VERSION = str(registered_version.version)
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

model_info = mlflow.models.get_model_info(MODEL_URI)
MODEL_INPUTS = model_info.signature.inputs.input_names()
assert MODEL_INPUTS, "The registered model must have a named input signature."

PREDICTION_KEYS = ["game_pk", "at_bat_index", "pitch_number"]
IDENTITY_COLUMNS = [
    *PREDICTION_KEYS,
    "season",
    "game_date",
    "pitcher_id",
    "pitcher_name",
    "pitch_type",
]

# COMMAND ----------
# MAGIC %md
# MAGIC ## Score the registered model
# MAGIC
# MAGIC We load the `@champion` model from Unity Catalog and apply it with a *pandas UDF*: a
# MAGIC Python function that Spark runs in parallel across the cluster, so each pitch is scored
# MAGIC where its data already lives and nothing is pulled onto the driver. Here we score a full
# MAGIC season; a scheduled production job would score only the new batch of pitches.

# COMMAND ----------

assert spark.catalog.tableExists(SILVER_TABLE), f"{SILVER_TABLE} does not exist. Run notebook 00."
source = (
    spark.table(SILVER_TABLE)
    .where((F.col("season") == INFERENCE_SEASON) & F.col("pitch_type").isin("FF", "SI"))
)
missing_inputs = sorted(set(MODEL_INPUTS) - set(source.columns))
assert not missing_inputs, f"Source data is missing model inputs: {missing_inputs}"

ROWS_TO_SCORE = source.count()
assert ROWS_TO_SCORE > 0, "No rows matched the requested inference season."

import pandas as pd
import mlflow.sklearn
from pyspark.sql.functions import pandas_udf

# Load the model once on the driver. Referencing it in the UDF makes Spark serialize the
# model into the closure and ship it to the workers a single time, instead of every worker
# fetching and rebuilding it from Unity Catalog. We can't use sparkContext.broadcast
# (SparkContext is not accessible on serverless), and mlflow.pyfunc.spark_udf can't be used
# either: it parses the runtime version, which raises InvalidVersion on preview runtimes with
# non-numeric minors (e.g. "18.x-photon-scala2").
model = mlflow.sklearn.load_model(MODEL_URI)


@pandas_udf("double")
def predict_udf(*feature_cols):
    frame = pd.concat(feature_cols, axis=1)
    frame.columns = MODEL_INPUTS
    return pd.Series(model.predict(frame), index=frame.index)

# Build the row we store for each scored pitch: the prediction, the observed outcome (kept
# so notebook 05 can measure quality against it), and provenance columns saying which model
# produced it and when.
scored = (
    source.select(*IDENTITY_COLUMNS, *MODEL_INPUTS, "pitch_rv")
    .withColumn(
        "predicted_run_value",
        predict_udf(*[F.col(feature) for feature in MODEL_INPUTS]),
    )
    .withColumn("actual_run_value", F.col("pitch_rv").cast("double"))
    .drop("pitch_rv")
    .withColumn("model_name", F.lit(MODEL_NAME))
    .withColumn("model_version", F.lit(MODEL_VERSION))
    .withColumn("model_alias", F.lit(MODEL_ALIAS))
    .withColumn("scored_at", F.current_timestamp())
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Write the predictions
# MAGIC
# MAGIC Save the scored rows to one Delta table, overwriting it each run so it always holds the
# MAGIC latest scoring. Change Data Feed is enabled because notebook 05's monitor uses it to read
# MAGIC new rows incrementally.

# COMMAND ----------

(
    scored.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(PREDICTIONS_TABLE)
)
spark.sql(
    f"ALTER TABLE {PREDICTIONS_TABLE} "
    "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Inspect the predictions
# MAGIC
# MAGIC As a quick sanity check, build a leaderboard of the pitchers whose fastballs have the
# MAGIC lowest predicted run value (the best "stuff"), among those with enough fastballs to be
# MAGIC meaningful.

# COMMAND ----------

predictions = spark.table(PREDICTIONS_TABLE)
leaderboard = (
    predictions.where(F.col("season") == INFERENCE_SEASON)
    .groupBy("season", "pitcher_id", "pitcher_name")
    .agg(
        F.count("*").alias("fastballs"),
        F.avg("predicted_run_value").alias("stuff_rv"),
        F.expr("percentile_approx(predicted_run_value, 0.5)").alias("median_stuff_rv"),
    )
    .where("fastballs >= 250")
)

display(leaderboard.orderBy("stuff_rv").limit(20))

# COMMAND ----------

# dbutils.notebook.exit stops the notebook the moment it runs, so keep it in its own cell;
# in the same cell as display() above it would cut off the leaderboard output.
dbutils.notebook.exit(
    json.dumps(
        {
            "model_version": MODEL_VERSION,
            "model_alias": MODEL_ALIAS,
            "rows_scored": ROWS_TO_SCORE,
            "predictions_table": PREDICTIONS_TABLE,
        }
    )
)
