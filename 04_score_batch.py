# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "environment.yaml"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 04 · Batch Inference
# MAGIC
# MAGIC Score a season of pitches with the registered model and save the predictions to one
# MAGIC table in Unity Catalog. That table is an append-only history of every prediction the
# MAGIC model has made: it is both the audit log and the input that monitoring (notebook 05)
# MAGIC reads. On top of it we create a SQL view (a saved query that stores no data of its own)
# MAGIC that returns just the latest prediction for each pitch.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

dbutils.widgets.text("model_alias", "champion")
dbutils.widgets.text("inference_season", "2025")
dbutils.widgets.text("batch_id", "")

MODEL_ALIAS = dbutils.widgets.get("model_alias").strip()
INFERENCE_SEASON = int(dbutils.widgets.get("inference_season"))
REQUESTED_BATCH_ID = dbutils.widgets.get("batch_id").strip()

SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_pitches"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.fastball_stuff_rv"
PREDICTION_EVENTS_TABLE = f"{CATALOG}.{SCHEMA}.gold_pitch_prediction_events"
CURRENT_PREDICTIONS_VIEW = f"{CATALOG}.{SCHEMA}.gold_current_pitch_predictions"

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

# A Lakeflow Job should pass {{job.run_id}} as batch_id. The stable fallback keeps an
# interactive workshop rerun idempotent for the same season and registered model version.
BATCH_ID = REQUESTED_BATCH_ID or f"workshop-{INFERENCE_SEASON}-model-{MODEL_VERSION}"

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
duplicate_key_exists = (
    source.groupBy(*PREDICTION_KEYS)
    .count()
    .where(F.col("count") > 1)
    .limit(1)
    .count()
    > 0
)
assert not duplicate_key_exists, "Prediction keys are not unique."

import pandas as pd
import mlflow.sklearn
from pyspark.sql.functions import pandas_udf

# mlflow.pyfunc.spark_udf parses the Databricks runtime version, which raises
# InvalidVersion on preview runtimes with non-numeric minors (e.g. "18.x-photon-scala2").
# Apply the model through a pandas UDF so scoring stays distributed. The closure captures
# only MODEL_URI (a string) and lazy-loads the model once per worker into a module-level
# cache. We avoid sparkContext.broadcast (SparkContext is not accessible on serverless) and
# avoid embedding the pickled model in the closure, which would ship the whole artifact with
# every task.
_MODEL_CACHE = {}


@pandas_udf("double")
def predict_udf(*feature_cols):
    import warnings

    # A booster pickled by an older xgboost patch warns harmlessly when a newer xgboost
    # unpickles it; predictions are unaffected, and retraining on the current env clears it
    # at the source. Filter on the worker, where the load actually happens.
    warnings.filterwarnings("ignore", message=".*serialized model.*", category=UserWarning)
    model = _MODEL_CACHE.get(MODEL_URI)
    if model is None:
        model = mlflow.sklearn.load_model(MODEL_URI)
        _MODEL_CACHE[MODEL_URI] = model
    frame = pd.concat(feature_cols, axis=1)
    frame.columns = MODEL_INPUTS
    return pd.Series(model.predict(frame), index=frame.index)

# Build the exact row we store for each scored pitch:
#   - predicted_run_value: the model's output.
#   - actual_run_value: the observed outcome, kept so notebook 05 can measure quality later.
#   - model_name/version/alias, batch_id, scored_at: provenance, so every prediction says
#     which model produced it, in which run, and when.
#   - inference_id: a deterministic fingerprint used to avoid duplicate rows on write (below).
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
    .withColumn("batch_id", F.lit(BATCH_ID))
    .withColumn("scored_at", F.current_timestamp())
    .withColumn(
        # sha2 hashes (pitch keys + model version + batch id) into one id. The same pitch
        # scored by the same model version in the same batch always gets the same id, so a
        # re-run inserts nothing new. A new model version or batch changes the id, so a
        # genuinely new prediction is still recorded.
        "inference_id",
        F.sha2(
            F.concat_ws(
                "||",
                *[F.col(key).cast("string") for key in PREDICTION_KEYS],
                F.lit(MODEL_VERSION),
                F.lit(BATCH_ID),
            ),
            256,
        ),
    )
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Write the predictions to the events table
# MAGIC
# MAGIC Each scored pitch becomes one row in `gold_pitch_prediction_events`, an append-only
# MAGIC Delta table (a versioned table in Unity Catalog). We only ever add rows, never update
# MAGIC or delete, so the table keeps the full history of every prediction the model has made.
# MAGIC
# MAGIC Two Databricks features make this safe to run more than once:
# MAGIC
# MAGIC - **`MERGE ... WHEN NOT MATCHED`** inserts a scored row only if its `inference_id` is
# MAGIC   not already in the table, so re-running the batch (or an automatic job retry) never
# MAGIC   creates duplicate rows.
# MAGIC - **Change Data Feed** makes Delta record each row-level insert, so notebook 05 can
# MAGIC   process only the new rows on each refresh instead of re-reading the whole table.

# COMMAND ----------

# First run: the table does not exist yet, so create it by writing the scored rows.
# Later runs: the table exists, so MERGE in only the rows whose inference_id is new.
if spark.catalog.tableExists(PREDICTION_EVENTS_TABLE):
    # Turn on Change Data Feed so notebook 05's monitor reads only new rows on each refresh.
    spark.sql(
        f"ALTER TABLE {PREDICTION_EVENTS_TABLE} "
        "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )
    # MERGE reads its new rows from SQL, so expose the scored DataFrame as a temporary view.
    scored.createOrReplaceTempView("scored_batch")
    # Insert a scored pitch only when its inference_id is not already present (no duplicates).
    spark.sql(
        f"""
        MERGE INTO {PREDICTION_EVENTS_TABLE} AS target
        USING scored_batch AS source
        ON target.inference_id = source.inference_id
        WHEN NOT MATCHED THEN INSERT *
        """
    )
    spark.catalog.dropTempView("scored_batch")
else:
    # Create the table from the first batch, then enable Change Data Feed for future refreshes.
    scored.write.format("delta").mode("append").saveAsTable(PREDICTION_EVENTS_TABLE)
    spark.sql(
        f"ALTER TABLE {PREDICTION_EVENTS_TABLE} "
        "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )

# The log can hold several predictions for the same pitch (for example after a newer model
# version re-scores it). This view is a saved query that stores no data; it returns just the
# most recent prediction per pitch (highest scored_at), which is what dashboards should read.
spark.sql(
    f"""
    CREATE OR REPLACE VIEW {CURRENT_PREDICTIONS_VIEW} AS
    SELECT *
    FROM {PREDICTION_EVENTS_TABLE}
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY game_pk, at_bat_index, pitch_number
        ORDER BY scored_at DESC, inference_id DESC
    ) = 1
    """
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Read the latest prediction per pitch
# MAGIC
# MAGIC Querying the view gives one row per pitch with no data copied. As a quick sanity check
# MAGIC we build a leaderboard of the pitchers whose fastballs have the lowest predicted run
# MAGIC value (the best "stuff"), among those with enough fastballs to be meaningful.

# COMMAND ----------

current_predictions = spark.table(CURRENT_PREDICTIONS_VIEW)
leaderboard = (
    current_predictions.where(F.col("season") == INFERENCE_SEASON)
    .groupBy("season", "pitcher_id", "pitcher_name")
    .agg(
        F.count("*").alias("fastballs"),
        F.avg("predicted_run_value").alias("stuff_rv"),
        F.expr("percentile_approx(predicted_run_value, 0.5)").alias("median_stuff_rv"),
    )
    .where("fastballs >= 250")
)

display(leaderboard.orderBy("stuff_rv").limit(20))
dbutils.notebook.exit(
    json.dumps(
        {
            "model_version": MODEL_VERSION,
            "model_alias": MODEL_ALIAS,
            "batch_id": BATCH_ID,
            "rows_scored": ROWS_TO_SCORE,
            "prediction_events_table": PREDICTION_EVENTS_TABLE,
            "current_predictions_view": CURRENT_PREDICTIONS_VIEW,
        }
    )
)
