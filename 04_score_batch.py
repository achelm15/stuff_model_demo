# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "/Workspace/Users/andrew.helmreich@databricks.com/rockies-mlflow-demo/rockies-mlflow-demo.yaml"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 04 · Batch Inference
# MAGIC
# MAGIC Score with the registered model and write one production prediction-events table.
# MAGIC The table is both the durable inference log and the direct input to Data Quality
# MAGIC Monitoring. A zero-copy SQL view exposes only the latest prediction for each pitch.

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
# MAGIC ## Score the registered artifact
# MAGIC
# MAGIC The registered pyfunc is applied as a Spark UDF, so feature rows remain distributed and
# MAGIC are never collected to the driver. In routine production inference, the source would be
# MAGIC restricted to the new batch; a promotion workflow can deliberately select the full scoring
# MAGIC population.

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

predict_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=MODEL_URI,
    result_type="double",
    env_manager="local",
)

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
# MAGIC ## Publish one idempotent inference log
# MAGIC
# MAGIC `inference_id` makes a repaired job run a no-op while preserving a genuinely new
# MAGIC scoring event. Predictions are immutable events; notebook 05 monitors this table
# MAGIC directly, so there is no second monitoring copy.

# COMMAND ----------

if spark.catalog.tableExists(PREDICTION_EVENTS_TABLE):
    spark.sql(
        f"ALTER TABLE {PREDICTION_EVENTS_TABLE} "
        "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )
    scored.createOrReplaceTempView("scored_batch")
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
    scored.write.format("delta").mode("append").saveAsTable(PREDICTION_EVENTS_TABLE)
    spark.sql(
        f"ALTER TABLE {PREDICTION_EVENTS_TABLE} "
        "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )

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
# MAGIC ## Use the current-state view
# MAGIC
# MAGIC Consumers see one row per pitch without duplicating the data. The event table retains
# MAGIC the evidence monitoring and audit workflows need when a new model version takes over.

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
