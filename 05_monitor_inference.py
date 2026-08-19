# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "environment.yaml"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 05 · Monitor Production Inference
# MAGIC
# MAGIC Databricks can watch a table of predictions and compute quality and drift metrics over
# MAGIC time on its own. This notebook points that managed monitoring (a "Data Quality"
# MAGIC profile) at the prediction-events table from notebook 04. You write no metric code and
# MAGIC copy no data: Databricks profiles the predictions and the inputs, compares each day's
# MAGIC window against the day before, and writes the results to two metrics tables plus a
# MAGIC generated dashboard.
# MAGIC
# MAGIC The steps are: describe the table to Databricks (the config), create the monitor and
# MAGIC run the first computation, then read the metrics tables it produced.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

# MAGIC %run ./_monitoring_helpers

# COMMAND ----------

dbutils.widgets.text("warehouse_id", "")

WAREHOUSE_ID = dbutils.widgets.get("warehouse_id").strip()

PREDICTION_EVENTS_TABLE = f"{CATALOG}.{SCHEMA}.gold_pitch_prediction_events"
ASSETS_DIR = (
    "/Workspace/Shared/rockies_mlflow_monitoring/"
    + PREDICTION_EVENTS_TABLE.replace(".", "_")
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Describe the prediction-events table to Databricks
# MAGIC
# MAGIC The config below is the whole handoff: it tells Databricks that this table is an
# MAGIC inference log and which columns mean what. Because we point it at the ground-truth
# MAGIC label as well as the prediction, it can compute regression-quality metrics (not just
# MAGIC data profiles), broken down per day and per model version.

# COMMAND ----------

import json

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.dataquality import (
    AggregationGranularity,
    DataProfilingConfig,
    InferenceLogConfig,
    InferenceProblemType,
)
from pyspark.sql import functions as F

assert spark.catalog.tableExists(PREDICTION_EVENTS_TABLE), "Run notebook 04 first."

# WorkspaceClient is the Databricks SDK entry point. The monitor is configured with the
# numeric IDs of the schema (where its output tables go) and of the table being monitored.
workspace = WorkspaceClient()
schema_info = workspace.schemas.get(full_name=f"{CATALOG}.{SCHEMA}")
table_info = workspace.tables.get(full_name=PREDICTION_EVENTS_TABLE)

config = DataProfilingConfig(
    output_schema_id=schema_info.schema_id,  # where Databricks creates the metrics tables
    assets_dir=ASSETS_DIR,                   # workspace folder for the generated dashboard
    slicing_exprs=["pitch_type"],            # also compute every metric per pitch type (FF, SI)
    inference_log=InferenceLogConfig(
        problem_type=InferenceProblemType.INFERENCE_PROBLEM_TYPE_REGRESSION,
        prediction_column="predicted_run_value",
        label_column="actual_run_value",  # having the actual outcome enables quality metrics
        timestamp_column="scored_at",      # the column that defines the daily time windows
        model_id_column="model_version",   # metrics are tracked separately per model version
        granularities=[AggregationGranularity.AGGREGATION_GRANULARITY_1_DAY],  # one window per day
    ),
)
# Optional: run the metric queries on a specific SQL warehouse instead of the default.
if WAREHOUSE_ID:
    config.warehouse_id = WAREHOUSE_ID

# COMMAND ----------
# MAGIC %md
# MAGIC ## Create the monitor and compute the first metrics
# MAGIC
# MAGIC Both calls below are small wrappers defined in `_monitoring_helpers`:
# MAGIC
# MAGIC - `ensure_monitor` creates the managed monitor on the table the first time, and reuses
# MAGIC   the existing one on later runs.
# MAGIC - `refresh_and_wait` triggers one metric computation and blocks until it finishes, so a
# MAGIC   failure fails this notebook instead of passing silently.
# MAGIC
# MAGIC On the first refresh Databricks reads the preceding 30 days of events to establish a
# MAGIC baseline; after that, Change Data Feed lets each refresh process only the new rows.
# MAGIC When it finishes, the monitor exposes the names of the two tables it created (profile
# MAGIC metrics and drift metrics) and the ID of its generated dashboard.

# COMMAND ----------

monitor = ensure_monitor(workspace, table_info.table_id, config)
refresh = refresh_and_wait(workspace, table_info.table_id)

# The monitor tells us where it wrote its results.
monitor_config = monitor.data_profiling_config
profile_table = monitor_config.profile_metrics_table_name  # per-window column/quality metrics
drift_table = monitor_config.drift_metrics_table_name      # window-over-window change metrics

print("refresh:", refresh.refresh_id, refresh.state)
print("profile metrics:", profile_table)
print("drift metrics:", drift_table)
print("dashboard:", monitor_config.dashboard_id)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Read the metrics tables Databricks produced
# MAGIC
# MAGIC The first refresh produces profile and regression-quality metrics. Drift compares one
# MAGIC daily window to the previous one, so it only appears once a second day of predictions
# MAGIC exists. That is the real production behavior: the initial 30-day scan is a lookback
# MAGIC boundary, not an invented baseline.

# COMMAND ----------

model_id_column = monitor_config.inference_log.model_id_column

# profile_metrics has one row per (time window, model version, column). The special
# column_name ":table" is the whole-table summary rather than a single column, and log_type
# "INPUT" is the scored data (vs the monitor's own baseline). So this shows the per-day,
# per-version row counts and prediction summary.
display(
    spark.table(profile_table)
    .where((F.col("column_name") == ":table") & (F.col("log_type") == "INPUT"))
    .orderBy(F.col("window.start"), model_id_column)
)

# drift_metrics with drift_type "CONSECUTIVE" is each daily window compared to the one before
# it. The table only exists once there are at least two daily windows to compare.
if drift_table and spark.catalog.tableExists(drift_table):
    display(
        spark.table(drift_table)
        .where(F.col("drift_type") == "CONSECUTIVE")
        .orderBy(F.col("window.start"), "column_name", model_id_column)
    )
else:
    print("No drift table yet: another daily inference window is required.")

dbutils.notebook.exit(
    json.dumps(
        {
            "prediction_events_table": PREDICTION_EVENTS_TABLE,
            "profile_metrics_table": profile_table,
            "drift_metrics_table": drift_table,
            "dashboard_id": monitor_config.dashboard_id,
            "refresh_id": refresh.refresh_id,
            "refresh_state": str(refresh.state),
        }
    )
)
