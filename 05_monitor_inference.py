# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "/Workspace/Users/andrew.helmreich@databricks.com/rockies-mlflow-demo/rockies-mlflow-demo.yaml"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 05 · Monitor Production Inference
# MAGIC
# MAGIC Create or reuse a managed Data Quality Inference profile on the prediction-events
# MAGIC table written by notebook 04. Databricks computes model quality, feature profiles,
# MAGIC prediction profiles, and consecutive-window drift—with no copied monitor-input table
# MAGIC and no custom baseline pipeline.

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
# MAGIC ## Define the monitoring contract
# MAGIC
# MAGIC This is the complete handoff to Databricks: the prediction, optional ground-truth
# MAGIC label, inference timestamp, immutable model version, and daily aggregation window.
# MAGIC `pitch_type` is the one useful business slice.

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

workspace = WorkspaceClient()
schema_info = workspace.schemas.get(full_name=f"{CATALOG}.{SCHEMA}")
table_info = workspace.tables.get(full_name=PREDICTION_EVENTS_TABLE)

config = DataProfilingConfig(
    output_schema_id=schema_info.schema_id,
    assets_dir=ASSETS_DIR,
    slicing_exprs=["pitch_type"],
    inference_log=InferenceLogConfig(
        problem_type=InferenceProblemType.INFERENCE_PROBLEM_TYPE_REGRESSION,
        prediction_column="predicted_run_value",
        label_column="actual_run_value",
        timestamp_column="scored_at",
        model_id_column="model_version",
        granularities=[AggregationGranularity.AGGREGATION_GRANULARITY_1_DAY],
    ),
)
if WAREHOUSE_ID:
    config.warehouse_id = WAREHOUSE_ID

# COMMAND ----------
# MAGIC %md
# MAGIC ## Refresh the managed profile
# MAGIC
# MAGIC On creation, Databricks reads inference events from the preceding 30 days. After that,
# MAGIC Change Data Feed lets refreshes process new events incrementally. In a production Job,
# MAGIC this task runs after inference; profile failures fail the task rather than being hidden.

# COMMAND ----------

monitor = ensure_monitor(workspace, table_info.table_id, config)
refresh = refresh_and_wait(workspace, table_info.table_id)

monitor_config = monitor.data_profiling_config
profile_table = monitor_config.profile_metrics_table_name
drift_table = monitor_config.drift_metrics_table_name

print("refresh:", refresh.refresh_id, refresh.state)
print("profile metrics:", profile_table)
print("drift metrics:", drift_table)
print("dashboard:", monitor_config.dashboard_id)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Inspect Databricks-managed outputs
# MAGIC
# MAGIC The first batch produces profile and regression-quality metrics. Consecutive drift
# MAGIC appears only after a second daily window exists. That is the real production behavior:
# MAGIC the initial 30-day scan is a lookback boundary, not an invented baseline.

# COMMAND ----------

model_id_column = monitor_config.inference_log.model_id_column
display(
    spark.table(profile_table)
    .where((F.col("column_name") == ":table") & (F.col("log_type") == "INPUT"))
    .orderBy(F.col("window.start"), model_id_column)
)

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
