# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Deploy Model Serving Endpoint
# MAGIC
# MAGIC Creates or updates a Databricks Model Serving endpoint for the selected Unity
# MAGIC Catalog model alias. The endpoint resolves the alias to an immutable model version
# MAGIC and uses scale-to-zero for workshop cost control.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

dbutils.widgets.text("model_alias", "champion")
dbutils.widgets.text("endpoint_name", "rockies-stuff-rv")

MODEL_ALIAS = dbutils.widgets.get("model_alias").strip()
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name").strip()
MODEL_NAME = f"{CATALOG}.{SCHEMA}.fastball_stuff_rv"

# COMMAND ----------

import json
from datetime import timedelta

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import ResourceDoesNotExist
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    Route,
    ServedEntityInput,
    TrafficConfig,
)
from mlflow.tracking import MlflowClient

# Resolve the alias to a concrete version now. The endpoint pins that immutable version, so
# moving @champion later does not silently repoint a running endpoint; you re-run this notebook.
mlflow.set_registry_uri("databricks-uc")
registry = MlflowClient(registry_uri="databricks-uc")
selected = registry.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
VERSION = str(selected.version)

# One served entity, all traffic to it. SERVED_NAME is a stable label we choose and reference
# from the traffic route, so the route does not depend on the version string.
SERVED_NAME = "fastball-stuff-rv"
served_entity = ServedEntityInput(
    name=SERVED_NAME,
    entity_name=MODEL_NAME,
    entity_version=VERSION,
    workload_size="Small",
    scale_to_zero_enabled=True,  # workshop cost control: idle endpoints cost nothing
)
traffic_config = TrafficConfig(
    routes=[Route(served_entity_name=SERVED_NAME, traffic_percentage=100)]
)

# Create the endpoint if it is new, otherwise update its config. The *_and_wait calls block
# until the endpoint finishes provisioning and reaches a ready, not-updating state, and raise
# on timeout or a failed rollout, so there is no hand-rolled polling loop to maintain.
w = WorkspaceClient()
try:
    w.serving_endpoints.get(ENDPOINT_NAME)
    exists = True
except ResourceDoesNotExist:
    exists = False

if exists:
    endpoint = w.serving_endpoints.update_config_and_wait(
        name=ENDPOINT_NAME,
        served_entities=[served_entity],
        traffic_config=traffic_config,
        timeout=timedelta(minutes=30),
    )
    action = "updated"
else:
    endpoint = w.serving_endpoints.create_and_wait(
        name=ENDPOINT_NAME,
        config=EndpointCoreConfigInput(
            served_entities=[served_entity],
            traffic_config=traffic_config,
        ),
        timeout=timedelta(minutes=30),
    )
    action = "created"

last_state = {
    "ready": str(endpoint.state.ready),
    "config_update": str(endpoint.state.config_update),
}
print(
    f"{action} endpoint {ENDPOINT_NAME} serving {MODEL_NAME} "
    f"version {VERSION} resolved from @{MODEL_ALIAS}"
)
dbutils.notebook.exit(json.dumps({
    "endpoint_name": ENDPOINT_NAME,
    "model_name": MODEL_NAME,
    "model_version": VERSION,
    "model_alias": MODEL_ALIAS,
    "action": action,
    "state": last_state,
}))
