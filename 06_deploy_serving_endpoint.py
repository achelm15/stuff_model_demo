# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Deploy Model Serving Endpoint
# MAGIC
# MAGIC Creates or updates a Databricks Model Serving endpoint for the selected Unity
# MAGIC Catalog model alias. The endpoint resolves the alias to an immutable model version
# MAGIC and uses scale-to-zero for workshop cost control.

# COMMAND ----------

dbutils.widgets.text("catalog", "ahelmreich_demo")
dbutils.widgets.text("schema", "rockies_mlflow_workshop")
dbutils.widgets.text("model_alias", "champion")
dbutils.widgets.text("endpoint_name", "rockies-stuff-rv")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_ALIAS = dbutils.widgets.get("model_alias").strip()
ENDPOINT_NAME = dbutils.widgets.get("endpoint_name")
MODEL_NAME = f"{CATALOG}.{SCHEMA}.fastball_stuff_rv"

# COMMAND ----------

import json
import time

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.deployments import get_deploy_client
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
registry = MlflowClient(registry_uri="databricks-uc")
selected = registry.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
VERSION = str(selected.version)

served_entity = {
    "entity_name": MODEL_NAME,
    "entity_version": VERSION,
    "workload_size": "Small",
    "scale_to_zero_enabled": True,
}
config = {
    "served_entities": [served_entity],
    "traffic_config": {
        "routes": [{
            "served_model_name": f"fastball_stuff_rv-{VERSION}",
            "traffic_percentage": 100,
        }]
    },
}

deploy = get_deploy_client("databricks")
w = WorkspaceClient()

try:
    deploy.create_endpoint(name=ENDPOINT_NAME, config=config)
    action = "created"
except Exception as create_error:
    print("Create failed or endpoint already exists; attempting update:", str(create_error)[:500])
    deploy.update_endpoint(endpoint=ENDPOINT_NAME, config=config)
    action = "updated"

deadline = time.time() + 30 * 60
last_state = None
while time.time() < deadline:
    endpoint = w.serving_endpoints.get(ENDPOINT_NAME)
    ready = getattr(endpoint.state, "ready", None)
    config_update = getattr(endpoint.state, "config_update", None)
    ready_value = str(getattr(ready, "value", ready)).split(".")[-1]
    config_update_value = str(getattr(config_update, "value", config_update)).split(".")[-1]
    last_state = {"ready": ready_value, "config_update": config_update_value}
    print(time.strftime("%H:%M:%S"), last_state)
    if ready_value == "READY" and config_update_value == "NOT_UPDATING":
        break
    time.sleep(30)

assert last_state is not None and last_state["ready"] == "READY", (
    f"Endpoint did not become ready before the timeout: {last_state}"
)
assert last_state["config_update"] == "NOT_UPDATING", (
    f"Endpoint configuration is still updating: {last_state}"
)

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
