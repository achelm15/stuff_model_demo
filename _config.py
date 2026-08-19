# Databricks notebook source
# MAGIC %md
# MAGIC # _config · Shared catalog / schema configuration
# MAGIC
# MAGIC `%run` this notebook from 00-07 to get `CATALOG` and `SCHEMA` in one place.
# MAGIC Both widgets are intentionally **blank**. Set them once (at the top of the
# MAGIC notebook you are running, or as job parameters) before executing anything.
# MAGIC Change the target here and every notebook that `%run`s this file follows.

# COMMAND ----------

dbutils.widgets.text("catalog", "")
dbutils.widgets.text("schema", "")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()

assert CATALOG and SCHEMA, (
    "Set the `catalog` and `schema` widgets before running; they are intentionally "
    "blank. Fill them in at the top of the notebook (or pass them as job parameters)."
)

print(f"Unity Catalog target: {CATALOG}.{SCHEMA}")
