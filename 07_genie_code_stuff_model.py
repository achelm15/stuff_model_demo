# Databricks notebook source
# /// script
# [tool.databricks.environment]
# base_environment = "environment.yaml"
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # 07 · Bootstrapping the Stuff Model with Genie Code
# MAGIC
# MAGIC This notebook starts empty on purpose. You fill it in live with **Databricks Genie Code**
# MAGIC (the in-workspace agent-mode coding assistant) from a single prompt, to show that an agent
# MAGIC can scaffold the same governed training workflow as notebooks 02 and 03 because the prompt
# MAGIC `@`-mentions a shared **workspace skill** that encodes our MLflow + Unity Catalog conventions.
# MAGIC
# MAGIC The point is not the model. Notebook 03 remains the governed, service-principal training
# MAGIC entry point. The point is how fast an agent gets you to a correct first draft that already
# MAGIC follows house style, so every teammate's agent produces the same shape of code.
# MAGIC
# MAGIC The skill lives at `Workspace/.assistant/skills/rockies-mlflow-conventions/SKILL.md` (the
# MAGIC `rockies-mlflow-conventions/` folder in this repo) and is available to everyone in the workspace.
# MAGIC
# MAGIC ## Try it
# MAGIC
# MAGIC Set the `catalog` and `schema` widgets (or run `_config` first), then open Genie Code and
# MAGIC give it this prompt:
# MAGIC
# MAGIC > `@rockies-mlflow-conventions` Train an XGBoost "stuff" regressor that predicts `pitch_rv`
# MAGIC > for four-seam and sinker fastballs in `<catalog>.<schema>.silver_pitches` (the target you
# MAGIC > set on the `_config` widgets). Use the engineered feature set, tune with Optuna as nested
# MAGIC > MLflow runs, evaluate on an untouched holdout, log the model with a signature and input
# MAGIC > example, and register a **candidate** version to Unity Catalog without moving the
# MAGIC > `@champion` alias.
# MAGIC
# MAGIC ## What you should see happen
# MAGIC
# MAGIC 1. Genie Code loads the `rockies-mlflow-conventions` skill and follows it instead of guessing
# MAGIC    a generic training recipe.
# MAGIC 2. It writes the cells in order: `%run ./_config`, widgets and names, setup, load
# MAGIC    `silver_pitches` with a 20% untouched holdout, a tuning function, and the Optuna plus
# MAGIC    registration cell. The code should look like notebook 03, not a generic tutorial.
# MAGIC 3. Running it produces one parent MLflow run with a nested run per Optuna trial, each logging
# MAGIC    its validation RMSE.
# MAGIC 4. It logs the final model with a signature and input example, writes a `feature_contract.json`
# MAGIC    artifact, and registers a new version of `<catalog>.<schema>.fastball_stuff_rv` to Unity
# MAGIC    Catalog.
# MAGIC 5. The new version is tagged `promotion_status=candidate` and the `@champion` alias does not
# MAGIC    move. Held-out test metrics print at the end.
# MAGIC
# MAGIC ## How to verify
# MAGIC
# MAGIC - The MLflow experiment shows the parent run with nested trial runs underneath it.
# MAGIC - The model in Unity Catalog has a new version tagged `promotion_status=candidate`, and
# MAGIC   `@champion` still points at whatever notebook 03 last promoted.
# MAGIC
# MAGIC Review the generated recipe, and move anything worth keeping into notebook 03 for governed
# MAGIC promotion. Genie Code gets you a correct draft fast; promotion stays a deliberate, governed step.

# COMMAND ----------
