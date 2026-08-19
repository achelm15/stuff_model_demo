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
# MAGIC `@`-mentions a **personal skill** that encodes our MLflow + Unity Catalog conventions.
# MAGIC
# MAGIC The point is not the model. Notebook 03 remains the governed, service-principal training
# MAGIC entry point. The point is how fast an agent gets you to a correct first draft that already
# MAGIC follows house style, so your agent produces the same shape of code every time.
# MAGIC
# MAGIC The skill lives in your own user folder at
# MAGIC `/Workspace/Users/<your-username>/.assistant/skills/rockies-mlflow-conventions/SKILL.md` (the
# MAGIC `rockies-mlflow-conventions/` folder in this repo). It is scoped to you; a workspace admin can
# MAGIC later promote it to the shared `/Workspace/.assistant/skills/` path to give the whole team the
# MAGIC same conventions.
# MAGIC
# MAGIC ## Install the skill (run the cell below first)
# MAGIC
# MAGIC The next cell copies this repo's `rockies-mlflow-conventions/SKILL.md` into your personal
# MAGIC Assistant skills folder so Genie Code can find it. Run it once. If the skill is already there
# MAGIC (for example a teammate installed the workspace-wide copy), you can skip it.
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

# Install this repo's copy of the skill into your personal Assistant skills folder. You can write
# under your own /Workspace/Users/<you> path yourself; the workspace-wide /Workspace/.assistant
# path needs admin rights.
import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat

w = WorkspaceClient()
user_name = w.current_user.me().user_name

# This notebook and the skill folder are siblings in the Git folder. Resolve the notebook's own
# directory from its workspace path so the read works regardless of the current working directory.
notebook_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
repo_dir = os.path.dirname(f"/Workspace{notebook_path}")
skill_source = f"{repo_dir}/rockies-mlflow-conventions/SKILL.md"

skill_dir = f"/Workspace/Users/{user_name}/.assistant/skills/rockies-mlflow-conventions"
with open(skill_source, "rb") as handle:
    skill_bytes = handle.read()

w.workspace.mkdirs(skill_dir)
w.workspace.upload(
    f"{skill_dir}/SKILL.md", skill_bytes, format=ImportFormat.RAW, overwrite=True
)
print(f"Installed the skill at {skill_dir}/SKILL.md")
print("Reference it in Genie Code as @rockies-mlflow-conventions.")

# COMMAND ----------
