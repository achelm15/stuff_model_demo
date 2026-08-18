# Databricks notebook source
# MAGIC %md
# MAGIC # 00b · Load Pitch Data from the Published CSV
# MAGIC
# MAGIC A fast alternative to notebook `00`. Instead of pulling MLB GUMBO feeds from the
# MAGIC API, this downloads the published training CSV from the repo's GitHub Release,
# MAGIC extracts it into a Unity Catalog Volume, and writes `catalog.schema.silver_pitches`
# MAGIC directly. Use this to get a working `silver_pitches` in a couple of minutes;
# MAGIC use `00` when you want to rebuild the table from source.
# MAGIC
# MAGIC The CSV is the same feature set with the `pitch_rv` target that `00` produces.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

dbutils.widgets.text(
    "release_url",
    "https://github.com/achelm15/stuff_model_demo/releases/download/data-v1/Pitch_Stuff_Model_Training.csv.zip",
)
dbutils.widgets.text("volume", "gumbo_raw")
dbutils.widgets.dropdown("overwrite_silver", "true", ["true", "false"])

RELEASE_URL = dbutils.widgets.get("release_url").strip()
VOLUME = dbutils.widgets.get("volume").strip()
OVERWRITE = dbutils.widgets.get("overwrite_silver").lower() == "true"

SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_pitches"
IMPORT_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/release_import"
ZIP_PATH = f"{IMPORT_DIR}/pitch_stuff_training.zip"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download and extract
# MAGIC The zip lands in a Volume, not the Git folder. The archive carries a macOS
# MAGIC `__MACOSX` resource-fork entry alongside the real CSV, so pick the CSV explicitly.

# COMMAND ----------

import os
import urllib.request
import zipfile

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
os.makedirs(IMPORT_DIR, exist_ok=True)

print(f"Downloading {RELEASE_URL}")
urllib.request.urlretrieve(RELEASE_URL, ZIP_PATH)

with zipfile.ZipFile(ZIP_PATH) as archive:
    csv_members = [
        name for name in archive.namelist()
        if name.lower().endswith(".csv") and not name.startswith("__MACOSX")
    ]
    assert len(csv_members) == 1, f"Expected one CSV in the zip, found: {csv_members}"
    csv_member = csv_members[0]
    archive.extract(csv_member, IMPORT_DIR)

CSV_PATH = f"{IMPORT_DIR}/{csv_member}"
print(f"Extracted {CSV_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write `silver_pitches`
# MAGIC `inferSchema` reads types from the CSV; notebook `00` remains the canonical schema
# MAGIC if you ever need to reconcile columns. `overwrite_silver=false` refuses to clobber
# MAGIC an existing table.

# COMMAND ----------

raw = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(CSV_PATH)
)

(
    raw.write.format("delta")
    .option("overwriteSchema", "true")
    .mode("overwrite" if OVERWRITE else "errorifexists")
    .saveAsTable(SILVER_TABLE)
)

print(f"Wrote {spark.table(SILVER_TABLE).count():,} rows to {SILVER_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify
# MAGIC Check the season coverage before you set the `train_season` / `inference_season`
# MAGIC widgets in later notebooks - the loaded table only has the seasons in this export.

# COMMAND ----------

from pyspark.sql import functions as F

silver = spark.table(SILVER_TABLE)
display(
    silver.groupBy("season")
    .agg(F.count("*").alias("pitches"), F.countDistinct("pitcher_id").alias("pitchers"))
    .orderBy("season")
)
display(silver.limit(5))
