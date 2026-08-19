# Fastball "stuff" model demo

Databricks notebooks that build a pitch-level "stuff" model end to end: pull MLB
pitch data, train and register an XGBoost run-value model, score in batch, monitor
it, and serve it. The notebooks are the demo; there is no app here.

`pitch_rv` is the target (run value per pitch, lower is better for the pitcher). The
model predicts it from release speed, spin, movement, and release geometry.

## What's here

| File | What it does |
|------|--------------|
| `_config.py` | Shared `catalog` / `schema` widgets. Every notebook `%run`s this so the target lives in one place. |
| `00_materialize_pitch_data.py` | Pull MLB GUMBO feeds through a bronze/silver pipeline into the `silver_pitches` table. |
| `00b_load_from_release_csv.py` | Fast alternative to `00`: download the published CSV from the Release, extract it to a Volume, and write `silver_pitches` directly. |
| `01_current_training_workflow.py` | The anti-pattern: train, print a metric, tweak in place, lose track. No MLflow. Shown on purpose as the "before". |
| `02_explore_models.py` | Development notebook. Tune candidates in a personal MLflow experiment. |
| `03_train_register_model.py` | Production entrypoint. Tune, evaluate on a holdout, register to Unity Catalog, set the `@champion` alias. |
| `04_score_batch.py` | Batch inference. Score with the registered model and write one predictions table. |
| `05_monitor_inference.py` | Managed Data Quality / inference monitoring on the predictions table. |
| `06_deploy_serving_endpoint.py` | Create or update a Model Serving endpoint for the champion model. |
| `07_genie_code_stuff_model.py` | Alternative bootstrap: the same model scaffolded by Databricks Genie Code. |
| `_monitoring_helpers.py` | Helpers `%run` by notebook 05. |
| `environment.yaml` | Pinned dependencies (mlflow, xgboost, optuna, scikit-learn, shap, lightgbm) used by the serverless environment header. |
| `rockies-mlflow-conventions/` | The workspace skill notebook `07` `@`-mentions: the house MLflow + Unity Catalog conventions for training the stuff model. |

## Prerequisites

- A Databricks workspace with Unity Catalog and serverless notebooks (the notebooks
  pin their libraries through `environment.yaml`). Databricks Runtime ML also works
  if you match the versions.
- Rights to create a schema, a volume, tables, an MLflow experiment, and a registered
  model in the catalog/schema you point at.
- A SQL warehouse id for notebook 05 (monitoring).

## Setup

Steps:

1. Clone this repo into Databricks as a Git folder (Workspace -> Git folder -> add
   this repo URL). Cloning as a Git folder is what makes the relative
   `base_environment = "environment.yaml"` resolve.
2. Open `_config` (or any notebook) and set the `catalog` and `schema` widgets. They
   are intentionally blank and the notebooks error until you set them. You set them on
   whichever notebook you are running, or pass them as job parameters.
3. Run the notebooks in order (see below).

## Run order

Run `00` first, then `02 -> 03 -> 04 -> 05 -> 06`. `01` is an illustration you can run
any time to contrast with the MLflow workflow. `07` is an alternative to `02`/`03`.

1. `00` materializes `silver_pitches` from the API, or run `00b` to load the published CSV instead (see Training data).
2. `02` explores and tunes candidates in a personal experiment.
3. `03` trains the reviewed model, registers it, and sets `@champion`.
4. `04` scores the current season into the predictions table.
5. `05` attaches monitoring. Set the `warehouse_id` widget here.
6. `06` deploys the champion to a serving endpoint.

Per-notebook widgets (besides `catalog`/`schema`) let you set season, tuning trials,
training device (`gpu`/`cpu`), model alias, endpoint name, and the warehouse id.

## Training data

The full training set is a Release asset, not part of the git tree, so cloning the
repo does not bring it and it never bloats the repo:

https://github.com/achelm15/stuff_model_demo/releases/tag/data-v1

`Pitch_Stuff_Model_Training.csv.zip` is 235 MB zipped, about 803 MB unzipped. It is the
same four-seam/sinker feature set with the `pitch_rv` target that notebook `00` produces.

You have two ways to get data:

- **Rebuild it**: run notebook `00`. This pulls from the MLB API into `silver_pitches`
  and needs no download. This is the canonical path.
- **Load the CSV**: run notebook `00b`. It downloads the Release zip into a UC Volume,
  extracts the CSV, and writes `silver_pitches` for you. Faster than the API pull, and
  it keeps the file in a Volume, not the Git folder. The `release_url` and `volume` are
  widgets if you need to change them.

The loaded table only covers the seasons in the export, so check `00b`'s season summary
before you set the `train_season` / `inference_season` widgets in later notebooks.

## Gotchas

- `catalog` and `schema` are blank by default and every notebook stops until you set
  them. This is deliberate so you never run against the wrong target by accident.
- `base_environment` is a relative path. It resolves when you run from a Git folder. If
  you import the notebooks into a plain workspace folder instead, point it at wherever
  you place `environment.yaml`.
- Keep the CSV in a UC Volume. Git folders are not the place for an 800 MB file.
