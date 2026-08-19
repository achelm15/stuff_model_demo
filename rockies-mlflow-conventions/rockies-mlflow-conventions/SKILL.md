---
name: rockies-mlflow-conventions
description: The Rockies house conventions for training an XGBoost run-value ("stuff") model on Databricks with MLflow and Unity Catalog. Load this skill when someone asks an agent to train, tune, or prototype the fastball stuff model, or any regressor that should follow the same MLflow tracking, model logging, and UC registration pattern. Produces a governed candidate, never a promotion.
---

# Rockies MLflow + Unity Catalog conventions

This skill encodes how this team trains and registers models so that an agent produces the
same shape of code every time, matching the governed workflow in notebook 03. Generate all
code directly in notebook cells so the user can read and edit it. Do not import from a hidden
module.

The output is a Unity Catalog model version tagged as a **candidate**. Promotion (moving the
`@champion` alias) is a separate, governed step and this skill never does it.

## When to use

- Train or tune the fastball stuff model (predict `pitch_rv` for four-seam and sinker fastballs).
- Prototype any regressor that should follow the same tracking, logging, and registration rules.
- Reproduce the notebook 03 pattern quickly from a natural-language prompt.

## House rules (do not deviate)

1. **Serverless environment, not `%pip`.** The notebook declares its libraries through the
   `environment.yaml` script header, so do not emit a `%pip install` cell. Assume `xgboost`,
   `optuna`, `shap`, `scikit-learn`, and `mlflow` are already present.
2. **Read governed data from Unity Catalog.** Source is `{catalog}.{schema}.silver_pitches`
   (the target set on the `_config` widgets), filtered to `pitch_type IN ('FF', 'SI')`. Cast the
   target to double, pull to pandas, and drop nulls in the model inputs and target.
3. **Honest evaluation.** Hold out 20% as an untouched test set that is never fit on. Split the
   remaining data again into train and validation for tuning. Use `random_state=42` everywhere a
   split or sampler takes a seed, so runs are reproducible.
4. **Registry URI is `databricks-uc`.** Set it before logging or registering.
5. **Register as a candidate, never promote.** Tag the new version `promotion_status=candidate`
   and `workflow=<the workflow name>`. Do not touch the `@champion` alias.

## Model shape

A scikit-learn `Pipeline` with two steps:

- `preprocess`: a `ColumnTransformer` that one-hot encodes the categorical features
  (`OneHotEncoder(handle_unknown="ignore", sparse_output=False)`) and passes the numeric
  features through unchanged. Set `verbose_feature_names_out=False` so downstream names stay clean.
- `model`: `XGBRegressor(objective="reg:squarederror", tree_method="hist", device=<device>,
  n_jobs=-1, random_state=42, **tuned_params)`.

Feature groups for the stuff model:

- Numeric: `release_speed`, `spin_rate`, `spin_direction`, `pfx_x_hnorm`, `pfx_z`,
  `release_x_hnorm`, `release_z`, `extension`, `vaa`.
- Categorical: `p_throws`, `stand`.
- Target: `pitch_rv` (run value per pitch, lower is better for the pitcher).

The `device` comes from a `training_device` widget with values `auto` / `gpu` / `cpu`. `auto`
uses CUDA when `nvidia-smi` is present and falls back to CPU; `gpu` asserts a GPU is present;
`cpu` forces CPU.

## Notebook cell structure

Generate these cells in order, each with a clear title.

### Cell 1: `%run ./_config`

Load the shared `catalog` / `schema` widgets. Do not hardcode the target.

### Cell 2: Widgets and names

Define widgets `train_season` (default 2024), `n_trials` (default 8), and a `training_device`
dropdown (`auto`, `gpu`, `cpu`). Then set:

- `SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_pitches"`
- `MODEL_NAME = f"{CATALOG}.{SCHEMA}.fastball_stuff_rv"`
- `EXPERIMENT_NAME = f"/Users/{current_user}/rockies-mlflow-demo/<descriptive-name>"`

### Cell 3: Setup

Set the registry URI to `databricks-uc`, set the experiment, and set Optuna verbosity to
WARNING. Serverless blocks the py4j call MLflow uses to resolve some run-context tags, so raise
`mlflow.tracking.context.registry` to ERROR to silence that one benign warning (runs still log
fine). Resolve the XGBoost `device` from the `training_device` widget as described above. Assert
the silver table exists with a message pointing at notebook 00.

### Cell 4: Load fastball rows and reserve a holdout

Read `silver_pitches` for the season, filter to `('FF', 'SI')`, select the model inputs plus the
cast target, `toPandas`, drop nulls, reset the index. Split off 20% as an untouched test set,
then split the remainder into train and validation. Print the three row counts.

Define two helpers here:

- `make_model(params)` returns the `Pipeline` described above.
- `regression_metrics(actual, prediction, prefix)` returns a dict of `{prefix}_rmse`,
  `{prefix}_mae`, `{prefix}_r2`, each cast to `float`.

### Cell 5: Tune with Optuna as nested MLflow runs, then register a candidate

Open one parent MLflow run and set tags describing the workflow (for example `workflow`,
`algorithm=xgboost`, `feature_set`, and who authored it). Inside it:

1. Define `suggest_params(trial)` with this search space:

   | Hyperparameter | Range | Scale |
   |---|---|---|
   | `n_estimators` | 200-800 | int |
   | `learning_rate` | 0.01-0.16 | log |
   | `max_depth` | 3-10 | int |
   | `min_child_weight` | 1.0-20.0 | log float |
   | `subsample` | 0.70-1.00 | float |
   | `colsample_bytree` | 0.70-1.00 | float |
   | `reg_lambda` | 1e-3-5.0 | log |

2. Define the Optuna `objective(trial)` as a nested function. Each trial opens a **nested**
   MLflow run named `trial-{number:03d}`, fits `make_model(params)` on the train split, predicts
   the validation split, logs the params and the validation metrics, and returns the validation
   RMSE.

3. Create the study with `direction="minimize"` and `TPESampler(seed=42)`, and optimize for
   `n_trials`. Direction is `minimize` because the objective returns RMSE directly (not a negated
   sklearn score).

4. Refit `make_model(study.best_params)` on train plus validation, predict the untouched test
   set, and log the best params and the test metrics on the parent run. Log a
   `feature_contract.json` artifact with the numeric features, categorical features, and target.

5. Make the artifact portable: set the estimator and its booster `device` to `cpu` before
   logging, so batch and serving inference do not require a GPU.

6. Log the model with `mlflow.sklearn.log_model` using an `input_example`, a `signature` from
   `infer_signature`, `registered_model_name=MODEL_NAME`, and the cloudpickle serialization
   format. Then reload it with `mlflow.pyfunc.load_model` and assert it predicts, as a sanity
   check that the artifact round-trips.

7. After the run, tag the new version: `workflow=<name>` and `promotion_status=candidate`. Print
   the registered version, the held-out test metrics, and a reminder that promotion happens in
   notebook 03. Do not move `@champion`.

## Key implementation rules

1. All code goes directly in notebook cells. Never import from a separate module.
2. Tuning and training live in one parent run; the Optuna objective is a nested function that
   opens a nested run per trial.
3. Seed everything with `random_state=42` (splits and the TPE sampler) for reproducibility.
4. Set `n_jobs=-1` on the XGBoost estimator.
5. Adapt feature selection to the modeling question, but keep the numeric/categorical split and
   the `ColumnTransformer` shape.
6. Register to Unity Catalog as a candidate only. Promotion is out of scope for this skill.
