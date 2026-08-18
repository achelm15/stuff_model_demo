---
name: xgboost-tuning
description: Tune and train XGBoost models (regression or classification) with hyperparameter search, MLflow experiment logging, optional calibration plots, and SHAP feature importance. Load this skill when the user asks to train, tune, or optimize an XGBoost model, or wants a reusable XGBoost training function.
---

# XGBoost Tuning & Training Skill

This skill guides the agent to build out an XGBoost training pipeline directly in notebook cells. Do NOT import from a module — generate all code inline so the user has full visibility and can customize.

## When to use

- The user wants to train an XGBoost regressor or classifier with automatic hyperparameter tuning.
- The user wants MLflow tracking of tuning trials and the final model.
- The user asks for calibration or SHAP explainability visualizations.

## Prerequisites

Install these packages in a `%pip` cell (all available on Databricks Runtime ML; for standard runtime, pip install them):

```
xgboost, optuna, shap, scikit-learn, mlflow, matplotlib, numpy, pandas
```

## Notebook Cell Structure

Generate the following cells in order. Each cell should be self-contained and titled clearly.

### Cell 1: Install dependencies

```python
%pip install xgboost optuna shap scikit-learn mlflow matplotlib numpy pandas --quiet
```

### Cell 2: Load and prepare data

- Load the user's data into a pandas DataFrame.
- Define the feature columns and target column based on the user's request.
- Drop rows with nulls in the relevant columns.
- Print dataset size, target distribution, and feature count.
- Perform a train/test split (default 80/20, stratified for classification).

### Cell 3: Define and run the training function

Generate a **single function** that encapsulates the entire tuning + training + evaluation + MLflow logging pipeline. The Optuna objective must be defined as a **nested function** inside it.

Function signature:

```python
def tune_and_train_xgboost(
    X_train, y_train, X_test, y_test,
    task="classification",
    experiment_name="/Users/{user_email}/{name}-experiment",
    n_tuning_trials=50,
    early_stopping_rounds=50,
    random_state=42,
    produce_calibration_plot=False,
    produce_shap_plot=False,
    registered_model_name=None,
):
```

Inside the function, implement in this order:

1. **Setup** — suppress warnings, set Optuna verbosity to WARNING, determine task type and XGBoost objective (`binary:logistic` / `multi:softprob` / `reg:squarederror`), set eval_metric (`logloss` / `mlogloss` / `rmse`).

2. **Nested objective function** — define `def objective(trial):` inside the main function. It should:
   - Suggest hyperparameters from this search space:

     | Hyperparameter | Range | Scale |
     |---------------|-------|-------|
     | `max_depth` | 3–10 | int |
     | `learning_rate` | 0.01–0.3 | log |
     | `n_estimators` | 100–1500 | int (step 100) |
     | `subsample` | 0.5–1.0 | float |
     | `colsample_bytree` | 0.5–1.0 | float |
     | `reg_alpha` | 1e-8–10 | log |
     | `reg_lambda` | 1e-8–10 | log |
     | `min_child_weight` | 1–10 | int |
     | `gamma` | 0–5 | float |

   - Run 3-fold cross-validation with `cross_val_score`.
   - Scoring: `neg_log_loss` for classification, `neg_root_mean_squared_error` for regression.
   - Return `scores.mean()`.

3. **Run Optuna study** — `TPESampler` with fixed seed, direction `maximize`, `n_tuning_trials` trials with progress bar. Print best score and params.

4. **Train final model** — instantiate `XGBClassifier` or `XGBRegressor` with best params + early stopping. Fit on full training set with eval on test set.

5. **Evaluate** — compute metrics:
   - Classification: accuracy, F1 (weighted), ROC-AUC (binary or OVR for multiclass)
   - Regression: RMSE, MAE, R²

6. **Log to MLflow** — set experiment, start run, log params + metrics + model artifact. Optionally register to Unity Catalog.

7. **Optional plots** (still inside the function):
   - Calibration plot (binary classification only): `CalibrationDisplay.from_estimator()`, log as artifact.
   - SHAP plot: `shap.Explainer(model)` → `shap.summary_plot()`, log as artifact.

8. **Return** a dict with keys: `model`, `best_params`, `metrics`, `run_id`.

After defining the function, **call it** in the same cell with the user's data and print the results.

## Key Implementation Rules

1. **All code goes directly in notebook cells** — never import from a separate module.
2. **Tuning + training live in one function** with the objective as a nested closure.
3. **Each cell should be runnable independently** once prior cells have been run.
4. **Use descriptive cell titles** that explain what each cell does.
5. **Print progress** at each stage so the user sees what's happening.
6. **Always set `verbosity=0` and `n_jobs=-1`** on XGBoost models.
7. **Always suppress warnings** with `warnings.filterwarnings("ignore", category=UserWarning)`.
8. **Adapt feature selection** to the user's domain — use only the columns that make sense for their modeling question.
9. **Default experiment path**: `/Users/{user_email}/{descriptive-name}-experiment`
10. **Study direction**: Always `maximize` (since sklearn scoring returns negative values for loss metrics).
