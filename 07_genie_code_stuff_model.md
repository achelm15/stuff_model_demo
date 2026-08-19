# 07 · Bootstrapping the Stuff Model with Genie Code

This step is a live demo, not a notebook you run cell by cell. It shows that
**Databricks Genie Code** (the in-workspace agent-mode coding assistant) can scaffold the
same governed training workflow as notebooks 02 and 03 from a single natural-language prompt,
because the prompt `@`-mentions a shared **workspace skill** that encodes our MLflow + Unity
Catalog conventions.

The point is not the model. Notebook 03 remains the governed, service-principal training
entry point. The point is how fast an agent gets you to a correct first draft that already
follows house style, so every teammate's agent produces the same shape of code.

The skill lives at `Workspace/.assistant/skills/rockies-mlflow-conventions/SKILL.md` (the
`rockies-mlflow-conventions/` folder in this repo) and is available to everyone in the
workspace.

## Try it

Open a new notebook, set the `catalog` and `schema` widgets (or run `_config` first), then open
Genie Code and give it this prompt:

> `@rockies-mlflow-conventions` Train an XGBoost "stuff" regressor that predicts `pitch_rv` for
> four-seam and sinker fastballs in `<catalog>.<schema>.silver_pitches` (the target you set on
> the `_config` widgets). Use the engineered feature set, tune with Optuna as nested MLflow
> runs, evaluate on an untouched holdout, log the model with a signature and input example, and
> register a **candidate** version to Unity Catalog without moving the `@champion` alias.

## What you should see happen

1. Genie Code loads the `rockies-mlflow-conventions` skill and follows it instead of guessing a
   generic training recipe.
2. It writes the cells in order: `%run ./_config`, widgets and names, setup, load
   `silver_pitches` with a 20% untouched holdout, a tuning function, and the Optuna plus
   registration cell. The code should look like notebook 03, not a generic tutorial.
3. Running it produces one parent MLflow run with a nested run per Optuna trial, each logging
   its validation RMSE.
4. It logs the final model with a signature and input example, writes a `feature_contract.json`
   artifact, and registers a new version of `<catalog>.<schema>.fastball_stuff_rv` to Unity
   Catalog.
5. The new version is tagged `promotion_status=candidate` and the `@champion` alias does not
   move. Held-out test metrics print at the end.

## How to verify

- The MLflow experiment shows the parent run with nested trial runs underneath it.
- The model in Unity Catalog has a new version tagged `promotion_status=candidate`, and
  `@champion` still points at whatever notebook 03 last promoted.

Review the generated recipe, and move anything worth keeping into notebook 03 for governed
promotion. Genie Code gets you a correct draft fast; promotion stays a deliberate, governed
step.
