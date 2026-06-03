# UFC Fight Predictor

[![Tests](https://github.com/ArjunAdelaide/UFC-Fight-Predictor/actions/workflows/tests.yml/badge.svg)](https://github.com/ArjunAdelaide/UFC-Fight-Predictor/actions/workflows/tests.yml)

A Python machine learning project for predicting UFC fight outcomes from
historical fighter data.

The project predicts a matchup in the form:

```text
fighter_a vs fighter_b -> probability that fighter_a wins
```

The current version is intentionally focused on a reliable modeling pipeline:
chronological feature engineering, leakage-safe historical fighter profiles, a
date-based train/test split, and a transparent NumPy logistic regression
baseline.

## Highlights

- Builds pre-fight features from historical data only
- Processes fights chronologically to avoid data leakage
- Handles the raw dataset's winner/loser column structure safely
- Randomly orients fights into neutral `fighter_a` / `fighter_b` matchups
- Trains a logistic regression baseline implemented with NumPy
- Evaluates with a future-dated train/test split
- Provides a CLI for predicting future matchups

## Results

Current baseline performance on a date-based test split:

| Model / Rule | Accuracy | Log Loss |
| --- | ---: | ---: |
| NumPy logistic regression | 62.1% | 0.650 |
| Majority class baseline | 50.2% | - |
| Higher Elo rule | 55.4% | - |
| Higher win percentage rule | 60.5% | - |
| Better recent form rule | 57.9% | - |

Split:

```text
train: fights before 2024-01-01
test:  fights on or after 2024-01-01
```

This split is designed to approximate the real use case: training on past
fights and evaluating on future fights.

## Methodology

The raw CSV stores each fight with:

```text
fighter_1 = winner
fighter_2 = loser
```

Training directly on those columns would create a trivial and useless target,
because `fighter_1` always wins. To avoid that, each fight is converted into a
neutral matchup:

```text
fighter_a vs fighter_b -> fighter_a_wins
```

For each historical fight, the row is randomly oriented:

- `fighter_a = winner`, `fighter_b = loser`, `fighter_a_wins = 1`
- or `fighter_a = loser`, `fighter_b = winner`, `fighter_a_wins = 0`

The feature pipeline also avoids using post-fight information from the fight
being predicted. Stats such as method, round, finish time, knockdowns,
significant strikes, takedowns, and control time are only used after the
training row has been created, when updating each fighter's history for later
fights.

At a high level:

```text
sort fights by date
for each fight:
  build pre-fight profiles for both fighters
  create fighter_a vs fighter_b features
  assign fighter_a_wins target
  update fighter histories with the completed fight
```

For a deeper explanation of the modeling choices, see
[`docs/methodology.md`](docs/methodology.md). For shorter beginner notes, see
[`LEARNING_NOTES.md`](LEARNING_NOTES.md).

## Features

The model mostly sees matchup differences:

```text
fighter_a value - fighter_b value
```

Examples include:

- age, height, reach, weight, and reach-to-height differences
- total fights, wins, losses, and win percentage differences
- finish, KO/TKO, submission, and decision win-rate differences
- striking, takedown, knockdown, and control-time historical rates
- current streak, longest win streak, and recent form
- days since last fight
- simple Elo difference

Categorical inputs include:

- weight class
- Fighter A stance
- Fighter B stance
- stance matchup

## Project Structure

```text
.
|-- LEARNING_NOTES.md
|-- README.md
|-- requirements.txt
|-- .github
|   `-- workflows
|       `-- tests.yml
|-- data
|   `-- .gitkeep
|-- docs
|   `-- methodology.md
|-- src
|   |-- feature_engineering.py
|   |-- predict_matchup.py
|   `-- train_baseline.py
|-- tests
|   `-- test_feature_engineering.py
`-- outputs
    |-- baseline_coefficients.csv
    |-- calibration_buckets.csv
    |-- confidence_buckets.csv
    |-- baseline_logistic_model.npz
    |-- current_fighter_profiles.pkl
    |-- baseline_metrics.csv
    |-- model_report.md
    `-- prefight_features.csv
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

The current code expects a cleaned UFC dataset CSV. The recommended public
workflow is to supply the local dataset path with `--data`:

```bash
python3 src/train_baseline.py --data data/clean_ufc_dataset.csv
```

You can also set an environment variable:

```bash
UFC_DATASET_PATH=data/clean_ufc_dataset.csv python3 src/train_baseline.py
```

If no path is supplied, the fallback path is:

```python
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "clean_ufc_dataset.csv"
```

The dataset itself is intentionally ignored by Git. Place a local copy in
`data/` or pass the path to wherever it lives on your machine.

## Training

Run from the project root:

```bash
python3 src/train_baseline.py --data data/clean_ufc_dataset.csv
```

Training creates or updates:

```text
outputs/prefight_features.csv
outputs/baseline_metrics.csv
outputs/baseline_coefficients.csv
outputs/calibration_buckets.csv
outputs/confidence_buckets.csv
outputs/baseline_logistic_model.npz
outputs/current_fighter_profiles.pkl
outputs/model_report.md
```

Output files:

| File | Purpose |
| --- | --- |
| `prefight_features.csv` | Leakage-safe model dataset |
| `baseline_metrics.csv` | Accuracy, log loss, Brier score, and baseline comparisons |
| `baseline_coefficients.csv` | Learned logistic regression coefficients |
| `calibration_buckets.csv` | Predicted probability buckets vs. actual Fighter A win rate |
| `confidence_buckets.csv` | Accuracy grouped by model confidence |
| `baseline_logistic_model.npz` | Saved model used by the prediction CLI |
| `current_fighter_profiles.pkl` | Cached latest fighter histories for faster prediction |
| `model_report.md` | Markdown summary of the latest training run |

The generated [model report](outputs/model_report.md) summarizes the latest
dataset split, metrics, calibration buckets, confidence buckets, and largest
coefficients.

## Prediction

Train the model first. This creates both the saved model and a cache of current
fighter profiles.

Then run:

```bash
python3 src/predict_matchup.py "Islam Makhachev" "Ilia Topuria" --date 2026-06-03 --weight-class Lightweight
```

By default, prediction loads `outputs/current_fighter_profiles.pkl` instead of
rebuilding fighter histories from the CSV. To force a rebuild from the dataset,
use:

```bash
python3 src/predict_matchup.py "Islam Makhachev" "Ilia Topuria" --date 2026-06-03 --weight-class Lightweight --data data/clean_ufc_dataset.csv --rebuild-profiles
```

Example output:

```text
Prediction
  Islam Makhachev: 68.1%
  Ilia Topuria: 31.9%
  fight date: 2026-06-03
  weight class: Lightweight
  profiles: loaded from outputs/current_fighter_profiles.pkl

Quick comparison
  Elo: Islam Makhachev 1740 vs Ilia Topuria 1656
  Record in dataset: Islam Makhachev 17-1 vs Ilia Topuria 9-0
  Recent win pct: Islam Makhachev 1.00 vs Ilia Topuria 1.00
```

The first fighter passed to the CLI is `fighter_a`, so the reported probability
answers:

```text
What is the probability that fighter_a beats fighter_b?
```

## Tests

GitHub Actions runs the test suite automatically on pushes and pull requests to
`main`.

Run the test suite with:

```bash
python3 -m unittest discover -s tests
```

The current tests focus on the most important ML safety checks:

- `fighter_a_wins` matches the randomized Fighter A / Fighter B orientation
- current-fight post-fight stats do not leak into that fight's pre-fight row

## Limitations

This is a baseline model and does not currently include:

- betting odds
- UFC rankings
- injury context
- short-notice fight information
- weight misses
- fight camp or team data
- title fight or main event flags
- opponent-quality adjusted statistics
- advanced Elo variants

Cached fighter profiles make predictions faster after training, but the cache
must be regenerated when the underlying dataset changes.

## Roadmap

- Plot calibration curves and confidence bucket charts
- Reduce overlapping features for clearer coefficient interpretation
- Add rolling last-3 and last-5 fight features
- Improve Elo with recency, method, finish, and opponent-quality adjustments
- Compare against scikit-learn, tree-based models, XGBoost, or LightGBM
- Add external data such as odds, rankings, injuries, and weight misses

## Disclaimer

This project is for machine learning experimentation and sports analytics
practice. It is not betting advice. Combat sports are noisy, high-variance, and
affected by many factors that are not included in the current dataset.
