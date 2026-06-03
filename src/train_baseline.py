from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering import DATASET_PATH, build_current_fighter_data, build_prefight_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FEATURE_OUTPUT = OUTPUT_DIR / "prefight_features.csv"
MODEL_OUTPUT = OUTPUT_DIR / "baseline_logistic_model.npz"
PROFILE_CACHE_OUTPUT = OUTPUT_DIR / "current_fighter_profiles.pkl"
METRICS_OUTPUT = OUTPUT_DIR / "baseline_metrics.csv"
COEFFICIENT_OUTPUT = OUTPUT_DIR / "baseline_coefficients.csv"
CALIBRATION_OUTPUT = OUTPUT_DIR / "calibration_buckets.csv"
CONFIDENCE_OUTPUT = OUTPUT_DIR / "confidence_buckets.csv"
MODEL_REPORT_OUTPUT = OUTPUT_DIR / "model_report.md"
SPLIT_DATE = pd.Timestamp("2024-01-01")


DROP_COLUMNS = {
    "event_date",
    "event_name",
    "source_row",
    "fighter_a",
    "fighter_b",
    "winner",
    "method",
    "method_bucket",
    "fighter_a_wins",
}

CATEGORICAL_COLUMNS = ["weight_class", "a_stance", "b_stance", "stance_matchup"]

ALWAYS_KEEP_COLUMNS = {
    "weight_class",
    "a_stance",
    "b_stance",
    "stance_matchup",
    "a_is_debut",
    "b_is_debut",
}


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in DROP_COLUMNS and (col.endswith("_diff") or col in ALWAYS_KEEP_COLUMNS)
    ]


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-values))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob >= 0.5).astype(int) == y_true))


def train_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    learning_rate: float = 0.05,
    epochs: int = 4_000,
    l2: float = 0.01,
) -> np.ndarray:
    weights = np.zeros(x_train.shape[1], dtype=float)
    n_rows = float(len(y_train))

    for _ in range(epochs):
        predictions = sigmoid(x_train @ weights)
        gradient = (x_train.T @ (predictions - y_train)) / n_rows
        gradient[1:] += l2 * weights[1:] / n_rows
        weights -= learning_rate * gradient

    return weights


def prepare_design_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.Series, pd.Series]:
    feature_columns = model_feature_columns(train)

    train_x = train[feature_columns].copy()
    test_x = test[feature_columns].copy()

    for col in CATEGORICAL_COLUMNS:
        train_x[col] = train_x[col].fillna("Unknown").astype(str)
        test_x[col] = test_x[col].fillna("Unknown").astype(str)

    numeric_columns = [col for col in feature_columns if col not in CATEGORICAL_COLUMNS]
    medians = train_x[numeric_columns].median(numeric_only=True).fillna(0.0)
    train_x[numeric_columns] = train_x[numeric_columns].fillna(medians)
    test_x[numeric_columns] = test_x[numeric_columns].fillna(medians)

    train_x = pd.get_dummies(train_x, columns=CATEGORICAL_COLUMNS, drop_first=False)
    test_x = pd.get_dummies(test_x, columns=CATEGORICAL_COLUMNS, drop_first=False)
    train_x, test_x = train_x.align(test_x, join="left", axis=1, fill_value=0)

    means = train_x.mean()
    stds = train_x.std(ddof=0).replace(0, 1.0)
    train_scaled = (train_x - means) / stds
    test_scaled = (test_x - means) / stds

    train_matrix = np.column_stack([np.ones(len(train_scaled)), train_scaled.to_numpy(dtype=float)])
    test_matrix = np.column_stack([np.ones(len(test_scaled)), test_scaled.to_numpy(dtype=float)])
    feature_names = ["intercept", *train_scaled.columns.tolist()]

    return train_matrix, test_matrix, feature_names, means, stds


def flipped_matchup_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Create the opposite Fighter A/B orientation for training rows."""
    flipped = frame.copy()

    for column in frame.columns:
        if column.startswith("a_"):
            opposite = "b_" + column[2:]
            if opposite in frame.columns:
                flipped[column] = frame[opposite]
        elif column.startswith("b_"):
            opposite = "a_" + column[2:]
            if opposite in frame.columns:
                flipped[column] = frame[opposite]

    for column in frame.columns:
        if column.endswith("_diff"):
            flipped[column] = -frame[column]

    flipped["fighter_a"] = frame["fighter_b"]
    flipped["fighter_b"] = frame["fighter_a"]
    flipped["fighter_a_wins"] = 1 - frame["fighter_a_wins"]

    if {"a_stance", "b_stance", "stance_matchup"}.issubset(flipped.columns):
        flipped["stance_matchup"] = (
            flipped["a_stance"].fillna("Unknown").astype(str)
            + "_vs_"
            + flipped["b_stance"].fillna("Unknown").astype(str)
        )

    return flipped


def augment_training_matchups(train: pd.DataFrame) -> pd.DataFrame:
    flipped = flipped_matchup_rows(train)
    return pd.concat([train, flipped], ignore_index=True)


def evaluate_rule(name: str, y_true: np.ndarray, score: pd.Series) -> dict[str, float | str]:
    probabilities = np.where(score > 0, 0.60, np.where(score < 0, 0.40, 0.50))
    return {
        "model": name,
        "accuracy": accuracy(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities),
        "brier": brier_score(y_true, probabilities),
    }


def summarize_metrics(metrics: list[dict[str, float | str]]) -> pd.DataFrame:
    frame = pd.DataFrame(metrics)
    return frame.sort_values("log_loss", ascending=True).reset_index(drop=True)


def bucket_label(start: float, end: float) -> str:
    return f"{int(round(start * 100))}-{int(round(end * 100))}%"


def probability_calibration_buckets(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    bucket_width: float = 0.10,
) -> pd.DataFrame:
    rows = []
    for start in np.arange(0.0, 1.0, bucket_width):
        end = min(start + bucket_width, 1.0)
        if end >= 1.0:
            mask = (y_prob >= start) & (y_prob <= end)
        else:
            mask = (y_prob >= start) & (y_prob < end)

        rows.append(
            {
                "probability_bucket": bucket_label(start, end),
                "count": int(mask.sum()),
                "avg_predicted_probability": float(np.mean(y_prob[mask])) if mask.any() else np.nan,
                "actual_fighter_a_win_rate": float(np.mean(y_true[mask])) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def confidence_buckets(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    bucket_width: float = 0.10,
) -> pd.DataFrame:
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    predicted = (y_prob >= 0.5).astype(int)
    correct = predicted == y_true

    rows = []
    for start in np.arange(0.5, 1.0, bucket_width):
        end = min(start + bucket_width, 1.0)
        if end >= 1.0:
            mask = (confidence >= start) & (confidence <= end)
        else:
            mask = (confidence >= start) & (confidence < end)

        rows.append(
            {
                "confidence_bucket": bucket_label(start, end),
                "count": int(mask.sum()),
                "avg_confidence": float(np.mean(confidence[mask])) if mask.any() else np.nan,
                "accuracy": float(np.mean(correct[mask])) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_profile_cache(dataset_path: Path) -> int:
    states, bios = build_current_fighter_data(dataset_path)
    payload = {
        "dataset_path": str(dataset_path),
        "states": states,
        "bios": bios,
    }
    with PROFILE_CACHE_OUTPUT.open("wb") as cache_file:
        pickle.dump(payload, cache_file)
    return len(states)


def markdown_table(
    rows: list[dict[str, object]],
    columns: list[str],
) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def write_model_report(
    *,
    dataset_path: Path,
    features: pd.DataFrame,
    train: pd.DataFrame,
    train_examples: int,
    test: pd.DataFrame,
    metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    confidence: pd.DataFrame,
    top_positive: pd.DataFrame,
    top_negative: pd.DataFrame,
    cached_fighters: int,
) -> None:
    metric_rows = [
        {
            "Model / Rule": row["model"],
            "Accuracy": f"{row['accuracy']:.1%}",
            "Log Loss": f"{row['log_loss']:.3f}",
            "Brier": f"{row['brier']:.3f}",
        }
        for _, row in metrics.iterrows()
    ]
    calibration_rows = [
        {
            "Probability Bucket": row["probability_bucket"],
            "Fights": int(row["count"]),
            "Avg Predicted": (
                "" if pd.isna(row["avg_predicted_probability"]) else f"{row['avg_predicted_probability']:.1%}"
            ),
            "Actual Win Rate": (
                "" if pd.isna(row["actual_fighter_a_win_rate"]) else f"{row['actual_fighter_a_win_rate']:.1%}"
            ),
        }
        for _, row in calibration.iterrows()
    ]
    confidence_rows = [
        {
            "Confidence Bucket": row["confidence_bucket"],
            "Fights": int(row["count"]),
            "Avg Confidence": "" if pd.isna(row["avg_confidence"]) else f"{row['avg_confidence']:.1%}",
            "Accuracy": "" if pd.isna(row["accuracy"]) else f"{row['accuracy']:.1%}",
        }
        for _, row in confidence.iterrows()
    ]
    positive_rows = [
        {"Feature": row["feature"], "Coefficient": f"{row['coefficient']:.3f}"}
        for _, row in top_positive.head(8).iterrows()
    ]
    negative_rows = [
        {"Feature": row["feature"], "Coefficient": f"{row['coefficient']:.3f}"}
        for _, row in top_negative.head(8).iterrows()
    ]

    report = f"""# Model Report

This report is generated by `python3 src/train_baseline.py`.

## Dataset

| Item | Value |
| --- | --- |
| Dataset file | `{dataset_path.name}` |
| Feature rows | {len(features):,} |
| Feature columns | {features.shape[1]:,} |
| Train rows | {len(train):,} |
| Training examples | {train_examples:,} |
| Test rows | {len(test):,} |
| Train period | {train["event_date"].min().date()} to {train["event_date"].max().date()} |
| Test period | {test["event_date"].min().date()} to {test["event_date"].max().date()} |
| Cached fighter profiles | {cached_fighters:,} |

## Evaluation

{markdown_table(metric_rows, ["Model / Rule", "Accuracy", "Log Loss", "Brier"])}

## Calibration

{markdown_table(calibration_rows, ["Probability Bucket", "Fights", "Avg Predicted", "Actual Win Rate"])}

## Confidence

{markdown_table(confidence_rows, ["Confidence Bucket", "Fights", "Avg Confidence", "Accuracy"])}

## Largest Positive Coefficients

These features pushed predictions toward Fighter A in the current baseline.

{markdown_table(positive_rows, ["Feature", "Coefficient"])}

## Largest Negative Coefficients

These features pushed predictions toward Fighter B in the current baseline.

{markdown_table(negative_rows, ["Feature", "Coefficient"])}

## Notes

This is a baseline model. Coefficients are useful for debugging, but they should
not be treated as causal explanations because many features overlap.
"""
    MODEL_REPORT_OUTPUT.write_text(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the baseline UFC fight prediction model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATASET_PATH,
        help="Path to the cleaned UFC dataset CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = args.data.expanduser()

    features = build_prefight_dataset(dataset_path)
    features.to_csv(FEATURE_OUTPUT, index=False)

    train = features[features["event_date"] < SPLIT_DATE].copy()
    test = features[features["event_date"] >= SPLIT_DATE].copy()
    train_model = augment_training_matchups(train)

    y_train_original = train["fighter_a_wins"].to_numpy(dtype=float)
    y_train = train_model["fighter_a_wins"].to_numpy(dtype=float)
    y_test = test["fighter_a_wins"].to_numpy(dtype=float)

    x_train, x_test, feature_names, means, stds = prepare_design_matrix(train_model, test)
    weights = train_logistic_regression(x_train, y_train)
    test_prob = sigmoid(x_test @ weights)
    train_prob = sigmoid(x_train @ weights)

    majority_prob = np.repeat(y_train_original.mean(), len(y_test))
    metrics = [
        {
            "model": "logistic_regression_numpy",
            "accuracy": accuracy(y_test, test_prob),
            "log_loss": log_loss(y_test, test_prob),
            "brier": brier_score(y_test, test_prob),
        },
        {
            "model": "majority_class",
            "accuracy": accuracy(y_test, majority_prob),
            "log_loss": log_loss(y_test, majority_prob),
            "brier": brier_score(y_test, majority_prob),
        },
        evaluate_rule("higher_elo_rule", y_test, test["elo_diff"]),
        evaluate_rule("higher_win_pct_rule", y_test, test["win_pct_diff"]),
        evaluate_rule("higher_experience_rule", y_test, test["fights_diff"]),
        evaluate_rule("better_recent_form_rule", y_test, test["recent_win_pct_diff"]),
    ]
    metrics_frame = summarize_metrics(metrics)

    coef_frame = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": weights,
            "abs_coefficient": np.abs(weights),
        }
    )
    coef_frame = coef_frame[coef_frame["feature"] != "intercept"]
    top_positive = coef_frame.sort_values("coefficient", ascending=False).head(12)
    top_negative = coef_frame.sort_values("coefficient", ascending=True).head(12)
    metrics_frame.to_csv(METRICS_OUTPUT, index=False)
    coef_frame.sort_values("abs_coefficient", ascending=False).to_csv(COEFFICIENT_OUTPUT, index=False)
    calibration_frame = probability_calibration_buckets(y_test, test_prob)
    confidence_frame = confidence_buckets(y_test, test_prob)
    calibration_frame.to_csv(CALIBRATION_OUTPUT, index=False)
    confidence_frame.to_csv(CONFIDENCE_OUTPUT, index=False)

    np.savez(
        MODEL_OUTPUT,
        weights=weights,
        feature_names=np.array(feature_names, dtype=object),
        means=means.to_numpy(dtype=float),
        stds=stds.to_numpy(dtype=float),
        train_start=str(train["event_date"].min().date()),
        train_end=str(train["event_date"].max().date()),
        test_start=str(test["event_date"].min().date()),
        test_end=str(test["event_date"].max().date()),
    )
    cached_fighters = save_profile_cache(dataset_path)
    write_model_report(
        dataset_path=dataset_path,
        features=features,
        train=train,
        train_examples=len(train_model),
        test=test,
        metrics=metrics_frame,
        calibration=calibration_frame,
        confidence=confidence_frame,
        top_positive=top_positive,
        top_negative=top_negative,
        cached_fighters=cached_fighters,
    )

    print("\nDataset")
    print(f"  source: {dataset_path}")
    print(f"  feature rows: {len(features):,}")
    print(f"  feature columns: {features.shape[1]:,}")
    print(f"  train rows: {len(train):,} ({train['event_date'].min().date()} to {train['event_date'].max().date()})")
    print(f"  training examples after flipped augmentation: {len(train_model):,}")
    print(f"  test rows: {len(test):,} ({test['event_date'].min().date()} to {test['event_date'].max().date()})")
    print(f"  train target mean: {y_train_original.mean():.3f}")
    print(f"  augmented training target mean: {y_train.mean():.3f}")
    print(f"  test target mean: {y_test.mean():.3f}")

    print("\nEvaluation on future-dated test split")
    print(metrics_frame.to_string(index=False, formatters={
        "accuracy": "{:.3f}".format,
        "log_loss": "{:.3f}".format,
        "brier": "{:.3f}".format,
    }))

    print("\nCalibration buckets")
    print(calibration_frame.to_string(index=False, formatters={
        "avg_predicted_probability": "{:.3f}".format,
        "actual_fighter_a_win_rate": "{:.3f}".format,
    }))

    print("\nConfidence buckets")
    print(confidence_frame.to_string(index=False, formatters={
        "avg_confidence": "{:.3f}".format,
        "accuracy": "{:.3f}".format,
    }))

    print("\nTraining sanity check")
    print(f"  train accuracy: {accuracy(y_train, train_prob):.3f}")
    print(f"  train log loss: {log_loss(y_train, train_prob):.3f}")

    print("\nFeatures pushing prediction toward Fighter A")
    print(top_positive[["feature", "coefficient"]].to_string(index=False, formatters={
        "coefficient": "{:.3f}".format,
    }))

    print("\nFeatures pushing prediction toward Fighter B")
    print(top_negative[["feature", "coefficient"]].to_string(index=False, formatters={
        "coefficient": "{:.3f}".format,
    }))

    print("\nSaved")
    print(f"  features: {FEATURE_OUTPUT}")
    print(f"  model: {MODEL_OUTPUT}")
    print(f"  metrics: {METRICS_OUTPUT}")
    print(f"  coefficients: {COEFFICIENT_OUTPUT}")
    print(f"  calibration: {CALIBRATION_OUTPUT}")
    print(f"  confidence: {CONFIDENCE_OUTPUT}")
    print(f"  profile cache: {PROFILE_CACHE_OUTPUT} ({cached_fighters:,} fighters)")
    print(f"  model report: {MODEL_REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
