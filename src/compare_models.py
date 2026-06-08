from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

from feature_engineering import DATASET_PATH, build_prefight_dataset
from train_baseline import (
    OUTPUT_DIR,
    SPLIT_DATE,
    accuracy,
    augment_training_matchups,
    brier_score,
    log_loss,
    prepare_design_matrix,
    sigmoid,
    train_logistic_regression,
)


MODEL_COMPARISON_OUTPUT = OUTPUT_DIR / "sklearn_model_comparison.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare stronger scikit-learn models against the NumPy baseline."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATASET_PATH,
        help="Path to the cleaned UFC dataset CSV.",
    )
    parser.add_argument(
        "--use-existing-features",
        action="store_true",
        help="Use outputs/prefight_features.csv instead of rebuilding features.",
    )
    return parser.parse_args()


def model_specs() -> list[tuple[str, object]]:
    return [
        (
            "sklearn_logistic_regression_c_0_1",
            LogisticRegression(C=0.1, max_iter=2_000, solver="lbfgs"),
        ),
        (
            "sklearn_logistic_regression_c_1",
            LogisticRegression(C=1.0, max_iter=2_000, solver="lbfgs"),
        ),
        (
            "random_forest_depth_6",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=1,
            ),
        ),
        (
            "gradient_boosting_depth_3",
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.03,
                max_depth=3,
                min_samples_leaf=20,
                random_state=42,
            ),
        ),
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.03,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            ),
        ),
    ]


def evaluate_model(name: str, y_true, probabilities) -> dict[str, float | str]:
    return {
        "model": name,
        "accuracy": accuracy(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities),
        "brier": brier_score(y_true, probabilities),
    }


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_path = OUTPUT_DIR / "prefight_features.csv"
    if args.use_existing_features and feature_path.exists():
        features = pd.read_csv(feature_path, parse_dates=["event_date"])
    else:
        features = build_prefight_dataset(args.data.expanduser())

    train = features[features["event_date"] < SPLIT_DATE].copy()
    test = features[features["event_date"] >= SPLIT_DATE].copy()
    train_model = augment_training_matchups(train)

    y_train = train_model["fighter_a_wins"].to_numpy(dtype=int)
    y_test = test["fighter_a_wins"].to_numpy(dtype=int)
    x_train_with_intercept, x_test_with_intercept, *_ = prepare_design_matrix(train_model, test)
    x_train = x_train_with_intercept[:, 1:]
    x_test = x_test_with_intercept[:, 1:]

    rows = []
    weights = train_logistic_regression(x_train_with_intercept, y_train.astype(float))
    numpy_prob = sigmoid(x_test_with_intercept @ weights)
    rows.append(evaluate_model("logistic_regression_numpy", y_test, numpy_prob))

    for name, model in model_specs():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        rows.append(evaluate_model(name, y_test, probabilities))

    comparison = pd.DataFrame(rows).sort_values("log_loss", ascending=True)
    comparison.to_csv(MODEL_COMPARISON_OUTPUT, index=False)

    print("\nModel comparison on future-dated test split")
    print(
        comparison.to_string(
            index=False,
            formatters={
                "accuracy": "{:.3f}".format,
                "log_loss": "{:.3f}".format,
                "brier": "{:.3f}".format,
            },
        )
    )
    print(f"\nSaved: {MODEL_COMPARISON_OUTPUT}")


if __name__ == "__main__":
    main()
