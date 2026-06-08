from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402

from ufc_predictor.config import DATASET_PATH, DEFAULT_PATHS, ProjectPaths, TrainingConfig
from ufc_predictor.evaluation import evaluate_probabilities
from ufc_predictor.features import build_prefight_dataset
from ufc_predictor.model import sigmoid, train_logistic_regression
from ufc_predictor.preprocessing import augment_training_matchups, prepare_design_matrix
from ufc_predictor.training import split_features


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


def compare_models(
    *,
    features: pd.DataFrame,
    config: TrainingConfig = TrainingConfig(),
) -> pd.DataFrame:
    train, test = split_features(features, config.split_date)
    train_model = augment_training_matchups(train)

    y_train = train_model["fighter_a_wins"].to_numpy(dtype=int)
    y_test = test["fighter_a_wins"].to_numpy(dtype=int)
    x_train_with_intercept, x_test_with_intercept, *_ = prepare_design_matrix(train_model, test)
    x_train = x_train_with_intercept[:, 1:]
    x_test = x_test_with_intercept[:, 1:]

    rows = []
    lr_config = config.logistic_regression
    weights = train_logistic_regression(
        x_train_with_intercept,
        y_train.astype(float),
        learning_rate=lr_config.learning_rate,
        epochs=lr_config.epochs,
        l2=lr_config.l2,
    )
    numpy_prob = sigmoid(x_test_with_intercept @ weights)
    rows.append(evaluate_probabilities("logistic_regression_numpy", y_test, numpy_prob))

    for name, model in model_specs():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, 1]
        rows.append(evaluate_probabilities(name, y_test, probabilities))

    return pd.DataFrame(rows).sort_values("log_loss", ascending=True)


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


def main(paths: ProjectPaths = DEFAULT_PATHS) -> None:
    args = parse_args()
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    if args.use_existing_features and paths.feature_output.exists():
        features = pd.read_csv(paths.feature_output, parse_dates=["event_date"])
    else:
        features = build_prefight_dataset(args.data.expanduser())

    comparison = compare_models(features=features)
    comparison.to_csv(paths.model_comparison_output, index=False)

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
    print(f"\nSaved: {paths.model_comparison_output}")


if __name__ == "__main__":
    main()
