from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ufc_predictor.artifacts import ModelArtifact, save_model_artifact, save_profile_cache
from ufc_predictor.config import DATASET_PATH, DEFAULT_PATHS, ProjectPaths, TrainingConfig
from ufc_predictor.evaluation import (
    accuracy,
    brier_score,
    confidence_buckets,
    evaluate_probabilities,
    evaluate_rule,
    log_loss,
    probability_calibration_buckets,
    summarize_metrics,
)
from ufc_predictor.features import build_current_fighter_data, build_prefight_dataset
from ufc_predictor.model import sigmoid, train_logistic_regression
from ufc_predictor.preprocessing import augment_training_matchups, prepare_design_matrix
from ufc_predictor.reporting import write_model_report


@dataclass(frozen=True)
class TrainingRun:
    features: pd.DataFrame
    train: pd.DataFrame
    train_model: pd.DataFrame
    test: pd.DataFrame
    metrics: pd.DataFrame
    coefficients: pd.DataFrame
    calibration: pd.DataFrame
    confidence: pd.DataFrame
    train_probabilities: np.ndarray
    test_probabilities: np.ndarray
    feature_names: list[str]
    weights: np.ndarray
    cached_fighters: int


def split_features(
    features: pd.DataFrame,
    split_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = features[features["event_date"] < split_date].copy()
    test = features[features["event_date"] >= split_date].copy()
    return train, test


def save_current_profile_cache(dataset_path: Path, paths: ProjectPaths) -> int:
    states, bios = build_current_fighter_data(dataset_path)
    save_profile_cache(
        paths.profile_cache_output,
        dataset_path=dataset_path,
        states=states,
        bios=bios,
    )
    return len(states)


def run_baseline_training(
    *,
    dataset_path: Path,
    paths: ProjectPaths = DEFAULT_PATHS,
    config: TrainingConfig = TrainingConfig(),
) -> TrainingRun:
    paths.output_dir.mkdir(parents=True, exist_ok=True)

    features = build_prefight_dataset(
        dataset_path,
        random_seed=config.random_seed,
        include_dirty_methods=config.include_dirty_methods,
    )
    features.to_csv(paths.feature_output, index=False)

    train, test = split_features(features, config.split_date)
    train_model = augment_training_matchups(train)

    y_train_original = train["fighter_a_wins"].to_numpy(dtype=float)
    y_train = train_model["fighter_a_wins"].to_numpy(dtype=float)
    y_test = test["fighter_a_wins"].to_numpy(dtype=float)

    x_train, x_test, feature_names, means, stds = prepare_design_matrix(train_model, test)
    lr_config = config.logistic_regression
    weights = train_logistic_regression(
        x_train,
        y_train,
        learning_rate=lr_config.learning_rate,
        epochs=lr_config.epochs,
        l2=lr_config.l2,
    )
    test_prob = sigmoid(x_test @ weights)
    train_prob = sigmoid(x_train @ weights)

    majority_prob = np.repeat(y_train_original.mean(), len(y_test))
    metrics = [
        evaluate_probabilities("logistic_regression_numpy", y_test, test_prob),
        evaluate_probabilities("majority_class", y_test, majority_prob),
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

    metrics_frame.to_csv(paths.metrics_output, index=False)
    coef_frame.sort_values("abs_coefficient", ascending=False).to_csv(paths.coefficient_output, index=False)
    calibration_frame = probability_calibration_buckets(y_test, test_prob)
    confidence_frame = confidence_buckets(y_test, test_prob)
    calibration_frame.to_csv(paths.calibration_output, index=False)
    confidence_frame.to_csv(paths.confidence_output, index=False)

    save_model_artifact(
        paths.model_output,
        ModelArtifact(
            weights=weights,
            feature_names=feature_names,
            means=means,
            stds=stds,
            metadata={
                "train_start": str(train["event_date"].min().date()),
                "train_end": str(train["event_date"].max().date()),
                "test_start": str(test["event_date"].min().date()),
                "test_end": str(test["event_date"].max().date()),
                "split_date": str(config.split_date.date()),
                "random_seed": config.random_seed,
                "learning_rate": lr_config.learning_rate,
                "epochs": lr_config.epochs,
                "l2": lr_config.l2,
            },
        ),
    )

    cached_fighters = save_current_profile_cache(dataset_path, paths)
    write_model_report(
        path=paths.model_report_output,
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

    return TrainingRun(
        features=features,
        train=train,
        train_model=train_model,
        test=test,
        metrics=metrics_frame,
        coefficients=coef_frame,
        calibration=calibration_frame,
        confidence=confidence_frame,
        train_probabilities=train_prob,
        test_probabilities=test_prob,
        feature_names=feature_names,
        weights=weights,
        cached_fighters=cached_fighters,
    )


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


def print_training_summary(run: TrainingRun, *, dataset_path: Path, paths: ProjectPaths) -> None:
    y_train_original = run.train["fighter_a_wins"].to_numpy(dtype=float)
    y_train = run.train_model["fighter_a_wins"].to_numpy(dtype=float)
    y_test = run.test["fighter_a_wins"].to_numpy(dtype=float)
    top_positive = run.coefficients.sort_values("coefficient", ascending=False).head(12)
    top_negative = run.coefficients.sort_values("coefficient", ascending=True).head(12)

    print("\nDataset")
    print(f"  source: {dataset_path}")
    print(f"  feature rows: {len(run.features):,}")
    print(f"  feature columns: {run.features.shape[1]:,}")
    print(f"  train rows: {len(run.train):,} ({run.train['event_date'].min().date()} to {run.train['event_date'].max().date()})")
    print(f"  training examples after flipped augmentation: {len(run.train_model):,}")
    print(f"  test rows: {len(run.test):,} ({run.test['event_date'].min().date()} to {run.test['event_date'].max().date()})")
    print(f"  train target mean: {y_train_original.mean():.3f}")
    print(f"  augmented training target mean: {y_train.mean():.3f}")
    print(f"  test target mean: {y_test.mean():.3f}")

    print("\nEvaluation on future-dated test split")
    print(run.metrics.to_string(index=False, formatters={
        "accuracy": "{:.3f}".format,
        "log_loss": "{:.3f}".format,
        "brier": "{:.3f}".format,
    }))

    print("\nCalibration buckets")
    print(run.calibration.to_string(index=False, formatters={
        "avg_predicted_probability": "{:.3f}".format,
        "actual_fighter_a_win_rate": "{:.3f}".format,
    }))

    print("\nConfidence buckets")
    print(run.confidence.to_string(index=False, formatters={
        "avg_confidence": "{:.3f}".format,
        "accuracy": "{:.3f}".format,
    }))

    print("\nTraining sanity check")
    print(f"  train accuracy: {accuracy(y_train, run.train_probabilities):.3f}")
    print(f"  train log loss: {log_loss(y_train, run.train_probabilities):.3f}")

    print("\nFeatures pushing prediction toward Fighter A")
    print(top_positive[["feature", "coefficient"]].to_string(index=False, formatters={
        "coefficient": "{:.3f}".format,
    }))

    print("\nFeatures pushing prediction toward Fighter B")
    print(top_negative[["feature", "coefficient"]].to_string(index=False, formatters={
        "coefficient": "{:.3f}".format,
    }))

    print("\nSaved")
    print(f"  features: {paths.feature_output}")
    print(f"  model: {paths.model_output}")
    print(f"  metrics: {paths.metrics_output}")
    print(f"  coefficients: {paths.coefficient_output}")
    print(f"  calibration: {paths.calibration_output}")
    print(f"  confidence: {paths.confidence_output}")
    print(f"  profile cache: {paths.profile_cache_output} ({run.cached_fighters:,} fighters)")
    print(f"  model report: {paths.model_report_output}")


def main() -> None:
    args = parse_args()
    dataset_path = args.data.expanduser()
    run = run_baseline_training(dataset_path=dataset_path)
    print_training_summary(run, dataset_path=dataset_path, paths=DEFAULT_PATHS)


if __name__ == "__main__":
    main()
