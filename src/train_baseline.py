from __future__ import annotations

from pathlib import Path

from ufc_predictor.config import DATASET_PATH, DEFAULT_PATHS, PROJECT_ROOT, SPLIT_DATE
from ufc_predictor.evaluation import (
    accuracy,
    brier_score,
    bucket_label,
    confidence_buckets,
    evaluate_rule,
    log_loss,
    probability_calibration_buckets,
    summarize_metrics,
)
from ufc_predictor.features import build_current_fighter_data, build_prefight_dataset
from ufc_predictor.model import sigmoid, train_logistic_regression
from ufc_predictor.preprocessing import (
    ALWAYS_KEEP_COLUMNS,
    CATEGORICAL_COLUMNS,
    DROP_COLUMNS,
    augment_training_matchups,
    flipped_matchup_rows,
    model_feature_columns,
    prepare_design_matrix,
)
from ufc_predictor.reporting import markdown_table, write_model_report
from ufc_predictor.training import main, parse_args, run_baseline_training


OUTPUT_DIR = DEFAULT_PATHS.output_dir
FEATURE_OUTPUT = DEFAULT_PATHS.feature_output
MODEL_OUTPUT = DEFAULT_PATHS.model_output
PROFILE_CACHE_OUTPUT = DEFAULT_PATHS.profile_cache_output
METRICS_OUTPUT = DEFAULT_PATHS.metrics_output
COEFFICIENT_OUTPUT = DEFAULT_PATHS.coefficient_output
CALIBRATION_OUTPUT = DEFAULT_PATHS.calibration_output
CONFIDENCE_OUTPUT = DEFAULT_PATHS.confidence_output
MODEL_REPORT_OUTPUT = DEFAULT_PATHS.model_report_output


def save_profile_cache(dataset_path: Path) -> int:
    from ufc_predictor.training import save_current_profile_cache

    return save_current_profile_cache(dataset_path, DEFAULT_PATHS)


__all__ = [
    "ALWAYS_KEEP_COLUMNS",
    "CALIBRATION_OUTPUT",
    "CATEGORICAL_COLUMNS",
    "COEFFICIENT_OUTPUT",
    "CONFIDENCE_OUTPUT",
    "DATASET_PATH",
    "DROP_COLUMNS",
    "FEATURE_OUTPUT",
    "METRICS_OUTPUT",
    "MODEL_OUTPUT",
    "MODEL_REPORT_OUTPUT",
    "OUTPUT_DIR",
    "PROFILE_CACHE_OUTPUT",
    "PROJECT_ROOT",
    "SPLIT_DATE",
    "accuracy",
    "augment_training_matchups",
    "brier_score",
    "bucket_label",
    "build_current_fighter_data",
    "build_prefight_dataset",
    "confidence_buckets",
    "evaluate_rule",
    "flipped_matchup_rows",
    "log_loss",
    "main",
    "markdown_table",
    "model_feature_columns",
    "parse_args",
    "prepare_design_matrix",
    "probability_calibration_buckets",
    "run_baseline_training",
    "save_profile_cache",
    "sigmoid",
    "summarize_metrics",
    "train_logistic_regression",
    "write_model_report",
]


if __name__ == "__main__":
    main()
