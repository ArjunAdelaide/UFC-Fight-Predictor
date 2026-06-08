from __future__ import annotations

from ufc_predictor.config import DATASET_PATH, DEFAULT_PATHS
from ufc_predictor.prediction import (
    PredictionResult,
    display_path,
    is_dummy_column,
    load_fighter_profiles,
    load_model,
    main,
    parse_args,
    predict_matchup,
    prepare_single_matchup_matrix,
    print_prediction,
    resolve_fighter_name,
)


MODEL_OUTPUT = DEFAULT_PATHS.model_output
PROFILE_CACHE_OUTPUT = DEFAULT_PATHS.profile_cache_output

__all__ = [
    "DATASET_PATH",
    "MODEL_OUTPUT",
    "PROFILE_CACHE_OUTPUT",
    "PredictionResult",
    "display_path",
    "is_dummy_column",
    "load_fighter_profiles",
    "load_model",
    "main",
    "parse_args",
    "predict_matchup",
    "prepare_single_matchup_matrix",
    "print_prediction",
    "resolve_fighter_name",
]


if __name__ == "__main__":
    main()
