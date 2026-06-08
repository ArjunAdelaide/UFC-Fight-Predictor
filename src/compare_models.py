from __future__ import annotations

from ufc_predictor.config import DEFAULT_PATHS
from ufc_predictor.model_comparison import compare_models, main, model_specs, parse_args


MODEL_COMPARISON_OUTPUT = DEFAULT_PATHS.model_comparison_output

__all__ = [
    "MODEL_COMPARISON_OUTPUT",
    "compare_models",
    "main",
    "model_specs",
    "parse_args",
]


if __name__ == "__main__":
    main()
