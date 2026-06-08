from __future__ import annotations

from ufc_predictor.charts import (
    bar_chart,
    calibration_chart,
    confidence_chart,
    generate_charts,
    main,
    model_label,
    pct,
    write_svg,
)
from ufc_predictor.config import DEFAULT_PATHS


OUTPUT_DIR = DEFAULT_PATHS.output_dir
CHART_DIR = DEFAULT_PATHS.chart_dir
METRICS_PATH = DEFAULT_PATHS.metrics_output
CALIBRATION_PATH = DEFAULT_PATHS.calibration_output
CONFIDENCE_PATH = DEFAULT_PATHS.confidence_output

__all__ = [
    "CALIBRATION_PATH",
    "CHART_DIR",
    "CONFIDENCE_PATH",
    "METRICS_PATH",
    "OUTPUT_DIR",
    "bar_chart",
    "calibration_chart",
    "confidence_chart",
    "generate_charts",
    "main",
    "model_label",
    "pct",
    "write_svg",
]


if __name__ == "__main__":
    main()
