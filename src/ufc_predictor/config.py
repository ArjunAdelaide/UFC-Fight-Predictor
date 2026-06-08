from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path = PROJECT_ROOT
    dataset_path: Path = Path(
        os.environ.get("UFC_DATASET_PATH", PROJECT_ROOT / "data" / "clean_ufc_dataset.csv")
    ).expanduser()
    output_dir: Path = PROJECT_ROOT / "outputs"

    @property
    def feature_output(self) -> Path:
        return self.output_dir / "prefight_features.csv"

    @property
    def model_output(self) -> Path:
        return self.output_dir / "baseline_logistic_model.npz"

    @property
    def profile_cache_output(self) -> Path:
        return self.output_dir / "current_fighter_profiles.pkl"

    @property
    def metrics_output(self) -> Path:
        return self.output_dir / "baseline_metrics.csv"

    @property
    def coefficient_output(self) -> Path:
        return self.output_dir / "baseline_coefficients.csv"

    @property
    def calibration_output(self) -> Path:
        return self.output_dir / "calibration_buckets.csv"

    @property
    def confidence_output(self) -> Path:
        return self.output_dir / "confidence_buckets.csv"

    @property
    def model_report_output(self) -> Path:
        return self.output_dir / "model_report.md"

    @property
    def model_comparison_output(self) -> Path:
        return self.output_dir / "sklearn_model_comparison.csv"

    @property
    def chart_dir(self) -> Path:
        return self.output_dir / "charts"


@dataclass(frozen=True)
class LogisticRegressionConfig:
    learning_rate: float = 0.05
    epochs: int = 4_000
    l2: float = 0.01


@dataclass(frozen=True)
class TrainingConfig:
    split_date: pd.Timestamp = pd.Timestamp("2024-01-01")
    random_seed: int = 42
    include_dirty_methods: bool = False
    logistic_regression: LogisticRegressionConfig = LogisticRegressionConfig()


DEFAULT_PATHS = ProjectPaths()
DEFAULT_DATASET_PATH = DEFAULT_PATHS.dataset_path
DATASET_PATH = DEFAULT_DATASET_PATH
OUTPUT_DIR = DEFAULT_PATHS.output_dir
SPLIT_DATE = TrainingConfig().split_date
