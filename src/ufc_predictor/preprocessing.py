from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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


@dataclass(frozen=True)
class DesignMatrix:
    values: np.ndarray
    feature_names: list[str]


@dataclass(frozen=True)
class FittedPreprocessor:
    feature_columns: list[str]
    encoded_columns: list[str]
    means: pd.Series
    stds: pd.Series
    medians: pd.Series

    def transform(self, frame: pd.DataFrame) -> DesignMatrix:
        frame_x = frame[self.feature_columns].copy()

        for col in CATEGORICAL_COLUMNS:
            frame_x[col] = frame_x[col].fillna("Unknown").astype(str)

        numeric_columns = [col for col in self.feature_columns if col not in CATEGORICAL_COLUMNS]
        frame_x[numeric_columns] = frame_x[numeric_columns].fillna(self.medians)
        frame_x = pd.get_dummies(frame_x, columns=CATEGORICAL_COLUMNS, drop_first=False)
        frame_x = frame_x.reindex(columns=self.encoded_columns, fill_value=0)
        scaled = (frame_x - self.means) / self.stds
        matrix = np.column_stack([np.ones(len(scaled)), scaled.to_numpy(dtype=float)])
        return DesignMatrix(matrix, ["intercept", *self.encoded_columns])


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        col
        for col in frame.columns
        if col not in DROP_COLUMNS and (col.endswith("_diff") or col in ALWAYS_KEEP_COLUMNS)
    ]


def fit_preprocessor(train: pd.DataFrame) -> FittedPreprocessor:
    feature_columns = model_feature_columns(train)
    train_x = train[feature_columns].copy()

    for col in CATEGORICAL_COLUMNS:
        train_x[col] = train_x[col].fillna("Unknown").astype(str)

    numeric_columns = [col for col in feature_columns if col not in CATEGORICAL_COLUMNS]
    medians = train_x[numeric_columns].median(numeric_only=True).fillna(0.0)
    train_x[numeric_columns] = train_x[numeric_columns].fillna(medians)

    train_x = pd.get_dummies(train_x, columns=CATEGORICAL_COLUMNS, drop_first=False)
    means = train_x.mean()
    stds = train_x.std(ddof=0).replace(0, 1.0)
    return FittedPreprocessor(
        feature_columns=feature_columns,
        encoded_columns=train_x.columns.tolist(),
        means=means,
        stds=stds,
        medians=medians,
    )


def prepare_design_matrix(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.Series, pd.Series]:
    preprocessor = fit_preprocessor(train)
    train_matrix = preprocessor.transform(train)
    test_matrix = preprocessor.transform(test)
    return (
        train_matrix.values,
        test_matrix.values,
        train_matrix.feature_names,
        preprocessor.means,
        preprocessor.stds,
    )


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
