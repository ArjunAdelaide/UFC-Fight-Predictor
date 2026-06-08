from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ufc_predictor.config import LogisticRegressionConfig


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -500, 500)
    return 1.0 / (1.0 + np.exp(-values))


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


@dataclass(frozen=True)
class NumpyLogisticRegression:
    weights: np.ndarray

    @classmethod
    def fit(
        cls,
        x_train: np.ndarray,
        y_train: np.ndarray,
        config: LogisticRegressionConfig = LogisticRegressionConfig(),
    ) -> "NumpyLogisticRegression":
        weights = train_logistic_regression(
            x_train,
            y_train,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            l2=config.l2,
        )
        return cls(weights=weights)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        return sigmoid(matrix @ self.weights)
