from __future__ import annotations

import numpy as np
import pandas as pd


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-12
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob >= 0.5).astype(int) == y_true))


def evaluate_probabilities(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | str]:
    return {
        "model": name,
        "accuracy": accuracy(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "brier": brier_score(y_true, y_prob),
    }


def evaluate_rule(name: str, y_true: np.ndarray, score: pd.Series) -> dict[str, float | str]:
    probabilities = np.where(score > 0, 0.60, np.where(score < 0, 0.40, 0.50))
    return evaluate_probabilities(name, y_true, probabilities)


def summarize_metrics(metrics: list[dict[str, float | str]]) -> pd.DataFrame:
    frame = pd.DataFrame(metrics)
    return frame.sort_values("log_loss", ascending=True).reset_index(drop=True)


def bucket_label(start: float, end: float) -> str:
    return f"{int(round(start * 100))}-{int(round(end * 100))}%"


def probability_calibration_buckets(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    bucket_width: float = 0.10,
) -> pd.DataFrame:
    rows = []
    for start in np.arange(0.0, 1.0, bucket_width):
        end = min(start + bucket_width, 1.0)
        if end >= 1.0:
            mask = (y_prob >= start) & (y_prob <= end)
        else:
            mask = (y_prob >= start) & (y_prob < end)

        rows.append(
            {
                "probability_bucket": bucket_label(start, end),
                "count": int(mask.sum()),
                "avg_predicted_probability": float(np.mean(y_prob[mask])) if mask.any() else np.nan,
                "actual_fighter_a_win_rate": float(np.mean(y_true[mask])) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def confidence_buckets(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    bucket_width: float = 0.10,
) -> pd.DataFrame:
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    predicted = (y_prob >= 0.5).astype(int)
    correct = predicted == y_true

    rows = []
    for start in np.arange(0.5, 1.0, bucket_width):
        end = min(start + bucket_width, 1.0)
        if end >= 1.0:
            mask = (confidence >= start) & (confidence <= end)
        else:
            mask = (confidence >= start) & (confidence < end)

        rows.append(
            {
                "confidence_bucket": bucket_label(start, end),
                "count": int(mask.sum()),
                "avg_confidence": float(np.mean(confidence[mask])) if mask.any() else np.nan,
                "accuracy": float(np.mean(correct[mask])) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)
