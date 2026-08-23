"""Evaluation metrics reported in original traffic-flow units."""

from __future__ import annotations

import math

import numpy as np
from sklearn import metrics


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > np.finfo(float).eps
    if not np.any(mask):
        return 0.0
    return float(200.0 * np.mean(np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum(np.abs(y_true)))
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(100.0 * np.sum(np.abs(y_true - y_pred)) / denominator)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = metrics.mean_squared_error(y_true, y_pred)
    return {
        "explained_variance": float(metrics.explained_variance_score(y_true, y_pred)),
        "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "r2": float(metrics.r2_score(y_true, y_pred)),
        "smape_percent": smape(y_true, y_pred),
        "wape_percent": wape(y_true, y_pred),
    }

