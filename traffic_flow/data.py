"""Leakage-safe preparation of chronological traffic-flow data.

The CSV rows are treated as ordered observations.  Training, validation and
test windows are built independently so that a raw CSV row cannot occur in
both the training and validation arrays.  Every scaler is fitted only on the
chronological training portion.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


DEFAULT_TARGET_COLUMN = "Lane 1 Flow (Veh/5 Minutes)"


def _read_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8")
    if frame.empty:
        raise ValueError(f"CSV file is empty: {path}")
    return frame


def _numeric_values(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    missing_columns = [column for column in columns if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns}")

    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    bad_rows = np.flatnonzero(~np.isfinite(numeric.to_numpy()).all(axis=1))
    if bad_rows.size:
        preview = ", ".join(map(str, bad_rows[:10]))
        raise ValueError(
            "Missing or non-numeric flow values remain at zero-based rows "
            f"{preview}. Run preprc.py before training."
        )
    return numeric.to_numpy(dtype=np.float32)


def _chronological_parts(
    frame: pd.DataFrame,
    lags: int,
    validation_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if lags < 1:
        raise ValueError("lags must be at least 1")
    if not 0.0 < validation_ratio < 0.5:
        raise ValueError("validation_ratio must be between 0 and 0.5")

    split_index = int(np.floor(len(frame) * (1.0 - validation_ratio)))
    train_frame = frame.iloc[:split_index].copy()
    validation_frame = frame.iloc[split_index:].copy()

    if len(train_frame) <= lags or len(validation_frame) <= lags:
        raise ValueError(
            "Both chronological training and validation portions must contain "
            f"more than {lags} rows. Got {len(train_frame)} and "
            f"{len(validation_frame)} rows."
        )
    return train_frame, validation_frame


def _make_windows(
    values: np.ndarray,
    lags: int,
    target_index: int = 0,
    timestamps: pd.Series | None = None,
    expected_interval_seconds: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one-step-ahead windows without crossing missing time intervals."""
    if values.ndim == 1:
        values = values[:, None]
    if len(values) <= lags:
        raise ValueError(f"Need more than {lags} rows to create windows")

    endpoints = np.arange(lags, len(values))
    if timestamps is not None:
        if expected_interval_seconds is None:
            raise ValueError("expected_interval_seconds is required with timestamps")
        differences = timestamps.diff().dt.total_seconds().to_numpy(dtype=np.float64)
        tolerance = max(1e-6, expected_interval_seconds * 1e-6)
        endpoints = np.asarray(
            [
                endpoint
                for endpoint in endpoints
                if np.all(
                    np.abs(differences[endpoint - lags + 1 : endpoint + 1] - expected_interval_seconds)
                    <= tolerance
                )
            ],
            dtype=int,
        )
        if not endpoints.size:
            raise ValueError("No contiguous windows remain after timestamp-gap filtering")

    X = np.stack([values[i - lags : i] for i in endpoints])
    y = values[endpoints, target_index]
    return X.astype(np.float32), y.astype(np.float32)


def process_data(
    train: str | Path,
    test: str | Path,
    lags: int,
    validation_ratio: float = 0.15,
    target_column: str = DEFAULT_TARGET_COLUMN,
    timestamp_column: str | None = "5 Minutes",
    timestamp_format: str | None = "%d/%m/%Y %H:%M",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Prepare independent chronological windows for a univariate model.

    Returns ``X_train, y_train, X_val, y_val, X_test, y_test, scaler``.
    All X arrays have shape ``(samples, lags, 1)``.  The scaler is fitted only
    on the training portion before validation begins.
    """
    train_frame = _read_csv(train)
    test_frame = _read_csv(test)
    train_frame, validation_frame = _chronological_parts(train_frame, lags, validation_ratio)

    train_timestamps = None
    validation_timestamps = None
    test_timestamps = None
    reference_seconds = None
    if timestamp_column:
        train_timestamps = _parse_timestamps(train_frame, timestamp_column, timestamp_format)
        validation_timestamps = _parse_timestamps(validation_frame, timestamp_column, timestamp_format)
        test_timestamps = _parse_timestamps(test_frame, timestamp_column, timestamp_format)
        reference_seconds = _reference_interval_seconds(train_timestamps)
        if validation_timestamps.iloc[0] <= train_timestamps.iloc[-1]:
            raise ValueError("Validation timestamps must follow training timestamps")
        if test_timestamps.iloc[0] <= validation_timestamps.iloc[-1]:
            raise ValueError("Test timestamps must follow training/validation timestamps")

    train_values = _numeric_values(train_frame, [target_column])
    validation_values = _numeric_values(validation_frame, [target_column])
    test_values = _numeric_values(test_frame, [target_column])

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    train_scaled = scaler.fit_transform(train_values)
    validation_scaled = scaler.transform(validation_values)
    test_scaled = scaler.transform(test_values)

    X_train, y_train = _make_windows(
        train_scaled, lags, timestamps=train_timestamps, expected_interval_seconds=reference_seconds
    )
    X_val, y_val = _make_windows(
        validation_scaled,
        lags,
        timestamps=validation_timestamps,
        expected_interval_seconds=reference_seconds,
    )
    X_test, y_test = _make_windows(
        test_scaled, lags, timestamps=test_timestamps, expected_interval_seconds=reference_seconds
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


def _parse_timestamps(
    frame: pd.DataFrame,
    timestamp_column: str,
    timestamp_format: str | None = None,
) -> pd.Series:
    if timestamp_column not in frame.columns:
        raise KeyError(f"Timestamp column not found: {timestamp_column}")
    timestamps = pd.to_datetime(
        frame[timestamp_column],
        format=timestamp_format,
        errors="coerce",
    )
    if timestamps.isna().any():
        rows = np.flatnonzero(timestamps.isna().to_numpy())[:10]
        raise ValueError(f"Invalid timestamps at zero-based rows: {rows.tolist()}")
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValueError("Timestamps must be strictly increasing and unique")
    return timestamps


def _reference_interval_seconds(timestamps: pd.Series) -> float:
    differences = timestamps.diff().dt.total_seconds().dropna()
    differences = differences[differences > 0]
    if differences.empty:
        raise ValueError("At least two distinct timestamps are required")
    return float(differences.median())


def _normalised_deltas(timestamps: pd.Series, reference_seconds: float, maximum: float = 12.0) -> np.ndarray:
    differences = timestamps.diff().dt.total_seconds().to_numpy(
        dtype=np.float64,
        copy=True
    )
    differences[0] = reference_seconds
    deltas = np.clip(differences / reference_seconds, 0.0, maximum)
    return deltas.astype(np.float32)


def _make_delta_windows(values: np.ndarray, deltas: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    features = np.column_stack([values.reshape(-1), deltas])
    return _make_windows(features, lags, target_index=0)


def process_delta_data(
    train: str | Path,
    test: str | Path,
    lags: int,
    timestamp_column: str,
    timestamp_format: str | None = None,
    validation_ratio: float = 0.15,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Prepare flow plus real normalised time gaps for DeltaRelaxLSTM."""
    full_train_frame = _read_csv(train)
    test_frame = _read_csv(test)
    train_frame, validation_frame = _chronological_parts(full_train_frame, lags, validation_ratio)

    train_timestamps = _parse_timestamps(train_frame, timestamp_column, timestamp_format)
    validation_timestamps = _parse_timestamps(validation_frame, timestamp_column, timestamp_format)
    test_timestamps = _parse_timestamps(test_frame, timestamp_column, timestamp_format)
    reference_seconds = _reference_interval_seconds(train_timestamps)
    if validation_timestamps.iloc[0] <= train_timestamps.iloc[-1]:
        raise ValueError("Validation timestamps must follow training timestamps")
    if test_timestamps.iloc[0] <= validation_timestamps.iloc[-1]:
        raise ValueError("Test timestamps must follow training/validation timestamps")

    train_values = _numeric_values(train_frame, [target_column])
    validation_values = _numeric_values(validation_frame, [target_column])
    test_values = _numeric_values(test_frame, [target_column])
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    train_scaled = scaler.fit_transform(train_values)
    validation_scaled = scaler.transform(validation_values)
    test_scaled = scaler.transform(test_values)

    X_train, y_train = _make_delta_windows(
        train_scaled, _normalised_deltas(train_timestamps, reference_seconds), lags
    )
    X_val, y_val = _make_delta_windows(
        validation_scaled, _normalised_deltas(validation_timestamps, reference_seconds), lags
    )
    X_test, y_test = _make_delta_windows(
        test_scaled, _normalised_deltas(test_timestamps, reference_seconds), lags
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

