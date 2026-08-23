"""Optional causal cleaning and chronological splitting for raw traffic-flow CSV data.

This module deliberately does *not* normalise or round flow values. Scaling is
fitted later, using only the model-training portion in ``traffic_flow/data.py``.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TARGET_COLUMN = "Lane 1 Flow (Veh/5 Minutes)"


def _causal_hampel(
    values: pd.Series,
    window: int = 24,
    n_sigma: float = 4.0,
) -> pd.Series:
    """Causally impute missing values and replace only robust local outliers.

    Each decision uses preceding cleaned observations only.  It therefore does
    not leak future validation/test information into earlier rows.
    """
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    cleaned = np.empty_like(numeric)
    history: list[float] = []

    for index, value in enumerate(numeric):
        recent = np.asarray(history[-window:], dtype=np.float64)

        if not np.isfinite(value):
            value = history[-1] if history else 0.0
        elif recent.size >= max(5, window // 4):
            median = float(np.median(recent))
            mad = float(np.median(np.abs(recent - median)))
            robust_sigma = 1.4826 * mad
            deviation = abs(value - median)
            if robust_sigma > 0.0 and deviation > n_sigma * robust_sigma:
                value = median
            elif robust_sigma == 0.0:
                # A perfectly flat recent window has zero MAD.  Use a small,
                # scale-aware fallback so an isolated sensor spike is not
                # silently accepted merely because the robust scale is zero.
                fallback_scale = max(1.0, 0.05 * abs(median))
                if deviation > n_sigma * fallback_scale:
                    value = median

        cleaned[index] = value
        history.append(float(value))

    return pd.Series(cleaned, index=values.index, name=values.name)


def _flow_columns(frame: pd.DataFrame, requested: Iterable[str] | None) -> list[str]:
    if requested:
        columns = list(dict.fromkeys(requested))
    else:
        columns = [column for column in frame.columns if "flow" in column.casefold()]
    if not columns:
        raise ValueError("No flow columns found. Pass --flow-columns explicitly.")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing flow columns: {missing}")
    return columns


def preprocess_and_split(
    input_file: str | Path,
    train_output: str | Path,
    test_output: str | Path,
    train_ratio: float = 0.8,
    flow_columns: Iterable[str] | None = None,
    timestamp_column: str | None = None,
    timestamp_format: str | None = None,
    hampel_window: int = 24,
    hampel_sigma: float = 4.0,
) -> tuple[Path, Path]:
    if not 0.5 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0.5 and 1.0")

    frame = pd.read_csv(input_file)
    if frame.empty:
        raise ValueError(f"Input CSV is empty: {input_file}")

    if timestamp_column:
        if timestamp_column not in frame.columns:
            raise KeyError(f"Timestamp column not found: {timestamp_column}")
        parsed = pd.to_datetime(
            frame[timestamp_column],
            format=timestamp_format,
            errors="coerce",
        )
        if parsed.isna().any():
            raise ValueError("Timestamp column contains invalid values")
        frame = frame.assign(_parsed_timestamp=parsed).sort_values("_parsed_timestamp", kind="stable")
        if frame["_parsed_timestamp"].duplicated().any():
            raise ValueError("Timestamp column contains duplicates")
        frame = frame.drop(columns="_parsed_timestamp").reset_index(drop=True)

    columns = _flow_columns(frame, flow_columns)
    for column in columns:
        frame[column] = _causal_hampel(frame[column], hampel_window, hampel_sigma)

    split_index = int(np.floor(len(frame) * train_ratio))
    if split_index < 2 or len(frame) - split_index < 2:
        raise ValueError("The requested split leaves too few rows in train or test")

    train_frame = frame.iloc[:split_index].copy()
    test_frame = frame.iloc[split_index:].copy()
    train_path = Path(train_output)
    test_path = Path(test_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)
    train_frame.to_csv(train_path, index=False)
    test_frame.to_csv(test_path, index=False)
    return train_path, test_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="transformed_data.csv")
    parser.add_argument("--train-output", default="data/train.csv")
    parser.add_argument("--test-output", default="data/test.csv")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--timestamp-column")
    parser.add_argument(
        "--timestamp-format",
        default="%d/%m/%Y %H:%M",
        help="Explicit pandas datetime format; default matches the supplied CSV.",
    )
    parser.add_argument(
        "--flow-columns",
        nargs="+",
        help="Flow columns to clean. By default all columns containing 'flow' are used.",
    )
    parser.add_argument("--hampel-window", type=int, default=24)
    parser.add_argument("--hampel-sigma", type=float, default=4.0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    train_path, test_path = preprocess_and_split(
        input_file=arguments.input,
        train_output=arguments.train_output,
        test_output=arguments.test_output,
        train_ratio=arguments.train_ratio,
        flow_columns=arguments.flow_columns,
        timestamp_column=arguments.timestamp_column,
        timestamp_format=arguments.timestamp_format,
        hampel_window=arguments.hampel_window,
        hampel_sigma=arguments.hampel_sigma,
    )
    print(f"Saved chronological train data to {train_path}")
    print(f"Saved chronological test data to {test_path}")

