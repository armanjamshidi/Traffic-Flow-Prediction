from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_data_module():
    path = Path(__file__).parents[1] / "traffic_flow" / "data.py"
    spec = importlib.util.spec_from_file_location("traffic_flow_data_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


data_module = _load_data_module()


def test_gap_filter_removes_every_window_that_crosses_the_gap():
    values = np.arange(20, dtype=np.float32)[:, None]
    timestamps = pd.Series(pd.date_range("2026-01-01", periods=20, freq="5min"))
    timestamps.iloc[10:] += pd.Timedelta(hours=2)

    X, y = data_module._make_windows(
        values,
        lags=3,
        timestamps=timestamps,
        expected_interval_seconds=300.0,
    )

    assert X.shape == (14, 3, 1)
    assert y.shape == (14,)
    assert not np.any(y == 10)
    assert not np.any(y == 11)
    assert not np.any(y == 12)


def test_scaler_is_fitted_before_the_validation_period(tmp_path):
    train_timestamps = pd.date_range("2026-01-01", periods=30, freq="5min")
    test_timestamps = pd.date_range(train_timestamps[-1] + pd.Timedelta(minutes=5), periods=8, freq="5min")
    train_values = np.concatenate([np.arange(24), np.arange(100, 106)])
    test_values = np.arange(200, 208)

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame(
        {
            "5 Minutes": train_timestamps.strftime("%d/%m/%Y %H:%M"),
            "Lane 1 Flow (Veh/5 Minutes)": train_values,
        }
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        {
            "5 Minutes": test_timestamps.strftime("%d/%m/%Y %H:%M"),
            "Lane 1 Flow (Veh/5 Minutes)": test_values,
        }
    ).to_csv(test_path, index=False)

    result = data_module.process_data(train_path, test_path, lags=3, validation_ratio=0.2)
    X_train, _, X_val, _, X_test, _, scaler = result

    assert X_train.shape[0] == 21
    assert X_val.shape[0] == 3
    assert X_test.shape[0] == 5
    assert scaler.data_max_.item() == 23
    assert X_val.min() > 1.0
    assert X_test.min() > X_val.min()

