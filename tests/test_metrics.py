from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_metrics_module():
    path = Path(__file__).parents[1] / "traffic_flow" / "metrics.py"
    spec = importlib.util.spec_from_file_location("traffic_flow_metrics_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


metrics_module = _load_metrics_module()


def test_percentage_metrics_handle_zero_targets():
    y_true = np.array([0.0, 10.0, 20.0])
    y_pred = np.array([0.0, 12.0, 18.0])

    assert np.isfinite(metrics_module.smape(y_true, y_pred))
    assert np.isfinite(metrics_module.wape(y_true, y_pred))


def test_regression_metrics_are_reported_in_original_scale():
    result = metrics_module.regression_metrics(
        np.array([10.0, 20.0, 30.0]),
        np.array([12.0, 18.0, 30.0]),
    )

    assert result["mae"] == 4.0 / 3.0
    assert result["mse"] == 8.0 / 3.0
    assert np.isclose(result["rmse"], np.sqrt(8.0 / 3.0))

