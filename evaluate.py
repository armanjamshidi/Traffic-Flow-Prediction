"""Evaluate trained models on the untouched chronological test period."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from traffic_flow.data import process_data, process_delta_data
from traffic_flow.models import (
    AttentionLayer,
    BSplineKANLayer,
    DeltaRelaxLSTMCell,
    GatedSelfAttentionBlock,
)
from traffic_flow.metrics import regression_metrics


MODEL_NAMES = ("lstm", "lstm_no_attention", "gsa_kan", "delta_relax_lstm")
CORE_MODEL_NAMES = ("lstm", "lstm_no_attention")
DATA_DEFAULTS = {
    "train_file": "data/train.csv",
    "test_file": "data/test.csv",
    "target_column": "Lane 1 Flow (Veh/5 Minutes)",
    "timestamp_column": "5 Minutes",
    "timestamp_format": "%d/%m/%Y %H:%M",
    "lag": 12,
    "validation_ratio": 0.15,
}
DISPLAY_NAMES = {
    "lstm": "LSTM with Attention",
    "lstm_no_attention": "LSTM without Attention",
    "gsa_kan": "GSA-KAN",
    "delta_relax_lstm": "DeltaRelaxLSTM",
}
CUSTOM_OBJECTS = {
    "AttentionLayer": AttentionLayer,
    "BSplineKANLayer": BSplineKANLayer,
    "GatedSelfAttentionBlock": GatedSelfAttentionBlock,
    "DeltaRelaxLSTMCell": DeltaRelaxLSTMCell,
}


def _metadata(model_directory: Path, model_name: str) -> dict:
    path = model_directory / f"{model_name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _resolved_value(args, metadata: dict, key: str):
    command_line_value = getattr(args, key)
    if command_line_value is not None:
        return command_line_value
    if key in metadata:
        return metadata[key]
    return DATA_DEFAULTS[key]


def _test_data(args, model_name: str, metadata: dict):
    common = {
        "train": _resolved_value(args, metadata, "train_file"),
        "test": _resolved_value(args, metadata, "test_file"),
        "lags": int(_resolved_value(args, metadata, "lag")),
        "validation_ratio": float(_resolved_value(args, metadata, "validation_ratio")),
        "target_column": _resolved_value(args, metadata, "target_column"),
    }
    timestamp_column = _resolved_value(args, metadata, "timestamp_column")
    timestamp_format = _resolved_value(args, metadata, "timestamp_format")
    if model_name == "delta_relax_lstm":
        if not timestamp_column:
            raise ValueError("A timestamp column is required to evaluate DeltaRelaxLSTM")
        result = process_delta_data(
            **common,
            timestamp_column=timestamp_column,
            timestamp_format=timestamp_format,
        )
        return result[4], result[5], result[6]
    result = process_data(
        **common,
        timestamp_column=timestamp_column,
        timestamp_format=timestamp_format,
    )
    return result[4], result[5], result[6]


def plot_results(y_true, predictions, output_path: Path, maximum_points: int = 288) -> None:
    point_count = min(len(y_true), maximum_points)
    x = np.arange(point_count)
    figure, axis = plt.subplots(figsize=(15, 7))
    axis.plot(x, y_true[:point_count], label="True flow", linewidth=2.2, color="black")
    for name, prediction in predictions.items():
        axis.plot(x, prediction[:point_count], label=DISPLAY_NAMES[name], linewidth=1.6)
    axis.set_xlabel("Chronological test sample")
    axis.set_ylabel("Flow (vehicles / 5 minutes)")
    axis.set_title("Traffic-flow prediction comparison")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(CORE_MODEL_NAMES))
    parser.add_argument("--model-dir", default="artifacts/models")
    parser.add_argument("--output-dir", default="artifacts/evaluation")
    parser.add_argument("--train-file")
    parser.add_argument("--test-file")
    parser.add_argument("--target-column")
    parser.add_argument("--timestamp-column")
    parser.add_argument("--timestamp-format")
    parser.add_argument("--lag", type=int)
    parser.add_argument("--validation-ratio", type=float)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_directory = Path(args.model_dir)
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, float | str]] = []
    predictions: dict[str, np.ndarray] = {}
    truths: dict[str, np.ndarray] = {}
    common_truth: np.ndarray | None = None

    for model_name in args.models:
        model_path = model_directory / f"{model_name}.keras"
        if not model_path.exists():
            print(f"Skipping {model_name}: model file not found at {model_path}")
            continue
        metadata = _metadata(model_directory, model_name)
        try:
            X_test, y_test, scaler = _test_data(args, model_name, metadata)
            model = load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
            predicted_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
        except (ValueError, KeyError) as error:
            print(f"Skipping {model_name}: {error}")
            continue

        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
        y_pred = scaler.inverse_transform(predicted_scaled).reshape(-1)
        model_metrics = regression_metrics(y_true, y_pred)
        results.append(
            {"model": DISPLAY_NAMES[model_name], "samples": int(len(y_true)), **model_metrics}
        )
        predictions[model_name] = y_pred
        truths[model_name] = y_true
        common_truth = y_true if common_truth is None else common_truth

        print(f"\n{DISPLAY_NAMES[model_name]}")
        for metric_name, value in model_metrics.items():
            print(f"  {metric_name:<20} {value:.6f}")

    if not results or common_truth is None:
        raise RuntimeError("No trained models were successfully evaluated")

    summary = pd.DataFrame(results).set_index("model")
    summary.to_csv(output_directory / "model_comparison_metrics.csv")
    aligned_predictions = {
        name: prediction
        for name, prediction in predictions.items()
        if len(truths[name]) == len(common_truth)
        and np.allclose(truths[name], common_truth, rtol=0.0, atol=1e-6)
    }
    omitted = [name for name in predictions if name not in aligned_predictions]
    if omitted:
        labels = ", ".join(DISPLAY_NAMES[name] for name in omitted)
        print(
            "\nNot plotting models evaluated on different target samples: "
            f"{labels}. Their aggregate metrics remain in the CSV with sample counts."
        )
    plot_results(common_truth, aligned_predictions, output_directory / "overlay_predictions.png")
    print(f"\n{summary}")
    print(f"\nSaved evaluation files to {output_directory}")


if __name__ == "__main__":
    main()

