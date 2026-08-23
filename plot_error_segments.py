"""Plot pointwise absolute and squared errors for the two core LSTM models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from traffic_flow.data import process_data
from traffic_flow.models import AttentionLayer


MODEL_FILES = {
    "lstm": "lstm.keras",
    "lstm_no_attention": "lstm_no_attention.keras",
}
DISPLAY_NAMES = {
    "lstm": "LSTM with Attention",
    "lstm_no_attention": "LSTM without Attention",
}


def _predictions(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    result = process_data(
        args.train_file,
        args.test_file,
        args.lag,
        validation_ratio=args.validation_ratio,
        target_column=args.target_column,
        timestamp_column=args.timestamp_column,
        timestamp_format=args.timestamp_format,
    )
    X_test, y_test, scaler = result[4], result[5], result[6]
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)
    predictions: dict[str, np.ndarray] = {}
    for name, filename in MODEL_FILES.items():
        path = Path(args.model_dir) / filename
        custom_objects = {"AttentionLayer": AttentionLayer} if name == "lstm" else None
        model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
        scaled = model.predict(X_test, batch_size=args.batch_size, verbose=0)
        predictions[name] = scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(-1)
    return y_true, predictions


def _plot_segments(
    errors: dict[str, np.ndarray],
    output_dir: Path,
    metric: str,
    segment_size: int,
    maximum_points: int,
) -> None:
    count = min(maximum_points, min(map(len, errors.values())))
    for start in range(0, count, segment_size):
        end = min(start + segment_size, count)
        figure, axis = plt.subplots(figsize=(9, 5))
        for name, values in errors.items():
            axis.plot(np.arange(start, end), values[start:end], label=DISPLAY_NAMES[name], linewidth=1.7)
        axis.set_xlabel("Chronological test-sample index")
        axis.set_ylabel(metric)
        axis.set_title(f"{metric}: samples {start}-{end}")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        figure.tight_layout()
        stem = output_dir / f"{metric.lower().replace(' ', '_')}_{start:03d}_{end:03d}"
        figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="artifacts/models")
    parser.add_argument("--train-file", default="data/train.csv")
    parser.add_argument("--test-file", default="data/test.csv")
    parser.add_argument("--output-dir", default="artifacts/error_segments")
    parser.add_argument("--target-column", default="Lane 1 Flow (Veh/5 Minutes)")
    parser.add_argument("--timestamp-column", default="5 Minutes")
    parser.add_argument("--timestamp-format", default="%d/%m/%Y %H:%M")
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--segment-size", type=int, default=50)
    parser.add_argument("--maximum-points", type=int, default=288)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true, predictions = _predictions(args)
    table = pd.DataFrame({"sample_index": np.arange(len(y_true)), "true_flow": y_true})
    absolute_errors: dict[str, np.ndarray] = {}
    squared_errors: dict[str, np.ndarray] = {}
    for name, prediction in predictions.items():
        absolute_errors[name] = np.abs(y_true - prediction)
        squared_errors[name] = np.square(y_true - prediction)
        table[f"{name}_prediction"] = prediction
        table[f"{name}_absolute_error"] = absolute_errors[name]
        table[f"{name}_squared_error"] = squared_errors[name]
    table.to_csv(output_dir / "pointwise_errors.csv", index=False)
    _plot_segments(absolute_errors, output_dir, "Absolute error", args.segment_size, args.maximum_points)
    _plot_segments(squared_errors, output_dir, "Squared error", args.segment_size, args.maximum_points)
    print(f"Saved segmented error plots to {output_dir}")


if __name__ == "__main__":
    main()

