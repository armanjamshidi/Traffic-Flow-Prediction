"""Train one leakage-safe traffic-flow forecasting model."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN

from traffic_flow.data import process_data, process_delta_data
from traffic_flow.models import (
    get_delta_relax_lstm,
    get_gsa_kan,
    get_lstm_functional,
    get_lstm_no_attention,
)


MODEL_NAMES = ("lstm", "lstm_no_attention", "gsa_kan", "delta_relax_lstm")


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def _prepare_data(args):
    common = {
        "train": args.train_file,
        "test": args.test_file,
        "lags": args.lag,
        "validation_ratio": args.validation_ratio,
        "target_column": args.target_column,
    }
    metadata: dict[str, object] = {}

    if args.model == "delta_relax_lstm":
        if not args.timestamp_column:
            raise ValueError("--timestamp-column is required for delta_relax_lstm")
        result = process_delta_data(
            **common,
            timestamp_column=args.timestamp_column,
            timestamp_format=args.timestamp_format,
        )
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = result
        metadata["timestamp_column"] = args.timestamp_column
        metadata["timestamp_format"] = args.timestamp_format
    else:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = process_data(
            **common,
            timestamp_column=args.timestamp_column,
            timestamp_format=args.timestamp_format,
        )

    metadata.update(
        {
            "model": args.model,
            "lag": args.lag,
            "validation_ratio": args.validation_ratio,
            "target_column": args.target_column,
            "train_file": args.train_file,
            "test_file": args.test_file,
            "input_shape": list(X_train.shape[1:]),
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "hidden_units": args.hidden_units,
            "dropout": args.dropout,
            "early_stopping_patience": args.early_stopping_patience,
            "reduce_lr_patience": args.reduce_lr_patience,
            "reduce_lr_factor": args.reduce_lr_factor,
            "min_learning_rate": args.min_learning_rate,
        }
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, metadata


def _build_model(args, input_shape):
    units = [args.lag, args.hidden_units, args.hidden_units, 1]
    if args.model == "lstm":
        return get_lstm_functional(units, input_features=input_shape[-1], dropout=args.dropout)
    if args.model == "lstm_no_attention":
        return get_lstm_no_attention(units, input_features=input_shape[-1], dropout=args.dropout)
    if args.model == "gsa_kan":
        return get_gsa_kan(
            units,
            input_features=input_shape[-1],
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            grid_size=args.kan_grid_size,
            dropout=args.dropout,
        )
    if args.model == "delta_relax_lstm":
        return get_delta_relax_lstm(units, signal_features=input_shape[-1] - 1, dropout=args.dropout)
    raise ValueError(f"Unknown model: {args.model}")


def train_model(model, X_train, y_train, X_val, y_val, args):
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=args.reduce_lr_factor,
            patience=args.reduce_lr_patience,
            min_lr=args.min_learning_rate,
            verbose=1,
        ),
    ]
    return model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )


def evaluate_computational_cost(model, X_test, maximum_samples: int = 256) -> dict[str, float | int | str]:
    sample = X_test[: min(len(X_test), maximum_samples)]
    if not len(sample):
        raise ValueError("X_test is empty")

    # Warm-up excludes graph tracing and initial allocation from timed latency.
    model.predict(sample[:1], batch_size=1, verbose=0)
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss
    start = time.perf_counter()
    model.predict(sample, batch_size=1, verbose=0)
    elapsed = time.perf_counter() - start
    memory_after = process.memory_info().rss
    device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"

    return {
        "device": device,
        "samples": int(len(sample)),
        "parameter_count": int(model.count_params()),
        "total_seconds": float(elapsed),
        "latency_ms_per_sample": float(1000.0 * elapsed / len(sample)),
        "rss_memory_change_mb": float((memory_after - memory_before) / (1024**2)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, default="lstm")
    parser.add_argument("--train-file", default="data/train.csv")
    parser.add_argument("--test-file", default="data/test.csv")
    parser.add_argument("--target-column", default="Lane 1 Flow (Veh/5 Minutes)")
    parser.add_argument("--timestamp-column", default="5 Minutes")
    parser.add_argument("--timestamp-format", default="%d/%m/%Y %H:%M")
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--kan-grid-size", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=100)
    parser.add_argument("--reduce-lr-patience", type=int, default=10)
    parser.add_argument("--reduce-lr-factor", type=float, default=0.2)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=3261)
    parser.add_argument("--output-dir", default="artifacts/models")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    set_random_seed(args.seed)
    X_train, y_train, X_val, y_val, X_test, y_test, _, metadata = _prepare_data(args)
    model = _build_model(args, X_train.shape[1:])
    model.summary()
    history = train_model(model, X_train, y_train, X_val, y_val, args)

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    model_path = output_directory / f"{args.model}.keras"
    model.save(model_path)
    pd.DataFrame(history.history).to_csv(output_directory / f"{args.model}_history.csv", index=False)

    metadata["seed"] = args.seed
    metadata["best_validation_loss"] = float(min(history.history["val_loss"]))
    metadata["computational_cost"] = evaluate_computational_cost(model, X_test)
    (output_directory / f"{args.model}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()

