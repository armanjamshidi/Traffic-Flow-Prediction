"""Create interpretable summaries of the LSTM temporal-attention weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.cluster import KMeans

from traffic_flow.data import process_data
from traffic_flow.models import AttentionLayer


def _save_figure(figure: plt.Figure, output_stem: Path) -> None:
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _load_attention_weights(args: argparse.Namespace) -> np.ndarray:
    result = process_data(
        args.train_file,
        args.test_file,
        args.lag,
        validation_ratio=args.validation_ratio,
        target_column=args.target_column,
        timestamp_column=args.timestamp_column,
        timestamp_format=args.timestamp_format,
    )
    X_test = result[4]
    model = tf.keras.models.load_model(
        args.model_file,
        custom_objects={"AttentionLayer": AttentionLayer},
        compile=False,
    )
    attention_layer = model.get_layer("attention_layer")
    sequence_model = tf.keras.Model(model.inputs, attention_layer.input)
    sequence_output = sequence_model.predict(X_test, batch_size=args.batch_size, verbose=0)
    weights = attention_layer.attention_weights(sequence_output).numpy()
    if weights.ndim != 2:
        raise ValueError(f"Expected 2-D attention weights, got {weights.shape}")
    return weights


def create_attention_report(weights: np.ndarray, output_dir: Path, clusters: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_count, time_steps = weights.shape
    lags = np.arange(time_steps)

    mean = weights.mean(axis=0)
    standard_deviation = weights.std(axis=0)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(lags, mean, color="navy", linewidth=2, label="Mean attention")
    axis.fill_between(
        lags,
        mean - standard_deviation,
        mean + standard_deviation,
        color="skyblue",
        alpha=0.4,
        label="+/- 1 standard deviation",
    )
    axis.set(xlabel="Lag index (oldest to newest)", ylabel="Attention weight", title="Attention by lag")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    _save_figure(figure, output_dir / "attention_by_lag")

    clipped = np.clip(weights, np.finfo(float).eps, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.histplot(entropy, bins=30, kde=True, ax=axis, color="steelblue")
    axis.set(xlabel="Attention entropy (nats)", ylabel="Sample count", title="Attention entropy")
    _save_figure(figure, output_dir / "attention_entropy")

    heatmap_count = min(50, sample_count)
    figure, axis = plt.subplots(figsize=(10, 7))
    sns.heatmap(weights[:heatmap_count], cmap="viridis", ax=axis)
    axis.set(xlabel="Lag index (oldest to newest)", ylabel="Test-sample index", title="Attention heatmap")
    _save_figure(figure, output_dir / "attention_heatmap")

    cluster_count = min(max(1, clusters), sample_count)
    kmeans = KMeans(n_clusters=cluster_count, random_state=seed, n_init=10).fit(weights)
    figure, axis = plt.subplots(figsize=(7, 4))
    for cluster in range(cluster_count):
        members = weights[kmeans.labels_ == cluster]
        axis.plot(lags, members.mean(axis=0), linewidth=2, label=f"Cluster {cluster + 1} (n={len(members)})")
    axis.set(xlabel="Lag index (oldest to newest)", ylabel="Mean attention weight", title="Attention-pattern clusters")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    _save_figure(figure, output_dir / "attention_clusters")

    pd.DataFrame(weights, columns=[f"lag_{index}" for index in lags]).assign(
        entropy=entropy,
        cluster=kmeans.labels_ + 1,
    ).to_csv(output_dir / "attention_weights.csv", index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", default="artifacts/models/lstm.keras")
    parser.add_argument("--train-file", default="data/train.csv")
    parser.add_argument("--test-file", default="data/test.csv")
    parser.add_argument("--output-dir", default="artifacts/attention")
    parser.add_argument("--target-column", default="Lane 1 Flow (Veh/5 Minutes)")
    parser.add_argument("--timestamp-column", default="5 Minutes")
    parser.add_argument("--timestamp-format", default="%d/%m/%Y %H:%M")
    parser.add_argument("--lag", type=int, default=12)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    weights = _load_attention_weights(args)
    create_attention_report(weights, Path(args.output_dir), args.clusters, args.seed)
    print(f"Saved attention analysis to {args.output_dir}")


if __name__ == "__main__":
    main()

