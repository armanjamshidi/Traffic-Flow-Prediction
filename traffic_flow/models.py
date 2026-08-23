"""Serializable TensorFlow models for one-step traffic-flow forecasting."""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)
from tensorflow.keras.utils import register_keras_serializable


DEFAULT_DROPOUT = 0.1


@register_keras_serializable(package="TrafficFlow")
class AttentionLayer(Layer):
    """Learn a normalised importance weight for every recurrent time step."""

    def build(self, input_shape):
        feature_count = int(input_shape[-1])
        self.score_kernel = self.add_weight(
            name="score_kernel",
            shape=(feature_count, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.score_bias = self.add_weight(
            name="score_bias",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def attention_weights(self, inputs):
        scores = tf.squeeze(tf.tanh(tf.matmul(inputs, self.score_kernel) + self.score_bias), axis=-1)
        return tf.nn.softmax(scores, axis=1)

    def call(self, inputs):
        weights = self.attention_weights(inputs)
        return tf.reduce_sum(inputs * weights[..., None], axis=1)


def get_lstm_functional(units, input_features: int = 1, dropout: float = DEFAULT_DROPOUT) -> Model:
    time_steps = int(units[0])
    inputs = Input(shape=(time_steps, input_features), name="flow_sequence")
    x = LSTM(int(units[1]), return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = LSTM(int(units[2]), return_sequences=True)(x)
    x = AttentionLayer(name="attention_layer")(x)
    x = Dropout(dropout)(x)
    outputs = Dense(int(units[3]), activation="sigmoid", name="scaled_flow")(x)
    return Model(inputs=inputs, outputs=outputs, name="LSTM_Attention")


def get_lstm_no_attention(units, input_features: int = 1, dropout: float = DEFAULT_DROPOUT) -> Model:
    time_steps = int(units[0])
    model = Sequential(name="LSTM_NoAttention")
    model.add(Input(shape=(time_steps, input_features), name="flow_sequence"))
    model.add(LSTM(int(units[1]), return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(int(units[2])))
    model.add(Dropout(dropout))
    model.add(Dense(int(units[3]), activation="sigmoid", name="scaled_flow"))
    return model


@register_keras_serializable(package="TrafficFlow")
class BSplineKANLayer(Layer):
    """A KAN layer with learnable cubic B-spline edge functions.

    For every input-output edge, the layer learns independent spline
    coefficients.  A SiLU base branch is retained for stable extrapolation,
    following the common practical KAN formulation.
    """

    def __init__(
        self,
        output_dim: int,
        grid_size: int = 8,
        spline_order: int = 3,
        grid_min: float = 0.0,
        grid_max: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if output_dim < 1 or grid_size < 2 or spline_order < 1:
            raise ValueError("Invalid KAN dimensions")
        if grid_max <= grid_min:
            raise ValueError("grid_max must be greater than grid_min")
        self.output_dim = int(output_dim)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.grid_min = float(grid_min)
        self.grid_max = float(grid_max)

    def build(self, input_shape):
        self.input_dim = int(input_shape[-1])
        basis_count = self.grid_size + self.spline_order
        step = (self.grid_max - self.grid_min) / self.grid_size
        knot_values = self.grid_min + step * np.arange(
            -self.spline_order,
            self.grid_size + self.spline_order + 1,
            dtype=np.float32,
        )
        self.knots = tf.constant(knot_values, dtype=self.dtype or tf.float32)
        self.base_kernel = self.add_weight(
            name="base_kernel",
            shape=(self.input_dim, self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.spline_kernel = self.add_weight(
            name="spline_kernel",
            shape=(self.input_dim, self.output_dim, basis_count),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.05),
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.output_dim,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def _basis(self, inputs):
        epsilon = tf.cast(tf.keras.backend.epsilon(), inputs.dtype)
        clipped = tf.clip_by_value(inputs, self.grid_min, self.grid_max - epsilon)
        x = clipped[..., None]
        knots = tf.cast(self.knots, inputs.dtype)
        basis = tf.cast((x >= knots[:-1]) & (x < knots[1:]), inputs.dtype)

        for degree in range(1, self.spline_order + 1):
            left_denominator = knots[degree:-1] - knots[: -(degree + 1)]
            right_denominator = knots[degree + 1 :] - knots[1:-degree]
            left = ((x - knots[: -(degree + 1)]) / left_denominator) * basis[..., :-1]
            right = ((knots[degree + 1 :] - x) / right_denominator) * basis[..., 1:]
            basis = left + right
        return basis

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        base = tf.einsum("...i,io->...o", tf.nn.silu(inputs), self.base_kernel)
        spline = tf.einsum("...ik,iok->...o", self._basis(inputs), self.spline_kernel)
        return base + spline + self.bias

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "grid_size": self.grid_size,
                "spline_order": self.spline_order,
                "grid_min": self.grid_min,
                "grid_max": self.grid_max,
            }
        )
        return config


@register_keras_serializable(package="TrafficFlow")
class GatedSelfAttentionBlock(Layer):
    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.embed_dim // self.num_heads,
            output_shape=self.embed_dim,
        )
        self.gate = Dense(self.embed_dim, activation="sigmoid")
        self.dropout = Dropout(self.dropout_rate)
        self.normalisation = LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        """Build all stateful sublayers before model deserialisation."""
        self.attention.build(input_shape, input_shape)
        self.gate.build(input_shape)
        self.normalisation.build(input_shape)
        super().build(input_shape)

    def call(self, inputs, training=None):
        attended = self.attention(inputs, inputs, training=training)
        attended = self.dropout(attended, training=training)
        gated = attended * self.gate(inputs)
        return self.normalisation(inputs + gated)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout_rate,
            }
        )
        return config


def get_gsa_kan(
    units,
    input_features: int = 1,
    embed_dim: int = 64,
    num_heads: int = 4,
    grid_size: int = 8,
    dropout: float = 0.2,
) -> Model:
    """Gated self-attention over a genuine B-spline KAN representation."""
    time_steps = int(units[0])
    output_dim = int(units[3])
    inputs = Input(shape=(time_steps, input_features), name="gsa_kan_input")
    x = BSplineKANLayer(embed_dim, grid_size=grid_size, name="spline_kan")(inputs)
    x = Activation("gelu")(x)
    x = GatedSelfAttentionBlock(embed_dim, num_heads, dropout, name="gated_attention")(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(output_dim, activation="sigmoid", name="scaled_flow")(x)
    return Model(inputs=inputs, outputs=outputs, name="GSA_KAN")


@register_keras_serializable(package="TrafficFlow")
class DeltaRelaxLSTMCell(Layer):
    """LSTM cell whose hidden and cell states decay with the observed time gap."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.state_size = [self.units, self.units]
        self.output_size = self.units

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        if input_dim < 2:
            raise ValueError("DeltaRelaxLSTMCell expects signal features plus a final delta-time feature")
        self.signal_dim = input_dim - 1
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.signal_dim, 4 * self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=(self.units, 4 * self.units),
            initializer="orthogonal",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(4 * self.units,),
            initializer="zeros",
            trainable=True,
        )
        # softplus(raw_alpha) guarantees a positive relaxation rate.
        self.raw_alpha = self.add_weight(
            name="raw_alpha",
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(-2.2521685),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, states):
        previous_hidden, previous_cell = states
        signal = inputs[..., : self.signal_dim]
        delta = tf.maximum(inputs[..., self.signal_dim : self.signal_dim + 1], 0.0)
        alpha = tf.nn.softplus(tf.cast(self.raw_alpha, inputs.dtype))[None, :]
        decay = tf.exp(-alpha * delta)
        relaxed_hidden = previous_hidden * decay
        relaxed_cell = previous_cell * decay

        gates = (
            tf.matmul(signal, self.kernel)
            + tf.matmul(relaxed_hidden, self.recurrent_kernel)
            + self.bias
        )
        input_gate, forget_gate, candidate, output_gate = tf.split(gates, 4, axis=1)
        input_gate = tf.sigmoid(input_gate)
        forget_gate = tf.sigmoid(forget_gate)
        candidate = tf.tanh(candidate)
        output_gate = tf.sigmoid(output_gate)
        cell = forget_gate * relaxed_cell + input_gate * candidate
        hidden = output_gate * tf.tanh(cell)
        return hidden, [hidden, cell]

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def get_delta_relax_lstm(units, signal_features: int = 1, dropout: float = 0.2) -> Model:
    time_steps = int(units[0])
    output_dim = int(units[3])
    inputs = Input(shape=(time_steps, signal_features + 1), name="delta_relax_input")
    x = tf.keras.layers.RNN(DeltaRelaxLSTMCell(int(units[2])), name="delta_relax_lstm")(inputs)
    x = Dropout(dropout)(x)
    outputs = Dense(output_dim, activation="sigmoid", name="scaled_flow")(x)
    return Model(inputs=inputs, outputs=outputs, name="DeltaRelaxLSTM")

