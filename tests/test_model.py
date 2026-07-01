"""Architecture tests for the SimpleCNN / AttentionCNN builders.

These require TensorFlow (2.9.x) and are skipped where it is unavailable.
"""
import pytest

pytest.importorskip("tensorflow")

import numpy as np
import tensorflow as tf  # noqa: E402

import train  # noqa: E402
from attentionModule import attach_attention_module  # noqa: E402


def test_simple_cnn_builds_with_grayscale_input():
    net = train.NeuralNetwork()
    model = net.build_simple_cnn(learning_rate=1e-4)
    assert model.input_shape == (None, 256, 256, 1)
    assert model.output_shape == (None, 1)


def test_attention_cnn_builds_with_grayscale_input():
    net = train.NeuralNetwork()
    model = net.build_attention_cnn(learning_rate=1e-4)
    assert model.input_shape == (None, 256, 256, 1)
    assert model.output_shape == (None, 1)


def test_cbam_block_runs_on_dummy_tensor():
    x = tf.keras.Input(shape=(32, 32, 8))
    out = attach_attention_module(x, attention_module="cbam_block")
    # CBAM is an attention gate: it preserves the feature-map shape.
    assert out.shape.as_list() == [None, 32, 32, 8]


def test_simple_cnn_predicts_probabilities():
    net = train.NeuralNetwork()
    model = net.build_simple_cnn()
    x = np.zeros((2, 256, 256, 1), dtype="float32")
    preds = model.predict(x)
    assert preds.shape == (2, 1)
    assert ((preds >= 0.0) & (preds <= 1.0)).all()  # sigmoid outputs
