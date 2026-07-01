"""End-to-end smoke test: render a tiny synthetic dataset and train 1 epoch.

Confirms the fixed training path (grayscale input, generator-driven batch size,
EarlyStopping wired in) executes without error. Requires TensorFlow + Pillow;
skipped where unavailable.
"""
import os

import pytest

pytest.importorskip("tensorflow")
pytest.importorskip("PIL.Image")
wandb = pytest.importorskip("wandb")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import train  # noqa: E402


def _make_images(class_dir, prefix, n):
    os.makedirs(class_dir, exist_ok=True)
    try:
        fnt = ImageFont.truetype("../data/ARIAL.TTF", 28)
    except OSError:
        fnt = ImageFont.load_default()
    for i in range(n):
        img = Image.new("L", (256, 256))
        ImageDraw.Draw(img).text((0, 128), "%s%d" % (prefix, i), font=fnt, fill=255)
        img.save(os.path.join(class_dir, "%s%d.png" % (prefix, i)), "PNG")


def _build_split(root, n_per_class):
    for split in ("train", "valid"):
        _make_images(os.path.join(root, split, "real"), "real", n_per_class)
        _make_images(os.path.join(root, split, "fake"), "fake", n_per_class)


def test_one_epoch_simple_cnn(tmp_path):
    root = str(tmp_path)
    _build_split(root, n_per_class=5)

    wandb.init(mode="disabled")  # SimpleCNN uses WandbCallback; keep it offline

    net = train.NeuralNetwork()
    net.DataGenerator(os.path.join(root, "train"), os.path.join(root, "valid"), batch_size=2)
    config = {"learning_rate": 1e-4, "epochs": 1, "steps_per_epoch": 2, "batch_size": 2}

    # SimpleCNN saves to ../models; run from a temp cwd with that dir present.
    cwd = os.getcwd()
    os.chdir(root)
    try:
        os.makedirs("../models", exist_ok=True)
        net.SimpleCNN(config)
    finally:
        os.chdir(cwd)

    assert net.history is not None
    assert "loss" in net.history.history
