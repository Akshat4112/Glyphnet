"""Test that a domain string renders to a 256x256 grayscale PNG."""
import os

import pytest

Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")
ImageFont = pytest.importorskip("PIL.ImageFont")


def render_domain(text, path):
    """Mirror the img() helper used across the pipeline (grayscale, 256x256).

    Falls back to PIL's built-in bitmap font when ARIAL.TTF is unavailable so
    the test does not depend on the (non-committed) font file.
    """
    img = Image.new("L", (256, 256))
    try:
        fnt = ImageFont.truetype("../data/ARIAL.TTF", 28)
    except OSError:
        fnt = ImageFont.load_default()
    d = ImageDraw.Draw(img)
    d.text((0, 128), text, font=fnt, fill=255)
    out = os.path.join(path, text + ".png")
    img.save(out, "PNG")
    return out


def test_render_produces_grayscale_256(tmp_path):
    out = render_domain("google.com", str(tmp_path))
    assert os.path.exists(out)
    with Image.open(out) as im:
        assert im.size == (256, 256)
        assert im.mode == "L"  # grayscale, matching what the CNN now expects
