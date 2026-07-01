# GlyphNet: Homoglyph domains dataset and detection using attention-based Convolutional Neural Networks

GlyphNet detects [homoglyph](https://en.wikipedia.org/wiki/IDN_homograph_attack)
phishing domains (e.g. `facebοοk.com` using Greek letters) by rendering each
domain name to a 256×256 grayscale image and classifying it as **real** or
**fake/phish** with a CNN — a plain `SimpleCNN` and an attention-augmented
`AttentionCNN` (CBAM/SE attention blocks).

[Website](https://akshat4112.github.io/Glyphnet/) | [Paper](https://arxiv.org/abs/2306.10392)

## Repository layout

- `code/` — all pipeline scripts (run from inside this directory).
- `notebooks/` — Jupyter notebooks with exploratory work and a full walkthrough.
- `models/` — a couple of representative saved `.h5` models (other checkpoints
  are gitignored; see `models/.gitignore`).
- `figures/` — accuracy/loss plots.
- `Backup/domains_final.txt` — the real-domain list used as pipeline input.
- `data/` — working directory (gitignored); every stage reads/writes here at runtime.
- `tests/` — fast, dataset-free smoke tests.

## Getting started

```bash
pip install -r requirements.txt
```

Requires **Python 3.10** with **TensorFlow 2.9.1 / Keras 2.9** (pinned; the
attention module relies on version-specific Keras APIs — do not bump casually).

You must supply two inputs that are **not** committed to the repo:

- `<path_data>/domains_final.txt` — copy from `Backup/domains_final.txt`.
- `<path_data>/ARIAL.TTF` — the TrueType font used to render domain images.

## Pipeline

Scripts communicate only through files under the data directory and are run
**from inside `code/`**, in order. Each stage takes `--path_data` (default
`../data`):

```bash
cd code
python dataGeneration.py  --path_data ../data   # domains -> homoglyph pairs -> dataset_final.csv
python ImageGeneration.py --path_data ../data   # dataset_final.csv -> real/ + fake/ PNGs (256x256, grayscale)
python dataSplit.py       --path_data ../data   # real/ + fake/ -> train/ valid/ test/ (70/20/10, reproducible)
python dataPreprocess.py  --path_data ../data   # reorganize into final_{train,test,valid}/{real,phish}
python train.py           --path_data ../data   # train SimpleCNN + AttentionCNN; save models/ + figures/
python predict.py         --path_data ../data --model ../models/model_v1.h5   # evaluate a saved model
```

Training prints evaluation metrics (accuracy, precision, recall, F1, kappa, ROC
AUC, confusion matrix) and saves accuracy/loss curves under `figures/`.

## Tests

```bash
pytest tests/
```

Tests skip automatically when their dependencies (TensorFlow, Pillow) are
unavailable, so the homoglyph/image checks run even without a full ML stack.

## Contribution

Contributions to enhance detection methods and dataset quality are welcome.

## License

This project is under the MIT License.
