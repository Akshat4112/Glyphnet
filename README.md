# GlyphNet

**Homoglyph domain detection with attention-based Convolutional Neural Networks.**

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2306.10392-b31b1b.svg)](https://arxiv.org/abs/2306.10392)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Akshat4112%2FGlyphnet-yellow.svg)](https://huggingface.co/datasets/Akshat4112/Glyphnet)
[![Website](https://img.shields.io/badge/Website-Project%20Page-2563eb.svg)](https://akshat4112.github.io/Glyphnet/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

GlyphNet detects [**homoglyph**](https://en.wikipedia.org/wiki/IDN_homograph_attack)
phishing domains — fake domains built from characters that look identical to
the real ones (e.g. `facebοοk.com` using a Greek omicron `ο` instead of `o`).
Instead of comparing strings, GlyphNet **renders each domain name to an image**
and classifies it as **real** or **phish** with a CNN, so it catches visual
look-alikes that string methods miss and works even when only the suspicious
domain is available (no real/fake pairing required at inference time).

> 📄 Paper: [GlyphNet: Homoglyph domains dataset and detection using attention-based Convolutional Neural Networks](https://arxiv.org/abs/2306.10392) (Gupta, Tomar & Garg, 2023) · reports **0.93 AUC** on the GlyphNet dataset.

---

## Highlights

- 🖼️ **Vision-based approach** — domains are rendered to 256×256 grayscale images and classified, sidestepping brittle string comparisons.
- 🧠 **Two models** — a plain `SimpleCNN` and an `AttentionCNN` that adds a **CBAM** attention block after each convolution block.
- 🔤 **Novel homoglyph generator** — builds synthetic look-alike domains by substituting one or two characters with Unicode confusables.
- 🤗 **Public dataset** — 1.28M domain pairs / ~2.57M labelled images on the [Hugging Face Hub](https://huggingface.co/datasets/Akshat4112/Glyphnet) with ready-made train/validation/test splits.
- ✅ **Reproducible** — a staged, file-driven pipeline plus a dataset-free smoke-test suite.

## Dataset

The dataset is published on the Hugging Face Hub:
**[`Akshat4112/Glyphnet`](https://huggingface.co/datasets/Akshat4112/Glyphnet)**.

| Config | Rows | Fields |
|--------|------|--------|
| `pairs`  | 1,285,579 | `domain` (genuine), `homoglyphs` (spoofed), `pair_id` |
| `images` | 2,571,158 | `image` (256×256 grayscale), `label` (`real`/`phish`), `text`, `pair_id` |

Both configs share the same **train/validation/test** split (70/20/10), assigned
*per pair* so a genuine domain and its homoglyph never land in different splits.

```python
from datasets import load_dataset

images = load_dataset("Akshat4112/Glyphnet", "images")
train, test = images["train"], images["test"]
print(train[0]["label"], train[0]["image"].size)   # e.g. 0 (256, 256)
```

> Note: the published images are rendered with the **DejaVu Sans** font (Arial,
> used in the paper, is proprietary), so glyph shapes may differ slightly from
> the original figures.

## Installation

```bash
pip install -r requirements.txt
```

Requires **Python 3.10** with **TensorFlow 2.9.1 / Keras 2.9** (pinned — the
attention module relies on version-specific Keras APIs, so avoid bumping these
casually).

Two inputs are **not** committed to the repo and must be supplied under your
data directory (default `../data` when running from `code/`):

- `domains_final.txt` — the real-domain list (copy from [`Backup/domains_final.txt`](Backup/domains_final.txt)).
- `ARIAL.TTF` — the TrueType font used to render domain images.

## Pipeline

The project is a sequence of standalone scripts that communicate through files
under the data directory. Run them **from inside `code/`**, in order; each takes
`--path_data` (default `../data`):

| # | Command | Does |
|---|---------|------|
| 1 | `python dataGeneration.py --path_data ../data`  | Real domains → homoglyph variants → `dataset_final.csv` |
| 2 | `python ImageGeneration.py --path_data ../data` | `dataset_final.csv` → `real/` + `fake/` PNGs (256×256 grayscale) |
| 3 | `python dataSplit.py --path_data ../data`       | Split into `train/` `valid/` `test/` (70/20/10, reproducible) |
| 4 | `python dataPreprocess.py --path_data ../data`  | Reorganize into `final_{train,test,valid}/{real,phish}` |
| 5 | `python train.py --path_data ../data`           | Train `SimpleCNN` + `AttentionCNN`; save models & figures |
| 6 | `python predict.py --path_data ../data --model ../models/model_v1.h5` | Evaluate a saved model |

Training prints evaluation metrics (accuracy, precision, recall, F1, Cohen's
kappa, ROC AUC, confusion matrix) and saves accuracy/loss curves to `figures/`.

To (re)build and publish the Hugging Face dataset from `dataset_final.csv`:

```bash
pip install -r requirements-dataset.txt
cd code
python build_hf_dataset.py --stage build --path_data ../data      # render + save locally
HF_TOKEN=hf_xxx python build_hf_dataset.py --stage push --repo <user>/<repo>
```

## Model architecture

Both models take a `(256, 256, 1)` grayscale image and end in a single sigmoid
(thresholded at 0.5):

- **`SimpleCNN`** — four `Conv2D → MaxPool` blocks (32→64→64→128 filters, with
  batch norm) → `Dense(128)` → `Dense(1, sigmoid)`.
- **`AttentionCNN`** — the same backbone with a **CBAM** block
  (channel attention → spatial attention) after each convolution block.

Attention modules live in [`code/attentionModule.py`](code/attentionModule.py):
`cbam_block` ([CBAM](https://arxiv.org/abs/1807.06521), wired into `AttentionCNN`)
and `se_block` ([Squeeze-and-Excitation](https://arxiv.org/abs/1709.01507)).
Loss is binary cross-entropy; optimizer is RMSprop (lr `1e-4`); training uses
early stopping.

## Repository layout

```
code/               # pipeline scripts (run from inside this dir)
  dataGeneration.py   #   domains -> homoglyph pairs -> dataset_final.csv
  ImageGeneration.py  #   csv -> real/ + fake/ PNGs
  dataSplit.py        #   split into train/valid/test
  dataPreprocess.py   #   reorganize into final_{train,test,valid}/{real,phish}
  attentionModule.py  #   SE and CBAM attention blocks
  train.py            #   SimpleCNN + AttentionCNN training and evaluation
  predict.py          #   evaluate a saved .h5 model
  build_hf_dataset.py #   build + publish the dataset to the Hugging Face Hub
  archive/            #   older/superseded scripts (not part of the pipeline)
models/             # a couple of representative .h5 models (rest gitignored)
figures/            # accuracy/loss plots (generated at runtime, gitignored)
Backup/             # domains_final.txt (pipeline input)
tests/              # fast, dataset-free smoke tests
data/               # gitignored working dir (created at runtime)
```

## Tests

```bash
pytest tests/
```

Tests skip automatically when their heavier dependencies (TensorFlow, Pillow)
are absent, so the homoglyph-generation and image-rendering checks still run
without a full ML stack.

## Citation

If you use GlyphNet in your research, please cite:

```bibtex
@article{gupta2023glyphnet,
  title   = {GlyphNet: Homoglyph domains dataset and detection using attention-based Convolutional Neural Networks},
  author  = {Gupta, Akshat and Tomar, Laxman Singh and Garg, Ridhima},
  journal = {arXiv preprint arXiv:2306.10392},
  year    = {2023}
}
```

## Acknowledgements

Real domains are sourced from the [Domains Project](https://github.com/tb0hdan/domains),
one of the largest public collections of active domains.

## License

Released under the [MIT License](LICENSE).
