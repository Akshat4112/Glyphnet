# CLAUDE.md

Guidance for AI assistants (and humans) working in this repository.

## What this project is

**GlyphNet** detects [homoglyph](https://en.wikipedia.org/wiki/IDN_homograph_attack)
phishing domains (e.g. `facebοοk.com` using a Greek omicron) by treating each
domain name as an **image** and classifying it with a CNN. The pipeline
generates synthetic homoglyph variants of real domains, renders every domain to
a 256×256 grayscale PNG, and trains a binary classifier — a plain CNN
(`SimpleCNN`) and an attention-augmented CNN (`AttentionCNN`, using CBAM/SE
attention blocks) — to distinguish **real** from **fake/phish** domains.

- Paper: https://arxiv.org/abs/2306.10392
- Site: https://akshat4112.github.io/Glyphnet/
- License: MIT

This is a **research codebase**, not a packaged library or service. It is a
collection of standalone scripts run in sequence, coordinated by a shared
on-disk `data/` directory rather than by imports. There is no test suite, no
setup.py/pyproject, and no CLI entry point beyond running individual scripts.

## Repository layout

```
code/               # All source scripts (run from inside this dir)
  dataGeneration.py   # real domains -> homoglyph pairs -> dataset_final.csv
  ImageGeneration.py  # dataset_final.csv -> real/ + fake/ PNGs (256x256)
  dataSplit.py        # real/ + fake/ -> train/ valid/ test/ (70/20/10)
  dataPreprocess.py   # reorganize split dirs into final_{train,test,valid}/{real,phish}
  attentionModule.py  # SE and CBAM attention blocks (imported by train.py)
  train.py            # trains SimpleCNN + AttentionCNN; prints metrics; saves models/figures
  predict.py          # loads a saved .h5 model and evaluates on final_test
  SingleImage.py      # renders one domain string to a single PNG (ad hoc utility)
  archive/            # older/superseded scripts (Streamlit app, Flask api, baselines) - not part of the current pipeline
models/             # Saved Keras .h5 models (generated artifacts, checked in)
figures/            # Saved accuracy/loss plots per run (generated at runtime, gitignored)
Backup/domains_final.txt  # ~22 MB list of real domains, one per line (pipeline input)
data/               # gitignored working dir - all scripts read/write here at runtime
Dockerfile          # Ubuntu 20.04 + pip install; runs code/train.py
requirements.txt    # pinned deps (TF 2.9.1, Keras 2.9.0, etc.)
```

## The end-to-end pipeline

Scripts communicate **only through files under `data/`**. There is no shared
config or orchestrator — you run each stage manually, in order. Scripts are
launched **from inside `code/`** so their hardcoded `../data/`, `../models/`,
and `../figures/` relative paths resolve correctly.

Most scripts accept a `--path_data <dir>` argument pointing at the working data
directory (e.g. `../data`). Note the inconsistency: some scripts use
`--path_data`, while `dataSplit.py` and parts of `train.py`/`predict.py`
hardcode `../data/...` paths. Check the top of each script before running.

1. **`dataGeneration.py --path_data ../data`**
   Reads `<path_data>/domains_final.txt` (copy `Backup/domains_final.txt` here),
   takes the first 2,000,000 domains, and for each generates a homoglyph variant
   by substituting one (`homo_gen_1`) or two (`homo_gen_2`) characters using the
   `glyphs` lookup table (maps each ASCII char to visually-similar Unicode
   confusables). Writes `data/dataset_final.csv` with columns
   `domain,homoglyphs`. Uses `multiprocessing`; failures are appended to an
   `errors.txt`.

2. **`ImageGeneration.py --path_data ../data`**
   Reads `dataset_final.csv` and renders each real domain and each homoglyph to a
   256×256 grayscale (`L` mode) PNG using `ARIAL.TTF` at size 28. Outputs to
   `data/real/` and `data/fake/`. **`data/ARIAL.TTF` must exist** — it is not in
   the repo and must be supplied. Heavily parallelized via `multiprocessing`.

3. **`dataSplit.py`** (hardcodes `BASE_PATH = '../data/'`)
   Splits `real/` and `fake/` into `train/`, `valid/`, `test/` at **70/20/10**,
   each with `real/` and `fake/` subdirs, by moving files.

4. **`dataPreprocess.py --path_data ../data`**
   Reorganizes split image dirs (`domain_pics/*`, `fake_pics/*`) into
   `final_train/`, `final_test/`, `final_valid/`, each with `real/` and `phish/`
   class subdirectories — the layout Keras `flow_from_directory` consumes.
   (Directory names here differ from stage 3's output; reconcile paths to your
   actual data layout before running.)

5. **`train.py --path_data ../data`**
   The core training script. Defines a `NeuralNetwork` class with:
   - `DataGenerator` — Keras `ImageDataGenerator(rescale=1./255)` +
     `flow_from_directory`, `class_mode='binary'`, `color_mode='grayscale'`,
     `target_size=(256,256)`, batch size set here.
   - `SimpleCNN(config)` — 4 conv blocks → dense → sigmoid.
   - `AttentionCNN(config)` — same backbone with a CBAM attention block after
     each conv block (via `attach_attention_module`).
   - `plotGraphs` — saves acc/loss curves to `../figures/`.
   - `Evaluation` — computes accuracy/precision/recall/F1/kappa/AUC/confusion
     matrix on the test set.
   Runs both models, printing evaluation metrics to stdout. Saves models
   to `../models/modelSimpleCNN<timestamp>.h5` and
   `../models/modelAttentionCNN<timestamp>.h5`.

6. **`predict.py --path_data ../data`**
   Loads a saved model (`../models/model_v1.h5`) and runs it over
   `final_test/` via `flow_from_directory`.

`SingleImage.py` is a one-off helper to render an individual domain string to an
image for manual inspection.

## Model architecture notes

- Input: `(256, 256, 3)`, pixel values rescaled to `[0,1]`.
- `SimpleCNN`: `Conv2D(32,5x5) → MaxPool → BN → Conv2D(64,3x3) → MaxPool → BN →
  Conv2D(64,3x3) → MaxPool → Conv2D(128,3x3) → MaxPool → Flatten → Dense(128) →
  Dense(1, sigmoid)`.
- `AttentionCNN`: identical backbone, with a **CBAM** block after each conv block.
- `attentionModule.py` implements two attention mechanisms:
  - `se_block` — Squeeze-and-Excitation (https://arxiv.org/abs/1709.01507).
  - `cbam_block` — Convolutional Block Attention Module = channel attention then
    spatial attention (https://arxiv.org/abs/1807.06521). This is the one wired
    into `AttentionCNN`.
- Loss: `binary_crossentropy`; optimizer: `RMSprop(lr=1e-4)`; metric: `acc`.
- `EarlyStopping(monitor='loss', patience=3)` is passed into `fit`.
- Images are rendered grayscale and loaded as **1 channel** (`color_mode='grayscale'`,
  input shape `(256,256,1)`). The batch size is set on the generator in
  `DataGenerator`, not on `fit` (where it is ignored for iterators).
- `Evaluation`/`predict.py` threshold the single sigmoid output at `0.5`
  (`(preds > 0.5)`), not `argmax` — argmax over a 1-element row is always 0.
- Model construction is separated from training: `build_simple_cnn()` /
  `build_attention_cnn()` return a compiled model; `SimpleCNN()`/`AttentionCNN()`
  build then fit. Importing `train`/`dataGeneration` no longer runs the pipeline
  (guarded by `if __name__ == "__main__"`).

## Environment & conventions

- **Python 3.10**, TensorFlow **2.9.1** / Keras **2.9.0** (pinned in
  `requirements.txt`). This is an older TF stack — do not casually bump versions;
  `attentionModule.py` mixes the modern `.shape` API with the legacy
  `._keras_shape` API (in `se_block`), so it is version-sensitive.
- Install: `pip install -r requirements.txt`.
- No experiment-tracking service. `train.py` prints evaluation metrics to stdout
  and `plotGraphs` writes accuracy/loss curves to `figures/`. (Weights & Biases
  was previously used and has been removed.)
- Coding style is research-grade and inconsistent (mixed tabs/spaces across
  files). **Match the style of the file you are editing** rather than
  reformatting the whole file.
- `models/*.h5` and `figures/*.png` are generated artifacts. Only a couple of
  representative models are tracked (`model_v1.h5`, `30epochs_model.h5`); the rest
  are gitignored via `models/.gitignore`. `figures/` is gitignored entirely
  (kept as a directory via `figures/.gitignore`). Treat them as data — don't hand-edit.
- `data/` and `code/wandb/` are gitignored; large inputs/outputs and W&B run logs
  live there and are never committed.

## Known rough edges (do not "fix" silently)

- Inter-stage directory names still differ by design (`real`/`fake` →
  `domain_pics`/`fake_pics` → `final_*`); all stages now take `--path_data`
  (default `../data`), but trace the actual subdir layout for the stage you run.
- `ARIAL.TTF` is a required, non-committed input expected at
  `<path_data>/ARIAL.TTF` — rendering scripts fail without it.
- `attentionModule.se_block` still uses the legacy `._keras_shape` API and is
  unused (only `cbam_block` is wired in). Leave it unless explicitly asked.
- `code/archive/` holds superseded code (a Streamlit `app.py`, a Flask-style
  `api.py`, baseline CNN, and a `get_data.py` that pulls a spoof-domain pickle
  from the endgameinc/homoglyph repo). It is **not** part of the current pipeline
  — don't wire it in without asking.

## Working agreements for AI assistants

- When adding a new pipeline stage, follow the existing pattern: a standalone
  script under `code/`, run from `code/`, reading/writing under `../data/`,
  taking `--path_data` where practical.
- Don't introduce a new dependency without adding it (pinned) to
  `requirements.txt`.
- Don't commit anything under `data/`; don't regenerate `models/` or `figures/`
  unless that is the explicit task.
- If a change touches the TF/Keras API surface, verify it against the pinned
  2.9.x versions rather than current TensorFlow.

## Git workflow

- Active development branch for AI-assisted work: `claude/claude-md-docs-slms70`.
- Commit with clear messages; push with `git push -u origin <branch>`.
- Do **not** open a pull request unless explicitly asked.
