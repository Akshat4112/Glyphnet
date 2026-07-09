"""Build and publish the GlyphNet homoglyph dataset to the Hugging Face Hub.

Two configs, each with train/validation/test splits (70/20/10):
  - "pairs":  (domain, homoglyphs, pair_id) rows from dataset_final.csv
  - "images": each domain rendered to a 256x256 grayscale PNG, labelled
              real (genuine) or phish (homoglyph), with the rendered text and
              a pair_id linking the two rows that came from the same source row.

Splits are assigned per PAIR (seeded), so a real domain and its homoglyph always
land in the same split -- this avoids leakage from near-identical pairs being
scattered across train/test.

Images are rendered on the fly inside a generator (parallelized with num_proc),
so the ~2.5M PNGs are never all held on disk at once.

Auth: set HF_TOKEN in the environment (or pass --token). Nothing is hardcoded.

    # build locally (no token):
    cd code && python build_hf_dataset.py --stage build --path_data ../data
    # then upload:
    HF_TOKEN=hf_xxx python build_hf_dataset.py --stage push --repo Akshat4112/Glyphnet
"""
import argparse
import io
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from datasets import (Dataset, DatasetDict, Features, Image as HfImage,
                      ClassLabel, Value)
from huggingface_hub import HfApi, DatasetCard

IMAGE_SIZE = (256, 256)
FONT_SIZE = 28
SEED = 42
SPLITS = [("train", 0.70), ("validation", 0.20), ("test", 0.10)]
STALE_ZIPS = ["train.zip", "valid.zip", "test.zip"]

CARD_PROSE = """
# GlyphNet: Homoglyph Domains Dataset

Data for detecting [homoglyph](https://en.wikipedia.org/wiki/IDN_homograph_attack)
phishing domains (e.g. `facebook.com` spoofed with visually-similar Unicode
characters). Every genuine domain is paired with a synthetically generated
homoglyph variant, and each domain is also rendered to a 256x256 grayscale image
so the task can be tackled as text **or** image classification.

Paper: [arXiv:2306.10392](https://arxiv.org/abs/2306.10392) ·
Code: [github.com/Akshat4112/Glyphnet](https://github.com/Akshat4112/Glyphnet)

## Configs & splits

Both configs share the same `train` / `validation` / `test` split (70/20/10).
Splits are assigned **per pair** (seeded), so a genuine domain and its homoglyph
always fall in the same split -- preventing leakage between near-identical pairs.

| Config | Rows | Fields |
|--------|------|--------|
| `pairs`  | 1,285,579 pairs | `domain` (genuine), `homoglyphs` (spoofed), `pair_id` |
| `images` | 2,571,158 images | `image` (256x256 grayscale PNG), `label` (`real`/`phish`), `text`, `pair_id` |

`pair_id` links the two `images` rows (one `real`, one `phish`) that came from
the same source pair, and matches the `pair_id` in the `pairs` config.

```python
from datasets import load_dataset

pairs  = load_dataset("Akshat4112/Glyphnet", "pairs")
images = load_dataset("Akshat4112/Glyphnet", "images")
train_img = images["train"]        # or "validation" / "test"
```

## How it was generated

Homoglyphs are produced by substituting one or two characters with Unicode
confusables (`code/dataGeneration.py`). Images are rendered with the **DejaVu
Sans** font -- Arial (used in the paper) is proprietary and not redistributed
here, so glyph shapes may differ slightly from the original figures.

## Intended uses & limitations

- Intended for research on homoglyph / IDN-homograph phishing detection.
- Homoglyphs are **synthetic**, not harvested from real attacks, so the
  distribution may not match live phishing in the wild.
- The genuine and spoofed strings within a pair are near-identical; always
  respect the provided splits (or group by `pair_id`) to avoid leakage.
- Rendering font differs from the paper (DejaVu Sans vs Arial).

## Citation

```bibtex
@article{gupta2023glyphnet,
  title   = {GlyphNet: Homoglyph domains dataset and detection using attention-based Convolutional Neural Networks},
  author  = {Gupta, Akshat and Tomar, Laxman Singh and Garg, Ridhima},
  journal = {arXiv preprint arXiv:2306.10392},
  year    = {2023}
}
```

## License

MIT (same as the source repository).
"""


def render(text, font):
    """Render one domain string to a 256x256 grayscale PNG (bytes)."""
    img = Image.new("L", IMAGE_SIZE)
    ImageDraw.Draw(img).text((0, 128), str(text), font=font, fill=255)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def assign_splits(n_pairs):
    """Return an int array (len n_pairs) mapping each pair_id -> split index.

    Seeded shuffle then contiguous 70/20/10 slices, so the assignment is
    reproducible and split membership is by pair (not by row)."""
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n_pairs)
    split_of = np.empty(n_pairs, dtype=np.int8)
    start = 0
    for idx, (_, frac) in enumerate(SPLITS):
        end = n_pairs if idx == len(SPLITS) - 1 else start + int(round(frac * n_pairs))
        split_of[order[start:end]] = idx
        start = end
    return split_of


IMAGE_FEATURES = Features({
    "image": HfImage(),
    "label": ClassLabel(names=["real", "phish"]),
    "text": Value("string"),
    "pair_id": Value("int32"),
})


def image_generator(csv_path, font_path, shards, split_code, split_of_path):
    """Yield real+phish image records for pairs in the given shards that belong
    to split `split_code`. `shards` is a list of (start, end) pair-id ranges."""
    font = ImageFont.truetype(font_path, FONT_SIZE)
    df = pd.read_csv(csv_path)
    split_of = np.load(split_of_path)
    for start, end in shards:
        sub = df.iloc[start:end]
        for pair_id, (real, phish) in zip(range(start, end),
                                          zip(sub["domain"].astype(str),
                                              sub["homoglyphs"].astype(str))):
            if split_of[pair_id] != split_code:
                continue
            yield {"image": {"bytes": render(real, font), "path": None},
                   "label": "real", "text": real, "pair_id": pair_id}
            yield {"image": {"bytes": render(phish, font), "path": None},
                   "label": "phish", "text": phish, "pair_id": pair_id}


def build_pairs_dd(csv_path, split_of):
    df = pd.read_csv(csv_path)
    df["pair_id"] = np.arange(len(df), dtype=np.int32)
    dd = DatasetDict()
    for code, (name, _) in enumerate(SPLITS):
        sub = df[split_of == code]
        dd[name] = Dataset.from_pandas(sub, preserve_index=False)
    return dd


def build_images_dd(csv_path, font_path, split_of_path, n_pairs, num_proc):
    bounds = [round(i * n_pairs / num_proc) for i in range(num_proc + 1)]
    shards = list(zip(bounds[:-1], bounds[1:]))
    dd = DatasetDict()
    for code, (name, _) in enumerate(SPLITS):
        print("  rendering split '%s'" % name)
        dd[name] = Dataset.from_generator(
            image_generator,
            features=IMAGE_FEATURES,
            gen_kwargs={"csv_path": csv_path, "font_path": font_path,
                        "shards": shards, "split_code": code,
                        "split_of_path": split_of_path},
            num_proc=num_proc,
        )
    return dd


def update_card(repo, token):
    """Set metadata (license/tags/pretty_name) and append prose, keeping the
    auto-generated dataset_info/configs YAML intact."""
    card = DatasetCard.load(repo, repo_type="dataset", token=token)
    card.data.license = "mit"
    card.data.pretty_name = "GlyphNet Homoglyph Domains"
    card.data.tags = sorted(set((card.data.tags or []) +
                                ["homoglyph", "phishing", "idn-homograph",
                                 "cybersecurity", "domain-spoofing", "security"]))
    if "GlyphNet: Homoglyph Domains Dataset" not in card.text:
        card.text = card.text.rstrip() + "\n\n" + CARD_PROSE.strip() + "\n"
    card.push_to_hub(repo, repo_type="dataset", token=token)


def remove_stale_zips(repo, token):
    api = HfApi(token=token)
    existing = set(api.list_repo_files(repo, repo_type="dataset"))
    for z in STALE_ZIPS:
        if z in existing:
            api.delete_file(z, repo_id=repo, repo_type="dataset",
                            commit_message="Remove stale %s (superseded by parquet configs)" % z)
            print("  deleted", z)


def main():
    parser = argparse.ArgumentParser(description="Build/publish the GlyphNet dataset for the HF Hub.")
    parser.add_argument("--path_data", default="../data")
    parser.add_argument("--work_dir", default="../data/hf_build")
    parser.add_argument("--stage", choices=["build", "push", "all"], default="all")
    parser.add_argument("--repo", help="target HF dataset repo id (required for push)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--max_shard_size", default="500MB")
    parser.add_argument("--num_proc", type=int, default=os.cpu_count())
    parser.add_argument("--skip_images", action="store_true")
    parser.add_argument("--keep_zips", action="store_true", help="do not delete stale zips on push")
    args = parser.parse_args()

    csv_path = os.path.join(args.path_data, "dataset_final.csv")
    font_path = os.path.join(args.path_data, "ARIAL.TTF")
    pairs_dir = os.path.join(args.work_dir, "pairs")
    images_dir = os.path.join(args.work_dir, "images")
    split_of_path = os.path.join(args.work_dir, "split_of.npy")

    if args.stage in ("build", "all"):
        os.makedirs(args.work_dir, exist_ok=True)
        n_pairs = len(pd.read_csv(csv_path, usecols=["domain"]))
        split_of = assign_splits(n_pairs)
        np.save(split_of_path, split_of)
        counts = {name: int((split_of == code).sum()) for code, (name, _) in enumerate(SPLITS)}
        print("== BUILD: %d pairs, split (by pair):" % n_pairs, counts)
        build_pairs_dd(csv_path, split_of).save_to_disk(pairs_dir)
        if not args.skip_images:
            print("== BUILD: rendering %d images across %d procs" % (2 * n_pairs, args.num_proc))
            build_images_dd(csv_path, font_path, split_of_path, n_pairs, args.num_proc).save_to_disk(images_dir)
        print("Build complete ->", args.work_dir)

    if args.stage in ("push", "all"):
        if not args.token:
            raise SystemExit("No HF token: pass --token or set HF_TOKEN (needed for push).")
        if not args.repo:
            raise SystemExit("No --repo given (needed for push).")
        from datasets import load_from_disk
        print("== PUSH: pairs ->", args.repo)
        load_from_disk(pairs_dir).push_to_hub(args.repo, config_name="pairs", private=args.private,
                                              token=args.token, max_shard_size=args.max_shard_size)
        if not args.skip_images:
            print("== PUSH: images ->", args.repo)
            load_from_disk(images_dir).push_to_hub(args.repo, config_name="images", private=args.private,
                                                   token=args.token, max_shard_size=args.max_shard_size)
        if not args.keep_zips:
            print("== PUSH: removing stale zips")
            remove_stale_zips(args.repo, args.token)
        print("== PUSH: updating dataset card")
        update_card(args.repo, args.token)
        print("Done ->", "https://huggingface.co/datasets/" + args.repo)


if __name__ == "__main__":
    main()
