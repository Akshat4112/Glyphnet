"""Build and publish the GlyphNet homoglyph dataset to the Hugging Face Hub.

Produces two configs in one dataset repo:
  - "pairs":  the (domain, homoglyphs) table from dataset_final.csv
  - "images": each domain rendered to a 256x256 grayscale PNG, labelled
              real (the genuine domain) or phish (its homoglyph variant),
              with the rendered text kept alongside.

Images are rendered on the fly inside a generator, so the ~2.5M PNGs are never
all held on disk at once -- `datasets` streams them straight into parquet shards.

Auth: set HF_TOKEN in the environment (or pass --token). Nothing is hardcoded.

Example:
    export HF_TOKEN=hf_xxx
    cd code
    python build_hf_dataset.py --path_data ../data --repo akshat4112/glyphnet
"""
import argparse
import io
import os

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from datasets import Dataset, Features, Image as HfImage, ClassLabel, Value
from huggingface_hub import HfApi

IMAGE_SIZE = (256, 256)
FONT_SIZE = 28

CARD_PROSE = """
# GlyphNet homoglyph domains dataset

Data for detecting [homoglyph](https://en.wikipedia.org/wiki/IDN_homograph_attack)
phishing domains (e.g. `facebook.com` spoofed with visually-similar Unicode
characters). Real domains are paired with a synthetically generated homoglyph
variant, and every domain is also rendered to a 256x256 grayscale image so the
problem can be tackled as image classification.

See the [GlyphNet paper](https://arxiv.org/abs/2306.10392) and
[code](https://github.com/Akshat4112/Glyphnet).

## Configs

- **pairs** - table of `domain` (genuine) and `homoglyphs` (spoofed variant).
- **images** - each domain rendered to a 256x256 grayscale PNG, with
  `label` (`real` / `phish`) and the rendered `text`.

```python
from datasets import load_dataset
pairs  = load_dataset("{repo}", "pairs")
images = load_dataset("{repo}", "images")
```

## Generation

Homoglyphs are produced by substituting one or two characters with Unicode
confusables (see `code/dataGeneration.py`). Images are rendered with the DejaVu
Sans font (Arial is proprietary and not redistributed here), so glyph shapes may
differ slightly from the original paper.

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


def image_generator(csv_path, font_path):
    """Yield one real + one phish image record per CSV row."""
    font = ImageFont.truetype(font_path, FONT_SIZE)
    df = pd.read_csv(csv_path)
    for real, phish in zip(df["domain"].astype(str), df["homoglyphs"].astype(str)):
        yield {"image": {"bytes": render(real, font), "path": None},
               "label": "real", "text": real}
        yield {"image": {"bytes": render(phish, font), "path": None},
               "label": "phish", "text": phish}


def main():
    parser = argparse.ArgumentParser(description="Publish the GlyphNet dataset to the HF Hub.")
    parser.add_argument("--path_data", default="../data", help="dir containing dataset_final.csv and ARIAL.TTF")
    parser.add_argument("--repo", required=True, help="target HF dataset repo id, e.g. akshat4112/glyphnet")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (or set HF_TOKEN)")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    parser.add_argument("--max_shard_size", default="500MB")
    parser.add_argument("--skip_images", action="store_true", help="upload only the pairs config")
    args = parser.parse_args()

    if not args.token:
        raise SystemExit("No HF token: pass --token or set HF_TOKEN in the environment.")

    csv_path = os.path.join(args.path_data, "dataset_final.csv")
    font_path = os.path.join(args.path_data, "ARIAL.TTF")

    # --- config: pairs (the domain/homoglyph table) ---
    print("Uploading 'pairs' config from", csv_path)
    pairs = Dataset.from_pandas(pd.read_csv(csv_path), preserve_index=False)
    pairs.push_to_hub(args.repo, config_name="pairs", private=args.private,
                      token=args.token, max_shard_size=args.max_shard_size)

    # --- config: images (rendered PNGs, labelled real/phish) ---
    if not args.skip_images:
        print("Rendering + uploading 'images' config (this streams; no bulk PNGs on disk)")
        features = Features({
            "image": HfImage(),
            "label": ClassLabel(names=["real", "phish"]),
            "text": Value("string"),
        })
        images = Dataset.from_generator(
            image_generator,
            features=features,
            gen_kwargs={"csv_path": csv_path, "font_path": font_path},
        )
        images.push_to_hub(args.repo, config_name="images", private=args.private,
                           token=args.token, max_shard_size=args.max_shard_size)

    update_card(args.repo, args.token)
    print("Done ->", "https://huggingface.co/datasets/" + args.repo)


def update_card(repo, token):
    """Append the human-readable prose to the auto-generated dataset card.

    push_to_hub writes a README.md with YAML front matter (configs / features);
    we keep that intact and append the prose below it, only if not already there.
    """
    api = HfApi(token=token)
    path = api.hf_hub_download(repo_id=repo, repo_type="dataset", filename="README.md")
    with open(path, encoding="utf-8") as fh:
        current = fh.read()
    if "GlyphNet homoglyph domains dataset" in current:
        return  # prose already present
    combined = current.rstrip() + "\n\n" + CARD_PROSE.strip().format(repo=repo) + "\n"
    api.upload_file(path_or_fileobj=combined.encode("utf-8"), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")


if __name__ == "__main__":
    main()
