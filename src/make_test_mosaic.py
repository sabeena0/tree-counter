"""Stitch one copy of each distinct test photo into a big image (3x3 grid).

The true tree count of the mosaic is the sum of the individual label counts, so
it is a controlled check of the tiled-inference pipeline on a large image.

Usage:  python src/make_test_mosaic.py --root data/tree-counting
"""
import argparse
import re
from pathlib import Path

from PIL import Image

EXT = {".jpg", ".jpeg", ".png"}


def source_id(name):
    """Roboflow names copies <source>_jpg.rf.<hash>.jpg; strip the suffix."""
    return re.split(r"_(?:jpg|jpeg|png)\.rf\.", name)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/tree-counting")
    ap.add_argument("--split", default="test")
    ap.add_argument("--grid", type=int, default=3)
    ap.add_argument("--out", default="outputs/test_mosaic.png")
    a = ap.parse_args()

    img_dir = Path(a.root) / a.split / "images"
    lbl_dir = Path(a.root) / a.split / "labels"
    chosen = {}
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() in EXT:
            chosen.setdefault(source_id(p.name), p)   # first copy per source
    paths = list(chosen.values())[: a.grid ** 2]

    w, h = Image.open(paths[0]).size
    canvas = Image.new("RGB", (w * a.grid, h * a.grid))
    total = 0
    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize((w, h))
        canvas.paste(im, ((i % a.grid) * w, (i // a.grid) * h))
        lbl = lbl_dir / (p.stem + ".txt")
        total += sum(1 for ln in lbl.read_text().splitlines() if ln.strip())

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"Mosaic {canvas.size[0]}x{canvas.size[1]} from {len(paths)} distinct "
          f"photos; true tree count = {total}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()