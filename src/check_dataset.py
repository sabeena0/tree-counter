"""Check a YOLO-format dataset before training.

Usage:  python src/check_dataset.py --root data/tree-counting
Reports, per split: image count, box count, boxes per image, missing or empty
labels, malformed lines, image sizes. Also reads data.yaml (classes).
"""
import argparse
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def check_split(root: Path, split: str, nc):
    img_dir, lbl_dir = root / split / "images", root / split / "labels"
    if not img_dir.exists():
        print(f"[{split}] not found ({img_dir})")
        return
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    sizes, classes = Counter(), Counter()
    per_img = []
    no_label = empty = bad = polygons = 0
    for img in images:
        with Image.open(img) as im:
            sizes[im.size] += 1
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            no_label += 1
            continue
        lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
        if not lines:
            empty += 1
        n = 0
        for ln in lines:
            parts = ln.split()
            try:
                cls = int(parts[0])
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                bad += 1
                continue
            if len(vals) > 4:           # polygon (segmentation) label
                polygons += 1
                continue
            if len(vals) != 4 or not all(0 <= v <= 1 for v in vals) \
                    or (nc is not None and not 0 <= cls < nc):
                bad += 1
                continue
            classes[cls] += 1
            n += 1
        per_img.append(n)
    total = sum(per_img)
    avg = total / len(per_img) if per_img else 0
    print(f"[{split}] images={len(images)}  boxes={total}  avg/img={avg:.1f}  "
          f"min={min(per_img, default=0)}  max={max(per_img, default=0)}")
    print(f"         missing labels={no_label}  empty labels={empty}  "
          f"malformed={bad}  polygon lines={polygons}")
    print(f"         classes={dict(classes)}  image sizes={dict(sizes.most_common(3))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset folder with data.yaml")
    root = Path(ap.parse_args().root)
    nc = None
    yml = root / "data.yaml"
    if yml.exists():
        cfg = yaml.safe_load(yml.read_text())
        nc = cfg.get("nc")
        print(f"data.yaml: nc={nc}  names={cfg.get('names')}\n")
    else:
        print("data.yaml not found\n")
    for split in ("train", "valid", "test"):
        check_split(root, split, nc)


if __name__ == "__main__":
    main()