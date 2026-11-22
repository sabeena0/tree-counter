"""Stage 1 (alt. dataset) - Convert OAM-TCD (MS-COCO) into tiled YOLO detection data.

OAM-TCD images are 2048x2048 px at 10 cm/px. We cut them into overlapping
640x640 tiles (what YOLOv8 trains on) and turn each COCO box into a YOLO label.

Step 1: run once with --list-categories to see the category ids/names.
Step 2: run again with --category-id <id of the single-tree class>.

Example:
  python src/convert_tcd_to_yolo.py --images tcd/images --train-json tcd/train.json \
         --test-json tcd/test.json --out Dataset --list-categories
  python src/convert_tcd_to_yolo.py --images tcd/images --train-json tcd/train.json \
         --test-json tcd/test.json --out Dataset --category-id 1 --max-images 800

Leakage guard: files are named <oam_id>_<image_id>.tif. Tiles from the same
OAM survey look alike, so the train/valid split is done BY oam_id, not by image.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def load_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:  # some GeoTIFFs need Pillow
        from PIL import Image
        img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return img


def box_of(ann):
    """Return (x1, y1, x2, y2) from a COCO annotation."""
    if ann.get("bbox"):
        x, y, w, h = ann["bbox"]
        return x, y, x + w, y + h
    seg = ann["segmentation"]
    pts = np.concatenate([np.array(p, dtype=float).reshape(-1, 2) for p in seg])
    return pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()


def tile_positions(size, tile, overlap):
    step = tile - overlap
    pos = list(range(0, max(size - tile, 0) + 1, step))
    if pos[-1] + tile < size:
        pos.append(size - tile)
    return pos


def process_split(coco, images_dir, split_dir, category_id, tile, overlap,
                  keep_ratio, min_px, file_filter, empty_keep):
    imgs = {i["id"]: i for i in coco["images"] if file_filter(i["file_name"])}
    anns = defaultdict(list)
    for a in coco["annotations"]:
        if a["category_id"] == category_id and a["image_id"] in imgs:
            anns[a["image_id"]].append(box_of(a))
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    n_tiles = n_boxes = 0
    rng = random.Random(0)
    for iid, info in imgs.items():
        path = Path(images_dir) / Path(info["file_name"]).name
        if not path.exists():
            continue
        img = load_image(path)
        H, W = img.shape[:2]
        boxes = anns[iid]
        for ty in tile_positions(H, tile, overlap):
            for tx in tile_positions(W, tile, overlap):
                lines = []
                for (x1, y1, x2, y2) in boxes:
                    cx1, cy1 = max(x1, tx), max(y1, ty)
                    cx2, cy2 = min(x2, tx + tile), min(y2, ty + tile)
                    w, h = cx2 - cx1, cy2 - cy1
                    full = max((x2 - x1) * (y2 - y1), 1e-6)
                    if w < min_px or h < min_px or (w * h) / full < keep_ratio:
                        continue
                    lines.append("0 %.6f %.6f %.6f %.6f" % (
                        ((cx1 + cx2) / 2 - tx) / tile, ((cy1 + cy2) / 2 - ty) / tile,
                        w / tile, h / tile))
                if not lines and rng.random() > empty_keep:
                    continue  # drop most empty tiles so training isn't all background
                stem = f"{path.stem}_{tx}_{ty}"
                cv2.imwrite(str(split_dir / "images" / f"{stem}.jpg"),
                            img[ty:ty + tile, tx:tx + tile])
                (split_dir / "labels" / f"{stem}.txt").write_text("\n".join(lines))
                n_tiles += 1
                n_boxes += len(lines)
    return n_tiles, n_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder with the .tif images")
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--test-json", required=True)
    ap.add_argument("--out", default="Dataset")
    ap.add_argument("--category-id", type=int)
    ap.add_argument("--list-categories", action="store_true")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--keep-ratio", type=float, default=0.5,
                    help="min fraction of a box that must lie inside the tile")
    ap.add_argument("--min-px", type=float, default=8)
    ap.add_argument("--empty-keep", type=float, default=0.1,
                    help="fraction of tiles with no trees to keep")
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--max-images", type=int, default=0, help="0 = all (use less on small machines)")
    a = ap.parse_args()

    train = json.load(open(a.train_json))
    test = json.load(open(a.test_json))
    if a.list_categories or a.category_id is None:
        print("Categories in train json:")
        for c in train["categories"]:
            n = sum(1 for x in train["annotations"] if x["category_id"] == c["id"])
            print(f"  id={c['id']}  name={c['name']}  annotations={n}")
        print("\nRe-run with --category-id <id of the single-tree class>.")
        return

    names = [i["file_name"] for i in train["images"]]
    surveys = sorted({Path(n).stem.split("_")[0] for n in names})
    random.Random(0).shuffle(surveys)
    n_val = max(1, int(len(surveys) * a.valid_frac))
    val_surveys = set(surveys[:n_val])

    if a.max_images:
        keep = set(random.Random(1).sample(names, min(a.max_images, len(names))))
        base = lambda n: n in keep
    else:
        base = lambda n: True
    sv = lambda n: Path(n).stem.split("_")[0]
    out = Path(a.out)
    kw = dict(category_id=a.category_id, tile=a.tile, overlap=a.overlap,
              keep_ratio=a.keep_ratio, min_px=a.min_px, empty_keep=a.empty_keep)
    for split, coco, filt in (
        ("train", train, lambda n: base(n) and sv(n) not in val_surveys),
        ("valid", train, lambda n: base(n) and sv(n) in val_surveys),
        ("test", test, lambda n: True),
    ):
        t, b = process_split(coco, a.images, out / split, file_filter=filt, **kw)
        print(f"[{split}] tiles={t} boxes={b}")

    # dataset config for YOLO (absolute path so it works from any folder)
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n"
        "nc: 1\nnames: ['tree']\n")
    print(f"Wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()