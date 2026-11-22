"""Count trees in a large aerial image using tiled (sliced) YOLOv8 inference.

Why tiling: the model was trained on 640x640 images where trees are a few dozen
pixels wide. A large image is cut into overlapping 640x640 tiles, each tile is
detected separately, and the boxes are merged back into one list.

Usage:
  python src/count_trees.py --weights models/best.pt --image path/to/image.png
"""
import argparse
import json
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def count_trees(image, weights, conf=0.25, tile=640, overlap=0.2, iou=0.7,
                device="cpu"):
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=str(weights),
        confidence_threshold=conf, device=device, image_size=tile)
    return get_sliced_prediction(
        str(image), model,
        slice_height=tile, slice_width=tile,
        overlap_height_ratio=overlap, overlap_width_ratio=overlap,
        perform_standard_pred=False,        # no extra full-image pass
        postprocess_type="NMS",             # drop duplicates in tile overlaps
        postprocess_match_metric="IOU",
        postprocess_match_threshold=iou)    # 0.7: keep touching neighbours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    result = count_trees(a.image, a.weights, a.conf, a.tile, a.overlap, a.iou, a.device)
    preds = result.object_prediction_list
    stem = Path(a.image).stem
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    result.export_visuals(export_dir=str(out), file_name=f"{stem}_pred",
                          hide_labels=True, hide_conf=True, rect_th=1)
    boxes = [{"xyxy": [round(v, 1) for v in p.bbox.to_xyxy()],
              "score": round(float(p.score.value), 3)} for p in preds]
    (out / f"{stem}_pred.json").write_text(json.dumps(
        {"image": str(a.image), "count": len(boxes), "boxes": boxes}, indent=1))
    print(f"Trees detected: {len(boxes)}")
    print(f"Saved: {out / (stem + '_pred.png')}  and  {out / (stem + '_pred.json')}")


if __name__ == "__main__":
    main()