# Results log

## Baseline: YOLOv8s on Roboflow "Tree Counting" (tree-counting-gz3xz v1)

Dataset: 91 source images (73 train / 9 valid / 9 test), expanded to 273 by
the dataset authors' augmentation (3 copies each). No source image is shared
between splits. 640x640, one class (tree), CC BY 4.0.

Training: yolov8s.pt, 62 of 100 epochs (early stopping, patience 20, best at
epoch 42), batch 16, max_det 1000, flips only. Colab T4, about 5.5 minutes.

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Validation | 0.892 | 0.862 | 0.880 | 0.416 |
| Test | 0.897 | 0.817 | 0.876 | 0.395 |

Counting on test (conf 0.25, iou 0.7): 2349 true vs 2502 predicted (+6.5%);
mean absolute error 7.4 trees per image (9.2%).

Caveats: the test set has only 9 distinct source images, so these figures
are noisy. Box tightness (mAP50-95) is limited by overlapping crowns and
imprecise hand-drawn boxes.
## Caveat found after the baseline run

The dataset images appear to contain small red dots (and possibly faint grey
boxes) on or around the labelled trees. If these marks are part of the images,
the model may partly rely on them and the scores above would be optimistic.
This dependence was not measured. The main model is therefore trained on
OAM-TCD (clean images) instead, and the Roboflow baseline is kept only as a
documented comparison.