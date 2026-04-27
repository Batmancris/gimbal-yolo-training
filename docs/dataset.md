# Dataset Notes

YOLO datasets use one image file and one matching label file per sample:

```text
images/train/example.jpg
labels/train/example.txt
```

Each non-empty label line uses:

```text
class_id x_center y_center width height
```

All box values are normalized to image width and height. A negative sample should still have a matching `.txt` label file, but the file must be empty.

This repository keeps only templates in Git. Raw images, generated labels, cache files, and numpy arrays should stay local.
