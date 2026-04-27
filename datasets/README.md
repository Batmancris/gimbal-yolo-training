# Datasets

Keep only dataset templates and placeholder files in Git.

Expected YOLO layout:

```text
dataset_name/
  data.yaml
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

For negative samples, keep the matching label file empty. Do not commit raw images, generated caches, or numpy arrays.
