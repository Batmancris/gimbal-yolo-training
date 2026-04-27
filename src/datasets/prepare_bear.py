from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_ID = 0


def import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("cv2 is required for dataset image conversion. Install it with: pip install opencv-python") from exc
    return cv2


@dataclass(frozen=True)
class Sample:
    image_path: Path
    boxes: tuple[tuple[float, float, float, float], ...]
    is_background: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a bear LabelMe dataset into YOLO format.")
    parser.add_argument("--source-root", type=Path, required=True, help="raw dataset root")
    parser.add_argument("--output-root", type=Path, required=True, help="YOLO dataset output root")
    parser.add_argument("--target-train-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260423)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true", help="replace existing images and labels under output root")
    return parser.parse_args()


def image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def labelme_boxes(json_path: Path) -> tuple[tuple[float, float, float, float], ...]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    width = float(data["imageWidth"])
    height = float(data["imageHeight"])
    boxes: list[tuple[float, float, float, float]] = []

    for shape in data.get("shapes", []):
        if shape.get("label") != "bear":
            continue
        points = shape.get("points", [])
        if not points:
            continue
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        x1 = max(0.0, min(xs))
        y1 = max(0.0, min(ys))
        x2 = min(width, max(xs))
        y2 = min(height, max(ys))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            boxes.append((x1, y1, x2, y2))

    if not boxes:
        raise ValueError(f"no valid bear boxes found: {json_path}")
    return tuple(boxes)


def load_samples(source_root: Path) -> tuple[list[Sample], list[Path]]:
    bear_dir = source_root / "bear"
    background_dir = source_root / "background"
    external_test_dir = source_root / "test"

    positives: list[Sample] = []
    for image_path in image_files(bear_dir):
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(f"missing LabelMe json for {image_path.name}: {json_path}")
        positives.append(Sample(image_path=image_path, boxes=labelme_boxes(json_path)))

    backgrounds = [Sample(image_path=path, boxes=tuple(), is_background=True) for path in image_files(background_dir)]
    external_tests = image_files(external_test_dir)
    return positives + backgrounds, external_tests


def split_samples(samples: list[Sample], val_ratio: float, seed: int) -> dict[str, list[Sample]]:
    rng = random.Random(seed)
    positives = [sample for sample in samples if not sample.is_background]
    backgrounds = [sample for sample in samples if sample.is_background]
    rng.shuffle(positives)
    rng.shuffle(backgrounds)

    def split_group(group: list[Sample]) -> tuple[list[Sample], list[Sample]]:
        total = len(group)
        val_count = max(1, round(total * val_ratio)) if total >= 5 else 0
        return group[: total - val_count], group[total - val_count :]

    pos_train, pos_val = split_group(positives)
    bg_train, bg_val = split_group(backgrounds)
    return {"train": pos_train + bg_train, "val": pos_val + bg_val}


def assert_output_can_be_replaced(root: Path, overwrite: bool) -> None:
    if overwrite or not root.exists():
        return
    payload = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != ".gitkeep" and path.name != "data.yaml"
    ]
    if payload:
        raise FileExistsError(f"output root is not empty; pass --overwrite to replace it: {root}")


def reset_output(root: Path, overwrite: bool) -> None:
    assert_output_can_be_replaced(root, overwrite)
    if overwrite:
        for child in ("images", "labels"):
            target = root / child
            if target.exists():
                shutil.rmtree(target)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "images" / "test").mkdir(parents=True, exist_ok=True)


def yolo_lines(boxes: tuple[tuple[float, float, float, float], ...], width: int, height: int) -> list[str]:
    lines: list[str] = []
    for x1, y1, x2, y2 in boxes:
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def write_sample(image: np.ndarray, boxes: tuple[tuple[float, float, float, float], ...], root: Path, split: str, stem: str) -> None:
    cv2 = import_cv2()
    height, width = image.shape[:2]
    image_path = root / "images" / split / f"{stem}.jpg"
    label_path = root / "labels" / split / f"{stem}.txt"
    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError(f"failed to write image: {image_path}")
    label_path.write_text("\n".join(yolo_lines(boxes, width, height)), encoding="utf-8")


def copy_original(sample: Sample, root: Path, split: str) -> None:
    cv2 = import_cv2()
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {sample.image_path}")
    write_sample(image, sample.boxes, root, split, f"{sample.image_path.stem}_orig")


def transform_boxes(
    boxes: tuple[tuple[float, float, float, float], ...],
    matrix: np.ndarray,
    width: int,
    height: int,
) -> tuple[tuple[float, float, float, float], ...]:
    transformed: list[tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in boxes:
        corners = np.array([[x1, y1, 1.0], [x2, y1, 1.0], [x2, y2, 1.0], [x1, y2, 1.0]], dtype=np.float32)
        warped = corners @ matrix.T
        xs = np.clip(warped[:, 0], 0, width)
        ys = np.clip(warped[:, 1], 0, height)
        nx1, ny1, nx2, ny2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        if nx2 - nx1 >= 4 and ny2 - ny1 >= 4:
            transformed.append((nx1, ny1, nx2, ny2))
    return tuple(transformed)


def augment(sample: Sample, rng: random.Random, index: int) -> tuple[np.ndarray, tuple[tuple[float, float, float, float], ...]]:
    cv2 = import_cv2()
    image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {sample.image_path}")
    height, width = image.shape[:2]

    angle = rng.uniform(-8.0, 8.0)
    scale = rng.uniform(0.92, 1.08)
    tx = rng.uniform(-0.04, 0.04) * width
    ty = rng.uniform(-0.04, 0.04) * height
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty

    augmented = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    boxes = transform_boxes(sample.boxes, matrix, width, height)

    if rng.random() < 0.5:
        augmented = cv2.flip(augmented, 1)
        boxes = tuple((width - x2, y1, width - x1, y2) for x1, y1, x2, y2 in boxes)

    alpha = rng.uniform(0.75, 1.25)
    beta = rng.uniform(-35.0, 35.0)
    augmented = cv2.convertScaleAbs(augmented, alpha=alpha, beta=beta)

    if rng.random() < 0.25:
        hsv = cv2.cvtColor(augmented, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= rng.uniform(0.75, 1.25)
        hsv[:, :, 2] *= rng.uniform(0.80, 1.20)
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        augmented = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if rng.random() < 0.15:
        noise = rng.uniform(3.0, 8.0)
        generator = np.random.default_rng(index)
        augmented = np.clip(augmented.astype(np.float32) + generator.normal(0, noise, augmented.shape), 0, 255).astype(np.uint8)

    if rng.random() < 0.10:
        augmented = cv2.GaussianBlur(augmented, (3, 3), 0)

    return augmented, boxes


def create_augmentations(train_samples: list[Sample], root: Path, target_count: int, seed: int) -> int:
    rng = random.Random(seed + 99)
    needed = max(0, target_count - len(train_samples))
    if needed == 0:
        return 0

    positives = [sample for sample in train_samples if not sample.is_background]
    backgrounds = [sample for sample in train_samples if sample.is_background]
    if not positives:
        raise ValueError("no positive train samples available for augmentation")

    augmented_count = 0
    for index in range(needed):
        use_background = backgrounds and rng.random() < 0.18
        sample = rng.choice(backgrounds if use_background else positives)
        image, boxes = augment(sample, rng, index)
        if not sample.is_background and not boxes:
            continue
        write_sample(image, boxes, root, "train", f"{sample.image_path.stem}_aug_{index:04d}")
        augmented_count += 1
    return augmented_count


def write_data_yaml(root: Path) -> None:
    data_yaml = """path: .
train: images/train
val: images/val

names:
  0: bear
"""
    (root / "data.yaml").write_text(data_yaml, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    samples, external_tests = load_samples(source_root)
    if not samples:
        raise FileNotFoundError(f"no bear or background samples found under: {source_root}")

    splits = split_samples(samples, args.val_ratio, args.seed)
    reset_output(output_root, args.overwrite)

    for split, samples_for_split in splits.items():
        for sample in samples_for_split:
            copy_original(sample, output_root, split)

    augmented = create_augmentations(splits["train"], output_root, args.target_train_count, args.seed)
    for path in external_tests:
        shutil.copy2(path, output_root / "images" / "test" / path.name)
    write_data_yaml(output_root)

    counts = {split: len(list((output_root / "images" / split).glob("*.jpg"))) for split in ("train", "val")}
    print(f"source samples: {len(samples)}")
    print(f"splits: train={len(splits['train'])}, val={len(splits['val'])}")
    print(f"augmented train images: {augmented}")
    print(f"final images: train={counts['train']}, val={counts['val']}")
    print(f"separate unlabeled test images: {len(external_tests)}")
    print(f"dataset root: {output_root}")


if __name__ == "__main__":
    main()
