"""Report image and label counts for a downloaded YOLO dataset.

Counts, for each split (train/valid/test), how many images and label
files are present and how many bounding boxes belong to each class.
Class names are read from the dataset's ``data.yaml``.

Usage (from the project root, with the virtual environment active)::

    python scripts/dataset_stats.py data/pork_v1
"""

import sys
from collections import Counter
from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")


def load_class_names(dataset_dir: Path) -> list[str]:
    """Read the ordered list of class names from ``data.yaml``.

    Args:
        dataset_dir: Folder holding the downloaded dataset.

    Returns:
        Class names in class-id order, so index 0 is class 0.

    Raises:
        SystemExit: If ``data.yaml`` is missing.
    """
    yaml_path = dataset_dir / "data.yaml"
    if not yaml_path.exists():
        sys.exit(f"No data.yaml found in {dataset_dir}")
    with yaml_path.open() as handle:
        return yaml.safe_load(handle)["names"]


def count_split(split_dir: Path) -> tuple[int, int, int, Counter]:
    """Count images, labels, empty label files and boxes per class.

    Args:
        split_dir: Folder for one split, holding ``images`` and ``labels``.

    Returns:
        A tuple of (image count, label-file count, empty label-file count,
        Counter mapping class id to number of boxes).
    """
    images = list((split_dir / "images").glob("*"))
    labels = list((split_dir / "labels").glob("*.txt"))

    boxes: Counter = Counter()
    empty = 0
    for label_file in labels:
        lines = [ln for ln in label_file.read_text().splitlines() if ln.strip()]
        if not lines:
            empty += 1
        for line in lines:
            boxes[int(line.split()[0])] += 1

    return len(images), len(labels), empty, boxes


def main(dataset_dir: Path) -> None:
    """Print a per-split summary of the dataset at ``dataset_dir``."""
    names = load_class_names(dataset_dir)

    for split in SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"{split}: missing")
            continue

        n_images, n_labels, n_empty, boxes = count_split(split_dir)
        print(f"\n== {split} ==")
        print(f"images: {n_images}  labels: {n_labels}  empty labels: {n_empty}")
        for class_id, name in enumerate(names):
            print(f"  {class_id} {name:<16} {boxes.get(class_id, 0)}")


if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "data/pork_v1")
    main(folder.resolve())