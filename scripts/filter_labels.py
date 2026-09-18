"""Remove unwanted classes from a downloaded YOLO dataset.

Reads the Roboflow export in ``data/pork_v1`` and writes a filtered copy to
``data/pork_v4class``. The original download is never modified.

What the filter does:

* Converts polygon annotations (which Roboflow mixes into the YOLOv8
  export) into their enclosing boxes, so the output is boxes only.
* Deletes every box whose class id is in ``DROP_CLASS_IDS`` (by default
  class 4, ``wrinkle``). The remaining class ids are unchanged, because
  ``wrinkle`` is the last class, so no renumbering is needed.
* Optionally drops images whose *only* boxes were removed classes. Without
  this, a tray that was labelled only as wrinkled would become an empty
  label file, and the model would be taught it is a defect-free tray.
* Writes a new ``data.yaml`` with the reduced class list and split paths
  that are relative to the yaml file itself, so the dataset works both on
  the laptop and in Colab.

Usage (from the project root, with the virtual environment active)::

    python scripts/filter_labels.py
"""

import shutil
import sys
from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")
DROP_CLASS_IDS = {4}  # 4 = wrinkle
DROP_IMAGES_LEFT_EMPTY = True

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "pork_v1"
OUTPUT_DIR = ROOT / "data" / "pork_v4class"


def polygon_to_box(coords: list[float]) -> list[float]:
    """Convert polygon points to the smallest enclosing YOLO box.

    Args:
        coords: Flat list ``[x1, y1, x2, y2, ...]`` of normalised points
            (0 to 1, as fractions of image width and height).

    Returns:
        ``[x_centre, y_centre, width, height]``, also normalised. Points
        slightly outside the image are clipped to its edges first.
    """
    xs = [min(max(x, 0.0), 1.0) for x in coords[0::2]]
    ys = [min(max(y, 0.0), 1.0) for y in coords[1::2]]
    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
    return [(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min]


def parse_label_line(line: str) -> tuple[int, list[float], bool]:
    """Read one YOLO label line, as either a box or a polygon.

    Roboflow's YOLOv8 export mixes two formats in the same files:

    * box: ``class x_centre y_centre width height`` (5 fields);
    * polygon: ``class x1 y1 x2 y2 ... xn yn`` (at least 3 points).

    Polygons are converted to their enclosing box.

    Args:
        line: One non-blank line from a label file.

    Returns:
        ``(class_id, [x_centre, y_centre, width, height], was_polygon)``.

    Raises:
        ValueError: If the line is neither a valid box nor a valid polygon.
    """
    fields = line.split()
    class_id = int(fields[0])
    values = [float(v) for v in fields[1:]]
    if len(values) == 4:
        return class_id, values, False
    if len(values) >= 6 and len(values) % 2 == 0:
        return class_id, polygon_to_box(values), True
    raise ValueError(f"Not a valid YOLO box or polygon line: {line!r}")


def filter_label_lines(
    lines: list[str], drop_ids: set[int]
) -> tuple[list[str], dict[int, int]]:
    """Remove unwanted classes from one label file and output boxes only.

    Box lines are kept exactly as written. Polygon lines are converted to
    box lines. Blank lines are discarded.

    Args:
        lines: Lines read from one label file.
        drop_ids: Class ids to remove.

    Returns:
        ``(kept_lines, polygons_converted)``, where ``polygons_converted``
        maps class id to the number of kept polygons turned into boxes.

    Raises:
        ValueError: If a line is neither a valid box nor a valid polygon.
    """
    kept = []
    converted: dict[int, int] = {}
    for line in lines:
        if not line.strip():
            continue
        class_id, box, was_polygon = parse_label_line(line)
        if class_id in drop_ids:
            continue
        if was_polygon:
            converted[class_id] = converted.get(class_id, 0) + 1
            kept.append(f"{class_id} " + " ".join(f"{v:.6f}" for v in box))
        else:
            kept.append(line.strip())
    return kept, converted


def filter_split(
    in_split: Path, out_split: Path, drop_ids: set[int], drop_left_empty: bool
) -> dict[str, int]:
    """Filter one split (train, valid or test) into the output folder.

    Args:
        in_split: Input split folder holding ``images`` and ``labels``.
        out_split: Output split folder; created if missing.
        drop_ids: Class ids to remove.
        drop_left_empty: If True, skip images whose boxes were all removed.

    Returns:
        Counts for this split: images kept, images dropped, boxes removed,
        and polygons converted to boxes per class id.

    Raises:
        FileNotFoundError: If an image has no matching label file.
    """
    (out_split / "images").mkdir(parents=True, exist_ok=True)
    (out_split / "labels").mkdir(parents=True, exist_ok=True)
    counts = {"kept": 0, "dropped": 0, "boxes_removed": 0, "polygons": {}}

    for image in sorted((in_split / "images").iterdir()):
        label = in_split / "labels" / f"{image.stem}.txt"
        if not label.exists():
            raise FileNotFoundError(f"No label file for {image.name}")

        original = [ln for ln in label.read_text().splitlines() if ln.strip()]
        kept, converted = filter_label_lines(original, drop_ids)
        counts["boxes_removed"] += len(original) - len(kept)
        for class_id, n in converted.items():
            counts["polygons"][class_id] = counts["polygons"].get(class_id, 0) + n

        # An image that had boxes, but lost all of them, showed only a
        # removed defect. Keeping it would label a defective tray as normal.
        if drop_left_empty and original and not kept:
            counts["dropped"] += 1
            continue

        shutil.copy2(image, out_split / "images" / image.name)
        (out_split / "labels" / label.name).write_text("\n".join(kept) + ("\n" if kept else ""))
        counts["kept"] += 1

    return counts


def write_data_yaml(in_dir: Path, out_dir: Path, drop_ids: set[int]) -> list[str]:
    """Write a ``data.yaml`` for the filtered dataset.

    Split paths are written relative to the yaml file (``train/images``),
    with no ``path`` key, so the dataset folder can be moved or copied to
    Colab without editing the file. Roboflow's original ``../train/images``
    paths are not reused.

    Args:
        in_dir: Original dataset folder (to read the class names).
        out_dir: Filtered dataset folder.
        drop_ids: Class ids that were removed.

    Returns:
        The new class-name list.

    Raises:
        ValueError: If a dropped class is not the last class, because then
            the remaining ids would need renumbering, which this script
            does not do.
    """
    names = yaml.safe_load((in_dir / "data.yaml").read_text())["names"]
    kept_ids = [i for i in range(len(names)) if i not in drop_ids]
    if kept_ids != list(range(len(kept_ids))):
        raise ValueError("Dropped classes must be the highest ids; renumbering is not supported.")

    new_names = [names[i] for i in kept_ids]
    config = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(new_names),
        "names": new_names,
    }
    (out_dir / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return new_names


def main() -> None:
    """Build ``OUTPUT_DIR`` from ``INPUT_DIR`` and print a summary."""
    if not (INPUT_DIR / "data.yaml").exists():
        sys.exit(f"No dataset at {INPUT_DIR}. Run scripts/download_data.py first.")

    # Rebuild from scratch each time, so stale files never survive a rerun.
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    names = yaml.safe_load((INPUT_DIR / "data.yaml").read_text())["names"]
    for split in SPLITS:
        counts = filter_split(
            INPUT_DIR / split, OUTPUT_DIR / split, DROP_CLASS_IDS, DROP_IMAGES_LEFT_EMPTY
        )
        print(
            f"{split}: kept {counts['kept']} images, dropped {counts['dropped']}, "
            f"removed {counts['boxes_removed']} boxes"
        )
        for class_id, n in sorted(counts["polygons"].items()):
            print(f"  converted {n} {names[class_id]} polygons to boxes")

    new_names = write_data_yaml(INPUT_DIR, OUTPUT_DIR, DROP_CLASS_IDS)
    print(f"Classes: {new_names}")
    print(f"Filtered dataset saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
