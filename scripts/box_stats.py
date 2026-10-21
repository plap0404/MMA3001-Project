"""Measure the size of annotated boxes, per class.

Object detectors generally do worse on small objects, and shrinking an image for
training makes that worse. Whether that matters here depends on how large the
boxes actually are, which this script measures rather than assumes.

Sizes are reported two ways:

* **As a fraction of the image** (area, width and height, each 0 to 1). This is how
  the labels are stored, and it does not depend on the image size.
* **In pixels at a given inference size** (``--imgsz``, default 640). A detector
  resizes every image to a square of this size before looking at it, so this is the
  size the model actually sees. The common rule of thumb calls an object "small"
  below roughly 32 x 32 pixels, which is the threshold reported here.

The **median** is reported alongside the mean, because a handful of very large
boxes pull a mean upwards and would make a class look larger than it typically is.

Usage (from the project root)::

    python scripts/box_stats.py                       # data/pork_v4class, 640 px
    python scripts/box_stats.py data/pork_v4class --imgsz 960
"""

import argparse
import statistics
from pathlib import Path

import yaml

SPLITS = ("train", "valid", "test")
SMALL_OBJECT_PIXELS = 32  # common convention: below 32 x 32 px counts as "small"


def read_boxes(dataset_dir: Path, splits: tuple[str, ...]) -> dict[int, list[tuple[float, float]]]:
    """Collect every box's normalised width and height, grouped by class.

    Args:
        dataset_dir: Folder holding the dataset splits.
        splits: Split names to read.

    Returns:
        Class id to a list of ``(width, height)`` pairs, each a fraction of the
        image dimension.

    Raises:
        ValueError: If a label line does not have exactly 5 fields, since this
            script expects the filtered, boxes-only dataset.
    """
    boxes: dict[int, list[tuple[float, float]]] = {}
    for split in splits:
        labels_dir = dataset_dir / split / "labels"
        if not labels_dir.exists():
            continue
        for label_file in sorted(labels_dir.glob("*.txt")):
            for line in label_file.read_text().splitlines():
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 5:
                    raise ValueError(
                        f"Expected a 5-field box in {label_file.name}, got: {line!r}"
                    )
                class_id = int(fields[0])
                width, height = float(fields[3]), float(fields[4])
                boxes.setdefault(class_id, []).append((width, height))
    return boxes


def summarise_class(sizes: list[tuple[float, float]], imgsz: int) -> dict:
    """Summarise the box sizes of one class.

    Args:
        sizes: ``(width, height)`` pairs as fractions of the image.
        imgsz: Inference image size in pixels, used to convert to pixels.

    Returns:
        Counts, median and mean area as a fraction of the image, the median box
        in pixels at ``imgsz``, and how many boxes fall below the small-object
        threshold.

    Raises:
        ValueError: If ``sizes`` is empty.
    """
    if not sizes:
        raise ValueError("No boxes for this class.")

    areas = [w * h for w, h in sizes]
    widths_px = [w * imgsz for w, _ in sizes]
    heights_px = [h * imgsz for _, h in sizes]
    small = sum(
        1 for w_px, h_px in zip(widths_px, heights_px)
        if w_px < SMALL_OBJECT_PIXELS and h_px < SMALL_OBJECT_PIXELS
    )

    return {
        "boxes": len(sizes),
        "median_area_frac": statistics.median(areas),
        "mean_area_frac": statistics.fmean(areas),
        "min_area_frac": min(areas),
        "max_area_frac": max(areas),
        "median_w_px": statistics.median(widths_px),
        "median_h_px": statistics.median(heights_px),
        "small_boxes": small,
        "small_fraction": small / len(sizes),
    }


def format_table(summaries: dict[str, dict], imgsz: int) -> str:
    """Render per-class box-size summaries as a Markdown table.

    Args:
        summaries: Class name to the dict returned by :func:`summarise_class`.
        imgsz: Inference image size the pixel columns refer to.

    Returns:
        A Markdown table.
    """
    lines = [
        f"| Class | Boxes | Median area (% of image) | Median box at {imgsz} px "
        f"| Smallest / largest area (%) | Under {SMALL_OBJECT_PIXELS}x{SMALL_OBJECT_PIXELS} px |",
        "|---|---|---|---|---|---|",
    ]
    for name in sorted(summaries):
        s = summaries[name]
        lines.append(
            f"| {name} | {s['boxes']} | {100 * s['median_area_frac']:.2f}% "
            f"| {s['median_w_px']:.0f} x {s['median_h_px']:.0f} px "
            f"| {100 * s['min_area_frac']:.3f}% / {100 * s['max_area_frac']:.1f}% "
            f"| {s['small_boxes']} ({100 * s['small_fraction']:.0f}%) |"
        )
    return "\n".join(lines)


def main() -> None:
    """Print per-class box-size statistics for the dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="data/pork_v4class")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset).resolve()
    names = yaml.safe_load((dataset_dir / "data.yaml").read_text())["names"]
    boxes = read_boxes(dataset_dir, SPLITS)

    summaries = {
        names[class_id]: summarise_class(sizes, args.imgsz)
        for class_id, sizes in sorted(boxes.items())
    }
    print(f"\nBox sizes in {dataset_dir.name} (all splits), at inference size {args.imgsz} px\n")
    print(format_table(summaries, args.imgsz))
    print(
        f"\n'Small' means both sides under {SMALL_OBJECT_PIXELS} px at this inference size."
        "\nMedian is given rather than mean: a few large boxes would otherwise inflate it."
    )


if __name__ == "__main__":
    main()
