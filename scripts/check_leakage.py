"""Check for near-duplicate video frames shared between dataset splits.

The images are frames cut from video, so neighbouring frames of the same
tray can end up on both sides of the train/test split. A model tested on
such frames has effectively seen the answer already, which inflates its
test scores. This script looks for that problem using two independent
kinds of evidence:

1. **Frame numbers.** Filenames such as ``frame_with_red_10200_jpg.rf.x.jpg``
   carry the frame's position in the video. For every valid/test image, the
   train image with the closest frame number is found (the "frame gap").
2. **Visual similarity.** Each image is reduced to a 256-bit difference
   hash (dHash): shrink to 17 x 16 greyscale pixels, then record for each
   pixel whether it is brighter than its right-hand neighbour. Similar
   images give similar hashes. The number of differing bits (the Hamming
   distance, 0 to 256) measures how different two images look.

The hash distances are calibrated against (a) neighbouring train frames
and (b) train frames far apart. On this dataset the two overlap heavily:
every image shows the same conveyor, lighting and tray positions, so the
hash responds to scene layout rather than to the individual tray. Manual
inspection confirmed that the visually closest pairs can be different
trays. The hash is therefore reported for information only; conclusions
rest on frame gaps, confirmed by inspecting pictures.

Outputs (in ``reports/leakage``):

* ``leakage_report.csv`` - one row per valid/test image;
* ``frame_pairs/*.jpg`` - a random sample of valid/test images shown
  beside their nearest train frame by number, for checking by eye
  whether they show the same physical tray.

Usage (from the project root, with the virtual environment active)::

    python scripts/check_leakage.py
"""

import csv
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data" / "pork_v4class"
REPORT_DIR = ROOT / "reports" / "leakage"

EVAL_SPLITS = ("valid", "test")
HASH_SIZE = 16            # hash is HASH_SIZE x HASH_SIZE = 256 bits
NEIGHBOUR_STEPS = 2       # consecutive frames within 2 sampling steps are "neighbours"
FAR_STEPS = 100           # frames 100+ sampling steps apart are "unrelated"
N_FAR_PAIRS = 5000        # random unrelated pairs sampled for calibration
N_PAIR_IMAGES = 20        # random valid/test images saved beside nearest train frame
SEED = 0                  # fixed, so the sampled pairs are repeatable

# Everything before the last number is the "prefix" (a guess at which
# video the frame came from); Roboflow's "_jpg.rf.<hash>" suffix is skipped.
FRAME_RE = re.compile(
    r"^(?P<prefix>.*?)_?(?P<num>\d+)(?:_(?:jpg|jpeg|png))?(?:\.rf\.\w+)?\.\w+$",
    re.IGNORECASE,
)


def parse_frame(filename: str) -> tuple[str, int] | None:
    """Extract the video prefix and frame number from an image filename.

    Args:
        filename: Image file name, e.g. ``frame_with_red_10200_jpg.rf.ab12.jpg``.

    Returns:
        ``(prefix, frame_number)``, e.g. ``("frame_with_red", 10200)``, or
        ``None`` if the name contains no frame number.
    """
    match = FRAME_RE.match(filename)
    if not match:
        return None
    return match["prefix"], int(match["num"])


def dhash(image_path: Path, size: int = HASH_SIZE) -> np.ndarray:
    """Compute a difference hash of an image.

    Args:
        image_path: Path to the image file.
        size: Hash side length; the hash has ``size * size`` bits.

    Returns:
        A flat boolean array of length ``size * size``.
    """
    with Image.open(image_path) as img:
        small = img.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    pixels = np.asarray(small, dtype=np.int16)
    return (pixels[:, 1:] > pixels[:, :-1]).flatten()


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Count differing bits between every hash in ``a`` and every hash in ``b``.

    Args:
        a: Boolean array of shape (n, bits).
        b: Boolean array of shape (m, bits).

    Returns:
        Integer array of shape (n, m).
    """
    return (a[:, None, :] != b[None, :, :]).sum(axis=2)


def nearest_frame(
    frame: tuple[str, int] | None, train_frames: list[tuple[str, int] | None]
) -> tuple[int | None, int | None]:
    """Find the train image with the closest frame number from the same video.

    Args:
        frame: ``(prefix, number)`` of the image being checked, or None.
        train_frames: ``(prefix, number)`` (or None) for every train image.

    Returns:
        ``(index into train_frames, gap in frames)``, or ``(None, None)``
        if there is no train frame with the same prefix.
    """
    if frame is None:
        return None, None
    best_idx, best_gap = None, None
    for idx, other in enumerate(train_frames):
        if other is None or other[0] != frame[0]:
            continue
        gap = abs(other[1] - frame[1])
        if best_gap is None or gap < best_gap:
            best_idx, best_gap = idx, gap
    return best_idx, best_gap


def sampling_step(frames: list[tuple[str, int] | None]) -> int:
    """Estimate how many video frames separate consecutive dataset images.

    Frames are usually extracted at a fixed interval (e.g. every 10th
    frame), so the typical gap between consecutive frame numbers in the
    same video reveals that interval.

    Args:
        frames: ``(prefix, number)`` (or None) for each image.

    Returns:
        The median gap between consecutive frame numbers (at least 1).
    """
    order = sorted(f for f in frames if f is not None)
    gaps = [n2 - n1 for (p1, n1), (p2, n2) in zip(order, order[1:]) if p1 == p2 and n2 > n1]
    return max(1, int(np.median(gaps))) if gaps else 1


def calibrate(
    hashes: np.ndarray,
    frames: list[tuple[str, int] | None],
    step: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure hash distances for neighbouring and for unrelated train frames.

    Args:
        hashes: Train hashes, shape (n, bits).
        frames: ``(prefix, number)`` (or None) for each train image.
        step: Typical frame spacing, from :func:`sampling_step`.
        rng: Random generator, for sampling unrelated pairs repeatably.

    Returns:
        ``(neighbour_distances, unrelated_distances)`` as integer arrays.
    """
    order = sorted(
        (f[0], f[1], i) for i, f in enumerate(frames) if f is not None
    )
    neighbours = [
        int((hashes[i] != hashes[j]).sum())
        for (p1, n1, i), (p2, n2, j) in zip(order, order[1:])
        if p1 == p2 and 0 < n2 - n1 <= NEIGHBOUR_STEPS * step
    ]

    unrelated = []
    known = [i for i, f in enumerate(frames) if f is not None]
    for _ in range(N_FAR_PAIRS * 20):  # attempt cap, so this always ends
        if len(unrelated) >= N_FAR_PAIRS or len(known) < 2:
            break
        i, j = rng.sample(known, 2)
        (p1, n1), (p2, n2) = frames[i], frames[j]
        if p1 != p2 or abs(n1 - n2) >= FAR_STEPS * step:
            unrelated.append(int((hashes[i] != hashes[j]).sum()))

    return np.array(neighbours, dtype=int), np.array(unrelated, dtype=int)


def save_pair(path_a: Path, path_b: Path, caption: str, out_path: Path) -> None:
    """Save two images side by side with a caption, for checking by eye."""
    with Image.open(path_a) as a, Image.open(path_b) as b:
        a, b = a.convert("RGB"), b.convert("RGB").resize(a.size)
        sheet = Image.new("RGB", (a.width * 2, a.height + 30), "white")
        sheet.paste(a, (0, 30))
        sheet.paste(b, (a.width, 30))
    ImageDraw.Draw(sheet).text((5, 8), caption, fill="black")
    sheet.save(out_path, quality=85)


def summarise(name: str, values: np.ndarray) -> str:
    """Return a one-line percentile summary of a set of distances."""
    if values.size == 0:
        return f"{name}: no pairs found"
    p5, p50, p95 = np.percentile(values, [5, 50, 95])
    return (
        f"{name}: n={values.size}, 5th pct={p5:.0f}, "
        f"median={p50:.0f}, 95th pct={p95:.0f}"
    )


def main() -> None:
    """Run the leakage check on ``DATASET_DIR`` and write the report."""
    if not (DATASET_DIR / "train" / "images").exists():
        sys.exit(f"No dataset at {DATASET_DIR}. Run scripts/filter_labels.py first.")

    rng = random.Random(SEED)
    pair_dir = REPORT_DIR / "frame_pairs"
    pair_dir.mkdir(parents=True, exist_ok=True)
    for old in pair_dir.glob("*.jpg"):
        old.unlink()

    train_paths = sorted((DATASET_DIR / "train" / "images").iterdir())
    train_frames = [parse_frame(p.name) for p in train_paths]
    print(f"Hashing {len(train_paths)} train images...")
    train_hashes = np.stack([dhash(p) for p in train_paths])

    prefixes = sorted({f[0] for f in train_frames if f})
    unparsed = sum(f is None for f in train_frames)
    print(f"Video prefixes in train: {prefixes}  (names without a frame number: {unparsed})")

    step = sampling_step(train_frames)
    print(f"Typical spacing between consecutive train frames: {step} video frames")

    neighbours, unrelated = calibrate(train_hashes, train_frames, step, rng)
    print("\nCalibration (hash distance, 0 = identical, 256 = opposite):")
    print("  " + summarise(f"neighbouring frames (gap <= {NEIGHBOUR_STEPS * step})", neighbours))
    print("  " + summarise(f"unrelated frames (gap >= {FAR_STEPS * step})", unrelated))
    print(
        "  (If these two ranges overlap, the hash cannot tell same-tray from\n"
        "   different-tray images; treat hash distances as information only.)"
    )

    rows = []
    for split in EVAL_SPLITS:
        paths = sorted((DATASET_DIR / split / "images").iterdir())
        hashes = np.stack([dhash(p) for p in paths])
        distances = hamming_matrix(hashes, train_hashes)

        for k, path in enumerate(paths):
            frame = parse_frame(path.name)
            f_idx, gap = nearest_frame(frame, train_frames)
            h_idx = int(distances[k].argmin())
            h_dist = int(distances[k, h_idx])
            rows.append({
                "split": split,
                "image": path.name,
                "frame": frame[1] if frame else "",
                "nearest_train_by_frame": train_paths[f_idx].name if f_idx is not None else "",
                "frame_gap": gap if gap is not None else "",
                "nearest_train_by_look": train_paths[h_idx].name,
                "hash_distance": h_dist,
                "_path": path,
                "_frame_match": train_paths[f_idx] if f_idx is not None else None,
            })

    fields = [k for k in rows[0] if not k.startswith("_")]
    with (REPORT_DIR / "leakage_report.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("\nResults:")
    for split in EVAL_SPLITS:
        sub = [r for r in rows if r["split"] == split]
        gaps = np.array([r["frame_gap"] for r in sub if r["frame_gap"] != ""], dtype=int)
        dists = np.array([r["hash_distance"] for r in sub], dtype=int)
        print(f"\n== {split} ({len(sub)} images) ==")
        if gaps.size:
            print(
                f"  nearest train frame is within 1 step: {(gaps <= step).sum()}, "
                f"within 5 steps: {(gaps <= 5 * step).sum()}, "
                f"within 50 steps: {(gaps <= 50 * step).sum()}, "
                f"median gap: {np.median(gaps):.0f} frames"
            )
        print("  " + summarise("nearest train image by look (information only)", dists))

    candidates = [r for r in rows if r["_frame_match"] is not None]
    sample = sorted(rng.sample(candidates, min(N_PAIR_IMAGES, len(candidates))),
                    key=lambda r: (r["split"], r["frame"]))
    for rank, r in enumerate(sample, start=1):
        caption = (
            f"{r['split']} frame {r['frame']} (left)  |  nearest train frame "
            f"{parse_frame(r['_frame_match'].name)[1]} (right)  |  gap {r['frame_gap']} frames"
        )
        save_pair(r["_path"], r["_frame_match"], caption,
                  pair_dir / f"{rank:02d}_{r['split']}_gap{r['frame_gap']}.jpg")

    print(f"\nReport: {REPORT_DIR / 'leakage_report.csv'}")
    print(f"{len(sample)} random frame pairs saved in {pair_dir}")


if __name__ == "__main__":
    main()
