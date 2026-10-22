"""Measure how fast a trained model runs, and how much arithmetic it costs.

The engineering question this answers is whether the method could keep up with a
production line. Accuracy alone does not settle that: a model that is accurate but
takes a second per tray is useless on a line moving a tray every 300 ms.

Three things are measured:

* **Latency** - milliseconds to process one image, split into preprocessing,
  the model itself (inference), and post-processing (turning raw model output
  into a tidy list of boxes, which includes discarding overlapping duplicates).
* **Throughput** - images per second, and trays per minute, derived from the
  median latency.
* **Compute cost** - the model's parameter count and its GFLOPs: billions of
  floating-point operations per image. Unlike timing, this does not depend on
  which machine you run on, so it is the fair way to compare two models' cost.

**Timing on a GPU needs care.** GPU work is queued and runs asynchronously, so a
naive timer records how long it took to *queue* the work, not to do it. Every
measurement here calls ``torch.cuda.synchronize()`` to wait for the GPU to finish.
Warm-up runs are discarded because the first few passes include one-off costs
(memory allocation, kernel selection) that would inflate the result.

The **median** is reported rather than the mean, because occasional scheduling
delays produce outliers that drag a mean upwards; the 95th percentile is reported
alongside it, since a line's worst case matters more than its average.

Usage (from the project root)::

    python scripts/benchmark.py --weights <path>/best.pt --name baseline_n640
    python scripts/benchmark.py --weights <path>/best.pt --name baseline_n640_cpu \\
                                --device cpu --repeats 20
"""

import argparse
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES = ROOT / "data" / "pork_v4class" / "test" / "images"
REPORT_DIR = ROOT / "reports" / "metrics"


def summarise_times(times_ms: list[float]) -> dict[str, float]:
    """Summarise a list of per-image timings.

    Args:
        times_ms: One measurement per image, in milliseconds. Must not be empty.

    Returns:
        Median, mean, 95th percentile, minimum and maximum latency in
        milliseconds, plus throughput in images per second derived from the
        median.

    Raises:
        ValueError: If ``times_ms`` is empty or contains a non-positive time,
            which would mean the timer failed rather than that the model is fast.
    """
    if not times_ms:
        raise ValueError("No timings recorded.")
    if min(times_ms) <= 0:
        raise ValueError(f"Non-positive timing recorded: {min(times_ms)} ms")

    ordered = sorted(times_ms)
    median = statistics.median(ordered)
    # Nearest-rank 95th percentile: the smallest value at or above 95% of the data.
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
    return {
        "median_ms": median,
        "mean_ms": statistics.fmean(ordered),
        "p95_ms": p95,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "images_per_second": 1000.0 / median,
        "n": len(ordered),
    }


def trays_per_minute(median_ms: float) -> float:
    """Convert median latency into trays processed per minute.

    Args:
        median_ms: Median milliseconds per image.

    Returns:
        Trays per minute, assuming one tray per image and no other bottleneck.
    """
    return 60_000.0 / median_ms


def format_report(name: str, device: str, imgsz: int, stages: dict, info: dict) -> str:
    """Render the benchmark result as a Markdown section.

    Args:
        name: Run name.
        device: Device the timings were taken on.
        imgsz: Inference image size.
        stages: Stage name to the dict returned by :func:`summarise_times`.
        info: Model parameter count and GFLOPs.

    Returns:
        Markdown text ready to paste into the report.
    """
    lines = [
        f"# {name} - speed ({device}, imgsz {imgsz})",
        "",
        f"Model: {info['parameters']:,} parameters, {info['gflops']:.1f} GFLOPs per image.",
        "",
        "| Stage | Median (ms) | Mean (ms) | 95th pct (ms) | Min (ms) | Max (ms) |",
        "|---|---|---|---|---|---|",
    ]
    for stage, s in stages.items():
        lines.append(
            f"| {stage} | {s['median_ms']:.2f} | {s['mean_ms']:.2f} | "
            f"{s['p95_ms']:.2f} | {s['min_ms']:.2f} | {s['max_ms']:.2f} |"
        )
    total = stages["total"]
    lines += [
        "",
        f"Throughput: **{total['images_per_second']:.1f} images/s** "
        f"= **{trays_per_minute(total['median_ms']):.0f} trays/minute** "
        f"(median of {total['n']} timed images, warm-up discarded).",
    ]
    return "\n".join(lines)


def main() -> None:
    """Time one set of weights and write the benchmark report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="path to best.pt")
    parser.add_argument("--name", required=True, help="run name, used for output files")
    parser.add_argument("--imgsz", type=int, default=640, help="must match training")
    parser.add_argument("--device", default="0", help="'0' for the first GPU, or 'cpu'")
    parser.add_argument("--warmup", type=int, default=10, help="discarded initial runs")
    parser.add_argument("--repeats", type=int, default=3, help="passes over the images")
    parser.add_argument("--images", default=str(DEFAULT_IMAGES))
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    image_paths = sorted(Path(args.images).iterdir())
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    from ultralytics.utils.torch_utils import get_flops, get_num_params

    model = YOLO(args.weights)
    # Computed directly, not via model.info(): in Ultralytics 8.4.183
    # model.info(verbose=False) returns None, and model.info() always costs the
    # model at 640 px. get_flops() is given the real inference size instead.
    info = {
        "parameters": int(get_num_params(model.model)),
        "gflops": float(get_flops(model.model, imgsz=args.imgsz)),
    }

    def synchronise() -> None:
        """Wait for queued GPU work to finish, so timings measure real work."""
        if args.device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()

    for path in image_paths[: args.warmup]:
        model.predict(str(path), imgsz=args.imgsz, device=args.device, verbose=False)
    synchronise()

    stage_times: dict[str, list[float]] = {
        "preprocess": [], "inference": [], "postprocess": [], "total": []
    }
    for _ in range(args.repeats):
        for path in image_paths:
            start = time.perf_counter()
            result = model.predict(
                str(path), imgsz=args.imgsz, device=args.device, verbose=False
            )[0]
            synchronise()
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            for stage in ("preprocess", "inference", "postprocess"):
                stage_times[stage].append(float(result.speed[stage]))
            stage_times["total"].append(elapsed_ms)

    stages = {stage: summarise_times(times) for stage, times in stage_times.items()}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": args.name, "device": args.device, "imgsz": args.imgsz,
        "images": len(image_paths), "repeats": args.repeats,
        "model": info, "stages": stages,
        "trays_per_minute": trays_per_minute(stages["total"]["median_ms"]),
    }
    (REPORT_DIR / f"{args.name}_speed.json").write_text(json.dumps(payload, indent=2))

    report = format_report(args.name, args.device, args.imgsz, stages, info)
    (REPORT_DIR / f"{args.name}_speed.md").write_text(report + "\n")
    print("\n" + report)
    print(f"\nWritten to {REPORT_DIR}")


if __name__ == "__main__":
    main()
