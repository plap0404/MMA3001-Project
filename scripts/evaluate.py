"""Evaluate a trained detection model on a held-out split and save the results.

Training reports scores on the *validation* split, which was used to choose the
saved weights. The **test** split is held back and scored only here, so these are
the numbers that belong in the report.

For each run this writes, under ``reports/metrics``:

* ``<name>_<split>.json`` - every metric, for exact quoting later;
* ``<name>_<split>.md``   - a table ready to paste into the report.

Metrics reported (all between 0 and 1, higher is better):

* **Precision** - of the defects the model reported, the fraction that were real.
* **Recall** - of the defects actually present, the fraction the model found.
* **mAP50** - mean average precision, counting a box correct when it overlaps the
  true box by at least 50%. The usual headline detection score.
* **mAP50-95** - the same, averaged over overlap thresholds from 50% to 95%. A
  stricter score that rewards tight boxes.

Usage (from the project root)::

    python scripts/evaluate.py --weights runs/baseline_n640/weights/best.pt \\
                               --name baseline_n640 [--imgsz 640] [--split test]
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "pork_v4class" / "data.yaml"
REPORT_DIR = ROOT / "reports" / "metrics"


def build_rows(
    names: dict[int, str],
    ap_class_index: list[int],
    class_result,
    boxes_per_class: list[int],
) -> list[dict]:
    """Assemble one results row per class.

    Args:
        names: Class id to class name, from the model.
        ap_class_index: Class ids that appear in the evaluated split, in the
            order used by ``class_result``.
        class_result: Callable taking a position in ``ap_class_index`` and
            returning ``(precision, recall, mAP50, mAP50-95)``.
        boxes_per_class: Number of true boxes per class id in the split.

    Returns:
        One dict per class with its name, box count and four metrics. Classes
        absent from the split are included with ``None`` metrics, so a class
        that was never tested is visible rather than silently missing.
    """
    rows = []
    for position, class_id in enumerate(ap_class_index):
        precision, recall, map50, map50_95 = class_result(position)
        rows.append({
            "class": names[class_id],
            "boxes": int(boxes_per_class[class_id]),
            "precision": float(precision),
            "recall": float(recall),
            "mAP50": float(map50),
            "mAP50-95": float(map50_95),
        })

    for class_id, name in names.items():
        if class_id not in ap_class_index:
            rows.append({
                "class": name,
                "boxes": int(boxes_per_class[class_id]),
                "precision": None,
                "recall": None,
                "mAP50": None,
                "mAP50-95": None,
            })
    return rows


def format_table(rows: list[dict], overall: dict) -> str:
    """Render the per-class rows and the overall row as a Markdown table.

    Args:
        rows: Per-class rows from :func:`build_rows`.
        overall: Metrics averaged over all classes.

    Returns:
        A Markdown table. Missing metrics are shown as ``-``.
    """
    header = "| Class | Boxes | Precision | Recall | mAP50 | mAP50-95 |"
    lines = [header, "|---|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda r: r["class"]):
        cells = [
            "-" if row[key] is None else f"{row[key]:.3f}"
            for key in ("precision", "recall", "mAP50", "mAP50-95")
        ]
        lines.append(f"| {row['class']} | {row['boxes']} | " + " | ".join(cells) + " |")
    cells = [f"{overall[k]:.3f}" for k in ("precision", "recall", "mAP50", "mAP50-95")]
    total = sum(row["boxes"] for row in rows)
    lines.append(f"| **all** | **{total}** | " + " | ".join(f"**{c}**" for c in cells) + " |")
    return "\n".join(lines)


def main() -> None:
    """Evaluate one set of weights and write the report files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="path to best.pt")
    parser.add_argument("--name", required=True, help="run name, used for output files")
    parser.add_argument("--imgsz", type=int, default=640, help="must match training")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    args = parser.parse_args()

    from ultralytics import YOLO  # imported here so --help works without it

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        plots=True,
        name=f"{args.name}_{args.split}_eval",
    )

    names = dict(model.names)
    rows = build_rows(
        names,
        list(metrics.box.ap_class_index),
        metrics.box.class_result,
        list(metrics.nt_per_class),
    )
    overall = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "run": args.name,
        "split": args.split,
        "weights": args.weights,
        "imgsz": args.imgsz,
        "overall": overall,
        "per_class": rows,
        "speed_ms_per_image": {k: float(v) for k, v in metrics.speed.items()},
    }
    (REPORT_DIR / f"{args.name}_{args.split}.json").write_text(json.dumps(payload, indent=2))

    table = format_table(rows, overall)
    (REPORT_DIR / f"{args.name}_{args.split}.md").write_text(
        f"# {args.name} - {args.split} split (imgsz {args.imgsz})\n\n{table}\n"
    )

    print(f"\n{args.name} - {args.split} split (imgsz {args.imgsz})\n")
    print(table)
    print("\nSpeed per image (ms):", {k: round(v, 2) for k, v in metrics.speed.items()})
    print(f"\nWritten to {REPORT_DIR}")


if __name__ == "__main__":
    main()
