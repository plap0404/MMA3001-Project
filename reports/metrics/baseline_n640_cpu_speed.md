# baseline_n640_cpu - speed (cpu, imgsz 640)

Model: 3,138,732 parameters, 8.8 GFLOPs per image.

| Stage | Median (ms) | Mean (ms) | 95th pct (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|---|
| preprocess | 4.54 | 4.57 | 5.19 | 3.42 | 7.89 |
| inference | 97.54 | 99.28 | 115.13 | 91.52 | 138.08 |
| postprocess | 0.86 | 0.86 | 1.06 | 0.49 | 1.26 |
| total | 106.15 | 108.09 | 123.70 | 99.71 | 147.88 |

Throughput: **9.4 images/s** = **565 trays/minute** (median of 76 timed images, warm-up discarded).
