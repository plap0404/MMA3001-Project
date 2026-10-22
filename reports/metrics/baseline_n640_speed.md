# baseline_n640 - speed (0, imgsz 640)

Model: 3,138,732 parameters, 8.8 GFLOPs per image.

| Stage | Median (ms) | Mean (ms) | 95th pct (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|---|
| preprocess | 1.46 | 1.49 | 1.74 | 1.36 | 2.24 |
| inference | 5.63 | 6.00 | 9.03 | 5.27 | 10.58 |
| postprocess | 1.10 | 1.08 | 1.31 | 0.44 | 1.87 |
| total | 10.79 | 11.21 | 14.64 | 9.79 | 18.07 |

Throughput: **92.7 images/s** = **5562 trays/minute** (median of 228 timed images, warm-up discarded).
