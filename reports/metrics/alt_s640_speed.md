# alt_s640 - speed (0, imgsz 640)

Model: 11,137,148 parameters, 28.6 GFLOPs per image.

| Stage | Median (ms) | Mean (ms) | 95th pct (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|---|
| preprocess | 1.45 | 1.52 | 2.10 | 1.36 | 2.94 |
| inference | 9.65 | 9.75 | 10.90 | 9.32 | 11.84 |
| postprocess | 1.12 | 1.10 | 1.62 | 0.46 | 2.03 |
| total | 14.84 | 15.05 | 17.26 | 13.87 | 19.47 |

Throughput: **67.4 images/s** = **4043 trays/minute** (median of 228 timed images, warm-up discarded).
