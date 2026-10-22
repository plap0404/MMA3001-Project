# Automated Detection of Pork Rasher Packaging Defects

MMA3001 — Numerical Methods and Machine Learning, individual project.
Pantelis Lappas, Monash University, 2026.

A computer-vision pipeline that locates packaging defects in photographs of pork
rasher trays on a production line, so trays can be inspected automatically instead
of by eye.

---

## 1. Engineering problem

Packaged meat is inspected manually at the end of a production line. Manual
inspection is slow, inconsistent between inspectors, and tiring over a shift, so
defective trays reach distribution. The engineering goal is a method that takes a
photograph of a tray and reports **which defects are present and where they are**,
accurately enough and fast enough to run on a moving line.

**Scope of this project.** Four defect types are detected and localised:

| Class | Meaning |
|---|---|
| `loose-meat` | A small piece of meat sitting away from the main group of rashers |
| `packaging-error` | A crease in the film near the edge of the container |
| `twisted-meat` | A rasher folded or twisted rather than laid flat |
| `unsealed` | Film not sealed to the tray rim |

The dataset ships no written definitions, so these describe what the annotations
consistently mark in the images. `packaging-error` and `unsealed` are both sealing
faults: a crease near the rim risks a seal that does not hold, while `unsealed` is
a seal that has already failed.

A fifth annotated class, `wrinkle`, is **excluded**: only 18 examples exist across
the whole dataset, too few to train or to evaluate meaningfully. Excluding it is a
scope decision, not a claim that wrinkles do not matter.

**Out of scope.** Grading meat quality, counting rashers, deciding whether a whole
tray passes or fails, and running on live video. The method reports detections on a
single still image.

---

## 2. Inputs and outputs

### Input

| Property | Value |
|---|---|
| Format | RGB JPEG images |
| Size | 720 × 540 pixels, auto-oriented, **stretched** to that size (Roboflow export) |
| Content | One or two pork rasher trays on a conveyor, fixed overhead camera |
| Source | [Roboflow project](https://universe.roboflow.com/plap0002-student-monash-edu/pork-rasher-error-packaging-hvffk), Version 1 |

### Output

For each image, a list of detections. Each detection is:

* a **class** — one of the four defect types above;
* a **bounding box** — the rectangle containing the defect, in YOLO format
  (`class x_centre y_centre width height`, each coordinate a fraction of image
  width or height, so values lie between 0 and 1);
* a **confidence** between 0 and 1 — how certain the model is.

An image with no detections means no defect of these four types was found. This is
not the same as "the tray is good": a defect outside the four classes, or one the
model misses, also produces no detection.

### Handling invalid or unexpected input

| Situation | Behaviour |
|---|---|
| Image with no matching label file | `filter_labels.py` stops with `FileNotFoundError`, naming the image |
| Label line that is neither a 5-field box nor an even-length polygon | Stops with `ValueError`, printing the offending line |
| Polygon annotation | Converted to the smallest enclosing box; points outside the image are clipped to its edges |
| Image whose only annotations were `wrinkle` | Excluded, so it is not presented to the model as a defect-free tray |
| Dataset folder missing | Scripts exit with a message naming the script to run first |

---

## 3. Dataset

Roboflow Version 1 (generated 16 September 2026): 1,955 images, auto-oriented and
resized to 720 × 540 using `Stretch to`, **no augmentation in the dataset**, split
1,764 / 115 / 76. (Ultralytics still applies its default training-time augmentation —
mosaic, colour shifts and flips — identically in every run.)
Stretching does not preserve the original aspect ratio, so every image — and every
annotation in it — is distorted identically. This is consistent across training and
test data, but it is a property of the dataset worth stating.

`scripts/filter_labels.py` produces the working dataset used throughout, by
removing `wrinkle` and converting polygon annotations to boxes. Counts below were
verified by running `scripts/dataset_stats.py`, and reproduced identically on the
laptop and in Colab.

| Class | Train | Valid | Test | Total |
|---|---|---|---|---|
| `loose-meat` | 127 | 5 | 3 | 135 |
| `packaging-error` | 1,372 | 45 | 29 | 1,446 |
| `twisted-meat` | 236 | 5 | 7 | 248 |
| `unsealed` | 479 | 88 | 52 | 619 |
| **Boxes** | **2,214** | **143** | **91** | **2,448** |
| **Images** | **1,752** | **115** | **76** | **1,943** |

Twenty training images and one test image carry no boxes; these are defect-free
trays and are kept deliberately as negative examples.

### Box sizes

Measured with `scripts/box_stats.py` over all splits. Sizes in pixels are given at
an inference size of 640, i.e. the size the model actually sees.

| Class | Boxes | Median area (% of image) | Median box at 640 px | Smallest / largest area | Under 32x32 px |
|---|---|---|---|---|---|
| `loose-meat` | 135 | 1.59% | 53 x 121 px | 0.149% / 3.9% | 2 (1%) |
| `packaging-error` | 1,446 | 0.21% | 28 x 28 px | 0.001% / 4.8% | 661 (46%) |
| `twisted-meat` | 248 | 1.74% | 51 x 137 px | 0.592% / 3.8% | 0 (0%) |
| `unsealed` | 619 | 9.61% | 221 x 181 px | 0.140% / 17.6% | 2 (0%) |

The classes span roughly a 46x range in median area. `unsealed` is a judgement about
the whole tray and its boxes are correspondingly large; `packaging-error` marks a
crease and is by far the smallest, with its smallest box covering about 4 square
pixels. The median is quoted rather than the mean because a few large boxes would
otherwise inflate it.

At an inference size of 960 the median `packaging-error` box becomes 43 x 43 px and the
share under 32 x 32 px falls from 46% to 17% (`python scripts/box_stats.py --imgsz 960`).
The source images are only 720 px wide, however, so this enlargement adds no real detail
— see section 5.

### Known data limitations

1. **Severe class imbalance.** `packaging-error` is 62% of training boxes;
   `loose-meat` is 6%.
2. **Thin test coverage.** `loose-meat` has 3 test boxes, `twisted-meat` 7. One box
   either way moves a per-class score by tens of percentage points, so per-class
   figures for these two are indicative only.
3. **Skewed test composition.** `unsealed` is 21% of training boxes but 57% of test
   boxes, so any single overall score largely reflects performance on `unsealed`.
4. **Annotations lost when the version was generated.** Roboflow's project-level
   class counts exceed the counts in the exported Version 1 by 200 boxes: 199 in
   `packaging-error` (1,645 vs 1,446) and 1 in `twisted-meat` (249 vs 248). The other
   three classes match exactly, and the project and the version contain the same
   1,955 images, so images were not excluded.

   The likely cause is the version's `Stretch to 720x540` preprocessing. Resizing
   shrinks every annotation, and an annotation too small to survive cannot be written
   to a label file. This is consistent with the measured box sizes above:
   `packaging-error` is by far the smallest class, with its smallest surviving box
   covering about 4 square pixels. It cannot be confirmed for the single
   `twisted-meat` box, whose surviving boxes are all large — any box that was dropped
   is by definition absent from the data and cannot be measured. All figures in this
   repository are counted from the exported labels, never taken from the dashboard.
6. **Split independence.** The images are video frames. Frame numbers in the
   filenames were shown **not** to represent time order, and the most visually
   similar train/test pair was confirmed by inspection to be different trays, so no
   near-duplicate leakage was found. It cannot be fully ruled out: the same tray
   photographed at a different point on the belt could escape both checks. See
   `reports/leakage/`.

---

## 4. Method

Detection uses **YOLOv8** (Ultralytics), a single-stage object detector, starting
from COCO-pretrained weights. Pretrained weights are used because the dataset is
small; the general visual features learned from a large dataset transfer, and only
the task-specific layers need substantial adaptation.

Three runs, each changing **one** setting from the baseline so that differences are
attributable:

| Run | Model | Image size | Question it answers |
|---|---|---|---|
| `baseline_n640` | YOLOv8n | 640 | Untuned reference point |
| `alt_s640` | YOLOv8s | 640 | Is model capacity the limit? |
| `alt_n960` | YOLOv8n | 960 | Is input resolution the limit? |

All runs share 100 epochs, batch size 16, `seed=0`, and Ultralytics defaults
otherwise.

The resolution run is motivated by measurement, not intuition. `scripts/box_stats.py`
reports that the median `packaging-error` box is **28 x 28 px** at an inference size
of 640, and **46% of them fall below the conventional 32 x 32 px small-object
threshold**. Since that class is 62% of all training boxes, how well small objects
are handled should dominate overall performance.

**Selected method: `baseline_n640`.** It scored highest on the held-out test split
(mAP50 0.630), highest on `packaging-error`, the most important class, and is also the
cheapest and fastest of the three. Neither alternative improved on it (section 5), so
there is no accuracy-versus-cost trade-off to make.

Training used the optimiser Ultralytics selects automatically (AdamW, initial learning
rate 0.00125, decayed linearly; confirmed from the logged learning rate), and the saved
weights are `best.pt`, the epoch with the best validation score.

---

## 5. Results

Reported on the **test** split, which was held out and scored only after training.
Validation scores are not quoted as final results, because the validation split
selected the saved weights.

### Overall

| Run | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| `baseline_n640` | 0.638 | 0.616 | 0.630 | 0.230 |
| `alt_s640` | 0.419 | 0.545 | 0.491 | 0.225 |
| `alt_n960` | 0.611 | 0.635 | 0.561 | 0.232 |

### Per class — mAP50, all runs

| Class | Test boxes | `baseline_n640` | `alt_s640` | `alt_n960` |
|---|---|---|---|---|
| `loose-meat` | 3 | 0.913 | 0.618 | 0.727 |
| `packaging-error` | 29 | **0.513** | 0.340 | 0.364 |
| `twisted-meat` | 7 | 0.318 | 0.287 | **0.397** |
| `unsealed` | 52 | **0.777** | 0.720 | 0.755 |

Full per-class tables for every run are in `reports/metrics/<run>_test.md`.

### Per class — `baseline_n640`

| Class | Test boxes | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| `loose-meat` | 3 | 0.747 | 0.667 | 0.913 | 0.275 |
| `packaging-error` | 29 | 0.593 | 0.414 | 0.513 | 0.132 |
| `twisted-meat` | 7 | 0.493 | 0.558 | 0.318 | 0.127 |
| `unsealed` | 52 | 0.719 | 0.827 | 0.777 | 0.385 |
| **all** | **91** | **0.638** | **0.616** | **0.630** | **0.230** |

**The 0.913 for `loose-meat` is not evidence of good performance.** It rests on 3
boxes in 2 images; a single box either way moves it by roughly 33 percentage points.
It is reported for completeness and excluded from any conclusion.

**`packaging-error` recall is 0.414**, so the baseline misses close to 60% of creases.
This is the class with the smallest boxes and the one that matters most for the stated
purpose, since a missed defect leaves the line undetected.

**`twisted-meat` scores lowest (mAP50 0.318) despite large boxes.** Box size alone
therefore does not explain the ranking: `twisted-meat` has only 236 training boxes
against 1,372 for `packaging-error`. Object size and number of training examples are
confounded in this dataset and these runs cannot fully separate them.

**mAP50-95 (0.230) is far below mAP50 (0.630)**, so detections are placed roughly
correctly but the boxes are not tight. The effect is strongest for `packaging-error`
(0.132), as expected: for a box 28 px across, being a few pixels out is a large
proportional error.

### Comparing the runs

**A larger model (`alt_s640`) was worse, not better** (mAP50 0.491). The loss is mainly
in precision — more false alarms. Its training log is consistent with overfitting:
training losses fall to the last epoch while validation losses rise from about epoch 65.
Its validation mAP50 (about 0.72) is far above its test score, partly because `best.pt`
is chosen *by* the validation split, which is why only test scores are quoted here.

**A larger input size (`alt_n960`) did not help `packaging-error`**, the class it was run
for (0.513 → 0.364), even though it lifts most of those boxes above the small-object
size. The likely reason is that the source images are 720 x 540: running at 960 enlarges
them, which adds pixels but no new detail. The smallest creases were also already lost
when the dataset was resized (section 3, limitation 4). This is the most likely
explanation, not a proven one. `twisted-meat` did improve (0.318 → 0.397, mAP50-95 0.127
→ 0.195), but on 7 test boxes this is indicative only.

**mAP50-95 is almost identical across runs (0.225–0.232)**: the runs differ in whether
defects are found, not in how tightly boxes fit them.

### Speed and compute cost

Measured with `scripts/benchmark.py`: one image at a time (as on a line), 10 warm-up images
discarded, 3 passes over the 76 test images, GPU synchronised before each timer stops.

| Run | Device | Parameters | GFLOPs / image | Median (ms) | 95th pct (ms) | Images / minute |
|---|---|---|---|---|---|---|
| **`baseline_n640`** | T4 GPU | 3.14 M | 8.8 | **10.8** | **14.6** | **5,562** |
| `alt_s640` | T4 GPU | 11.14 M | 28.6 | 14.8 | 17.3 | 4,043 |
| `alt_n960` | T4 GPU | 3.14 M | 19.7 | 17.0 | 20.6 | 3,523 |
| `baseline_n640` | Colab CPU | 3.14 M | 8.8 | 106.2 | 123.7 | 565 |

GFLOPs (billions of arithmetic operations per image) do not depend on the machine, so they
are the fair measure of cost. The 960 run costs 2.24x the 640 run, matching the 2.25x
increase in pixel count. Times include reading the image from disk; the figures printed
during evaluation came from batched inference and are not quoted.

Even at its 95th-percentile time on CPU (0.12 s), the selected model handles about 8 images
per second. Whether that keeps up with a real line depends on its speed, which is not known
for this dataset and is an assumption. Images rather than trays are counted, since some
images show two trays.

---

## 6. Repository layout

```
MMA3001-Project/
├── notebooks/
│   └── train_colab.ipynb    Colab runner: rebuilds everything from this repo, then trains
├── scripts/
│   ├── download_data.py     Fetches Roboflow Version 1 into data/pork_v1
│   ├── dataset_stats.py     Counts images, labels and boxes per class per split
│   ├── filter_labels.py     Removes wrinkle, converts polygons to boxes -> data/pork_v4class
│   ├── check_leakage.py     Near-duplicate investigation between splits
│   ├── evaluate.py          Scores trained weights on the held-out test split
│   ├── benchmark.py         Latency, throughput, parameters and GFLOPs
│   └── box_stats.py         Per-class box sizes at a given inference size
├── tests/                   pytest suite for the scripts above
├── reports/
│   ├── leakage/             Leakage report, evidence images
│   ├── metrics/             Per-run test-split and speed results (JSON + Markdown)
│   ├── docs/                HTML code documentation (pdoc)
│   └── tests/               pytest report
├── PROJECT_LOG.md           Dated session log: decisions, verified counts, AI-use record
├── requirements.txt
└── pytest.ini
```

`data/` is git-ignored; it is rebuilt by `download_data.py` and `filter_labels.py`.

---

## 7. Reproducing this work

### Environment

Developed on Ubuntu, Python 3.12.3. Training runs in Google Colab on a Tesla T4
(Python 3.13.15, PyTorch 2.11.0+cu130, Ultralytics 8.4.183). Data preparation and
tests were verified on both Python versions.

Colab upgraded Ultralytics between the September setup (8.4.155) and the October
training runs. `ultralytics` is now pinned in `requirements.txt` so that all three
runs are compared on the same version.

### Local setup

```bash
git clone https://github.com/plap0404/MMA3001-Project.git
cd MMA3001-Project
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A Roboflow API key is required to download the dataset. Place it in a `.env` file
in the project root (this file is git-ignored and must never be committed):

```
ROBOFLOW_API_KEY=your_key_here
```

### Build the dataset and check it

```bash
python scripts/download_data.py          # -> data/pork_v1
python scripts/filter_labels.py          # -> data/pork_v4class
python scripts/dataset_stats.py data/pork_v4class   # must match the table in section 3
python -m pytest                         # all tests must pass
```

### Train

Training requires a GPU. Open `notebooks/train_colab.ipynb` in Google Colab
(*Runtime -> Change runtime type -> T4 GPU*), add the Roboflow key as a Colab
secret named `ROBOFLOW_API_KEY` with notebook access enabled, and run the cells in
order. The notebook clones this repository, so results come from committed code.

### Evaluate

```bash
python scripts/evaluate.py --weights <path>/best.pt --name <run_name> [--imgsz 640]
```

Writes `reports/metrics/<run_name>_test.json` and `.md`.

---

## 8. Testing

```bash
python -m pytest            # run the suite
python -m pytest -v         # per-test detail, for the test report
```

Tests use small synthetic fixtures rather than the real dataset, so they run in
under a second and do not depend on downloaded data.

53 tests cover filtering, leakage checking, evaluation, benchmarking and box
statistics. The saved test report is `reports/tests/pytest_report.txt`.

### Code documentation

HTML documentation is generated from the docstrings with pdoc, and saved in
`reports/docs/` (open `reports/docs/index.html` in a browser):

```bash
python -m pytest -v > reports/tests/pytest_report.txt
pdoc scripts/*.py -o reports/docs
```

---

## 9. Limitations and further work

### Limitations

1. **`packaging-error` recall is 0.414**: the selected model misses close to 60% of
   creases, the most important defect. It is not ready to replace a human inspector.
2. **Small, skewed test set.** 76 images; 3 `loose-meat` and 7 `twisted-meat` boxes.
   Per-class scores for those two are indicative only, and the overall score is
   dominated by `unsealed` (57% of test boxes).
3. **One training run per configuration.** Run-to-run variation was not measured, so
   part of the gap between runs may be chance. A seed does not make GPU training fully
   repeatable.
4. **Object size and class frequency are confounded.** `twisted-meat` has large boxes but
   few examples and scores worst, so these experiments cannot say which matters more.
5. **Information lost before training.** The stretch-resize to 720 x 540 distorts aspect
   ratio and appears to have dropped ~200 of the smallest annotations; no setting
   chosen later can recover them.
6. **Out-of-scope defects pass silently.** `wrinkle` and anything outside the four
   classes produce no detection, which is not the same as a good tray.
7. **Possible leakage between frames.** No near-duplicates were found, but the same tray
   photographed at another belt position cannot be fully ruled out.
8. **Run log incomplete.** `results.csv` for `alt_s640` is missing epochs 78–80 and 85–87.
   The time column shows training ran continuously, so the most likely cause is a write
   lost while saving to Google Drive. The weights and test results are unaffected.

### Further work

1. Re-export the dataset from the original full-resolution images (or with `Fit`
   instead of `Stretch to`), so small creases survive and a larger input size can carry
   real detail.
2. Repeat each configuration over several seeds and report the spread.
3. Address imbalance: more `twisted-meat` and `loose-meat` examples, or class-weighted
   loss, to separate the effect of class frequency from object size.
4. Tune the confidence threshold for the line's priority — a missed unsealed tray costs
   more than a false alarm — and turn detections into a pass/fail verdict per tray.
5. Test on images from a different day or line, to check the model generalises beyond
   these recordings.

---

## 10. Licence

Released under the MIT Licence — see [LICENSE](LICENSE).

**Ultralytics** (used for training and evaluation) is released under **AGPL-3.0**,
with a separate commercial Enterprise licence available. AGPL-3.0 is acceptable for
this open, non-commercial coursework repository. A real deployment on a production
line would need the licence position reviewed, since AGPL-3.0 carries obligations
that a commercial operator may not want to accept.

The dataset originates from a Roboflow Universe project. *[TODO: confirm the dataset
licence on the ORIGINAL Universe page, not the fork — the fork lists this workspace
as author, which would be the wrong attribution. Project notes record it as CC BY 4.0,
which requires crediting the original creator.]*

---

## AI Acknowledgement

Artificial intelligence tools were used in the development of this project. In accordance with Monash University's guidance on [acknowledging the use of AI](https://www.monash.edu/student-academic-success/learning-with-ai/academic-integrity-and-ai/acknowledging-the-use-of-ai) and the responsible-use expectations set out in MMA3001, the following is a summary of what was used, how it was used, and the oversight applied.

### Tools used

| Tool | Version | Accessed | Used for |
|---|---|---|---|
| Claude ([Anthropic](https://claude.ai)) | Opus 5 | *[date range]* | *[e.g. debugging, code review, explaining library behaviour]* |
| Gemini (built into [Google Colab](https://colab.research.google.com)) | *[version]* | *[date range]* | *[e.g. generating initial code cells for X]* |
| *[other tool + link]* | *[version]* | *[date range]* | *[purpose]* |

### How AI was used

AI assistance was limited to *[scope — e.g. drafting boilerplate for the data-loading module, suggesting plotting approaches, and explaining runtime errors]*. No AI-generated content was incorporated verbatim.

Suggested code was rewritten to match the project's structure and naming conventions. For example, *[describe a specific change you made — e.g. the vectorised approach initially suggested was replaced with an explicit loop for readability]*.

Analysis, interpretation of results, and all engineering conclusions are my own.

**Known limitations.** These models can produce confident but incorrect output, particularly on library-specific APIs and numerical edge cases. Their training data has a knowledge cutoff that may predate the versions of the libraries used here, and outputs may reflect biases present in that training data.

### Data handling

*[If applicable]* No confidential, client-owned, or personally identifying data was provided to any external AI tool. *[Name anything you deliberately withheld and why.]*

### Human oversight and responsibility

All AI-assisted output was verified before inclusion:

- Code was executed and tested against *[known values / hand-calculated cases / documented examples]*.
- Factual and technical claims were checked against *[textbooks, standards, official library documentation — name them]*.
- *[Describe at least one thing AI got wrong that you caught and corrected.]*

I am solely responsible for the accuracy, quality, and fitness-for-purpose of the final submission, including any content developed with AI assistance.

---

*Last updated: [date]*
