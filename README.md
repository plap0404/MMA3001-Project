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
| `loose-meat` | A rasher sitting out of place within the tray |
| `packaging-error` | Meat or other material caught in the sealed edge |
| `twisted-meat` | A rasher folded or twisted rather than laid flat |
| `unsealed` | Film not sealed to the tray rim |

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
| Size | 720 × 540 pixels, auto-oriented (Roboflow export) |
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
resized to 720 × 540, **no augmentation**, split 1,764 / 115 / 76.

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

### Known data limitations

1. **Severe class imbalance.** `packaging-error` is 62% of training boxes;
   `loose-meat` is 6%.
2. **Thin test coverage.** `loose-meat` has 3 test boxes, `twisted-meat` 7. One box
   either way moves a per-class score by tens of percentage points, so per-class
   figures for these two are indicative only.
3. **Skewed test composition.** `unsealed` is 21% of training boxes but 57% of test
   boxes, so any single overall score largely reflects performance on `unsealed`.
4. **Unexplained count gap.** Roboflow's project-level class balance reports ~1,645
   `packaging-error` boxes against 1,446 counted in the Version 1 export. Figures
   in this repository are the counted ones.
5. **Possible glare shortcut.** Inspected `packaging-error` boxes consistently
   contain a bright specular highlight. The model may be learning "bright spot"
   rather than "meat in the seal". See validation notes.
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
otherwise. Source images are 720 px wide, so 640 shrinks them slightly; since
`packaging-error` boxes are small, resolution is a plausible limiting factor.

*[TODO after runs: state which was selected as the final method and why.]*

---

## 5. Results

Reported on the **test** split, which was held out and scored only after training.
Validation scores are not quoted as final results, because the validation split
selected the saved weights.

*[TODO: paste tables from `reports/metrics/`. Include per-class results, not only
the overall figure — see data limitation 3.]*

| Run | Precision | Recall | mAP50 | mAP50-95 | ms/image |
|---|---|---|---|---|---|
| `baseline_n640` | | | | | |
| `alt_s640` | | | | | |
| `alt_n960` | | | | | |

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
│   └── evaluate.py          Scores trained weights on the held-out test split
├── tests/                   pytest suite for the scripts above
├── reports/
│   ├── leakage/             Leakage report, evidence images
│   └── metrics/             Per-run test-split results (JSON + Markdown)
├── PROJECT_LOG.md           Dated session log: decisions, verified counts, AI-use record
├── requirements.txt
└── pytest.ini
```

`data/` is git-ignored; it is rebuilt by `download_data.py` and `filter_labels.py`.

---

## 7. Reproducing this work

### Environment

Developed on Ubuntu, Python 3.12.3. Training runs in Google Colab on a Tesla T4
(Python 3.13.15, PyTorch 2.11.0+cu128, Ultralytics 8.4.155). Data preparation and
tests were verified on both Python versions.

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

*[TODO: generate the test report and HTML code documentation (pdoc) before
submission.]*

---

## 9. Limitations and further work

*[TODO after results: what the method cannot establish, where it fails, and what
you would do next with more time.]*

---

## 10. Licence

Released under the MIT Licence — see [LICENSE](LICENSE).

The dataset originates from a Roboflow Universe project and is subject to its own
licence terms. *[TODO: state the dataset licence and check Ultralytics' licence
terms, which are AGPL-3.0 unless a commercial licence applies.]*

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
