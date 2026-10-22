# MMA3001 Project Log

Pork rasher packaging defect detection.
Repository: https://github.com/plap0404/MMA3001-Project

This log records decisions, their reasoning, verified facts and open questions.
It is the source material for the written report and the presentation.

---

## Session 1 — 18 September 2026

### 1. Project and dataset selection

- Selected **Dataset 3** from the teaching team's provided options:
  *Automated Detection of Pork Rasher Packaging Errors and Meat Quality.*
- Task type: **object detection** (draw boxes around defects in images of
  packaged pork rasher trays).
- Engineering context: automated quality inspection on a food packaging line,
  to reduce reliance on manual inspection and catch defective trays before
  distribution.
- Forked the dataset into a personal Roboflow workspace:
  - Fork URL: https://universe.roboflow.com/plap0002-student-monash-edu/pork-rasher-error-packaging-hvffk
  - Workspace slug: `plap0002-student-monash-edu`
  - Project slug: `pork-rasher-error-packaging-hvffk`
  - Licence: **CC BY 4.0** — the original creator must be credited in the report.
    Roboflow's auto-generated citation names my own workspace as author, which is
    incorrect for citation purposes; cite the **original** dataset from the unit link.

### 2. Roboflow dataset version

The fork copied images and annotations but **not** the teaching team's generated
versions (the project showed "Dataset versions: 0"). Version 1 was generated manually.

**Version 1 settings (this is the version in use):**

| Setting | Value |
|---|---|
| Created | 2026-09-16, by plap0002@student.monash.edu |
| Total images | 1,955 |
| Preprocessing | Auto-Orient; Resize to 720 x 540 |
| Augmentation | **None** (deliberate — this is the untouched baseline) |
| Split | Train 1,764 (90%) / Valid 115 (6%) / Test 76 (4%) |
| Export format used | YOLOv8 |

Note: the unit slides mention a "Version 4" with 3,549 images after augmentation.
That version was not carried over by the fork. The 1,955 images here are the
original, un-augmented images.

### 3. Class definitions

Classes are stored in label files as **numbers**, in this order (from `data.yaml`):

| ID | Class name |
|---|---|
| 0 | loose-meat |
| 1 | packaging-error |
| 2 | twisted-meat |
| 3 | unsealed |
| 4 | wrinkle |

**Observed label geometry (from inspecting annotated images):**

- `unsealed` boxes cover the **whole tray** — a judgement about the entire product.
- `packaging-error` boxes are **small and localised**, sitting on the end of a
  rasher where it meets the tray edge / seal area.
- `wrinkle` boxes are **very small thin strips** on the film over the meat.

**Working interpretation of `packaging-error` (UNCONFIRMED):** meat is touching or
overlapping the seal area, which would prevent the tray sealing correctly.
*Action required: confirm this with the unit's dataset notes or the teaching team.*

### 4. Verified dataset statistics

Counted from the downloaded YOLO labels (not from Roboflow's dashboard).

**Files per split — all complete, every image has a matching label file:**

| Split | Images | Label files | Empty label files |
|---|---|---|---|
| train | 1,764 | 1,764 | 20 |
| valid | 115 | 115 | 0 |
| test | 76 | 76 | 1 |

**Bounding boxes per class per split:**

| ID | Class | Train | Valid | Test | Total |
|---|---|---|---|---|---|
| 0 | loose-meat | 127 | 5 | 3 | 135 |
| 1 | packaging-error | 1,372 | 45 | 29 | 1,446 |
| 2 | twisted-meat | 236 | 5 | 7 | 248 |
| 3 | unsealed | 479 | 88 | 52 | 619 |
| 4 | wrinkle | 18 | 0 | 0 | 18 |
| | **Total** | **2,232** | **143** | **91** | **2,466** |

*Corrected in Session 2: this row originally read 291 (test) and 2,666 (grand total), an arithmetic error. Re-verified by re-running `dataset_stats.py`.*

~~Grand total (2,666) matches Roboflow's reported class balance, so the download is
verified complete.~~ *Corrected in Session 2:* the counted total is **2,466**, which
does **not** match Roboflow's class-balance total (about 2,666). The ~200-box gap is
entirely in `packaging-error` (Roboflow 1,645 vs 1,446 counted). Unexplained, possibly
images excluded from the generated version. Do not quote Roboflow's per-class figures
in the report — quote the counted ones.

**Counting pitfall (worth mentioning in the report):** an initial count using
`cat data/pork_v1/*/labels/*.txt | awk ...` gave badly wrong totals (535 instead of
2,466), because the label files do not end with a newline, so the last line of one
file and the first line of the next were concatenated into a single line. Passing
the files to `awk` directly fixed it. This is a good example of verifying a result
against an independent source rather than trusting the first number produced.

### 5. Decisions made, with reasoning

**D1 — Keep the teaching team's train/valid/test split (for now).**
It is the split the data shipped with, and it gets to a working model soonest.
*Limitation:* the test set is only 76 images, so one image moves accuracy by more
than 1%.

**D2 — Scope: detect 4 classes, drop `wrinkle`.**
Reasoning: `wrinkle` has only 18 boxes, **all of them in the training split**
(0 in valid, 0 in test). It therefore cannot be scored at all, no matter how well
the model learns it. There is also evidence that those 18 boxes come from only a
few physical trays, since the filenames cluster tightly (frame_with_red_10200 to
10550), i.e. neighbouring video frames of the same tray.
*Consequence to state in the report:* the model will pass a wrinkled tray, because
wrinkles are outside the defined scope.

**D3 — Remove `wrinkle` in code, not in Roboflow.**
Roboflow's "Modify Classes" preprocessing step requires a paid plan.
Doing it in code is arguably better anyway: the step is version-controlled,
reproducible by anyone who clones the repo, reversible via a single setting, and
directly testable with pytest ("after filtering, no class-4 boxes remain").

**D4 — Keep images that have no boxes (do not use "Filter Null").**
Images of trays with no defect teach the model what "normal" looks like and reduce
false alarms. 21 such images exist (20 train, 1 test).

**D5 — Version 1 has no augmentation.**
This gives a clean baseline. Augmentation can be added later as a deliberate
comparison, which feeds the "optimisation and comparison" requirement of the brief.

**D6 — The dataset is not stored in Git.**
`data/` and `.env` are in `.gitignore`. Reasons: repository size, and CC BY 4.0
redistribution is avoided. Reproducibility is preserved instead by committing
`scripts/download_data.py`, which re-downloads the exact version on demand.

**D7 — API key kept in `.env`, never in code.**
`python-dotenv` reads it at runtime. `.env` is git-ignored and was verified as
ignored via `git status`.

### 6. Risks and limitations identified (report material)

1. **Video-frame leakage.** Filenames (`frame_with_red_####.jpg`) indicate the
   images are frames extracted from video. Near-identical frames may sit on both
   sides of the train/test split, which would inflate test scores because the model
   is partly being tested on images it has effectively already seen.
   *Not yet investigated.*
2. **Severe class imbalance.** packaging-error (1,446) vs loose-meat (135).
   The model will favour common classes.
3. **Mixed box scales.** `unsealed` covers a whole tray; `packaging-error` and
   `wrinkle` are small. Detection models generally perform worse on small objects,
   especially after images are downscaled. Expect uneven per-class scores.
4. **Glare shortcut risk.** Every inspected `packaging-error` box also contained a
   bright white glare spot. The model may learn "bright spot" rather than
   "meat on the seal". *Test for this later* by checking whether the model flags
   glare on defect-free trays. Strong validation material.
5. **Thin test coverage for `loose-meat`.** Only 3 test boxes. One box either way
   swings its score by ~33 percentage points; treat any per-class figure for
   loose-meat as indicative only.
6. **Test set composition is skewed.** `unsealed` is 21% of training boxes but 57%
   of test boxes (*corrected in Session 2; originally 15% and 56%*), so the headline test score will mostly reflect performance on
   `unsealed`, not overall performance.
7. **`data.yaml` path quirk.** Roboflow writes `train: ../train/images` with a
   leading `../`, which commonly confuses training tools about the dataset root.
   To be corrected when the filtered dataset is written.

### 7. Environment and tooling set up

- **OS:** Ubuntu (laptop: pantelis-IdeaPad-Slim-3-15IRH10)
- **Python:** 3.12.3
- **Editor:** VS Code + Microsoft Python extension (v2026.4.0)
- **Project location:** `~/Documents/MMA3001/MMA3001-Project`
  (an earlier clone landed at a nested path `~/home/pantelis/Documents/...`;
  moved and the stray folders removed)
- **Virtual environment:** `.venv` in the project root.
  Activate each session with `source .venv/bin/activate`
- **GitHub CLI:** `gh` v2.45.0, logged in to **two** accounts:
  - `plap0404` — uni account, owns this repo (active when this log was written)
  - `pl0727` — personal account
  - Switch with `gh auth switch --user plap0404` / `--user pl0727`
  - Check with `gh auth status` before starting work
- **Git identity (set locally in this repo only, not globally):**
  - `user.name` = Pan Lappas
  - `user.email` = 161100943+plap0404@users.noreply.github.com
  - The global identity remains the personal account, deliberately untouched.

### 8. Repository state at end of session

```
MMA3001-Project/
├── .gitignore          (Python template + data/ + .env)
├── LICENSE             (MIT)
├── README.md           (still the default stub — TODO)
├── requirements.txt    (roboflow, python-dotenv, pytest, pdoc, pyyaml)
├── scripts/
│   ├── download_data.py    (downloads Roboflow v1 -> data/pork_v1)
│   └── dataset_stats.py    (counts images, labels, boxes per class)
└── data/               (git-ignored; holds pork_v1 locally)
```

Commits so far (most recent last):
1. Initial commit (README, LICENSE, .gitignore)
2. `Ignore local data folder and .env secrets file`
3. `Add initial project requirements`
4. `Add script to download dataset v1 from Roboflow`
5. `Add script to report dataset image and class counts` *(in progress at session end)*

Routine for every change: **edit → `git add <file>` → `git commit -m "clear message"` → `git push`**

### 9. Open items to pick up next session

1. **Confirm repository visibility.** Opening the repo link in an incognito window
   returned 404, suggesting it is Private. It must be accessible to markers before
   submission (Settings → Danger Zone → Change visibility, or add staff as
   collaborators). *Still unverified.*
2. **Confirm the meaning of `packaging-error`** with the unit's dataset notes or
   teaching team.
3. **Write `filter_labels.py`**: remove class 4 (wrinkle) from the downloaded
   dataset, write a corrected `data.yaml` with 4 classes and fixed paths, and
   output to `data/pork_v4class`.
4. **Write the first pytest test** for that filter (no class-4 boxes remain;
   all other boxes unchanged; file counts preserved).
5. **Decide whether to rebuild the splits.** Motivated by items 1 and 5 in the
   risks list above. If rebuilt, it must be grouped so that near-identical video
   frames stay within one split.
6. **Set up Colab for training** (the laptop has no suitable GPU). Plan: develop
   and test code locally, clone the repo into Colab, train there.
7. **Write the real README** (purpose, install, usage, environment, data
   provenance, licence).
8. Add exact version pins to `requirements.txt` once the toolchain settles.

### 10. AI use record (for the report's AI reflection section)

Tool: **Claude (Anthropic)**, used interactively throughout this session.

| Task | AI contribution | My contribution |
|---|---|---|
| Explaining what "forking" a dataset means | High — explanation only | Performed the fork myself |
| Deciding project scope (4 classes vs 5 vs merged) | Medium — laid out three options with trade-offs | Chose option B and can justify it |
| Roboflow / GitHub / Ubuntu setup steps | High — step-by-step instructions | Ran every command, reported results |
| Two-GitHub-account configuration | High — diagnosed and prescribed | Executed; verified with `gh auth status` |
| `scripts/download_data.py` | High — written by Claude | Reviewed, ran, verified output |
| `scripts/dataset_stats.py` | High — written by Claude | Reviewed, ran, cross-checked against shell counts |
| Dataset statistics interpretation | Medium — flagged imbalance and split issues | Verified counts independently |

**AI errors caught during this session (worth citing in the reflection):**

1. Claude initially claimed `packaging-error` appeared on almost every image, based
   on Roboflow page captions. Inspecting the actual annotated images showed the
   boxes are small and localised, and often absent. Claude corrected itself.
2. Claude supplied a shell command using `cat ... | awk` to count boxes, which
   silently undercounted (535 vs the true 2,466) because the label files lack
   trailing newlines. Caught only because the total disagreed with Roboflow's
   dashboard.

Both cases support the same lesson: AI output must be cross-checked against an
independent source before it is relied upon.

---

## Session 2 — 18 September 2026

### 1. Corrections to Session 1

- The box-count table's totals row was wrong: test is **91** boxes (not 291) and the
  grand total is **2,466** (not 2,666). Confirmed by re-running
  `python scripts/dataset_stats.py`, which reproduced every per-class figure.
- Consequence: the claim that the download "matches Roboflow" was false. Roboflow's
  class-balance total (~2,666) exceeds the counted total by ~200 boxes, all in
  `packaging-error`. Still unexplained (see open items).
- `unsealed` share corrected to 21% of train boxes and 57% of test boxes.
- Corrections have been made in place in Session 1, marked "corrected in Session 2".

### 2. Repository visibility confirmed

The repository is **public**: files were fetched from outside the account with no
login. Session 1 open item 1 is closed.

### 3. Filtering to 4 classes — `scripts/filter_labels.py`

Reads `data/pork_v1`, writes `data/pork_v4class`; the original download is never
modified. The output folder is rebuilt from scratch on every run.

**Discovery: some labels are polygons, not boxes.** The first run stopped on a label
line with 13 numbers instead of 5: a class id followed by 6 x,y points (a closed
outline). Roboflow's YOLOv8 export mixes box lines and polygon lines in the same
files. The script's format check caught it.

**D8 — Drop images whose only boxes were `wrinkle`.**
Removing the wrinkle box from such an image would leave an empty label file, which
the model reads as "defect-free tray". Keeping them would teach it that a wrinkled
tray is normal. Controlled by `DROP_IMAGES_LEFT_EMPTY = True`.

**D9 — Convert polygon annotations to their enclosing box.**
The output is boxes only, so any detection model (including the later comparison
model) reads it the same way, and metrics are unambiguous. Points outside the image
are clipped to its edges.

**Verified result of the run:**

| Split | Images kept | Images dropped | Boxes removed | Polygons converted |
|---|---|---|---|---|
| train | 1,752 | 12 | 18 | 0 |
| valid | 115 | 0 | 0 | 0 |
| test | 76 | 0 | 0 | 0 |

- **All polygons were `wrinkle` boxes.** None of the 4 kept classes contains a
  polygon, so every kept box is exactly as the annotator drew it.
- 12 train images were wrinkle-only and were dropped. The other 6 wrinkle boxes
  were on trays that also had in-scope defects; those images were kept with only the
  wrinkle box removed (consistent with D2: wrinkles are out of scope).
- Every other class count is unchanged in every split; the 20 + 1 empty label files
  (genuine defect-free trays, D4) are unchanged.
- The run checks every image has a label file **with the same name**. It passed, so
  the Session 1 statement "every image has a matching label file" is now verified
  (Session 1 had only shown the counts were equal).

**Filtered dataset (`data/pork_v4class`), verified with `dataset_stats.py`:**

| ID | Class | Train | Valid | Test | Total |
|---|---|---|---|---|---|
| 0 | loose-meat | 127 | 5 | 3 | 135 |
| 1 | packaging-error | 1,372 | 45 | 29 | 1,446 |
| 2 | twisted-meat | 236 | 5 | 7 | 248 |
| 3 | unsealed | 479 | 88 | 52 | 619 |
| | **Boxes** | **2,214** | **143** | **91** | **2,448** |
| | **Images** | **1,752** | **115** | **76** | **1,943** |

The new `data.yaml` lists 4 classes and uses split paths relative to the yaml file
(`train/images`), with no `path` key. This fixes Session 1 risk 7 (Roboflow's
`../train/images` paths).

### 4. Leakage investigation — `scripts/check_leakage.py`

**Question:** do near-identical video frames of the same tray appear in both train
and valid/test? If so, test scores would be inflated (Session 1 risk 1).

**Method (two independent kinds of evidence):**
1. *Frame numbers* from filenames (`frame_with_red_####`): for each valid/test image,
   the train image with the closest frame number.
2. *Visual similarity:* a 256-bit difference hash (dHash) of each image, compared by
   Hamming distance (number of differing bits).
3. *Manual inspection* of saved side-by-side pairs. Criteria: number of rashers,
   rasher shapes and fat patterns, tray positions.

**What happened:**
- Frame numbers first suggested severe leakage: all 76 test images had a train image
  within 25 frame numbers, median gap 2. Claude concluded the leak was "real and
  large".
- The hash was weak: "neighbouring" train frames (median distance 51) were barely
  more alike than far-apart ones (median 59). Its first flagging rule (92% flagged)
  was unsound — it compared each image's *best* match out of 1,752 against *random*
  pairs, so almost everything was flagged. That rule was removed.
- **Manual inspection overturned the frame-number conclusion:**
  - `evidence/04_test_gap1.jpg`: test frame 4523 vs train frame 4522 — one number
    apart, completely different trays and positions.
  - `evidence/13_valid_gap0.jpg`: valid frame 1946 vs train frame 1946 — the *same*
    number, different trays. The numbering is reused across several recordings that
    share the `frame_with_red` prefix, so frame numbers say nothing about time order.
  - The single most visually similar pair (hash distance 12; valid frame 4847 vs
    train frame 2754) was confirmed to be **different trays** (6 rashers vs 5,
    different fat patterns). The hash responds to scene layout — same conveyor,
    lighting and tray positions — not to the individual tray.

**Conclusion:** no evidence of near-duplicate leakage was found. **D1 (keep the
original split) is confirmed.**
*Limitation to state in the report:* leakage cannot be fully ruled out — the same
tray photographed at a different point on the belt could be missed by both methods.

Evidence committed in `reports/leakage/`: `leakage_report.csv` and `evidence/`.
The 20 generated pair images (`frame_pairs/`) are git-ignored.

### 5. Tests

- `pytest.ini` added (`testpaths = tests`, `pythonpath = scripts`).
- **28 tests**, all passing on the laptop and on Colab:
  16 in `tests/test_filter_labels.py` (wrinkle removal, polygon conversion using the
  real line that crashed the first run, missing labels, `data.yaml` output) and
  12 in `tests/test_check_leakage.py` (filename parsing, hashing, frame matching).

### 6. Colab set up — `notebooks/train_colab.ipynb`

Laptop has no suitable GPU, so training runs on Colab. The notebook rebuilds
everything from the GitHub repository: check GPU → clone/pull repo → install
requirements + `ultralytics` → download and filter data → verify stats and run tests
→ record versions → train. Anything not pushed to GitHub does not exist in Colab.

- **API key:** stored as a Colab secret `ROBOFLOW_API_KEY` (Notebook access on), read
  into an environment variable at runtime; never written in the notebook.
- **Results** are saved to Google Drive (`MyDrive/MMA3001/runs`), since Colab wipes
  `/content` when the session ends.
- `data.yaml` is passed to Ultralytics as an **absolute path**, because with a
  relative path Ultralytics looks for the dataset in its own datasets folder.
- **Dataset stats on Colab matched the laptop exactly; all 28 tests passed.**

**Software versions (Colab, 18 Sep 2026):**

| Component | Version |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| Ultralytics | 8.4.155 |
| GPU | Tesla T4 |

Laptop Python is 3.12.3; data preparation and tests were verified on both 3.12
(laptop) and 3.13 (Colab). Training runs only on Colab.

**Smoke test (1 epoch, YOLOv8n, imgsz 640, seed 0):** completed and saved to
`MyDrive/MMA3001/runs/smoke_test`. The model used the 4 correct classes and validated
against 5 / 45 / 5 / 88 boxes, matching the verified valid split. mAP50 = 0.018 —
meaningless after one epoch; the run only proves the pipeline works end to end.
Speed figures from this run are not to be quoted (a half-trained model produces many
junk boxes, which slows post-processing).

### 7. Repository state at end of session

```
MMA3001-Project/
├── .gitignore          (+ reports/leakage/frame_pairs/)
├── LICENSE             (MIT)
├── PROJECT_LOG.md
├── README.md           (still not written — TODO)
├── pytest.ini
├── requirements.txt    (roboflow, python-dotenv, pytest, pdoc, pyyaml, numpy, pillow)
├── notebooks/
│   └── train_colab.ipynb
├── reports/leakage/
│   ├── leakage_report.csv
│   └── evidence/       (04_test_gap1.jpg, 13_valid_gap0.jpg)
├── scripts/
│   ├── download_data.py
│   ├── dataset_stats.py
│   ├── filter_labels.py
│   └── check_leakage.py
└── tests/
    ├── test_filter_labels.py
    └── test_check_leakage.py
```

17 commits on `main` at end of session.

### 8. Open items for next session

1. **Train the baseline.** Proposed (not yet agreed): YOLOv8n pretrained, 100 epochs,
   imgsz 640, defaults otherwise, seed 0. Estimated 30–60 min on a T4; Ultralytics
   checkpoints each epoch to Drive, so an interrupted run can be resumed.
2. Notebook tidy-up: assign the training result (`results = model.train(...)`) so the
   full metrics object is not printed.
3. Add `closest_pair_distance12.jpg` (the distance-12 different-trays pair) to
   `reports/leakage/evidence/` — recovered from a screenshot.
4. Check Roboflow's figures for **Version 1 specifically** to explain the ~200-box
   `packaging-error` gap.
5. Confirm the meaning of `packaging-error` with the unit's dataset notes or teaching
   team (carried over from Session 1).
6. Write the real README (carried over).
7. Pin exact versions in `requirements.txt`, including `ultralytics` (carried over).
8. Licensing note for the report: check Ultralytics' licence terms against the
   repo's MIT licence.

### 9. AI use record — Session 2

| Task | AI contribution | My contribution |
|---|---|---|
| Finding the arithmetic errors in the Session 1 log | High — spotted by re-adding the columns | Re-ran the stats script to confirm |
| `scripts/filter_labels.py` and its tests | High — written by Claude | Chose D8; ran it; checked the output counts |
| Diagnosing the polygon crash | High — identified the line format | Ran the script that surfaced it |
| `scripts/check_leakage.py` and its tests | High — written by Claude | Ran it; did the manual inspection |
| Interpreting the leakage results | Mixed — Claude's first conclusion was wrong | **Overturned it by inspecting the pictures** |
| Colab notebook | High — written by Claude | Set up the secret and Drive; ran and verified every step |

**AI errors caught during this session:**

3. The Session 1 log (drafted with Claude) contained arithmetic errors in the
   box-count totals (291 and 2,666 instead of 91 and 2,466). Because of this, it
   wrongly claimed the download matched Roboflow's figures.
4. Claude wrote `filter_labels.py` assuming every label line was a 5-number box,
   without checking a real file. Its own format check crashed on a polygon line. The
   tests had passed because Claude's test data only contained boxes.
5. Claude's first near-duplicate flagging rule compared each image's best match
   against random pairs, so it flagged 92% of images regardless. A confident-looking
   number that did not measure what it claimed to.
6. **Most significant:** Claude concluded from filename frame numbers that the split
   leaked badly ("real and large"), treating its reading of the filenames as fact.
   My inspection of the saved pairs showed that neighbouring and even identical frame
   numbers are different scenes. The conclusion was withdrawn and the original split
   kept.

Lesson (reinforcing Session 1): check AI claims against the actual data — here, by
looking at the images — before building on them. Errors 4–6 were each caught by a
check on real data, not by reviewing the code.

---

## Session 3 — 21–22 October 2026

**Deadline confirmed: Sunday 25 October 2026, midnight.** (An earlier note in this
log said Monday 19 October; that was wrong — see AI errors below.)

### 1. Plan to submission

| Day | Work |
|---|---|
| Wed 21 Oct | Baseline training started; README drafted |
| Thu 22 Oct | `alt_s640` and `alt_n960` runs; evaluate each on the test split |
| Fri 23 Oct | `benchmark.py` on the chosen model; pdoc HTML docs; pytest report |
| Sat 24 Oct | Write the 5–10 page report and the AI reflection |
| Sun 25 Oct | Buffer, final checks, submit well before midnight |

### 2. New code this session

All with pytest coverage. **53 tests pass** (16 filter, 12 leakage, 6 evaluate,
9 benchmark, 10 box stats).

| Script | Purpose |
|---|---|
| `scripts/evaluate.py` | Scores trained weights on the **test** split; writes `reports/metrics/<run>_test.{json,md}` |
| `scripts/benchmark.py` | Latency (split by stage), throughput in trays/min, parameter count and GFLOPs |
| `scripts/box_stats.py` | Per-class box sizes: median area, median box in px at a given inference size, count below the 32x32 px small-object threshold |

`notebooks/train_colab.ipynb` extended with three training runs (cells 8, 10, 11),
each followed by a test-split evaluation. A `COMMON` dict fixes the settings shared
by every run, so only the setting under test differs — this is what makes the
comparison fair, and it is worth stating in the report.

`README.md` written in full (previously only the AI Acknowledgement stub).

### 3. Class definitions — CONFIRMED by me, not inferred

The Session 1 open item is closed. The dataset ships no written definitions; these
describe what the annotations consistently mark:

| Class | Meaning |
|---|---|
| `loose-meat` | A small piece of meat sitting away from the main group of rashers |
| `packaging-error` | A crease in the film near the edge of the container |
| `twisted-meat` | A rasher folded or twisted rather than laid flat |
| `unsealed` | Film not sealed to the tray rim |

`packaging-error` and `unsealed` are both sealing faults at different stages: a
crease that risks a seal failing, versus a seal that has already failed.

### 4. Box sizes measured (`scripts/box_stats.py`, all splits, at imgsz 640)

| Class | Boxes | Median area | Median box at 640 px | Smallest / largest | Under 32x32 px |
|---|---|---|---|---|---|
| `loose-meat` | 135 | 1.59% | 53 x 121 px | 0.149% / 3.9% | 2 (1%) |
| `packaging-error` | 1,446 | 0.21% | 28 x 28 px | 0.001% / 4.8% | 661 (46%) |
| `twisted-meat` | 248 | 1.74% | 51 x 137 px | 0.592% / 3.8% | 0 (0%) |
| `unsealed` | 619 | 9.61% | 221 x 181 px | 0.140% / 17.6% | 2 (0%) |

Replaces the earlier unmeasured claim that "`packaging-error` boxes are small". The
measured position is sharper: its median box sits almost exactly on the small-object
threshold and 46% fall below it, while the class spans a ~4,800x range in area.
This is the measured justification for the `alt_n960` run.

*Still to run: the same command with `--imgsz 960`, to show how many boxes cross back
above the threshold.*

### 5. Roboflow count gap — EXPLAINED (Session 2 open item 4 closed)

Roboflow's Classes & Tags page (project level) against the counted export:

| Class | Roboflow | Counted | Gap |
|---|---|---|---|
| `loose-meat` | 135 | 135 | — |
| `packaging-error` | 1,645 | 1,446 | **−199** |
| `twisted-meat` | 249 | 248 | **−1** |
| `unsealed` | 619 | 619 | — |
| `wrinkle` | 18 | 18 | — |

Three classes match exactly, and the project and Version 1 both contain 1,955
images — so the Session 1 guess that "images were excluded from the version" is
**disproven**.

**Likely cause:** Version 1 uses `Stretch to 720x540`. Resizing shrinks every
annotation, and one too small to survive cannot be written to a label file. This
fits `packaging-error` directly: it is by far the smallest class and its smallest
surviving box covers about 4 square pixels.

**Honest limit:** this cannot be confirmed for the single `twisted-meat` box. Its
surviving boxes are all large, and any box that was dropped is by definition absent
from the data and cannot be measured. One box is also within the range of "something
irregular happened" and does not require an explanation.

### 6. Version 1 preprocessing — `Stretch to`, not `Fit`

Confirmed on the Roboflow Versions page. Stretching does not preserve aspect ratio,
so every image and every annotation is distorted identically. Consistent across
train and test, but a property of the dataset worth stating in the report.

### 7. Baseline trained and evaluated — `baseline_n640`

YOLOv8n, pretrained, 100 epochs, imgsz 640, batch 16, seed 0. Completed all 100
epochs (`results.csv` has 101 rows including the header). Weights in Drive at
`MyDrive/MMA3001/runs/baseline_n640/weights/best.pt`.

**Test split (76 images, 91 boxes) — these are the report numbers:**

| Class | Boxes | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| `loose-meat` | 3 | 0.747 | 0.667 | 0.913 | 0.275 |
| `packaging-error` | 29 | 0.593 | 0.414 | 0.513 | 0.132 |
| `twisted-meat` | 7 | 0.493 | 0.558 | 0.318 | 0.127 |
| `unsealed` | 52 | 0.719 | 0.827 | 0.777 | 0.385 |
| **all** | **91** | **0.638** | **0.616** | **0.630** | **0.230** |

Readings to carry into the report:

1. **The 0.913 for `loose-meat` is not evidence of anything.** 3 boxes in 2 images;
   one box moves it ~33 points. Report it, exclude it from conclusions.
2. **`packaging-error` recall 0.414** — misses ~60% of creases. The smallest class
   and the most important one for the stated purpose.
3. **`twisted-meat` is worst (0.318) despite large boxes.** Box size alone does not
   explain the ranking; `twisted-meat` has 236 training boxes against 1,372 for
   `packaging-error`. Object size and training-set size are confounded here and
   these three runs cannot separate them. Say so rather than implying otherwise.
4. **mAP50-95 (0.230) far below mAP50 (0.630)** — detections roughly right, boxes
   not tight. Worst for `packaging-error` (0.132), as expected for a 28 px box.
5. Evaluation printed ~6.1 ms preprocess / 5.7 ms inference / 3.1 ms postprocess.
   **Do not quote these** — they come from batched inference. Use `benchmark.py`.

### 8. Software version drift — now pinned

The October run used **Ultralytics 8.4.183, PyTorch 2.11.0+cu130**; the September
setup recorded 8.4.155 / cu128. Colab upgraded underneath the project.
`ultralytics==8.4.183` is now pinned in `requirements.txt` so all three runs are
compared on the same version. README versions updated.

### 9. Licensing

**Ultralytics is AGPL-3.0** (verified on their repository, 22 Oct 2026), with a
separate commercial Enterprise licence. Fine for this open coursework repo; a real
production deployment would need the position reviewed. Worth a line in the report
as a genuine engineering constraint.

*Open:* confirm the dataset licence on the **original** Roboflow Universe page, not
the fork — the fork lists this workspace as author, which would be the wrong
attribution. Project notes record CC BY 4.0, requiring credit to the original creator.

### 10. Repository state

Added this session: `scripts/evaluate.py`, `scripts/benchmark.py`,
`scripts/box_stats.py`, their three test files, the rewritten `README.md`, the
extended notebook, and the `ultralytics` pin.

### 11. Open items

1. **`alt_s640`** (YOLOv8s, 640) — running at the time of writing, cell 10.
   Then evaluate it.
2. **`alt_n960`** (YOLOv8n, 960) — cell 11, then evaluate.
3. Run `box_stats.py --imgsz 960`.
4. Run `benchmark.py` on the chosen model (GPU, and CPU for contrast).
5. Choose the final method and justify it in README section 4.
6. Generate pdoc HTML documentation and a pytest report.
7. Fill README sections 5 (speed), 9 (limitations and further work) and 10 (dataset
   licence).
8. Write the 5–10 page report and the AI reflection.
9. **Undecided scope question:** should the model output a pass/fail verdict per
   tray rather than a list of detections? Would strengthen the engineering framing
   and give the confidence-threshold work a purpose, at the cost of roughly an hour.
   Deferred, not rejected.

### 12. AI use record — Session 3

| Task | AI contribution | My contribution |
|---|---|---|
| Three-run experiment design | High — proposed baseline + two one-variable alternatives | Agreed the scope; ran every run |
| `evaluate.py`, `benchmark.py`, `box_stats.py` + tests | High — written by Claude | Ran them; checked the output against expectations |
| Notebook extension | High | Ran it; confirmed dataset counts and tests matched the laptop |
| README | High — drafted by Claude | **Corrected the class definitions, rejected the unmeasured "boxes are small" claim, and rejected the glare limitation as overstated** |
| Roboflow count-gap investigation | Medium — proposed the resize hypothesis | Pulled the Roboflow figures that disproved the earlier explanation |
| Interpreting baseline results | High | Ran the training and evaluation |

**AI errors caught during this session:**

7. **Claude's stored note had the wrong deadline** (Monday 19 October instead of
   Sunday 25 October) and it opened the session by stating the deadline had passed.
   Same failure mode as error 6: an unverified source stated as fact.
8. **First drafts of the notebook and `evaluate.py` had two defects**, caught by
   Claude before they were run: metrics were written to Colab storage that is wiped
   when a session ends, and an argument was passed to the evaluation step that was
   not verified as supported.
9. **Claude predicted that box size would predict the per-class ranking.** It holds
   for `unsealed` (largest boxes, best score) and `packaging-error` (smallest, poor
   recall), but `twisted-meat` scored worst despite large boxes. The prediction was
   partly wrong and the confound with training-set size was not anticipated.
10. **Claude initially asserted "`packaging-error` boxes are small"** in the README
    from an informal look at a few images in September. I rejected it as unmeasured;
    `box_stats.py` was then written to measure it. The measured answer is more
    useful than the claim would have been.

Lesson, consistent across all three sessions: measure rather than assert, and treat
any AI claim without a source as a hypothesis to test.

---


## Session 4 — 22–23 October 2026

All three runs trained, evaluated on the test split and benchmarked. Final method
chosen: **`baseline_n640`**. README sections 3, 4, 5, 6, 8 and 9 filled from measured
results. Report drafting started in a separate document (sections 1–3 drafted).

### 1. `alt_s640` — log file missing rows, run complete

`results.csv` had 95 rows instead of 101, which first suggested training stopped at
epoch 94. Reading the file showed it reaches epoch 100 but **epochs 78–80 and 85–87
are missing**. The time column (about 28 s per epoch) shows no gap or restart:
epoch 77 ends at 2,148 s and epoch 81 at 2,259 s, exactly four epochs later. Training
ran continuously; six rows were lost from the log. Most likely cause: a write lost
while saving directly to Google Drive. Not proven. The weights and test results do
not depend on this file. Confirmed by opening the file in Drive that the rows really
are missing (not a copy-paste artefact). `alt_n960`'s `results.csv` is complete
(101 rows).

### 2. Test-split results (76 images, 91 boxes)

| Run | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| `baseline_n640` | 0.638 | 0.616 | **0.630** | 0.230 |
| `alt_s640` | 0.419 | 0.545 | 0.491 | 0.225 |
| `alt_n960` | 0.611 | 0.635 | 0.561 | 0.232 |

| Class (mAP50) | Boxes | `baseline_n640` | `alt_s640` | `alt_n960` |
|---|---|---|---|---|
| `loose-meat` | 3 | 0.913 | 0.618 | 0.727 |
| `packaging-error` | 29 | 0.513 | 0.340 | 0.364 |
| `twisted-meat` | 7 | 0.318 | 0.287 | 0.397 |
| `unsealed` | 52 | 0.777 | 0.720 | 0.755 |

Readings:

1. **Bigger model worse.** `alt_s640` loses mainly on precision. Training losses
   fall to the end while validation losses rise from about epoch 65 — consistent
   with overfitting, not proof. Its validation mAP50 (~0.72) far exceeds its test
   score; `best.pt` is selected on validation, so only test scores are quoted.
2. **Bigger input did not help `packaging-error`** (0.513 → 0.364), the class it was
   run for. Likely reason: the images are 720 x 540, so 960 enlarges them without
   adding detail, and the smallest creases were already lost in the Roboflow resize.
   `box_stats.py --imgsz 960`: median box 43 x 43 px, share under 32 x 32 px down
   from 46% to 17% — the boxes got bigger, the score did not.
3. `twisted-meat` improved at 960 (0.318 → 0.397; mAP50-95 0.127 → 0.195) — 7 boxes,
   indicative only.
4. mAP50-95 nearly identical across runs (0.225–0.232).
5. Each configuration trained once; run-to-run variation unmeasured.

### 3. Speed and compute — `scripts/benchmark.py`

One image at a time, 10 warm-up images discarded, 3 passes over 76 test images.

| Run | Device | Parameters | GFLOPs | Median (ms) | 95th pct (ms) | Images/min |
|---|---|---|---|---|---|---|
| `baseline_n640` | T4 | 3,138,732 | 8.8 | 10.8 | 14.6 | 5,562 |
| `alt_s640` | T4 | 11,137,148 | 28.6 | 14.8 | 17.3 | 4,043 |
| `alt_n960` | T4 | 3,138,732 | 19.7 | 17.0 | 20.6 | 3,523 |
| `baseline_n640` | CPU (1 pass) | 3,138,732 | 8.8 | 106.2 | 123.7 | 565 |

960 costs 2.24x the GFLOPs of 640, matching the 2.25x pixel ratio. The script labels
throughput "trays/minute"; the README says images per minute, since some images show
two trays. Real line speed is unknown — an assumption in the report.

### 4. Decision — final method `baseline_n640`

Highest test mAP50, highest on `packaging-error`, and cheapest and fastest. Neither
alternative improved on it, so there was no accuracy-versus-cost trade-off.

### 5. Corrections

- README said "no augmentation". True of the dataset only: Ultralytics applies its
  default training-time augmentation (mosaic, colour, flips) in every run. Corrected.
- Optimiser recorded: AdamW chosen automatically, initial learning rate 0.00125,
  linear decay (read from the logged learning rate).

### 6. Repository changes

- `scripts/benchmark.py` fixed (see AI error 11).
- README: selected method, results for all runs, comparison, speed table, box sizes at
  960, layout, test/docs instructions, limitations and further work.
- To add: `reports/tests/pytest_report.txt` and `reports/docs/` (pdoc HTML).

### 7. Open items

1. Generate and commit the pytest report and pdoc docs.
2. Confirm the dataset licence on the **original** Roboflow Universe page (README
   section 10).
3. Report: sections 4–8, then cut to 5–10 pages. AI reflection written by me.
4. Fill the README AI Acknowledgement placeholders.
5. Check the repository is public before submitting.
6. Pass/fail-per-tray scope question — now listed as further work, not done.

### 8. AI use record — Session 4

| Task | AI contribution | My contribution |
|---|---|---|
| Diagnosing the 95-row `results.csv` | High | Pasted the file; confirmed the rows were missing in Drive |
| Interpreting the three runs | High | Ran every training, evaluation and benchmark |
| Benchmark fix | High — written by Claude | Applied, committed and pushed it; re-ran |
| README results and limitations | High — drafted by Claude | Reviewed and committed |
| Report sections 1–3 | High — drafted by Claude from the log and README | To review |

**AI errors caught during this session:**

11. **`benchmark.py` crashed on first use.** Claude had assumed `model.info(verbose=False)`
    returns the model summary; in Ultralytics 8.4.183 it returns `None`. The old call
    would also have costed the 960 run at 640 px. Fixed by computing parameters and
    GFLOPs directly at the real inference size, after reading the library source.
12. **Claude first read the 95-row `results.csv` as training stopped at epoch 94** and
    told me to resume. Reading the file showed missing rows, not an early stop;
    resuming would have failed. Same lesson: check the data before acting on a guess.
13. **The resolution hypothesis was wrong.** Claude expected 960 to help
    `packaging-error`; it made it worse. The source resolution (720 px) was not
    considered when the run was designed.

---
