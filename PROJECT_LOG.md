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
