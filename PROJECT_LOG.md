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
| | **Total** | **2,232** | **143** | **291** | **2,666** |

Grand total (2,666) matches Roboflow's reported class balance, so the download is
verified complete. Roboflow reported 1,645 for `packaging-error` against 1,446
counted here; unexplained, likely images excluded from the generated version.
Minor, but do not quote Roboflow's per-class figures in the report — quote the
counted ones.

**Counting pitfall (worth mentioning in the report):** an initial count using
`cat data/pork_v1/*/labels/*.txt | awk ...` gave badly wrong totals (535 instead of
2,666), because the label files do not end with a newline, so the last line of one
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
6. **Test set composition is skewed.** `unsealed` is 15% of training boxes but 56%
   of test boxes, so the headline test score will mostly reflect performance on
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
   silently undercounted (535 vs the true 2,666) because the label files lack
   trailing newlines. Caught only because the total disagreed with Roboflow's
   dashboard.

Both cases support the same lesson: AI output must be cross-checked against an
independent source before it is relied upon.

---
