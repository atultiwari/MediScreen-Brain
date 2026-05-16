# MediScreen-Brain — Training, Dataset & Reproducibility Guide

> Last updated: 2026-05-16
> Companion script: [`prepare_dataset.sh`](prepare_dataset.sh)
> Project analysis: [`ARCHITECTURE_ANALYSIS.md`](ARCHITECTURE_ANALYSIS.md)

This document collects everything the repository says (and does not say) about
how the bundled `pt_models/*.pt` and `onnx_models/*.onnx` weights were trained,
plus a reproducible recipe you can run with a public dataset.

---

## 1. What is shipped

| Artefact | Path | Notes |
|---|---|---|
| Class taxonomy | `utils/data.yaml` | 3 classes: `glioma`, `meningioma`, `pituitary` |
| Training script | `utils/train_all_models.py` | Trains 5 nano YOLO variants back-to-back |
| Inference weights (PyTorch) | `pt_models/Brain_Tumor.pt`, `pt_models/best.pt` | 6.0 MB / 5.2 MB — sizes consistent with YOLOv*n* |
| Inference weights (ONNX) | `onnx_models/Brain_Tumor.onnx`, `onnx_models/segmentation.onnx` | Exported via `utils/convert_pt_to_onnx.py` |
| Validation harness | `utils/batch_validate_models.py` | Sweep all 5 checkpoints across GPU & CPU |
| Slicing | `utils/nii_slice_to_images.py` | NIfTI → PNG (random axes; non-deterministic) |
| DICOM ingest | `utils/data_utils/dcm2nii_ui.py` | DICOM → NIfTI via `dcm2niix` |
| Label utilities | `utils/{batch_modify_labels,modify_labels,update_class_id_to_0,create_empty_txt_for_images,check_empty_txt}.py` | Post-process YOLO `.txt` label files |
| Dataset analytics | `utils/{analyze_valid_set,analyze_image_overlap,analyze_fn_fp,random_prune_images}.py` | Sanity checks, FN/FP audits |

---

## 2. Pipeline overview

```
┌────────┐   dcm2niix    ┌────────┐   nii→png    ┌──────────┐   labelling   ┌─────────────┐
│ DICOM  │ ────────────▶ │ NIfTI  │ ───────────▶ │ 2D PNG/  │ ────────────▶ │ YOLO dataset│
│ series │               │  .nii  │              │  JPG     │  (manual /    │ Train/, val/│
└────────┘               └────────┘              └──────────┘   semi-auto)  └─────┬───────┘
                                                                                  │
                                                                                  ▼
                                                                       ┌─────────────────────┐
                                                                       │ data.yaml           │
                                                                       │ (path, classes, nc) │
                                                                       └─────────┬───────────┘
                                                                                 │
                                                                                 ▼
                                                                       ┌─────────────────────┐
                                                                       │ train_all_models.py │
                                                                       │ → runs/detect/*/    │
                                                                       │   weights/best.pt   │
                                                                       └─────────┬───────────┘
                                                                                 │
                                                  ┌──────────────────────────────┼────────────────────────────┐
                                                  ▼                              ▼                            ▼
                                       batch_validate_models.py        convert_pt_to_onnx.py           Brain_Tumor_detection_ui.py
                                       (metrics on GPU + CPU)         (PT → ONNX export)              (GUI inference)
```

---

## 3. Training the model

### 3.1 Environment

Training is heavy — use a CUDA-capable Linux/Windows box (or macOS for tiny
test runs on MPS). On a MacBook the script *runs* but is slow.

```bash
# Re-use the GUI venv created by ./run.sh
source .venv/bin/activate

# Add training-only extras (already covered by the base install,
# but make explicit):
pip install "ultralytics==8.3.168" "torch>=2.2,<2.6"
```

### 3.2 Configure the dataset

Edit `utils/data.yaml` so `path:` points at *your* YOLO dataset root. The
expected layout (matches Ultralytics defaults):

```text
<dataset_root>/
├── Train/
│   ├── images/   *.png|*.jpg
│   └── labels/   *.txt    # YOLO format: <class_id> <cx> <cy> <w> <h>  (normalised 0–1)
└── val/
    ├── images/
    └── labels/
```

Class IDs follow the order in `data.yaml/names`:
- `0 → glioma`
- `1 → meningioma`
- `2 → pituitary`

### 3.3 Train

```bash
cd utils
python train_all_models.py
```

This trains **yolov8n → yolov9t → yolov10n → yolo11n → yolo26n** sequentially
with the in-script hyperparameters:

| Knob | Value | Where to change |
|---|---|---|
| Image size | 512 | `IMG_SIZE` in `train_all_models.py` |
| Epochs | 100 | `EPOCHS` |
| Batch size | 16 | `BATCH_SIZE` |
| Optimizer / LR / scheduler / augmentations | Ultralytics defaults | Pass extra kwargs to `model.train(...)` inside `train_model()` |

Outputs land in `utils/runs/detect/train_<model>/weights/best.pt`.

### 3.4 Validate

```bash
cd utils
python batch_validate_models.py
```

This iterates every `runs/detect/train_*/weights/best.pt`, runs Ultralytics
`model.val()` on the `val/` split, and prints mAP@50 / mAP@50-95 / per-class
P/R for each model on GPU **and** CPU.

### 3.5 Export to ONNX

```bash
cd utils
python convert_pt_to_onnx.py
```

> ⚠️ Note: `convert_pt_to_onnx.py` has hard-coded absolute Windows paths at
> the top — repoint `PT_DIR` and `ONNX_DIR` before running. Training uses
> `imgsz=512` but the exporter calls `imgsz=640`. Align these if you depend
> on exact tensor shapes.

### 3.6 Drop the new weights into the GUI

Copy `best.pt` from the run you like into `pt_models/` (and its ONNX export
into `onnx_models/`). The GUI scans both folders on launch, so no code
change is needed.

---

## 4. Public datasets compatible with this project

The original `MRI_2D_DATA_USE_3` referenced by `utils/data.yaml` is **not
bundled, not linked, and not cited**. The author's filename pattern in
`utils/YOLOONNX.py` (`BrainTumorYolov8_copy`) strongly suggests a
Roboflow-style YOLO export. The substitutes below match either the
3-class taxonomy, the YOLO bbox format, or both.

| Dataset | Taxonomy | Format | Size | Best fit | License | Where |
|---|---|---|---|---|---|---|
| **Cheng et al. brain tumor dataset (figshare, 2017)** | glioma / meningioma / pituitary | `.mat` (image + tumor mask) | 3,064 T1-CE slices, 233 patients | ✅ Exact taxonomy. Requires mask → bbox conversion. | CC-BY 4.0 | https://figshare.com/articles/dataset/brain_tumor_dataset/1512427 |
| **Brain Tumor MRI Dataset (Nickparvar, Kaggle)** | glioma / meningioma / pituitary / no_tumor | Class folders (classification) | 7,023 images | ⚠️ Taxonomy matches +1. No bboxes — needs labelling. | CC0 | https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset |
| **Brain Tumor Classification (Sartaj, Kaggle)** | glioma / meningioma / pituitary / no_tumor | Class folders | 3,264 images | ⚠️ Same as above. | (per-dataset) | https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri |
| **Roboflow Universe — "brain tumor detection"** (multiple authors) | varies; several with glioma / meningioma / pituitary | YOLO `.txt` (bbox) | varies | ✅ Drop-in for training. Quality varies — inspect first. | Per-author (often CC-BY 4.0) | https://universe.roboflow.com/search?q=brain+tumor |
| **BraTS 2021 / 2023** | HGG / LGG (+ sub-region masks: enhancing, edema, necrosis) | NIfTI volumes + multi-label masks | ~2,000 cases | ❌ Different taxonomy. Useful only after class remapping. | Restricted (research) | https://www.synapse.org/brats |
| **TCIA — REMBRANDT / TCGA-GBM / TCGA-LGG** | by molecular subtype | DICOM | thousands | ❌ Mostly classification by genomics, not by tumor *type* in this taxonomy. | TCIA terms | https://www.cancerimagingarchive.net |
| **Brain MRI Segmentation (LGG, Mateusz Buda, Kaggle)** | LGG mask | PNG + mask | 3,929 slices, 110 patients | ❌ Single class only. Useful for the `segmentation.onnx` model lineage. | (per-dataset) | https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation |

### Recommended substitute

For a **drop-in, fully reproducible run** with this repo:

1. **Roboflow Universe** — search "brain tumor glioma meningioma pituitary",
   pick a version with YOLOv8 export and the 3-class taxonomy, download the
   YOLO export zip, point `utils/data.yaml` at it.
2. **Cheng et al. figshare** — if you want a citable, well-known source.
   You will need a one-time script to convert the per-image `tumorMask`
   (provided as `.mat`) into a YOLO bbox (`min/max` of mask non-zero region).
   A starter snippet:
   ```python
   import h5py, numpy as np
   with h5py.File(p, 'r') as f:
       label = int(f['cjdata/label'][0,0])  # 1=meningioma 2=glioma 3=pituitary
       img   = np.array(f['cjdata/image']).T
       mask  = np.array(f['cjdata/tumorMask']).T
   ys, xs = np.where(mask > 0)
   cx = (xs.min()+xs.max())/2 / img.shape[1]
   cy = (ys.min()+ys.max())/2 / img.shape[0]
   w  = (xs.max()-xs.min())    / img.shape[1]
   h  = (ys.max()-ys.min())    / img.shape[0]
   # YOLO id (remap to 0-indexed glioma/meningioma/pituitary)
   ```
3. **Nickparvar Kaggle** — easiest to download (`kagglehub` works in this
   venv) but classification-only; you'd be training **classification**, not
   **detection**. Not recommended unless you re-label.

---

## 5. Gaps, impact & TODO

### 5.1 Reproducibility gaps

| # | Missing artefact | Impact | Suggested action | Priority |
|---|---|---|---|---|
| 1 | **Dataset citation & download link** | Cannot reproduce training; can't verify what the bundled weights saw | Add `DATASET.md` with DOI, license, download command, version. Pick one of §4. | 🔴 P0 |
| 2 | **Splits manifest** (which files in train vs val, fixed seed) | Each re-run gets a different split → metrics non-comparable | Commit a CSV/JSON manifest under `data/splits/` + record the random seed | 🔴 P0 |
| 3 | **Patient-level split policy** | If slices from one patient land in both train and val, metrics are leaked and optimistic | Enforce patient-id–aware grouping in the splitter | 🔴 P0 |
| 4 | **Reported metrics table** (mAP@50, mAP@50-95, per-class P/R, latency) | No baseline to compare against | Wire `batch_validate_models.py` output into `docs/METRICS.md`, commit per-release | 🟠 P1 |
| 5 | **Random seed** | Two clean training runs disagree | Pass `seed=` to `model.train()`; record in run name | 🟠 P1 |
| 6 | **MRI sequence(s) used** (T1 / T1c / T2 / FLAIR) | Model may underperform on the wrong modality without warning | Document in `DATASET.md`; consider gating GUI input by sequence | 🟠 P1 |
| 7 | **Annotation provenance** (annotator credentials, tool, IRR) | Cannot vouch for label quality; blocks clinical claims | Add `ANNOTATION.md` with annotator count, qualifications, inter-rater agreement | 🟠 P1 |
| 8 | **Training preprocessing pipeline** (intensity normalisation, skull-strip y/n, percentile clip values) | Train/inference mismatch — runtime applies p99 clip, training preprocessing is opaque | Encode preprocessing in a single `preprocess.py` shared by training and `Brain_Tumor_detection_ui.py` | 🟠 P1 |
| 9 | **Image size inconsistency** (train=512, export=640) | Different padding/letterbox behaviour at inference; subtle accuracy drift | Pick one; document the choice | 🟠 P1 |
| 10 | **Hardware / wall-clock baselines** | New contributors can't sanity-check training speed | Add a small "expected ~X min/epoch on RTX 3060" table | 🟡 P2 |
| 11 | **Ablations** (img size, batch, epochs, augmentations) | No defence of chosen hyperparameters | Run a small ablation grid; commit results | 🟡 P2 |
| 12 | **Class balance + per-class slice counts** | Possible class imbalance hidden; affects loss weighting | Print histogram from `analyze_valid_set.py`; commit | 🟡 P2 |
| 13 | **Paper / preprint** | README says "DOI not published yet" — methods aren't peer-reviewed | Link the preprint when available | 🟡 P2 |
| 14 | **Per-model checkpoints** (currently only `best.pt` and `Brain_Tumor.pt`) | Can't compare YOLO architectures qualitatively | Publish each of the 5 trained `best.pt` to GitHub Releases | 🟡 P2 |

### 5.2 Code gaps (training-pipeline only)

| # | Issue | Impact | Suggested action | Priority |
|---|---|---|---|---|
| 1 | `utils/data.yaml` `path:` is `H:\...` (Windows absolute) | Training breaks on any other machine | Use a relative path or env var (`${MEDISCREEN_DATA_ROOT}`) | 🔴 P0 |
| 2 | `utils/convert_pt_to_onnx.py` hard-codes absolute Windows paths | Export breaks elsewhere | Take `--pt-dir` / `--onnx-dir` CLI args | 🔴 P0 |
| 3 | `utils/nii_slice_to_images.py` picks **random** axes per file | Non-reproducible slice generation | Add deterministic axis flags and a fixed RNG seed | 🟠 P1 |
| 4 | `__main__` blocks of utility scripts hard-code `H:\...` inputs | Can't run any of them out-of-the-box | Convert each to `argparse` | 🟠 P1 |
| 5 | No CI for training-pipeline scripts | Silent regressions on data prep | Add a tiny smoke test: 2 NIfTI volumes → slice → split → 1-epoch train | 🟡 P2 |
| 6 | No tests at all | Refactoring blocked | Pin behaviour with `pytest` + snapshot tests on `data_test/MRBrainTumor2.nii.gz` | 🟡 P2 |

### 5.3 Clinical / safety gaps

| # | Issue | Impact | Suggested action | Priority |
|---|---|---|---|---|
| 1 | No external-test-set evaluation | Performance on out-of-distribution scanners/protocols unknown | Hold out an entire institution / scanner vendor | 🔴 P0 |
| 2 | No bias analysis (age, sex, scanner) | Risk of subgroup failure | Stratified metrics in `METRICS.md` | 🟠 P1 |
| 3 | No failure-mode catalogue | Hard to advise users on limitations | Add `LIMITATIONS.md` with known failure modes (low-contrast lesions, post-op cavities, motion artefacts, ...) | 🟠 P1 |
| 4 | "Not for clinical diagnosis" disclaimer is only in PDF reports | Users may not see it in the GUI | Surface it in the splash / "About" dialog | 🟡 P2 |

Legend: 🔴 P0 = blocks reproducibility/credibility · 🟠 P1 = should fix before next public release · 🟡 P2 = nice-to-have

---

## 6. One-line recipe with `prepare_dataset.sh`

```bash
# 0. Have a directory of NIfTI volumes, e.g. ./raw_nii/
./prepare_dataset.sh slice   --input ./raw_nii --output ./dataset_raw --axes axial
./prepare_dataset.sh split   --input ./dataset_raw --output ./dataset --val-ratio 0.2 --seed 42
./prepare_dataset.sh update-yaml --dataset ./dataset
./prepare_dataset.sh validate    --dataset ./dataset
# 1. Train
source .venv/bin/activate
cd utils && python train_all_models.py
```

See `./prepare_dataset.sh --help` for full subcommand reference.

---

## 7. Citation

When/if you publish results trained with this pipeline, also cite:

- Ultralytics YOLO (v8/v9/v10/v11): https://github.com/ultralytics/ultralytics
- The original Cheng dataset (if used): *Cheng J. et al. (2015). Enhanced
  Performance of Brain Tumor Classification via Tumor Region Augmentation
  and Partition. PLoS ONE.*
- MediScreen-Brain itself (per the upstream README — citation TBD when DOI
  is issued).
