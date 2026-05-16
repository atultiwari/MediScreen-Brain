#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# MediScreen-Brain — dataset preparation pipeline (macOS / Linux)
#
# Subcommand-driven helper that wires together the NIfTI slicing, DICOM
# conversion, train/val split, YOLO dataset validation, and data.yaml
# rewriting steps. Designed to take you from raw MRI to an Ultralytics-ready
# `Train/` + `val/` layout that `utils/train_all_models.py` can consume.
#
# Usage:
#   ./prepare_dataset.sh <subcommand> [options]
#
# Subcommands:
#   slice         Convert a folder of NIfTI volumes (.nii / .nii.gz) into
#                 2D PNG slices (deterministic; seed-controlled).
#   dicom         Convert a folder of DICOM series into NIfTI via dcm2niix.
#   split         Split a flat folder of images (+ optional YOLO labels)
#                 into Train/ and val/ subsets with patient-aware grouping
#                 when possible.
#   update-yaml   Rewrite utils/data.yaml so `path:` points at a new dataset.
#   validate      Sanity-check a YOLO dataset (images/labels parity, class
#                 IDs in range, empty-label warnings, image-format check).
#   download      Print instructions for fetching public substitute datasets
#                 (Roboflow Universe, Cheng et al. figshare, Kaggle).
#   pipeline      Run slice → split → update-yaml → validate end-to-end.
#   help          Show this message.
#
# Global notes:
#   - Run inside the project root (the script enforces this).
#   - The script reuses the project's .venv (created by ./run.sh). If it does
#     not exist, the script will offer to create it.
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR=".venv"

# -------- Logging ----------------------------------------------------------
COLOR_BLUE="\033[1;34m"
COLOR_GREEN="\033[1;32m"
COLOR_YELLOW="\033[1;33m"
COLOR_RED="\033[1;31m"
COLOR_RESET="\033[0m"
log()  { printf "${COLOR_BLUE}[prepare]${COLOR_RESET} %s\n" "$*"; }
ok()   { printf "${COLOR_GREEN}[prepare]${COLOR_RESET} %s\n" "$*"; }
warn() { printf "${COLOR_YELLOW}[prepare]${COLOR_RESET} %s\n" "$*" >&2; }
die()  { printf "${COLOR_RED}[prepare]${COLOR_RESET} %s\n" "$*" >&2; exit "${2:-2}"; }

# -------- Help -------------------------------------------------------------
print_help() {
  sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
}

# -------- Venv activation --------------------------------------------------
activate_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    die "No venv at $VENV_DIR. Run './run.sh --setup' first."
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
}

# ===========================================================================
# Subcommand: slice
# ===========================================================================
cmd_slice() {
  local INPUT="" OUTPUT="" AXES="axial" STEP=2 SEED=42 NORMALIZE="percentile"
  local DRY_RUN=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --input)      INPUT="$2"; shift 2 ;;
      --output)     OUTPUT="$2"; shift 2 ;;
      --axes)       AXES="$2"; shift 2 ;;     # axial | sagittal | coronal | all
      --step)       STEP="$2"; shift 2 ;;     # take every Nth slice (default 2)
      --seed)       SEED="$2"; shift 2 ;;
      --normalize)  NORMALIZE="$2"; shift 2 ;; # minmax | percentile
      --dry-run)    DRY_RUN=1; shift ;;
      -h|--help)
        cat <<EOF
slice — NIfTI volumes → 2D PNG slices (deterministic).

Options:
  --input    <dir>    folder containing *.nii or *.nii.gz (recursive)
  --output   <dir>    destination folder for PNGs
  --axes     <ax>     axial|sagittal|coronal|all (default: axial)
  --step     <N>      keep every Nth slice in the chosen axis (default: 2)
  --seed     <N>      RNG seed for tie-breaking (default: 42)
  --normalize <m>     percentile (clip @ p99 then min-max) | minmax  (default: percentile)
  --dry-run           list what would be produced; write nothing

Output filename pattern: <volume_stem>_<axis>_<index>.png
EOF
        return 0 ;;
      *) die "slice: unknown flag $1" 1 ;;
    esac
  done
  [[ -n "$INPUT"  ]] || die "slice: --input is required"
  [[ -n "$OUTPUT" ]] || die "slice: --output is required"
  [[ -d "$INPUT"  ]] || die "slice: input dir not found: $INPUT"

  activate_venv
  log "slice: input=$INPUT  output=$OUTPUT  axes=$AXES  step=$STEP  seed=$SEED  norm=$NORMALIZE  dry=$DRY_RUN"

  INPUT_ABS="$(cd "$INPUT" && pwd)"
  python - <<PYEOF
import os, sys, random
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image

INPUT     = Path(r"$INPUT_ABS")
OUTPUT    = Path(r"$OUTPUT")
AXES      = "$AXES"
STEP      = int("$STEP")
SEED      = int("$SEED")
NORMALIZE = "$NORMALIZE"
DRY_RUN   = bool(int("$DRY_RUN"))

random.seed(SEED); np.random.seed(SEED)
AXIS_MAP = {"sagittal": 0, "coronal": 1, "axial": 2}
if AXES == "all":
    axes = [0, 1, 2]
else:
    axes = [AXIS_MAP[a] for a in AXES.split(",")]

if not DRY_RUN:
    OUTPUT.mkdir(parents=True, exist_ok=True)

def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    if NORMALIZE == "percentile":
        hi = np.percentile(arr, 99) if arr.size else 1.0
        arr = np.clip(arr, 0, hi if hi > 0 else 1.0)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(arr, dtype=np.uint8)
    return (((arr - lo) / (hi - lo)) * 255).astype(np.uint8)

niis = sorted([*INPUT.rglob("*.nii"), *INPUT.rglob("*.nii.gz")])
if not niis:
    sys.exit("slice: no NIfTI files found under " + str(INPUT))

total = 0
for nii in niis:
    try:
        data = nib.load(str(nii)).get_fdata()
    except Exception as e:
        print(f"  ! skip {nii.name}: {e}", file=sys.stderr); continue
    while data.ndim > 3:
        data = data[..., 0]
    if data.ndim != 3:
        print(f"  ! skip {nii.name}: shape {data.shape}", file=sys.stderr); continue

    stem = nii.name.removesuffix(".gz").removesuffix(".nii")
    for ax in axes:
        depth = data.shape[ax]
        for idx in range(0, depth, STEP):
            sl = np.take(data, idx, axis=ax)
            label = {0:"sagittal",1:"coronal",2:"axial"}[ax]
            out = OUTPUT / f"{stem}_{label}_{idx:04d}.png"
            if DRY_RUN:
                print("  would write", out)
            else:
                Image.fromarray(normalize(sl)).save(out)
            total += 1

print(f"slice: produced {total} PNGs from {len(niis)} volume(s)")
PYEOF
  ok "slice: done"
}

# ===========================================================================
# Subcommand: dicom
# ===========================================================================
cmd_dicom() {
  local INPUT="" OUTPUT=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --input)  INPUT="$2"; shift 2 ;;
      --output) OUTPUT="$2"; shift 2 ;;
      -h|--help)
        cat <<EOF
dicom — DICOM series → NIfTI volumes via dcm2niix.

Options:
  --input  <dir>   folder containing DICOM series (recursive)
  --output <dir>   destination folder for *.nii.gz volumes

Requires dcm2niix on PATH. Install on macOS: 'brew install dcm2niix'.
EOF
        return 0 ;;
      *) die "dicom: unknown flag $1" 1 ;;
    esac
  done
  [[ -n "$INPUT"  ]] || die "dicom: --input is required"
  [[ -n "$OUTPUT" ]] || die "dicom: --output is required"
  command -v dcm2niix >/dev/null 2>&1 || die "dcm2niix not found. brew install dcm2niix"

  mkdir -p "$OUTPUT"
  log "dicom: dcm2niix -z y -f %p_%s -o $OUTPUT $INPUT"
  dcm2niix -z y -f "%p_%s" -o "$OUTPUT" "$INPUT"
  ok "dicom: done"
}

# ===========================================================================
# Subcommand: split
# ===========================================================================
cmd_split() {
  local INPUT="" OUTPUT="" VAL_RATIO=0.2 SEED=42 PATIENT_GROUP=1 ALLOW_NO_LABELS=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --input)         INPUT="$2"; shift 2 ;;
      --output)        OUTPUT="$2"; shift 2 ;;
      --val-ratio)     VAL_RATIO="$2"; shift 2 ;;
      --seed)          SEED="$2"; shift 2 ;;
      --no-patient)    PATIENT_GROUP=0; shift ;;
      --allow-no-labels) ALLOW_NO_LABELS=1; shift ;;
      -h|--help)
        cat <<EOF
split — Split flat images/labels into Train/ and val/ subsets.

Expected input layout:
  <INPUT>/
    images/   *.png|*.jpg|*.jpeg
    labels/   *.txt              (YOLO format; same stem as the image)

  OR a flat directory of images (no labels), if --allow-no-labels is passed.

Output layout (matches utils/data.yaml expectations):
  <OUTPUT>/
    Train/{images,labels}/
    val/{images,labels}/
  <OUTPUT>/splits.csv             (manifest: stem,patient_id,split)

Options:
  --input       <dir>   prepared dataset folder
  --output      <dir>   destination dataset root
  --val-ratio   <f>     fraction reserved for val (default: 0.2)
  --seed        <N>     RNG seed (default: 42)
  --no-patient          do not attempt patient-level grouping
  --allow-no-labels     proceed without a labels/ subdir (slice-only datasets)

Patient grouping heuristic: stems matching /^([^_]+)_/ are grouped by the
leading token (e.g. "PAT001_axial_0123.png" → patient "PAT001"). Override
with --no-patient.
EOF
        return 0 ;;
      *) die "split: unknown flag $1" 1 ;;
    esac
  done
  [[ -n "$INPUT"  ]] || die "split: --input is required"
  [[ -n "$OUTPUT" ]] || die "split: --output is required"
  [[ -d "$INPUT"  ]] || die "split: input dir not found: $INPUT"

  activate_venv
  log "split: input=$INPUT  output=$OUTPUT  val_ratio=$VAL_RATIO  seed=$SEED  patient=$PATIENT_GROUP"

  python - <<PYEOF
import csv, random, re, sys, shutil
from pathlib import Path

INPUT  = Path(r"$INPUT")
OUTPUT = Path(r"$OUTPUT")
VAL    = float("$VAL_RATIO")
SEED   = int("$SEED")
PATIENT = bool(int("$PATIENT_GROUP"))
ALLOW_NO_LABELS = bool(int("$ALLOW_NO_LABELS"))

random.seed(SEED)

# Find images dir
img_dir = INPUT / "images" if (INPUT / "images").is_dir() else INPUT
lbl_dir = INPUT / "labels" if (INPUT / "labels").is_dir() else None
if lbl_dir is None and not ALLOW_NO_LABELS:
    sys.exit("split: no labels/ subdir found. Pass --allow-no-labels if intentional.")

exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
if not images:
    sys.exit(f"split: no images under {img_dir}")

def patient_of(stem: str) -> str:
    if not PATIENT:
        return stem
    m = re.match(r"^([^_]+)_", stem)
    return m.group(1) if m else stem

# Group by patient, then split groups
groups = {}
for img in images:
    groups.setdefault(patient_of(img.stem), []).append(img)
group_keys = sorted(groups)
random.shuffle(group_keys)
n_val = max(1, int(round(len(group_keys) * VAL)))
val_keys = set(group_keys[:n_val])

for split in ("Train", "val"):
    (OUTPUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / split / "labels").mkdir(parents=True, exist_ok=True)

manifest_rows = []
copied_imgs = 0
copied_lbls = 0
missing_lbls = 0
for pkey, files in groups.items():
    split = "val" if pkey in val_keys else "Train"
    for img in files:
        dst_img = OUTPUT / split / "images" / img.name
        shutil.copy2(img, dst_img); copied_imgs += 1
        if lbl_dir is not None:
            lbl_src = lbl_dir / (img.stem + ".txt")
            if lbl_src.exists():
                shutil.copy2(lbl_src, OUTPUT / split / "labels" / lbl_src.name)
                copied_lbls += 1
            else:
                missing_lbls += 1
        manifest_rows.append((img.stem, pkey, split))

with (OUTPUT / "splits.csv").open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["stem", "patient_id", "split"]); w.writerows(manifest_rows)

print(f"split: groups={len(groups)} train_groups={len(group_keys)-n_val} val_groups={n_val}")
print(f"split: images_copied={copied_imgs} labels_copied={copied_lbls} missing_labels={missing_lbls}")
print(f"split: manifest -> {OUTPUT/'splits.csv'}")
PYEOF
  ok "split: done"
}

# ===========================================================================
# Subcommand: update-yaml
# ===========================================================================
cmd_update_yaml() {
  local DATASET="" YAML_PATH="utils/data.yaml"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset) DATASET="$2"; shift 2 ;;
      --yaml)    YAML_PATH="$2"; shift 2 ;;
      -h|--help)
        cat <<EOF
update-yaml — Repoint utils/data.yaml at a new dataset root.

Options:
  --dataset <dir>   absolute or relative path to the dataset root
                    (folder containing Train/ and val/)
  --yaml    <file>  yaml file to update (default: utils/data.yaml)

Class names and class count (nc=3, glioma/meningioma/pituitary) are left
untouched; edit them by hand if you intend to train on a different taxonomy.
EOF
        return 0 ;;
      *) die "update-yaml: unknown flag $1" 1 ;;
    esac
  done
  [[ -n "$DATASET" ]] || die "update-yaml: --dataset is required"
  [[ -d "$DATASET" ]] || die "update-yaml: dataset dir not found: $DATASET"
  [[ -f "$YAML_PATH" ]] || die "update-yaml: yaml file not found: $YAML_PATH"

  DATASET_ABS="$(cd "$DATASET" && pwd)"
  log "update-yaml: setting path: $DATASET_ABS  in $YAML_PATH"

  python - <<PYEOF
from pathlib import Path
p = Path(r"$YAML_PATH")
lines = p.read_text().splitlines()
out = []
seen_path = False
for line in lines:
    if line.startswith("path:"):
        out.append(f"path: {r'$DATASET_ABS'}"); seen_path = True
    else:
        out.append(line)
if not seen_path:
    out.insert(0, f"path: {r'$DATASET_ABS'}")
p.write_text("\n".join(out) + "\n")
print("update-yaml: wrote", p)
PYEOF
  ok "update-yaml: done"
}

# ===========================================================================
# Subcommand: validate
# ===========================================================================
cmd_validate() {
  local DATASET=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --dataset) DATASET="$2"; shift 2 ;;
      -h|--help)
        cat <<EOF
validate — Verify a YOLO dataset layout.

Checks:
  - Train/ and val/ directories exist with images/ and labels/ subdirs
  - Every image has a matching .txt label (or a known-empty marker)
  - Class IDs in labels are within [0, nc-1] from utils/data.yaml
  - Image files open without errors
  - Reports class-balance histogram

Options:
  --dataset <dir>   dataset root to check
EOF
        return 0 ;;
      *) die "validate: unknown flag $1" 1 ;;
    esac
  done
  [[ -n "$DATASET" ]] || die "validate: --dataset is required"
  [[ -d "$DATASET" ]] || die "validate: dataset dir not found: $DATASET"

  activate_venv

  python - <<PYEOF
import sys, yaml
from pathlib import Path
from PIL import Image

DATASET = Path(r"$DATASET")
yaml_path = Path("utils/data.yaml")
nc = 3
names = ["glioma", "meningioma", "pituitary"]
if yaml_path.exists():
    try:
        cfg = yaml.safe_load(yaml_path.read_text()) or {}
        nc = int(cfg.get("nc", nc))
        names = cfg.get("names", names)
    except Exception:
        pass

problems = 0
hist = {i: 0 for i in range(nc)}
empty_labels = 0
missing_labels = 0
checked_images = 0

for split in ("Train", "val"):
    sp = DATASET / split
    if not sp.is_dir():
        print(f"  ! missing split dir: {sp}"); problems += 1; continue
    img_dir = sp / "images"
    lbl_dir = sp / "labels"
    if not img_dir.is_dir():
        print(f"  ! missing images dir: {img_dir}"); problems += 1; continue
    if not lbl_dir.is_dir():
        print(f"  ! missing labels dir: {lbl_dir}"); problems += 1; continue

    for img in img_dir.iterdir():
        if img.suffix.lower() not in {".png",".jpg",".jpeg",".bmp",".tiff",".webp"}:
            continue
        checked_images += 1
        try:
            Image.open(img).verify()
        except Exception as e:
            print(f"  ! corrupt image {img}: {e}"); problems += 1

        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            missing_labels += 1; continue
        rows = lbl.read_text().strip().splitlines()
        if not rows:
            empty_labels += 1; continue
        for r in rows:
            parts = r.split()
            if len(parts) != 5:
                print(f"  ! malformed label {lbl}: {r}"); problems += 1; continue
            try:
                cid = int(parts[0])
                coords = list(map(float, parts[1:]))
            except ValueError:
                print(f"  ! non-numeric label {lbl}: {r}"); problems += 1; continue
            if not (0 <= cid < nc):
                print(f"  ! class id out of range in {lbl}: {cid}"); problems += 1
            else:
                hist[cid] += 1
            if not all(0.0 <= c <= 1.0 for c in coords):
                print(f"  ! coords out of [0,1] in {lbl}: {coords}"); problems += 1

print()
print(f"validate: checked_images={checked_images}")
print(f"validate: missing_labels={missing_labels}  empty_labels={empty_labels}")
print(f"validate: class histogram:")
for i, n in hist.items():
    print(f"           {i:>2} {names[i] if i < len(names) else '?':<12}  {n}")
print()
sys.exit(1 if problems else 0)
PYEOF
  if [[ $? -eq 0 ]]; then
    ok "validate: dataset OK"
  else
    die "validate: dataset has problems (see above)" 3
  fi
}

# ===========================================================================
# Subcommand: download
# ===========================================================================
cmd_download() {
  cat <<'EOF'
download — recommended public substitutes for the (unpublished) MRI_2D_DATA_USE_3.

This subcommand does NOT fetch anything automatically because every option
requires user-side authentication or license acceptance.

1) Roboflow Universe (recommended — already YOLO-formatted)
   Browse: https://universe.roboflow.com/search?q=brain+tumor
   Pick a 3-class (glioma/meningioma/pituitary) project, click "Download",
   choose "YOLOv8" format, extract its zip, then:

       ./prepare_dataset.sh split        --input ./roboflow_extract --output ./dataset
       ./prepare_dataset.sh update-yaml  --dataset ./dataset
       ./prepare_dataset.sh validate     --dataset ./dataset

2) Cheng et al. brain tumor dataset (citable; needs mask→bbox conversion)
   Project page: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
   Files are .mat (HDF5) — see TRAINING.md §4 for a starter conversion snippet.

3) Kaggle: masoudnickparvar/brain-tumor-mri-dataset
   pip install kagglehub
   python -c "import kagglehub; print(kagglehub.dataset_download('masoudnickparvar/brain-tumor-mri-dataset'))"
   (classification format — class folders only, no bboxes; not drop-in.)

4) BraTS 2023 (NIfTI segmentations — different taxonomy)
   https://www.synapse.org/brats — research access only.

See TRAINING.md §4 for a comparison table.
EOF
}

# ===========================================================================
# Subcommand: pipeline
# ===========================================================================
cmd_pipeline() {
  local NII_IN="" RAW_OUT="" DATASET="" AXES="axial" STEP=2 VAL_RATIO=0.2 SEED=42
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --nii)        NII_IN="$2"; shift 2 ;;
      --raw)        RAW_OUT="$2"; shift 2 ;;
      --dataset)    DATASET="$2"; shift 2 ;;
      --axes)       AXES="$2"; shift 2 ;;
      --step)       STEP="$2"; shift 2 ;;
      --val-ratio)  VAL_RATIO="$2"; shift 2 ;;
      --seed)       SEED="$2"; shift 2 ;;
      -h|--help)
        cat <<EOF
pipeline — slice → split → update-yaml → validate (one-shot).

Options:
  --nii        <dir>   folder of NIfTI volumes (input)
  --raw        <dir>   intermediate folder for sliced PNGs
                       (defaults to <dataset>/_raw_slices)
  --dataset    <dir>   final dataset root (will contain Train/, val/)
  --axes       <ax>    axial|sagittal|coronal|all (default: axial)
  --step       <N>     slice every Nth (default: 2)
  --val-ratio  <f>     val fraction (default: 0.2)
  --seed       <N>     RNG seed (default: 42)

Note: this pipeline produces slices WITHOUT labels. You must annotate them
(or copy in existing YOLO .txt label files matching the image stems) before
running train_all_models.py.
EOF
        return 0 ;;
      *) die "pipeline: unknown flag $1" 1 ;;
    esac
  done
  [[ -n "$NII_IN"   ]] || die "pipeline: --nii is required"
  [[ -n "$DATASET"  ]] || die "pipeline: --dataset is required"
  [[ -z "$RAW_OUT"  ]] && RAW_OUT="$DATASET/_raw_slices"

  cmd_slice       --input "$NII_IN" --output "$RAW_OUT" --axes "$AXES" --step "$STEP" --seed "$SEED"
  cmd_split       --input "$RAW_OUT" --output "$DATASET" --val-ratio "$VAL_RATIO" --seed "$SEED" --allow-no-labels
  cmd_update_yaml --dataset "$DATASET"
  warn "pipeline: dataset has NO labels yet — annotate Train/labels/ and val/labels/ before training."
  cmd_validate    --dataset "$DATASET" || warn "validate flagged issues (likely missing labels — expected)"
  ok "pipeline: done"
}

# ===========================================================================
# Dispatch
# ===========================================================================
[[ $# -ge 1 ]] || { print_help; exit 0; }
SUBCMD="$1"; shift || true
case "$SUBCMD" in
  slice)        cmd_slice "$@" ;;
  dicom)        cmd_dicom "$@" ;;
  split)        cmd_split "$@" ;;
  update-yaml)  cmd_update_yaml "$@" ;;
  validate)     cmd_validate "$@" ;;
  download)     cmd_download "$@" ;;
  pipeline)     cmd_pipeline "$@" ;;
  help|-h|--help) print_help ;;
  *) die "Unknown subcommand: $SUBCMD (try './prepare_dataset.sh help')" 1 ;;
esac
