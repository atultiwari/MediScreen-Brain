# MediScreen-Brain — Architecture & Application Analysis

> Date analysed: 2026-05-16
> Repository root: `/Users/atultiwari/Downloads/Projects/YOLO-GUI/MediScreen-Brain`
> Upstream: https://github.com/JingW-ui/MediScreen-Brain
> License: MIT

---

## 1. Executive Summary

**MediScreen-Brain** is a **single-process, monolithic Python desktop GUI application** for brain-tumour detection on MRI imagery. It wraps **Ultralytics YOLO** (PyTorch + ONNX Runtime) inside a **PySide6 (Qt 6)** UI and adds medical-imaging utilities (NIfTI tri-planar slicing, DICOM/NIfTI conversion, structured PDF/Excel reporting, multi-camera capture). It is explicitly **CPU-optimized**, ships pre-trained `.pt` and `.onnx` model weights inside the repo, and is distributed to Windows users as a frozen **PyInstaller** EXE (no Python install required).

It is **not** a web/server app, has no database, no auth, no network surface beyond optional `huggingface_hub` model fetches and outbound PDF/Excel writes.

| Aspect | Value |
|---|---|
| Type | Desktop GUI (single-window, multi-tab) |
| Language | Python 3.8+ (`3.10`/`3.11` recommended by deps) |
| UI Framework | PySide6 6.6.3 (Qt 6) — PyQt5 also pulled in but unused at runtime |
| ML Backend | Ultralytics YOLOv8 (`8.3.168`) + ONNX Runtime |
| Tensor Backend | PyTorch 2.4.1 + cu121 (CPU also works) |
| Medical I/O | NiBabel (NIfTI), DICOM via `pydicom`-like flow (`dcm2niix.exe`) |
| Reporting | ReportLab (PDF), OpenPyXL (XLSX) |
| Packager | PyInstaller 6.14 → single-file `.exe` (Windows) |
| Distribution | GitHub Release artefact `MediScreen-Brain.exe` |
| Entry point | `Brain_Tumor_detection_ui.py` (6,453 lines) |
| Secondary tool | `batch_compare_ui.py` — standalone batch-compare UI |

---

## 2. Source Layout

```text
MediScreen-Brain/
├── Brain_Tumor_detection_ui.py     # Main app, ALL UI + business logic (6,453 LOC)
├── batch_compare_ui.py             # Secondary CLI/UI for batch model comparison (759 LOC)
├── requirements.txt                # Frozen deps (Windows/conda pinned)
├── icon.ico                        # Window/app icon
├── index.html                      # GitHub Pages landing site
├── README.md / LICENSE
├── assets/
│   ├── img/                        # logo.ico, logo.png, screenshots
│   └── video/                      # Demo MP4s (H.264, converted)
├── pt_models/
│   ├── Brain_Tumor.pt              # 6.0 MB YOLOv8 detection weights
│   └── best.pt                     # 5.2 MB experimental weights
├── onnx_models/
│   ├── Brain_Tumor.onnx            # 11.7 MB ONNX export
│   └── segmentation.onnx           # 12.7 MB segmentation export
├── data_test/
│   └── MRBrainTumor2.nii.gz        # Sample 3D MRI (~5.8 MB)
├── utils/
│   ├── TumorSliceFinder.py         # 3D NIfTI → best 2D slice locator
│   ├── YOLOONNX.py                 # Lightweight ONNX inference wrapper
│   ├── Compares_two_methods.py     # Cross-method benchmarking
│   ├── convert_pt_to_onnx.py       # Model exporter
│   ├── nii_slice_to_images.py      # NIfTI → PNG sequence
│   ├── find_slice_plus.py          # Enhanced slice search
│   ├── analyze_*.py / batch_*.py   # Dataset analytics & label tools
│   ├── data.yaml                   # YOLO training config
│   ├── data_utils/, results_utils/, utils_for_data/
├── doc/
│   ├── PROJECT_DOCUMENTATION_EN.md (full feature ref)
│   ├── PROJECT_DOCUMENTATION_ZN.md (中文)
│   ├── RELEASE_NOTES_v1.0.3*.md
│   └── 打包命令.md                  # PyInstaller commands
└── .github/workflows/             # Only deploys GitHub Pages (no build CI)
```

### Notable shape

- The **entire application logic lives in one 6,453-line file** (`Brain_Tumor_detection_ui.py`). It violates the project-rule `<800 lines` budget by ~8×. This is *the* primary refactor target if the codebase is ever maintained beyond freeze-and-ship.
- `utils/` is a mix of **runtime helpers** (`TumorSliceFinder`, `YOLOONNX`) and **dev/training scripts** (`train_all_models.py`, `analyze_*`). Only `TumorSliceFinder` is imported at runtime by the main UI.
- `get_runtime_dir` is duplicated **three times** in the main module — a clear copy-paste smell.

---

## 3. Architecture

### 3.1 High-level diagram

```text
                    ┌────────────────────────────────────────────────┐
                    │  EnhancedDetectionUI (QMainWindow)             │
                    │  - Tabbed UI (6 tabs)                          │
                    │  - Owns managers + worker threads              │
                    └──────────────────────┬─────────────────────────┘
                                           │
        ┌──────────────────────────────────┼──────────────────────────────────┐
        │                                  │                                  │
        ▼                                  ▼                                  ▼
  Managers (sync)                  Workers (QThread)                  Recorders / Snapshots
  ─────────────────                ──────────────────                 ───────────────────────
  • ModelManager        ▸ scan        • DetectionThread             • DetectionVideoRecorder
    /load .pt/.onnx       /load           (single image / video)      (MP4 + JSON metadata)
  • CameraManager       ▸ enumerate   • BatchDetectionThread        • CameraVideoRecorder
  • StyleManager          /probe         (folder sweep + XLSX)        • SnapshotWidget
  • TumorSliceFinder    ▸ axial,      • MultiCameraMonitorThread      (history list / playback)
    (utils/…)             sagittal,      (≤ 4 cams, auto-reconnect)
                          coronal      • CameraThread (per-cam)
                                       • Per-thread Qt Signals
                                          → UI slots on main thread
        │                                  │                                  │
        └──────────────────────────────────┴──────────────────────────────────┘
                                           │
                                           ▼
                                   Filesystem outputs
                                   ──────────────────
                                   detection_history/   monitor_history/
                                   *.mp4 + *.json       *.xlsx  *.pdf  *.txt
```

### 3.2 Key classes (all defined in `Brain_Tumor_detection_ui.py`)

| Line | Class | Responsibility |
|---:|---|---|
| 72   | `StyleManager` | Centralised Qt stylesheet (gradients, hover, rounded controls) |
| 417  | `CameraManager` | Enumerate local cameras (OpenCV `VideoCapture`) |
| 557  | `ModelManager` | Scan & load `.pt` / `.onnx` weights via Ultralytics `YOLO()` |
| 651  | `DetectionThread(QThread)` | Async inference on a single image/video |
| 847  | `BatchDetectionThread(QThread)` | Async folder sweep → XLSX/TXT reports |
| 936  | `MultiCameraMonitorThread(QThread)` | Multiple cameras with auto-reconnect & pause |
| 1074 | `ModelSelectionDialog(QDialog)` | Advanced model picker (table) |
| 1168 | `DetectionResultWidget(QWidget)` | Side-by-side original vs annotated view |
| 1343 | `MonitoringWidget(QWidget)` | Live multi-cam grid + stats |
| 1959 | `CameraVideoRecorder` | Buffered MP4 capture |
| 2085 | `VideoWidget(QWidget)` | Playback of recorded snapshots |
| 2282 | `CameraThread(QThread)` | One thread per camera |
| 2367 | `EnhancedMonitoringWidget(QWidget)` | Outer monitoring tab |
| 2851 | `SliceDetailDialog(QDialog)` | Detailed NIfTI slice inspector |
| 2976 | `SnapshotWidget(QWidget)` | Snapshot list/playback/management |
| 3640 | `EnhancedDetectionUI(QMainWindow)` | **Root window** — composes everything |
| 6298 | `DetectionVideoRecorder` | Detection-triggered video recorder |
| 6429 | `main()` | App bootstrap, `QApplication`, show window |

### 3.3 Threading model

All long-running work runs in `QThread` subclasses that emit `Signal` payloads back to the main UI thread. There is no `asyncio`, no multiprocessing, no IPC. Inference (`ultralytics.YOLO.__call__`) is invoked from the worker thread directly; PyTorch handles its own thread pool.

### 3.4 Resource resolution (PyInstaller-aware)

Three near-identical helpers (lines 25, 36, 51, 62) implement the standard PyInstaller pattern:

```python
if getattr(sys, 'frozen', False):
    base_path = Path(sys._MEIPASS)        # bundled resources
else:
    base_path = Path(__file__).parent     # dev mode
```

Models, icons, and demo NIfTI are loaded through `get_resource_path()`; per-run output dirs (`detection_history/`, `monitor_history/`) go through `get_runtime_dir()` which **anchors at `Path(sys.executable).parent`** when frozen — so reports/snapshots land beside the EXE, not in `_MEIPASS`.

---

## 4. Runtime Data Flow

### 4.1 Single-image detection

```
User picks image
   └─► EnhancedDetectionUI.start_detection()
        └─► DetectionThread.run()
             ├─► ModelManager.current_model(image_path, conf=…)        ← Ultralytics call
             ├─► boxes → numpy (xyxy, conf, cls)
             └─► result_ready.emit(orig, annotated, ms, dets, classes)
                  └─► UI slot renders QPixmap on DetectionResultWidget
```

### 4.2 NIfTI tri-planar detection

```
.nii(.gz) → nibabel.load → np.ndarray (X,Y,Z)
        └─► TumorSliceFinder
             ├─ AXIAL_SAMPLING_STEP=2, SAGITTAL=2, CORONAL=2
             ├─ thresholded sweep → coarse best slice
             ├─ REFINE_RADIUS=1     → local refinement
             └─ returns best (axial, sagittal, coronal) indices + boxes
        └─► PDF/Excel exporter → diameter (mm) / area (mm²) / volume (mm³)
```

Voxel spacing is read from the NIfTI affine and used for the mm/mm²/mm³ conversions.

### 4.3 Batch folder sweep

`BatchDetectionThread` iterates supported extensions, runs inference per file, accumulates a list of records, and at the end materialises:

- `detection_report.txt` — human-readable summary
- `detection_report.xlsx` — two sheets (Results + Summary) using OpenPyXL with custom styles

### 4.4 Multi-camera live monitoring

- Up to 4 cameras via `MultiCameraMonitorThread` + per-cam `CameraThread`.
- Per-cam buffer; recording only triggers when ≥ 1 detection in frame.
- Output: `monitor_history/CameraN_<unix_ts>.mp4` + matching `.json` metadata.
- Auto-prune when directory exceeds **500 MB**.

---

## 5. Dependencies & Risk

`requirements.txt` is a **conda-frozen Windows lock file** — many entries are `file:///C:/...` paths, not pip-resolvable on macOS/Linux. The *functional* runtime deps are:

| Group | Key packages |
|---|---|
| GUI | `PySide6==6.6.3.1`, `pyqtgraph`, `PyQt5` (unused at runtime, removable) |
| Vision/ML | `ultralytics==8.3.168`, `torch==2.4.1+cu121`, `torchvision`, `onnxruntime`, `opencv-python==4.11.0.86` |
| Medical I/O | `nibabel==5.2.1`, `pynrrd`, `fslpy`, `pyvista`, `vtk` |
| Reporting | `reportlab==4.0.0`, `openpyxl==3.1.5`, `Pillow` |
| Misc heavy | `gradio`, `chattts`, `ChatGPT/whisper`, `realesrgan`, `gfpgan`, `Selenium`, `DrissionPage`, `huggingface_hub`, `tensorboard*` |

Roughly **80 % of `requirements.txt` is unused by the GUI** (chattts, whisper, realesrgan, gfpgan, selenium, DrissionPage, gradio, tensorboard, etc.). A clean minimal lock would be ~25 packages. This bloats the PyInstaller bundle and slows install considerably.

Secrets / network: none required. `huggingface_hub` is present but optional.

---

## 6. How to Run

### 6.1 Prerequisites (both platforms)

- Python **3.10** or **3.11** (3.8+ documented, but `numpy==1.23.5` and `torch==2.4.1` are most compatible with 3.10/3.11)
- Git
- ≥ 8 GB RAM, ~2 GB free disk
- (Recommended) a virtual env

### 6.2 macOS (Apple Silicon — arm64)

```bash
# 1. Clone
git clone https://github.com/JingW-ui/MediScreen-Brain.git
cd MediScreen-Brain

# 2. Python venv
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 3. Install a SLIM dep set (requirements.txt is Windows-locked, do not use as-is)
pip install \
  "PySide6>=6.6,<6.8" \
  "ultralytics==8.3.168" \
  "torch>=2.2,<2.6"  "torchvision" \
  "onnxruntime" \
  "opencv-python==4.11.0.86" \
  "nibabel==5.2.1" \
  "numpy<2" "scipy" "pandas" "matplotlib" "Pillow" \
  "reportlab==4.0.0" "openpyxl==3.1.5" \
  "pyqtgraph" "vtk" "pyvista"

# 4. Run
python Brain_Tumor_detection_ui.py
```

Notes for macOS:
- On Apple Silicon, `pip install torch` resolves to the universal2/arm64 wheel (CPU + MPS). The CUDA pin in `requirements.txt` (`torch==2.4.1+cu121`) does **not** install on macOS — drop the `+cu121` suffix.
- Camera access requires granting Terminal/iTerm **Camera** permission in *System Settings → Privacy & Security*.
- NIfTI / Excel / PDF features have no platform-specific gotchas.

### 6.3 Windows (x64) — from source

```powershell
# 1. Clone
git clone https://github.com/JingW-ui/MediScreen-Brain.git
cd MediScreen-Brain

# 2. Python venv
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 3. Install (the included requirements.txt is Windows-oriented but conda-locked.
#    Use the same slim set as macOS to avoid file:// paths)
pip install `
  "PySide6>=6.6,<6.8" `
  "ultralytics==8.3.168" `
  "torch>=2.2,<2.6" "torchvision" `
  "onnxruntime" `
  "opencv-python==4.11.0.86" `
  "nibabel==5.2.1" `
  "numpy<2" "scipy" "pandas" "matplotlib" "Pillow" `
  "reportlab==4.0.0" "openpyxl==3.1.5" `
  "pyqtgraph" "vtk" "pyvista"

# 4. Run
python Brain_Tumor_detection_ui.py
```

### 6.4 Windows — zero-config (end-user)

```text
Download MediScreen-Brain_1.0.3.exe from:
  https://github.com/JingW-ui/MediScreen-Brain/releases
Double-click → app starts. No Python required.
```

The EXE bundles weights and resources via `_MEIPASS`; reports and snapshots are written next to the EXE.

### 6.5 Secondary tool — batch model comparison

```bash
python batch_compare_ui.py
```

---

## 7. Standalone Desktop App — Is It Feasible?

### 7.1 Windows x64 — **Already shipped**

Yes. v1.0.3 of the upstream project ships exactly this: a single `MediScreen-Brain_1.0.3.exe`. The build recipe (from `doc/打包命令.md`):

```powershell
python -m PyInstaller `
  --onefile --windowed `
  --name="MediScreen-Brain_1.0.3" `
  --icon="assets/img/logo.ico" `
  --add-data "assets/img;assets/img" `
  --add-data "pt_models;pt_models" `
  --add-data "onnx_models;onnx_models" `
  --hidden-import=torch `
  --clean --noconfirm `
  Brain_Tumor_detection_ui.py
```

For development/debug a `--onedir` build is recommended (faster cold start, easier to inspect).

Expected EXE size: ~700 MB – 1.2 GB (Torch + PySide6 + bundled weights). Cold-start ~15–25 s on first launch from `_MEIPASS` extraction; subsequent ~5–8 s.

### 7.2 macOS arm64 (Apple Silicon) — **Feasible, not yet provided**

There is **no macOS build today**, but nothing in the code blocks one. All key dependencies have native arm64 wheels:

| Dep | arm64 wheel | Notes |
|---|---|---|
| PySide6 6.6+ | ✅ | Native universal2 |
| PyTorch 2.2+ | ✅ | CPU + MPS, native arm64 |
| Ultralytics | ✅ | Pure Python |
| onnxruntime | ✅ | `onnxruntime` 1.17+ has arm64 macOS wheels (also `onnxruntime-coreml` for ANE) |
| opencv-python | ✅ | 4.10+ has arm64 wheels |
| nibabel / numpy / scipy | ✅ | |
| vtk / pyvista | ✅ | 9.3+ on Python 3.11 |
| reportlab / openpyxl | ✅ | Pure-Python or universal |

#### Recommended packaging recipe (macOS arm64)

```bash
# Build environment
arch -arm64 python3.11 -m venv .venv-arm64
source .venv-arm64/bin/activate
pip install --upgrade pip
pip install <slim deps from §6.2>
pip install pyinstaller==6.14

# One-file .app build
pyinstaller \
  --onefile --windowed \
  --name "MediScreen-Brain" \
  --icon "assets/img/logo.icns"  \
  --add-data "assets/img:assets/img" \
  --add-data "pt_models:pt_models" \
  --add-data "onnx_models:onnx_models" \
  --hidden-import=torch \
  --collect-submodules=ultralytics \
  --osx-bundle-identifier "com.uestc.neuro.mediscreen-brain" \
  --target-arch arm64 \
  --clean --noconfirm \
  Brain_Tumor_detection_ui.py

# (Optional) sign + notarize for Gatekeeper-friendly distribution
codesign --deep --force --options runtime \
  --sign "Developer ID Application: <Your Name> (<TEAMID>)" \
  dist/MediScreen-Brain.app
xcrun notarytool submit dist/MediScreen-Brain.app.zip \
  --apple-id <appleid> --team-id <TEAMID> --password <app-specific-pw> --wait
xcrun stapler staple dist/MediScreen-Brain.app
```

#### Pre-flight changes required for a clean macOS build

1. **Drop the `+cu121` Torch pin.** macOS has no CUDA. Use a plain `torch>=2.2,<2.6` requirement.
2. **Resource-path separator.** PyInstaller uses `;` on Windows and `:` on macOS/Linux for `--add-data`. The shipped `打包命令.md` uses `;` — convert to `:` for the macOS build.
3. **Convert the icon.** Create `assets/img/logo.icns` from `assets/img/logo.ico`/`logo.png` (`iconutil -c icns logo.iconset/`).
4. **`dcm2niix`.** The Windows recipe bundles `dcm2niix.exe`. For macOS, install via `brew install dcm2niix` and either bundle the arm64 binary at `--add-binary` or detect a system install.
5. **Camera entitlement.** A bundled `Info.plist` needs `NSCameraUsageDescription`; PyInstaller's `--windowed` produces an `.app` skeleton, edit `Contents/Info.plist` after build (or pass via `--info-plist` patches).
6. **Optionally** swap ONNX Runtime for `onnxruntime-coreml` to expose Apple Neural Engine acceleration on M-series silicon — gains ~2–4× over CPU for the bundled ONNX weights.

#### Alternative packagers worth considering

| Tool | When to pick | Trade-offs |
|---|---|---|
| **PyInstaller** | Parity with current Windows build | Familiar; large bundle |
| **Briefcase (BeeWare)** | Want a real `.app` with proper code-signing flow | Excellent macOS support; rewriting bundle spec |
| **py2app** | Pure-macOS focus | Maintained; fewer hidden-import surprises than older versions |
| **Nuitka** | Want smaller + faster startup | Longer build; compiles Python to C |
| **Conda-pack + launcher** | Reproducibility for research labs | Heavier on disk; trivial to update |

### 7.3 Verdict

- **Windows x64 standalone:** already exists as `MediScreen-Brain_1.0.3.exe`; recipe is in-repo and reproducible.
- **macOS arm64 standalone:** technically feasible with ~1–2 days of work — primarily (a) sanitising `requirements.txt`, (b) producing an `.icns` icon, (c) swapping `;` → `:` in `--add-data`, (d) optional codesign/notarize. No architectural blockers.

A reasonable next step would be a CI matrix (`windows-latest`, `macos-14`/arm64) in `.github/workflows/` that runs PyInstaller per OS and uploads the artefacts to the GitHub Release — today the only workflow there deploys the marketing page to GitHub Pages.

---

## 8. Observations & Caveats

1. **Monolith risk.** `Brain_Tumor_detection_ui.py` at 6.4 k LOC concentrates UI, threading, business logic, recorders, and PDF building. Even modest refactors (split into `app/{ui,workers,recorders,reporting}`) would dramatically lower maintenance cost.
2. **Duplicate code.** `get_runtime_dir` is defined three times back-to-back (lines 36/51/62). Harmless but symptomatic.
3. **`requirements.txt` is not portable.** Conda lock paths and irrelevant heavy deps (`chattts`, `realesrgan`, `gfpgan`, `selenium`, `gradio`, `tensorboard*`) inflate install size and break non-Windows installs. A `pyproject.toml` with a minimal core + extras is the obvious upgrade.
4. **No tests.** No `tests/`, no pytest config. Refactoring is therefore high-risk; before any restructure, pin behaviour with snapshot/integration tests over `data_test/MRBrainTumor2.nii.gz`.
5. **Type hints sparse.** Only `TumorSliceFinder` uses typing meaningfully. PEP 484 coverage is otherwise low.
6. **PHI / clinical-safety.** README and PDF disclaimer flag this as a **prototype, not for clinical diagnosis**. Treat any deployment with that disclaimer in mind.
7. **CPU optimisation claim.** The codebase loads `torch==2.4.1+cu121` but ONNX export + `task='detect'` is the actual CPU path used by most users — that is what makes startup feasible on commodity hardware.

---

## 9. Quick Command Reference

| Goal | macOS arm64 | Windows x64 |
|---|---|---|
| Run from source | `python Brain_Tumor_detection_ui.py` | `python Brain_Tumor_detection_ui.py` |
| Run batch tool | `python batch_compare_ui.py` | `python batch_compare_ui.py` |
| Export PT→ONNX | `python utils/convert_pt_to_onnx.py` | same |
| NIfTI→PNG slices | `python utils/nii_slice_to_images.py` | same |
| Build standalone | `pyinstaller --onefile --windowed --target-arch arm64 ... Brain_Tumor_detection_ui.py` (see §7.2) | `python -m PyInstaller --onefile --windowed ... Brain_Tumor_detection_ui.py` (see §7.1) |

---

## 10. TL;DR

- **What it is:** a PySide6 desktop app that runs YOLOv8 on brain-MRI imagery (single image, batch, NIfTI tri-planar, multi-camera) and exports PDF/Excel reports.
- **How to run it now:** `pip install` the slim dependency set in §6, then `python Brain_Tumor_detection_ui.py`.
- **Standalone Windows x64:** ✅ already provided as an EXE in the upstream GitHub Releases; reproducible via the included PyInstaller `--onefile` recipe.
- **Standalone macOS arm64:** ✅ feasible with the modifications listed in §7.2 — no architectural blockers, ~1–2 days of packaging work, optionally with Apple Notarisation.
