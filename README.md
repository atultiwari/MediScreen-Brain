# [MediScreen-Brain](https://jingw-ui.github.io/MediScreen-Brain/)   
## A YOLO-Based, CPU-Optimized GUI Platform for Brain Tumor Detection and Reporting

MediScreen-Brain is a lightweight brain tumor detection system designed for both clinical and research applications. It integrates YOLO object detection models with CPU-specific inference optimization, providing a user-friendly graphical interface (GUI) for non-experts to perform image analysis and generate reports efficiently.
![Screenshot 1](assets/img/1.png)
## software download
[MediScreen-Brain.exe](https://github.com/JingW-ui/MediScreen-Brain/releases/tag/MediScreen-Brain)

## source code
```commandline
git clone https://github.com/JingW-ui/MediScreen-Brain.git
```

---

## Quick start (macOS / Linux) — single command

The repo ships with `run.sh`, a one-shot bootstrap that creates a local
virtualenv, installs the slim macOS/Linux dependency set
(`requirements-macos.txt`), and launches the GUI.

```bash
cd MediScreen-Brain
./run.sh
```

That's it. First run takes a few minutes (downloading ~2 GB of wheels:
PySide6, PyTorch, VTK, etc.). Subsequent runs skip install and launch in
seconds.

### Requirements

- Python 3.10 or 3.11 (`brew install python@3.11` on macOS)
- ~3 GB free disk for the virtualenv + bundled model weights
- macOS arm64 (Apple Silicon) tested; macOS x86_64 and Linux should also work

### `run.sh` flags

| Flag | Effect |
|---|---|
| (none) | Setup if needed + launch the main GUI |
| `--setup` | Create venv and install deps; **do not** launch the app |
| `--run` | Skip setup; launch from the existing venv |
| `--reinstall` | Force-reinstall dependencies into the existing venv |
| `--clean` | Remove the venv and `__pycache__` |
| `--batch` | Launch `batch_compare_ui.py` instead of the main GUI |
| `--python <bin>` | Use a specific interpreter (default auto-picks `python3.11` → `3.10` → `python3`) |
| `--venv <dir>` | Override venv directory (default `.venv`) |
| `--requirements <f>` | Use a different requirements file |
| `--no-cache` | Pass `--no-cache-dir` to pip |
| `--upgrade-pip` | Upgrade pip/setuptools/wheel before installing |
| `-h`, `--help` | Show inline help |

### Common recipes

```bash
# Fresh install + run
./run.sh

# Just rebuild the env from scratch
./run.sh --clean --setup

# Launch the batch comparison tool
./run.sh --batch

# Use a specific Python
./run.sh --python /opt/homebrew/bin/python3.11

# After pulling new commits, re-install pinned deps
./run.sh --reinstall --run
```

### macOS notes

- First launch will request **Camera** permission if you open the Monitoring
  tab. Grant it in *System Settings → Privacy & Security → Camera*.
- The PDF report module auto-detects a CJK font (STHeiti / Songti / Arial
  Unicode). If none are present it falls back to Helvetica.
- macOS uses the `CAP_AVFOUNDATION` capture backend automatically.

### Windows users

The simplest path on Windows is still the prebuilt
`MediScreen-Brain_1.0.3.exe` from the GitHub release page — no Python
install required. To run from source on Windows, see the slim install
recipe in [`ARCHITECTURE_ANALYSIS.md`](ARCHITECTURE_ANALYSIS.md#62-windows-x64--from-source).

---

## Citation
If you use this software in your research, please cite our paper:
> Jing W, et al. (2026). *MediScreen-Brain: A Portable, YOLO-powered GUI for Multi-Modal Brain Tumor Detection, 3D Localization, and Structured Reporting​*. Computer Methods and Programs in Biomedicine. DOI: not published yet

---

## Contact
For questions or collaboration opportunities, please contact:  
📧 202421140108@std.uestc.edu.cn
## my Bio
[JingWang](https://jingw-ui.github.io/)

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
