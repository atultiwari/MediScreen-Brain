# MediScreen-Brain v1.0.3 🧠

> **Zero-Configuration Brain Tumor Detection System** - Double-click to use, no dependencies installation required

---

## ✨ Key Features

- 🎯 **High-Precision Detection**: Based on YOLO deep learning model for accurate brain tumor region identification
- ⚡ **Zero-Configuration Launch**: Download EXE file and double-click to run, no Python installation or environment setup needed
- 📊 **Multi-Format Support**: Supports NIfTI (.nii/.nii.gz), DICOM, JPG/PNG and other medical imaging formats
- 🔄 **Batch Processing**: Folder-level batch detection with automatic Excel/TXT report generation
- 📹 **Real-Time Monitoring**: Multi-camera synchronous monitoring with automatic abnormal segment recording
- 📝 **Report Export**: One-click PDF diagnostic report and Excel statistical data export

---

## 🚀 Quick Start

### Windows Users (Recommended)

1. **Download**: Click the `MediScreen-Brain_1.0.3.exe` link above to download
2. **Run**: Double-click the EXE file to launch
3. **Use**: Select model → Load image → Start detection

**No Python installation, no environment variable configuration, no dependency library downloads required!**

### Other Platforms

To run from source on macOS/Linux:

```bash
git clone https://github.com/JingW-ui/MediScreen-Brain.git
cd MediScreen-Brain
pip install -r requirements.txt
python Brain_Tumor_detection_ui.py
```

---

## 📸 Feature Preview

### Real-Time Detection
- Single image/video/camera real-time analysis
- Three-plane (axial/sagittal/coronal) NIfTI medical image analysis
- Automatic tumor diameter, area, and volume measurement

### Batch Processing
- Automatic scanning of all images within folders
- Generate annotated result images
- Export Excel statistical reports (including detection count, confidence, dimensions, etc.)

### Real-Time Monitoring
- Support up to 4 camera streams simultaneously
- Automatic MP4 video recording when targets are detected
- Historical snapshot playback and management

---

## 📋 System Requirements

| Item | Minimum Configuration | Recommended Configuration |
|------|----------------------|--------------------------|
| **Operating System** | Windows 10 (64-bit) | Windows 11 |
| **CPU** | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| **Memory** | 8 GB | 16 GB |
| **Graphics Card** | Integrated graphics | NVIDIA GTX 1060+ (optional GPU acceleration) |
| **Storage Space** | 2 GB available space | 10 GB SSD |

---

## 📦 Version Information

### v1.0.3 Updates

- ✅ Packaged as standalone EXE file for true zero-configuration deployment
- ✅ Optimized model loading speed for faster startup
- ✅ Fixed NIfTI file parsing compatibility issues
- ✅ Improved Chinese character display in PDF reports
- ✅ Enhanced batch processing stability

---

## ⚠️ Important Disclaimer

> **This software is a prototype version intended only for medical research and educational demonstration purposes, not suitable for clinical diagnosis or disease classification in medical applications.**

- All detection results are for reference only and must be confirmed by professional physicians
- Data is processed locally and will not be uploaded to the cloud
- Please comply with HIPAA/GDPR and other data protection regulations when using

---

## 💬 Technical Support

Encountering issues?

1. Check the runtime logs at the bottom of the interface
2. Review console error output
3. Consult the "FAQ" section in the complete documentation

---

## 👥 Development Team

- **Research Team**: [Neuroengineering Laboratory, University of Electronic Science and Technology of China](https://www.neuro.uestc.edu.cn/)
- **Developer**: [JingW-ui](https://jingw-ui.github.io/resume/)

---

## 📄 License

This project is provided for academic research and educational use only. For commercial use, please contact the author for authorization.

---

**Release Date**: 2026-05-15  
**File Size**: Approximately 342 MB (including model files)  
**Compatible Systems**: Windows 10/11 (64-bit)