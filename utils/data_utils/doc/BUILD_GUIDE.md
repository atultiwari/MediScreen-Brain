
### 方式二：手动执行命令

#### 📦 单文件夹模式完整命令

```powershell
pyinstaller --name="DCM2NII_Converter" --onedir --windowed --icon="icon.ico" --add-data "icon.ico;." --add-data "dcm2niix.exe;." --hidden-import=nibabel --hidden-import=numpy --hidden-import=PySide6.QtCore --hidden-import=PySide6.QtGui --hidden-import=PySide6.QtWidgets --clean --noconfirm "dcm2nii_ui.py"
```

#### 🎯 单文件模式完整命令

```powershell
   pyinstaller --name="DCM2NII_Converter1.0.1" --onefile --windowed --icon="icon.ico" --add-data "icon.ico;." --add-data "dcm2niix.exe;." --hidden-import=nibabel --hidden-import=numpy --hidden-import=PySide6.QtCore --hidden-import=PySide6.QtGui --hidden-import=PySide6.QtWidgets --runtime-tmpdir="%TEMP%" --clean --noconfirm "utils\data_utils\dcm2nii_ui.py"
```

---

## 📊 两种打包模式对比

| 特性 | 单文件夹模式 (--onedir) | 单文件模式 (--onefile) |
|------|------------------------|------------------------|
| **输出形式** | 文件夹 + exe | 单个 exe |
| **启动速度** | ⚡ 快（直接运行） | 🐢 慢（需解压到临时目录） |
| **文件大小** | 📦 250-350 MB | 📦 200-300 MB |
| **分发便利性** | ⭐⭐⭐ 一般 | ⭐⭐⭐⭐⭐ 优秀 |
| **更新维护** | ✅ 可替换部分文件 | ❌ 需重新打包 |
| **反病毒扫描** | ✅ 较少误报 | ⚠️ 可能误报 |
| **适用场景** | 内部使用、频繁更新 | 对外分发、最终版本 |

---