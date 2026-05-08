# PyInstaller 打包命令 - 速查卡

## 🎯 一句话总结
使用 PyInstaller 将 Python 应用打包成独立的 Windows exe 文件。

---

## 📦 完整打包命令（复制即用）

### 单文件模式（推荐用于分发）

```powershell
pyinstaller --name="DCM2NII_Converter" --onefile --windowed --icon="icon.ico" --add-data "icon.ico;." --add-data "dcm2niix.exe;." --hidden-import=nibabel --hidden-import=numpy --hidden-import=PySide6 --collect-all=PySide6 --clean --noconfirm "utils\data_utils\dcm2nii_ui.py"
```

### 单文件夹模式（推荐用于开发）

```powershell
pyinstaller --name="DCM2NII_Converter" --onedir --windowed --icon="icon.ico" --add-data "icon.ico;." --add-data "dcm2niix.exe;." --hidden-import=nibabel --hidden-import=numpy --hidden-import=PySide6 --collect-all=PySide6 --clean --noconfirm "utils\data_utils\dcm2nii_ui.py"
```

---

## 🔧 核心参数说明

| 参数 | 作用 | 示例 |
|------|------|------|
| `--name` | 程序名称 | `--name="MyApp"` |
| `--onefile` | 单文件模式 | （生成单个 exe） |
| `--onedir` | 单文件夹模式 | （默认） |
| `--windowed` | 无控制台窗口 | （GUI 应用必备） |
| `--icon` | 程序图标 | `--icon="app.ico"` |
| `--add-data` | 添加资源文件 | `--add-data "file;dest"` |
| `--hidden-import` | 隐藏导入模块 | `--hidden-import=numpy` |
| `--collect-all` | 收集包的所有子模块 | `--collect-all=PySide6` |
| `--clean` | 清理缓存 | （建议每次使用） |
| `--noconfirm` | 不询问确认 | （自动化脚本用） |

---

## 📁 资源文件添加格式

### Windows（分号分隔）
```bash
--add-data "源文件;目标位置"
```

**示例：**
```bash
--add-data "icon.ico;."              # 放到根目录
--add-data "config.json;config"      # 放到 config 子目录
--add-data "data/file.txt;data"      # 放到 data 子目录
```

### Linux/Mac（冒号分隔）
```bash
--add-data "源文件:目标位置"
```

---

## 🚀 快速开始

### 方法一：使用批处理脚本

```bash
# 单文件模式
build_dcm2nii_onefile.bat

# 单文件夹模式
build_dcm2nii.bat
```

### 方法二：使用 PowerShell 脚本

```powershell
# 单文件模式（默认）
.\build_dcm2nii.ps1

# 单文件夹模式
.\build_dcm2nii.ps1 -Mode "onedir"
```

### 方法三：手动执行

```powershell
# 进入项目目录
cd H:\pycharm_project\MediScreen-Brain

# 执行打包命令（复制上面的完整命令）
pyinstaller --name="DCM2NII_Converter" --onefile ...
```

---

## 💡 代码适配

### 资源路径访问

在代码中使用 `get_resource_path()` 函数：

```python
import sys
import os

def get_resource_path(relative_path: str) -> str:
    """获取资源文件的绝对路径（兼容打包环境）"""
    if getattr(sys, 'frozen', False):
        # 打包后的环境
        base_path = sys._MEIPASS
    else:
        # 开发环境
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 使用示例
icon_path = get_resource_path("icon.ico")
exe_path = get_resource_path("dcm2niix.exe")
```

### ❌ 错误写法
```python
# 这样在打包后会找不到文件
icon_path = "icon.ico"
```

---

## 📊 两种模式对比

| 特性 | 单文件 (--onefile) | 单文件夹 (--onedir) |
|------|-------------------|-------------------|
| 输出形式 | 单个 exe | 文件夹 + exe |
| 启动速度 | 🐢 慢（需解压） | ⚡ 快 |
| 文件大小 | 200-300 MB | 250-350 MB |
| 分发便利性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 适用场景 | 对外分发 | 内部测试 |

---

## ❓ 常见问题

### Q1: 找不到资源文件
**解决：** 使用 `get_resource_path()` 函数

### Q2: 打包体积过大
**解决：** 
- 排除不必要的模块：`--exclude-module=tkinter`
- 使用 UPX 压缩：`--upx-dir=path`

### Q3: 杀毒软件误报
**解决：** 改用单文件夹模式，或提交白名单

### Q4: dcm2niix.exe 无法运行
**解决：** 检查路径是否正确
```python
exe_path = get_resource_path("dcm2niix.exe")
print(f"存在: {os.path.exists(exe_path)}")
```

---

## 📝 检查清单

打包前：
- [ ] 安装 PyInstaller：`pip install pyinstaller`
- [ ] 准备 icon.ico 文件
- [ ] 准备 dcm2niix.exe 文件
- [ ] 代码使用 `get_resource_path()`

打包后：
- [ ] 测试 exe 能正常启动
- [ ] 图标显示正确
- [ ] dcm2niix.exe 能被调用
- [ ] 完整流程测试通过

---

## 🔗 相关文件

- 批处理脚本：`build_dcm2nii.bat`, `build_dcm2nii_onefile.bat`
- PowerShell 脚本：`build_dcm2nii.ps1`
- 详细文档：`BUILD_GUIDE.md`

---

**更新日期：** 2026-05-06  
**版本：** v1.0
