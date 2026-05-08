"""
NIfTI 文件信息查看器
功能：通过 PySide6 GUI 读取并展示指定 .nii / .nii.gz 文件的维度与分辨率信息
依赖：PySide6, nibabel, numpy
"""

import sys
import os
import numpy as np

import nibabel as nib
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QFileDialog, QTableWidget,
    QTableWidgetItem, QGroupBox, QTextEdit, QSplitter, QHeaderView,
    QMessageBox, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor, QPalette, QIcon


# ─────────────────────────── 后台加载线程 ───────────────────────────
class NiiLoadThread(QThread):
    """在后台线程中加载 NIfTI 文件，避免 UI 卡顿"""
    finished = Signal(dict)   # 成功时发射包含信息的字典
    error    = Signal(str)    # 失败时发射错误信息

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            img = nib.load(self.filepath)
            hdr = img.header

            # ── 基础维度信息 ──
            shape   = img.shape                      # (I, J, K[, T...])
            ndim    = len(shape)
            zooms   = hdr.get_zooms()                # 各维度分辨率(mm 或 s)
            affine  = img.affine                     # 4x4 仿射矩阵
            data_type = hdr.get_data_dtype()

            # ── 扩展信息 ──
            try:
                xyzt_units = hdr.get_xyzt_units()    # ('mm', 's') 等
            except Exception:
                xyzt_units = ('unknown', 'unknown')

            try:
                dim_info = hdr.get_dim_info()        # (freq, phase, slice)
            except Exception:
                dim_info = ('unknown', 'unknown', 'unknown')

            try:
                qform_code = int(hdr.get('qform_code', 0))
                sform_code = int(hdr.get('sform_code', 0))
            except Exception:
                qform_code = sform_code = 0

            # ── 数据统计（可选，不读全量数据，仅读 header proxy）──
            try:
                data_proxy = img.dataobj
                data_min = float(np.min(data_proxy))
                data_max = float(np.max(data_proxy))
            except Exception:
                data_min = data_max = float('nan')

            result = {
                'filepath'    : self.filepath,
                'filename'    : os.path.basename(self.filepath),
                'shape'       : shape,
                'ndim'        : ndim,
                'zooms'       : zooms,
                'affine'      : affine,
                'data_type'   : str(data_type),
                'xyzt_units'  : xyzt_units,
                'dim_info'    : dim_info,
                'qform_code'  : qform_code,
                'sform_code'  : sform_code,
                'data_min'    : data_min,
                'data_max'    : data_max,
                'file_size_mb': os.path.getsize(self.filepath) / 1024 / 1024,
            }
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ─────────────────────────── 主窗口 ───────────────────────────
class NiiInfoViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NIfTI 文件信息查看器 — MediScreen-Brain")
        self.setMinimumSize(900, 680)
        self.setWindowIcon(QIcon(str("look_icon.ico")))
        self._load_thread = None
        self._build_ui()
        self._apply_style()

    # ── UI 构建 ──────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(10)
        root_layout.setContentsMargins(14, 14, 14, 14)

        # 文件选择区
        file_group = QGroupBox("文件选择")
        file_layout = QHBoxLayout(file_group)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("请选择或拖入 .nii / .nii.gz 文件路径 …")
        self.path_edit.setReadOnly(False)
        self.path_edit.returnPressed.connect(self._on_load_clicked)
        browse_btn = QPushButton("浏览…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._on_browse)
        self.load_btn = QPushButton("读取信息")
        self.load_btn.setFixedWidth(90)
        self.load_btn.clicked.connect(self._on_load_clicked)
        file_layout.addWidget(self.path_edit)
        file_layout.addWidget(browse_btn)
        file_layout.addWidget(self.load_btn)
        root_layout.addWidget(file_group)

        # 状态栏
        self.status_label = QLabel("就绪 — 请选择 NIfTI 文件")
        self.status_label.setStyleSheet("color: #888; font-size: 12px;")
        root_layout.addWidget(self.status_label)

        # 主内容区（上下分割）
        splitter = QSplitter(Qt.Vertical)
        root_layout.addWidget(splitter, stretch=1)

        # ── 上半：维度 & 分辨率表格 ──
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)

        # 维度信息表
        dim_group = QGroupBox("维度信息 (Shape)")
        dim_vbox = QVBoxLayout(dim_group)
        self.dim_table = self._make_table(["轴", "维度名称", "体素数量", "分辨率 (mm)", "物理范围 (mm)"])
        dim_vbox.addWidget(self.dim_table)
        top_layout.addWidget(dim_group, stretch=3)

        # 基础属性表
        attr_group = QGroupBox("文件属性")
        attr_vbox = QVBoxLayout(attr_group)
        self.attr_table = self._make_table(["属性", "值"])
        self.attr_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.attr_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        attr_vbox.addWidget(self.attr_table)
        top_layout.addWidget(attr_group, stretch=2)

        splitter.addWidget(top_widget)

        # ── 下半：仿射矩阵 & 原始 Header ──
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)

        affine_group = QGroupBox("仿射矩阵 (Affine 4×4)")
        affine_vbox = QVBoxLayout(affine_group)
        self.affine_table = QTableWidget(4, 4)
        self.affine_table.setHorizontalHeaderLabels(["X", "Y", "Z", "T"])
        self.affine_table.setVerticalHeaderLabels(["行0", "行1", "行2", "行3"])
        self.affine_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.affine_table.setEditTriggers(QTableWidget.NoEditTriggers)
        affine_vbox.addWidget(self.affine_table)
        bottom_layout.addWidget(affine_group, stretch=2)

        header_group = QGroupBox("Header 原始信息")
        header_vbox = QVBoxLayout(header_group)
        self.header_text = QTextEdit()
        self.header_text.setReadOnly(True)
        self.header_text.setFont(QFont("Consolas", 9))
        header_vbox.addWidget(self.header_text)
        bottom_layout.addWidget(header_group, stretch=3)

        splitter.addWidget(bottom_widget)
        splitter.setSizes([320, 300])

    def _make_table(self, headers: list) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.setAlternatingRowColors(True)
        t.setSelectionBehavior(QTableWidget.SelectRows)
        return t

    # ── 样式 ──────────────────────────────────────────────
    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #f5f6fa; }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #d0d4e0;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 6px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #3a5adb;
            }
            QPushButton {
                background: #3a5adb;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 12px;
                font-size: 13px;
            }
            QPushButton:hover  { background: #2d48c0; }
            QPushButton:pressed { background: #2039a0; }
            QLineEdit {
                border: 1px solid #c8cde8;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 13px;
            }
            QTableWidget {
                border: none;
                font-size: 12px;
                gridline-color: #e8eaf0;
            }
            QHeaderView::section {
                background: #eef0fa;
                color: #3a5adb;
                font-weight: bold;
                font-size: 12px;
                border: none;
                padding: 4px;
            }
            QTextEdit {
                border: none;
                font-size: 12px;
                background: #f8f9fe;
            }
        """)

    # ── 槽函数 ──────────────────────────────────────────────
    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 NIfTI 文件", "",
            "NIfTI 文件 (*.nii *.nii.gz);;所有文件 (*)"
        )
        if path:
            self.path_edit.setText(path)
            self._on_load_clicked()

    def _on_load_clicked(self):
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "提示", "请先选择或输入 NIfTI 文件路径！")
            return
        if not os.path.isfile(path):
            QMessageBox.warning(self, "错误", f"文件不存在：\n{path}")
            return

        self.load_btn.setEnabled(False)
        self.status_label.setText("⏳ 正在加载文件，请稍候…")
        self.status_label.setStyleSheet("color: #e67e00; font-size: 12px;")

        self._load_thread = NiiLoadThread(path)
        self._load_thread.finished.connect(self._on_load_finished)
        self._load_thread.error.connect(self._on_load_error)
        self._load_thread.start()

    def _on_load_finished(self, info: dict):
        self.load_btn.setEnabled(True)
        self._fill_dim_table(info)
        self._fill_attr_table(info)
        self._fill_affine_table(info['affine'])
        self._fill_header_text(info)
        self.status_label.setText(
            f"✅ 已加载：{info['filename']}  |  大小：{info['file_size_mb']:.2f} MB"
        )
        self.status_label.setStyleSheet("color: #27ae60; font-size: 12px;")

    def _on_load_error(self, msg: str):
        self.load_btn.setEnabled(True)
        self.status_label.setText(f"❌ 加载失败：{msg}")
        self.status_label.setStyleSheet("color: #c0392b; font-size: 12px;")
        QMessageBox.critical(self, "加载失败", f"无法读取 NIfTI 文件：\n\n{msg}")

    # ── 填充各控件 ──────────────────────────────────────────
    def _fill_dim_table(self, info: dict):
        """填充维度 & 分辨率表格"""
        axis_names = {0: "I (Left-Right)", 1: "J (Posterior-Anterior)",
                      2: "K (Inferior-Superior)", 3: "T (时间)"}
        shape  = info['shape']
        zooms  = info['zooms']

        self.dim_table.setRowCount(0)
        for idx in range(len(shape)):
            row = self.dim_table.rowCount()
            self.dim_table.insertRow(row)
            voxels   = shape[idx]
            res      = zooms[idx] if idx < len(zooms) else float('nan')
            phy_range = voxels * res

            items = [
                str(idx),
                axis_names.get(idx, f"轴 {idx}"),
                str(voxels),
                f"{res:.4f}",
                f"{phy_range:.2f}",
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if idx < 3:                       # 空间轴蓝色高亮
                    item.setForeground(QColor("#1a3cc8"))
                self.dim_table.setItem(row, col, item)

    def _fill_attr_table(self, info: dict):
        """填充文件属性表格"""
        units_spatial = info['xyzt_units'][0] if len(info['xyzt_units']) > 0 else 'N/A'
        units_temporal = info['xyzt_units'][1] if len(info['xyzt_units']) > 1 else 'N/A'

        dim_info_labels = ('频率编码方向', '相位编码方向', '层面方向')
        dim_info_str = ', '.join(
            f"{dim_info_labels[i]}={v}" if v is not None else f"{dim_info_labels[i]}=未设置"
            for i, v in enumerate(info['dim_info'])
            if i < 3
        )

        rows = [
            ("文件名",       info['filename']),
            ("文件大小",     f"{info['file_size_mb']:.3f} MB"),
            ("维度数 (ndim)",str(info['ndim'])),
            ("体积形状 (Shape)", str(info['shape'])),
            ("数据类型",     info['data_type']),
            ("空间单位",     units_spatial),
            ("时间单位",     units_temporal),
            ("空间分辨率 X", f"{info['zooms'][0]:.4f} {units_spatial}" if len(info['zooms']) > 0 else "N/A"),
            ("空间分辨率 Y", f"{info['zooms'][1]:.4f} {units_spatial}" if len(info['zooms']) > 1 else "N/A"),
            ("空间分辨率 Z", f"{info['zooms'][2]:.4f} {units_spatial}" if len(info['zooms']) > 2 else "N/A"),
            ("时间分辨率 TR",f"{info['zooms'][3]:.4f} {units_temporal}" if len(info['zooms']) > 3 else "N/A (非4D)"),
            ("qform_code",   str(info['qform_code'])),
            ("sform_code",   str(info['sform_code'])),
            ("dim_info",     dim_info_str),
            ("体素数据最小值", f"{info['data_min']:.4f}"),
            ("体素数据最大值", f"{info['data_max']:.4f}"),
        ]

        self.attr_table.setRowCount(0)
        for key, val in rows:
            row = self.attr_table.rowCount()
            self.attr_table.insertRow(row)
            k_item = QTableWidgetItem(key)
            k_item.setFont(QFont("", -1, QFont.Bold))
            k_item.setForeground(QColor("#333"))
            v_item = QTableWidgetItem(val)
            v_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            self.attr_table.setItem(row, 0, k_item)
            self.attr_table.setItem(row, 1, v_item)

    def _fill_affine_table(self, affine: np.ndarray):
        """填充仿射矩阵"""
        for r in range(4):
            for c in range(4):
                val = affine[r, c] if affine is not None else 0.0
                item = QTableWidgetItem(f"{val:.4f}")
                item.setTextAlignment(Qt.AlignCenter)
                # 对角线绿色
                if r == c:
                    item.setForeground(QColor("#1a7a3c"))
                    item.setFont(QFont("", -1, QFont.Bold))
                self.affine_table.setItem(r, c, item)

    def _fill_header_text(self, info: dict):
        """填充 Header 原始文本"""
        lines = [
            f"文件路径：{info['filepath']}",
            f"文件大小：{info['file_size_mb']:.3f} MB",
            "",
            "── 维度与分辨率 ──",
            f"  Shape (体素数量)：{info['shape']}",
            f"  Zooms (各轴分辨率)：{tuple(round(z, 4) for z in info['zooms'])}",
            f"  Ndim：{info['ndim']}",
            "",
            "── 坐标系信息 ──",
            f"  空间单位：{info['xyzt_units'][0] if info['xyzt_units'] else 'N/A'}",
            f"  时间单位：{info['xyzt_units'][1] if len(info['xyzt_units'])>1 else 'N/A'}",
            f"  qform_code：{info['qform_code']}",
            f"  sform_code：{info['sform_code']}",
            "",
            "── 仿射矩阵 (Affine) ──",
        ]
        if info['affine'] is not None:
            for row in info['affine']:
                lines.append("  " + "  ".join(f"{v:10.4f}" for v in row))
        lines += [
            "",
            "── 数据范围 ──",
            f"  数据类型：{info['data_type']}",
            f"  最小值：{info['data_min']:.4f}",
            f"  最大值：{info['data_max']:.4f}",
        ]
        self.header_text.setPlainText("\n".join(lines))


# ─────────────────────────── 入口 ───────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = NiiInfoViewer()
    window.show()

    # 如果命令行传入了文件路径，自动加载
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        window.path_edit.setText(sys.argv[1])
        window._on_load_clicked()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
