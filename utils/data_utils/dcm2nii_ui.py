"""
DCM → NII 一键转换 & 筛查工具
工作流：
  1. 扫描输入目录下所有 Admin_* 文件夹的子文件夹（参照 script.py）
  2. 逐个调用 dcm2niix.exe 进行转换（参照 process.py）
  3. 每转换一个文件夹后立即筛查输出目录新增文件：
       保留 XYZ 空间分辨率均为 1.0 mm（±0.05 容差）的 .nii.gz 及同名 .json
       不满足条件的立即删除（仅删除本次新增文件，绝不触碰输入目录）
依赖：PySide6, nibabel, numpy
"""

import os
import sys
import json
import subprocess
import time
import traceback
import csv
from pathlib import Path
from typing import List, Tuple, Set, Optional

# Windows平台下隐藏子进程控制台窗口
if sys.platform == 'win32':
    CREATE_NO_WINDOW = 0x08000000
else:
    CREATE_NO_WINDOW = 0


def get_resource_path(relative_path: str) -> str:
    """
    获取资源文件的绝对路径（兼容打包环境）
    
    Args:
        relative_path: 相对路径
        
    Returns:
        资源文件的绝对路径
    """
    if getattr(sys, 'frozen', False):
        # 打包后的环境（PyInstaller）
        base_path = sys._MEIPASS
    else:
        # 开发环境
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

import nibabel as nib
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QFileDialog, QTextEdit,
    QGroupBox, QProgressBar, QSplitter, QMessageBox, QFrame,
    QSpinBox, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal, QMutex, QMutexLocker, QTimer
from PySide6.QtGui import QFont, QColor, QTextCursor, QIcon

# ════════════════════════════════════════════════════════════════
#  常量
# ════════════════════════════════════════════════════════════════
RES_TOLERANCE   = 0.05          # 分辨率判定容差 (mm)
TARGET_RES      = 1.0           # 目标分辨率 (mm)
DCM2NIIX_FLAGS  = ['-f', '%f_%p_%t_%s', '-p', 'y', '-z', 'y']


# ════════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════════
def format_time(seconds: float) -> str:
    """
    将秒数格式化为可读的时间字符串
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串，如 "5m 30s" 或 "1h 25m 10s"
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def save_batch_results_csv(
    output_dir: str,
    log_records: List[Tuple[str, str]],
    timestamp: str
) -> str:
    """
    将批处理日志保存为 CSV 文件
    
    Args:
        output_dir: 输出目录
        log_records: 日志记录列表 [(消息, 级别), ...]
        timestamp: 时间戳字符串
        
    Returns:
        保存的 CSV 文件路径
    """
    csv_path = Path(output_dir) / f"batch_log_{timestamp}.csv"
    
    try:
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['时间戳', '级别', '消息'])
            
            # 写入日志记录
            for msg, level in log_records:
                # 提取时间（如果有）
                timestamp_col = time.strftime('%Y-%m-%d %H:%M:%S')
                writer.writerow([timestamp_col, level, msg])
        
        return str(csv_path)
    except Exception as e:
        return f"Error: {str(e)}"


def save_batch_summary_txt(
    output_dir: str,
    summary_data: dict,
    timestamp: str
) -> str:
    """
    将批处理统计摘要保存为 TXT 文件
    
    Args:
        output_dir: 输出目录
        summary_data: 统计数据字典
        timestamp: 时间戳字符串
        
    Returns:
        保存的 TXT 文件路径
    """
    txt_path = Path(output_dir) / f"batch_summary_{timestamp}.txt"
    
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("DCM → NII 批处理统计摘要\n")
            f.write("="*70 + "\n\n")
            
            # 基本信息
            f.write("【处理信息】\n")
            f.write(f"处理时间: {summary_data.get('timestamp', 'N/A')}\n")
            f.write(f"输入目录: {summary_data.get('input_dir', 'N/A')}\n")
            f.write(f"输出目录: {summary_data.get('output_dir', 'N/A')}\n")
            f.write(f"总耗时:   {summary_data.get('elapsed_time', 'N/A')}\n\n")
            
            # 统计信息
            f.write("【统计结果】\n")
            f.write(f"发现序列总数: {summary_data.get('total_sequences', 0)}\n")
            f.write(f"成功转换:     {summary_data.get('converted', 0)}\n")
            f.write(f"保留文件:     {summary_data.get('kept', 0)}\n")
            f.write(f"删除文件:     {summary_data.get('deleted', 0)}\n")
            f.write(f"处理失败:     {summary_data.get('failed', 0)}\n\n")
            
            # 计算成功率
            total = summary_data.get('total_sequences', 0)
            converted = summary_data.get('converted', 0)
            if total > 0:
                success_rate = (converted / total) * 100
                f.write(f"【成功率】\n")
                f.write(f"转换成功率: {success_rate:.1f}% ({converted}/{total})\n\n")
            
            # 分辨率统计（如果有）
            if 'resolution_stats' in summary_data:
                res_stats = summary_data['resolution_stats']
                f.write("【分辨率统计】\n")
                f.write(f"符合 1mm 标准: {res_stats.get('compliant', 0)}\n")
                f.write(f"不符合标准:   {res_stats.get('non_compliant', 0)}\n\n")
            
            # 文件类型统计（如果有）
            if 'file_type_stats' in summary_data:
                type_stats = summary_data['file_type_stats']
                f.write("【文件类型统计】\n")
                for file_type, count in type_stats.items():
                    f.write(f"{file_type}: {count}\n")
                f.write("\n")
            
            # 错误信息（如果有）
            if summary_data.get('errors'):
                f.write("【错误信息】\n")
                for error in summary_data['errors']:
                    f.write(f"- {error}\n")
                f.write("\n")
            
            f.write("="*70 + "\n")
            f.write("生成时间: " + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
            f.write("工具版本: MediScreen-Brain DCM2NII v1.0\n")
            f.write("="*70 + "\n")
        
        return str(txt_path)
    except Exception as e:
        return f"Error: {str(e)}"


# ════════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════════
def scan_input_dir(base_dir: str) -> List[str]:
    """
    智能扫描输入目录下的所有 DCM 序列文件夹。
    
    支持多种目录结构：
    1. 直接包含 DCM 文件的目录（无子文件夹）
    2. 包含 Admin_* 子文件夹的目录（原有逻辑）
    3. 任意嵌套的子文件夹结构
    4. 混合结构（部分有 Admin_，部分没有）
    
    识别规则：
    - 如果目录下有 .dcm/.DCM 文件，则该目录本身是 DCM 序列目录
    - 如果目录下有 Admin_* 子文件夹，则递归扫描其子文件夹
    - 否则，将所有子文件夹视为潜在的 DCM 序列目录
    
    Args:
        base_dir: 输入目录路径
        
    Returns:
        DCM 序列文件夹路径列表（已排序）
    """
    paths = []
    base = Path(base_dir).resolve()
    
    if not base.exists() or not base.is_dir():
        return paths
    
    def has_dicom_files(dir_path: Path) -> bool:
        """检查目录是否包含 DICOM 文件"""
        dicom_extensions = {'.dcm', '.DCM', '.dicom', '.DICOM'}
        try:
            for f in dir_path.iterdir():
                if f.is_file():
                    # 检查扩展名或无扩展名但可能是 DICOM
                    if f.suffix.lower() in dicom_extensions:
                        return True
                    # DICOM 文件可能没有扩展名，检查文件大小（DICOM 通常 > 100KB）
                    if not f.suffix and f.stat().st_size > 100 * 1024:
                        return True
        except (PermissionError, OSError):
            pass
        return False
    
    def scan_directory(dir_path: Path, depth: int = 0):
        """递归扫描目录"""
        # 限制递归深度，避免过深嵌套
        if depth > 5:
            return
        
        try:
            items = sorted(dir_path.iterdir())
        except (PermissionError, OSError):
            return
        
        # 检查当前目录是否包含 DICOM 文件
        if has_dicom_files(dir_path):
            paths.append(str(dir_path))
            return  # 找到 DICOM 文件后不再深入扫描
        
        # 检查是否有 Admin_* 子文件夹
        admin_folders = [item for item in items if item.is_dir() and item.name.startswith('Admin_')]
        
        if admin_folders:
            # 如果有 Admin_* 文件夹，只扫描这些文件夹的直接子文件夹
            for admin_folder in admin_folders:
                try:
                    for sub in sorted(admin_folder.iterdir()):
                        if sub.is_dir():
                            # 递归检查子文件夹
                            scan_directory(sub, depth + 1)
                except (PermissionError, OSError):
                    continue
        else:
            # 没有 Admin_* 文件夹，扫描所有子文件夹
            subdirs = [item for item in items if item.is_dir()]
            
            if subdirs:
                # 有子文件夹，继续递归扫描
                for subdir in subdirs:
                    scan_directory(subdir, depth + 1)
            else:
                # 没有子文件夹，但之前已检查过没有 DICOM 文件
                # 这可能是空的或不相关的目录，跳过
                pass
    
    # 开始扫描
    scan_directory(base)
    
    # 去重并排序
    unique_paths = sorted(set(paths))
    
    return unique_paths


def snapshot_output_dir(output_dir: str) -> Set[str]:
    """返回输出目录当前所有文件的绝对路径集合（快照）"""
    d = Path(output_dir)
    if not d.exists():
        return set()
    return {str(p.resolve()) for p in d.iterdir() if p.is_file()}


def check_resolution(nii_path: str) -> Tuple[bool, Tuple[float, ...]]:
    """
    读取 .nii.gz Header，返回 (是否满足 1mm, 实际分辨率 zooms[:3])。
    仅读 Header，不加载体素数据，速度极快。
    """
    try:
        hdr = nib.load(nii_path).header
        zooms = tuple(float(z) for z in hdr.get_zooms()[:3])
        ok = all(abs(z - TARGET_RES) <= RES_TOLERANCE for z in zooms)
        return ok, zooms
    except Exception:
        return False, (0.0, 0.0, 0.0)


def check_voxel_dimensions(nii_path: str) -> Tuple[bool, Tuple[int, ...]]:
    """
    读取 .nii.gz Header，返回 (是否各轴体素数量 > 100, 实际体素维度)。
    仅读 Header，不加载体素数据，速度极快。
    
    Args:
        nii_path: NIfTI 文件路径
        
    Returns:
        (是否满足条件, 体素维度元组 (dim_x, dim_y, dim_z))
    """
    try:
        img = nib.load(nii_path)
        # 获取图像数据的形状（体素维度）
        shape = img.shape[:3]  # 只取前三个维度 (x, y, z)
        voxel_dims = tuple(int(d) for d in shape)
        # 检查所有维度是否都大于 100
        ok = all(d > 100 for d in voxel_dims)
        return ok, voxel_dims
    except Exception:
        return False, (0, 0, 0)


def filter_new_files(
    before: Set[str],
    after: Set[str],
    output_dir: str,
) -> Tuple[int, int, List[str]]:
    """
    对本次新增的 .nii.gz 文件做分辨率和体素数量筛查：
      - 满足条件：各轴分辨率为 1mm（±0.05）且各轴体素数量 > 100
      - 保留：满足条件的文件（连同同名 .json/.bval/.bvec）
      - 删除：不满足条件的文件（连同同名 .json/.bval/.bvec）
    
    支持的配套文件类型：
      - .json: BIDS 元数据文件
      - .bval: DTI b-values 文件
      - .bvec: DTI b-vectors 文件
    
    Args:
        before: 转换前的文件快照
        after: 转换后的文件快照
        output_dir: 输出目录路径
        
    Returns:
        (保留数, 删除数, 日志行列表)
        
    Note:
        绝不处理 output_dir 之外的文件。
    """
    new_files = after - before
    out_root = str(Path(output_dir).resolve())
    logs = []
    kept = deleted = 0
    
    # 定义所有需要处理的配套文件扩展名
    companion_extensions = ['.json', '.bval', '.bvec']

    nii_files = [f for f in new_files if f.endswith('.nii.gz')]

    for nii_path in nii_files:
        # 安全检查：确保文件在输出目录内
        if not nii_path.startswith(out_root):
            logs.append(f"  [跳过] 路径不在输出目录内（安全保护）: {nii_path}")
            continue

        # 检查分辨率条件
        res_ok, zooms = check_resolution(nii_path)
        # 检查体素数量条件
        voxel_ok, voxel_dims = check_voxel_dimensions(nii_path)
        
        # 两个条件都必须满足
        ok = res_ok and voxel_ok
        
        base_stem = nii_path[:-len('.nii.gz')]   # 去除 .nii.gz 后缀
        
        # 构建所有可能的配套文件路径
        companion_files = {}
        for ext in companion_extensions:
            companion_path = base_stem + ext
            if os.path.isfile(companion_path):
                companion_files[ext] = companion_path

        if ok:
            # 保留主文件和所有配套文件
            kept += 1
            log_msg = f"  [保留] {os.path.basename(nii_path)}  分辨率={zooms}, 体素维度={voxel_dims}"
            
            # 记录保留的配套文件
            if companion_files:
                kept_companions = [os.path.basename(p) for p in companion_files.values()]
                log_msg += f" (+ {', '.join(kept_companions)})"
            
            logs.append(log_msg)
        else:
            # 构建详细的删除原因
            reasons = []
            if not res_ok:
                reasons.append(f"分辨率={zooms} ≠ 1mm")
            if not voxel_ok:
                reasons.append(f"体素维度={voxel_dims} ≤ 100")
            reason_str = ", ".join(reasons)
            
            # 删除主文件 .nii.gz
            try:
                os.remove(nii_path)
                logs.append(f"  [删除] {os.path.basename(nii_path)}  {reason_str}")
            except Exception as e:
                logs.append(f"  [删除失败] {os.path.basename(nii_path)}: {e}")

            # 删除所有配套文件
            for ext, comp_path in companion_files.items():
                try:
                    os.remove(comp_path)
                    file_type = {
                        '.json': '配套json',
                        '.bval': '配套bval',
                        '.bvec': '配套bvec'
                    }.get(ext, ext)
                    logs.append(f"  [删除] {os.path.basename(comp_path)} ({file_type})")
                except Exception as e:
                    logs.append(f"  [删除失败] {os.path.basename(comp_path)}: {e}")
            
            deleted += 1

    # 处理孤立的配套文件（nii.gz 已被删，但配套文件未统计到）
    for f in new_files:
        # 安全检查
        if not f.startswith(out_root):
            continue
            
        # 检查是否是配套文件类型
        is_companion = False
        for ext in companion_extensions:
            if f.endswith(ext):
                is_companion = True
                break
        
        if is_companion:
            # 获取基础文件名（去掉扩展名）
            stem = Path(f).stem  # 去掉最后一个扩展名
            # 对于 .nii.gz 需要特殊处理
            if stem.endswith('.nii'):
                stem = stem[:-4]  # 再去掉 .nii
            
            # 检查对应的 .nii.gz 是否存在
            nii_equiv = stem + '.nii.gz'
            if not os.path.isfile(nii_equiv):
                # 确定文件类型
                file_ext = Path(f).suffix
                if f.endswith('.nii.gz'):
                    file_ext = '.nii.gz'
                
                file_type = {
                    '.json': '孤立json',
                    '.bval': '孤立bval',
                    '.bvec': '孤立bvec'
                }.get(file_ext, '孤立文件')
                
                try:
                    os.remove(f)
                    logs.append(f"  [删除] {os.path.basename(f)} ({file_type})")
                except Exception:
                    pass

    return kept, deleted, logs


# ════════════════════════════════════════════════════════════════
#  后台工作线程
# ════════════════════════════════════════════════════════════════
class WorkerThread(QThread):
    # 信号定义
    log        = Signal(str, str)   # (消息, 颜色标识: 'info'|'ok'|'warn'|'error')
    progress   = Signal(int, int)   # (当前, 总数)
    stats      = Signal(int, int, int, int)  # (已转换, 已保留, 已删除, 失败)
    finished   = Signal(bool, str, dict)  # (成功, 摘要, 统计数据)

    def __init__(self, input_dir: str, output_dir: str, exe_path: str):
        super().__init__()
        self.input_dir  = input_dir
        self.output_dir = output_dir
        self.exe_path   = exe_path
        self._abort     = False
        self._mutex     = QMutex()
        self._log_records: List[Tuple[str, str]] = []  # 收集所有日志记录

    def abort(self):
        with QMutexLocker(self._mutex):
            self._abort = True
    
    def _add_log_record(self, msg: str, level: str):
        """添加日志记录到收集列表"""
        self._log_records.append((msg, level))
    
    def _emit_log(self, msg: str, level: str):
        """发送日志信号并收集记录"""
        self._add_log_record(msg, level)
        self.log.emit(msg, level)

    def _is_aborted(self):
        with QMutexLocker(self._mutex):
            return self._abort

    def run(self):
        try:
            self._execute()
        except Exception as e:
            error_msg = f"[严重错误] {traceback.format_exc()}"
            self._add_log_record(error_msg, 'error')
            self.log.emit(error_msg, 'error')
            self.finished.emit(False, str(e), {})

    def _execute(self):
        t0 = time.time()

        # ── 步骤1：扫描输入目录 ──
        self._emit_log("▶ 开始扫描输入目录…", 'info')
        paths = scan_input_dir(self.input_dir)
        total = len(paths)

        if total == 0:
            self._emit_log("⚠ 未找到任何 DCM 序列文件夹，请确认输入目录结构。", 'warn')
            self.finished.emit(False, "未发现可处理路径", {})
            return

        self._emit_log(f"✅ 扫描完成，共发现 {total} 个 DCM 序列文件夹", 'ok')
        self.progress.emit(0, total)

        # ── 步骤2：创建输出目录 ──
        os.makedirs(self.output_dir, exist_ok=True)
        self._emit_log(f"📁 输出目录：{self.output_dir}", 'info')

        # ── 步骤3：逐个转换 + 立即筛查 ──
        converted = kept = deleted = failed = 0

        for idx, dcm_path in enumerate(paths, start=1):
            if self._is_aborted():
                self._emit_log("⏹ 用户已中止处理", 'warn')
                break

            self._emit_log(
                f"\n[{idx}/{total}] 转换: {os.path.basename(dcm_path)}", 'info'
            )
            self._emit_log(f"  路径: {dcm_path}", 'info')

            # 转换前快照
            before = snapshot_output_dir(self.output_dir)

            # 调用 dcm2niix（Windows下隐藏控制台窗口）
            cmd = [self.exe_path] + DCM2NIIX_FLAGS + ['-o', self.output_dir, dcm_path]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    creationflags=CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
            except subprocess.TimeoutExpired:
                self._emit_log(f"  ⚠ 超时（300s），跳过: {dcm_path}", 'warn')
                failed += 1
                self.progress.emit(idx, total)
                self.stats.emit(converted, kept, deleted, failed)
                continue
            except FileNotFoundError:
                self._emit_log(
                    f"  ❌ 找不到 dcm2niix.exe：{self.exe_path}\n  请确认路径是否正确", 'error'
                )
                self.finished.emit(False, "dcm2niix.exe 不存在", {})
                return

            if result.returncode != 0:
                failed += 1
                self._emit_log(f"  ❌ 转换失败 (code={result.returncode})", 'error')
                if result.stderr.strip():
                    for line in result.stderr.strip().splitlines()[-5:]:
                        self._emit_log(f"    {line}", 'error')
            else:
                converted += 1
                self._emit_log(f"  ✅ dcm2niix 转换完成", 'ok')

            # 转换后快照 + 立即筛查
            after = snapshot_output_dir(self.output_dir)
            new_count = len(after - before)
            self._emit_log(f"  📊 本次新增文件: {new_count} 个，开始筛查…", 'info')

            k, d, filter_logs = filter_new_files(before, after, self.output_dir)
            kept    += k
            deleted += d
            for line in filter_logs:
                color = 'ok' if '保留' in line else ('warn' if '跳过' in line else 'error')
                self._emit_log(line, color)

            self._emit_log(
                f"  筛查结果：保留 {k} 个 / 删除 {d} 个", 'info'
            )

            self.progress.emit(idx, total)
            self.stats.emit(converted, kept, deleted, failed)

        # ── 完成 ──
        elapsed = time.time() - t0
        summary = (
            f"处理完成！耗时 {elapsed:.1f}s | "
            f"转换 {converted} | 保留 {kept} | 删除 {deleted} | 失败 {failed}"
        )
        self.log.emit(f"\n{'═'*60}", 'info')
        self.log.emit(f"🎉 {summary}", 'ok')
        
        # 生成时间戳
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # 准备统计数据
        summary_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'input_dir': self.input_dir,
            'output_dir': self.output_dir,
            'elapsed_time': format_time(elapsed),
            'total_sequences': total,
            'converted': converted,
            'kept': kept,
            'deleted': deleted,
            'failed': failed,
        }
        
        # 保存日志和统计信息
        try:
            # 保存 CSV 日志
            csv_path = save_batch_results_csv(self.output_dir, self._log_records, timestamp)
            self.log.emit(f"\n📄 日志已保存: {csv_path}", 'info')
            
            # 保存 TXT 统计摘要
            txt_path = save_batch_summary_txt(self.output_dir, summary_data, timestamp)
            self.log.emit(f"📊 统计摘要已保存: {txt_path}", 'info')
        except Exception as e:
            self.log.emit(f"⚠ 保存文件失败: {str(e)}", 'warn')
        
        self.finished.emit(True, summary, summary_data)


# ════════════════════════════════════════════════════════════════
#  主窗口
# ════════════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 窗口基础配置
        self.setWindowTitle("DCM → NII 转换 & 筛查工具 — MediScreen-Brain")
        self.setMinimumSize(1000, 720)
        
        # 尝试设置窗口图标（如果存在）
        icon_path = get_resource_path("icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # 状态变量初始化
        self._worker: Optional[WorkerThread] = None  # 后台工作线程
        self._last_input_dir = ""   # 上次使用的输入目录
        self._last_output_dir = ""  # 上次使用的输出目录
        self._start_time: float = 0  # 处理开始时间
        self._time_history: List[float] = []  # 每个任务耗时历史记录
        
        # 构建界面和应用样式
        self._build_ui()
        self._apply_style()
        
        # 将窗口居中显示
        self._center_window()
        
        # 创建状态栏
        self.statusBar().showMessage("就绪 | 选择输入目录和输出目录后点击“开始处理”")
        
        # 延迟显示安全说明（等待主窗口完全显示后）
        QTimer.singleShot(500, self._show_safety_notice)

    # ── UI 构建 ─────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 14)

        # ── 路径配置区 ──
        cfg_group = QGroupBox("路径配置")
        cfg_layout = QVBoxLayout(cfg_group)
        cfg_layout.setSpacing(8)

        # 输入目录（支持多种结构）
        row_in = QHBoxLayout()
        lbl_in = QLabel("输入目录  ")
        lbl_in.setFixedWidth(90)
        lbl_in.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("DCM 文件所在目录（支持 Admin_* 结构或直接包含 DCM 的文件夹）")
        self.input_edit.setToolTip("选择包含 DCM 文件的目录，程序将智能扫描：\n1. 直接包含 DCM 文件的目录\n2. Admin_* 子文件夹结构\n3. 任意嵌套的子文件夹")
        btn_in = QPushButton("浏览…")
        btn_in.setMinimumWidth(90)
        btn_in.clicked.connect(lambda: self._browse_dir(self.input_edit))
        row_in.addWidget(lbl_in)
        row_in.addWidget(self.input_edit)
        row_in.addWidget(btn_in)

        # 输出目录
        row_out = QHBoxLayout()
        lbl_out = QLabel("输出目录  ")
        lbl_out.setFixedWidth(90)
        lbl_out.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("转换结果保存目录（如 H:\\data\\tumor_data_swust\\data2_niigz）")
        self.output_edit.setToolTip("选择 NIfTI 文件的输出目录，不能是输入目录的子目录")
        btn_out = QPushButton("浏览…")
        btn_out.setMinimumWidth(90)
        btn_out.clicked.connect(lambda: self._browse_dir(self.output_edit))
        row_out.addWidget(lbl_out)
        row_out.addWidget(self.output_edit)
        row_out.addWidget(btn_out)

        cfg_layout.addLayout(row_in)
        cfg_layout.addLayout(row_out)

        # 筛查参数提示
        tip = QLabel(
            f"筛查规则：保留 X/Y/Z 空间分辨率均为 {TARGET_RES} mm（容差 ±{RES_TOLERANCE} mm）且各轴体素数量 > 100 的 .nii.gz 及同名 .json/.bval/.bvec，其余删除"
        )
        tip.setStyleSheet("color:#555; font-size:11px; padding-left:6px;")
        cfg_layout.addWidget(tip)

        root.addWidget(cfg_group)

        # ── 操作按钮区 ──
        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶  开始处理")
        self.start_btn.setFixedHeight(36)
        self.start_btn.setToolTip("开始执行 DCM 到 NIfTI 的转换和筛查任务")
        self.start_btn.clicked.connect(self._on_start)

        self.abort_btn = QPushButton("⏹  中止")
        self.abort_btn.setFixedHeight(36)
        self.abort_btn.setEnabled(False)
        self.abort_btn.setToolTip("中止当前正在进行的处理任务")
        self.abort_btn.clicked.connect(self._on_abort)

        self.clear_btn = QPushButton("清空日志")
        self.clear_btn.setFixedHeight(36)
        self.clear_btn.setToolTip("清空日志显示区域")
        self.clear_btn.clicked.connect(self._clear_log)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.abort_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.clear_btn)
        root.addLayout(btn_row)

        # ── 进度区 ──
        prog_group = QGroupBox("处理进度")
        prog_layout = QVBoxLayout(prog_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% | %v / %m")
        prog_layout.addWidget(self.progress_bar)
        
        # 时间信息行
        time_row = QHBoxLayout()
        self.elapsed_label = QLabel("已用时间: 0s")
        self.elapsed_label.setStyleSheet("color:#555; font-size:12px;")
        self.eta_label = QLabel("预计剩余: --")
        self.eta_label.setStyleSheet("color:#3a5adb; font-size:12px; font-weight:bold;")
        time_row.addWidget(self.elapsed_label)
        time_row.addStretch()
        time_row.addWidget(self.eta_label)
        prog_layout.addLayout(time_row)

        # 统计数字行
        stat_row = QHBoxLayout()
        self._stat_labels = {}
        for key, color, text in [
            ('converted', '#1a5adb', '已转换'),
            ('kept',      '#1a7a3c', '已保留'),
            ('deleted',   '#c0392b', '已删除'),
            ('failed',    '#e67e00', '失败'),
        ]:
            lbl = QLabel(f"{text}: 0")
            lbl.setStyleSheet(f"color:{color}; font-weight:bold; font-size:13px;")
            self._stat_labels[key] = lbl
            stat_row.addWidget(lbl)
        stat_row.addStretch()
        prog_layout.addLayout(stat_row)

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color:#888; font-size:12px;")
        prog_layout.addWidget(self.status_label)

        root.addWidget(prog_group)

        # ── 日志区 ──
        log_group = QGroupBox("处理日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setLineWrapMode(QTextEdit.NoWrap)
        log_layout.addWidget(self.log_text)
        root.addWidget(log_group, stretch=1)

    # ── 样式 ────────────────────────────────────────────────
    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #f5f6fa; }
            QGroupBox {
                font-weight: bold; font-size: 13px;
                border: 1px solid #d0d4e0; border-radius: 6px;
                margin-top: 8px; padding-top: 6px; background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px;
                padding: 0 4px; color: #3a5adb;
            }
            QPushButton {
                background: #3a5adb; color: white;
                border: none; border-radius: 4px;
                padding: 5px 14px; font-size: 13px;
            }
            QPushButton:hover  { background: #2d48c0; }
            QPushButton:pressed { background: #1e32a0; }
            QPushButton:disabled { background: #b0b8d8; color: #e0e4f0; }
            QPushButton#clear_btn { background: #7f8c8d; }
            QPushButton#clear_btn:hover { background: #6d7a7b; }
            QLineEdit {
                border: 1px solid #c8cde8; border-radius: 4px;
                padding: 4px 8px; font-size: 13px;
            }
            QProgressBar {
                border: 1px solid #c8cde8; border-radius: 4px;
                text-align: center; font-size: 12px; height: 22px;
            }
            QProgressBar::chunk { background: #3a5adb; border-radius: 3px; }
            QTextEdit { border: none; background: #1e1e2e; color: #cdd6f4; }
        """)
        self.abort_btn.setObjectName("abort_btn")
        self.clear_btn.setObjectName("clear_btn")
    
    def _center_window(self):
        """将窗口居中显示在屏幕中央"""
        # 使用 PySide6 的现代 API 获取屏幕几何信息
        from PySide6.QtGui import QScreen
        from PySide6.QtWidgets import QApplication
        
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        window_geometry = self.frameGeometry()
        
        # 计算居中位置
        x = (screen_geometry.width() - window_geometry.width()) // 2
        y = (screen_geometry.height() - window_geometry.height()) // 2
        
        self.move(x, y)
    
    def _show_safety_notice(self):
        """显示安全说明对话框，告知用户软件的删除规则"""
        notice_text = (
            "欢迎使用 DCM → NII 转换 & 筛查工具！\n\n"
            "【安全说明】\n"
            "• 本软件仅对 dcm2niix 转换后的输出文件进行筛查和删除操作\n"
            "• 绝不会修改或删除原始输入目录中的任何 DCM 文件\n"
            "• 程序会自动检测并确保输出目录不是输入目录的子目录\n\n"
            "【筛查规则】\n"
            f"• 保留：各轴分辨率均为 {TARGET_RES} mm（±{RES_TOLERANCE}）且体素维度 > 100 的文件\n"
            "• 删除：不满足上述条件的 .nii.gz 及其配套文件（.json/.bval/.bvec）\n\n"
            "请放心使用，您的原始数据始终安全！"
        )
        
        QMessageBox.information(
            self,
            "安全说明",
            notice_text,
            QMessageBox.Ok
        )

    # ── 槽函数 ──────────────────────────────────────────────
    def _browse_dir(self, edit: QLineEdit):
        """浏览选择目录，并记住用户的选择"""
        # 优先使用当前输入框的值，其次使用上次选择的目录
        default_dir = edit.text() or (self._last_input_dir if edit == self.input_edit else self._last_output_dir)
        d = QFileDialog.getExistingDirectory(self, "选择目录", default_dir or "")
        if d:
            edit.setText(d)
            # 保存最后使用的目录
            if edit == self.input_edit:
                self._last_input_dir = d
            elif edit == self.output_edit:
                self._last_output_dir = d

    def _browse_file(self, edit: QLineEdit, filt: str):
        f, _ = QFileDialog.getOpenFileName(self, "选择文件", edit.text() or "", filt)
        if f:
            edit.setText(f)

    def _clear_log(self):
        self.log_text.clear()

    def _on_start(self):
        input_dir  = self.input_edit.text().strip()
        output_dir = self.output_edit.text().strip()
        
        # 自动获取 dcm2niix.exe 路径（使用打包资源路径）
        exe_path = get_resource_path("dcm2niix.exe")

        # 输入校验
        errors = []
        if not input_dir:
            errors.append("请填写输入目录")
        elif not os.path.isdir(input_dir):
            errors.append(f"输入目录不存在：{input_dir}")
        if not output_dir:
            errors.append("请填写输出目录")
        if not os.path.isfile(exe_path):
            errors.append(f"找不到 dcm2niix.exe：{exe_path}")
        if errors:
            QMessageBox.warning(self, "参数不完整", "\n".join(errors))
            return

        # 安全确认：输出目录不能是输入目录的子目录或相同
        in_abs  = str(Path(input_dir).resolve())
        out_abs = str(Path(output_dir).resolve())
        if out_abs == in_abs or out_abs.startswith(in_abs + os.sep):
            QMessageBox.critical(
                self, "安全警告",
                "输出目录不能是输入目录或其子目录！\n这会导致误删原始 DCM 数据。"
            )
            return

        # 更新 UI 状态
        self.start_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(1)
        self._update_stats(0, 0, 0, 0)
        self.status_label.setText("⏳ 处理中…")
        self.status_label.setStyleSheet("color:#e67e00; font-size:12px;")
        self.statusBar().showMessage("正在处理 DCM 文件，请稍候...")
        
        # 初始化时间跟踪
        import time as time_module
        self._start_time = time_module.time()
        self._time_history = []
        self.elapsed_label.setText("已用时间: 0s")
        self.eta_label.setText("预计剩余: --")

        # 启动工作线程
        self._worker = WorkerThread(input_dir, output_dir, exe_path)
        self._worker.log.connect(self._append_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.stats.connect(self._update_stats)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_abort(self):
        if self._worker:
            self._worker.abort()
        self.abort_btn.setEnabled(False)
        self.status_label.setText("⏹ 正在中止，等待当前任务完成…")
        self.statusBar().showMessage("用户已请求中止处理...")

    def _on_progress(self, cur: int, total: int):
        """
        更新进度条并计算预计完成时间
        
        Args:
            cur: 当前完成的数量
            total: 总数量
        """
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(cur)
        
        # 计算时间信息
        if cur > 0 and self._start_time > 0:
            import time as time_module
            current_time = time_module.time()
            elapsed = current_time - self._start_time
            
            # 更新已用时间
            self.elapsed_label.setText(f"已用时间: {format_time(elapsed)}")
            
            # 记录每个任务的耗时（用于更准确的预测）
            avg_time_per_task = elapsed / cur
            self._time_history.append(avg_time_per_task)
            
            # 计算预计剩余时间
            remaining_tasks = total - cur
            if remaining_tasks > 0:
                # 使用最近 5 个任务的平均时间（如果有的话），否则使用总体平均
                if len(self._time_history) >= 5:
                    recent_avg = sum(self._time_history[-5:]) / 5
                    eta_seconds = recent_avg * remaining_tasks
                else:
                    eta_seconds = avg_time_per_task * remaining_tasks
                
                self.eta_label.setText(f"预计剩余: {format_time(eta_seconds)}")
            else:
                self.eta_label.setText("预计剩余: 即将完成")
        else:
            # 刚开始，还没有数据
            if cur == 0:
                self.elapsed_label.setText("已用时间: 0s")
                self.eta_label.setText("预计剩余: 计算中...")

    def _update_stats(self, converted: int, kept: int, deleted: int, failed: int):
        self._stat_labels['converted'].setText(f"已转换: {converted}")
        self._stat_labels['kept'].setText(f"已保留: {kept}")
        self._stat_labels['deleted'].setText(f"已删除: {deleted}")
        self._stat_labels['failed'].setText(f"失败: {failed}")

    def _on_finished(self, success: bool, summary: str, summary_data: dict = None):
        self.start_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        
        # 计算总耗时
        if self._start_time > 0:
            import time as time_module
            total_elapsed = time_module.time() - self._start_time
            self.elapsed_label.setText(f"总耗时: {format_time(total_elapsed)}")
            self.eta_label.setText("✅ 已完成")
        
        if success:
            self.status_label.setText(f"✅ {summary}")
            self.status_label.setStyleSheet("color:#27ae60; font-size:12px;")
            self.statusBar().showMessage(f"处理完成！{summary}")
            
            # 显示保存文件的通知
            if summary_data:
                QMessageBox.information(
                    self,
                    "批处理完成",
                    f"{summary}\n\n日志和统计摘要已保存到输出目录：\n{self.output_edit.text()}"
                )
        else:
            self.status_label.setText(f"❌ {summary}")
            self.status_label.setStyleSheet("color:#c0392b; font-size:12px;")
            self.statusBar().showMessage(f"处理失败：{summary}")

    # ── 日志着色输出 ──────────────────────────────────────
    _COLOR_MAP = {
        'info':  '#cdd6f4',   # 默认白
        'ok':    '#a6e3a1',   # 绿
        'warn':  '#f9e2af',   # 黄
        'error': '#f38ba8',   # 红
    }

    def _append_log(self, msg: str, level: str):
        color = self._COLOR_MAP.get(level, '#cdd6f4')
        # 使用 HTML 保留换行
        html = (
            f'<span style="color:{color}; white-space:pre;">'
            + msg.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            + '</span><br>'
        )
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
        self.log_text.insertHtml(html)
        self.log_text.ensureCursorVisible()

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            reply = QMessageBox.question(
                self, "确认退出",
                "处理任务仍在运行，确定要退出吗？\n（当前文件夹处理完后才会停止）",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._worker.abort()
                self._worker.wait(5000)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
    
    def resizeEvent(self, event):
        """窗口大小改变事件，可用于保存窗口状态"""
        super().resizeEvent(event)
        # 这里可以添加保存窗口大小的逻辑，例如保存到配置文件


# ════════════════════════════════════════════════════════════════
#  入口
# ════════════════════════════════════════════════════════════════
def main():
    os.environ.setdefault('PYTHONUTF8', '1')
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()

    # 命令行预填参数（可选）
    if len(sys.argv) >= 3:
        win.input_edit.setText(sys.argv[1])
        win.output_edit.setText(sys.argv[2])

    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
