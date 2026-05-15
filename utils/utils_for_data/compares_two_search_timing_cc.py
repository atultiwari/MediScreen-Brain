import cv2
import nibabel as nib
import numpy as np
from scipy import ndimage
from ultralytics import YOLO
import os
import time
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, List, Dict


@dataclass
class Config:
    """配置类 - 用于方法1"""
    model_path: str = r'H:\pycharm_project\PI-MAPP\project\detection_train\tumor\runs\detect\train_yolo12_try_owndata2\weights\best.pt'
    nii_path: str = r'../data_test/545140_swust.nii.gz'
    output_project: str = '../best_slices_results_test/545140_swust'
    conf: float = 0.65

    AXIAL_SAMPLING_STEP: int = 2
    SAGITTAL_SAMPLING_STEP: int = 2
    CORONAL_SAMPLING_STEP: int = 2
    SLICE_THRESHOLD: Optional[float] = None  # 如果为None则使用自适应阈值
    STD_THRESHOLD: Optional[float] = None  # 如果为None则使用自适应阈值
    REFINE_RADIUS: int = 1
    HIGH_CONF_FOR_OTHER_AXES: float = 0.92
    USE_ADAPTIVE_THRESHOLDS: bool = True  # 是否使用自适应阈值


@dataclass
class SearchResult:
    """搜索结果数据类"""
    success: bool
    has_tumor: bool = True
    best_slice_idx: Optional[int] = None
    best_axis: Optional[str] = None
    max_area: float = 0.0
    processing_time: float = 0.0
    slices_processed: int = 0
    slices_skipped: int = 0
    message: str = ""

    axial_slice: Optional[int] = None
    sagittal_slice: Optional[int] = None
    coronal_slice: Optional[int] = None
    tumor_center: Optional[Tuple[int, int, int]] = None
    detection_areas: Optional[Dict[str, float]] = None
    units: Optional[Dict] = None
    statistics: Optional[Dict] = None


class FullSearchDetector:
    """
    方法1: 全轴暴力搜索检测器

    特点:
    - 遍历所有三个轴的所有切片
    - 支持无肿瘤检测
    - 简单直接,但计算量大
    """

    def __init__(self, model, nii_path: str, output_project: str,
                 conf: float = 0.65, high_conf_for_other_axes: float = 0.92,
                 tumor_area_threshold: float = 100.0,
                 slice_threshold: Optional[float] = None,
                 std_threshold: Optional[float] = None,
                 use_adaptive_thresholds: bool = True):
        """
        初始化全轴搜索检测器

        Args:
            model: 已加载的YOLO模型实例
            nii_path: NIfTI文件路径
            output_project: 输出目录
            conf: 轴向搜索置信度阈值
            high_conf_for_other_axes: 矢状面和冠状面搜索置信度阈值
            tumor_area_threshold: 判断是否存在肿瘤的面积阈值
            slice_threshold: 切片预过滤阈值(非零像素比例)，如果为None且use_adaptive_thresholds=True则自动计算
            std_threshold: 切片标准差阈值，如果为None且use_adaptive_thresholds=True则自动计算
            use_adaptive_thresholds: 是否使用自适应阈值（推荐True）
        """
        self.model = model
        self.nii_path = nii_path
        self.output_project = output_project
        self.conf = conf
        self.high_conf_for_other_axes = high_conf_for_other_axes
        self.tumor_area_threshold = tumor_area_threshold
        self.use_adaptive_thresholds = use_adaptive_thresholds

        self.data = None
        self.I, self.J, self.K = 0, 0, 0
        self.spacing = None
        self.adaptive_thresholds = None

        self._load_data()

        # 初始化自适应阈值
        if self.use_adaptive_thresholds:
            self.adaptive_thresholds = AdaptiveThresholds(self.data)
            self.slice_threshold = slice_threshold if slice_threshold is not None else self.adaptive_thresholds.slice_nonzero_thresh()
            self.std_threshold = std_threshold if std_threshold is not None else self.adaptive_thresholds.slice_std_thresh()
            print(f"📊 自适应阈值: slice_threshold={self.slice_threshold:.4f}, std_threshold={self.std_threshold:.4f}")
        else:
            # 使用固定阈值
            self.slice_threshold = slice_threshold if slice_threshold is not None else 0.05
            self.std_threshold = std_threshold if std_threshold is not None else 5.0
            print(f"📊 固定阈值: slice_threshold={self.slice_threshold}, std_threshold={self.std_threshold}")

        os.makedirs(self.output_project, exist_ok=True)

    def _load_data(self):
        """加载NIfTI数据"""
        print("🔄 Loading NIfTI data...")
        try:
            img = nib.load(self.nii_path)
            data_raw = img.get_fdata()
            self.spacing = img.header.get_zooms()

            if data_raw.dtype != np.uint8:
                data_raw = np.clip(data_raw, 0, np.percentile(data_raw, 99))
                data_raw = ((data_raw - data_raw.min()) /
                            (data_raw.max() - data_raw.min()) * 255).astype(np.uint8)

            self.data = data_raw
            self.I, self.J, self.K = self.data.shape

            print(f"✅ Data loaded: shape={self.data.shape}, spacing={self.spacing}")
        except Exception as e:
            raise RuntimeError(f"Failed to load NIfTI data: {e}")

    @staticmethod
    def pre_filter_slice(slice_2d, non_zero_thresh=0.05, std_thresh=5.0):
        """预过滤切片:基于非零比例和标准差"""
        non_zero_ratio = np.count_nonzero(slice_2d) / slice_2d.size
        if non_zero_ratio < non_zero_thresh:
            return False
        slice_std = np.std(slice_2d)
        if slice_std < std_thresh:
            return False
        return True

    def run_inference_on_slice(self, slice_2d, conf_threshold):
        """在单个切片上运行推理"""
        slice_rgb = np.stack([slice_2d] * 3, axis=-1)
        results = self.model.predict(
            source=slice_rgb,
            conf=conf_threshold,
            verbose=False,
            device=None,
            save=False
        )

        max_area = 0
        best_box = None
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xywh.cpu().numpy()
                areas = boxes[:, 2] * boxes[:, 3]
                max_idx = areas.argmax()
                if areas[max_idx] > max_area:
                    max_area = areas[max_idx]
                    best_box = boxes[max_idx]

        return max_area, best_box

    def extract_slice(self, axis: str, index: int) -> np.ndarray:
        """根据轴和索引提取切片"""
        if axis == 'axial':
            return self.data[:, :, index]
        elif axis == 'sagittal':
            return self.data[index, :, :]
        elif axis == 'coronal':
            return self.data[:, index, :]
        else:
            raise ValueError(f"Invalid axis: {axis}")

    def save_best_slice(self, result: SearchResult, output_dir: str):
        """
        保存最佳切片图像（仅保存，不绘制检测框）

        Args:
            result: 搜索结果
            output_dir: 输出目录

        Returns:
            保存的文件路径列表
        """
        if not result.has_tumor or result.best_axis is None or result.best_slice_idx is None:
            print("⚠️  未检测到肿瘤，跳过保存切片")
            return []

        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 提取并保存最佳切片所在的轴
        slice_2d = self.extract_slice(result.best_axis, result.best_slice_idx)
        slice_rgb = np.ascontiguousarray(np.stack([slice_2d] * 3, axis=-1))

        filename = f"best_slice_{result.best_axis}_{result.best_slice_idx}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, slice_rgb)
        saved_files.append(filepath)
        print(f"💾 已保存最佳切片 ({result.best_axis}): {filepath}")

        return saved_files

    def save_all_axis_slices(self, result: SearchResult, output_dir: str):
        """
        保存所有三个轴的最佳切片

        Args:
            result: 搜索结果
            output_dir: 输出目录

        Returns:
            保存的文件路径字典
        """
        if not result.has_tumor:
            print("⚠️  未检测到肿瘤，跳过保存切片")
            return {}

        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存轴向切片 (axial: k轴)
        if result.best_axis == 'axial' and result.best_slice_idx is not None:
            k_idx = result.best_slice_idx
        else:
            # 如果没有轴向信息，使用中间切片
            k_idx = self.K // 2

        slice_2d = self.data[:, :, k_idx]
        slice_rgb = np.ascontiguousarray(np.stack([slice_2d] * 3, axis=-1))
        filepath = os.path.join(output_dir, f"axial_slice_{k_idx}_{timestamp}.png")
        cv2.imwrite(filepath, slice_rgb)
        saved_files['axial'] = filepath

        # 保存矢状面切片 (sagittal: i轴)
        if result.best_axis == 'sagittal' and result.best_slice_idx is not None:
            i_idx = result.best_slice_idx
        else:
            i_idx = self.I // 2

        slice_2d = self.data[i_idx, :, :]
        slice_rgb = np.ascontiguousarray(np.stack([slice_2d] * 3, axis=-1))
        filepath = os.path.join(output_dir, f"sagittal_slice_{i_idx}_{timestamp}.png")
        cv2.imwrite(filepath, slice_rgb)
        saved_files['sagittal'] = filepath

        # 保存冠状面切片 (coronal: j轴)
        if result.best_axis == 'coronal' and result.best_slice_idx is not None:
            j_idx = result.best_slice_idx
        else:
            j_idx = self.J // 2

        slice_2d = self.data[:, j_idx, :]
        slice_rgb = np.ascontiguousarray(np.stack([slice_2d] * 3, axis=-1))
        filepath = os.path.join(output_dir, f"coronal_slice_{j_idx}_{timestamp}.png")
        cv2.imwrite(filepath, slice_rgb)
        saved_files['coronal'] = filepath

        print(f"💾 已保存 {len(saved_files)} 个切片到: {output_dir}")
        for axis, path in saved_files.items():
            print(f"   - {axis}: {path}")

        return saved_files

    def search(self) -> SearchResult:
        """
        执行全轴暴力搜索

        Returns:
            SearchResult: 包含搜索结果的データ类
        """
        start_time = time.time()
        total_processed = 0
        all_areas = []

        best_area = -1
        best_slice_info = None

        print("💥 Starting full axial search...")
        for k in range(self.K):
            area, _ = self.run_inference_on_slice(
                self.extract_slice('axial', k),
                self.conf
            )
            total_processed += 1
            all_areas.append(area)
            if area > best_area:
                best_area = area
                best_slice_info = ('axial', k)

        print("💥 Starting full sagittal search...")
        for i in range(self.I):
            area, _ = self.run_inference_on_slice(
                self.extract_slice('sagittal', i),
                self.high_conf_for_other_axes
            )
            total_processed += 1
            all_areas.append(area)
            if area > best_area:
                best_area = area
                best_slice_info = ('sagittal', i)

        print("💥 Starting full coronal search...")
        for j in range(self.J):
            area, _ = self.run_inference_on_slice(
                self.extract_slice('coronal', j),
                self.high_conf_for_other_axes
            )
            total_processed += 1
            all_areas.append(area)
            if area > best_area:
                best_area = area
                best_slice_info = ('coronal', j)

        processing_time = time.time() - start_time

        has_tumor = best_area > self.tumor_area_threshold

        statistics = {
            'all_areas': all_areas,
            'max_area_found': best_area,
            'area_threshold': self.tumor_area_threshold
        }

        result = SearchResult(
            success=True,
            has_tumor=has_tumor,
            best_axis=best_slice_info[0] if best_slice_info else None,
            best_slice_idx=best_slice_info[1] if best_slice_info else None,
            max_area=best_area if best_area > 0 else 0,
            processing_time=processing_time,
            slices_processed=total_processed,
            slices_skipped=0,
            message="Full search completed" + (" - No tumor detected" if not has_tumor else ""),
            statistics=statistics
        )

        # 保存所有三个轴的切片
        if has_tumor:
            saved_files = self.save_all_axis_slices(result, self.output_project)
            result.message += f" | Slices saved: {list(saved_files.keys())}"

        return result


class AdaptiveThresholds:
    """自适应阈值计算器 - 根据3D数据动态计算最优阈值"""

    def __init__(self, data_3d):
        """
        初始化自适应阈值计算器

        Args:
            data_3d: 3D图像数据 (numpy array)
        """
        flat = data_3d.flatten()
        self.global_mean = flat.mean()
        self.global_std = flat.std()
        self.global_nz_ratio = np.count_nonzero(flat) / flat.size

    def slice_nonzero_thresh(self, quantile=0.1):
        """
        计算切片非零比例阈值

        Args:
            quantile: 分位数因子，默认0.1

        Returns:
            自适应的非零比例阈值
        """
        return self.global_nz_ratio * quantile

    def slice_std_thresh(self, factor=0.5):
        """
        计算切片标准差阈值

        Args:
            factor: 标准差因子，默认0.5

        Returns:
            自适应的标准差阈值
        """
        return self.global_std * factor


class TumorSliceFinder:
    """
    方法2: 智能分层搜索检测器

    特点:
    - 先沿轴向全局搜索
    - 基于检测结果在其他两轴进行局部搜索
    - 使用粗搜索+精细搜索策略
    - 计算效率高,适合快速定位
    """

    def __init__(self, model, nii_path: str, output_project: str,
                 conf: float = 0.65, voxel_spacing: Optional[Tuple[float, float, float]] = None,
                 axial_sampling_step: int = 2,
                 sagittal_sampling_step: int = 2,
                 coronal_sampling_step: int = 2,
                 slice_threshold: Optional[float] = None,
                 std_threshold: Optional[float] = None,
                 refine_radius: int = 1,
                 other_axes_conf: float = 0.92,
                 use_adaptive_thresholds: bool = True):
        """
        初始化肿瘤切片查找器

        Args:
            model: 已加载的YOLO模型实例
            nii_path: NIfTI文件路径
            output_project: 输出目录
            conf: 置信度阈值
            voxel_spacing: 体素间距(mm),如果为None则从文件读取
            axial_sampling_step: 轴向采样步长
            sagittal_sampling_step: 矢状面采样步长
            coronal_sampling_step: 冠状面采样步长
            slice_threshold: 切片预过滤阈值(非零像素比例)，如果为None且use_adaptive_thresholds=True则自动计算
            std_threshold: 切片标准差阈值，如果为None且use_adaptive_thresholds=True则自动计算
            refine_radius: 精细搜索半径
            other_axes_conf: 其他轴的置信度阈值
            use_adaptive_thresholds: 是否使用自适应阈值（推荐True）
        """
        self.model = model
        self.nii_path = nii_path
        self.output_project = output_project
        self.conf = conf
        self.voxel_spacing = voxel_spacing
        self.use_adaptive_thresholds = use_adaptive_thresholds

        self.AXIAL_SAMPLING_STEP = axial_sampling_step
        self.SAGITTAL_SAMPLING_STEP = sagittal_sampling_step
        self.CORONAL_SAMPLING_STEP = coronal_sampling_step
        self.REFINE_RADIUS = refine_radius
        self.OTHER_AXES_CONF = other_axes_conf

        self.data = None
        self.I, self.J, self.K = 0, 0, 0
        self.spacing = None
        self.adaptive_thresholds = None

        self._load_data()

        # 初始化自适应阈值
        if self.use_adaptive_thresholds:
            self.adaptive_thresholds = AdaptiveThresholds(self.data)
            self.SLICE_THRESHOLD = slice_threshold if slice_threshold is not None else self.adaptive_thresholds.slice_nonzero_thresh()
            self.STD_THRESHOLD = std_threshold if std_threshold is not None else self.adaptive_thresholds.slice_std_thresh()
            print(f"📊 自适应阈值: SLICE_THRESHOLD={self.SLICE_THRESHOLD:.4f}, STD_THRESHOLD={self.STD_THRESHOLD:.4f}")
        else:
            # 使用固定阈值
            self.SLICE_THRESHOLD = slice_threshold if slice_threshold is not None else 0.05
            self.STD_THRESHOLD = std_threshold if std_threshold is not None else 5.0
            print(f"📊 固定阈值: SLICE_THRESHOLD={self.SLICE_THRESHOLD}, STD_THRESHOLD={self.STD_THRESHOLD}")

        os.makedirs(self.output_project, exist_ok=True)

        print(f"📏 Voxel spacing (mm): {self.spacing}")

    def _load_data(self):
        """加载NIfTI数据"""
        print("🔄 Loading NIfTI data...")
        img = nib.load(self.nii_path)
        self.data = img.get_fdata()

        if self.voxel_spacing is None:
            self.spacing = img.header.get_zooms()
        else:
            self.spacing = self.voxel_spacing

        self.I, self.J, self.K = self.data.shape

        if self.data.dtype != np.uint8:
            self.data = np.clip(self.data, 0, np.percentile(self.data, 99))
            self.data = ((self.data - self.data.min()) /
                         (self.data.max() - self.data.min()) * 255).astype(np.uint8)

        print(f"✅ Data loaded: shape={self.data.shape}")

    def get_units(self) -> Dict[str, dict]:
        """获取各轴的单位信息"""
        units = {
            'axial': {
                'pixel_unit': 'pixel',
                'physical_unit': 'mm',
                'conversion_factor': float(self.spacing[2]),
                'description': f'轴向切片索引 -> 物理位置: 索引 × {self.spacing[2]:.3f} mm'
            },
            'sagittal': {
                'pixel_unit': 'pixel',
                'physical_unit': 'mm',
                'conversion_factor': float(self.spacing[0]),
                'description': f'矢状面切片索引 -> 物理位置: 索引 × {self.spacing[0]:.3f} mm'
            },
            'coronal': {
                'pixel_unit': 'pixel',
                'physical_unit': 'mm',
                'conversion_factor': float(self.spacing[1]),
                'description': f'冠状面切片索引 -> 物理位置: 索引 × {self.spacing[1]:.3f} mm'
            }
        }
        return units

    def save_slices(self, result: SearchResult, output_dir: str) -> Dict[str, str]:
        """
        保存所有最佳切片图像

        Args:
            result: 搜索结果
            output_dir: 输出目录

        Returns:
            保存的文件路径字典
        """
        if not result.has_tumor:
            print("⚠️  未检测到肿瘤，跳过保存切片")
            return {}

        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存轴向切片
        if result.axial_slice is not None:
            slice_2d = self.data[:, :, result.axial_slice]
            slice_rgb = np.ascontiguousarray(np.stack([slice_2d] * 3, axis=-1))
            filepath = os.path.join(output_dir, f"axial_slice_{result.axial_slice}_{timestamp}.png")
            cv2.imwrite(filepath, slice_rgb)
            saved_files['axial'] = filepath

        # 保存矢状面切片
        if result.sagittal_slice is not None:
            slice_2d = self.data[result.sagittal_slice, :, :]
            slice_rgb = np.ascontiguousarray(np.stack([slice_2d] * 3, axis=-1))
            filepath = os.path.join(output_dir, f"sagittal_slice_{result.sagittal_slice}_{timestamp}.png")
            cv2.imwrite(filepath, slice_rgb)
            saved_files['sagittal'] = filepath

        # 保存冠状面切片
        if result.coronal_slice is not None:
            slice_2d = self.data[:, result.coronal_slice, :]
            slice_rgb = np.ascontiguousarray(np.stack([slice_2d] * 3, axis=-1))
            filepath = os.path.join(output_dir, f"coronal_slice_{result.coronal_slice}_{timestamp}.png")
            cv2.imwrite(filepath, slice_rgb)
            saved_files['coronal'] = filepath

        print(f"💾 已保存 {len(saved_files)} 个切片到: {output_dir}")
        for axis, path in saved_files.items():
            print(f"   - {axis}: {path}")

        return saved_files

    @staticmethod
    def pre_filter_slice(slice_2d, non_zero_thresh=0.05, std_thresh=5.0):
        """预筛选切片:检查切片是否值得推理"""
        non_zero_ratio = np.count_nonzero(slice_2d) / slice_2d.size
        slice_std = np.std(slice_2d)

        if non_zero_ratio < non_zero_thresh:
            return False
        if slice_std < std_thresh:
            return False

        return True

    def search_with_refinement(self, data_axis, axis_type, slice_indices, sampling_step,
                               search_range_name="", conf_threshold=None):
        """
        在指定轴上执行粗搜索和精细搜索

        Args:
            data_axis: 切片提取函数
            axis_type: 轴类型名称
            slice_indices: 搜索范围(start, end)
            sampling_step: 采样步长
            search_range_name: 搜索范围描述
            conf_threshold: 置信度阈值

        Returns:
            best_index: 最佳切片索引
            max_area: 最大检测面积
            processed_slices: 处理的切片数
            skipped_slices: 跳过的切片数
        """
        if conf_threshold is None:
            conf_threshold = self.conf

        start_idx, end_idx = slice_indices
        print(f"🔍 {axis_type}搜索 ({search_range_name}): 范围 [{start_idx}, {end_idx})")

        max_area_coarse = -1
        best_idx_coarse = -1
        best_box_coarse = None
        skipped_slices = 0
        processed_coarse = 0

        for idx in range(start_idx, end_idx, sampling_step):
            slice_2d = data_axis(idx)

            if not self.pre_filter_slice(slice_2d, self.SLICE_THRESHOLD, self.STD_THRESHOLD):
                skipped_slices += 1
                continue

            processed_coarse += 1
            slice_rgb = np.stack([slice_2d] * 3, axis=-1)
            results = self.model.predict(
                source=slice_rgb,
                conf=conf_threshold,
                verbose=False,
                device=None,
                save=False
            )

            current_max_area = 0
            best_box = None
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xywh.cpu().numpy()
                    areas = boxes[:, 2] * boxes[:, 3]
                    max_idx = areas.argmax()
                    if areas[max_idx] > current_max_area:
                        current_max_area = areas[max_idx]
                        best_box = boxes[max_idx]

            if current_max_area > max_area_coarse:
                max_area_coarse = current_max_area
                best_idx_coarse = idx
                best_box_coarse = best_box

        print(f"✅ {axis_type}粗搜索: 最佳切片 (idx={best_idx_coarse}), 最大面积: {max_area_coarse:.2f}")
        print(f"⏭️  跳过 {skipped_slices}/{len(range(start_idx, end_idx, sampling_step))} 切片")

        if max_area_coarse <= 0:
            return -1, 0, processed_coarse, skipped_slices

        print(f"🔍 {axis_type}精细搜索...")
        max_area_fine = -1
        best_idx_fine = -1
        best_box_fine = None

        fine_search_range = []
        for offset in range(-self.REFINE_RADIUS, self.REFINE_RADIUS + 1):
            fine_idx = best_idx_coarse + offset
            if start_idx <= fine_idx < end_idx:
                fine_search_range.append(fine_idx)

        if best_idx_coarse not in fine_search_range:
            fine_search_range.append(best_idx_coarse)

        fine_search_range = sorted(set(fine_search_range))
        print(f"🔍 精细搜索范围: {fine_search_range}")

        for fine_idx in fine_search_range:
            slice_2d = data_axis(fine_idx)
            slice_rgb = np.stack([slice_2d] * 3, axis=-1)
            results = self.model.predict(
                source=slice_rgb,
                conf=conf_threshold,
                verbose=False,
                device=None,
                save=False
            )

            current_max_area = 0
            best_box = None
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xywh.cpu().numpy()
                    areas = boxes[:, 2] * boxes[:, 3]
                    max_idx = areas.argmax()
                    if areas[max_idx] > current_max_area:
                        current_max_area = areas[max_idx]
                        best_box = boxes[max_idx]

            if current_max_area > max_area_fine:
                max_area_fine = current_max_area
                best_idx_fine = fine_idx
                best_box_fine = best_box

        print(f"✅ {axis_type}精细搜索: 最佳切片 (idx={best_idx_fine}), 最大面积: {max_area_fine:.2f}")

        if best_box_fine is None:
            if best_box_coarse is None:
                return -1, 0, processed_coarse + len(fine_search_range), skipped_slices
            else:
                print(f"⚠️  {axis_type}精细搜索未找到目标,使用粗搜索结果")
                best_idx_fine = best_idx_coarse
                best_box_fine = best_box_coarse
                max_area_fine = max_area_coarse

        return best_idx_fine, max_area_fine, processed_coarse + len(fine_search_range), skipped_slices

    def _build_3d_tumor_mask(self, axis_type="axial", top_k=5, conf=None):
        """
        从 axial 多 slice 检测结果构建 3D 肿瘤掩膜
        返回:
            mask (I,J,K)
            axial_processed (int)
            axial_skipped (int)
            axial_refine_processed (int)
        """
        if conf is None:
            conf = self.conf

        print(f"🧠 Building 3D tumor mask from top-{top_k} axial slices...")

        candidates = []
        axial_processed = 0
        axial_skipped = 0

        for k in range(0, self.K, self.AXIAL_SAMPLING_STEP):
            slice_2d = self.data[:, :, k]
            if not self.pre_filter_slice(slice_2d, self.SLICE_THRESHOLD, self.STD_THRESHOLD):
                axial_skipped += 1
                continue

            axial_processed += 1
            slice_rgb = np.stack([slice_2d] * 3, axis=-1)
            results = self.model.predict(
                source=slice_rgb,
                conf=conf,
                verbose=False,
                save=False
            )

            max_area = 0
            best_box = None
            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xywh.cpu().numpy()
                    areas = boxes[:, 2] * boxes[:, 3]
                    idx = areas.argmax()
                    if areas[idx] > max_area:
                        max_area = areas[idx]
                        best_box = boxes[idx]

            if best_box is not None:
                candidates.append((k, max_area))

        if len(candidates) == 0:
            return None, axial_processed, axial_skipped, 0

        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:top_k]
        selected_k = [k for k, _ in candidates]
        print(f"✅ Selected axial slices: {selected_k}")

        mask = np.zeros_like(self.data, dtype=np.uint8)
        axial_refine_processed = 0

        for k in selected_k:
            axial_refine_processed += 1
            slice_2d = self.data[:, :, k]
            slice_rgb = np.stack([slice_2d] * 3, axis=-1)
            results = self.model.predict(
                source=slice_rgb,
                conf=conf,
                verbose=False,
                save=False
            )

            for r in results:
                if r.masks is not None:
                    m = r.masks.data.cpu().numpy().astype(np.uint8)
                    if m.ndim == 3:
                        m = m[0]
                    mask[:, :, k] = np.logical_or(mask[:, :, k], m > 0)
                elif r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    for x1, y1, x2, y2 in boxes:
                        mask[
                        int(y1):int(y2),
                        int(x1):int(x2),
                        k
                        ] = 1

        return mask, axial_processed, axial_skipped, axial_refine_processed

    def search(self) -> SearchResult:
        start_time = time.time()

        print("🔍 Step 1: Axial multi-slice detection + 3D CC")

        tumor_mask, axial_processed, axial_skipped, axial_refine_processed = (
            self._build_3d_tumor_mask(axis_type="axial", top_k=5)
        )

        if tumor_mask is None or tumor_mask.sum() == 0:
            processing_time = time.time() - start_time
            print("❌ No tumor mask found, returning middle slices")

            return SearchResult(
                success=True,
                has_tumor=False,
                axial_slice=self.K // 2,
                sagittal_slice=self.I // 2,
                coronal_slice=self.J // 2,
                tumor_center=None,
                units=self.get_units(),
                processing_time=processing_time,
                slices_processed=axial_processed,
                slices_skipped=axial_skipped,
                message="No tumor detected",
                statistics=None
            )

        # Connected components
        labeled, num_cc = ndimage.label(tumor_mask)
        sizes = ndimage.sum_labels(tumor_mask, labeled, range(1, num_cc + 1))
        largest_cc = np.argmax(sizes) + 1
        cc_mask = (labeled == largest_cc).astype(np.uint8)

        coords = np.argwhere(cc_mask)
        i_min, j_min, k_min = coords.min(axis=0)
        i_max, j_max, k_max = coords.max(axis=0) + 1

        print(f"🎯 Largest CC bbox: "
              f"i=[{i_min},{i_max}), "
              f"j=[{j_min},{j_max}), "
              f"k=[{k_min},{k_max})")

        # Local search margin (physical space)
        margin_mm = 3.0
        margin_i = int(np.ceil(margin_mm / self.spacing[0]))
        margin_j = int(np.ceil(margin_mm / self.spacing[1]))

        i_start = max(0, i_min - margin_i)
        i_end = min(self.I, i_max + margin_i)
        j_start = max(0, j_min - margin_j)
        j_end = min(self.J, j_max + margin_j)

        print("\n" + "=" * 50)
        print("Step 2: Sagittal local search (CC-based)")

        def get_sagittal_slice(i):
            return self.data[i, :, :]

        best_i_fine, max_area_i, sagittal_processed, sagittal_skipped = (
            self.search_with_refinement(
                data_axis=get_sagittal_slice,
                axis_type="矢状面",
                slice_indices=(i_start, i_end),
                sampling_step=self.SAGITTAL_SAMPLING_STEP,
                search_range_name=f"CC[{i_start},{i_end})",
                conf_threshold=self.OTHER_AXES_CONF
            )
        )

        print("\n" + "=" * 50)
        print("Step 3: Coronal local search (CC-based)")

        def get_coronal_slice(j):
            return self.data[:, j, :]

        best_j_fine, max_area_j, coronal_processed, coronal_skipped = (
            self.search_with_refinement(
                data_axis=get_coronal_slice,
                axis_type="冠状面",
                slice_indices=(j_start, j_end),
                sampling_step=self.CORONAL_SAMPLING_STEP,
                search_range_name=f"CC[{j_start},{j_end})",
                conf_threshold=self.OTHER_AXES_CONF
            )
        )

        processing_time = time.time() - start_time

        total_processed = (
                axial_processed +
                axial_refine_processed +
                sagittal_processed +
                coronal_processed
        )

        total_skipped = (
                axial_skipped +
                sagittal_skipped +
                coronal_skipped
        )

        result = SearchResult(
            success=True,
            has_tumor=True,
            axial_slice=(k_min + k_max) // 2,
            sagittal_slice=best_i_fine if best_i_fine != -1 else (i_min + i_max) // 2,
            coronal_slice=best_j_fine if best_j_fine != -1 else (j_min + j_max) // 2,
            tumor_center=(
                (i_min + i_max) // 2,
                (j_min + j_max) // 2,
                (k_min + k_max) // 2
            ),
            detection_areas={
                "axial": None,
                "sagittal": max_area_i if best_i_fine != -1 else None,
                "coronal": max_area_j if best_j_fine != -1 else None
            },
            units=self.get_units(),
            processing_time=processing_time,
            slices_processed=total_processed,
            slices_skipped=total_skipped,
            message="CC-based hierarchical search completed",
            statistics={
                "cc_bbox": {
                    "i": (i_min, i_max),
                    "j": (j_min, j_max),
                    "k": (k_min, k_max)
                }
            }
        )

        saved_files = self.save_slices(result, self.output_project)
        result.message += f" | Saved: {list(saved_files.keys())}"

        return result


def convert_to_serializable(obj):
    """
    将numpy类型转换为Python原生类型，以便JSON序列化

    Args:
        obj: 需要转换的对象

    Returns:
        转换后的对象
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def save_comparison_report(results: Dict, output_dir: str, nii_filename: str = ""):
    """
    保存对比结果报告

    Args:
        results: 包含两种方法结果的字典
        output_dir: 报告输出目录
        nii_filename: NIfTI文件名（用于标识）
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 准备报告数据
    report = {
        'comparison_timestamp': timestamp,
        'nii_file': nii_filename,
        'methods': {}
    }

    # 处理方法1的结果
    if results.get('full_search'):
        r1 = results['full_search']
        report['methods']['full_search'] = {
            'name': '全轴暴力搜索',
            'success': r1.success,
            'has_tumor': r1.has_tumor,
            'best_axis': r1.best_axis,
            'best_slice_idx': r1.best_slice_idx,
            'max_area': r1.max_area,
            'processing_time_seconds': round(r1.processing_time, 3),
            'slices_processed': r1.slices_processed,
            'slices_skipped': r1.slices_skipped,
            'message': r1.message
        }

    # 处理方法2的结果
    if results.get('hierarchical_search'):
        r2 = results['hierarchical_search']
        report['methods']['hierarchical_search'] = {
            'name': '智能分层搜索',
            'success': r2.success,
            'has_tumor': r2.has_tumor,
            'axial_slice': r2.axial_slice,
            'sagittal_slice': r2.sagittal_slice,
            'coronal_slice': r2.coronal_slice,
            'tumor_center': list(r2.tumor_center) if r2.tumor_center else None,
            'detection_areas': r2.detection_areas,
            'processing_time_seconds': round(r2.processing_time, 3),
            'slices_processed': r2.slices_processed,
            'slices_skipped': r2.slices_skipped,
            'message': r2.message,
            'units': r2.units
        }

    # 计算性能对比
    if results.get('full_search') and results.get('hierarchical_search'):
        r1 = results['full_search']
        r2 = results['hierarchical_search']

        report['performance_comparison'] = {
            'time_full_search': round(r1.processing_time, 3),
            'time_hierarchical_search': round(r2.processing_time, 3),
            'speedup_ratio': round(r1.processing_time / r2.processing_time, 2) if r2.processing_time > 0 else None,
            'slices_full_search': r1.slices_processed,
            'slices_hierarchical_search': r2.slices_processed,
            'efficiency_improvement': round((1 - r2.slices_processed / r1.slices_processed) * 100,
                                            2) if r1.slices_processed > 0 else None
        }

    # 保存JSON报告
    json_path = os.path.join(output_dir, f"comparison_report_{timestamp}.json")
    # 转换所有numpy类型为Python原生类型
    serializable_report = convert_to_serializable(report)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_report, f, ensure_ascii=False, indent=2)
    print(f"\n📄 JSON报告已保存: {json_path}")

    # 保存文本报告
    txt_path = os.path.join(output_dir, f"comparison_report_{timestamp}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("肿瘤检测方法对比报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"NIfTI文件: {nii_filename}\n\n")

        # 方法1详情
        if results.get('full_search'):
            r1 = results['full_search']
            f.write("-" * 80 + "\n")
            f.write("方法1: 全轴暴力搜索\n")
            f.write("-" * 80 + "\n")
            f.write(f"  检测到肿瘤: {'是' if r1.has_tumor else '否'}\n")
            if r1.has_tumor:
                f.write(f"  最佳轴: {r1.best_axis}\n")
                f.write(f"  最佳切片索引: {r1.best_slice_idx}\n")
                f.write(f"  最大检测面积: {r1.max_area:.2f}\n")
            f.write(f"  处理时间: {r1.processing_time:.3f} 秒\n")
            f.write(f"  处理切片数: {r1.slices_processed}\n")
            f.write(f"  跳过切片数: {r1.slices_skipped}\n")
            f.write(f"  说明: {r1.message}\n\n")

        # 方法2详情
        if results.get('hierarchical_search'):
            r2 = results['hierarchical_search']
            f.write("-" * 80 + "\n")
            f.write("方法2: 智能分层搜索\n")
            f.write("-" * 80 + "\n")
            f.write(f"  检测到肿瘤: {'是' if r2.has_tumor else '否'}\n")
            if r2.has_tumor:
                f.write(f"  轴向切片: {r2.axial_slice}\n")
                f.write(f"  矢状面切片: {r2.sagittal_slice}\n")
                f.write(f"  冠状面切片: {r2.coronal_slice}\n")
                f.write(f"  肿瘤中心: {r2.tumor_center}\n")
                if r2.detection_areas:
                    f.write(f"  检测面积:\n")
                    for axis, area in r2.detection_areas.items():
                        if area is not None:
                            f.write(f"    - {axis}: {area:.2f}\n")
            f.write(f"  处理时间: {r2.processing_time:.3f} 秒\n")
            f.write(f"  处理切片数: {r2.slices_processed}\n")
            f.write(f"  跳过切片数: {r2.slices_skipped}\n")
            f.write(f"  说明: {r2.message}\n\n")

        # 性能对比
        if results.get('full_search') and results.get('hierarchical_search'):
            r1 = results['full_search']
            r2 = results['hierarchical_search']

            f.write("=" * 80 + "\n")
            f.write("性能对比\n")
            f.write("=" * 80 + "\n")
            f.write(f"  方法1处理时间: {r1.processing_time:.3f} 秒\n")
            f.write(f"  方法2处理时间: {r2.processing_time:.3f} 秒\n")
            if r2.processing_time > 0:
                speedup = r1.processing_time / r2.processing_time
                f.write(f"  加速比: {speedup:.2f}x\n")

            f.write(f"\n  方法1处理切片数: {r1.slices_processed}\n")
            f.write(f"  方法2处理切片数: {r2.slices_processed}\n")
            if r1.slices_processed > 0:
                improvement = (1 - r2.slices_processed / r1.slices_processed) * 100
                f.write(f"  效率提升: {improvement:.2f}%\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("总结\n")
            f.write("=" * 80 + "\n")
            f.write(f"  两种方法检测结果一致: {r1.has_tumor == r2.has_tumor}\n")
            if r1.has_tumor and r2.has_tumor:
                f.write(f"  推荐使用: 方法2 (智能分层搜索) - 速度更快，效率更高\n")
            f.write("\n")

    print(f"📄 文本报告已保存: {txt_path}")
    return json_path, txt_path


def load_yolo_model(model_path: str) -> YOLO:
    """
    加载YOLO模型（只加载一次，可复用）

    Args:
        model_path: YOLO模型路径

    Returns:
        加载好的YOLO模型实例
    """
    print("🔄 正在加载YOLO模型...")
    try:
        model = YOLO(model_path)
        print("✅ 模型加载成功")
        return model
    except Exception as e:
        raise RuntimeError(f"模型加载失败: {e}")


def compare_methods(model, nii_path: str, output_project: str, conf: float = 0.65,
                    slice_threshold: Optional[float] = None, std_threshold: Optional[float] = None,
                    use_adaptive_thresholds: bool = True):
    """
    比较两种方法的性能和结果

    Args:
        model: 已加载的YOLO模型实例
        nii_path: NIfTI文件路径
        output_project: 输出目录
        conf: 置信度阈值
        slice_threshold: 切片预过滤阈值(非零像素比例)，如果为None且use_adaptive_thresholds=True则自动计算
        std_threshold: 切片标准差阈值，如果为None且use_adaptive_thresholds=True则自动计算
        use_adaptive_thresholds: 是否使用自适应阈值（推荐True）

    Returns:
        包含两种方法结果的字典
    """
    print("=" * 70)
    print("开始比较两种检测方法")
    print("=" * 70)

    results = {}
    nii_filename = os.path.basename(nii_path)

    print("\n" + "=" * 70)
    print("方法1: 全轴暴力搜索")
    print("=" * 70)
    try:
        detector1 = FullSearchDetector(
            model=model,
            nii_path=nii_path,
            output_project=os.path.join(output_project, "full_search"),
            conf=conf,
            slice_threshold=slice_threshold,
            std_threshold=std_threshold,
            use_adaptive_thresholds=use_adaptive_thresholds
        )
        result1 = detector1.search()
        results['full_search'] = result1

        print(f"\n✅ 方法1完成:")
        print(f"   - 检测到肿瘤: {result1.has_tumor}")
        print(f"   - 最佳轴: {result1.best_axis}")
        print(f"   - 最佳切片: {result1.best_slice_idx}")
        print(f"   - 最大面积: {result1.max_area:.2f}")
        print(f"   - 处理时间: {result1.processing_time:.2f}s")
        print(f"   - 处理切片数: {result1.slices_processed}")
    except Exception as e:
        print(f"❌ 方法1失败: {e}")
        import traceback
        traceback.print_exc()
        results['full_search'] = None

    print("\n" + "=" * 70)
    print("方法2: 智能分层搜索")
    print("=" * 70)
    try:
        detector2 = TumorSliceFinder(
            model=model,
            nii_path=nii_path,
            output_project=os.path.join(output_project, "hierarchical_search"),
            conf=conf,
            slice_threshold=slice_threshold,
            std_threshold=std_threshold,
            use_adaptive_thresholds=use_adaptive_thresholds
        )
        result2 = detector2.search()
        results['hierarchical_search'] = result2

        print(f"\n✅ 方法2完成:")
        print(f"   - 检测到肿瘤: {result2.has_tumor}")
        print(f"   - 轴向切片: {result2.axial_slice}")
        print(f"   - 矢状面切片: {result2.sagittal_slice}")
        print(f"   - 冠状面切片: {result2.coronal_slice}")
        print(f"   - 肿瘤中心: {result2.tumor_center}")
        print(f"   - 处理时间: {result2.processing_time:.2f}s")
        print(f"   - 处理切片数: {result2.slices_processed}")
        print(f"   - 跳过切片数: {result2.slices_skipped}")
    except Exception as e:
        print(f"❌ 方法2失败: {e}")
        import traceback
        traceback.print_exc()
        results['hierarchical_search'] = None

    # 保存对比报告
    print("\n" + "=" * 70)
    print("生成对比报告")
    print("=" * 70)
    report_dir = os.path.join(output_project, "comparison_reports")
    json_path, txt_path = save_comparison_report(results, report_dir, nii_filename)

    print("\n" + "=" * 70)
    print("比较总结")
    print("=" * 70)
    if results['full_search'] and results['hierarchical_search']:
        r1 = results['full_search']
        r2 = results['hierarchical_search']

        print(f"方法1处理时间: {r1.processing_time:.2f}s, 处理切片: {r1.slices_processed}")
        print(f"方法2处理时间: {r2.processing_time:.2f}s, 处理切片: {r2.slices_processed}")

        if r1.processing_time > 0:
            speedup = r1.processing_time / r2.processing_time
            print(f"方法2加速比: {speedup:.2f}x")

        print(f"\n📊 详细报告已保存到:")
        print(f"   JSON: {json_path}")
        print(f"   TXT:  {txt_path}")

    return results


def batch_compare_methods(model_path: str, nii_dir: str, output_base_dir: str, conf: float = 0.65,
                          slice_threshold: Optional[float] = None, std_threshold: Optional[float] = None,
                          use_adaptive_thresholds: bool = True):
    """
    批量比较两种方法

    Args:
        model_path: 模型路径
        nii_dir: 包含nii.gz文件的目录
        output_base_dir: 输出根目录
        conf: 置信度阈值
        slice_threshold: 切片预过滤阈值(非零像素比例)，如果为None且use_adaptive_thresholds=True则自动计算
        std_threshold: 切片标准差阈值，如果为None且use_adaptive_thresholds=True则自动计算
        use_adaptive_thresholds: 是否使用自适应阈值（推荐True）

    Returns:
        所有文件的对比结果列表
    """
    # 一次性加载模型
    model = load_yolo_model(model_path)

    # 获取所有nii.gz文件
    nii_files = [f for f in os.listdir(nii_dir) if f.endswith('.nii.gz')]

    if not nii_files:
        print(f"❌ 在目录 {nii_dir} 中未找到nii.gz文件")
        return []

    print(f"📁 找到 {len(nii_files)} 个nii.gz文件")
    print("=" * 80)

    all_results = []

    for idx, nii_file in enumerate(nii_files, 1):
        nii_path = os.path.join(nii_dir, nii_file)
        # 从文件名提取标识符（去掉.nii.gz后缀）
        file_id = nii_file.replace('.nii.gz', '')
        output_project = os.path.join(output_base_dir, file_id)

        print(f"\n{'=' * 80}")
        print(f"处理文件 [{idx}/{len(nii_files)}]: {nii_file}")
        print(f"{'=' * 80}")

        try:
            results = compare_methods(
                model=model,
                nii_path=nii_path,
                output_project=output_project,
                conf=conf,
                slice_threshold=slice_threshold,
                std_threshold=std_threshold,
                use_adaptive_thresholds=use_adaptive_thresholds
            )

            # 收集统计信息
            summary = {
                'file_name': nii_file,
                'file_id': file_id,
                'full_search': None,
                'hierarchical_search': None
            }

            if results.get('full_search'):
                r1 = results['full_search']
                summary['full_search'] = {
                    'has_tumor': r1.has_tumor,
                    'best_axis': r1.best_axis,
                    'best_slice_idx': r1.best_slice_idx,
                    'max_area': float(r1.max_area),
                    'processing_time': round(r1.processing_time, 3),
                    'slices_processed': r1.slices_processed
                }

            if results.get('hierarchical_search'):
                r2 = results['hierarchical_search']
                summary['hierarchical_search'] = {
                    'has_tumor': r2.has_tumor,
                    'axial_slice': r2.axial_slice,
                    'sagittal_slice': r2.sagittal_slice,
                    'coronal_slice': r2.coronal_slice,
                    'tumor_center': list(r2.tumor_center) if r2.tumor_center else None,
                    'processing_time': round(r2.processing_time, 3),
                    'slices_processed': r2.slices_processed,
                    'slices_skipped': r2.slices_skipped
                }

            # 计算加速比
            if results.get('full_search') and results.get('hierarchical_search'):
                r1 = results['full_search']
                r2 = results['hierarchical_search']
                if r2.processing_time > 0:
                    summary['speedup_ratio'] = round(r1.processing_time / r2.processing_time, 2)
                if r1.slices_processed > 0:
                    summary['efficiency_improvement'] = round(
                        (1 - r2.slices_processed / r1.slices_processed) * 100, 2
                    )

            all_results.append(summary)

        except Exception as e:
            print(f"❌ 处理文件 {nii_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'file_name': nii_file,
                'file_id': file_id,
                'error': str(e)
            })

    # 生成汇总报告
    print("\n" + "=" * 80)
    print("生成批量处理汇总报告")
    print("=" * 80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = os.path.join(output_base_dir, "batch_summary")
    os.makedirs(summary_dir, exist_ok=True)

    # 保存JSON汇总
    json_summary_path = os.path.join(summary_dir, f"batch_summary_{timestamp}.json")
    serializable_summary = convert_to_serializable(all_results)
    with open(json_summary_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_summary, f, ensure_ascii=False, indent=2)
    print(f"📄 JSON汇总报告已保存: {json_summary_path}")

    # 保存TXT汇总
    txt_summary_path = os.path.join(summary_dir, f"batch_summary_{timestamp}.txt")
    with open(txt_summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("肿瘤检测方法批量对比汇总报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"处理文件数: {len(all_results)}\n")
        f.write(f"NIfTI目录: {nii_dir}\n\n")

        # 统计信息
        total_files = len(all_results)
        files_with_tumor_method1 = sum(1 for r in all_results if r.get('full_search', {}).get('has_tumor', False))
        files_with_tumor_method2 = sum(
            1 for r in all_results if r.get('hierarchical_search', {}).get('has_tumor', False))

        f.write("-" * 80 + "\n")
        f.write("总体统计\n")
        f.write("-" * 80 + "\n")
        f.write(f"  总文件数: {total_files}\n")
        f.write(f"  方法1检测到肿瘤的文件数: {files_with_tumor_method1}\n")
        f.write(f"  方法2检测到肿瘤的文件数: {files_with_tumor_method2}\n")

        # 计算平均性能
        valid_results = [r for r in all_results if 'error' not in r]
        if valid_results:
            avg_time_method1 = np.mean(
                [r['full_search']['processing_time'] for r in valid_results if r.get('full_search')])
            avg_time_method2 = np.mean(
                [r['hierarchical_search']['processing_time'] for r in valid_results if r.get('hierarchical_search')])
            avg_slices_method1 = np.mean(
                [r['full_search']['slices_processed'] for r in valid_results if r.get('full_search')])
            avg_slices_method2 = np.mean(
                [r['hierarchical_search']['slices_processed'] for r in valid_results if r.get('hierarchical_search')])
            avg_speedup = np.mean([r['speedup_ratio'] for r in valid_results if 'speedup_ratio' in r])
            avg_efficiency = np.mean(
                [r['efficiency_improvement'] for r in valid_results if 'efficiency_improvement' in r])

            f.write(f"\n  平均处理时间:\n")
            f.write(f"    方法1: {avg_time_method1:.3f} 秒\n")
            f.write(f"    方法2: {avg_time_method2:.3f} 秒\n")
            f.write(f"    平均加速比: {avg_speedup:.2f}x\n")
            f.write(f"\n  平均处理切片数:\n")
            f.write(f"    方法1: {avg_slices_method1:.1f}\n")
            f.write(f"    方法2: {avg_slices_method2:.1f}\n")
            f.write(f"    平均效率提升: {avg_efficiency:.2f}%\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("详细结果\n")
        f.write("=" * 80 + "\n\n")

        for idx, result in enumerate(all_results, 1):
            f.write(f"[{idx}] {result['file_name']}\n")
            f.write("-" * 80 + "\n")

            if 'error' in result:
                f.write(f"  ❌ 错误: {result['error']}\n\n")
                continue

            # 方法1结果
            if result.get('full_search'):
                fs = result['full_search']
                f.write(f"  方法1 (全轴暴力搜索):\n")
                f.write(f"    检测到肿瘤: {'是' if fs['has_tumor'] else '否'}\n")
                if fs['has_tumor']:
                    f.write(f"    最佳轴: {fs['best_axis']}\n")
                    f.write(f"    最佳切片: {fs['best_slice_idx']}\n")
                    f.write(f"    最大面积: {fs['max_area']:.2f}\n")
                f.write(f"    处理时间: {fs['processing_time']:.3f} 秒\n")
                f.write(f"    处理切片数: {fs['slices_processed']}\n")

            f.write("\n")

            # 方法2结果
            if result.get('hierarchical_search'):
                hs = result['hierarchical_search']
                f.write(f"  方法2 (智能分层搜索):\n")
                f.write(f"    检测到肿瘤: {'是' if hs['has_tumor'] else '否'}\n")
                if hs['has_tumor']:
                    f.write(f"    轴向切片: {hs['axial_slice']}\n")
                    f.write(f"    矢状面切片: {hs['sagittal_slice']}\n")
                    f.write(f"    冠状面切片: {hs['coronal_slice']}\n")
                    f.write(f"    肿瘤中心: {hs['tumor_center']}\n")
                f.write(f"    处理时间: {hs['processing_time']:.3f} 秒\n")
                f.write(f"    处理切片数: {hs['slices_processed']}\n")
                f.write(f"    跳过切片数: {hs['slices_skipped']}\n")

            f.write("\n")

            # 性能对比
            if 'speedup_ratio' in result:
                f.write(f"  性能对比:\n")
                f.write(f"    加速比: {result['speedup_ratio']:.2f}x\n")
                if 'efficiency_improvement' in result:
                    f.write(f"    效率提升: {result['efficiency_improvement']:.2f}%\n")

            f.write("\n" + "=" * 80 + "\n\n")

    print(f"📄 TXT汇总报告已保存: {txt_summary_path}")
    print(f"\n✅ 批量处理完成！共处理 {len(all_results)} 个文件")

    return all_results


if __name__ == "__main__":
    model_path = r'H:\pycharm_project\PI-MAPP\project\detection_train\tumor\runs\detect\train_yolo12_try_owndata2\weights\best.pt'
    nii_dir = r"H:\data\tumor_data_swust\data1_output\filtered"
    output_base_dir = r'H:\data\tumor_data_swust\data1_output\filtered_output5'

    # 批量处理所有文件
    all_results = batch_compare_methods(
        model_path=model_path,
        nii_dir=nii_dir,
        output_base_dir=output_base_dir,
        conf=0.65
    )
