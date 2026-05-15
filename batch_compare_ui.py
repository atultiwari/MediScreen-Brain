import sys
import os
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                                QLineEdit, QGroupBox, QTextEdit, QProgressBar,
                                QSpinBox, QDoubleSpinBox, QFormLayout, QMessageBox,
                                QSplitter, QTabWidget)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont


class BatchCompareThread(QThread):
    """批量对比线程 - 增强版，支持实时进度反馈"""
    progress_updated = Signal(int)
    status_changed = Signal(str)
    file_started = Signal(str, int, int)  # 文件名, 当前索引, 总数
    file_completed = Signal(str, dict)  # 文件名, 结果摘要
    batch_finished = Signal(list)  # 所有结果
    error_occurred = Signal(str)
    log_message = Signal(str)  # 日志消息信号
    
    def __init__(self, model_path, nii_dir, output_base_dir, conf=0.65):
        super().__init__()
        self.model_path = model_path
        self.nii_dir = nii_dir
        self.output_base_dir = output_base_dir
        self.conf = conf
        self.is_running = True
        self._original_batch_compare_methods = None
        
    def run(self):
        """执行批量对比，带实时进度更新"""
        try:
            import sys
            import os
            from datetime import datetime
            import numpy as np
            
            # 导入对比模块
            from utils.Compares_two_methods import compare_methods, save_comparison_report, convert_to_serializable
            import json
            
            self.status_changed.emit("开始批量对比分析...")
            self.log_message.emit("=" * 80)
            self.log_message.emit("开始批量对比分析")
            self.log_message.emit(f"模型: {self.model_path}")
            self.log_message.emit(f"NIfTI文件夹: {self.nii_dir}")
            self.log_message.emit(f"输出文件夹: {self.output_base_dir}")
            self.log_message.emit(f"置信度阈值: {self.conf}")
            
            # 获取所有nii.gz文件
            nii_files = [f for f in os.listdir(self.nii_dir) if f.endswith('.nii.gz')]
            total_files = len(nii_files)
            
            if total_files == 0:
                self.error_occurred.emit(f"在目录 {self.nii_dir} 中未找到nii.gz文件")
                return
            
            self.log_message.emit(f"找到 {total_files} 个nii.gz文件")
            self.log_message.emit("=" * 80)
            
            all_results = []
            
            for idx, nii_file in enumerate(nii_files, 1):
                if not self.is_running:
                    self.log_message.emit("用户取消处理")
                    break
                
                nii_path = os.path.join(self.nii_dir, nii_file)
                file_id = nii_file.replace('.nii.gz', '')
                output_project = os.path.join(self.output_base_dir, file_id)
                
                # 发送文件开始信号
                self.file_started.emit(nii_file, idx, total_files)
                self.log_message.emit(f"\n[{idx}/{total_files}] 开始处理: {nii_file}")
                
                try:
                    # 执行单个文件的对比
                    results = compare_methods(
                        model_path=self.model_path,
                        nii_path=nii_path,
                        output_project=output_project,
                        conf=self.conf
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
                    
                    # 发送文件完成信号
                    self.file_completed.emit(nii_file, summary)
                    
                    # 更新进度
                    progress = int((idx / total_files) * 100)
                    self.progress_updated.emit(progress)
                    
                except Exception as e:
                    error_msg = f"处理文件 {nii_file} 时出错: {str(e)}"
                    self.log_message.emit(f"  ❌ {error_msg}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        'file_name': nii_file,
                        'file_id': file_id,
                        'error': str(e)
                    })
                    self.file_completed.emit(nii_file, {'error': str(e)})
            
            # 生成汇总报告
            if self.is_running and all_results:
                self.log_message.emit("\n" + "=" * 80)
                self.log_message.emit("生成批量处理汇总报告")
                self.log_message.emit("=" * 80)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                summary_dir = os.path.join(self.output_base_dir, "batch_summary")
                os.makedirs(summary_dir, exist_ok=True)
                
                # 保存JSON汇总
                json_summary_path = os.path.join(summary_dir, f"batch_summary_{timestamp}.json")
                serializable_summary = convert_to_serializable(all_results)
                with open(json_summary_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_summary, f, ensure_ascii=False, indent=2)
                self.log_message.emit(f"📄 JSON汇总报告已保存: {json_summary_path}")
                
                # 保存TXT汇总
                txt_summary_path = os.path.join(summary_dir, f"batch_summary_{timestamp}.txt")
                with open(txt_summary_path, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("肿瘤检测方法批量对比汇总报告\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"处理文件数: {len(all_results)}\n")
                    f.write(f"NIfTI目录: {self.nii_dir}\n\n")
                    
                    # 统计信息
                    total_files_count = len(all_results)
                    files_with_tumor_method1 = sum(1 for r in all_results if r.get('full_search', {}).get('has_tumor', False))
                    files_with_tumor_method2 = sum(1 for r in all_results if r.get('hierarchical_search', {}).get('has_tumor', False))
                    
                    f.write("-" * 80 + "\n")
                    f.write("总体统计\n")
                    f.write("-" * 80 + "\n")
                    f.write(f"  总文件数: {total_files_count}\n")
                    f.write(f"  方法1检测到肿瘤的文件数: {files_with_tumor_method1}\n")
                    f.write(f"  方法2检测到肿瘤的文件数: {files_with_tumor_method2}\n")
                    
                    # 计算平均性能
                    valid_results = [r for r in all_results if 'error' not in r]
                    if valid_results:
                        avg_time_method1 = np.mean([r['full_search']['processing_time'] for r in valid_results if r.get('full_search')])
                        avg_time_method2 = np.mean([r['hierarchical_search']['processing_time'] for r in valid_results if r.get('hierarchical_search')])
                        avg_slices_method1 = np.mean([r['full_search']['slices_processed'] for r in valid_results if r.get('full_search')])
                        avg_slices_method2 = np.mean([r['hierarchical_search']['slices_processed'] for r in valid_results if r.get('hierarchical_search')])
                        avg_speedup = np.mean([r['speedup_ratio'] for r in valid_results if 'speedup_ratio' in r])
                        avg_efficiency = np.mean([r['efficiency_improvement'] for r in valid_results if 'efficiency_improvement' in r])
                        
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
                
                self.log_message.emit(f"📄 TXT汇总报告已保存: {txt_summary_path}")
                self.log_message.emit(f"\n✅ 批量处理完成！共处理 {len(all_results)} 个文件")
                
                self.batch_finished.emit(all_results)
                self.status_changed.emit(f"批量处理完成！共处理 {len(all_results)} 个文件")
                
        except Exception as e:
            error_msg = f"批量处理错误: {str(e)}"
            self.error_occurred.emit(error_msg)
            self.log_message.emit(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
    
    def stop(self):
        self.is_running = False


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 肿瘤检测方法批量对比工具")
        self.resize(1000, 700)
        
        # 默认路径配置
        self.default_model_path = r'H:\pycharm_project\PI-MAPP\project\detection_train\tumor\runs\detect\train_yolo12_try_owndata2\weights\best.pt'
        self.default_nii_dir = r"H:\data\tumor_data_swust\data1_output\filterred1"
        
        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
            }
            
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                border: 2px solid rgba(52, 152, 219, 0.7);
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 15px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 255, 255, 0.9), stop:1 rgba(245, 245, 245, 0.9));
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
                color: #2c3e50;
                font-size: 14px;
                font-weight: bold;
            }
            
            QPushButton {
                padding: 8px 16px;
                font-size: 13px;
                font-weight: bold;
                border: none;
                border-radius: 8px;
                color: white;
                min-width: 100px;
                min-height: 35px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3498db, stop:1 #2980b9);
            }
            
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5dade2, stop:1 #3498db);
            }
            
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2980b9, stop:1 #1f618d);
            }
            
            QPushButton:disabled {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #bdc3c7, stop:1 #95a5a6);
                color: #7f8c8d;
            }
            
            QPushButton#startBtn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #10B981, stop:1 #059669);
            }
            
            QPushButton#startBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #059669, stop:1 #047857);
            }
            
            QPushButton#stopBtn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #EF4444, stop:1 #DC2626);
            }
            
            QPushButton#stopBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #DC2626, stop:1 #B91C1C);
            }
            
            QLineEdit {
                padding: 8px 12px;
                border: 2px solid rgba(189, 195, 199, 0.5);
                border-radius: 6px;
                background: white;
                font-size: 12px;
                min-height: 30px;
            }
            
            QLineEdit:focus {
                border-color: #3498db;
            }
            
            QSpinBox, QDoubleSpinBox {
                padding: 6px 10px;
                border: 2px solid rgba(189, 195, 199, 0.5);
                border-radius: 6px;
                background: white;
                min-width: 100px;
                min-height: 30px;
                font-size: 12px;
            }
            
            QTextEdit {
                border: 2px solid rgba(189, 195, 199, 0.5);
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.95);
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                padding: 10px;
            }
            
            QProgressBar {
                border: 2px solid rgba(189, 195, 199, 0.5);
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                font-size: 12px;
                max-height: 25px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #ecf0f1, stop:1 #d5dbdb);
            }
            
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2ecc71, stop:1 #27ae60);
                border-radius: 6px;
                margin: 1px;
            }
            
            QLabel {
                font-size: 12px;
                color: #2c3e50;
            }
        """)
        
        # 初始化变量
        self.compare_thread = None
        self.init_ui()
        
    def init_ui(self):
        """初始化UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # ===== 参数配置区域 =====
        config_group = QGroupBox("⚙️ 参数配置")
        config_layout = QFormLayout(config_group)
        config_layout.setSpacing(10)
        
        # 模型路径
        model_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setText(self.default_model_path)
        self.model_path_edit.setPlaceholderText("选择YOLO模型文件 (.pt)")
        model_browse_btn = QPushButton("📂 浏览")
        model_browse_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(model_browse_btn)
        config_layout.addRow("模型路径:", model_layout)
        
        # NIfTI文件夹
        nii_layout = QHBoxLayout()
        self.nii_dir_edit = QLineEdit()
        self.nii_dir_edit.setText(self.default_nii_dir)
        self.nii_dir_edit.setPlaceholderText("选择包含nii.gz文件的文件夹")
        nii_browse_btn = QPushButton("📂 浏览")
        nii_browse_btn.clicked.connect(self.browse_nii_dir)
        nii_layout.addWidget(self.nii_dir_edit)
        nii_layout.addWidget(nii_browse_btn)
        config_layout.addRow("NIfTI文件夹:", nii_layout)
        
        # 输出文件夹（自动设置默认值）
        output_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        # 根据默认NIfTI目录自动生成默认输出目录
        default_output = os.path.join(os.path.dirname(self.default_nii_dir), 
                                     f"{os.path.basename(self.default_nii_dir)}_comparison_results")
        self.output_dir_edit.setText(default_output)
        self.output_dir_edit.setPlaceholderText("选择输出结果文件夹")
        output_browse_btn = QPushButton("📂 浏览")
        output_browse_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(output_browse_btn)
        config_layout.addRow("输出文件夹:", output_layout)
        
        # 置信度阈值
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.01, 1.0)
        self.conf_spinbox.setValue(0.65)
        self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.setDecimals(2)
        config_layout.addRow("置信度阈值:", self.conf_spinbox)
        
        main_layout.addWidget(config_group)
        
        # ===== 控制按钮区域 =====
        control_group = QGroupBox("🎮 控制面板")
        control_layout = QHBoxLayout(control_group)
        control_layout.setSpacing(15)
        
        self.start_btn = QPushButton("▶️ 开始批量对比")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self.start_comparison)
        control_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("⏹️ 停止")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self.stop_comparison)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        control_layout.addStretch()
        
        main_layout.addWidget(control_group)
        
        # ===== 进度显示区域 =====
        progress_group = QGroupBox("📊 处理进度")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #3498db;")
        progress_layout.addWidget(self.status_label)
        
        main_layout.addWidget(progress_group)
        
        # ===== 日志输出区域 =====
        log_group = QGroupBox("📝 处理日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        log_layout.addWidget(self.log_text)
        
        main_layout.addWidget(log_group)
        
    def browse_model(self):
        """浏览选择模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择YOLO模型文件",
            "",
            "模型文件 (*.pt *.onnx);;所有文件 (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            self.log_message(f"已选择模型: {file_path}")
    
    def browse_nii_dir(self):
        """浏览选择NIfTI文件夹"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择包含nii.gz文件的文件夹",
            ""
        )
        if dir_path:
            self.nii_dir_edit.setText(dir_path)
            self.log_message(f"已选择NIfTI文件夹: {dir_path}")
            
            # 自动设置输出文件夹（如果未设置）
            if not self.output_dir_edit.text():
                default_output = os.path.join(os.path.dirname(dir_path), 
                                             f"{os.path.basename(dir_path)}_comparison_results")
                self.output_dir_edit.setText(default_output)
    
    def browse_output_dir(self):
        """浏览选择输出文件夹"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择输出结果文件夹",
            ""
        )
        if dir_path:
            self.output_dir_edit.setText(dir_path)
            self.log_message(f"已选择输出文件夹: {dir_path}")
    
    def log_message(self, message):
        """添加日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def start_comparison(self):
        """开始批量对比"""
        # 验证输入
        model_path = self.model_path_edit.text().strip()
        nii_dir = self.nii_dir_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        conf = self.conf_spinbox.value()
        
        if not model_path:
            QMessageBox.warning(self, "警告", "请选择模型文件！")
            return
        
        if not nii_dir:
            QMessageBox.warning(self, "警告", "请选择NIfTI文件夹！")
            return
        
        if not output_dir:
            QMessageBox.warning(self, "警告", "请选择输出文件夹！")
            return
        
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "错误", f"模型文件不存在: {model_path}")
            return
        
        if not os.path.exists(nii_dir):
            QMessageBox.critical(self, "错误", f"NIfTI文件夹不存在: {nii_dir}")
            return
        
        # 检查nii.gz文件
        nii_files = [f for f in os.listdir(nii_dir) if f.endswith('.nii.gz')]
        if not nii_files:
            QMessageBox.warning(self, "警告", f"在文件夹中未找到nii.gz文件: {nii_dir}")
            return
        
        self.log_message("=" * 80)
        self.log_message(f"开始批量对比分析")
        self.log_message(f"模型: {model_path}")
        self.log_message(f"NIfTI文件夹: {nii_dir}")
        self.log_message(f"输出文件夹: {output_dir}")
        self.log_message(f"置信度阈值: {conf}")
        self.log_message(f"找到 {len(nii_files)} 个nii.gz文件")
        self.log_message("=" * 80)
        
        # 禁用开始按钮，启用停止按钮
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # 创建并启动线程
        self.compare_thread = BatchCompareThread(
            model_path=model_path,
            nii_dir=nii_dir,
            output_base_dir=output_dir,
            conf=conf
        )
        
        # 连接信号
        self.compare_thread.progress_updated.connect(self.on_progress_updated)
        self.compare_thread.status_changed.connect(self.on_status_changed)
        self.compare_thread.file_started.connect(self.on_file_started)
        self.compare_thread.file_completed.connect(self.on_file_completed)
        self.compare_thread.batch_finished.connect(self.on_batch_finished)
        self.compare_thread.error_occurred.connect(self.on_error_occurred)
        self.compare_thread.log_message.connect(self.log_message)  # 连接日志信号
        
        # 启动线程
        self.compare_thread.start()
    
    def stop_comparison(self):
        """停止批量对比"""
        if self.compare_thread and self.compare_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "确认停止",
                "确定要停止批量对比吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.compare_thread.stop()
                self.log_message("用户请求停止处理...")
                self.stop_btn.setEnabled(False)
    
    def on_progress_updated(self, value):
        """进度更新"""
        self.progress_bar.setValue(value)
    
    def on_status_changed(self, status):
        """状态改变"""
        self.status_label.setText(status)
        self.log_message(status)
    
    def on_file_started(self, filename, current, total):
        """文件开始处理"""
        self.log_message(f"\n[{current}/{total}] 开始处理: {filename}")
    
    def on_file_completed(self, filename, summary):
        """文件处理完成"""
        if 'error' in summary:
            self.log_message(f"  ❌ 处理失败: {summary['error']}")
        else:
            method1 = summary.get('full_search', {})
            method2 = summary.get('hierarchical_search', {})
            
            if method1 and method2:
                speedup = summary.get('speedup_ratio', 'N/A')
                efficiency = summary.get('efficiency_improvement', 'N/A')
                
                self.log_message(f"  ✅ 方法1: 时间={method1.get('processing_time', 'N/A')}s, "
                               f"切片数={method1.get('slices_processed', 'N/A')}")
                self.log_message(f"  ✅ 方法2: 时间={method2.get('processing_time', 'N/A')}s, "
                               f"切片数={method2.get('slices_processed', 'N/A')}")
                self.log_message(f"  📊 加速比: {speedup}x, 效率提升: {efficiency}%")
    
    def on_batch_finished(self, results):
        """批量处理完成"""
        self.progress_bar.setValue(100)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        successful = sum(1 for r in results if 'error' not in r)
        failed = len(results) - successful
        
        self.log_message("\n" + "=" * 80)
        self.log_message(f"✅ 批量处理完成！")
        self.log_message(f"   总文件数: {len(results)}")
        self.log_message(f"   成功: {successful}")
        self.log_message(f"   失败: {failed}")
        self.log_message("=" * 80)
        
        QMessageBox.information(
            self,
            "完成",
            f"批量对比完成！\n\n"
            f"总文件数: {len(results)}\n"
            f"成功: {successful}\n"
            f"失败: {failed}\n\n"
            f"详细报告已保存到输出文件夹。"
        )
    
    def on_error_occurred(self, error_msg):
        """错误发生"""
        self.log_message(f"❌ 错误: {error_msg}")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        QMessageBox.critical(self, "错误", error_msg)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.compare_thread and self.compare_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "确认退出",
                "批量处理正在进行中，确定要退出吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.compare_thread.stop()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
