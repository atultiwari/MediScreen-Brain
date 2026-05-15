import json
import os
import numpy as np
from collections import defaultdict
from datetime import datetime


def load_json_data(json_path):
    """加载JSON数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def analyze_batch_results(data):
    """分析批量处理结果"""
    
    # 初始化统计变量
    total_files = len(data)
    tumor_files = 0
    non_tumor_files = 0
    
    full_search_times = []
    hierarchical_search_times = []
    speedup_ratios = []
    efficiency_improvements = []
    
    full_slices_processed = []
    hierarchical_slices_processed = []
    slices_skipped = []
    
    # 用于存储不同轴的最佳切片分布
    best_axis_distribution = defaultdict(int)
    
    for item in data:
        # 统计有无肿瘤的文件数量
        if item['full_search']['has_tumor']:
            tumor_files += 1
        else:
            non_tumor_files += 1
            
        # 收集全面搜索的时间和处理切片数
        full_search_times.append(item['full_search']['processing_time'])
        full_slices_processed.append(item['full_search']['slices_processed'])
        
        # 收集分层搜索的时间和处理切片数
        hierarchical_search_times.append(item['hierarchical_search']['processing_time'])
        hierarchical_slices_processed.append(item['hierarchical_search']['slices_processed'])
        slices_skipped.append(item['hierarchical_search'].get('slices_skipped', 0))
        
        # 收集加速比和效率提升
        speedup_ratios.append(item['speedup_ratio'])
        efficiency_improvements.append(item['efficiency_improvement'])
        
        # 统计最佳轴向分布
        if item['full_search']['has_tumor']:
            best_axis_distribution[item['full_search']['best_axis']] += 1
    
    # 计算统计数据
    stats = {
        'total_files': total_files,
        'tumor_files': tumor_files,
        'non_tumor_files': non_tumor_files,
        
        'full_search_avg_time': np.mean(full_search_times),
        'full_search_std_time': np.std(full_search_times),
        'full_search_min_time': np.min(full_search_times),
        'full_search_max_time': np.max(full_search_times),
        
        'hierarchical_search_avg_time': np.mean(hierarchical_search_times),
        'hierarchical_search_std_time': np.std(hierarchical_search_times),
        'hierarchical_search_min_time': np.min(hierarchical_search_times),
        'hierarchical_search_max_time': np.max(hierarchical_search_times),
        
        'avg_speedup_ratio': np.mean(speedup_ratios),
        'std_speedup_ratio': np.std(speedup_ratios),
        'min_speedup_ratio': np.min(speedup_ratios),
        'max_speedup_ratio': np.max(speedup_ratios),
        
        'avg_efficiency_improvement': np.mean(efficiency_improvements),
        'std_efficiency_improvement': np.std(efficiency_improvements),
        'min_efficiency_improvement': np.min(efficiency_improvements),
        'max_efficiency_improvement': np.max(efficiency_improvements),
        
        'full_search_avg_slices': np.mean(full_slices_processed),
        'hierarchical_search_avg_slices': np.mean(hierarchical_slices_processed),
        'avg_slices_skipped': np.mean(slices_skipped),
        
        'best_axis_distribution': dict(best_axis_distribution)
    }
    
    return stats


def print_statistics(stats):
    """打印统计结果"""
    print("=" * 60)
    print("批量处理结果统计分析")
    print("=" * 60)
    print(f"总文件数: {stats['total_files']}")
    print(f"含肿瘤文件数: {stats['tumor_files']}")
    print(f"不含肿瘤文件数: {stats['non_tumor_files']}")
    print()
    
    print("全面搜索性能:")
    print(f"  平均处理时间: {stats['full_search_avg_time']:.3f} ± {stats['full_search_std_time']:.3f} 秒")
    print(f"  最小处理时间: {stats['full_search_min_time']:.3f} 秒")
    print(f"  最大处理时间: {stats['full_search_max_time']:.3f} 秒")
    print(f"  平均处理切片数: {stats['full_search_avg_slices']:.1f}")
    print()
    
    print("分层搜索性能:")
    print(f"  平均处理时间: {stats['hierarchical_search_avg_time']:.3f} ± {stats['hierarchical_search_std_time']:.3f} 秒")
    print(f"  最小处理时间: {stats['hierarchical_search_min_time']:.3f} 秒")
    print(f"  最大处理时间: {stats['hierarchical_search_max_time']:.3f} 秒")
    print(f"  平均处理切片数: {stats['hierarchical_search_avg_slices']:.1f}")
    print(f"  平均跳过切片数: {stats['avg_slices_skipped']:.1f}")
    print()
    
    print("性能提升:")
    print(f"  平均加速比: {stats['avg_speedup_ratio']:.2f}x ± {stats['std_speedup_ratio']:.2f}x")
    print(f"  最小加速比: {stats['min_speedup_ratio']:.2f}x")
    print(f"  最大加速比: {stats['max_speedup_ratio']:.2f}x")
    print()
    print(f"  平均效率提升: {stats['avg_efficiency_improvement']:.2f}% ± {stats['std_efficiency_improvement']:.2f}%")
    print(f"  最小效率提升: {stats['min_efficiency_improvement']:.2f}%")
    print(f"  最大效率提升: {stats['max_efficiency_improvement']:.2f}%")
    print()
    
    print("最佳轴向分布:")
    for axis, count in stats['best_axis_distribution'].items():
        print(f"  {axis}: {count} 个文件")
    print("=" * 60)


def save_statistics_to_file(stats, output_path=None):
    """将统计结果保存到文件"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"batch_analysis_{timestamp}.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("批量处理结果统计分析\n")
        f.write("=" * 60 + "\n")
        f.write(f"总文件数: {stats['total_files']}\n")
        f.write(f"含肿瘤文件数: {stats['tumor_files']}\n")
        f.write(f"不含肿瘤文件数: {stats['non_tumor_files']}\n\n")
        
        f.write("全面搜索性能:\n")
        f.write(f"  平均处理时间: {stats['full_search_avg_time']:.3f} ± {stats['full_search_std_time']:.3f} 秒\n")
        f.write(f"  最小处理时间: {stats['full_search_min_time']:.3f} 秒\n")
        f.write(f"  最大处理时间: {stats['full_search_max_time']:.3f} 秒\n")
        f.write(f"  平均处理切片数: {stats['full_search_avg_slices']:.1f}\n\n")
        
        f.write("分层搜索性能:\n")
        f.write(f"  平均处理时间: {stats['hierarchical_search_avg_time']:.3f} ± {stats['hierarchical_search_std_time']:.3f} 秒\n")
        f.write(f"  最小处理时间: {stats['hierarchical_search_min_time']:.3f} 秒\n")
        f.write(f"  最大处理时间: {stats['hierarchical_search_max_time']:.3f} 秒\n")
        f.write(f"  平均处理切片数: {stats['hierarchical_search_avg_slices']:.1f}\n")
        f.write(f"  平均跳过切片数: {stats['avg_slices_skipped']:.1f}\n\n")
        
        f.write("性能提升:\n")
        f.write(f"  平均加速比: {stats['avg_speedup_ratio']:.2f}x ± {stats['std_speedup_ratio']:.2f}x\n")
        f.write(f"  最小加速比: {stats['min_speedup_ratio']:.2f}x\n")
        f.write(f"  最大加速比: {stats['max_speedup_ratio']:.2f}x\n\n")
        f.write(f"  平均效率提升: {stats['avg_efficiency_improvement']:.2f}% ± {stats['std_efficiency_improvement']:.2f}%\n")
        f.write(f"  最小效率提升: {stats['min_efficiency_improvement']:.2f}%\n")
        f.write(f"  最大效率提升: {stats['max_efficiency_improvement']:.2f}%\n\n")
        
        f.write("最佳轴向分布:\n")
        for axis, count in stats['best_axis_distribution'].items():
            f.write(f"  {axis}: {count} 个文件\n")
        f.write("=" * 60 + "\n")
    
    return output_path


def main():
    """主函数"""
    # JSON文件路径
    json_path = r"H:\data\tumor_data_swust\data2_output\filtered_merged_output\batch_summary\batch_summary_20260509_190210.json"
    
    # 检查文件是否存在
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        return
    
    # 加载数据
    print("正在加载数据...")
    data = load_json_data(json_path)
    
    # 分析数据
    print("正在分析数据...")
    stats = analyze_batch_results(data)
    
    # 打印统计结果
    print_statistics(stats)
    
    # 保存统计结果到文件
    output_file = save_statistics_to_file(stats)
    print(f"\n统计结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
