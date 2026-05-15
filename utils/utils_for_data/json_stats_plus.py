#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本名称: json_stats.py
功能: 统计指定目录下所有 JSON 文件中的关键字段（SliceThickness, BaseResolution, ReconMatrixPE, 层数, ProtocolName），
     每个文件一行，输出 CSV 文件。
使用: python json_stats.py /path/to/dir
选项: -o 指定输出 CSV 路径（默认在目录下生成 json_metadata.csv）
     --no-recursive 不递归子目录
"""

import os
import sys
import json
import argparse
from pathlib import Path


def read_json_metadata(filepath):
    """
    读取 JSON 文件，返回所需字段的字典
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f,strict=False)
    except Exception as e:
        print(f"警告: 无法读取文件 {filepath} - {e}", file=sys.stderr)
        return None

    # 从 SliceTiming 数组长度获取层数
    slice_timing = data.get('SliceTiming')
    num_slices = len(slice_timing) if slice_timing is not None and isinstance(slice_timing, list) else None

    record = {
        'file_path': str(filepath),
        'file_name': os.path.basename(filepath),
        'ProtocolName': data.get('ProtocolName'),  # 新增字段
        'SliceThickness': data.get('SliceThickness'),
        'BaseResolution': data.get('BaseResolution'),
        'ReconMatrixPE': data.get('ReconMatrixPE'),
        'NumSlices': num_slices,  # 从 SliceTiming 长度获取的层数
        'SliceTiming_Exists': 'Yes' if slice_timing is not None else 'No',  # 标记是否存在
    }
    return record


def collect_stats(directory, recursive=True, output_csv=None):
    """
    遍历目录，收集所有 JSON 文件的元数据，逐条记录
    """
    pattern = '**/*.json' if recursive else '*.json'
    json_files = list(Path(directory).glob(pattern))

    # 按文件名排序
    json_files.sort()

    if not json_files:
        print(f"在目录 {directory} 中未找到任何 JSON 文件。")
        return

    print(f"找到 {len(json_files)} 个 JSON 文件，开始处理...")

    records = []
    failed = []

    # 带进度的处理
    total = len(json_files)
    for idx, json_path in enumerate(json_files, 1):
        # 显示进度和当前文件名
        sys.stdout.write(f"\r进度: {idx}/{total} ({idx * 100 // total}%) - 处理: {json_path.name[:50]}")
        sys.stdout.flush()

        meta = read_json_metadata(json_path)
        if meta is not None:
            records.append(meta)
        else:
            failed.append(str(json_path))

    print("\n处理完成。")

    if failed:
        print(f"警告: {len(failed)} 个文件读取失败:")
        for f in failed[:10]:  # 最多显示10个
            print(f"  {f}")
        if len(failed) > 10:
            print(f"  ... 还有 {len(failed) - 10} 个")

    # 输出统计信息
    print("\n" + "=" * 80)
    print("统计摘要:")
    print(f"总文件数: {len(json_files)}")
    print(f"成功读取: {len(records)}")
    print(f"读取失败: {len(failed)}")

    if records:
        # 统计各字段的缺失情况
        missing_protocol = sum(1 for r in records if r['ProtocolName'] is None)
        missing_slice_thick = sum(1 for r in records if r['SliceThickness'] is None)
        missing_base_res = sum(1 for r in records if r['BaseResolution'] is None)
        missing_recon = sum(1 for r in records if r['ReconMatrixPE'] is None)
        missing_num_slices = sum(1 for r in records if r['NumSlices'] is None)
        has_slice_timing = sum(1 for r in records if r['SliceTiming_Exists'] == 'Yes')

        print(f"\n字段缺失统计:")
        print(f"  ProtocolName 缺失: {missing_protocol}/{len(records)} ({missing_protocol * 100 // len(records)}%)")
        print(
            f"  SliceThickness 缺失: {missing_slice_thick}/{len(records)} ({missing_slice_thick * 100 // len(records)}%)")
        print(f"  BaseResolution 缺失: {missing_base_res}/{len(records)} ({missing_base_res * 100 // len(records)}%)")
        print(f"  ReconMatrixPE 缺失: {missing_recon}/{len(records)} ({missing_recon * 100 // len(records)}%)")
        print(
            f"  NumSlices (层数) 缺失: {missing_num_slices}/{len(records)} ({missing_num_slices * 100 // len(records)}%)")
        print(f"  SliceTiming 存在: {has_slice_timing}/{len(records)} ({has_slice_timing * 100 // len(records)}%)")

        # ProtocolName 分布统计
        protocol_names = [r['ProtocolName'] for r in records if r['ProtocolName'] is not None]
        if protocol_names:
            from collections import Counter
            protocol_counter = Counter(protocol_names)
            print(f"\nProtocolName 分布 (Top 10):")
            for name, count in protocol_counter.most_common(10):
                # 限制显示长度，避免过长
                display_name = name if len(name) <= 50 else name[:47] + "..."
                print(f"  {display_name}: {count} 个文件")

        # 层数分布统计（如果有数据）
        slice_counts = [r['NumSlices'] for r in records if r['NumSlices'] is not None]
        if slice_counts:
            from collections import Counter
            slice_counter = Counter(slice_counts)
            print(f"\n层数分布 (Top 10):")
            for num, count in slice_counter.most_common(10):
                print(f"  {num} 层: {count} 个文件")

        # SliceThickness 分布统计
        slice_thick_vals = [r['SliceThickness'] for r in records if r['SliceThickness'] is not None]
        if slice_thick_vals:
            from collections import Counter
            thick_counter = Counter(slice_thick_vals)
            print(f"\nSliceThickness 分布:")
            for thick, count in sorted(thick_counter.items()):
                print(f"  {thick} mm: {count} 个文件")

    print("=" * 80)

    # 确定输出 CSV 路径
    if output_csv is None:
        output_csv = os.path.join(directory, 'json_metadata.csv')
    else:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)

    # 写入 CSV
    import csv
    try:
        with open(output_csv, 'w', encoding='utf-8', newline='') as csvfile:
            fieldnames = ['file_path', 'file_name', 'ProtocolName', 'SliceThickness',
                          'BaseResolution', 'ReconMatrixPE', 'NumSlices', 'SliceTiming_Exists']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"\n结果已保存至: {output_csv}")
        print(f"成功导出 {len(records)} 条记录。")

        # 额外输出一个简化版统计报告
        summary_path = output_csv.replace('.csv', '_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"JSON 文件统计报告\n")
            f.write(f"目录: {directory}\n")
            f.write(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            if records:
                all_protocols = [r['ProtocolName'] for r in records if r['ProtocolName'] is not None]
                all_slice_thick = [r['SliceThickness'] for r in records if r['SliceThickness'] is not None]
                all_base_res = [r['BaseResolution'] for r in records if r['BaseResolution'] is not None]
                all_recon = [r['ReconMatrixPE'] for r in records if r['ReconMatrixPE'] is not None]
                all_num_slices = [r['NumSlices'] for r in records if r['NumSlices'] is not None]

                f.write("数值统计:\n")
                if all_slice_thick:
                    f.write(f"  SliceThickness: 最小={min(all_slice_thick)}, 最大={max(all_slice_thick)}, "
                            f"平均={sum(all_slice_thick) / len(all_slice_thick):.2f}\n")
                if all_base_res:
                    f.write(f"  BaseResolution: 最小={min(all_base_res)}, 最大={max(all_base_res)}, "
                            f"平均={sum(all_base_res) / len(all_base_res):.2f}\n")
                if all_recon:
                    f.write(f"  ReconMatrixPE: 最小={min(all_recon)}, 最大={max(all_recon)}, "
                            f"平均={sum(all_recon) / len(all_recon):.2f}\n")
                if all_num_slices:
                    f.write(f"  NumSlices (层数): 最小={min(all_num_slices)}, 最大={max(all_num_slices)}, "
                            f"平均={sum(all_num_slices) / len(all_num_slices):.2f}\n")

                f.write("\nProtocolName 统计:\n")
                if all_protocols:
                    from collections import Counter
                    protocol_counter = Counter(all_protocols)
                    f.write(f"  不同 ProtocolName 数量: {len(protocol_counter)}\n")
                    f.write("  出现频率 (Top 10):\n")
                    for name, count in protocol_counter.most_common(10):
                        f.write(f"    {name}: {count} 次\n")
                else:
                    f.write("  无 ProtocolName 数据\n")

        print(f"统计摘要已保存至: {summary_path}")

    except Exception as e:
        print(f"写入 CSV 失败: {e}", file=sys.stderr)
        sys.exit(1)


def main():

    target_dir = r'H:\data\tumor_data_swust\data2_niigz'
    if not os.path.isdir(target_dir):
        print(f"错误: '{target_dir}' 不是一个有效的目录")
        sys.exit(1)

    collect_stats(target_dir)


if __name__ == '__main__':
    main()