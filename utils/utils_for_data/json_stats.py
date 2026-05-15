#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本名称: json_stats.py
功能: 统计指定目录下所有 JSON 文件中的关键字段（SliceThickness, BaseResolution, ReconMatrixPE），
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
            data = json.load(f, strict=False)
    except Exception as e:
        print(f"警告: 无法读取文件 {filepath} - {e}", file=sys.stderr)
        return None

    record = {
        'file_path': str(filepath),
        'file_name': os.path.basename(filepath),
        'SliceThickness': data.get('SliceThickness'),
        'BaseResolution': data.get('BaseResolution'),
        'ReconMatrixPE': data.get('ReconMatrixPE'),
    }
    return record


def collect_stats(directory, recursive=True, output_csv=None):
    """
    遍历目录，收集所有 JSON 文件的元数据，逐条记录
    """
    pattern = '**/*.json' if recursive else '*.json'
    json_files = list(Path(directory).glob(pattern))

    if not json_files:
        print(f"在目录 {directory} 中未找到任何 JSON 文件。")
        return

    print(f"找到 {len(json_files)} 个 JSON 文件，开始处理...")

    records = []
    failed = []

    # 带进度的处理
    total = len(json_files)
    for idx, json_path in enumerate(json_files, 1):
        # 显示进度
        sys.stdout.write(f"\r进度: {idx}/{total} ({idx * 100 // total}%)")
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
            fieldnames = ['file_path', 'file_name', 'SliceThickness', 'BaseResolution', 'ReconMatrixPE']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"结果已保存至: {output_csv}")
        print(f"成功导出 {len(records)} 条记录。")
    except Exception as e:
        print(f"写入 CSV 失败: {e}", file=sys.stderr)
        sys.exit(1)


def main():

    target_dir = r'H:\data\tumor_data_swust\data2_niigz'
    if not os.path.isdir(target_dir):
        print(f"错误: '{target_dir}' 不是一个有效的目录", file=sys.stderr)
        sys.exit(1)

    collect_stats(target_dir)


if __name__ == '__main__':
    main()