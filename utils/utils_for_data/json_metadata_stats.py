#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本名称: json_metadata_stats.py
功能: 统计指定目录下所有 JSON 文件中的关键字段，按 ID 汇总。
ID 定义: 文件名第一个下划线前的数字部分（例如 "123_ep2d.json" -> ID 为 "123"）
统计字段: SliceThickness, BaseResolution, ReconMatrixPE
"""

import os
import sys
import json
import argparse
from collections import defaultdict
from pathlib import Path

def extract_first_id(filename):
    """
    提取第一个下划线前的数字作为 ID
    例如 "12345_series.json" -> "12345"
    """
    base = os.path.splitext(os.path.splitext(filename)[0])[0]
    # 取第一个下划线之前的部分
    if '_' in base:
        id_part = base.split('_', 1)[0]
        # 检查是否以数字开头（允许纯数字或数字开头）
        if id_part.isdigit() or (id_part[0].isdigit() and id_part[1:].replace('.', '').isdigit()):
            return id_part
    return None

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

    # 提取目标字段，缺失则设为 None
    record = {
        'SliceThickness': data.get('SliceThickness'),
        'BaseResolution': data.get('BaseResolution'),
        'ReconMatrixPE': data.get('ReconMatrixPE'),
        'file': filepath
    }
    return record

def collect_stats(directory, recursive=True):
    """
    遍历目录，收集所有 JSON 文件的元数据，按 ID 汇总
    """
    pattern = '**/*.json' if recursive else '*.json'
    json_files = list(Path(directory).glob(pattern))
    if not json_files:
        print(f"在目录 {directory} 中未找到任何 JSON 文件。")
        return

    # 按 ID 收集记录
    records_by_id = defaultdict(list)
    skipped = []

    for json_path in json_files:
        filename = json_path.name
        file_id = extract_first_id(filename)
        if file_id is None:
            print(f"跳过: {json_path} (无法从文件名提取数字 ID)")
            skipped.append(str(json_path))
            continue

        meta = read_json_metadata(json_path)
        if meta is not None:
            records_by_id[file_id].append(meta)

    # 输出汇总
    print(f"\n{'='*80}")
    print(f"统计目录: {directory}")
    print(f"共找到 {len(json_files)} 个 JSON 文件")
    print(f"成功处理 ID: {len(records_by_id)} 个")
    if skipped:
        print(f"跳过文件数: {len(skipped)} (无法提取 ID)")
        print("跳过的文件:")
        for s in skipped:
            print(f"  {s}")
    print(f"{'='*80}\n")

    # 打印每个 ID 的汇总信息
    print(f"{'ID':<12} {'文件数':<6} {'SliceThickness (mm)':<30} {'BaseResolution':<20} {'ReconMatrixPE':<20}")
    print("-" * 100)

    for id_, recs in sorted(records_by_id.items()):
        n_files = len(recs)

        # 收集各字段的值（保留原始值，去重）
        slice_vals = sorted(set(r['SliceThickness'] for r in recs if r['SliceThickness'] is not None))
        base_res_vals = sorted(set(r['BaseResolution'] for r in recs if r['BaseResolution'] is not None))
        recon_vals = sorted(set(r['ReconMatrixPE'] for r in recs if r['ReconMatrixPE'] is not None))

        # 格式化输出（多个值用逗号连接）
        slice_str = ', '.join(str(v) for v in slice_vals) if slice_vals else '缺失'
        base_str = ', '.join(str(v) for v in base_res_vals) if base_res_vals else '缺失'
        recon_str = ', '.join(str(v) for v in recon_vals) if recon_vals else '缺失'

        print(f"{id_:<12} {n_files:<6} {slice_str:<30} {base_str:<20} {recon_str:<20}")

    # 可选：输出到 CSV 文件
    output_csv = os.path.join(directory, 'metadata_summary.csv')
    try:
        with open(output_csv, 'w', encoding='utf-8') as csvfile:
            csvfile.write("ID,文件数,SliceThickness_values,BaseResolution_values,ReconMatrixPE_values\n")
            for id_, recs in sorted(records_by_id.items()):
                slice_vals = sorted(set(r['SliceThickness'] for r in recs if r['SliceThickness'] is not None))
                base_vals = sorted(set(r['BaseResolution'] for r in recs if r['BaseResolution'] is not None))
                recon_vals = sorted(set(r['ReconMatrixPE'] for r in recs if r['ReconMatrixPE'] is not None))
                csvfile.write(f"{id_},{len(recs)},\"{','.join(map(str, slice_vals))}\","
                              f"\"{','.join(map(str, base_vals))}\",\"{','.join(map(str, recon_vals))}\"\n")
        print(f"\n汇总结果已保存至: {output_csv}")
    except Exception as e:
        print(f"无法写入 CSV: {e}")

def main():


    target_dir = r'H:\data\tumor_data_swust\data2_niigz'
    if not os.path.isdir(target_dir):
        print(f"错误: '{target_dir}' 不是一个有效的目录", file=sys.stderr)
        sys.exit(1)

    collect_stats(target_dir)

if __name__ == '__main__':
    main()