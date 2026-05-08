import json
import subprocess
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

exe_path = r'dcm2niix.exe'
output_dir = r'H:\data\tumor_data_swust\data2_niigz'
json_path = 'subfolder_paths_data2.json'

def run_dcm2niix(path):
    command = [
        exe_path,
        '-f', '%f_%p_%t_%s',
        '-p', 'y',
        '-z', 'y',
        '-o', output_dir,
        path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return path, result.returncode, result.stderr


def main():
    with open(json_path, 'r', encoding='utf-8') as f:
        paths = json.load(f)

    max_workers = min(4, os.cpu_count())

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(run_dcm2niix, p): p
            for p in paths
        }

        for future in tqdm(
            as_completed(future_to_path),
            total=len(paths),
            desc="DCM → NII 转换进度"
        ):
            path, returncode, stderr = future.result()
            if returncode != 0:
                tqdm.write(f"❌ 失败: {path}")
                tqdm.write(stderr)
            else:
                tqdm.write(f"✅ 成功: {path}")


if __name__ == "__main__":
    main()