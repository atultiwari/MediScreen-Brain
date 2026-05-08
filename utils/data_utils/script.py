import os
import json

# 指定基础目录
base_dir = r'h:\data\tumor_data_swust\data2'

# 获取所有以 'Admin_' 开头的文件夹
admin_folders = [f for f in os.listdir(base_dir) if f.startswith('Admin_') and os.path.isdir(os.path.join(base_dir, f))]

# 初始化列表来存储子文件夹的绝对路径
subfolder_paths = []

# 遍历每个 Admin_ 文件夹
for admin_folder in admin_folders:
    admin_path = os.path.join(base_dir, admin_folder)
    # 获取该文件夹下的所有子文件夹
    for subfolder in os.listdir(admin_path):
        subfolder_path = os.path.join(admin_path, subfolder)
        if os.path.isdir(subfolder_path):
            # 添加绝对路径到列表
            subfolder_paths.append(os.path.abspath(subfolder_path))

# 保存列表到 JSON 文件
with open('subfolder_paths_data2.json', 'w', encoding='utf-8') as f:
    json.dump(subfolder_paths, f, ensure_ascii=False, indent=4)

print("子文件夹绝对路径已保存到 subfolder_paths.json")