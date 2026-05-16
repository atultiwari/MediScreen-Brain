"""
视频格式转换工具 - 将视频转换为浏览器兼容的 H.264 编码
"""
import subprocess
import os
from pathlib import Path

def check_video_info(video_path):
    """检查视频信息"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', video_path],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        print(f"\n{'='*60}")
        print(f"视频文件: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"错误: {result.stderr}")
        return result.returncode == 0
    except FileNotFoundError:
        print("未找到 ffprobe，请确保已安装 FFmpeg")
        return False

def convert_to_h264(input_path, output_path=None):
    """
    将视频转换为 H.264 编码 (浏览器兼容)
    
    参数:
        input_path: 输入视频路径
        output_path: 输出视频路径（可选，默认为原文件名_h264.mp4）
    """
    if output_path is None:
        path = Path(input_path)
        output_path = str(path.parent / f"{path.stem}_h264{path.suffix}")
    
    print(f"\n正在转换: {os.path.basename(input_path)}")
    print(f"输出到: {os.path.basename(output_path)}")
    
    # FFmpeg 命令：转换为 H.264 + AAC 音频
    cmd = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',           # H.264 视频编码
        '-preset', 'medium',          # 编码速度预设
        '-crf', '23',                 # 质量等级 (18-28, 越低质量越高)
        '-c:a', 'aac',                # AAC 音频编码
        '-b:a', '128k',               # 音频比特率
        '-movflags', '+faststart',    # 优化网页播放
        '-y',                         # 覆盖输出文件
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            print(f"✓ 转换成功: {output_path}")
            return True
        else:
            print(f"✗ 转换失败: {result.stderr}")
            return False
    except FileNotFoundError:
        print("✗ 未找到 ffmpeg，请确保已安装 FFmpeg")
        print("下载地址: https://ffmpeg.org/download.html")
        return False

def main():
    """主函数"""
    video_dir = Path(__file__).parent / "assets" / "video_new"
    
    if not video_dir.exists():
        print(f"视频目录不存在: {video_dir}")
        return
    
    video_files = list(video_dir.glob("*.mp4"))
    
    if not video_files:
        print("未找到 MP4 视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件\n")
    
    # 检查每个视频的信息
    print("步骤 1: 检查视频信息")
    for video in video_files:
        check_video_info(str(video))
    
    # 询问是否转换
    print("\n" + "="*60)
    choice = input("是否要转换为 H.264 编码？(y/n): ").strip().lower()
    
    if choice == 'y':
        print("\n步骤 2: 开始转换")
        for video in video_files:
            output = str(video.parent / f"{video.stem}_converted{video.suffix}")
            convert_to_h264(str(video), output)
        
        print("\n" + "="*60)
        print("转换完成！")
        print("请将转换后的文件重命名为原文件名，或更新 HTML 中的路径")
    else:
        print("已取消转换")

if __name__ == "__main__":
    main()
