import os
import subprocess
from pathlib import Path

def get_directory_size(path: str) -> int:
    """使用 du 命令无视差异快速统计目录实际大小（单位：字节）"""
    try:
        # 使用 du -sb 获取以 bytes 为单位的总大小
        result = subprocess.run(['du', '-sb', path], capture_output=True, text=True, check=True)
        size_in_bytes = int(result.stdout.split()[0])
        return size_in_bytes
    except Exception as e:
        print(f"⚠️ 无法计算 {path} 的体积: {e}")
        return 0

def format_size(size_in_bytes: float) -> str:
    """将字节数格式化为人类可读的格式 (KB, MB, GB, TB)"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.2f} PB"

def main():
    list_file = Path("cp_list")
    if not list_file.exists():
        print(f"❌ 找不到文件: {list_file.absolute()}")
        return

    # 读取文件中的路径
    with open(list_file, 'r', encoding='utf-8') as f:
        paths = [line.strip() for line in f if line.strip()]

    print(f"📦 开始统计 {len(paths)} 个目录的体积...\n")
    
    total_size = 0
    for file_path in paths:
        if os.path.exists(file_path):
            size = get_directory_size(file_path)
            total_size += size
            print(f"✅ {format_size(size).rjust(10)} | {file_path}")
        else:
            print(f"❌ {'目录不存在'.rjust(10)} | {file_path}")

    print("\n" + "="*80)
    print(f"📊 总体积: {format_size(total_size)}")
    print("="*80)

if __name__ == "__main__":
    main()
