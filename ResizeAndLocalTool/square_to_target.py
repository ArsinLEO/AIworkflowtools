#!/usr/bin/env python3
"""
方形化缩放工具 - 将PNG序列帧扩展为正方形并缩放到统一尺寸。

用法：
  1. 把PNG序列帧放入 input/ 文件夹
  2. 双击运行，输入目标尺寸（默认640）
  3. 结果输出到 output/ 文件夹
"""

import os
import re
from PIL import Image


INPUT_DIR = "input"
OUTPUT_DIR = "output"
DEFAULT_SIZE = 640


def _natural_sort_key(s: str) -> list:
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"([0-9]+)", s)]


def get_png_files(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    return sorted(files, key=_natural_sort_key)


def square_and_resize(src_path: str, dst_path: str, target_size: int) -> None:
    """扩展为正方形（居中，不拉伸），然后缩放到 target_size。"""
    img = Image.open(src_path).convert("RGBA")
    w, h = img.size
    max_dim = max(w, h)

    square = Image.new("RGBA", (max_dim, max_dim), (0, 0, 0, 0))
    paste_x = (max_dim - w) // 2
    paste_y = (max_dim - h) // 2
    square.paste(img, (paste_x, paste_y), img)

    resized = square.resize((target_size, target_size), Image.LANCZOS)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    resized.save(dst_path)


def main() -> None:
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tasks: list[tuple[str, str, str]] = []  # (src, dst_dir, group_name)

    root_files = get_png_files(INPUT_DIR)
    for fname in root_files:
        tasks.append((os.path.join(INPUT_DIR, fname), OUTPUT_DIR, ""))

    if os.path.isdir(INPUT_DIR):
        subdirs = [d for d in os.listdir(INPUT_DIR)
                   if os.path.isdir(os.path.join(INPUT_DIR, d))
                   and d not in ("output", "__pycache__")]
        for sub in sorted(subdirs):
            sub_input = os.path.join(INPUT_DIR, sub)
            for fname in get_png_files(sub_input):
                tasks.append((os.path.join(sub_input, fname),
                              os.path.join(OUTPUT_DIR, sub), sub))

    if not tasks:
        print(f"错误: {INPUT_DIR}/ 下未找到PNG文件")
        input("\n按回车键关闭窗口...")
        return

    print("=" * 60)
    print("方形化缩放工具")
    print("=" * 60)
    print(f"\n找到 {len(tasks)} 张图")

    # 输入目标尺寸
    try:
        s = input(f"\n请输入目标尺寸 (默认 {DEFAULT_SIZE}): ").strip()
        target_size = int(s) if s else DEFAULT_SIZE
    except ValueError:
        print(f"输入无效，使用默认值: {DEFAULT_SIZE}")
        target_size = DEFAULT_SIZE

    print(f"\n将扩展为正方形并缩放到 {target_size}x{target_size}...\n")

    for src_path, out_dir, _ in tasks:
        fname = os.path.basename(src_path)
        dst_path = os.path.join(out_dir, fname)
        square_and_resize(src_path, dst_path, target_size)
        print(f"  {fname} → {target_size}x{target_size}")

    print(f"\n{'=' * 60}")
    print(f"全部完成！输出尺寸: {target_size}x{target_size}")
    print(f"结果在 output/ 文件夹中")
    print("=" * 60)
    input("\n按回车键关闭窗口...")


if __name__ == "__main__":
    main()
